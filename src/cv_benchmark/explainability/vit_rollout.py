from typing import List, Optional
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F


class ViTAttentionRollout:
    """Computes Attention Rollout across Vision Transformer encoder blocks.

    Reference: Abnar & Zuidema, 'Quantifying Attention Flow in Transformers' (ACL 2020).
    """

    def __init__(self, model: nn.Module, head_fusion: str = "mean", discard_ratio: float = 0.9) -> None:
        self.model = model
        self.head_fusion = head_fusion
        self.discard_ratio = discard_ratio

    def generate(self, input_tensor: torch.Tensor) -> np.ndarray:
        """Compute spatial attention rollout heatmap for a single image tensor (1, C, H, W)."""
        self.model.eval()
        if input_tensor.dim() == 3:
            input_tensor = input_tensor.unsqueeze(0)

        device = next(self.model.parameters()).device
        input_tensor = input_tensor.to(device)

        # In torchvision ViT:
        # 1. Conv_proj + class_token + position_embedding
        x = self.model._process_input(input_tensor)
        n = x.shape[0]
        batch_class_token = self.model.class_token.expand(n, -1, -1)
        x = torch.cat([batch_class_token, x], dim=1)
        x = self.model.encoder.dropout(x + self.model.encoder.pos_embedding)

        # Collect attention matrices across encoder blocks
        attentions = []
        with torch.no_grad():
            for block in self.model.encoder.layers:
                # Input to self attention after ln_1
                normed = block.ln_1(x)
                # Compute Q, K, V
                mha = block.self_attention
                embed_dim = mha.embed_dim
                num_heads = mha.num_heads
                head_dim = embed_dim // num_heads

                qkv = F.linear(normed, mha.in_proj_weight, mha.in_proj_bias)
                q, k, v = qkv.chunk(3, dim=-1)

                q = q.contiguous().view(n, -1, num_heads, head_dim).transpose(1, 2)
                k = k.contiguous().view(n, -1, num_heads, head_dim).transpose(1, 2)

                attn_weights = torch.matmul(q, k.transpose(-2, -1)) / np.sqrt(head_dim)
                attn_probs = F.softmax(attn_weights, dim=-1)  # (n, num_heads, seq_len, seq_len)
                attentions.append(attn_probs)

                # Standard block forward to continue sequence
                attn_out, _ = mha(normed, normed, normed, need_weights=False)
                x = x + block.dropout(attn_out)
                x = x + block.mlp(block.ln_2(x))

        mask = self.compute_rollout_from_attention_matrices(attentions)
        # Interpolate mask to image dimensions
        mask_t = torch.from_numpy(mask).unsqueeze(0).unsqueeze(0)
        mask_resized = F.interpolate(
            mask_t,
            size=(input_tensor.shape[2], input_tensor.shape[3]),
            mode="bilinear",
            align_corners=False,
        ).squeeze().numpy()

        return mask_resized

    def compute_rollout_from_attention_matrices(
        self,
        attentions: List[torch.Tensor],
    ) -> np.ndarray:
        """Compute rollout matrix R from a list of attention weight tensors (B, H, N, N)."""
        result = torch.eye(attentions[0].size(-1))
        with torch.no_grad():
            for attention in attentions:
                if self.head_fusion == "mean":
                    attn_fused = torch.mean(attention, dim=1)
                elif self.head_fusion == "max":
                    attn_fused = torch.max(attention, dim=1)[0]
                elif self.head_fusion == "min":
                    attn_fused = torch.min(attention, dim=1)[0]
                else:
                    raise ValueError(f"Unknown head fusion: {self.head_fusion}")

                # Account for residual connection: A_res = 0.5 * A + 0.5 * I
                batch_size = attn_fused.size(0)
                I = torch.eye(attn_fused.size(-1)).unsqueeze(0).to(attn_fused.device)
                a = (attn_fused + I) / 2.0
                a = a / a.sum(dim=-1, keepdim=True)

                result = torch.matmul(a[0].cpu(), result)

        # Look at the first row (the [CLS] token's attention to all tokens)
        mask = result[0, 1:]  # Exclude CLS token self-attention
        grid_size = int(np.sqrt(mask.size(0)))
        mask = mask.reshape(grid_size, grid_size).numpy()

        # Min-max normalization
        mask_min, mask_max = mask.min(), mask.max()
        if mask_max - mask_min > 1e-8:
            mask = (mask - mask_min) / (mask_max - mask_min)
        else:
            mask = np.zeros_like(mask)

        return mask
