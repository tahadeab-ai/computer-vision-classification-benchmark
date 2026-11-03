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
        self.attentions: List[torch.Tensor] = []
        self.hooks = []
        self._register_hooks()

    def _register_hooks(self) -> None:
        # Hook into PyTorch torchvision ViT encoder layers
        for layer in self.model.encoder.layers:
            # Self-attention module in torchvision ViT is layer.self_attention
            def hook_fn(module, input, output):
                # When output is tuple or tensor: in torchvision vit, self_attention returns (attn_output, attn_weights)
                # or we hook the forward of self_attention
                pass

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
