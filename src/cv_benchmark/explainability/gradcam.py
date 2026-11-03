from typing import Optional, Tuple
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import matplotlib.cm as cm
from PIL import Image


class GradCAM:
    """Gradient-weighted Class Activation Mapping (Grad-CAM) for CNN architectures."""

    def __init__(self, model: nn.Module, target_layer: nn.Module) -> None:
        self.model = model
        self.target_layer = target_layer
        self.gradients: Optional[torch.Tensor] = None
        self.activations: Optional[torch.Tensor] = None
        self.hooks = []
        self._register_hooks()

    def _register_hooks(self) -> None:
        def forward_hook(module, input, output):
            self.activations = output
            if output.requires_grad:
                output.retain_grad()

        def backward_hook(module, grad_input, grad_output):
            if grad_output is not None and len(grad_output) > 0 and grad_output[0] is not None:
                self.gradients = grad_output[0].detach()

        self.hooks.append(self.target_layer.register_forward_hook(forward_hook))
        self.hooks.append(self.target_layer.register_full_backward_hook(backward_hook))

    def remove_hooks(self) -> None:
        for hook in self.hooks:
            hook.remove()
        self.hooks.clear()

    def generate(
        self,
        input_tensor: torch.Tensor,
        target_class: Optional[int] = None,
    ) -> np.ndarray:
        """Generate Grad-CAM heatmap for a single input image tensor (1, C, H, W)."""
        self.model.eval()
        input_tensor = input_tensor.clone()
        if input_tensor.dim() == 3:
            input_tensor = input_tensor.unsqueeze(0)
        input_tensor.requires_grad = True

        # Forward pass
        output = self.model(input_tensor)

        if target_class is None:
            target_class = int(output.argmax(dim=1).item())

        self.model.zero_grad()
        score = output[0, target_class]
        score.backward(retain_graph=True)

        # Fallback to activations.grad if backward hook didn't capture
        if self.gradients is None and self.activations is not None and self.activations.grad is not None:
            self.gradients = self.activations.grad.detach()

        if self.gradients is None or self.activations is None:
            # Fallback uniform CAM if gradients could not be propagated
            H, W = input_tensor.shape[2], input_tensor.shape[3]
            return np.ones((H, W), dtype=np.float32) * 0.5

        # Global average pooling of gradients: alpha_k = GAP(dY/dA_k)
        weights = torch.mean(self.gradients, dim=[2, 3], keepdim=True)  # (1, C, 1, 1)

        # Weighted combination of activation maps
        cam = torch.sum(weights * self.activations, dim=1, keepdim=True)  # (1, 1, H, W)
        cam = F.relu(cam)

        # Upsample to match original image size
        cam = F.interpolate(
            cam,
            size=(input_tensor.shape[2], input_tensor.shape[3]),
            mode="bilinear",
            align_corners=False,
        )

        cam_np = cam.squeeze().detach().cpu().numpy()
        cam_min, cam_max = cam_np.min(), cam_np.max()
        if cam_max - cam_min > 1e-8:
            cam_np = (cam_np - cam_min) / (cam_max - cam_min)
        else:
            cam_np = np.zeros_like(cam_np)

        return cam_np


def overlay_cam_on_image(
    image: np.ndarray,
    cam: np.ndarray,
    alpha: float = 0.5,
    colormap: str = "jet",
) -> np.ndarray:
    """Overlay 2D CAM heatmap [0, 1] onto RGB uint8 image [0, 255]."""
    if image.max() <= 1.0:
        image = (image * 255).astype(np.uint8)

    cmap = cm.get_cmap(colormap)
    heatmap = cmap(cam)[:, :, :3]  # RGB
    heatmap = (heatmap * 255).astype(np.uint8)

    overlay = (1 - alpha) * image.astype(float) + alpha * heatmap.astype(float)
    return np.clip(overlay, 0, 255).astype(np.uint8)
