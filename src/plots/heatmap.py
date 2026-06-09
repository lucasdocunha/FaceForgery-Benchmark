from __future__ import annotations

from pathlib import Path
from typing import Any

import cv2
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F


def _unwrap_model(model: nn.Module) -> nn.Module:
    if isinstance(model, nn.DataParallel):
        return model.module
    return model


def find_target_layer(model: nn.Module) -> nn.Module:
    """Return the layer used to compute a Grad-CAM heatmap."""
    base_model = _unwrap_model(model)
    patch_embed = getattr(base_model, "patch_embed", None)
    patch_proj = getattr(patch_embed, "proj", None)
    if isinstance(patch_proj, nn.Conv2d):
        return patch_proj

    target_layer: nn.Module | None = None
    for module in base_model.modules():
        if isinstance(module, nn.Conv2d):
            target_layer = module

    if target_layer is None:
        raise ValueError(
            "No compatible layer found for Grad-CAM. Expected patch_embed.proj "
            "or at least one nn.Conv2d."
        )
    return target_layer


def _as_batched_image(image_tensor: torch.Tensor, device: torch.device) -> torch.Tensor:
    if image_tensor.ndim == 3:
        image_tensor = image_tensor.unsqueeze(0)
    if image_tensor.ndim != 4:
        raise ValueError("image_tensor must have shape (C, H, W) or (1, C, H, W).")
    if image_tensor.shape[0] != 1:
        raise ValueError("gradcam_heatmap expects a single image.")
    return image_tensor.detach().clone().to(device=device, dtype=torch.float32)


def _model_device(model: nn.Module, image_tensor: torch.Tensor) -> torch.device:
    try:
        return next(model.parameters()).device
    except StopIteration:
        return image_tensor.device


def _normalize_heatmap(heatmap: torch.Tensor) -> torch.Tensor:
    heatmap = torch.nan_to_num(heatmap, nan=0.0, posinf=0.0, neginf=0.0)
    heatmap = heatmap - heatmap.min()
    max_value = heatmap.max()
    if float(max_value.detach()) <= 1e-12:
        return torch.zeros_like(heatmap)
    return heatmap / max_value


def gradcam_heatmap(
    model: nn.Module,
    image_tensor: torch.Tensor,
    target_class: int | None = None,
    target_layer: nn.Module | None = None,
) -> torch.Tensor:
    """Compute a normalized Grad-CAM heatmap for one image."""
    layer = target_layer if target_layer is not None else find_target_layer(model)
    device = _model_device(model, image_tensor)
    image = _as_batched_image(image_tensor, device)
    image.requires_grad_(True)
    height, width = image.shape[-2:]
    was_training = model.training
    activations: dict[str, torch.Tensor] = {}
    gradients: dict[str, torch.Tensor] = {}

    def forward_hook(_module: nn.Module, _inputs: Any, output: torch.Tensor) -> None:
        if not isinstance(output, torch.Tensor) or output.ndim != 4:
            raise ValueError("Target layer must return a 4D tensor.")
        activations["value"] = output
        output.register_hook(lambda grad: gradients.__setitem__("value", grad))

    handle = layer.register_forward_hook(forward_hook)
    try:
        model.eval()
        model.zero_grad(set_to_none=True)
        logits = model(image)
        if logits.ndim != 2:
            raise ValueError("Model output must have shape (batch, classes).")
        if target_class is None:
            class_idx = int(logits.detach().argmax(dim=1).item())
        else:
            class_idx = int(target_class)
        if class_idx < 0 or class_idx >= logits.shape[1]:
            raise ValueError("target_class is outside the model output range.")

        logits[0, class_idx].backward()
        if "value" not in activations or "value" not in gradients:
            raise RuntimeError("Grad-CAM hooks did not capture activations and gradients.")

        activation = activations["value"]
        gradient = gradients["value"]
        weights = gradient.mean(dim=(2, 3), keepdim=True)
        heatmap = torch.relu((weights * activation).sum(dim=1, keepdim=True))
        heatmap = F.interpolate(
            heatmap,
            size=(height, width),
            mode="bilinear",
            align_corners=False,
        )[0, 0]
        return _normalize_heatmap(heatmap).detach().cpu()
    finally:
        handle.remove()
        model.zero_grad(set_to_none=True)
        if was_training:
            model.train()


def _first_image(image_tensor: torch.Tensor) -> torch.Tensor:
    if image_tensor.ndim == 4:
        if image_tensor.shape[0] != 1:
            raise ValueError("overlay_heatmap expects a single image.")
        image_tensor = image_tensor[0]
    if image_tensor.ndim != 3:
        raise ValueError("image_tensor must have shape (C, H, W) or (1, C, H, W).")
    return image_tensor.detach().cpu().float()


def _to_display_rgb(image_tensor: torch.Tensor) -> torch.Tensor:
    image = _first_image(image_tensor)
    channels, height, width = image.shape

    if channels == 1:
        rgb = image.repeat(3, 1, 1)
    elif channels == 2:
        rgb = torch.cat([image, torch.zeros(1, height, width)], dim=0)
    else:
        rgb = image[:3]

    channel_min = rgb.amin(dim=(1, 2), keepdim=True)
    channel_max = rgb.amax(dim=(1, 2), keepdim=True)
    denom = (channel_max - channel_min).clamp_min(1e-8)
    rgb = (rgb - channel_min) / denom
    return torch.nan_to_num(rgb, nan=0.0, posinf=1.0, neginf=0.0).clamp(0.0, 1.0)


def _prepare_heatmap(heatmap: torch.Tensor | np.ndarray, size: tuple[int, int]) -> np.ndarray:
    if isinstance(heatmap, torch.Tensor):
        heatmap_array = heatmap.detach().cpu().float().numpy()
    else:
        heatmap_array = np.asarray(heatmap, dtype=np.float32)
    if heatmap_array.ndim != 2:
        raise ValueError("heatmap must be a 2D array.")
    heatmap_array = np.nan_to_num(heatmap_array, nan=0.0, posinf=0.0, neginf=0.0)
    if heatmap_array.shape != size:
        heatmap_array = cv2.resize(
            heatmap_array,
            (size[1], size[0]),
            interpolation=cv2.INTER_LINEAR,
        )
    heatmap_array = heatmap_array - float(heatmap_array.min())
    max_value = float(heatmap_array.max())
    if max_value <= 1e-12:
        return np.zeros(size, dtype=np.float32)
    return (heatmap_array / max_value).astype(np.float32)


def overlay_heatmap(
    image_tensor: torch.Tensor,
    heatmap: torch.Tensor | np.ndarray,
    alpha: float = 0.4,
) -> torch.Tensor:
    """Overlay a heatmap on the displayable channels of an image tensor."""
    alpha = float(np.clip(alpha, 0.0, 1.0))
    rgb = _to_display_rgb(image_tensor)
    height, width = rgb.shape[-2:]
    heatmap_array = _prepare_heatmap(heatmap, (height, width))

    base = (rgb.permute(1, 2, 0).numpy() * 255.0).astype(np.uint8)
    heat_uint8 = (heatmap_array * 255.0).astype(np.uint8)
    colored = cv2.applyColorMap(heat_uint8, cv2.COLORMAP_JET)
    colored = cv2.cvtColor(colored, cv2.COLOR_BGR2RGB)
    overlay = ((1.0 - alpha) * base.astype(np.float32)) + (
        alpha * colored.astype(np.float32)
    )
    overlay = np.clip(overlay, 0, 255).astype(np.uint8)
    return torch.from_numpy(overlay)


def save_heatmap_overlay(
    image_tensor: torch.Tensor,
    heatmap: torch.Tensor | np.ndarray,
    output_path: str | Path,
    alpha: float = 0.4,
) -> Path:
    """Save a heatmap overlay PNG and return its path."""
    path = Path(output_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    overlay = overlay_heatmap(image_tensor, heatmap, alpha=alpha).numpy()
    cv2.imwrite(str(path), cv2.cvtColor(overlay, cv2.COLOR_RGB2BGR))
    return path
