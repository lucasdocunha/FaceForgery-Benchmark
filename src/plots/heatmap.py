from __future__ import annotations

import warnings

import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision.utils import make_grid

CNN_FAMILIES = {"resnet", "xception", "mobilenet"}
TRANSFORMER_FAMILIES = {"vit", "clip", "dino"}


def _normalize(heatmap: torch.Tensor) -> torch.Tensor:
    heatmap = heatmap - heatmap.amin(dim=(-2, -1), keepdim=True)
    return heatmap / heatmap.amax(dim=(-2, -1), keepdim=True).clamp_min(1e-8)


def _unwrap(model: nn.Module) -> nn.Module:
    return model.module if isinstance(model, nn.DataParallel) else model


def _last_conv(module: nn.Module) -> nn.Conv2d:
    layers = [layer for layer in module.modules() if isinstance(layer, nn.Conv2d)]
    if not layers:
        raise ValueError("No Conv2d layer is available for Grad-CAM")
    return layers[-1]


def grad_cam(model: nn.Module, image: torch.Tensor, target_class: int | None = None,
             target_layer: nn.Module | None = None) -> torch.Tensor:
    """Standard Grad-CAM: channel weights are spatially pooled activation gradients."""
    model = _unwrap(model)
    image = image.detach().clone().requires_grad_(True)
    layer = target_layer or _last_conv(model)
    captured: dict[str, torch.Tensor] = {}

    def forward_hook(_module, _inputs, output):
        activation = output if isinstance(output, torch.Tensor) else output[0]
        captured["activations"] = activation
        activation.register_hook(lambda gradient: captured.__setitem__("gradients", gradient))

    forward_handle = layer.register_forward_hook(forward_hook)
    try:
        model.zero_grad(set_to_none=True)
        logits = model(image)
        targets = logits.argmax(1) if target_class is None else torch.full(
            (logits.shape[0],), target_class, device=logits.device, dtype=torch.long,
        )
        logits.gather(1, targets[:, None]).sum().backward()
        activations, gradients = captured["activations"], captured["gradients"]
        weights = gradients.mean(dim=(-2, -1), keepdim=True)
        heatmap = torch.relu((weights * activations).sum(dim=1, keepdim=True))
        return _normalize(F.interpolate(heatmap, image.shape[-2:], mode="bilinear", align_corners=False))
    finally:
        forward_handle.remove()


def attention_rollout(model: nn.Module, image: torch.Tensor) -> torch.Tensor:
    """Attention rollout with residual connections over HF ViT/CLIP attention matrices."""
    model = _unwrap(model)
    previous = getattr(model, "capture_attentions", None)
    if previous is not None:
        model.capture_attentions = True
    try:
        with torch.no_grad():
            model(image)
    finally:
        if previous is not None:
            model.capture_attentions = previous
    attentions = getattr(model, "last_attentions", None)
    if not attentions or any(attention is None for attention in attentions):
        raise ValueError("Model did not expose attention matrices")
    rollout = None
    for attention in attentions:
        matrix = attention.mean(dim=1)
        identity = torch.eye(matrix.shape[-1], device=matrix.device, dtype=matrix.dtype)[None]
        matrix = matrix + identity
        matrix = matrix / matrix.sum(dim=-1, keepdim=True)
        rollout = matrix if rollout is None else matrix @ rollout
    tokens = rollout[:, 0, 1:]
    side = int(tokens.shape[-1] ** .5)
    if side * side != tokens.shape[-1]:
        raise ValueError("Attention tokens cannot be reshaped to a square spatial grid")
    heatmap = tokens.reshape(-1, 1, side, side)
    return _normalize(F.interpolate(heatmap, image.shape[-2:], mode="bilinear", align_corners=False))


def generate(model: nn.Module, family: str, image: torch.Tensor, method: str = "auto") -> torch.Tensor:
    if family not in CNN_FAMILIES | TRANSFORMER_FAMILIES:
        raise ValueError(f"Unknown model family: {family}")
    if method == "auto":
        method = "gradcam" if family in CNN_FAMILIES else "attention"
    if method == "gradcam":
        return grad_cam(model, image)
    if method != "attention":
        raise ValueError("method must be auto, gradcam, or attention")
    try:
        return attention_rollout(model, image)
    except ValueError:
        if family != "dino":
            raise
        warnings.warn("DINOv3 ConvNeXt has no attention matrices; using Grad-CAM.", stacklevel=2)
        return grad_cam(model, image)


def overlay(display_image: torch.Tensor, heatmap: torch.Tensor) -> torch.Tensor:
    image = display_image.detach().cpu()
    if image.ndim == 3:
        image = image.unsqueeze(0)
    image = image[:, :3]
    image = (image-image.amin(dim=(-2, -1), keepdim=True)) / (
        image.amax(dim=(-2, -1), keepdim=True)-image.amin(dim=(-2, -1), keepdim=True)
    ).clamp_min(1e-8)
    heat = heatmap.detach().cpu()
    color = torch.cat((heat, torch.zeros_like(heat), 1-heat), dim=1)
    return (.6*image + .4*color).clamp(0, 1)


def grid(display_images: torch.Tensor, heatmaps: torch.Tensor, columns: int | None = None) -> torch.Tensor:
    overlays = overlay(display_images, heatmaps)
    paired = torch.stack((display_images.cpu(), overlays), dim=1).flatten(0, 1)
    return make_grid(paired, nrow=2 if columns is None else columns, padding=4)
