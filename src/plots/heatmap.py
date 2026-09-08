from __future__ import annotations

import warnings

import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision.utils import make_grid

CNN_FAMILIES = {"resnet", "xception", "mobilenet"}
TRANSFORMER_FAMILIES = {"vit", "clip", "dino"}
MOE_FAMILIES = {"moe_standard", "moe_frequency"}


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


def gradient_shap(
    model: nn.Module,
    image: torch.Tensor,
    target_class: int | None = None,
    baselines: torch.Tensor | None = None,
    n_steps: int = 32,
    stdevs: float = 0.0,
    **_kwargs,
) -> torch.Tensor:
    """Path-integrated Shapley values (Gradient SHAP / Integrated Gradients).

    Approximates Shapley values over a straight path between baseline and input,
    satisfying efficiency, symmetry, linearity, and dummy axioms.
    """
    model = _unwrap(model)
    was_training = model.training
    model.eval()
    device = image.device
    batch_size = image.shape[0]

    if baselines is None:
        baselines = torch.zeros_like(image)
    elif baselines.shape != image.shape:
        baselines = baselines.expand_as(image)

    alphas = torch.linspace(0.0, 1.0, max(2, n_steps), device=device)
    accumulated_grads = torch.zeros_like(image)

    for alpha in alphas:
        interpolated = baselines + alpha * (image - baselines)
        if stdevs > 0.0:
            interpolated = interpolated + torch.randn_like(interpolated) * stdevs
        interpolated = interpolated.detach().requires_grad_(True)

        logits = model(interpolated)
        if target_class is None:
            targets = logits.argmax(dim=-1)
        else:
            targets = torch.full((batch_size,), target_class, device=device, dtype=torch.long)

        score = logits.gather(1, targets.unsqueeze(1)).sum()
        grads = torch.autograd.grad(score, interpolated, retain_graph=False)[0]
        accumulated_grads += grads

    if was_training:
        model.train()

    avg_grads = accumulated_grads / len(alphas)
    attribution = (image - baselines) * avg_grads
    heatmap = attribution.abs().sum(dim=1, keepdim=True)
    return _normalize(heatmap)


def kernel_shap(
    model: nn.Module,
    image: torch.Tensor,
    target_class: int | None = None,
    patch_size: int = 16,
    n_samples: int = 128,
    baseline_value: float = 0.0,
    **_kwargs,
) -> torch.Tensor:
    """Superpixel / patch-based Kernel SHAP.

    Perturbs image patches to estimate Shapley values via weighted least squares.
    """
    model = _unwrap(model)
    was_training = model.training
    model.eval()
    device = image.device
    batch_size, channels, height, width = image.shape

    grid_h = max(1, height // patch_size)
    grid_w = max(1, width // patch_size)
    n_features = grid_h * grid_w

    heatmaps = []

    with torch.no_grad():
        for b in range(batch_size):
            img_b = image[b:b + 1]
            base_b = torch.full_like(img_b, baseline_value)

            orig_logits = model(img_b)
            cls_idx = orig_logits.argmax(dim=-1).item() if target_class is None else target_class
            f_base = model(base_b)[0, cls_idx].item()

            if n_features == 1:
                heatmaps.append(torch.ones(1, 1, height, width, device=device))
                continue

            p_probs = torch.rand(n_samples, device=device)
            masks = (torch.rand(n_samples, n_features, device=device) < p_probs[:, None]).float()
            subset_sizes = masks.sum(dim=1).clamp(1, n_features - 1)

            log_comb = (
                torch.lgamma(torch.tensor(n_features + 1, dtype=torch.float, device=device))
                - torch.lgamma(subset_sizes + 1)
                - torch.lgamma(torch.tensor(n_features, dtype=torch.float, device=device) - subset_sizes + 1)
            )
            weights = (n_features - 1) / (torch.exp(log_comb) * subset_sizes * (n_features - subset_sizes)).clamp_min(1e-12)
            weights = weights / weights.sum()

            perturbed_imgs = []
            for s in range(n_samples):
                mask_2d = masks[s].view(1, 1, grid_h, grid_w)
                mask_full = F.interpolate(mask_2d, size=(height, width), mode="nearest")
                perturbed = img_b * mask_full + base_b * (1.0 - mask_full)
                perturbed_imgs.append(perturbed[0])

            perturbed_batch = torch.stack(perturbed_imgs)
            eval_batch_size = 32
            out_scores = []
            for start_idx in range(0, n_samples, eval_batch_size):
                sub_batch = perturbed_batch[start_idx:start_idx + eval_batch_size]
                sub_out = model(sub_batch)[:, cls_idx]
                out_scores.append(sub_out)
            y_diff = torch.cat(out_scores) - f_base

            W_sqrt = torch.sqrt(weights).unsqueeze(1)
            Zw = masks * W_sqrt
            yw = y_diff * W_sqrt.squeeze(1)
            reg = 1e-4 * torch.eye(n_features, device=device)
            beta = torch.linalg.solve(Zw.T @ Zw + reg, Zw.T @ yw)

            beta_grid = beta.view(1, 1, grid_h, grid_w)
            heat_b = F.interpolate(beta_grid, size=(height, width), mode="bilinear", align_corners=False)
            heatmaps.append(torch.relu(heat_b))

    if was_training:
        model.train()

    res = torch.cat(heatmaps, dim=0)
    return _normalize(res)


def channel_shapley(
    model: nn.Module,
    image: torch.Tensor,
    target_class: int | None = None,
    domains: dict[str, tuple[int, int]] | None = None,
    baseline_value: float = 0.0,
) -> dict[str, float] | list[dict[str, float]]:
    """Exact Shapley values across input channels/domains (e.g. RGB, Mag, Phase, HighPass).

    Computes exact marginal contributions across all 2^|N| coalitions.
    """
    import math
    from itertools import combinations

    model = _unwrap(model)
    was_training = model.training
    model.eval()
    batch_size, total_channels, _, _ = image.shape

    if domains is None:
        if total_channels == 7:
            domains = {
                "spatial_rgb": (0, 3),
                "fft_magnitude": (3, 4),
                "fft_phase": (4, 5),
                "fft_highpass": (5, 6),
                "fft_lowpass": (6, 7),
            }
        elif total_channels == 6:
            domains = {
                "spatial_rgb": (0, 3),
                "fft_magnitude": (3, 4),
                "fft_phase": (4, 5),
                "fft_highpass": (5, 6),
            }
        else:
            domains = {f"channel_{c}": (c, c + 1) for c in range(total_channels)}

    domain_names = list(domains.keys())
    n_domains = len(domain_names)
    results = []

    with torch.no_grad():
        for b in range(batch_size):
            img_b = image[b:b + 1]
            base_b = torch.full_like(img_b, baseline_value)

            cls_idx = (
                model(img_b).argmax(dim=-1).item()
                if target_class is None
                else target_class
            )

            coalition_values = {}
            for mask_int in range(1 << n_domains):
                x_coalition = base_b.clone()
                active_set = []
                for idx, name in enumerate(domain_names):
                    if (mask_int >> idx) & 1:
                        c_start, c_end = domains[name]
                        x_coalition[:, c_start:c_end] = img_b[:, c_start:c_end]
                        active_set.append(idx)
                score = model(x_coalition)[0, cls_idx].item()
                coalition_values[frozenset(active_set)] = score

            shapley_values = {}
            all_indices = set(range(n_domains))
            for i, name in enumerate(domain_names):
                phi_i = 0.0
                others = all_indices - {i}
                for s_len in range(n_domains):
                    weight = math.factorial(s_len) * math.factorial(n_domains - s_len - 1) / math.factorial(n_domains)
                    for subset in combinations(others, s_len):
                        s_frozen = frozenset(subset)
                        s_with_i = s_frozen | {i}
                        marginal = coalition_values[s_with_i] - coalition_values[s_frozen]
                        phi_i += weight * marginal
                shapley_values[name] = float(phi_i)

            results.append(shapley_values)

    if was_training:
        model.train()

    return results[0] if batch_size == 1 else results


def generate(
    model: nn.Module,
    family: str,
    image: torch.Tensor,
    method: str = "auto",
    target_class: int | None = None,
    **kwargs,
) -> torch.Tensor:
    all_families = CNN_FAMILIES | TRANSFORMER_FAMILIES | MOE_FAMILIES
    if family not in all_families:
        raise ValueError(f"Unknown model family: {family}")

    if method == "auto":
        if family in MOE_FAMILIES:
            method = "shapley"
        elif family in CNN_FAMILIES:
            method = "gradcam"
        else:
            method = "attention"

    if method in ("shapley", "gradient_shap"):
        return gradient_shap(model, image, target_class=target_class, **kwargs)
    if method == "kernel_shap":
        return kernel_shap(model, image, target_class=target_class, **kwargs)
    if method == "gradcam":
        return grad_cam(model, image, target_class=target_class, **kwargs)
    if method != "attention":
        raise ValueError(
            f"method must be auto, gradcam, attention, shapley, gradient_shap, or kernel_shap (got {method})"
        )

    try:
        return attention_rollout(model, image)
    except ValueError:
        if family not in ("dino", *MOE_FAMILIES):
            raise
        warnings.warn(f"{family} has no attention matrices; falling back to Gradient SHAP.", stacklevel=2)
        return gradient_shap(model, image, target_class=target_class, **kwargs)


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
