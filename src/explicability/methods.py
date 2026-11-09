"""Attribution adapters. No silent method substitution or predicted-class switching.

All class-specific adapters explain z_fake - z_real in the model's encoded input
space. Gradient SHAP and Kernel SHAP use Captum, not the legacy visualizer's
custom approximations. Group densities are NOT individual-pixel Shapley values.
"""
from __future__ import annotations

from contextlib import contextmanager
import math
import random
from typing import Any
import numpy as np
import torch
from torch import nn
from torch.nn import functional as F

METHODS = ("gradcam", "integrated_gradients", "gradient_shap", "kernel_shap",
           "lime", "saliency", "input_x_gradient", "smoothgrad", "occlusion",
           "feature_ablation", "shapley_sampling", "attention_rollout")


class UnsupportedMethod(ValueError):
    """A declared architecture/method incompatibility, not an execution failure."""


class FakeMargin(nn.Module):
    def __init__(self, model: nn.Module):
        super().__init__()
        self.model = model

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        logits = self.model(x)
        if logits.ndim != 2 or logits.shape[1] != 2:
            raise ValueError("Expected two logits with real=0 and fake=1")
        return logits[:, 1] - logits[:, 0]


@contextmanager
def seeded(seed: int, device: torch.device):
    """Restore caller RNG state; independently seed each sample/method/replicate."""
    py_state, np_state = random.getstate(), np.random.get_state()
    devices = [device.index if device.index is not None else torch.cuda.current_device()] if device.type == "cuda" else []
    try:
        with torch.random.fork_rng(devices=devices):
            random.seed(seed); np.random.seed(seed % 2**32); torch.manual_seed(seed)
            if devices:
                torch.cuda.manual_seed_all(seed)
            yield
    finally:
        random.setstate(py_state); np.random.set_state(np_state)


def feature_mask(x: torch.Tensor, mode: str, patch: int = 32,
                 partition: str = "grid", radial_bins: int = 8) -> torch.Tensor:
    if x.ndim != 4 or x.shape[0] != 1 or patch < 1 or radial_bins < 2:
        raise ValueError("Expected one BCHW input and positive feature-group settings")
    _, channels, h, w = x.shape
    yy, xx = torch.meshgrid(torch.arange(h, device=x.device), torch.arange(w, device=x.device), indexing="ij")
    grid = yy.div(patch, rounding_mode="floor") * math.ceil(w / patch) + xx.div(patch, rounding_mode="floor")
    if partition not in {"grid", "radial"}:
        raise ValueError("partition must be grid or radial")
    if partition == "radial":
        fy = torch.fft.fftshift(torch.fft.fftfreq(h, device=x.device))
        fx = torch.fft.fftshift(torch.fft.fftfreq(w, device=x.device))
        radius = torch.sqrt(fy[:, None] ** 2 + fx[None, :] ** 2)
        spectral = (radius / math.sqrt(.5) * radial_bins).long().clamp_max(radial_bins - 1)
    else:
        spectral = grid
    # RGB is a joint colour feature. Complex real/imaginary components share bins.
    if mode == "none":
        groups = [(list(range(channels)), grid)]
    elif mode in {"concat", "concat_frequency"}:
        groups = [(list(range(3)), grid)] + [([c], spectral) for c in range(3, channels)]
    else:
        groups = [(list(range(channels)), spectral)]
    mask = torch.empty_like(x, dtype=torch.long); offset = 0
    for indices, spatial in groups:
        _, inverse = torch.unique(spatial, sorted=True, return_inverse=True)
        spatial = inverse.reshape(h, w)
        mask[:, indices] = spatial + offset
        offset += int(spatial.max()) + 1
    return mask


def cam_layer(model: nn.Module, family: str, explicit: str | None = None):
    if explicit:
        layer = model.get_submodule(explicit)
    elif family in {"vit", "clip"}:
        from src.models._hf_layers import encoder_layers
        last = encoder_layers(model.backbone)[-1]
        # Before final attention: patch tokens still influence the class token.
        attr = "layernorm_before" if family == "vit" else "layer_norm1"
        if not hasattr(last, attr):
            raise UnsupportedMethod(f"Declare a pre-attention CAM layer; {attr} is unavailable")
        layer = getattr(last, attr)
    else:
        layers = [m for m in model.modules() if isinstance(m, nn.Conv2d)]
        if not layers:
            raise UnsupportedMethod("No convolutional CAM target; declare a compatible layer")
        layer = layers[-1]
    name = next(name for name, m in model.named_modules() if m is layer)
    return name, layer


def gradcam(model: nn.Module, family: str, x: torch.Tensor, explicit: str | None):
    name, layer = cam_layer(model, family, explicit)
    activations = []
    handle = layer.register_forward_hook(lambda _m, _i, out: activations.append(out))
    try:
        score = FakeMargin(model)(x)
        if len(activations) != 1 or not isinstance(activations[0], torch.Tensor):
            raise UnsupportedMethod("CAM layer must return a single tensor and execute once")
        activation = activations[0]
        grad = torch.autograd.grad(score.sum(), activation)[0]
        if activation.ndim == 3:
            patches = activation.shape[1] - 1
            side = math.isqrt(patches)
            if side * side != patches:
                raise UnsupportedMethod("CAM token grid requires one CLS token and square patch grid")
            activation = activation[:, 1:].transpose(1, 2).reshape(1, -1, side, side)
            grad = grad[:, 1:].transpose(1, 2).reshape_as(activation)
        if activation.ndim != 4 or min(activation.shape[-2:]) <= 1:
            raise UnsupportedMethod("CAM target has no nontrivial spatial grid")
        signed = (activation * grad.mean((-2, -1), keepdim=True)).sum(1, keepdim=True)
        raw = signed.relu()
        return F.interpolate(raw, x.shape[-2:], mode="bilinear", align_corners=False), {
            "target_layer": name, "native_cam_shape": list(raw.shape),
            "semantics": "rectified layer localization for fake-minus-real margin; not additive input attribution",
            "signed_native_cam": signed.detach().cpu().numpy(),
        }
    finally:
        handle.remove()


def _group_density(coefficients: torch.Tensor, mask: torch.Tensor):
    coefficients = coefficients.reshape(-1)
    n = int(mask.max()) + 1
    if len(coefficients) != n:
        raise ValueError("Surrogate coefficient count differs from feature partition")
    counts = torch.bincount(mask.flatten(), minlength=n).to(coefficients)
    return coefficients[mask] / counts[mask]


def explain(model: nn.Module, family: str, mode: str, x: torch.Tensor,
            baseline: torch.Tensor, method: str, settings: dict[str, Any], seed: int):
    """Return dense raw attribution, raw feature mask, and estimator metadata.

    Numerical tests cannot establish the scientific validity of an explanation.
    Baselines and interpolation occur AFTER the existing dataset encoding.
    """
    if method not in METHODS:
        raise ValueError(f"Unknown method: {method}")
    if x.shape != baseline.shape or x.ndim != 4 or x.shape[0] != 1:
        raise ValueError("Inputs and baseline must be matching single-image BCHW tensors")
    if not torch.isfinite(x).all() or not torch.isfinite(baseline).all():
        raise ValueError("Non-finite encoded input or baseline")
    model.eval()
    x = x.detach().float().requires_grad_(True)
    baseline = baseline.detach().to(x)
    patch = int(settings.get("patch_size", 32))
    mask = feature_mask(x, mode, patch, settings.get("partition", "grid"), int(settings.get("radial_bins", 8)))
    batch = int(settings.get("perturbations_per_eval", 8))
    samples = int(settings.get("samples", 512)); steps = int(settings.get("ig_steps", 64))
    if min(samples, batch, steps) < 1 or steps < 2:
        raise ValueError("Invalid attribution budget")
    meta: dict[str, Any] = {"method": method, "target": "logit_fake_minus_real",
                          "seed": seed, "feature_groups": int(mask.max()) + 1,
                          "semantics": "encoded-input attribution", "settings": settings}
    margin = FakeMargin(model)
    with seeded(seed, x.device), torch.enable_grad():
        from captum.attr import (FeatureAblation, GradientShap, InputXGradient,
            IntegratedGradients, KernelShap, Lime, NoiseTunnel, Occlusion,
            Saliency, ShapleyValueSampling)
        if method == "gradcam":
            attr, extra = gradcam(model, family, x, settings.get("cam_layer")); meta.update(extra)
        elif method == "attention_rollout":
            if family not in {"vit", "clip"} or not hasattr(model, "capture_attentions"):
                raise UnsupportedMethod("Attention rollout requires exposed ViT/CLIP attention matrices")
            from src.plots.heatmap import attention_rollout
            attr = attention_rollout(model, x)
            meta.update(target="class_agnostic", semantics="residual attention rollout; not class-specific attribution")
        elif method == "integrated_gradients":
            attr, delta = IntegratedGradients(margin).attribute(x, baselines=baseline,
                n_steps=steps, method="gausslegendre", internal_batch_size=batch,
                return_convergence_delta=True)
            meta["completeness_residual"] = float(delta.abs().max().detach())
        elif method == "gradient_shap":
            attr, delta = GradientShap(margin).attribute(x, baselines=baseline,
                n_samples=int(settings.get("gradient_samples", 32)),
                stdevs=float(settings.get("noise_std", .02)), return_convergence_delta=True)
            meta["mean_absolute_completeness_residual"] = float(delta.abs().mean().detach())
            meta["baseline_distribution"] = "one declared reference; degenerate baseline distribution"
        elif method == "saliency":
            attr = Saliency(margin).attribute(x, abs=False)
            meta["semantics"] = "signed local derivative, not an additive contribution"
        elif method == "input_x_gradient":
            attr = InputXGradient(margin).attribute(x)
        elif method == "smoothgrad":
            attr = NoiseTunnel(Saliency(margin)).attribute(x, nt_type="smoothgrad", abs=False,
                nt_samples=int(settings.get("gradient_samples", 32)), nt_samples_batch_size=batch,
                stdevs=float(settings.get("noise_std", .02)))
            meta["semantics"] = "smoothed signed derivative; no robustness certification"
        elif method == "occlusion":
            window = (x.shape[1], min(patch, x.shape[2]), min(patch, x.shape[3]))
            attr = Occlusion(margin).attribute(x, baselines=baseline, sliding_window_shapes=window,
                strides=window, perturbations_per_eval=batch)
            meta["semantics"] = "window perturbation score allocated by Captum, not additive attribution"
        elif method in {"kernel_shap", "lime"}:
            if samples < 2 * meta["feature_groups"] + 2:
                raise ValueError("Surrogate sample budget must be at least 2*feature_groups+2")
            if method == "lime":
                from captum._utils.models.linear_model import SkLearnRidge
                estimator = Lime(margin, interpretable_model=SkLearnRidge(alpha=1.0))
                meta["surrogate"] = "ridge(alpha=1); Captum default cosine kernel"
            else:
                estimator = KernelShap(margin)
                meta["surrogate"] = "Captum baseline-substitution Kernel SHAP"
            coefficients = estimator.attribute(x, baselines=baseline, feature_mask=mask,
                n_samples=samples, perturbations_per_eval=batch, return_input_shape=False)
            attr = _group_density(coefficients, mask)
            meta["group_coefficients"] = coefficients.detach().cpu().numpy()
            meta["semantics"] = "group-contribution density, not individual-pixel Shapley values"
        elif method in {"feature_ablation", "shapley_sampling"}:
            estimator = FeatureAblation(margin) if method == "feature_ablation" else ShapleyValueSampling(margin)
            kwargs = {} if method == "feature_ablation" else {"n_samples": int(settings.get("shapley_permutations", 32))}
            repeated = estimator.attribute(x, baselines=baseline, feature_mask=mask,
                perturbations_per_eval=batch, **kwargs)
            coefficients = torch.stack([repeated[mask == k][0] for k in range(meta["feature_groups"])])
            attr = _group_density(coefficients, mask)
            meta["group_coefficients"] = coefficients.detach().cpu().numpy()
            meta["semantics"] = "group-effect density; ablation effects need not sum to score difference"
        else:
            raise AssertionError("Unreachable method")
        with torch.no_grad():
            meta["input_margin"] = float(margin(x).item())
            meta["baseline_margin"] = float(margin(baseline).item())
    if not torch.isfinite(attr).all():
        raise ValueError("Attribution contains non-finite values")
    meta["all_zero"] = bool(torch.all(attr == 0).item())
    meta["constant"] = bool((attr.max() == attr.min()).item())
    return attr.detach(), mask.detach(), meta


def perturbation_curves(model: nn.Module, x: torch.Tensor, baseline: torch.Tensor,
                        attr: torch.Tensor, mask: torch.Tensor, seed: int, points: int = 11):
    """Group insertion/deletion diagnostics; not ROAR or causal validation."""
    if points < 2:
        raise ValueError("Need at least two curve points")
    n = int(mask.max()) + 1
    # Layer maps have one channel; distribute display score, not CAM semantics.
    score = attr.expand_as(x) if attr.shape[1] == 1 else attr
    ranking = torch.stack([score[mask == k].sum() for k in range(n)]).argsort(descending=True, stable=True)
    rng = np.random.default_rng(seed)
    random_order = torch.as_tensor(rng.permutation(n), device=x.device)
    fractions = np.linspace(0, 1, points)
    out = {"fractions": fractions.tolist(), "target": "fake_minus_real_margin",
           "order": "descending signed group score", "interpretation": "off-manifold sensitivity diagnostic"}
    with torch.no_grad():
        for name, order in (("ranked", ranking), ("random", random_order)):
            deletion, insertion = [], []
            for fraction in fractions:
                selected = torch.isin(mask, order[:int(round(fraction * n))])
                deletion.append(float(FakeMargin(model)(torch.where(selected, baseline, x)).item()))
                insertion.append(float(FakeMargin(model)(torch.where(selected, x, baseline)).item()))
            out[name] = {"deletion": deletion, "insertion": insertion,
                         "deletion_auc": float(np.trapezoid(deletion, fractions)),
                         "insertion_auc": float(np.trapezoid(insertion, fractions))}
    return out
