"""Reconstruction objectives with explicit offline perceptual weights."""

from __future__ import annotations

from dataclasses import dataclass
import math
from pathlib import Path

import torch
from torch import nn
from torch.nn import functional as F

from .models import analytical_kl


@dataclass(frozen=True)
class BetaSchedule:
    kind: str = "linear"
    maximum: float = 0.0
    warmup_steps: int = 100
    cycle_steps: int = 100
    ramp_fraction: float = 0.5

    def __post_init__(self):
        if self.kind not in {"constant", "linear", "cyclic"}:
            raise ValueError("beta kind must be constant, linear, or cyclic")
        if not math.isfinite(self.maximum) or self.maximum < 0:
            raise ValueError("beta maximum must be finite and nonnegative")
        if self.warmup_steps < 0 or self.cycle_steps < 1 or not 0 < self.ramp_fraction <= 1:
            raise ValueError("Invalid beta schedule duration or ramp_fraction")

    def __call__(self, global_step: int):
        """global_step is the number of successfully completed optimizer updates."""
        if global_step < 0:
            raise ValueError("global_step cannot be negative")
        if self.kind == "constant":
            return self.maximum
        if self.kind == "linear":
            return self.maximum * min(1.0, global_step / max(self.warmup_steps, 1)) if self.warmup_steps else self.maximum
        phase = global_step % self.cycle_steps
        return self.maximum * min(1.0, phase / max(1.0, self.cycle_steps * self.ramp_fraction))


def structural_similarity(x, y, *, window_size=11, sigma=1.5):
    """Per-image RGB SSIM, population moments, unit data range, reflect padding."""
    if x.shape != y.shape or x.ndim != 4 or x.shape[1] != 3:
        raise ValueError("SSIM expects matching RGB BCHW tensors")
    with torch.autocast(device_type=x.device.type, enabled=False):
        x, y = x.float(), y.float()
        size = min(window_size, x.shape[-2], x.shape[-1])
        size -= 1 - size % 2
        if size < 3:
            raise ValueError("SSIM requires spatial dimensions >=3")
        grid = torch.arange(size, device=x.device, dtype=x.dtype) - size // 2
        gaussian = torch.exp(-grid.square() / (2 * sigma**2))
        gaussian = gaussian / gaussian.sum()
        kernel = (gaussian[:, None] * gaussian[None, :]).expand(3, 1, size, size)
        def smooth(value):
            return F.conv2d(F.pad(value, (size // 2,) * 4, mode="reflect"), kernel, groups=3)
        mx, my = smooth(x), smooth(y)
        vx = (smooth(x.square()) - mx.square()).clamp_min(0)
        vy = (smooth(y.square()) - my.square()).clamp_min(0)
        covariance = smooth(x * y) - mx * my
        similarity = ((2 * mx * my + 0.01**2) * (2 * covariance + 0.03**2)) / ((mx.square() + my.square() + 0.01**2) * (vx + vy + 0.03**2))
        return similarity.clamp(-1, 1).mean((1, 2, 3))


class OfflineLPIPS(nn.Module):
    """Load a complete LPIPS state_dict, including its pretrained feature network."""

    def __init__(self, state_path, *, net="alex"):
        super().__init__()
        if not state_path or not Path(state_path).is_file():
            raise FileNotFoundError("LPIPS requires lpips_state_path containing the complete pretrained LPIPS state_dict, including feature weights")
        try:
            import lpips
        except ImportError as exc:
            raise ImportError("LPIPS was requested; install the optional lpips package and stage its full offline state_dict") from exc
        # No torchvision download: random construction is immediately overwritten strictly.
        self.metric = lpips.LPIPS(net=net, pretrained=False, pnet_rand=True, verbose=False)
        self.metric.load_state_dict(torch.load(state_path, map_location="cpu", weights_only=True), strict=True)
        self.metric.requires_grad_(False)
        self.metric.eval()

    def train(self, mode=True):
        super().train(False)
        return self

    def forward(self, x, y):
        with torch.autocast(device_type=x.device.type, enabled=False):
            return self.metric(x.float() * 2 - 1, y.float() * 2 - 1).flatten(1).mean(1)


class CompositeReconstructionLoss(nn.Module):
    def __init__(self, *, lambda_ssim=0.1, lambda_lpips=0.0, beta=None, kl_reduction="sum", lpips_state_path=None, lpips_net="alex"):
        super().__init__()
        if any(not math.isfinite(float(v)) or v < 0 for v in (lambda_ssim, lambda_lpips)):
            raise ValueError("Loss weights must be finite and nonnegative")
        self.lambda_ssim, self.lambda_lpips = float(lambda_ssim), float(lambda_lpips)
        if kl_reduction not in {"sum", "mean_per_dim"}:
            raise ValueError("kl_reduction must be sum or mean_per_dim")
        self.kl_reduction = kl_reduction
        self.beta = beta if isinstance(beta, BetaSchedule) else BetaSchedule(**(beta or {}))
        self.perceptual = OfflineLPIPS(lpips_state_path, net=lpips_net) if lambda_lpips else None

    def forward(self, reconstruction, target, *, mu=None, logvar=None, global_step=0, real_mask=None):
        if reconstruction.shape != target.shape:
            raise ValueError("Reconstruction and target shapes differ")
        if real_mask is not None:
            if real_mask.ndim != 1 or len(real_mask) != len(target) or real_mask.dtype != torch.bool:
                raise ValueError("real_mask must be a boolean vector matching batch size")
            reconstruction, target = reconstruction[real_mask], target[real_mask]
            mu = mu[real_mask] if mu is not None else None
            logvar = logvar[real_mask] if logvar is not None else None
        beta = self.beta(global_step)
        if not len(target):
            zero = reconstruction.sum() * 0
            return {"loss": zero, "l1": zero, "ssim_loss": zero, "lpips": zero, "kl": zero, "kl_nats_per_sample": zero, "kl_nats_per_dimension": zero, "beta": zero + beta}
        l1 = (reconstruction.float() - target.float()).abs().mean()
        ssim = (1 - structural_similarity(reconstruction, target)).mean() if self.lambda_ssim else l1 * 0
        perceptual = self.perceptual(reconstruction, target).mean() if self.perceptual is not None else l1 * 0
        if (mu is None) != (logvar is None):
            raise ValueError("Provide both VAE mu and logvar")
        if beta and mu is None:
            raise ValueError("Nonzero KL beta requires a VAE posterior")
        kl_nats = analytical_kl(mu, logvar).mean() if logvar is not None else l1 * 0
        kl_per_dimension = kl_nats / mu.shape[1] if mu is not None else kl_nats
        kl = kl_nats if self.kl_reduction == "sum" else kl_per_dimension
        total = l1 + self.lambda_ssim * ssim + self.lambda_lpips * perceptual + beta * kl
        return {"loss": total, "l1": l1, "ssim_loss": ssim, "lpips": perceptual, "kl": kl, "kl_nats_per_sample": kl_nats, "kl_nats_per_dimension": kl_per_dimension, "beta": l1.new_tensor(beta)}
