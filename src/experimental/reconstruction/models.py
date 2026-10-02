"""Compact genuine-face reconstruction and raw-RGB forensic detectors."""

from __future__ import annotations

from contextlib import nullcontext
from dataclasses import asdict, dataclass
import math
from pathlib import Path

import torch
from torch import nn
from torch.nn import functional as F

from src.robustness.imaging import normalize_rgb


@dataclass(frozen=True)
class AutoencoderConfig:
    kind: str = "cae"
    image_size: int = 128
    width: int = 16
    latent_dim: int = 128
    skip_divisor: int = 8
    logvar_min: float = -12.0
    logvar_max: float = 8.0

    def __post_init__(self):
        if self.kind not in {"cae", "vae", "gated"}:
            raise ValueError("Autoencoder kind must be cae, vae, or gated")
        if self.image_size < 32 or self.image_size % 16:
            raise ValueError("Autoencoder image_size must be >=32 and divisible by 16")
        if self.width < 4 or self.latent_dim < 2:
            raise ValueError("Autoencoder width >=4 and latent_dim >=2 are required")
        if self.skip_divisor not in {4, 8}:
            raise ValueError("Skips must be restricted to H/4 or deeper (divisor 4 or 8)")
        if not self.logvar_min < self.logvar_max:
            raise ValueError("Invalid log-variance bounds")


def _block(in_channels, out_channels, stride=1):
    return nn.Sequential(
        nn.Conv2d(in_channels, out_channels, 3, stride=stride, padding=1),
        nn.GroupNorm(math.gcd(8, out_channels), out_channels),
        nn.SiLU(),
        nn.Conv2d(out_channels, out_channels, 3, padding=1),
        nn.GroupNorm(math.gcd(8, out_channels), out_channels),
        nn.SiLU(),
    )


def _check_rgb(x, size=None):
    if x.ndim != 4 or x.shape[1] != 3:
        raise ValueError("Expected raw RGB BCHW input")
    if size is not None and x.shape[-2:] != (size, size):
        raise ValueError(f"Expected {size}x{size} reconstruction input")
    if not torch.isfinite(x).all() or (x < 0).any() or (x > 1).any():
        raise ValueError("Reconstruction input must be finite raw RGB in [0,1]")


def analytical_kl(mu, logvar):
    """Per-sample KL(q(z|x) || N(0,I)), summed over latent dimensions."""
    if mu.shape != logvar.shape or mu.ndim != 2:
        raise ValueError("mu and logvar must have matching (batch, latent_dim) shapes")
    with torch.autocast(device_type=mu.device.type, enabled=False):
        mu, logvar = mu.float(), logvar.float()
        return 0.5 * (mu.square() + logvar.exp() - 1 - logvar).sum(dim=1)


class FaceAutoencoder(nn.Module):
    def __init__(self, config: AutoencoderConfig | dict | None = None):
        super().__init__()
        self.config = config if isinstance(config, AutoencoderConfig) else AutoencoderConfig(**(config or {}))
        c = self.config
        widths = [c.width * 2**i for i in range(4)]
        self.encoder = nn.ModuleList([_block(a, b, 2) for a, b in zip([3, *widths[:-1]], widths)])
        self.bottleneck_shape = (widths[-1], c.image_size // 16, c.image_size // 16)
        flat = math.prod(self.bottleneck_shape)
        self.to_mu = nn.Linear(flat, c.latent_dim)
        self.to_logvar = nn.Linear(flat, c.latent_dim) if c.kind == "vae" else None
        self.from_latent = nn.Linear(c.latent_dim, flat)
        self.decoder = nn.ModuleList([_block(widths[i + 1], widths[i]) for i in (2, 1, 0)])
        self.skip_indices = tuple(i for i in (2, 1, 0) if c.kind == "gated" and 2 ** (i + 1) >= c.skip_divisor)
        self.gates = nn.ModuleDict({str(i): nn.Conv2d(2 * widths[i], widths[i], 1) for i in self.skip_indices})
        for gate in self.gates.values():
            nn.init.zeros_(gate.weight)
            nn.init.constant_(gate.bias, -2.0)
        self.output = nn.Conv2d(widths[0], 3, 3, padding=1)
        self.skip_enabled = True

    def reconstruct(self, x, *, sample: bool | None = None):
        _check_rgb(x, self.config.image_size)
        features = []
        h = x
        for block in self.encoder:
            h = block(h)
            features.append(h)
        mu = self.to_mu(h.flatten(1))
        logvar = None
        z = mu
        if self.to_logvar is not None:
            logvar = self.to_logvar(h.flatten(1)).clamp(self.config.logvar_min, self.config.logvar_max)
            if self.training if sample is None else sample:
                z = mu + torch.exp(0.5 * logvar) * torch.randn_like(mu)
        h = self.from_latent(z).reshape(-1, *self.bottleneck_shape)
        for index, block in zip((2, 1, 0), self.decoder):
            h = block(F.interpolate(h, size=features[index].shape[-2:], mode="bilinear", align_corners=False))
            if self.skip_enabled and index in self.skip_indices:
                skip = features[index]
                if max(skip.shape[-2:]) > self.config.image_size // 4:
                    raise RuntimeError("A forbidden shallow skip reached the decoder")
                gate = torch.sigmoid(self.gates[str(index)](torch.cat((h, skip), dim=1)))
                h = h + gate * skip
        h = F.interpolate(h, size=x.shape[-2:], mode="bilinear", align_corners=False)
        return {"reconstruction": self.output(h).sigmoid(), "mu": mu, "logvar": logvar, "latent": z}

    def forward(self, x):
        return self.reconstruct(x)["reconstruction"]


class ResidualFusionBlock(nn.Module):
    """Raw [x, reconstruction, absolute error], optionally Sobel(error) magnitude."""

    def __init__(self, gradient: bool = False):
        super().__init__()
        self.gradient = bool(gradient)
        kernel = torch.tensor([[-1., 0., 1.], [-2., 0., 2.], [-1., 0., 1.]]) / 8
        self.register_buffer("sobel", torch.stack((kernel, kernel.T)).unsqueeze(1))
        self.register_buffer("luminance", torch.tensor([0.299, 0.587, 0.114]).reshape(1, 3, 1, 1))

    @property
    def out_channels(self):
        return 10 if self.gradient else 9

    def forward(self, x, reconstruction):
        if x.shape != reconstruction.shape or x.ndim != 4 or x.shape[1] != 3:
            raise ValueError("Fusion requires matching BCHW RGB tensors")
        residual = (x - reconstruction).abs()
        parts = [x, reconstruction, residual]
        if self.gradient:
            gray = (residual * self.luminance.to(residual)).sum(1, keepdim=True)
            derivatives = F.conv2d(F.pad(gray, (1, 1, 1, 1), mode="reflect"), self.sobel.to(gray))
            magnitude = torch.linalg.vector_norm(derivatives, dim=1, keepdim=True)
            parts.append(magnitude)
        return torch.cat(parts, dim=1)


class ReconstructionAnomaly(nn.Module):
    """Mean L1 is bounded in [0,1]; its orientation is always larger-is-faker."""

    def __init__(self, autoencoder):
        super().__init__()
        self.autoencoder = autoencoder

    def forward_details(self, x):
        result = self.autoencoder.reconstruct(x)
        result["anomaly_score"] = (x - result["reconstruction"]).abs().mean((1, 2, 3))
        return result

    def forward(self, x):
        score = self.forward_details(x)["anomaly_score"].float().clamp(1e-7, 1 - 1e-7)
        return torch.stack((torch.log1p(-score), torch.log(score)), dim=1)


class _Detector(nn.Module):
    def __init__(self, autoencoder, freeze_ae=True):
        super().__init__()
        self.autoencoder, self.freeze_ae = autoencoder, bool(freeze_ae)
        self.autoencoder.requires_grad_(not self.freeze_ae)
        if self.freeze_ae:
            self.autoencoder.eval()

    def train(self, mode=True):
        super().train(mode)
        if self.freeze_ae:
            self.autoencoder.eval()
        return self

    def _reconstruct(self, x):
        with torch.no_grad() if self.freeze_ae else nullcontext():
            # Fixed reconstruction also prevents stochastic VAE score noise in classification.
            return self.autoencoder.reconstruct(x, sample=False)

    def forward(self, x):
        return self.forward_details(x)["logits"]

    def parameter_groups(self, lr_ae, lr_backbone, lr_head):
        candidates = [("autoencoder", self.autoencoder, lr_ae), ("backbone", self.backbone, lr_backbone), ("head", self.classifier, lr_head)]
        groups = [{"name": name, "params": [p for p in module.parameters() if p.requires_grad], "lr": lr} for name, module, lr in candidates if module is not None]
        return [g for g in groups if g["params"]]


class ResidualDetector(_Detector):
    def __init__(self, autoencoder, *, freeze_ae=True, gradient=False, backbone="small", backbone_weights=None, width=16, dropout=0.1, initialize=True):
        super().__init__(autoencoder, freeze_ae)
        self.fusion = ResidualFusionBlock(gradient)
        channels = self.fusion.out_channels
        if backbone == "small":
            self.backbone = nn.Sequential(_block(channels, width, 2), _block(width, width * 2, 2), _block(width * 2, width * 4, 2), nn.AdaptiveAvgPool2d(1), nn.Flatten())
            features = width * 4
            if backbone_weights:
                raise ValueError("Small smoke backbone does not accept pretrained ResNet weights")
        elif backbone == "resnet18":
            from torchvision.models import resnet18
            self.backbone = resnet18(weights=None)
            if initialize:
                if not backbone_weights or not Path(backbone_weights).is_file():
                    raise FileNotFoundError("Generic ResNet-18 requires an explicit offline backbone_weights file")
                self.backbone.load_state_dict(torch.load(backbone_weights, map_location="cpu", weights_only=True), strict=True)
            old = self.backbone.conv1
            expanded = nn.Conv2d(channels, old.out_channels, old.kernel_size, old.stride, old.padding, bias=False)
            with torch.no_grad():
                expanded.weight.zero_()
                expanded.weight[:, :3].copy_(old.weight)
            self.backbone.conv1 = expanded
            features = self.backbone.fc.in_features
            self.backbone.fc = nn.Identity()
        else:
            raise ValueError("Residual backbone must be small or resnet18")
        self.classifier = nn.Sequential(nn.Dropout(dropout), nn.Linear(features, 2))

    def forward_details(self, x):
        details = self._reconstruct(x)
        # Classification gradients never update the AE from fake images.
        fused = self.fusion(x, details["reconstruction"].detach())
        scale = x.new_tensor([0.229, 0.224, 0.225]).reshape(1, 3, 1, 1)
        encoded = torch.cat((normalize_rgb(fused[:, :3]), normalize_rgb(fused[:, 3:6]), fused[:, 6:9] / scale, fused[:, 9:]), dim=1)
        features = self.backbone(encoded)
        return {**details, "features": features, "logits": self.classifier(features)}


class LatentDetector(_Detector):
    def __init__(self, autoencoder, *, freeze_ae=True, hidden_dim=64, dropout=0.1):
        if autoencoder.config.kind != "vae":
            raise ValueError("The latent detector requires a VAE")
        super().__init__(autoencoder, freeze_ae)
        size = autoencoder.config.latent_dim * 2 + 1
        self.backbone = None
        self.register_buffer("feature_mean", torch.zeros(size))
        self.register_buffer("feature_scale", torch.ones(size))
        self.classifier = nn.Sequential(nn.Linear(size, hidden_dim), nn.SiLU(), nn.Dropout(dropout), nn.Linear(hidden_dim, 2))

    @staticmethod
    def latent_features(details):
        return torch.cat((details["mu"].float(), details["logvar"].float(), analytical_kl(details["mu"], details["logvar"]).unsqueeze(1)), dim=1)

    def forward_details(self, x):
        details = self._reconstruct(x)
        features = self.latent_features(details).detach()
        features = (features - self.feature_mean) / self.feature_scale
        return {**details, "features": features, "logits": self.classifier(features)}


class MeanReconstructionEnsemble(nn.Module):
    """Fixed spatial/latent mean, requiring no fitting on target predictions."""

    def __init__(self, spatial, latent):
        super().__init__()
        self.spatial, self.latent = spatial, latent

    def forward(self, x):
        probabilities = (self.spatial(x).float().softmax(1) + self.latent(x).float().softmax(1)) / 2
        return probabilities.clamp_min(1e-7).log()


def autoencoder_config(model):
    return asdict(model.autoencoder.config)
