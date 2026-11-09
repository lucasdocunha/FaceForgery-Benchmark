"""Bounded spatial/spectral pilot, not an asserted novel or superior detector.

A learned gate is a hypothesis about cue usefulness, not calibrated uncertainty.
The spatial-control auxiliary branch has the SAME parameter count as the
spectral branch, using luminance instead of its FFT magnitude.
"""

from __future__ import annotations
import hashlib
import torch
from torch import nn
from torch.nn import functional as F
from .imaging import normalize_rgb, encode_tensor

KINDS = {"rgb", "spectral", "early_concat", "late_mean", "adaptive", "spatial_control"}


class TinyBranch(nn.Module):
    def __init__(self, width: int = 32):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv2d(1, width, 3, 2, 1),
            nn.GroupNorm(4, width),
            nn.GELU(),
            nn.Conv2d(width, 2 * width, 3, 2, 1),
            nn.GroupNorm(4, 2 * width),
            nn.GELU(),
            nn.AdaptiveAvgPool2d(1),
            nn.Flatten(),
        )
        self.num_features = 2 * width

    def forward(self, x):
        return self.net(x)


class PilotDetector(nn.Module):
    def __init__(
        self,
        backbone: nn.Module,
        kind: str = "rgb",
        width: int = 32,
        branch_dropout: float = 0.2,
    ):
        super().__init__()
        if kind not in KINDS or width < 4 or width % 4 or not 0 <= branch_dropout < 1:
            raise ValueError("Invalid pilot architecture parameters")
        self.kind, self.branch_dropout = kind, branch_dropout
        self.backbone = backbone
        n = int(backbone.num_features)
        self.rgb_head = nn.Linear(n, 2)
        self.auxiliary = TinyBranch(width)
        self.aux_head = nn.Linear(self.auxiliary.num_features, 2)
        self.gate = nn.Sequential(
            nn.Linear(n + self.auxiliary.num_features, 64), nn.GELU(), nn.Linear(64, 1)
        )
        nn.init.zeros_(self.gate[-1].weight)
        nn.init.constant_(self.gate[-1].bias, -2.0)
        # Remove unused trainable branches instead of inflating parameter counts.
        if kind in {"rgb", "early_concat"}:
            self.auxiliary = self.aux_head = self.gate = None
        elif kind == "spectral":
            self.backbone = self.rgb_head = self.gate = None
        elif kind == "late_mean":
            self.gate = None

    def forward_details(self, x):
        if x.ndim != 4 or x.shape[1] != 3 or not torch.isfinite(x).all():
            raise ValueError("Pilot input must be finite BCHW RGB")
        if self.kind == "early_concat":
            features = self.backbone(
                torch.cat([normalize_rgb(x), encode_tensor(x, "magnitude")], dim=1)
            )
            logits = self.rgb_head(features)
            return {"logits": logits, "rgb_logits": logits}
        if self.kind == "rgb":
            logits = self.rgb_head(self.backbone(normalize_rgb(x)))
            return {"logits": logits, "rgb_logits": logits}
        auxiliary = (
            (0.299 * x[:, 0:1] + 0.587 * x[:, 1:2] + 0.114 * x[:, 2:3]) * 2 - 1
            if self.kind == "spatial_control"
            else encode_tensor(x, "magnitude")
        )
        aux_features = self.auxiliary(auxiliary)
        aux_logits = self.aux_head(aux_features)
        if self.kind == "spectral":
            return {"logits": aux_logits, "aux_logits": aux_logits}
        features = self.backbone(normalize_rgb(x))
        rgb_logits = self.rgb_head(features)
        if self.kind == "late_mean":
            logits = (rgb_logits + aux_logits) / 2
            return {
                "logits": logits,
                "rgb_logits": rgb_logits,
                "aux_logits": aux_logits,
            }
        gate = torch.sigmoid(self.gate(torch.cat([features, aux_features], dim=1)))
        active = gate
        if self.training and self.branch_dropout:
            active = active * (torch.rand_like(active) >= self.branch_dropout)
        logits = rgb_logits + active * aux_logits
        return {
            "logits": logits,
            "rgb_logits": rgb_logits,
            "aux_logits": aux_logits,
            "gate": gate,
        }

    def forward(self, x):
        return self.forward_details(x)["logits"]


def build_model(config: dict, *, initialize_pretrained: bool = True):
    import timm

    kind = config["kind"]
    if kind not in KINDS:
        raise ValueError("Unsupported pilot kind")
    name = config["backbone"]
    channels = 4 if kind == "early_concat" else 3
    pretrained = (
        bool(config["pretrained"]) and initialize_pretrained and kind != "spectral"
    )
    backbone = timm.create_model(
        name, pretrained=pretrained, num_classes=0, in_chans=channels
    )
    return PilotDetector(
        backbone, kind, config.get("width", 32), config.get("branch_dropout", 0.2)
    )


def state_digest(model: nn.Module) -> str:
    h = hashlib.sha256()
    for name, value in sorted(model.state_dict().items()):
        tensor = value.detach().cpu().contiguous()
        h.update(name.encode())
        h.update(str(tensor.dtype).encode())
        h.update(str(tuple(tensor.shape)).encode())
        h.update(tensor.numpy().tobytes())
    return h.hexdigest()


def supervised_loss(outputs: dict, labels, aux_weight: float = 0.1):
    loss = F.cross_entropy(outputs["logits"], labels)
    if "aux_logits" in outputs and "rgb_logits" in outputs and aux_weight:
        loss = loss + aux_weight * 0.5 * (
            F.cross_entropy(outputs["aux_logits"], labels)
            + F.cross_entropy(outputs["rgb_logits"], labels)
        )
    return loss


def consistency_loss(a, b):
    p = F.softmax(a, dim=-1)
    q = F.softmax(b, dim=-1)
    m = (p + q) / 2
    tiny = torch.finfo(p.dtype).tiny
    return 0.5 * (
        (p * (p.clamp_min(tiny).log() - m.clamp_min(tiny).log())).sum(-1).mean()
        + (q * (q.clamp_min(tiny).log() - m.clamp_min(tiny).log())).sum(-1).mean()
    )


def distillation_loss(student, teacher, temperature: float = 2.0):
    if temperature <= 0:
        raise ValueError("Temperature must be positive")
    return (
        F.kl_div(
            F.log_softmax(student / temperature, dim=-1),
            F.softmax(teacher.detach() / temperature, dim=-1),
            reduction="batchmean",
        )
        * temperature**2
    )
