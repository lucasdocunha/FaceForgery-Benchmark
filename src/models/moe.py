from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F

from src.models._channel_adapt import adapt_conv2d_channels
from src.models import mobilenet, resnet


class MoERouter(nn.Module):
    """Lightweight convolutional gating network.

    Extracts spatial/spectral features using adaptive pooling and MLP,
    producing gating probabilities across experts.
    """

    def __init__(
        self,
        in_channels: int,
        num_experts: int,
        strategy: str = "dense",
        top_k: int = 2,
        hidden_dim: int = 64,
        temperature: float = 1.0,
    ):
        super().__init__()
        self.strategy = strategy
        self.num_experts = num_experts
        self.top_k = max(1, min(top_k, num_experts))
        self.temperature = max(1e-4, temperature)
        self.pool = nn.AdaptiveAvgPool2d((4, 4))
        self.mlp = nn.Sequential(
            nn.Flatten(),
            nn.Linear(in_channels * 4 * 4, hidden_dim),
            nn.ReLU(inplace=True),
            nn.Linear(hidden_dim, num_experts),
        )

    def forward(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        features = self.pool(x)
        logits = self.mlp(features) / self.temperature

        if self.strategy == "top_k" and self.top_k < self.num_experts:
            topk_logits, topk_indices = torch.topk(logits, self.top_k, dim=-1)
            topk_weights = F.softmax(topk_logits, dim=-1)
            weights = torch.zeros_like(logits).scatter_(-1, topk_indices, topk_weights)
        else:
            weights = F.softmax(logits, dim=-1)

        return weights, logits


def _build_expert(
    family: str,
    num_classes: int,
    in_channels: int,
    pretrained: bool,
    variant: str,
    dropout: float,
    allow_pretrained: bool,
) -> nn.Module:
    if family == "mobilenet":
        return mobilenet.mobilenet(
            num_classes=num_classes,
            in_channels=in_channels,
            pretrained=pretrained,
            variant=variant if variant in mobilenet._VARIANTS else "small",
            dropout=dropout,
            allow_pretrained=allow_pretrained,
        )
    if family == "resnet":
        arch = variant if variant in resnet._ARCHITECTURES else "resnet18"
        return resnet.resnet(
            num_classes=num_classes,
            pretrained=pretrained,
            architecture=arch,
            dropout=dropout,
            in_channels=in_channels,
            allow_pretrained=allow_pretrained,
        )
    raise ValueError(f"Unsupported expert family: {family}. Use 'mobilenet' or 'resnet'.")


class FrequencyMoE(nn.Module):
    """Frequency-Specialized Mixture of Experts (Version 1).

    Each expert receives a dedicated spatial or spectral representation of the input:
    - Expert 0: Spatial RGB (channels 0:3)
    - Expert 1: FFT Log-Magnitude (channel 3:4)
    - Expert 2: FFT Phase (channel 4:5)
    - Expert 3: FFT High-Pass Magnitude (channel 5:6)
    """

    def __init__(
        self,
        num_classes: int = 2,
        expert_family: str = "mobilenet",
        variant: str = "small",
        pretrained: bool = False,
        allow_pretrained: bool = False,
        dropout: float = 0.2,
        routing_strategy: str = "dense",
        top_k: int = 2,
    ):
        super().__init__()
        self.expert_family = expert_family
        self.slice_channels = (3, 1, 1, 1)
        self.num_experts = len(self.slice_channels)
        self.experts = nn.ModuleList([
            _build_expert(
                family=expert_family,
                num_classes=num_classes,
                in_channels=ch,
                pretrained=pretrained,
                variant=variant,
                dropout=dropout,
                allow_pretrained=allow_pretrained,
            )
            for ch in self.slice_channels
        ])
        total_in_channels = sum(self.slice_channels)
        self.router = MoERouter(
            in_channels=total_in_channels,
            num_experts=self.num_experts,
            strategy=routing_strategy,
            top_k=top_k,
        )
        self.last_routing_weights: torch.Tensor | None = None

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if x.shape[1] < 6:
            raise ValueError(
                f"FrequencyMoE expects 6-channel input (concat_frequency: RGB + Mag + Phase + HighPass), "
                f"but received tensor with {x.shape[1]} channels."
            )
        weights, _ = self.router(x)
        self.last_routing_weights = weights.detach()

        inputs = (
            x[:, 0:3, :, :],  # Spatial RGB
            x[:, 3:4, :, :],  # FFT Log-Magnitude
            x[:, 4:5, :, :],  # FFT Phase
            x[:, 5:6, :, :],  # FFT High-Pass Magnitude
        )

        expert_logits = [expert(inp) for expert, inp in zip(self.experts, inputs)]
        stacked = torch.stack(expert_logits, dim=1)  # (B, 4, num_classes)
        fused = torch.sum(stacked * weights.unsqueeze(-1), dim=1)  # (B, num_classes)
        return fused


class StandardMoE(nn.Module):
    """Standard Mixture of Experts (Version 2).

    All experts receive the exact same input tensor (e.g., standard RGB or configured mode),
    and the dynamic router learns soft or sparse gating weights over experts.
    """

    def __init__(
        self,
        num_classes: int = 2,
        in_channels: int = 3,
        num_experts: int = 4,
        expert_family: str = "mobilenet",
        variant: str = "small",
        pretrained: bool = False,
        allow_pretrained: bool = False,
        dropout: float = 0.2,
        routing_strategy: str = "dense",
        top_k: int = 2,
    ):
        super().__init__()
        self.expert_family = expert_family
        self.in_channels = in_channels
        self.num_experts = max(1, num_experts)
        self.experts = nn.ModuleList([
            _build_expert(
                family=expert_family,
                num_classes=num_classes,
                in_channels=in_channels,
                pretrained=pretrained,
                variant=variant,
                dropout=dropout,
                allow_pretrained=allow_pretrained,
            )
            for _ in range(self.num_experts)
        ])
        self.router = MoERouter(
            in_channels=in_channels,
            num_experts=self.num_experts,
            strategy=routing_strategy,
            top_k=top_k,
        )
        self.last_routing_weights: torch.Tensor | None = None

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        weights, _ = self.router(x)
        self.last_routing_weights = weights.detach()

        expert_logits = [expert(x) for expert in self.experts]
        stacked = torch.stack(expert_logits, dim=1)  # (B, num_experts, num_classes)
        fused = torch.sum(stacked * weights.unsqueeze(-1), dim=1)  # (B, num_classes)
        return fused


def freeze_backbone(model: nn.Module) -> None:
    """Freeze all expert backbones, leaving expert heads and router trainable."""
    for param in model.parameters():
        param.requires_grad = False

    # Router is always trainable
    if hasattr(model, "router"):
        for param in model.router.parameters():
            param.requires_grad = True

    # Expert classifier heads are trainable
    if hasattr(model, "experts"):
        for expert in model.experts:
            if hasattr(expert, "classifier"):
                for param in expert.classifier.parameters():
                    param.requires_grad = True
            elif hasattr(expert, "fc"):
                for param in expert.fc.parameters():
                    param.requires_grad = True


def unfreeze_for_finetune(model: nn.Module, n: int) -> None:
    """Unfreeze top classifier/router and last n blocks of each expert."""
    freeze_backbone(model)
    if not hasattr(model, "experts"):
        return
    for expert in model.experts:
        if model.expert_family == "mobilenet":
            mobilenet.unfreeze_last_blocks(expert, last_n_blocks=n)
        elif model.expert_family == "resnet":
            resnet.unfreeze_last_blocks(expert, train_layer3=n > 1)


def moe_parameter_groups(model: nn.Module, config: object) -> list[dict[str, object]]:
    """Group parameters into head/router (lr_head) and backbones (lr_backbone)."""
    head_params, backbone_params = [], []
    for name, param in model.named_parameters():
        if not param.requires_grad:
            continue
        if "router." in name or "classifier." in name or "fc." in name or "head." in name:
            head_params.append(param)
        else:
            backbone_params.append(param)

    groups: list[dict[str, object]] = []
    if backbone_params:
        groups.append({"params": backbone_params, "lr": getattr(config, "lr_backbone", 1e-4), "name": "backbone"})
    if head_params:
        groups.append({"params": head_params, "lr": getattr(config, "lr_head", 1e-3), "name": "head"})
    if not groups:
        raise ValueError("Model has no trainable parameters")
    return groups


def build_frequency_moe(config) -> nn.Module:
    return FrequencyMoE(
        num_classes=2,
        expert_family=getattr(config, "expert_family", "mobilenet"),
        variant=getattr(config, "variant", "small"),
        pretrained=config.regime == "finetune",
        allow_pretrained=config.allow_pretrained,
        dropout=config.dropout,
        routing_strategy=getattr(config, "routing_strategy", "dense"),
        top_k=getattr(config, "top_k", 2),
    )


def build_standard_moe(config) -> nn.Module:
    return StandardMoE(
        num_classes=2,
        in_channels=config.in_channels,
        num_experts=getattr(config, "num_experts", 4),
        expert_family=getattr(config, "expert_family", "mobilenet"),
        variant=getattr(config, "variant", "small"),
        pretrained=config.regime == "finetune",
        allow_pretrained=config.allow_pretrained,
        dropout=config.dropout,
        routing_strategy=getattr(config, "routing_strategy", "dense"),
        top_k=getattr(config, "top_k", 2),
    )
