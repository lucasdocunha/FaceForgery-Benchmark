from __future__ import annotations
import torch
import torch.nn as nn
from src.models._channel_adapt import replace_conv2d

class DINOVisionClassifier(nn.Module):
    def __init__(self, backbone: nn.Module, dropout: float = .2):
        super().__init__(); self.backbone = backbone
        self.classifier = nn.Sequential(nn.LayerNorm(backbone.num_features), nn.Dropout(dropout), nn.Linear(backbone.num_features, 2))
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        features = self.backbone.forward_features(x)
        if features.ndim == 4: features = features.mean((2, 3))
        elif features.ndim == 3: features = features[:, 0]
        return self.classifier(features)

def build(config) -> nn.Module:
    import timm
    names = {"tiny":"convnext_tiny.dinov3_lvd1689m", "small":"convnext_small.dinov3_lvd1689m",
             "base":"convnext_base.dinov3_lvd1689m", "large":"convnext_large.dinov3_lvd1689m"}
    pretrained = config.regime == "finetune"
    if pretrained and not config.allow_pretrained: raise ValueError("External pretrained DINO weights are disabled")
    backbone = timm.create_model(names[config.model_size], pretrained=pretrained, num_classes=0)
    if config.in_channels != 3:
        path = "stem.0" if hasattr(backbone, "stem") else "patch_embed.proj"
        replace_conv2d(backbone, path, config.in_channels)
    return DINOVisionClassifier(backbone, config.dropout)

def freeze_backbone(model: nn.Module) -> None:
    for p in model.backbone.parameters(): p.requires_grad = False
    for p in model.classifier.parameters(): p.requires_grad = True

def unfreeze_for_finetune(model: nn.Module, n: int) -> None:
    freeze_backbone(model)
    blocks = getattr(model.backbone, "stages", getattr(model.backbone, "blocks", []))
    for block in list(blocks)[-n:] if n > 0 else []:
        for p in block.parameters(): p.requires_grad = True
