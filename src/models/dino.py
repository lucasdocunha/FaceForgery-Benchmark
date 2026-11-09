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

from src.models._regime import uses_pretraining


def build(config) -> nn.Module:
    import timm
    names = {"tiny":"convnext_tiny.dinov3_lvd1689m", "small":"convnext_small.dinov3_lvd1689m",
             "base":"convnext_base.dinov3_lvd1689m", "large":"convnext_large.dinov3_lvd1689m"}
    pretrained = uses_pretraining(config, "dino")
    if pretrained and not config.allow_pretrained:
        raise ValueError("External pretrained DINO weights are disabled")
    if pretrained:
        from src.models._pretrained import get_candidate_pretrained_dirs
        candidate_dirs = get_candidate_pretrained_dirs("dino")
        backbone = None
        for d in candidate_dirs:
            p1 = d / f"{config.model_size}.pth"
            p2 = d / f"convnext_{config.model_size}.pth"
            target_path = p1 if p1.exists() else (p2 if p2.exists() else None)
            if target_path and target_path.is_file() and target_path.stat().st_size > 10 * 1024 * 1024:
                try:
                    print(f"[dino] Carregando pesos pré-treinados locais de: {target_path}", flush=True)
                    m = timm.create_model(names[config.model_size], pretrained=False, num_classes=0)
                    state = torch.load(target_path, map_location="cpu", weights_only=True)
                    m.load_state_dict(state)
                    backbone = m
                    break
                except Exception as e:
                    print(f"[dino] ⚠️ Falha ao carregar pesos de {target_path}: {e}. Tentando outros caminhos...", flush=True)
        if backbone is None:
            backbone = timm.create_model(names[config.model_size], pretrained=True, num_classes=0)
    else:
        backbone = timm.create_model(names[config.model_size], pretrained=False, num_classes=0)
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
