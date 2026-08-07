from __future__ import annotations
import torch
import torch.nn as nn
from transformers import ViTConfig, ViTModel
from src.models._channel_adapt import adapt_conv2d_channels

class ViTClassifier(nn.Module):
    def __init__(self, backbone: ViTModel, dropout: float = .2):
        super().__init__(); self.backbone = backbone
        self.classifier = nn.Sequential(nn.Dropout(dropout), nn.Linear(backbone.config.hidden_size, 2))
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        output = self.backbone(pixel_values=x, output_attentions=True)
        self.last_attentions = output.attentions
        return self.classifier(output.last_hidden_state[:, 0])

def build(config) -> nn.Module:
    if config.regime == "scratch":
        cfg = ViTConfig(image_size=config.image_size, patch_size=config.patch_size,
            hidden_size=config.hidden_size, num_hidden_layers=config.num_hidden_layers,
            num_attention_heads=config.num_attention_heads, intermediate_size=config.hidden_size * 4,
            num_channels=config.in_channels)
        backbone = ViTModel(cfg)
    else:
        if not config.allow_pretrained: raise ValueError("External pretrained ViT weights are disabled")
        backbone = ViTModel.from_pretrained("google/vit-base-patch16-224")
        pe = backbone.embeddings.patch_embeddings
        pe.projection = adapt_conv2d_channels(pe.projection, config.in_channels)
        pe.num_channels = config.in_channels
    return ViTClassifier(backbone, config.dropout)

def freeze_backbone(model: nn.Module) -> None:
    for p in model.backbone.parameters(): p.requires_grad = False
    for p in model.classifier.parameters(): p.requires_grad = True

def unfreeze_for_finetune(model: nn.Module, n: int) -> None:
    freeze_backbone(model)
    layers = model.backbone.encoder.layer
    for layer in layers[-max(0, n):]:
        for p in layer.parameters(): p.requires_grad = True
