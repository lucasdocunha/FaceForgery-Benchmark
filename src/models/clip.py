from __future__ import annotations
import torch
import torch.nn as nn
from transformers import CLIPVisionConfig, CLIPVisionModel
from src.models._channel_adapt import adapt_conv2d_channels

class CLIPClassifier(nn.Module):
    def __init__(self, backbone: CLIPVisionModel, dropout: float = .2):
        super().__init__(); self.backbone = backbone
        self.classifier = nn.Sequential(nn.Dropout(dropout), nn.Linear(backbone.config.hidden_size, 2))
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        output = self.backbone(pixel_values=x, output_attentions=True)
        self.last_attentions = output.attentions
        return self.classifier(output.pooler_output)

def build(config) -> nn.Module:
    if config.regime == "scratch":
        cfg = CLIPVisionConfig(image_size=config.image_size, patch_size=config.patch_size,
            hidden_size=config.hidden_size, num_hidden_layers=config.num_hidden_layers,
            num_attention_heads=config.num_attention_heads, intermediate_size=config.hidden_size * 4,
            num_channels=config.in_channels, projection_dim=config.projection_dim)
        backbone = CLIPVisionModel(cfg)
    else:
        if not config.allow_pretrained: raise ValueError("External pretrained CLIP weights are disabled")
        backbone = CLIPVisionModel.from_pretrained("openai/clip-vit-base-patch16")
        emb = backbone.vision_model.embeddings
        emb.patch_embedding = adapt_conv2d_channels(emb.patch_embedding, config.in_channels)
        backbone.config.num_channels = config.in_channels
    return CLIPClassifier(backbone, config.dropout)

def freeze_backbone(model: nn.Module) -> None:
    for p in model.backbone.parameters(): p.requires_grad = False
    for p in model.classifier.parameters(): p.requires_grad = True

def unfreeze_for_finetune(model: nn.Module, n: int) -> None:
    freeze_backbone(model)
    layers = model.backbone.vision_model.encoder.layers
    for layer in layers[-max(0, n):]:
        for p in layer.parameters(): p.requires_grad = True
