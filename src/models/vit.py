from __future__ import annotations
import torch
import torch.nn as nn
from transformers import ViTConfig, ViTModel
from src.models._channel_adapt import adapt_conv2d_channels
from src.models._hf_layers import encoder_layers

class ViTClassifier(nn.Module):
    def __init__(self, backbone: ViTModel, dropout: float = .2):
        super().__init__(); self.backbone = backbone
        self.capture_attentions = False
        self.classifier = nn.Sequential(nn.Dropout(dropout), nn.Linear(backbone.config.hidden_size, 2))
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        output = self.backbone(pixel_values=x, output_attentions=self.capture_attentions)
        self.last_attentions = output.attentions if self.capture_attentions else None
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
        # safetensors explícito para nunca cair no caminho .bin, que o transformers
        # recusa com torch < 2.6 (ver comentário em clip.py). Este repo já publica
        # model.safetensors, então aqui é só garantia.
        backbone = ViTModel.from_pretrained("google/vit-base-patch16-224", use_safetensors=True)
        pe = backbone.embeddings.patch_embeddings
        pe.projection = adapt_conv2d_channels(pe.projection, config.in_channels)
        pe.num_channels = config.in_channels
    backbone.set_attn_implementation("eager")
    return ViTClassifier(backbone, config.dropout)

def freeze_backbone(model: nn.Module) -> None:
    for p in model.backbone.parameters(): p.requires_grad = False
    for p in model.classifier.parameters(): p.requires_grad = True

def unfreeze_for_finetune(model: nn.Module, n: int) -> None:
    freeze_backbone(model)
    layers = encoder_layers(model.backbone)
    for layer in layers[-n:] if n > 0 else []:
        for p in layer.parameters(): p.requires_grad = True
