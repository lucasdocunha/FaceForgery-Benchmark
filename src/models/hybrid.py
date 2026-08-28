from __future__ import annotations
import torch
import torch.nn as nn
import torch.fft
from src.models._channel_adapt import replace_conv2d

class FrequencyBranch(nn.Module):
    """Extrai magnitude espectral via FFT 2D e projeta em tokens com convoluções."""
    def __init__(self, embed_dim: int = 768, in_channels: int = 3):
        super().__init__()
        self.conv_net = nn.Sequential(
            nn.Conv2d(in_channels, 64, kernel_size=3, stride=2, padding=1),
            nn.BatchNorm2d(64),
            nn.GELU(),
            nn.Conv2d(64, 128, kernel_size=3, stride=2, padding=1),
            nn.BatchNorm2d(128),
            nn.GELU(),
            nn.Conv2d(128, 256, kernel_size=3, stride=2, padding=1),
            nn.BatchNorm2d(256),
            nn.GELU(),
            nn.Conv2d(256, embed_dim, kernel_size=3, stride=2, padding=1),
            nn.BatchNorm2d(embed_dim),
            nn.GELU(),
        )
        self.proj = nn.Linear(embed_dim, embed_dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # Forçar FP32 durante a FFT para evitar overflow em FP16 (AMP)
        with torch.amp.autocast("cuda", enabled=False):
            x_fp32 = x.float()
            fft = torch.fft.rfft2(x_fp32, norm="ortho")
            # Magnitude com epsilon para estabilidade de gradiente (evitar sqrt(0) -> NaN)
            mag = torch.sqrt(fft.real.pow(2) + fft.imag.pow(2) + 1e-8)
            mag = torch.log1p(mag)
            mag = nn.functional.interpolate(
                mag, size=(x.shape[2], x.shape[3]), mode="bilinear", align_corners=False
            )
        feat_map = self.conv_net(mag.to(dtype=x.dtype))
        tokens = feat_map.flatten(2).transpose(1, 2)
        return self.proj(tokens)


class CrossAttentionFusion(nn.Module):
    """Fusão por Atenção Cruzada com Gating Adaptativo (DINOv3 = Query, FFT = Key/Value)."""
    def __init__(self, embed_dim: int = 768, num_heads: int = 8, dropout: float = 0.1):
        super().__init__()
        self.mha = nn.MultiheadAttention(embed_dim=embed_dim, num_heads=num_heads, dropout=dropout, batch_first=True)
        self.norm_q = nn.LayerNorm(embed_dim)
        self.norm_kv = nn.LayerNorm(embed_dim)
        self.gate = nn.Sequential(
            nn.Linear(embed_dim * 2, embed_dim),
            nn.Sigmoid()
        )
        self.mlp = nn.Sequential(
            nn.Linear(embed_dim, embed_dim * 2),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(embed_dim * 2, embed_dim),
        )
        self.norm_out = nn.LayerNorm(embed_dim)

    def forward(self, sem_tokens: torch.Tensor, freq_tokens: torch.Tensor) -> torch.Tensor:
        if sem_tokens.dim() == 2:
            sem_tokens = sem_tokens.unsqueeze(1)
        q = self.norm_q(sem_tokens)
        kv = self.norm_kv(freq_tokens)
        attn_out, _ = self.mha(query=q, key=kv, value=kv)
        gate_weight = self.gate(torch.cat([sem_tokens, attn_out], dim=-1))
        fused = sem_tokens + gate_weight * attn_out
        out = self.norm_out(fused + self.mlp(fused))
        return out.squeeze(1) if out.shape[1] == 1 else out.mean(dim=1)


class HybridVisionClassifier(nn.Module):
    """Classificador Híbrido Padronizado: Backbone DINOv3 + Ramo Espectral + Cross Attention."""
    def __init__(self, backbone: nn.Module, in_channels: int = 3, dropout: float = 0.2):
        super().__init__()
        self.backbone = backbone
        self.embed_dim = backbone.num_features
        self.freq_branch = FrequencyBranch(embed_dim=self.embed_dim, in_channels=in_channels)
        self.fusion = CrossAttentionFusion(embed_dim=self.embed_dim, num_heads=8, dropout=dropout)
        self.classifier = nn.Sequential(
            nn.LayerNorm(self.embed_dim),
            nn.Dropout(dropout),
            nn.Linear(self.embed_dim, 256),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(256, 2),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        features = self.backbone.forward_features(x)
        if features.ndim == 4:
            sem_tokens = features.mean((2, 3))
        elif features.ndim == 3:
            sem_tokens = features[:, 0]
        else:
            sem_tokens = features

        freq_tokens = self.freq_branch(x)
        fused = self.fusion(sem_tokens, freq_tokens)
        return self.classifier(fused)


def build(config) -> nn.Module:
    import timm
    names = {
        "tiny": "convnext_tiny.dinov3_lvd1689m",
        "small": "convnext_small.dinov3_lvd1689m",
        "base": "convnext_base.dinov3_lvd1689m",
        "large": "convnext_large.dinov3_lvd1689m",
    }
    pretrained = config.regime == "finetune"
    if pretrained and not config.allow_pretrained:
        raise ValueError("External pretrained DINO weights are disabled")
    backbone = timm.create_model(names[config.model_size], pretrained=pretrained, num_classes=0)
    if config.in_channels != 3:
        path = "stem.0" if hasattr(backbone, "stem") else "patch_embed.proj"
        replace_conv2d(backbone, path, config.in_channels)
    return HybridVisionClassifier(backbone, in_channels=config.in_channels, dropout=config.dropout)


def freeze_backbone(model: nn.Module) -> None:
    for p in model.backbone.parameters():
        p.requires_grad = False
    for p in model.freq_branch.parameters():
        p.requires_grad = True
    for p in model.fusion.parameters():
        p.requires_grad = True
    for p in model.classifier.parameters():
        p.requires_grad = True


def unfreeze_for_finetune(model: nn.Module, n: int) -> None:
    freeze_backbone(model)
    blocks = getattr(model.backbone, "stages", getattr(model.backbone, "blocks", []))
    for block in list(blocks)[-n:] if n > 0 else []:
        for p in block.parameters():
            p.requires_grad = True
