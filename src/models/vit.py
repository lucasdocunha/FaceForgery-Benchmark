from __future__ import annotations

import torch
import torch.nn as nn

from src.models.clip import PatchEmbedding


def drop_path(x: torch.Tensor, drop_prob: float = 0.0, training: bool = False, scale_by_keep: bool = True) -> torch.Tensor:
    """Drop paths (Stochastic Depth) per sample (when applied in representation of residual blocks)."""
    if drop_prob == 0.0 or not training:
        return x
    keep_prob = 1 - drop_prob
    shape = (x.shape[0],) + (1,) * (x.ndim - 1)  # work with arbitrary dimensions
    random_tensor = x.new_empty(shape).bernoulli_(keep_prob)
    if keep_prob > 0.0 and scale_by_keep:
        random_tensor.div_(keep_prob)
    return x * random_tensor


class DropPath(nn.Module):
    """Drop paths (Stochastic Depth) per sample (when applied in representation of residual blocks)."""
    def __init__(self, drop_prob: float = 0.0, scale_by_keep: bool = True):
        super().__init__()
        self.drop_prob = drop_prob
        self.scale_by_keep = scale_by_keep

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return drop_path(x, self.drop_prob, self.training, self.scale_by_keep)


class ResidualAttentionBlock(nn.Module):
    """Residual Attention Block with Pre-LN and DropPath (Stochastic Depth)."""
    def __init__(
        self,
        d_model: int,
        nhead: int,
        dim_feedforward: int,
        dropout: float = 0.0,
        drop_path: float = 0.0,
    ):
        super().__init__()
        self.attn = nn.MultiheadAttention(d_model, nhead, dropout=dropout, batch_first=True)
        self.ln_1 = nn.LayerNorm(d_model)
        self.mlp = nn.Sequential(
            nn.Linear(d_model, dim_feedforward),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(dim_feedforward, d_model),
            nn.Dropout(dropout),
        )
        self.ln_2 = nn.LayerNorm(d_model)
        self.drop_path = DropPath(drop_path) if drop_path > 0.0 else nn.Identity()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # Pre-LN attention block with Stochastic Depth
        norm_x = self.ln_1(x)
        attn_out, _ = self.attn(norm_x, norm_x, norm_x, need_weights=False)
        x = x + self.drop_path(attn_out)
        
        # Pre-LN MLP block with Stochastic Depth
        norm_x = self.ln_2(x)
        x = x + self.drop_path(self.mlp(norm_x))
        return x


class ConvStemPatchEmbedding(nn.Module):
    """Convolutional Stem Patch Embedding (SOTA).
    Replaces the single linear projection layer with a 3-layer CNN stem to improve training stability.
    """
    def __init__(
        self,
        image_size: int,
        patch_size: int,
        hidden_size: int,
        in_channels: int = 3,
    ):
        super().__init__()
        if image_size % patch_size != 0:
            raise ValueError("image_size must be divisible by patch_size")
        self.num_patches = (image_size // patch_size) ** 2
        
        # SOTA Conv Stem (Early Convolutions)
        stem_stride = patch_size // 4
        if stem_stride < 1:
            stem_stride = 1
            
        self.stem = nn.Sequential(
            nn.Conv2d(in_channels, hidden_size // 4, kernel_size=3, stride=2, padding=1, bias=False),
            nn.BatchNorm2d(hidden_size // 4),
            nn.GELU(),
            nn.Conv2d(hidden_size // 4, hidden_size // 2, kernel_size=3, stride=2, padding=1, bias=False),
            nn.BatchNorm2d(hidden_size // 2),
            nn.GELU(),
            nn.Conv2d(hidden_size // 2, hidden_size, kernel_size=3, stride=stem_stride, padding=1, bias=False),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.stem(x)
        return x.flatten(2).transpose(1, 2)


class VisionTransformerClassifier(nn.Module):
    def __init__(
        self,
        num_classes: int = 2,
        image_size: int = 224,
        patch_size: int = 16,
        hidden_size: int = 256,
        num_hidden_layers: int = 6,
        num_attention_heads: int = 8,
        dropout: float = 0.2,
        in_channels: int = 3,
        use_conv_stem: bool = False,
        pooling: str = "cls",
        drop_path_rate: float = 0.0,
    ):
        super().__init__()
        if hidden_size % num_attention_heads != 0:
            raise ValueError("hidden_size must be divisible by num_attention_heads")
        if pooling not in ("cls", "gap"):
            raise ValueError("pooling must be 'cls' or 'gap'")

        if use_conv_stem:
            self.patch_embed = ConvStemPatchEmbedding(
                image_size,
                patch_size,
                hidden_size,
                in_channels=in_channels,
            )
        else:
            self.patch_embed = PatchEmbedding(
                image_size,
                patch_size,
                hidden_size,
                in_channels=in_channels,
            )
            
        num_tokens = self.patch_embed.num_patches + (1 if pooling == "cls" else 0)
        self.cls_token = nn.Parameter(torch.zeros(1, 1, hidden_size)) if pooling == "cls" else None
        self.pos_embed = nn.Parameter(torch.zeros(1, num_tokens, hidden_size))
        self.pos_drop = nn.Dropout(dropout)

        # Stochastic depth decay rule
        dpr = [x.item() for x in torch.linspace(0, drop_path_rate, num_hidden_layers)]

        self.encoder = nn.Sequential(
            *[
                ResidualAttentionBlock(
                    d_model=hidden_size,
                    nhead=num_attention_heads,
                    dim_feedforward=hidden_size * 4,
                    dropout=dropout,
                    drop_path=dpr[i],
                )
                for i in range(num_hidden_layers)
            ]
        )
        self.pooling = pooling
        self.norm = nn.LayerNorm(hidden_size)
        self.classifier = nn.Sequential(
            nn.LayerNorm(hidden_size),
            nn.Dropout(dropout),
            nn.Linear(hidden_size, num_classes),
        )
        self._init_weights()

    def _init_weights(self) -> None:
        if self.cls_token is not None:
            nn.init.normal_(self.cls_token, std=0.02)
        nn.init.normal_(self.pos_embed, std=0.02)
        for module in self.modules():
            if isinstance(module, (nn.Linear, nn.Conv2d)):
                nn.init.trunc_normal_(module.weight, std=0.02)
                if getattr(module, "bias", None) is not None:
                    nn.init.zeros_(module.bias)
            elif isinstance(module, (nn.LayerNorm, nn.BatchNorm2d)):
                nn.init.ones_(module.weight)
                nn.init.zeros_(module.bias)

    def forward(self, pixel_values: torch.Tensor) -> torch.Tensor:
        x = self.patch_embed(pixel_values)
        if self.pooling == "cls":
            cls = self.cls_token.expand(x.shape[0], -1, -1)
            x = torch.cat([cls, x], dim=1)
        x = self.pos_drop(x + self.pos_embed)
        x = self.encoder(x)
        x = self.norm(x)
        if self.pooling == "cls":
            return self.classifier(x[:, 0])
        else:
            return self.classifier(x.mean(dim=1))

    def freeze_backbone(self) -> None:
        for module in (self.patch_embed, self.encoder):
            for param in module.parameters():
                param.requires_grad = False
        if self.cls_token is not None:
            self.cls_token.requires_grad = False
        self.pos_embed.requires_grad = False
