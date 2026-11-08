"""Implementação de SRM (Spatial Rich Models) para extração de resíduos forenses em PyTorch."""

from __future__ import annotations
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np


def get_srm_filters() -> torch.Tensor:
    """Retorna os 3 filtros canônicos de esteganálise SRM (Spatial Rich Models) em 5x5."""
    # Filtro 1: Derivada de 1ª ordem horizontal/borda
    f1 = np.array([
        [0, 0,  0, 0, 0],
        [0, 0,  0, 0, 0],
        [0, 1, -2, 1, 0],
        [0, 0,  0, 0, 0],
        [0, 0,  0, 0, 0]
    ], dtype=np.float32) / 2.0

    # Filtro 2: Derivada de 2ª ordem (Laplaciano 3x3 embutido em 5x5)
    f2 = np.array([
        [0,  0,  0,  0, 0],
        [0,  0, -1,  0, 0],
        [0, -1,  4, -1, 0],
        [0,  0, -1,  0, 0],
        [0,  0,  0,  0, 0]
    ], dtype=np.float32) / 4.0

    # Filtro 3: Filtro quadrado de borda de 3ª ordem (esteganálise rica)
    f3 = np.array([
        [-1,  2,  -2,  2, -1],
        [ 2, -6,   8, -6,  2],
        [-2,  8, -12,  8, -2],
        [ 2, -6,   8, -6,  2],
        [-1,  2,  -2,  2, -1]
    ], dtype=np.float32) / 12.0

    # Stack: [3, 1, 5, 5]
    weights = np.stack([f1, f2, f3], axis=0)[:, np.newaxis, :, :]
    return torch.from_numpy(weights)


class SRMConv2d(nn.Module):
    """Módulo convolucional fixo que extrai resíduos de ruído de alta frequência via SRM.
    
    Aplica 3 filtros passa-alta sobre o canal de luminância (ou canais RGB)
    seguido de truncamento não-linear (clamping) T=3.0, suprimindo semântica
    e isolando descontinuidades de interpolação e manipulação.
    """
    def __init__(self, mode: str = "concat_rgb", truncation: float = 3.0):
        super().__init__()
        self.mode = mode
        self.truncation = truncation
        
        # Filtros SRM [3, 1, 5, 5]
        srm_kernel = get_srm_filters()
        self.register_buffer("srm_kernel", srm_kernel)

    def extract_luminance(self, x: torch.Tensor) -> torch.Tensor:
        """Converte RGB [B, 3, H, W] em luminância [B, 1, H, W]."""
        r, g, b = x[:, 0:1], x[:, 1:2], x[:, 2:3]
        return 0.299 * r + 0.587 * g + 0.114 * b

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Processa tensor [B, 3, H, W] e retorna tensor com resíduos SRM."""
        lum = self.extract_luminance(x)
        
        # Convolução com padding=2 para manter mesmas dimensões espaciais [B, 3, H, W]
        residuals = F.conv2d(lum, self.srm_kernel, padding=2)
        
        # Truncamento não-linear característico de esteganálise
        if self.truncation > 0:
            residuals = torch.clamp(residuals, -self.truncation, self.truncation)
            # Normalização para escala aproximada [-1, 1]
            residuals = residuals / self.truncation

        if self.mode == "residual_only":
            return residuals  # 3 canais de resíduo
        elif self.mode == "concat_rgb":
            return torch.cat([x, residuals], dim=1)  # 6 canais (3 RGB + 3 SRM)
        else:
            raise ValueError(f"Modo SRM não suportado: {self.mode}")


def extract_srm_residuals(img_tensor: torch.Tensor) -> torch.Tensor:
    """Helper funcional para extração rápida de resíduos SRM em tensor [B, 3, H, W]."""
    conv = SRMConv2d(mode="residual_only").to(img_tensor.device)
    with torch.no_grad():
        return conv(img_tensor)
