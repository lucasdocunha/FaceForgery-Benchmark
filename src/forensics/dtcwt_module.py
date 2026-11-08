"""Implementação de Dual-Tree Complex Wavelet Transform (DTCWT) para perícia forense em PyTorch."""

from __future__ import annotations
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import dtcwt


class DTCWTExtractor(nn.Module):
    """Módulo para extração das 6 sub-bandas direcionais complexas via DTCWT 2D.
    
    Sub-bandas canônicas:
      1: +15° (quase horizontal)
      2: +45° (diagonal pura 1)
      3: +75° (quase vertical)
      4: -75° (quase vertical invertida)
      5: -45° (diagonal pura 2)
      6: -15° (quase horizontal invertida)
    """
    def __init__(self, mode: str = "concat_rgb", nlevels: int = 2):
        super().__init__()
        self.mode = mode
        self.nlevels = nlevels
        self.transform2d = dtcwt.Transform2d()

    def extract_luminance_np(self, x_np: np.ndarray) -> np.ndarray:
        """Converte RGB [B, 3, H, W] em luminância [B, H, W]."""
        return 0.299 * x_np[:, 0] + 0.587 * x_np[:, 1] + 0.114 * x_np[:, 2]

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Processa tensor PyTorch [B, 3, H, W] e retorna bandas DTCWT [B, C_out, H, W]."""
        device = x.device
        dtype = x.dtype
        b, c, h, w = x.shape

        x_np = x.detach().cpu().numpy()
        lum_np = self.extract_luminance_np(x_np)

        batch_magnitudes = []
        for i in range(b):
            img_gray = lum_np[i].astype(np.float32)
            res = self.transform2d.forward(img_gray, nlevels=self.nlevels)
            
            # Coleta as 6 sub-bandas do nível 0 (mais alta frequência, mais rico em bordas)
            # res.highpasses[0] tem shape [H/2, W/2, 6] com números complexos
            highpass_l0 = res.highpasses[0]
            mag_l0 = np.abs(highpass_l0)  # [H/2, W/2, 6]
            
            # Transpõe para [6, H/2, W/2]
            mag_l0 = np.transpose(mag_l0, (2, 0, 1))
            batch_magnitudes.append(mag_l0)

        batch_tensor = torch.from_numpy(np.stack(batch_magnitudes, axis=0)).to(device=device, dtype=dtype)
        # batch_tensor: [B, 6, H/2, W/2]

        # Interpola bilinearmente para resolução original [B, 6, H, W]
        dtcwt_resampled = F.interpolate(batch_tensor, size=(h, w), mode="bilinear", align_corners=False)

        # Normalização independente de escala por canal
        for ch in range(6):
            band = dtcwt_resampled[:, ch:ch+1]
            b_min = band.min()
            b_max = band.max()
            span = torch.clamp(b_max - b_min, min=1e-8)
            dtcwt_resampled[:, ch:ch+1] = (band - b_min) / span

        if self.mode == "directional_only":
            return dtcwt_resampled  # 6 canais direcionais
        elif self.mode == "concat_rgb":
            return torch.cat([x, dtcwt_resampled], dim=1)  # 9 canais (3 RGB + 6 DTCWT)
        else:
            raise ValueError(f"Modo DTCWT não suportado: {self.mode}")


def extract_dtcwt_features(img_tensor: torch.Tensor, mode: str = "directional_only") -> torch.Tensor:
    """Helper funcional para extração rápida de bandas direcionais DTCWT."""
    extractor = DTCWTExtractor(mode=mode).to(img_tensor.device)
    with torch.no_grad():
        return extractor(img_tensor)
