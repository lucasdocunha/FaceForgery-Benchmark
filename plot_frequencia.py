"""
Visualiza todos os modos Fourier do ImageDataset para duas imagens lado a lado.

Uso:
    python plot_fourier_modes.py \
        --img1 caminho/para/img1.jpg --label1 "Real" \
        --img2 caminho/para/img2.jpg --label2 "Falsa"

Ou edite as constantes IMAGE_1 / IMAGE_2 abaixo diretamente.
"""

import argparse
import numpy as np
import torch
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from PIL import Image
from torchvision import transforms
from pathlib import Path
from typing import Literal

# ── Cole aqui caminhos padrão para testes rápidos sem argparse ─────────────
IMAGE_1 = ("/media/ssd2/lucas.ocunha/datasets/phase1/testset/fc04255bec1f4591236af2617ee93f50.jpg", "0")
IMAGE_2 = ("/media/ssd2/lucas.ocunha/datasets/phase1/testset/70fac01d89420724e00dfc3e53370fa7.jpg", "1")
# ───────────────────────────────────────────────────────────────────────────

FourierMode = Literal[
    "none", "magnitude", "phase", "complex",
    "concat", "frequency_3", "concat_frequency",
]
 
ALL_FOURIER_MODES: tuple[FourierMode, ...] = (
    "none", "magnitude", "phase", "complex",
    "concat", "frequency_3",
)
 
MODE_TITLES = {
    "none":             "Espacial\n(RGB ImageNet)",
    "magnitude":        "Magnitude FFT\nlog(|F|+1)",
    "phase":            "Fase FFT\n[0,1]",
    "complex":          "FFT Complexo\nRe / Im",
    "concat":           "Concat\nRGB + Mag",
    "frequency_3":      "Passa-Alta\n(|F| × máscara)",
    "concat_frequency": "Concat Freq\nRGB + Mag + Fase + HP",
}
 
SPATIAL_SIZE = (128, 128)
 
normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
freq_norm  = transforms.Normalize(mean=[0.5], std=[0.5])
to_tensor  = transforms.ToTensor()
 
 
# ── helpers FFT (espelha a classe ImageDataset) ─────────────────────────────
 
def _to_grayscale(img: torch.Tensor) -> np.ndarray:
    if img.shape[0] == 3:
        return (0.299 * img[0] + 0.587 * img[1] + 0.114 * img[2]).numpy()
    return img.squeeze(0).numpy()
 
 
def _safe_norm(arr: np.ndarray) -> np.ndarray:
    den = arr.max() - arr.min()
    return np.zeros_like(arr) if den < 1e-8 else (arr - arr.min()) / den
 
 
def _fft_magnitude(img: torch.Tensor) -> torch.Tensor:
    gray = _to_grayscale(img)
    mag  = np.log(np.abs(np.fft.fftshift(np.fft.fft2(gray))) + 1)
    t    = torch.tensor(_safe_norm(mag), dtype=torch.float32).unsqueeze(0)
    return torch.nan_to_num(t, nan=0., posinf=1., neginf=0.)
 
 
def _fft_phase(img: torch.Tensor) -> torch.Tensor:
    gray  = _to_grayscale(img)
    phase = np.angle(np.fft.fftshift(np.fft.fft2(gray)))
    phase = (phase + np.pi) / (2 * np.pi)
    t     = torch.tensor(phase, dtype=torch.float32).unsqueeze(0)
    return torch.nan_to_num(t, nan=0., posinf=1., neginf=0.)
 
 
def _fft_complex(img: torch.Tensor) -> torch.Tensor:
    gray   = _to_grayscale(img)
    fshift = np.fft.fftshift(np.fft.fft2(gray))
    real   = torch.tensor(_safe_norm(np.real(fshift)), dtype=torch.float32)
    imag   = torch.tensor(_safe_norm(np.imag(fshift)), dtype=torch.float32)
    t      = torch.stack([real, imag], dim=0)
    return torch.nan_to_num(t, nan=0., posinf=1., neginf=0.)
 
 
def _fft_highpass(img: torch.Tensor) -> torch.Tensor:
    gray   = _to_grayscale(img)
    fshift = np.fft.fftshift(np.fft.fft2(gray))
    H, W   = fshift.shape
    y, x   = np.ogrid[:H, :W]
    radius = min(H, W) * 0.12
    mask   = ((y - H//2)**2 + (x - W//2)**2) >= radius**2
    hp     = _safe_norm(np.log1p(np.abs(fshift) * mask))
    t      = torch.tensor(hp, dtype=torch.float32).unsqueeze(0)
    return torch.nan_to_num(t, nan=0., posinf=1., neginf=0.)
 
 
def build_tensor(img_tensor: torch.Tensor, mode: FourierMode) -> torch.Tensor:
    image = normalize(img_tensor)
    if mode == "none":
        return image
    elif mode == "magnitude":
        return freq_norm(_fft_magnitude(img_tensor))
    elif mode == "phase":
        return freq_norm(_fft_phase(img_tensor))
    elif mode == "complex":
        return _fft_complex(img_tensor)
    elif mode == "frequency_3":
        return freq_norm(_fft_highpass(img_tensor))
    elif mode == "concat":
        return torch.cat([image, freq_norm(_fft_magnitude(img_tensor))], dim=0)
    elif mode == "concat_frequency":
        return torch.cat([
            image,
            freq_norm(_fft_magnitude(img_tensor)),
            freq_norm(_fft_phase(img_tensor)),
            freq_norm(_fft_highpass(img_tensor)),
        ], dim=0)
    raise ValueError(f"Modo inválido: {mode}")
 
 
# ── conversão tensor → imagem exibível ─────────────────────────────────────
 
def tensor_to_display(t: torch.Tensor, mode: FourierMode):
    """
    Retorna (imagem numpy para imshow, cmap, título dos canais).
    """
    c = t.shape[0]
    arr = t.detach().cpu().numpy()
 
    if mode == "none":
        # Desfaz normalização ImageNet para visualizar RGB bonito
        mean = np.array([0.485, 0.456, 0.406])[:, None, None]
        std  = np.array([0.229, 0.224, 0.225])[:, None, None]
        rgb  = np.clip(arr * std + mean, 0, 1).transpose(1, 2, 0)
        return [(rgb, None, "RGB")]
 
    elif mode in ("magnitude", "phase", "frequency_3"):
        ch = arr[0]
        ch = np.clip((ch + 1) / 2, 0, 1)          # desfaz Normalize([0.5],[0.5])
        return [(ch, "inferno", "canal único")]
 
    elif mode == "complex":
        titles = ["Re(F)", "Im(F)"]
        return [
            (np.clip((arr[i] + 1) / 2, 0, 1), "coolwarm", titles[i])
            for i in range(2)
        ]
 
    elif mode == "concat":
        # Canais 0-2: RGB, canal 3: magnitude
        mean = np.array([0.485, 0.456, 0.406])[:, None, None]
        std  = np.array([0.229, 0.224, 0.225])[:, None, None]
        rgb  = np.clip(arr[:3] * std + mean, 0, 1).transpose(1, 2, 0)
        mag  = np.clip((arr[3] + 1) / 2, 0, 1)
        return [(rgb, None, "RGB"), (mag, "inferno", "Magnitude")]
 
    elif mode == "concat_frequency":
        mean = np.array([0.485, 0.456, 0.406])[:, None, None]
        std  = np.array([0.229, 0.224, 0.225])[:, None, None]
        rgb  = np.clip(arr[:3] * std + mean, 0, 1).transpose(1, 2, 0)
        fft_titles = ["Magnitude", "Fase", "Passa-Alta"]
        panels = [(rgb, None, "RGB")]
        for i, title in enumerate(fft_titles):
            ch = np.clip((arr[3 + i] + 1) / 2, 0, 1)
            panels.append((ch, "inferno", title))
        return panels
 
    # fallback
    return [(arr[0], "gray", f"c0")]
 
 
# ── plot principal ──────────────────────────────────────────────────────────
 
def plot_all_modes(
    img1_path: str, label1: str,
    img2_path: str, label2: str,
    spatial_size: tuple[int, int] = SPATIAL_SIZE,
    figsize_per_panel: float = 2.2,
):
    resize = transforms.Resize(spatial_size)
 
    def load(path):
        with Image.open(path) as im:
            im = im.convert("RGB")
            return resize(im)
 
    pil1 = load(img1_path)
    pil2 = load(img2_path)
    t1   = to_tensor(pil1)
    t2   = to_tensor(pil2)
 
    # Pré-calcula painéis: dict[mode] → list de (arr, cmap, subtitle)
    panels1 = {m: tensor_to_display(build_tensor(t1, m), m) for m in ALL_FOURIER_MODES}
    panels2 = {m: tensor_to_display(build_tensor(t2, m), m) for m in ALL_FOURIER_MODES}
 
    # Número de sub-colunas por modo (máximo de painéis que um modo gera)
    subcols = {m: max(len(panels1[m]), len(panels2[m])) for m in ALL_FOURIER_MODES}
    total_cols = sum(subcols.values())
 
    fig_w = figsize_per_panel * total_cols
    fig_h = figsize_per_panel * 2 + 1.4          # 2 linhas + espaço para títulos
 
    fig = plt.figure(figsize=(fig_w, fig_h), facecolor="#0d0d0d")
    fig.suptitle(
        "Comparação de Modos Fourier",
        color="white", fontsize=14, fontweight="bold", y=1.01,
    )
 
    # GridSpec com sub-colunas variáveis por grupo de modo
    gs = gridspec.GridSpec(
        2, total_cols,
        figure=fig,
        hspace=0.08,
        wspace=0.04,
        left=0.01, right=0.99, top=0.96, bottom=0.04,
    )
 
    col_offset = 0
    for mode in ALL_FOURIER_MODES:
        n = subcols[mode]
        mid_col = col_offset + n // 2
 
        for row_idx, (panels, label) in enumerate([(panels1, label1), (panels2, label2)]):
            for sub_i, (arr, cmap, sub_title) in enumerate(panels[mode]):
                ax = fig.add_subplot(gs[row_idx, col_offset + sub_i])
                ax.set_facecolor("#0d0d0d")
 
                if arr.ndim == 3:
                    ax.imshow(arr)
                else:
                    ax.imshow(arr, cmap=cmap or "gray", vmin=0, vmax=1)
 
                ax.set_xticks([])
                ax.set_yticks([])
 
                # Subtítulo de canal (Re, Im, RGB…) — linha pequena
                ax.set_title(sub_title, color="#aaaaaa", fontsize=6.5, pad=2)
 
                for spine in ax.spines.values():
                    spine.set_edgecolor("#333333")
 
            # Label da imagem à esquerda (apenas na primeira sub-coluna do modo)
            if mode == ALL_FOURIER_MODES[0]:
                ax0 = fig.add_subplot(gs[row_idx, col_offset])
                ax0.set_facecolor("#0d0d0d")
                # rótulo externo à grade
                fig.text(
                    0.002, 0.75 - row_idx * 0.5,
                    label,
                    color="white", fontsize=9, fontweight="bold",
                    va="center", rotation=90,
                )
 
        # Título do grupo de modo (centralizado sobre suas sub-colunas)
        fig.text(
            (col_offset + n / 2) / total_cols,
            0.995,
            MODE_TITLES[mode],
            ha="center", va="bottom",
            color="#e0e0e0", fontsize=7.5, fontweight="bold",
            transform=fig.transFigure,
        )
 
        # Linha separadora entre grupos de modo
        if col_offset > 0:
            x_line = col_offset / total_cols
            line = plt.Line2D(
                [x_line, x_line], [0.03, 0.99],
                transform=fig.transFigure,
                color="#333333", linewidth=0.8, linestyle="--",
            )
            fig.add_artist(line)
 
        col_offset += n
 
    plt.tight_layout(pad=0.3)
    plt.savefig("fourier_modes_comparison.png", dpi=150, bbox_inches="tight",
                facecolor="#0d0d0d")
    print("Salvo em: fourier_modes_comparison.png")
    plt.show()
 
 
# ── CLI ─────────────────────────────────────────────────────────────────────
 
def parse_args():
    p = argparse.ArgumentParser(description="Plot todos os modos Fourier para duas imagens.")
    p.add_argument("--img1",   default=IMAGE_1[0], help="Caminho da imagem 1")
    p.add_argument("--label1", default=IMAGE_1[1], help="Label da imagem 1")
    p.add_argument("--img2",   default=IMAGE_2[0], help="Caminho da imagem 2")
    p.add_argument("--label2", default=IMAGE_2[1], help="Label da imagem 2")
    return p.parse_args()
 
 
if __name__ == "__main__":
    args = parse_args()
    plot_all_modes(
        img1_path=args.img1, label1=args.label1,
        img2_path=args.img2, label2=args.label2,
    )
 