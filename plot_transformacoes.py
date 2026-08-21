"""
Figura de artigo: 2 linhas (classes) × 6 colunas (transformações).
Uma imagem por classe para ilustrar cada representação.
"""

import numpy as np
import torch
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from pathlib import Path
from PIL import Image
from torchvision import transforms

plt.rcParams.update({
    "font.family":     "sans-serif",
    "font.sans-serif": ["DejaVu Sans", "Arial", "Helvetica"],
})

SOURCE_IMAGES = {
    0: "/media/ssd2/lucas.ocunha/datasets/phase1/testset/fc04255bec1f4591236af2617ee93f50.jpg",
    1: "/media/ssd2/lucas.ocunha/datasets/phase1/testset/70fac01d89420724e00dfc3e53370fa7.jpg",
}
SPATIAL_SIZE = (128, 128)

normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
freq_norm = transforms.Normalize(mean=[0.5], std=[0.5])
to_tensor = transforms.ToTensor()
resize    = transforms.Resize(SPATIAL_SIZE)


def _gray(t):
    return (0.299*t[0] + 0.587*t[1] + 0.114*t[2]).numpy()

def _safe_norm(a):
    d = a.max() - a.min()
    return np.zeros_like(a) if d < 1e-8 else (a - a.min()) / d

def _denorm_freq(ch):
    return np.clip((ch + 1) / 2, 0, 1)

def _denorm_rgb(arr):
    mean = np.array([0.485, 0.456, 0.406])[:, None, None]
    std  = np.array([0.229, 0.224, 0.225])[:, None, None]
    return np.clip(arr * std + mean, 0, 1).transpose(1, 2, 0)

def _pclip_bounds(arr, lo=1, hi=99):
    return np.percentile(arr, lo), np.percentile(arr, hi)

def _apply_bounds(arr, vlo, vhi):
    return np.clip((arr - vlo) / (vhi - vlo + 1e-8), 0, 1)


def load_fft(source_path):
    with Image.open(source_path) as pil:
        t = to_tensor(resize(pil.convert("RGB")))
    gray = _gray(t)
    return t, np.fft.fftshift(np.fft.fft2(gray))


def compute_panels(t, fft):
    re_bounds = _pclip_bounds(np.real(fft).ravel())
    im_bounds = _pclip_bounds(np.imag(fft).ravel())

    rgb = _denorm_rgb(normalize(t).numpy())

    mag = _denorm_freq(freq_norm(
        torch.tensor(_safe_norm(np.log(np.abs(fft) + 1)),
                     dtype=torch.float32).unsqueeze(0)
    ).numpy()[0])

    phase = _denorm_freq(freq_norm(
        torch.tensor((np.angle(fft) + np.pi) / (2 * np.pi),
                     dtype=torch.float32).unsqueeze(0)
    ).numpy()[0])

    re = _apply_bounds(np.real(fft), *re_bounds)
    im = _apply_bounds(np.imag(fft), *im_bounds)

    H, W = fft.shape
    y, x = np.ogrid[:H, :W]
    mask = ((y-H//2)**2 + (x-W//2)**2) >= (min(H,W)*0.12)**2
    hp   = _denorm_freq(freq_norm(
        torch.tensor(_safe_norm(np.log1p(np.abs(fft) * mask)),
                     dtype=torch.float32).unsqueeze(0)
    ).numpy()[0])

    return [
        {"img": rgb,   "cmap": None,       "title": "RGB",           "desc": "Spatial domain\n(ImageNet-normalized)"},
        {"img": mag,   "cmap": "inferno",   "title": "Log-Magnitude", "desc": r"$\log(|\mathcal{F}| + 1)$"},
        {"img": phase, "cmap": "inferno",   "title": "Phase",         "desc": r"$\angle\mathcal{F} \in [0,\,1]$"},
        {"img": re,    "cmap": "coolwarm",  "title": "Re(F)",         "desc": "Real part of FFT\n(shared scale)"},
        {"img": im,    "cmap": "coolwarm",  "title": "Im(F)",         "desc": "Imaginary part of FFT\n(shared scale)"},
        {"img": hp,    "cmap": "inferno",   "title": "High-Pass",     "desc": r"$\log(1+|\mathcal{F}|\cdot m_r)$"},
    ]


# ── Carrega imagens e computa painéis ──────────────────────────────────────
data = {}
for lbl, path in SOURCE_IMAGES.items():
    t, fft = load_fft(path)
    data[lbl] = compute_panels(t, fft)

N = len(data[0])

# ── Layout: 2 linhas × N colunas ──────────────────────────────────────────
CELL     = 1.6
TOP_PAD  = 0.70
LEFT_PAD = 0.80
BOTTOM   = 0.50

fig_w = LEFT_PAD + N * CELL + 0.15
fig_h = TOP_PAD + 2 * CELL + BOTTOM

fig = plt.figure(figsize=(fig_w, fig_h), facecolor="white")

gs = gridspec.GridSpec(
    2, N,
    figure=fig,
    hspace=0.10,
    wspace=0.06,
    left=LEFT_PAD / fig_w,
    right=(fig_w - 0.08) / fig_w,
    top=(fig_h - TOP_PAD * 0.50) / fig_h,
    bottom=BOTTOM / fig_h,
)

FS     = 11.5
FS_DESC = 10.5
BORDER = "#cccccc"
TITLE_COLOR = "#1a1a1a"
LABEL_COLOR = "#333333"
DESC_COLOR  = "#555555"

# ── Desenha células ────────────────────────────────────────────────────────
for row_idx, lbl in enumerate((0, 1)):
    for col_idx, panel in enumerate(data[lbl]):
        ax = fig.add_subplot(gs[row_idx, col_idx])
        img, cmap = panel["img"], panel["cmap"]

        if img.ndim == 3:
            ax.imshow(img, interpolation="lanczos")
        else:
            ax.imshow(img, cmap=cmap, vmin=0, vmax=1, interpolation="lanczos")

        ax.set_xticks([])
        ax.set_yticks([])
        for spine in ax.spines.values():
            spine.set_visible(True)
            spine.set_edgecolor(BORDER)
            spine.set_linewidth(0.6)

# ── Títulos de coluna (acima da linha 0) ──────────────────────────────────
gs_left  = LEFT_PAD / fig_w
gs_right = (fig_w - 0.08) / fig_w
gs_top   = (fig_h - TOP_PAD * 0.50) / fig_h

for col_idx, panel in enumerate(data[0]):
    xc = gs_left + ((col_idx + 0.5) / N) * (gs_right - gs_left)
    fig.text(
        xc, gs_top + (TOP_PAD * 0.30) / fig_h,
        panel["title"],
        ha="center", va="center",
        color=TITLE_COLOR, fontsize=FS, fontweight="bold",
        transform=fig.transFigure,
    )

# ── Descrições de coluna (abaixo da linha 1) ──────────────────────────────
gs_bottom = BOTTOM / fig_h

for col_idx, panel in enumerate(data[0]):
    xc = gs_left + ((col_idx + 0.5) / N) * (gs_right - gs_left)
    fig.text(
        xc, gs_bottom - 0.01,
        panel["desc"],
        ha="center", va="top",
        color=DESC_COLOR, fontsize=FS_DESC,
        multialignment="center", linespacing=1.35,
        transform=fig.transFigure,
    )

# ── Rótulos de linha ───────────────────────────────────────────────────────
ROW_LABELS = {0: "Class 0\n(Real)", 1: "Class 1\n(Fake)"}
row_span    = gs_top - BOTTOM / fig_h
hspace_frac = 0.10 * row_span / 2
row_h       = (row_span - hspace_frac) / 2

for row_idx, lbl in enumerate((0, 1)):
    y_center = gs_top - row_idx * (row_h + hspace_frac) - row_h / 2
    fig.text(
        (LEFT_PAD * 0.46) / fig_w, y_center,
        ROW_LABELS[lbl],
        ha="center", va="center",
        color=LABEL_COLOR, fontsize=FS, fontweight="bold",
        rotation=90, multialignment="center", linespacing=1.3,
        transform=fig.transFigure,
    )

# ── Linha horizontal separando títulos das imagens ────────────────────────
fig.add_artist(plt.Line2D(
    [gs_left, gs_right], [gs_top, gs_top],
    transform=fig.transFigure, color="#bbbbbb", linewidth=0.8,
))

plt.tight_layout()
OUT_DIR = Path(__file__).parent / "figures" / "fourier"
OUT_DIR.mkdir(parents=True, exist_ok=True)
OUT = OUT_DIR / "fourier_transformacoes.pdf"
plt.savefig(OUT, dpi=300, bbox_inches="tight", facecolor="white", pad_inches=0.08)
print(f"Salvo em: {OUT}")
