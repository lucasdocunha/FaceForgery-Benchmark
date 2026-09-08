"""
Salva uma imagem por modo Fourier para cada label (0 e 1).
Saída: fourier_samples/imagem_label_{label}_{mode}.png
"""

import sys
from pathlib import Path

import numpy as np
import torch
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from PIL import Image
from torchvision import transforms

# reutiliza funções do script original sem executa-lo
sys.path.insert(0, str(Path(__file__).parent))
from plot_frequencia import (
    ALL_FOURIER_MODES,
    MODE_TITLES,
    SPATIAL_SIZE,
    IMAGE_1,
    IMAGE_2,
    build_tensor,
    tensor_to_display,
    to_tensor,
)

OUT_DIR = Path(__file__).parent / "figures" / "fourier" / "samples"
OUT_DIR.mkdir(parents=True, exist_ok=True)

SAMPLES = [IMAGE_1, IMAGE_2]  # (path, label)


def load_tensor(path: str) -> torch.Tensor:
    resize = transforms.Resize(SPATIAL_SIZE)
    with Image.open(path) as im:
        return to_tensor(resize(im.convert("RGB")))


def save_mode(img_tensor: torch.Tensor, label: str, mode: str) -> None:
    panels = tensor_to_display(build_tensor(img_tensor, mode), mode)

    n = len(panels)
    fig, axes = plt.subplots(1, n, figsize=(2.8 * n, 2.8), facecolor="#0d0d0d")
    if n == 1:
        axes = [axes]

    for ax, (arr, cmap, subtitle) in zip(axes, panels):
        ax.set_facecolor("#0d0d0d")
        if arr.ndim == 3:
            ax.imshow(arr)
        else:
            ax.imshow(arr, cmap=cmap or "gray", vmin=0, vmax=1)
        ax.set_xticks([])
        ax.set_yticks([])
        for spine in ax.spines.values():
            spine.set_edgecolor("#333333")

    out = OUT_DIR / f"imagem_label_{label}_{mode}.png"
    plt.tight_layout(pad=0.4)
    plt.savefig(out, dpi=150, bbox_inches="tight", facecolor="#0d0d0d")
    plt.close(fig)
    print(f"Salvo: {out}")


def main() -> None:
    for img_path, label in SAMPLES:
        print(f"\n--- label {label}: {img_path} ---")
        tensor = load_tensor(img_path)
        for mode in ALL_FOURIER_MODES:
            save_mode(tensor, label, mode)

    print(f"\nTodos os arquivos salvos em: {OUT_DIR}/")


if __name__ == "__main__":
    main()
