"""
Gera um grid Fake 1 / Fake 2 / Real 1 / Real 2 (colunas) x modelos (linhas),
com Grad-CAM/attention rollout sobre a MESMA imagem para cada modelo.

Recriação do antigo generate_heatmap_grid.py / generate_extra_same_image_heatmaps.py
(removidos no refactor) usando a API unificada em src/plots/heatmap.py.
"""
from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd
import torch
from PIL import Image
from torchvision import transforms

from src.data.data import encode_pil_image
from src.data.paths import models_root, phase1_split_root
from src.pipelines.checkpoints import config_from_run, load_model_from_run, run_from_checkpoint
from src.plots.heatmap import generate, overlay

SEED = 42
FOURIER_MODE = "none"
REGIME = "finetune"

MODELS = ["resnet", "xception", "mobilenet", "clip", "vit"]
MODEL_LABELS = {
    "resnet": "ResNet",
    "xception": "Xception",
    "mobilenet": "MobileNet",
    "clip": "CLIP",
    "vit": "ViT",
}

# (column title, row index into data/raw/test.csv)
COL_SPECS = [
    ("Fake 1", 103884),
    ("Fake 2", 159795),
    ("Real 1", 3),
    ("Real 2", 181946),
]

OUTPUT_PDF = Path("heatmap.pdf")
OUTPUT_PNG = Path("heatmap.png")
DPI = 300

FONT_HEADER = 16
FONT_ROW_LABEL = 14


def main() -> None:
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    test_rows = pd.read_csv("data/raw/test.csv")
    test_rows.columns = test_rows.columns.str.strip()
    images_dir = phase1_split_root("test")

    row_overlays: dict[str, torch.Tensor] = {}
    for family in MODELS:
        checkpoint = models_root() / family / FOURIER_MODE / REGIME / f"seed_{SEED}" / "weights" / "best.pth"
        print(f"-> {family}: {checkpoint}")
        run = run_from_checkpoint(checkpoint)
        config = config_from_run(run)
        model = load_model_from_run(run, device)

        encoded, displays = [], []
        for _, row_id in COL_SPECS:
            img_name = str(test_rows.iloc[row_id]["img_name"])
            image = Image.open(images_dir / img_name).convert("RGB")
            encoded.append(encode_pil_image(image, run.fourier_mode, config.image_size))
            displays.append(transforms.ToTensor()(transforms.Resize((config.image_size, config.image_size))(image)))

        inputs = torch.stack(encoded).to(device)
        display_batch = torch.stack(displays)
        heatmaps = generate(model, run.model_family, inputs, method="auto")
        row_overlays[family] = overlay(display_batch, heatmaps).cpu()

        del model
        if device.type == "cuda":
            torch.cuda.empty_cache()

    n_rows, n_cols = len(MODELS), len(COL_SPECS)
    fig, axes = plt.subplots(n_rows, n_cols, figsize=(n_cols * 2.8, n_rows * 2.8), squeeze=False)
    plt.subplots_adjust(wspace=0.04, hspace=0.04, left=0.08, right=0.99, top=0.94, bottom=0.01)

    for row_idx, family in enumerate(MODELS):
        for col_idx, _ in enumerate(COL_SPECS):
            ax = axes[row_idx][col_idx]
            ax.axis("off")
            image = row_overlays[family][col_idx].permute(1, 2, 0).numpy()
            ax.imshow(image)

    fig.canvas.draw()

    for row_idx, family in enumerate(MODELS):
        pos = axes[row_idx][0].get_position()
        fig.text(0.07, (pos.y0 + pos.y1) / 2, MODEL_LABELS[family],
                  ha="right", va="center", fontsize=FONT_ROW_LABEL, fontweight="bold", rotation=90)

    for col_idx, (title, _) in enumerate(COL_SPECS):
        pos = axes[0][col_idx].get_position()
        fig.text((pos.x0 + pos.x1) / 2, pos.y1 + 0.01, title,
                  ha="center", va="bottom", fontsize=FONT_HEADER, fontweight="bold")

    fig.savefig(str(OUTPUT_PDF), dpi=DPI, bbox_inches="tight")
    fig.savefig(str(OUTPUT_PNG), dpi=DPI, bbox_inches="tight")
    plt.close(fig)

    print(f"[OK] PDF -> {OUTPUT_PDF}")
    print(f"[OK] PNG -> {OUTPUT_PNG}")


if __name__ == "__main__":
    main()
