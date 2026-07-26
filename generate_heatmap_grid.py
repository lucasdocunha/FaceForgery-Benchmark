"""
Gera heatmaps_same_image_grid.pdf com a MESMA imagem por classe em cada coluna,
para todos os modelos — 6 linhas (modelos) × N colunas (imagens compartilhadas).
Lê de heatmaps_same_image/<model>/ onde os arquivos não têm prefixo de modelo.
"""
from pathlib import Path
import re
import matplotlib.pyplot as plt
import matplotlib.image as mpimg

HEATMAPS_DIR = Path("heatmaps_same_image")
OUTPUT_PDF   = Path("heatmaps_same_image_grid.pdf")
OUTPUT_PNG   = Path("heatmaps_same_image_grid.png")
DPI          = 300

MODELS = ["resnet", "xception", "mobilenet", "clip", "vit", "dino"]
MODEL_LABELS = {
    "resnet":    "ResNet",
    "xception":  "Xception",
    "mobilenet": "MobileNet",
    "clip":      "CLIP",
    "vit":       "ViT",
    "dino":      "DINO",
}

FONT_HEADER    = 16
FONT_ROW_LABEL = 14

_PAT = re.compile(r"id_(?P<id>\d+)_true(?P<true>\d)_pred(?P<pred>\d)")


def _parse(filename: str) -> dict:
    m = _PAT.search(filename)
    if not m:
        return {}
    return {
        "id":   m.group("id"),
        "true": int(m.group("true")),
        "pred": int(m.group("pred")),
    }


def find_shared_ids(
    heatmaps_dir: Path, min_models: int | None = None
) -> tuple[list[str], list[str]]:
    """Return (fake_ids, real_ids) present in at least min_models subdirectories.

    Filters by ground-truth label only (pred value is ignored), so images that
    are misclassified by some models are still included.
    min_models defaults to len(MODELS) - 1 to tolerate one missing model.
    """
    if min_models is None:
        min_models = max(1, len(MODELS) - 1)

    from collections import Counter

    fake_counts: Counter[str] = Counter()
    real_counts: Counter[str] = Counter()
    for model in MODELS:
        seen_fake: set[str] = set()
        seen_real: set[str] = set()
        for img in (heatmaps_dir / model).glob("*.png"):
            info = _parse(img.name)
            if not info:
                continue
            if info["true"] == 0:
                seen_fake.add(info["id"])
            elif info["true"] == 1:
                seen_real.add(info["id"])
        for id_ in seen_fake:
            fake_counts[id_] += 1
        for id_ in seen_real:
            real_counts[id_] += 1

    common_fake = sorted(id_ for id_, n in fake_counts.items() if n >= min_models)
    common_real = sorted(id_ for id_, n in real_counts.items() if n >= min_models)
    return common_fake, common_real


def resolve_path(heatmaps_dir: Path, model: str, id_: str, true_: int) -> Path | None:
    """Find the heatmap file for a given id and ground-truth label (any pred value)."""
    for p in (heatmaps_dir / model).glob(f"id_{id_}_true{true_}_pred*.png"):
        return p
    return None


def main() -> None:
    fake_ids, real_ids = find_shared_ids(HEATMAPS_DIR)
    if not fake_ids and not real_ids:
        raise RuntimeError(
            f"Nenhum ID compartilhado entre todos os modelos em {HEATMAPS_DIR}. "
            "Execute generate_extra_same_image_heatmaps.py primeiro."
        )

    # columns: all fake IDs then all real IDs  (true label only — pred ignored)
    col_specs = [(id_, 0) for id_ in fake_ids] + [(id_, 1) for id_ in real_ids]
    n_rows = len(MODELS)
    n_cols = len(col_specs)

    fig, axes = plt.subplots(
        n_rows, n_cols,
        figsize=(n_cols * 2.8, n_rows * 2.8),
        squeeze=False,
    )

    n_fake = len(fake_ids)
    LEFT   = 0.12
    RIGHT  = 0.99
    TOP    = 0.93
    BOTTOM = 0.01

    plt.subplots_adjust(
        wspace=0.04, hspace=0.04,
        left=LEFT, right=RIGHT,
        top=TOP, bottom=BOTTOM,
    )

    for row_idx, model in enumerate(MODELS):
        for col_idx, (id_, true_) in enumerate(col_specs):
            ax = axes[row_idx][col_idx]
            ax.axis("off")
            path = resolve_path(HEATMAPS_DIR, model, id_, true_)
            if path and path.exists():
                ax.imshow(mpimg.imread(str(path)))

    fig.canvas.draw()

    # row labels
    for row_idx, model in enumerate(MODELS):
        ax = axes[row_idx][0]
        pos = ax.get_position()
        y_mid = (pos.y0 + pos.y1) / 2
        fig.text(
            LEFT - 0.01, y_mid,
            MODEL_LABELS[model],
            ha="right", va="center",
            fontsize=FONT_ROW_LABEL,
            fontweight="bold",
            rotation=90,
        )

    # column group headers
    groups: list[tuple[str, int, int]] = []
    if fake_ids:
        groups.append(("Fake", 0, n_fake - 1))
    if real_ids:
        groups.append(("Real", n_fake, n_cols - 1))

    for title, c_start, c_end in groups:
        pos_l = axes[0][c_start].get_position()
        pos_r = axes[0][c_end].get_position()
        x_mid = (pos_l.x0 + pos_r.x1) / 2
        y_top = pos_l.y1
        fig.text(
            x_mid, y_top + 0.01,
            title,
            ha="center", va="bottom",
            fontsize=FONT_HEADER,
            fontweight="bold",
        )

    fig.savefig(str(OUTPUT_PDF), dpi=DPI, bbox_inches="tight")
    fig.savefig(str(OUTPUT_PNG), dpi=DPI, bbox_inches="tight")
    plt.close(fig)

    print(f"[✓] PDF  → {OUTPUT_PDF}")
    print(f"[✓] PNG  → {OUTPUT_PNG}")


if __name__ == "__main__":
    main()
