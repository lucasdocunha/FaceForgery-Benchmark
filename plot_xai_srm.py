"""
Figura de Explicabilidade (XAI): Comparativo DINO (Apenas RGB) vs DINO (RGB + SRM).
Mostra para onde a atenção (Grad-CAM) do modelo campeão converge com e sem resíduos forenses SRM.
"""

from __future__ import annotations

import argparse
import os
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"

from pathlib import Path
from typing import List, Tuple

import matplotlib
matplotlib.use("Agg")
import matplotlib.gridspec as gridspec
import matplotlib.pyplot as plt
import numpy as np
import torch
from PIL import Image
from torchvision import transforms

plt.rcParams.update({
    "font.family": "sans-serif",
    "font.sans-serif": ["DejaVu Sans", "Arial", "Helvetica"],
})

from src.data.data import encode_pil_image
from src.data.paths import phase1_split_root
from src.forensics.srm import extract_srm_residuals
from src.pipelines.checkpoints import load_model_from_run, run_from_checkpoint
from src.plots.heatmap import grad_cam


def make_overlay(img_np: np.ndarray, heat_t: torch.Tensor, alpha: float = 0.55, cmap_name: str = "jet") -> np.ndarray:
    cmap = plt.get_cmap(cmap_name)
    h_np = heat_t.squeeze().detach().cpu().numpy()
    h_norm = (h_np - h_np.min()) / (h_np.max() - h_np.min() + 1e-8)
    color = cmap(h_norm)[..., :3]
    return np.clip((1.0 - alpha) * img_np + alpha * color, 0, 1)


def generate_xai_comparison(
    rows_info: List[Tuple[str, Path, int]],
    ckpt_rgb_path: Path,
    ckpt_srm_path: Path,
    out_pdf: Path,
    out_png: Path,
    device: str = "cpu",
) -> None:
    print(f"Carregando modelos DINO no dispositivo: {device} ...", flush=True)
    m_rgb = load_model_from_run(run_from_checkpoint(ckpt_rgb_path), device)
    m_srm = load_model_from_run(run_from_checkpoint(ckpt_srm_path), device)

    data_rows = []
    print("Processando inferências e mapas Grad-CAM ...", flush=True)

    for label, path, target in rows_info:
        if not path.exists():
            raise FileNotFoundError(f"Imagem não encontrada: {path}")

        img = Image.open(path).convert("RGB")
        disp = np.array(img.resize((224, 224))) / 255.0
        disp_t = transforms.ToTensor()(img.resize((224, 224))).unsqueeze(0)

        x_rgb = encode_pil_image(img, "none", 224).unsqueeze(0).to(device)
        x_srm = encode_pil_image(img, "srm", 224).unsqueeze(0).to(device)

        with torch.no_grad():
            p_rgb = torch.softmax(m_rgb(x_rgb), dim=1)[0, 1].item()
            p_srm = torch.softmax(m_srm(x_srm), dim=1)[0, 1].item()

        # Extrai Grad-CAM voltado para a classe alvo de detecção de fraude
        cam_rgb = grad_cam(m_rgb, x_rgb, target_class=1)
        cam_srm = grad_cam(m_srm, x_srm, target_class=1)

        # Extrai resíduos SRM para a coluna visual de sinal forense
        srm_res = extract_srm_residuals(disp_t)[0].numpy()
        v_scales = [float(np.percentile(np.abs(srm_res[f]), 99.5) + 1e-6) for f in range(3)]
        s1 = np.clip((srm_res[0] / (2.0 * v_scales[0])) + 0.5, 0.0, 1.0)
        s2 = np.clip((srm_res[1] / (2.0 * v_scales[1])) + 0.5, 0.0, 1.0)
        s3 = np.clip((srm_res[2] / (2.0 * v_scales[2])) + 0.5, 0.0, 1.0)
        comp = np.stack([s1, s2, s3], axis=-1)

        over_rgb = make_overlay(disp, cam_rgb, alpha=0.55, cmap_name="jet")
        over_srm = make_overlay(disp, cam_srm, alpha=0.55, cmap_name="jet")

        # Mapa de diferença absoluta de atenção (Shift de Foco)
        diff = torch.abs(cam_srm - cam_rgb)
        over_diff = make_overlay(disp, diff, alpha=0.55, cmap_name="magma")

        data_rows.append({
            "label": label,
            "disp": disp,
            "comp": comp,
            "over_rgb": over_rgb,
            "over_srm": over_srm,
            "over_diff": over_diff,
            "p_rgb": p_rgb,
            "p_srm": p_srm,
            "diff_max": float(diff.max().item()),
            "diff_mean": float(diff.mean().item()),
        })

    n_rows = len(data_rows)
    N = 5

    CELL = 2.0
    TOP_PAD = 0.60
    LEFT_PAD = 1.25
    BOTTOM = 0.50
    WSPACE = 0.06
    HSPACE = 0.08

    fig_w = LEFT_PAD + N * CELL + 0.15
    width_span = N * CELL + 0.07
    _cell_w = width_span / (N + (N - 1) * WSPACE)
    fig_h = _cell_w * (n_rows + (n_rows - 1) * HSPACE) + TOP_PAD * 0.5 + BOTTOM

    fig = plt.figure(figsize=(fig_w, fig_h), facecolor="white")
    gs = gridspec.GridSpec(
        n_rows,
        N,
        figure=fig,
        hspace=HSPACE,
        wspace=WSPACE,
        left=LEFT_PAD / fig_w,
        right=(fig_w - 0.08) / fig_w,
        top=(fig_h - TOP_PAD * 0.50) / fig_h,
        bottom=BOTTOM / fig_h,
    )

    FS_TITLE = 11.0
    FS_PROB = 10.5
    FS_LABEL = 11.5
    BORDER = "#888888"

    axes = np.empty((n_rows, N), dtype=object)
    for r in range(n_rows):
        for c in range(N):
            axes[r, c] = fig.add_subplot(gs[r, c])
            axes[r, c].set_xticks([])
            axes[r, c].set_yticks([])
            for spine in axes[r, c].spines.values():
                spine.set_visible(True)
                spine.set_edgecolor(BORDER)
                spine.set_linewidth(1.8)

    col_headers = [
        "Face de Entrada\n(RGB)",
        "Resíduos Forenses\n(SRM Combinado)",
        "DINO (Apenas RGB)\nGrad-CAM",
        "DINO (RGB + SRM)\nGrad-CAM",
        "Mudança de Foco Forense\n|SRM - RGB| (Magma)",
    ]

    for r, item in enumerate(data_rows):
        axes[r, 0].imshow(item["disp"])
        axes[r, 1].imshow(item["comp"])
        axes[r, 2].imshow(item["over_rgb"])
        axes[r, 3].imshow(item["over_srm"])
        axes[r, 4].imshow(item["over_diff"])

        # Predições numéricas acima de cada célula
        p_rgb = item["p_rgb"]
        p_srm = item["p_srm"]
        axes[r, 2].set_title(
            f"P(Fake) = {p_rgb:.1%}",
            fontsize=FS_PROB,
            fontweight="bold",
            color="darkred" if p_rgb > 0.5 else "darkgreen",
            pad=5,
        )
        axes[r, 3].set_title(
            f"P(Fake) = {p_srm:.1%}",
            fontsize=FS_PROB,
            fontweight="bold",
            color="darkred" if p_srm > 0.5 else "darkgreen",
            pad=5,
        )
        axes[r, 4].set_title(
            f"Shift máx = {item['diff_max']:.2f}",
            fontsize=FS_PROB - 0.5,
            color="#444444",
            pad=5,
        )

    fig.canvas.draw()

    # Títulos principais de coluna acima da linha 0
    for c in range(N):
        pos = axes[0, c].get_position()
        fig.text(
            (pos.x0 + pos.x1) / 2,
            pos.y1 + 0.038,
            col_headers[c],
            ha="center",
            va="bottom",
            color="#1a1a1a",
            fontsize=FS_TITLE,
            fontweight="bold",
            multialignment="center",
        )

    # Rótulos de linha
    for r in range(n_rows):
        pos = axes[r, 0].get_position()
        fig.text(
            (LEFT_PAD * 0.48) / fig_w,
            (pos.y0 + pos.y1) / 2,
            data_rows[r]["label"],
            ha="center",
            va="center",
            color="#333333",
            fontsize=FS_LABEL,
            fontweight="bold",
            multialignment="center",
            linespacing=1.2,
        )

    # Linha delimitadora superior
    pos_top = axes[0, 0].get_position()
    pos_top_r = axes[0, -1].get_position()
    y_top_line = pos_top.y1 + 0.030
    fig.add_artist(
        plt.Line2D(
            [pos_top.x0, pos_top_r.x1],
            [y_top_line, y_top_line],
            transform=fig.transFigure,
            color="#bbbbbb",
            linewidth=0.8,
        )
    )

    # Separador entre Real e Fake
    if n_rows == 4:
        pos_r1 = axes[1, 0].get_position()
        pos_r2 = axes[2, 0].get_position()
        y_sep = (pos_r1.y0 + pos_r2.y1) / 2
        fig.add_artist(
            plt.Line2D(
                [pos_top.x0 - 0.03, pos_top_r.x1],
                [y_sep, y_sep],
                transform=fig.transFigure,
                color="#bbbbbb",
                linewidth=1.0,
                linestyle="--",
            )
        )

    out_pdf.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_pdf, dpi=300, bbox_inches="tight", facecolor="white", pad_inches=0.08)
    print(f"[✓] PDF XAI salvo em: {out_pdf}")

    out_png.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_png, dpi=300, bbox_inches="tight", facecolor="white", pad_inches=0.08)
    print(f"[✓] PNG XAI salvo em: {out_png}")

    plt.close(fig)


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(
        description="Gera figura de explicabilidade comparando DINO (Apenas RGB) vs DINO (RGB + SRM)."
    )
    parser.add_argument(
        "--output-pdf",
        type=Path,
        default=Path("figures/srm/dino_rgb_vs_srm_xai.pdf"),
        help="Caminho do PDF de saída.",
    )
    parser.add_argument(
        "--device",
        default="cpu",
        help="Dispositivo ('cpu' ou 'cuda:0').",
    )
    args = parser.parse_args(argv)

    clean_dir = phase1_split_root("test")
    hard_dir = Path("/media/ssd2/lucas.ocunha/datasets/phase1/test_d")

    ckpt_rgb = Path("/media/ssd2/lucas.ocunha/models-tcc/dino/none/finetune_robust/seed_123/weights/best.pth")
    ckpt_srm = Path("/media/ssd2/lucas.ocunha/models-tcc/dino/srm/finetune_robust/seed_123/weights/best.pth")

    rows_info = [
        ("Real\n(Clean)", clean_dir / "fc04255bec1f4591236af2617ee93f50.jpg", 0),
        ("Real\n(Hard - Ruído)", hard_dir / "1507f38819c5a6207f637094de4856f3.jpg", 0),
        ("Fake\n(Clean - Swap)", clean_dir / "70fac01d89420724e00dfc3e53370fa7.jpg", 1),
        ("Fake\n(Hard - Blur)", hard_dir / "747c3e721133667b2ff157f24ebfacc0.jpg", 1),
    ]

    out_pdf = args.output_pdf
    out_png = out_pdf.with_suffix(".png")

    generate_xai_comparison(
        rows_info=rows_info,
        ckpt_rgb_path=ckpt_rgb,
        ckpt_srm_path=ckpt_srm,
        out_pdf=out_pdf,
        out_png=out_png,
        device=args.device,
    )


if __name__ == "__main__":
    main()
