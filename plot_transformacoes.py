"""
Gera o painel forense de transformações para artigos científicos e apresentações.
Unifica as representações do domínio Espacial (RGB), Espectral (Fourier 2D) e
Resíduos Forenses de Esteganálise (SRM - Spatial Rich Models) em alta resolução (300 DPI).

Suporta:
  - Mega Painel Completo (10 colunas: Espacial + 5 Fourier + 4 SRM)
  - Painel de 4 Linhas (Real Clean, Real Degradado, Fake Clean, Fake Degradado)
  - Painel de 2 Linhas (Classe 0 / Real vs Classe 1 / Fake)
  - Presets: 'challenge' (Ruído em Real, Blur em Fake) ou 'default' (Clássico SIBGRAPI)
  - Formatos: PDF vetorial + PNG de alta densidade
"""

from __future__ import annotations

import argparse
import os
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"

from pathlib import Path
from typing import Dict, List, Tuple
import numpy as np
import torch
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from PIL import Image
from torchvision import transforms

from src.data.paths import phase1_split_root
from src.forensics.srm import extract_srm_residuals

plt.rcParams.update({
    "font.family": "sans-serif",
    "font.sans-serif": ["DejaVu Sans", "Arial", "Helvetica"],
})

SPATIAL_SIZE = (256, 256)
to_tensor = transforms.ToTensor()
resize = transforms.Resize(SPATIAL_SIZE)


# ── Funções de Pré-processamento e Normalização ────────────────────────────

def _gray(t: torch.Tensor) -> np.ndarray:
    """Converte tensor RGB em canal de luminância padrão ITU-R BT.601."""
    return (0.299 * t[0] + 0.587 * t[1] + 0.114 * t[2]).numpy()


def _safe_norm(a: np.ndarray) -> np.ndarray:
    d = a.max() - a.min()
    return np.zeros_like(a) if d < 1e-8 else (a - a.min()) / d


def _pclip_bounds(arr: np.ndarray, lo: float = 1, hi: float = 99) -> Tuple[float, float]:
    return float(np.percentile(arr, lo)), float(np.percentile(arr, hi))


def _apply_bounds(arr: np.ndarray, vlo: float, vhi: float) -> np.ndarray:
    return np.clip((arr - vlo) / (vhi - vlo + 1e-8), 0, 1)


# ── Extração de Transformações ─────────────────────────────────────────────

def build_panels_for_rows(
    row_specs: List[Tuple[str, Path]],
    domain: str = "mega",
) -> Tuple[List[str], List[List[Dict]]]:
    """Computa os painéis para cada amostra com escalas conjuntas normalizadas."""
    loaded_imgs = []
    loaded_tensors = []
    loaded_ffts = []
    loaded_srms = []

    for label, path in row_specs:
        p = Path(path)
        if not p.exists():
            raise FileNotFoundError(f"Imagem não encontrada: {p}")
        with Image.open(p) as pil:
            img = pil.convert("RGB").resize(SPATIAL_SIZE)
        t = to_tensor(img)
        gray = _gray(t)
        fft = np.fft.fftshift(np.fft.fft2(gray))
        srm = extract_srm_residuals(t.unsqueeze(0))[0].numpy()

        loaded_imgs.append(np.array(img) / 255.0)
        loaded_tensors.append(t)
        loaded_ffts.append(fft)
        loaded_srms.append(srm)

    # Limites compartilhados para Fourier
    all_re = np.concatenate([np.real(f).ravel() for f in loaded_ffts])
    all_im = np.concatenate([np.imag(f).ravel() for f in loaded_ffts])
    re_bounds = _pclip_bounds(all_re, 1, 99)
    im_bounds = _pclip_bounds(all_im, 1, 99)

    # Escalas compartilhadas para SRM
    v_scales = [
        float(max(np.percentile(np.abs(srm[f]), 99.5) for srm in loaded_srms) + 1e-6)
        for f in range(3)
    ]

    rows_data = []
    labels = []

    for i, (label, path) in enumerate(row_specs):
        labels.append(label)
        rgb = loaded_imgs[i]
        fft = loaded_ffts[i]
        srm = loaded_srms[i]

        panels = []

        # 1. Espacial
        if domain in ("mega", "all", "fourier", "srm"):
            panels.append({
                "img": rgb, "cmap": None, "group": "Espacial",
                "title": "Face (RGB)", "desc": "Espaço Espacial\n(Nativo)"
            })

        # 2. Fourier
        if domain in ("mega", "all", "fourier"):
            mag = _safe_norm(np.log(np.abs(fft) + 1))
            phase = (np.angle(fft) + np.pi) / (2 * np.pi)
            re = _apply_bounds(np.real(fft), *re_bounds)
            im = _apply_bounds(np.imag(fft), *im_bounds)

            H, W = fft.shape
            y, x = np.ogrid[:H, :W]
            mask = ((y - H // 2) ** 2 + (x - W // 2) ** 2) >= (min(H, W) * 0.12) ** 2
            hp = _safe_norm(np.log1p(np.abs(fft) * mask))

            panels.extend([
                {"img": mag, "cmap": "inferno", "group": "Fourier", "title": "Log-Magnitude", "desc": r"$\log(|\mathcal{F}| + 1)$"},
                {"img": phase, "cmap": "inferno", "group": "Fourier", "title": "Fase", "desc": r"$\angle\mathcal{F} \in [0,\,1]$"},
                {"img": re, "cmap": "coolwarm", "group": "Fourier", "title": "Re(F)", "desc": "Parte Real FFT\n(Bipolar)"},
                {"img": im, "cmap": "coolwarm", "group": "Fourier", "title": "Im(F)", "desc": "Parte Imag. FFT\n(Bipolar)"},
                {"img": hp, "cmap": "inferno", "group": "Fourier", "title": "Passa-Alta", "desc": r"$\log(1+|\mathcal{F}|\cdot m_r)$"},
            ])

        # 3. SRM (4 canais operacionais do modelo: 3 derivadas + composto)
        if domain in ("mega", "all", "srm"):
            srm1 = np.clip((srm[0] / (2.0 * v_scales[0])) + 0.5, 0.0, 1.0)
            srm2 = np.clip((srm[1] / (2.0 * v_scales[1])) + 0.5, 0.0, 1.0)
            srm3 = np.clip((srm[2] / (2.0 * v_scales[2])) + 0.5, 0.0, 1.0)
            srm_comp = np.stack([srm1, srm2, srm3], axis=-1)

            panels.extend([
                {"img": srm1, "cmap": "gray", "group": "SRM", "title": "SRM 1 (1ª Ord.)", "desc": r"Borda Horizontal" "\n" r"$f_1 = [1, -2, 1]/2$"},
                {"img": srm2, "cmap": "gray", "group": "SRM", "title": "SRM 2 (2ª Ord.)", "desc": r"Laplaciano $3\times 3$" "\n" r"$f_2 = \nabla^2 / 4$"},
                {"img": srm3, "cmap": "gray", "group": "SRM", "title": "SRM 3 (3ª Ord.)", "desc": r"Filtro Quadrado $5\times 5$" "\n" r"$f_3\text{ (Esteganografia)}$"},
                {"img": srm_comp, "cmap": None, "group": "SRM", "title": "SRM Composto", "desc": r"Fusão Multicanal" "\n" r"$[s_1, s_2, s_3] \to \text{RGB}$"},
            ])

        rows_data.append(panels)

    return labels, rows_data


# ── Renderizador do Grid Visual ───────────────────────────────────────────

def render_figure(
    labels: List[str],
    rows_data: List[List[Dict]],
    out_pdf: Path,
    out_png: Path | None = None,
    domain: str = "mega",
) -> None:
    n_rows = len(labels)
    n_cols = len(rows_data[0])

    CELL = 1.60
    TOP_PAD = 0.95 if domain in ("mega", "all") else 0.55
    LEFT_PAD = 1.15
    BOTTOM = 0.55
    WSPACE = 0.05
    HSPACE = 0.08

    fig_w = LEFT_PAD + n_cols * CELL + 0.15
    width_span = n_cols * CELL + 0.07
    _cell_w = width_span / (n_cols + (n_cols - 1) * WSPACE)
    fig_h = _cell_w * (n_rows + (n_rows - 1) * HSPACE) + TOP_PAD + BOTTOM

    fig = plt.figure(figsize=(fig_w, fig_h), facecolor="white")

    gs = gridspec.GridSpec(
        n_rows, n_cols,
        figure=fig,
        hspace=HSPACE,
        wspace=WSPACE,
        left=LEFT_PAD / fig_w,
        right=(fig_w - 0.08) / fig_w,
        top=(fig_h - TOP_PAD) / fig_h,
        bottom=BOTTOM / fig_h,
    )

    FS_GROUP = 11.5
    FS_TITLE = 10.0 if n_cols > 8 else 11.0
    FS_DESC = 8.8 if n_cols > 8 else 9.5
    FS_LABEL = 10.5
    BORDER = "#888888"
    TITLE_COLOR = "#1a1a1a"
    LABEL_COLOR = "#333333"
    DESC_COLOR = "#555555"
    GROUP_COLOR = "#0f2027"

    axes = np.empty((n_rows, n_cols), dtype=object)
    for r in range(n_rows):
        for c in range(n_cols):
            ax = fig.add_subplot(gs[r, c])
            axes[r, c] = ax
            panel = rows_data[r][c]
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
                spine.set_linewidth(1.4)

    fig.canvas.draw()

    # Títulos das colunas
    for c in range(n_cols):
        pos = axes[0, c].get_position()
        fig.text(
            (pos.x0 + pos.x1) / 2, pos.y1 + 0.015,
            rows_data[0][c]["title"],
            ha="center", va="bottom",
            color=TITLE_COLOR, fontsize=FS_TITLE, fontweight="bold",
        )

    # Descrições das colunas
    for c in range(n_cols):
        pos = axes[-1, c].get_position()
        fig.text(
            (pos.x0 + pos.x1) / 2, pos.y0 - 0.015,
            rows_data[0][c]["desc"],
            ha="center", va="top",
            color=DESC_COLOR, fontsize=FS_DESC,
            multialignment="center", linespacing=1.25,
        )

    # Rótulos das linhas (à esquerda)
    for r in range(n_rows):
        pos = axes[r, 0].get_position()
        fig.text(
            (LEFT_PAD * 0.45) / fig_w, (pos.y0 + pos.y1) / 2,
            labels[r],
            ha="center", va="center",
            color=LABEL_COLOR, fontsize=FS_LABEL, fontweight="bold",
            multialignment="center", linespacing=1.2,
        )

    # Super-cabeçalhos de agrupamento (para o mega painel de 10 colunas)
    if domain in ("mega", "all") and n_cols == 10:
        pos_g1_left = axes[0, 0].get_position().x0
        pos_g1_right = axes[0, 0].get_position().x1

        pos_g2_left = axes[0, 1].get_position().x0
        pos_g2_right = axes[0, 5].get_position().x1

        pos_g3_left = axes[0, 6].get_position().x0
        pos_g3_right = axes[0, 9].get_position().x1

        y_group = axes[0, 0].get_position().y1 + 0.065
        y_line_group = axes[0, 0].get_position().y1 + 0.055

        groups = [
            ("Domínio Espacial", (pos_g1_left + pos_g1_right) / 2, pos_g1_left, pos_g1_right),
            ("Decomposição em Frequência (Fourier 2D)", (pos_g2_left + pos_g2_right) / 2, pos_g2_left, pos_g2_right),
            ("Resíduos Forenses e Esteganálise (SRM)", (pos_g3_left + pos_g3_right) / 2, pos_g3_left, pos_g3_right),
        ]

        for g_title, x_center, x0, x1 in groups:
            fig.text(
                x_center, y_group,
                g_title,
                ha="center", va="bottom",
                color=GROUP_COLOR, fontsize=FS_GROUP, fontweight="bold",
            )
            fig.add_artist(plt.Line2D(
                [x0, x1], [y_line_group, y_line_group],
                transform=fig.transFigure, color=GROUP_COLOR, linewidth=1.5
            ))

        # Divisórias verticais entre grupos
        y_bottom_sep = axes[-1, 0].get_position().y0 - 0.005
        y_top_sep = y_group + 0.025

        sep_x1 = (axes[0, 0].get_position().x1 + axes[0, 1].get_position().x0) / 2
        sep_x2 = (axes[0, 5].get_position().x1 + axes[0, 6].get_position().x0) / 2

        for sep_x in (sep_x1, sep_x2):
            fig.add_artist(plt.Line2D(
                [sep_x, sep_x], [y_bottom_sep, y_top_sep],
                transform=fig.transFigure, color="#aaaaaa", linewidth=1.0, linestyle="--"
            ))
    else:
        # Linha superior padrão
        pos_top = axes[0, 0].get_position()
        pos_top_r = axes[0, -1].get_position()
        y_top_line = pos_top.y1 + 0.012
        fig.add_artist(plt.Line2D(
            [pos_top.x0, pos_top_r.x1], [y_top_line, y_top_line],
            transform=fig.transFigure, color="#bbbbbb", linewidth=0.8
        ))

    # Divisória horizontal entre Real e Fake (quando há 4 linhas)
    if n_rows == 4:
        pos_r1 = axes[1, 0].get_position()
        pos_r2 = axes[2, 0].get_position()
        y_sep_rf = (pos_r1.y0 + pos_r2.y1) / 2
        fig.add_artist(plt.Line2D(
            [axes[0, 0].get_position().x0 - 0.02, axes[0, -1].get_position().x1 + 0.01],
            [y_sep_rf, y_sep_rf],
            transform=fig.transFigure, color="#bbbbbb", linewidth=1.2, linestyle="-"
        ))

    out_pdf.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_pdf, dpi=300, bbox_inches="tight", facecolor="white", pad_inches=0.08)
    print(f"PDF salvo: {out_pdf}")

    if out_png:
        out_png.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(out_png, dpi=300, bbox_inches="tight", facecolor="white", pad_inches=0.08)
        print(f"PNG salvo: {out_png}")

    plt.close(fig)


# ── Execução Principal e CLI ──────────────────────────────────────────────

def resolve_split_dirs() -> Tuple[Path, Path]:
    clean_dir = phase1_split_root("test")
    hard_candidates = [
        Path("/media/ssd2/lucas.ocunha/datasets/phase1/test_d"),
        clean_dir.parent / "test_d",
    ]
    hard_dir = next((p for p in hard_candidates if p.exists()), hard_candidates[0])
    return clean_dir, hard_dir


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(
        description="Gera figura forense com representações Espaciais, Frequência (Fourier) e SRM."
    )
    parser.add_argument(
        "--domain",
        choices=("mega", "all", "fourier", "srm"),
        default="mega",
        help="Domínio a visualizar: 'mega' (todas as 10 transformações: Espacial + Fourier + SRM, padrão), 'fourier' ou 'srm'.",
    )
    parser.add_argument(
        "--rows",
        choices=("4", "2", "4rows", "2rows"),
        default="4",
        help="Número de linhas: '4' (Clean vs Hard para Real e Fake, padrão) ou '2' (Real vs Fake limpos).",
    )
    parser.add_argument(
        "--preset",
        choices=("challenge", "default"),
        default="challenge",
        help="Preset de imagens: 'challenge' (Noise em Real, Blur em Fake) ou 'default' (imagens clássicas SIBGRAPI).",
    )
    parser.add_argument("--real", type=str, help="Caminho/nome da imagem Real (Clean).")
    parser.add_argument("--fake", type=str, help="Caminho/nome da imagem Fake (Clean).")
    parser.add_argument("--output", type=Path, default=None, help="Caminho de saída para o PDF.")

    args = parser.parse_args(argv)

    clean_dir, hard_dir = resolve_split_dirs()

    if args.preset == "challenge":
        real_fn = args.real or "1507f38819c5a6207f637094de4856f3.jpg"  # Ruído Gaussiano severo
        fake_fn = args.fake or "747c3e721133667b2ff157f24ebfacc0.jpg"  # Blur Gaussiano severo
    else:
        real_fn = args.real or "fc04255bec1f4591236af2617ee93f50.jpg"
        fake_fn = args.fake or "70fac01d89420724e00dfc3e53370fa7.jpg"

    p_real_clean = Path(real_fn) if Path(real_fn).is_file() else clean_dir / real_fn
    p_fake_clean = Path(fake_fn) if Path(fake_fn).is_file() else clean_dir / fake_fn

    n_rows_choice = 2 if args.rows in ("2", "2rows") else 4

    if n_rows_choice == 4:
        p_real_hard = hard_dir / Path(p_real_clean).name
        p_fake_hard = hard_dir / Path(p_fake_clean).name
        row_specs = [
            ("Real\n(Clean)", p_real_clean),
            ("Real\n(Degraded)", p_real_hard),
            ("Fake\n(Clean)", p_fake_clean),
            ("Fake\n(Degraded)", p_fake_hard),
        ]
        suffix = "4rows"
    else:
        row_specs = [
            ("Class 0\n(Real)", p_real_clean),
            ("Class 1\n(Fake)", p_fake_clean),
        ]
        suffix = "2rows"

    if args.output is None:
        out_dir = Path("figures/transformacoes")
        out_dir.mkdir(parents=True, exist_ok=True)
        if args.domain in ("mega", "all"):
            out_pdf = out_dir / f"mega_transformacoes_{suffix}.pdf"
        elif args.domain == "fourier":
            out_pdf = Path("figures/fourier") / f"fourier_transformacoes_{suffix}.pdf"
        else:
            out_pdf = Path("figures/srm") / f"srm_transformacoes_{suffix}.pdf"
    else:
        out_pdf = args.output

    out_png = out_pdf.with_suffix(".png")

    labels, rows_data = build_panels_for_rows(row_specs, domain=args.domain)
    render_figure(labels, rows_data, out_pdf=out_pdf, out_png=out_png, domain=args.domain)


if __name__ == "__main__":
    main()
