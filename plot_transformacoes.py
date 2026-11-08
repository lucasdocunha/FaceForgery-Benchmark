"""
Figura de artigo: 4 linhas (Real Clean, Real Degradado, Fake Clean, Fake Degradado) × Colunas forenses de resíduos SRM.
Configuração padrão solicitada: 1ª Derivada, 2ª Derivada, 3ª Derivada e Combinada (juntando todas).
"""

from __future__ import annotations

import argparse
import os
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"

from pathlib import Path
from typing import Dict, List, Tuple

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

from src.data.paths import phase1_split_root
from src.forensics.srm import extract_srm_residuals

SPATIAL_SIZE = (128, 128)

normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
freq_norm = transforms.Normalize(mean=[0.5], std=[0.5])
to_tensor = transforms.ToTensor()
resize = transforms.Resize(SPATIAL_SIZE)


# ── Helpers Gerais e Fourier ───────────────────────────────────────────────

def _gray(t: torch.Tensor) -> np.ndarray:
    return (0.299 * t[0] + 0.587 * t[1] + 0.114 * t[2]).numpy()


def _safe_norm(a: np.ndarray) -> np.ndarray:
    d = a.max() - a.min()
    return np.zeros_like(a) if d < 1e-8 else (a - a.min()) / d


def _denorm_freq(ch: np.ndarray) -> np.ndarray:
    return np.clip((ch + 1) / 2, 0, 1)


def _denorm_rgb(arr: np.ndarray) -> np.ndarray:
    mean = np.array([0.485, 0.456, 0.406])[:, None, None]
    std = np.array([0.229, 0.224, 0.225])[:, None, None]
    return np.clip(arr * std + mean, 0, 1).transpose(1, 2, 0)


def _pclip_bounds(arr: np.ndarray, lo: float = 1, hi: float = 99) -> Tuple[float, float]:
    return float(np.percentile(arr, lo)), float(np.percentile(arr, hi))


def _apply_bounds(arr: np.ndarray, vlo: float, vhi: float) -> np.ndarray:
    return np.clip((arr - vlo) / (vhi - vlo + 1e-8), 0, 1)


def load_fft(source_path: str | Path) -> Tuple[torch.Tensor, np.ndarray]:
    with Image.open(source_path) as pil:
        t = to_tensor(resize(pil.convert("RGB")))
    gray = _gray(t)
    return t, np.fft.fftshift(np.fft.fft2(gray))


def compute_fourier_panels(t: torch.Tensor, fft: np.ndarray) -> List[Dict]:
    re_bounds = _pclip_bounds(np.real(fft).ravel())
    im_bounds = _pclip_bounds(np.imag(fft).ravel())

    rgb = _denorm_rgb(normalize(t).numpy())

    mag = _denorm_freq(
        freq_norm(
            torch.tensor(_safe_norm(np.log(np.abs(fft) + 1)), dtype=torch.float32).unsqueeze(0)
        ).numpy()[0]
    )

    phase = _denorm_freq(
        freq_norm(
            torch.tensor((np.angle(fft) + np.pi) / (2 * np.pi), dtype=torch.float32).unsqueeze(0)
        ).numpy()[0]
    )

    re = _apply_bounds(np.real(fft), *re_bounds)
    im = _apply_bounds(np.imag(fft), *im_bounds)

    H, W = fft.shape
    y, x = np.ogrid[:H, :W]
    mask = ((y - H // 2) ** 2 + (x - W // 2) ** 2) >= (min(H, W) * 0.12) ** 2
    hp = _denorm_freq(
        freq_norm(
            torch.tensor(_safe_norm(np.log1p(np.abs(fft) * mask)), dtype=torch.float32).unsqueeze(0)
        ).numpy()[0]
    )

    return [
        {"img": rgb, "cmap": None, "title": "RGB", "desc": "Spatial domain\n(ImageNet-normalized)"},
        {"img": mag, "cmap": "inferno", "title": "Log-Magnitude", "desc": r"$\log(|\mathcal{F}| + 1)$"},
        {"img": phase, "cmap": "inferno", "title": "Phase", "desc": r"$\angle\mathcal{F} \in [0,\,1]$"},
        {"img": re, "cmap": "coolwarm", "title": "Re(F)", "desc": "Real part of FFT\n(shared scale)"},
        {"img": im, "cmap": "coolwarm", "title": "Im(F)", "desc": "Imaginary part of FFT\n(shared scale)"},
        {"img": hp, "cmap": "inferno", "title": "High-Pass", "desc": r"$\log(1+|\mathcal{F}|\cdot m_r)$"},
    ]


# ── Extração e Painéis SRM (Spatial Rich Models) ───────────────────────────

def compute_srm_panels_for_batch(
    row_specs: List[Tuple[str, Path]],
    columns: int = 4,
) -> List[Tuple[str, List[Dict]]]:
    """Computa os painéis forenses SRM com escala normalizada compartilhada entre as amostras.
    
    Opções de columns:
      - 4: 1ª Derivada, 2ª Derivada, 3ª Derivada, Combinada (Juntando todas)
      - 5: Face (RGB), 1ª Derivada, 2ª Derivada, 3ª Derivada, Combinada
      - 6: Face (RGB), 1ª Derivada, 2ª Derivada, 3ª Derivada, Combinada, Energia
    """
    raw_data = []
    all_srms = []

    for label, path in row_specs:
        if not Path(path).exists():
            raise FileNotFoundError(f"Imagem não encontrada: {path}")
        with Image.open(path) as pil:
            img_rgb = pil.convert("RGB").resize((256, 256))
        t = to_tensor(img_rgb).unsqueeze(0)
        srm = extract_srm_residuals(t)[0].numpy()  # [3, 256, 256]
        raw_data.append((label, np.array(img_rgb) / 255.0, srm))
        all_srms.append(srm)

    # Escalas compartilhadas em percentil 99.5
    v_scales = [
        float(max(np.percentile(np.abs(srm[f]), 99.5) for srm in all_srms) + 1e-6)
        for f in range(3)
    ]
    e_scale = float(max(np.percentile(np.mean(np.abs(srm), axis=0), 99.5) for srm in all_srms) + 1e-6)

    data = []
    for label, rgb, srm in raw_data:
        # Normalização dos resíduos individuais (0.5 = cinza neutro)
        srm1 = np.clip((srm[0] / (2.0 * v_scales[0])) + 0.5, 0.0, 1.0)
        srm2 = np.clip((srm[1] / (2.0 * v_scales[1])) + 0.5, 0.0, 1.0)
        srm3 = np.clip((srm[2] / (2.0 * v_scales[2])) + 0.5, 0.0, 1.0)

        # Imagem que junta todas as 3 derivadas: composto multicanal falso-colorido
        comp = np.stack([srm1, srm2, srm3], axis=-1)

        p_srm1 = {"img": srm1, "cmap": "gray", "title": "1ª Derivada\n(SRM 1)", "desc": r"Horizontal edge" "\n" r"$f_1 = [1, -2, 1]/2$"}
        p_srm2 = {"img": srm2, "cmap": "gray", "title": "2ª Derivada\n(SRM 2)", "desc": r"3×3 Laplacian" "\n" r"$f_2 = \nabla^2 / 4$"}
        p_srm3 = {"img": srm3, "cmap": "gray", "title": "3ª Derivada\n(SRM 3)", "desc": r"5×5 Steganalysis edge" "\n" r"$f_3 \text{ (Square filter)}$"}
        p_comp = {"img": comp, "cmap": None, "title": "Combinada\n(Todas)", "desc": r"Multichannel composite" "\n" r"$[s_1, s_2, s_3] \to \text{RGB}$"}

        if columns == 4:
            panels = [p_srm1, p_srm2, p_srm3, p_comp]
        elif columns == 5:
            p_rgb = {"img": rgb, "cmap": None, "title": "Face (RGB)", "desc": "Spatial domain\n(Native RGB)"}
            panels = [p_rgb, p_srm1, p_srm2, p_srm3, p_comp]
        else:
            p_rgb = {"img": rgb, "cmap": None, "title": "Face (RGB)", "desc": "Spatial domain\n(Native RGB)"}
            energy = np.clip(np.mean(np.abs(srm), axis=0) / e_scale, 0.0, 1.0)
            p_energy = {"img": energy, "cmap": "inferno", "title": "Noise Energy", "desc": r"Residual response" "\n" r"$\frac{1}{3}\sum |s_i|$"}
            panels = [p_rgb, p_srm1, p_srm2, p_srm3, p_comp, p_energy]

        data.append((label, panels))

    return data


# ── Renderizador do Grid ───────────────────────────────────────────────────

def render_grid(
    data: List[Tuple[str, List[Dict]]],
    out_pdf: Path,
    out_png: Path | None = None,
) -> None:
    n_rows = len(data)
    N = len(data[0][1])

    CELL = 1.8 if N <= 4 else 1.6
    TOP_PAD = 0.55
    LEFT_PAD = 1.15
    BOTTOM = 0.55
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

    FS = 11.5
    FS_DESC = 10.5
    BORDER = "#888888"
    TITLE_COLOR = "#1a1a1a"
    LABEL_COLOR = "#333333"
    DESC_COLOR = "#555555"

    axes = np.empty((n_rows, N), dtype=object)
    for r in range(n_rows):
        for c in range(N):
            ax = fig.add_subplot(gs[r, c])
            axes[r, c] = ax
            panel = data[r][1][c]
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
                spine.set_linewidth(1.8)

    fig.canvas.draw()

    # Títulos das colunas
    for c in range(N):
        pos = axes[0, c].get_position()
        fig.text(
            (pos.x0 + pos.x1) / 2,
            pos.y1 + 0.02,
            data[0][1][c]["title"],
            ha="center",
            va="bottom",
            color=TITLE_COLOR,
            fontsize=FS,
            fontweight="bold",
            multialignment="center",
        )

    # Descrições das colunas
    for c in range(N):
        pos = axes[-1, c].get_position()
        fig.text(
            (pos.x0 + pos.x1) / 2,
            pos.y0 - 0.02,
            data[0][1][c]["desc"],
            ha="center",
            va="top",
            color=DESC_COLOR,
            fontsize=FS_DESC,
            multialignment="center",
            linespacing=1.35,
        )

    # Rótulos das linhas perfeitamente centralizados
    for r in range(n_rows):
        pos = axes[r, 0].get_position()
        fig.text(
            (LEFT_PAD * 0.50) / fig_w,
            (pos.y0 + pos.y1) / 2,
            data[r][0],
            ha="center",
            va="center",
            color=LABEL_COLOR,
            fontsize=FS,
            fontweight="bold",
            multialignment="center",
            linespacing=1.2,
        )

    # Linha horizontal delimitadora superior
    pos_top = axes[0, 0].get_position()
    pos_top_r = axes[0, -1].get_position()
    y_top_line = pos_top.y1 + 0.012
    fig.add_artist(
        plt.Line2D(
            [pos_top.x0, pos_top_r.x1],
            [y_top_line, y_top_line],
            transform=fig.transFigure,
            color="#bbbbbb",
            linewidth=0.8,
        )
    )

    # Separador entre Real e Fake (entre linha 1 e linha 2)
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
    print(f"PDF salvo em: {out_pdf}")

    if out_png:
        out_png.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(out_png, dpi=300, bbox_inches="tight", facecolor="white", pad_inches=0.08)
        print(f"PNG salvo em: {out_png}")

    plt.close(fig)


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
        description="Gera figura em PDF com 4 linhas (Real Clean, Real Hard, Fake Clean, Fake Hard) x Colunas de resíduo SRM."
    )
    parser.add_argument(
        "--domain",
        choices=("srm", "fourier"),
        default="srm",
        help="Domínio forense: 'srm' (padrão) ou 'fourier'.",
    )
    parser.add_argument(
        "--columns",
        type=int,
        choices=(4, 5, 6),
        default=4,
        help="Quantidade de colunas para SRM: 4 (1ª, 2ª, 3ª e Combinada - padrão), 5 (+Face RGB) ou 6 (+Energia).",
    )
    parser.add_argument(
        "--preset",
        choices=("challenge", "default"),
        default="challenge",
        help="Preset de imagens: 'challenge' (Noise em Real, Blur em Fake) ou 'default' (imagens clássicas SIBGRAPI).",
    )
    parser.add_argument("--real", type=str, help="Nome do arquivo ou caminho da imagem Real (Clean).")
    parser.add_argument("--fake", type=str, help="Nome do arquivo ou caminho da imagem Fake (Clean).")
    parser.add_argument("--real-hard", type=str, help="Caminho explícito da imagem Real degradada (opcional).")
    parser.add_argument("--fake-hard", type=str, help="Caminho explícito da imagem Fake degradada (opcional).")
    parser.add_argument(
        "--output",
        type=Path,
        default=None,
        help="Caminho de saída para o PDF. Por padrão figures/srm/srm_transformacoes.pdf (ou fourier).",
    )

    args = parser.parse_args(argv)

    clean_dir, hard_dir = resolve_split_dirs()

    if args.preset == "challenge":
        real_fn = args.real or "1507f38819c5a6207f637094de4856f3.jpg"  # Heavy Noise
        fake_fn = args.fake or "747c3e721133667b2ff157f24ebfacc0.jpg"  # Gaussian Blur
    else:
        real_fn = args.real or "fc04255bec1f4591236af2617ee93f50.jpg"
        fake_fn = args.fake or "70fac01d89420724e00dfc3e53370fa7.jpg"

    p_real_clean = Path(real_fn) if Path(real_fn).is_file() else clean_dir / real_fn
    p_real_hard = Path(args.real_hard) if args.real_hard else hard_dir / Path(p_real_clean).name
    p_fake_clean = Path(fake_fn) if Path(fake_fn).is_file() else clean_dir / fake_fn
    p_fake_hard = Path(args.fake_hard) if args.fake_hard else hard_dir / Path(p_fake_clean).name

    rows = [
        ("Real\n(Clean)", p_real_clean),
        ("Real\n(Degraded)", p_real_hard),
        ("Fake\n(Clean)", p_fake_clean),
        ("Fake\n(Degraded)", p_fake_hard),
    ]

    if args.output is None:
        if args.domain == "srm":
            out_pdf = Path("figures/srm/srm_transformacoes.pdf")
        else:
            out_pdf = Path("figures/fourier/fourier_transformacoes.pdf")
    else:
        out_pdf = args.output

    out_png = out_pdf.with_suffix(".png")

    if args.domain == "srm":
        data = compute_srm_panels_for_batch(rows, columns=args.columns)
    else:
        data = []
        for label, path in rows:
            t, fft = load_fft(path)
            data.append((label, compute_fourier_panels(t, fft)))

    render_grid(data, out_pdf=out_pdf, out_png=out_png)


if __name__ == "__main__":
    main()
