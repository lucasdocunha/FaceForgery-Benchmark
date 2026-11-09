"""Exporta as 4 amostras do MFFI (2 reais, 2 falsas, e extras até 4 cada)
a partir da partição degradada (test_d / Test-Hard) para o repositório,
gerando também um painel comparativo visual entre Limpo (testset) e Degradado (test_d).
"""

from pathlib import Path
import shutil
import pandas as pd
import numpy as np
from PIL import Image
import matplotlib.pyplot as plt
import matplotlib.patches as patches

ROOT_DIR = Path(__file__).resolve().parent.parent
CLEAN_DIR = ROOT_DIR / "data" / "datasets" / "phase1" / "testset"
DEGRADED_DIR = ROOT_DIR / "data" / "datasets" / "phase1" / "test_d"
OUT_AMOSTRAS = ROOT_DIR / "figures" / "amostras"
OUT_DEDICATED = OUT_AMOSTRAS / "mffi_degraded"

# 4 Amostras oficiais de Reais e 4 de Falsos do MFFI
SAMPLES_REAL = [
    ("46f7e69cdc47b9ed7c3efd935c88993c.jpg", "Real #1"),
    ("c985cd75cc7d5f1c34b9dc6dee371245.jpg", "Real #2"),
    ("432000deea3d09ca9ed6adb11aaf0b16.jpg", "Real #3"),
    ("95cf0509bfbfedb3113eabeb78684e0d.jpg", "Real #4"),
]

SAMPLES_FAKE = [
    ("c5decfb888b08593e980537191776a84.jpg", "Falso #1"),
    ("4a5c44d7fa626fd5ef556166f07c669a.jpg", "Falso #2"),
    ("63a47345219a2f0b8c0215e9f405c745.jpg", "Falso #3"),
    ("36d40bca5f09e2c5d59ea826f7fe5f5a.jpg", "Falso #4"),
]


def export_images():
    OUT_AMOSTRAS.mkdir(parents=True, exist_ok=True)
    OUT_DEDICATED.mkdir(parents=True, exist_ok=True)

    manifest_path = OUT_AMOSTRAS / "manifest_amostras.csv"
    if manifest_path.exists():
        df_manifest = pd.read_csv(manifest_path)
    else:
        df_manifest = pd.DataFrame(columns=["base", "classe", "indice", "arquivo", "caminho_origem", "largura", "altura"])

    new_rows = []

    # Exportar Reais
    for idx, (filename, label) in enumerate(SAMPLES_REAL, start=1):
        src_path = DEGRADED_DIR / filename
        im = Image.open(src_path).convert("RGB")
        w, h = im.size

        # Formatos em figures/amostras/
        # 1. Padrão degraded
        p_deg = OUT_AMOSTRAS / f"imagem_mffi_degraded_real_{idx}.png"
        im.save(p_deg, "PNG", optimize=True)

        # 2. Padrão degradada (PT)
        p_pt = OUT_AMOSTRAS / f"imagem_mffi_degradada_real_{idx}.png"
        im.save(p_pt, "PNG", optimize=True)

        # 3. Padrão hard
        p_hard = OUT_AMOSTRAS / f"imagem_mffi_hard_real_{idx}.png"
        im.save(p_hard, "PNG", optimize=True)

        # 4. Padrão test_d
        p_testd = OUT_AMOSTRAS / f"imagem_mffi_test_d_real_{idx}.png"
        im.save(p_testd, "PNG", optimize=True)

        # Na pasta dedicada figures/amostras/mffi_degraded/
        im.save(OUT_DEDICATED / f"mffi_degraded_real_{idx}.jpg", "JPEG", quality=95)
        im.save(OUT_DEDICATED / f"mffi_degraded_real_{idx}.png", "PNG", optimize=True)

        new_rows.append({
            "base": "mffi_test_d",
            "classe": "real",
            "indice": idx,
            "arquivo": f"imagem_mffi_degraded_real_{idx}.png",
            "caminho_origem": str(src_path),
            "largura": w,
            "altura": h,
        })
        print(f"✅ Exportado Real #{idx} ({filename}) de test_d -> {p_deg.name}")

    # Exportar Falsos
    for idx, (filename, label) in enumerate(SAMPLES_FAKE, start=1):
        src_path = DEGRADED_DIR / filename
        im = Image.open(src_path).convert("RGB")
        w, h = im.size

        # Formatos em figures/amostras/
        # 1. Padrão degraded
        p_deg = OUT_AMOSTRAS / f"imagem_mffi_degraded_fake_{idx}.png"
        im.save(p_deg, "PNG", optimize=True)

        p_pt_falso = OUT_AMOSTRAS / f"imagem_mffi_degradada_falso_{idx}.png"
        im.save(p_pt_falso, "PNG", optimize=True)

        p_pt_fake = OUT_AMOSTRAS / f"imagem_mffi_degradada_fake_{idx}.png"
        im.save(p_pt_fake, "PNG", optimize=True)

        p_hard = OUT_AMOSTRAS / f"imagem_mffi_hard_fake_{idx}.png"
        im.save(p_hard, "PNG", optimize=True)

        p_testd = OUT_AMOSTRAS / f"imagem_mffi_test_d_fake_{idx}.png"
        im.save(p_testd, "PNG", optimize=True)

        # Na pasta dedicada figures/amostras/mffi_degraded/
        im.save(OUT_DEDICATED / f"mffi_degraded_fake_{idx}.jpg", "JPEG", quality=95)
        im.save(OUT_DEDICATED / f"mffi_degraded_fake_{idx}.png", "PNG", optimize=True)

        new_rows.append({
            "base": "mffi_test_d",
            "classe": "fake",
            "indice": idx,
            "arquivo": f"imagem_mffi_degraded_fake_{idx}.png",
            "caminho_origem": str(src_path),
            "largura": w,
            "altura": h,
        })
        print(f"✅ Exportado Falso #{idx} ({filename}) de test_d -> {p_deg.name}")

    # Atualizar manifesto removendo entradas antigas de mffi_test_d se houver
    df_manifest = df_manifest[df_manifest["base"] != "mffi_test_d"]
    df_manifest = pd.concat([df_manifest, pd.DataFrame(new_rows)], ignore_index=True)
    df_manifest.to_csv(manifest_path, index=False)
    print(f"📁 Manifesto atualizado com {len(new_rows)} novas entradas degradadas: {manifest_path}")


def generate_comparison_figure():
    """Gera um painel comparativo de 4 colunas com as 4 amostras do MFFI (2 Reais, 2 Falsos)
    contrastando a versão Limpa (testset) vs Degradada (test_d) com visualização dos resíduos de diferença.
    """
    selected = [
        (SAMPLES_REAL[0][0], "MFFI Real #1", "Câmera Original"),
        (SAMPLES_REAL[1][0], "MFFI Real #2", "Câmera Original"),
        (SAMPLES_FAKE[0][0], "MFFI Falso #1", "Síntese Facial"),
        (SAMPLES_FAKE[1][0], "MFFI Falso #2", "Síntese Facial"),
    ]

    fig, axes = plt.subplots(3, 4, figsize=(16, 12), dpi=200)
    plt.subplots_adjust(top=0.90, bottom=0.04, left=0.06, right=0.98, hspace=0.18, wspace=0.08)

    fig.patch.set_facecolor("#FFFFFF")
    fig.suptitle("Comparação Visual MFFI: Partição Limpa (testset) vs Partição Degradada (test_d / Test-Hard)",
                 fontsize=15, fontweight="bold", y=0.96, color="#0B132B")

    row_labels = ["Versão Limpa (testset)", "Versão Degradada (test_d)", "Diferença Absoluta (×5)"]

    for col_idx, (fname, title, tech) in enumerate(selected):
        p_clean = CLEAN_DIR / fname
        p_deg = DEGRADED_DIR / fname

        im_c = Image.open(p_clean).convert("RGB")
        im_d = Image.open(p_deg).convert("RGB")

        arr_c = np.array(im_c).astype(np.float32)
        arr_d = np.array(im_d).astype(np.float32)

        diff = np.abs(arr_c - arr_d)
        mean_diff = diff.mean()
        diff_amplified = np.clip(diff * 5.0, 0, 255).astype(np.uint8)

        # Linha 1: Limpa
        ax_c = axes[0, col_idx]
        ax_c.imshow(im_c)
        ax_c.set_title(f"{title}\n[{tech}]", fontsize=11, fontweight="bold", color="#1E293B", pad=8)
        ax_c.set_xticks([])
        ax_c.set_yticks([])
        for spine in ax_c.spines.values():
            spine.set_edgecolor("#2E7D32" if "Real" in title else "#C62828")
            spine.set_linewidth(2)

        # Linha 2: Degradada
        ax_d = axes[1, col_idx]
        ax_d.imshow(im_d)
        ax_d.set_title(f"test_d (MAE: {mean_diff:.1f} px)", fontsize=9.5, color="#475569", pad=4)
        ax_d.set_xticks([])
        ax_d.set_yticks([])
        for spine in ax_d.spines.values():
            spine.set_edgecolor("#EA580C")
            spine.set_linewidth(2)

        # Linha 3: Diferença
        ax_diff = axes[2, col_idx]
        ax_diff.imshow(diff_amplified)
        ax_diff.set_title(f"Ruído / Compressão Injetada", fontsize=9.5, color="#475569", pad=4)
        ax_diff.set_xticks([])
        ax_diff.set_yticks([])
        for spine in ax_diff.spines.values():
            spine.set_edgecolor("#94A3B8")
            spine.set_linewidth(1)

    # Rótulos das Linhas à esquerda
    for row_idx, label in enumerate(row_labels):
        axes[row_idx, 0].set_ylabel(label, fontsize=11, fontweight="bold", color="#0F172A", labelpad=10)

    out_comp = OUT_AMOSTRAS / "comparacao_mffi_limpo_vs_degradado.png"
    plt.savefig(out_comp, bbox_inches="tight", facecolor=fig.get_facecolor())
    plt.close()
    print(f"📊 Painel comparativo salvo em: {out_comp}")

    # Salvar cópia em PDF também
    out_comp_pdf = OUT_AMOSTRAS / "comparacao_mffi_limpo_vs_degradado.pdf"
    fig, axes = plt.subplots(3, 4, figsize=(16, 12), dpi=200)
    plt.subplots_adjust(top=0.90, bottom=0.04, left=0.06, right=0.98, hspace=0.18, wspace=0.08)
    fig.patch.set_facecolor("#FFFFFF")
    fig.suptitle("Comparação Visual MFFI: Partição Limpa (testset) vs Partição Degradada (test_d / Test-Hard)",
                 fontsize=15, fontweight="bold", y=0.96, color="#0B132B")
    for col_idx, (fname, title, tech) in enumerate(selected):
        p_clean = CLEAN_DIR / fname
        p_deg = DEGRADED_DIR / fname
        im_c = Image.open(p_clean).convert("RGB")
        im_d = Image.open(p_deg).convert("RGB")
        arr_c = np.array(im_c).astype(np.float32)
        arr_d = np.array(im_d).astype(np.float32)
        diff = np.abs(arr_c - arr_d)
        mean_diff = diff.mean()
        diff_amplified = np.clip(diff * 5.0, 0, 255).astype(np.uint8)

        axes[0, col_idx].imshow(im_c)
        axes[0, col_idx].set_title(f"{title}\n[{tech}]", fontsize=11, fontweight="bold", color="#1E293B", pad=8)
        axes[0, col_idx].set_xticks([])
        axes[0, col_idx].set_yticks([])
        for spine in axes[0, col_idx].spines.values():
            spine.set_edgecolor("#2E7D32" if "Real" in title else "#C62828")
            spine.set_linewidth(2)

        axes[1, col_idx].imshow(im_d)
        axes[1, col_idx].set_title(f"test_d (MAE: {mean_diff:.1f} px)", fontsize=9.5, color="#475569", pad=4)
        axes[1, col_idx].set_xticks([])
        axes[1, col_idx].set_yticks([])
        for spine in axes[1, col_idx].spines.values():
            spine.set_edgecolor("#EA580C")
            spine.set_linewidth(2)

        axes[2, col_idx].imshow(diff_amplified)
        axes[2, col_idx].set_title(f"Ruído / Compressão Injetada", fontsize=9.5, color="#475569", pad=4)
        axes[2, col_idx].set_xticks([])
        axes[2, col_idx].set_yticks([])
        for spine in axes[2, col_idx].spines.values():
            spine.set_edgecolor("#94A3B8")
            spine.set_linewidth(1)

    for row_idx, label in enumerate(row_labels):
        axes[row_idx, 0].set_ylabel(label, fontsize=11, fontweight="bold", color="#0F172A", labelpad=10)

    plt.savefig(out_comp_pdf, bbox_inches="tight", facecolor=fig.get_facecolor())
    plt.close()
    print(f"📄 PDF comparativo salvo em: {out_comp_pdf}")


if __name__ == "__main__":
    print("Iniciando exportação das imagens degradadas do MFFI (test_d)...")
    export_images()
    print("\nGerando painel comparativo visual...")
    generate_comparison_figure()
    print("\nConcluído!")
