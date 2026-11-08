"""Script para geração de PDF com amostras reais e falsas das bases MFFI, Celeb-DF v2 e DF-40.
Gera um documento profissional de 4 páginas:
- Página 1: Visão Geral Comparativa (Grid 3x4 com todas as 12 imagens)
- Página 2: Detalhamento MFFI (Multi-Family Facial Inconsistency)
- Página 3: Detalhamento Celeb-DF v2 (DeepFake Forensics Benchmark)
- Página 4: Detalhamento DF-40 (DeepFake-40 Benchmark)
"""

import os
import sys
import pandas as pd
import numpy as np
from PIL import Image
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from matplotlib.backends.backend_pdf import PdfPages

# Paleta de Cores Institucional
COLOR_REAL = "#1B5E20"          # Verde escuro para texto/bordas
COLOR_REAL_BG = "#E8F5E9"       # Verde pastel suave para badges
COLOR_REAL_BORDER = "#2E7D32"
COLOR_FAKE = "#B71C1C"          # Vermelho escuro para texto/bordas
COLOR_FAKE_BG = "#FFEBEE"       # Vermelho pastel suave para badges
COLOR_FAKE_BORDER = "#C62828"
COLOR_TEXT_MAIN = "#0F172A"     # Slate 900
COLOR_TEXT_MUTED = "#475569"    # Slate 600
COLOR_HEADER_BG = "#0B132B"     # Azul marinho profundo
COLOR_CARD_BG = "#FFFFFF"
COLOR_CARD_BORDER = "#CBD5E1"   # Slate 300

def get_samples_metadata():
    # 1. MFFI
    df_mffi = pd.read_csv("data/manifests/test.csv")
    root_mffi = "data/datasets/phase1/testset"
    mffi_r1 = df_mffi[df_mffi["label"] == 0].iloc[0]["img_name"]
    mffi_r2 = df_mffi[df_mffi["label"] == 0].iloc[1]["img_name"]
    mffi_f1 = df_mffi[df_mffi["label"] == 1].iloc[0]["img_name"]
    mffi_f2 = df_mffi[df_mffi["label"] == 1].iloc[1]["img_name"]

    # 2. Celeb-DF v2
    root_celeb = "data/datasets/celeb_df_crops"
    celeb_r1 = "id1_0007_f00020.jpg"        # Celeb-real
    celeb_r2 = "00208_f00028.jpg"           # YouTube-real
    celeb_f1 = "id1_id0_0007_f00020.jpg"    # Celeb-synthesis (id1 -> id0)
    celeb_f2 = "id3_id1_0001_f00023.jpg"    # Celeb-synthesis (id3 -> id1)

    # 3. DF-40
    df_df40 = pd.read_csv("data/df40/test.csv")
    df40_r1_row = df_df40[df_df40["method"] == "Real_whichfaceisreal"].iloc[0]
    df40_r2_row = df_df40[df_df40["method"] == "Real_heygen"].iloc[0]
    df40_f1_row = df_df40[df_df40["method"] == "faceswap"].iloc[0]
    df40_f2_row = df_df40[df_df40["method"] == "StyleGAN3"].iloc[0]

    datasets = [
        {
            "name": "MFFI (Multi-Family Facial Inconsistency)",
            "short_name": "MFFI",
            "accent": "#0D47A1",
            "description": "Base primária de treino e validação com múltiplas famílias de manipulação e imagens de alta resolução alinhadas.",
            "samples": [
                {
                    "type": "REAL",
                    "title": "Real #1",
                    "path": os.path.join(root_mffi, mffi_r1),
                    "id": mffi_r1,
                    "detail": "Split de Teste — Amostra Genuína",
                    "technique": "Câmera Original / Genuíno"
                },
                {
                    "type": "REAL",
                    "title": "Real #2",
                    "path": os.path.join(root_mffi, mffi_r2),
                    "id": mffi_r2,
                    "detail": "Split de Teste — Amostra Genuína",
                    "technique": "Câmera Original / Genuíno"
                },
                {
                    "type": "FALSO",
                    "title": "Falso #1",
                    "path": os.path.join(root_mffi, mffi_f1),
                    "id": mffi_f1,
                    "detail": "Split de Teste — Manipulação Facial",
                    "technique": "Síntese / Manipulação Facial"
                },
                {
                    "type": "FALSO",
                    "title": "Falso #2",
                    "path": os.path.join(root_mffi, mffi_f2),
                    "id": mffi_f2,
                    "detail": "Split de Teste — Manipulação Facial",
                    "technique": "Síntese / Manipulação Facial"
                }
            ]
        },
        {
            "name": "Celeb-DF v2 (DeepFake Forensics Benchmark)",
            "short_name": "Celeb-DF v2",
            "accent": "#5B21B6",
            "description": "Benchmark de generalização cross-dataset focado em trocas de faces (face swaps) em vídeos realistas de celebridades.",
            "samples": [
                {
                    "type": "REAL",
                    "title": "Real #1 (Celeb-real)",
                    "path": os.path.join(root_celeb, celeb_r1),
                    "id": celeb_r1,
                    "detail": "Vídeo original: id1_0007",
                    "technique": "Entrevista de Celebridade (Celeb-real)"
                },
                {
                    "type": "REAL",
                    "title": "Real #2 (YouTube-real)",
                    "path": os.path.join(root_celeb, celeb_r2),
                    "id": celeb_r2,
                    "detail": "Vídeo original: 00208",
                    "technique": "Vídeo in-the-wild (YouTube-real)"
                },
                {
                    "type": "FALSO",
                    "title": "Falso #1 (Face Swap)",
                    "path": os.path.join(root_celeb, celeb_f1),
                    "id": celeb_f1,
                    "detail": "Swap: id1 (fonte) -> id0 (alvo)",
                    "technique": "Celeb-synthesis (DeepFake Face Swap)"
                },
                {
                    "type": "FALSO",
                    "title": "Falso #2 (Face Swap)",
                    "path": os.path.join(root_celeb, celeb_f2),
                    "id": celeb_f2,
                    "detail": "Swap: id3 (fonte) -> id1 (alvo)",
                    "technique": "Celeb-synthesis (DeepFake Face Swap)"
                }
            ]
        },
        {
            "name": "DF-40 (DeepFake-40 Cross-Technique Benchmark)",
            "short_name": "DF-40",
            "accent": "#0F766E",
            "description": "Benchmark abrangente cobrindo 40 métodos geradores agrupados em 4 paradigmas (Face Swap, GANs, Difusão e T2I).",
            "samples": [
                {
                    "type": "REAL",
                    "title": "Real #1 (WhichFaceIsReal)",
                    "path": df40_r1_row["img_name"],
                    "id": os.path.basename(df40_r1_row["img_name"]),
                    "detail": "Subconjunto: Real_whichfaceisreal",
                    "technique": "Fotografia Real (FFHQ Benchmark)"
                },
                {
                    "type": "REAL",
                    "title": "Real #2 (HeyGen Real)",
                    "path": df40_r2_row["img_name"],
                    "id": os.path.basename(df40_r2_row["img_name"]),
                    "detail": "Subconjunto: Real_heygen",
                    "technique": "Avatar Comercial Real (Vídeo Base)"
                },
                {
                    "type": "FALSO",
                    "title": "Falso #1 (FaceSwap)",
                    "path": df40_f1_row["img_name"],
                    "id": os.path.basename(df40_f1_row["img_name"]),
                    "detail": "Paradigma: face_swap | Método: faceswap",
                    "technique": "Troca Facial Clássica (FaceSwap)"
                },
                {
                    "type": "FALSO",
                    "title": "Falso #2 (StyleGAN3)",
                    "path": df40_f2_row["img_name"],
                    "id": os.path.basename(df40_f2_row["img_name"]),
                    "detail": "Paradigma: gan | Método: StyleGAN3",
                    "technique": "Síntese Generativa Adversarial (GAN)"
                }
            ]
        }
    ]

    for ds in datasets:
        for s in ds["samples"]:
            im = Image.open(s["path"])
            s["size_str"] = f"{im.size[0]} × {im.size[1]} px"

    return datasets


def create_overview_page(datasets):
    """Página 1: Grid comparativo 3x4 (todas as 12 imagens)."""
    fig = plt.figure(figsize=(14.0, 9.5), dpi=300)
    fig.patch.set_facecolor("#FAFAFA")

    # Título Principal
    fig.text(0.5, 0.965, "Amostras Forenses dos Datasets de Avaliação", 
             ha="center", va="top", fontsize=18, fontweight="bold", color=COLOR_HEADER_BG)
    fig.text(0.5, 0.938, "Comparação Visual: 2 Imagens Reais e 2 Falsas por Base (MFFI, Celeb-DF v2 e DF-40)", 
             ha="center", va="top", fontsize=11, color=COLOR_TEXT_MUTED)

    # Linha divisória
    line = plt.Line2D([0.04, 0.96], [0.920, 0.920], transform=fig.transFigure, color="#E2E8F0", linewidth=1.5)
    fig.add_artist(line)

    # Cabeçalhos das 4 Colunas
    col_x_starts = [0.175, 0.380, 0.585, 0.790]
    col_w = 0.175
    
    col_headers = [
        ("REAL — Amostra #1", COLOR_REAL, COLOR_REAL_BG),
        ("REAL — Amostra #2", COLOR_REAL, COLOR_REAL_BG),
        ("FALSO — Amostra #1", COLOR_FAKE, COLOR_FAKE_BG),
        ("FALSO — Amostra #2", COLOR_FAKE, COLOR_FAKE_BG)
    ]

    for x_start, (label, text_col, bg_col) in zip(col_x_starts, col_headers):
        badge = patches.FancyBboxPatch(
            (x_start, 0.885), col_w, 0.024,
            boxstyle="round,pad=0.003,rounding_size=0.005",
            facecolor=bg_col, edgecolor=text_col, linewidth=1.2,
            transform=fig.transFigure, clip_on=False, zorder=2
        )
        fig.add_artist(badge)
        fig.text(x_start + col_w / 2.0, 0.897, label,
                 ha="center", va="center", fontsize=9.5, fontweight="bold", color=text_col, zorder=5)

    # 3 Linhas (MFFI, Celeb-DF v2, DF-40)
    row_y_centers = [0.725, 0.460, 0.195]
    row_img_h = 0.195
    row_img_w = 0.170

    for row_idx, (ds, y_c) in enumerate(zip(datasets, row_y_centers)):
        # Card lateral com nome do dataset
        ds_box = patches.FancyBboxPatch(
            (0.04, y_c - 0.095), 0.110, 0.205,
            boxstyle="round,pad=0.005,rounding_size=0.008",
            facecolor="#F1F5F9", edgecolor=ds["accent"], linewidth=1.5,
            transform=fig.transFigure, clip_on=False, zorder=2
        )
        fig.add_artist(ds_box)

        # Faixa de destaque lateral
        accent_bar = patches.Rectangle(
            (0.04, y_c - 0.095), 0.006, 0.205,
            facecolor=ds["accent"], edgecolor="none",
            transform=fig.transFigure, clip_on=False, zorder=3
        )
        fig.add_artist(accent_bar)

        fig.text(0.095, y_c + 0.045, ds["short_name"],
                 ha="center", va="center", fontsize=12, fontweight="bold", color=ds["accent"], zorder=5)
        
        if ds["short_name"] == "MFFI":
            sub_lbl = "Treinamento\nSupervisionado\nPrimário"
        elif ds["short_name"] == "Celeb-DF v2":
            sub_lbl = "Generalização\nCross-Dataset\n(Face Swaps)"
        else:
            sub_lbl = "Benchmark\nMulti-Método\n(40 Técnicas)"

        fig.text(0.095, y_c - 0.025, sub_lbl,
                 ha="center", va="center", fontsize=8.2, color=COLOR_TEXT_MUTED, multialignment="center", zorder=5)

        # 4 Imagens por Linha
        for col_idx, (s, x_start) in enumerate(zip(ds["samples"], col_x_starts)):
            is_real = (s["type"] == "REAL")
            border_col = COLOR_REAL_BORDER if is_real else COLOR_FAKE_BORDER
            
            ax = fig.add_axes([x_start + 0.002, y_c - 0.052, row_img_w, row_img_h])
            ax.set_zorder(10)
            im = Image.open(s["path"]).convert("RGB")
            ax.imshow(im)
            ax.set_xticks([])
            ax.set_yticks([])

            for spine in ax.spines.values():
                spine.set_edgecolor(border_col)
                spine.set_linewidth(1.8)

            # Legendas embaixo da imagem
            short_id = s['id'] if len(s['id']) <= 20 else s['id'][:18] + ".."
            info_text = f"{s['technique']}\n{short_id} | {s['size_str']}"
            fig.text(x_start + col_w / 2.0, y_c - 0.062, info_text,
                     ha="center", va="top", fontsize=7.2, color=COLOR_TEXT_MAIN, multialignment="center", zorder=12)

    # Rodapé da Página 1
    fig.text(0.04, 0.025, "Documento gerado automaticamente para inspeção forense e interpretabilidade — Projeto TCC (DeepFake Detection)", 
             fontsize=8.0, color="#64748B", zorder=10)
    fig.text(0.96, 0.025, "Página 1 de 4", fontsize=8.0, color="#64748B", ha="right", zorder=10)

    return fig


def create_detailed_page(ds, page_num, total_pages=4):
    """Página detalhada por base de dados (2 Reais e 2 Falsos em alta resolução)."""
    fig = plt.figure(figsize=(14.0, 9.5), dpi=300)
    fig.patch.set_facecolor("#FAFAFA")

    # Cabeçalho da Página
    fig.text(0.04, 0.965, f"Detalhamento de Amostras: {ds['name']}", 
             fontsize=16, fontweight="bold", color=COLOR_HEADER_BG, zorder=5)
    fig.text(0.04, 0.938, ds["description"], 
             fontsize=9.5, color=COLOR_TEXT_MUTED, zorder=5)

    line = plt.Line2D([0.04, 0.96], [0.920, 0.920], transform=fig.transFigure, color="#E2E8F0", linewidth=1.5, zorder=3)
    fig.add_artist(line)

    # 4 Colunas de Amostras em Destaque
    col_xs = [0.04, 0.28, 0.52, 0.76]
    card_w = 0.20
    card_h = 0.64
    card_y = 0.23
    img_h = 0.38
    img_w = 0.18

    for col_idx, (s, x_start) in enumerate(zip(ds["samples"], col_xs)):
        is_real = (s["type"] == "REAL")
        theme_col = COLOR_REAL if is_real else COLOR_FAKE
        bg_col = COLOR_REAL_BG if is_real else COLOR_FAKE_BG
        border_col = COLOR_REAL_BORDER if is_real else COLOR_FAKE_BORDER

        # Card de fundo (zorder=1 para ficar atrás do eixo da imagem)
        card = patches.FancyBboxPatch(
            (x_start, card_y), card_w, card_h,
            boxstyle="round,pad=0.008,rounding_size=0.010",
            facecolor="#FFFFFF", edgecolor=COLOR_CARD_BORDER, linewidth=1.2,
            transform=fig.transFigure, clip_on=False, zorder=1
        )
        fig.add_artist(card)

        # Header do Card com Badge
        badge = patches.FancyBboxPatch(
            (x_start + 0.01, card_y + card_h - 0.045), card_w - 0.02, 0.035,
            boxstyle="round,pad=0.003,rounding_size=0.006",
            facecolor=bg_col, edgecolor=theme_col, linewidth=1.2,
            transform=fig.transFigure, clip_on=False, zorder=3
        )
        fig.add_artist(badge)
        fig.text(x_start + card_w / 2.0, card_y + card_h - 0.028, f"[{s['type']}]  {s['title']}",
                 ha="center", va="center", fontsize=9.5, fontweight="bold", color=theme_col, zorder=5)

        # Eixo da Imagem (zorder=10 para ficar em primeiro plano)
        ax = fig.add_axes([x_start + 0.010, card_y + card_h - 0.445, img_w, img_h])
        ax.set_zorder(10)
        im = Image.open(s["path"]).convert("RGB")
        ax.imshow(im)
        ax.set_xticks([])
        ax.set_yticks([])
        for spine in ax.spines.values():
            spine.set_edgecolor(border_col)
            spine.set_linewidth(1.8)

        # Metadados abaixo da Imagem
        meta_y = card_y + 0.155
        fig.text(x_start + 0.012, meta_y, "Técnica / Subconjunto:", fontsize=8.2, fontweight="bold", color=COLOR_HEADER_BG, zorder=5)
        fig.text(x_start + 0.012, meta_y - 0.022, s["technique"], fontsize=7.6, color=COLOR_TEXT_MAIN, zorder=5)

        fig.text(x_start + 0.012, meta_y - 0.050, "Detalhes / Origem:", fontsize=8.2, fontweight="bold", color=COLOR_HEADER_BG, zorder=5)
        fig.text(x_start + 0.012, meta_y - 0.072, s["detail"], fontsize=7.6, color=COLOR_TEXT_MAIN, zorder=5)

        fig.text(x_start + 0.012, meta_y - 0.100, "Dimensões Originais:", fontsize=8.2, fontweight="bold", color=COLOR_HEADER_BG, zorder=5)
        fig.text(x_start + 0.012, meta_y - 0.122, s["size_str"], fontsize=7.6, color=COLOR_TEXT_MAIN, zorder=5)

    # Painel Informativo Inferior da Base
    info_box = patches.FancyBboxPatch(
        (0.04, 0.070), 0.92, 0.125,
        boxstyle="round,pad=0.006,rounding_size=0.008",
        facecolor="#F8FAFC", edgecolor=ds["accent"], linewidth=1.2,
        transform=fig.transFigure, clip_on=False, zorder=2
    )
    fig.add_artist(info_box)

    accent_strip = patches.Rectangle(
        (0.04, 0.070), 0.008, 0.125,
        facecolor=ds["accent"], edgecolor="none",
        transform=fig.transFigure, clip_on=False, zorder=3
    )
    fig.add_artist(accent_strip)

    fig.text(0.065, 0.165, f"Observações Forenses sobre a Base {ds['short_name']}:", 
             fontsize=9.5, fontweight="bold", color=ds["accent"], zorder=5)
    
    if ds["short_name"] == "MFFI":
        obs_text = (
            "• Imagens padronizadas em 512×512 px com recorte centralizado na face e iluminação balanceada.\n"
            "• Fakes focam em alterações texturais e manipulações sutis (olhos, boca, pele), ideais para análise de resíduos espectrais e SRM.\n"
            "• O split de teste oficial contém 181.947 amostras balanceadas entre múltiplas famílias geradoras e faces genuínas."
        )
    elif ds["short_name"] == "Celeb-DF v2":
        obs_text = (
            "• Reconhecido por sua elevada qualidade de síntese e ausência de bordas óbvias de blending nas trocas faciais.\n"
            "• Apresenta dois subgrupos reais: Celeb-real (entrevistas em estúdio de alta definição) e YouTube-real (vídeos naturais com compressão).\n"
            "• Principal desafio de generalização cross-dataset devido à variabilidade dinâmica de iluminação, pose e compressão temporal dos vídeos."
        )
    else: # DF-40
        obs_text = (
            "• Cobre 40 métodos geradores agrupados em 4 paradigmas: Face Swap (troca facial), GANs, Modelos de Difusão e Edição Text-to-Image (T2I).\n"
            "• Resoluções variam com a técnica de origem (desde 256×256 px em GANs clássicas até 1024×1024 px em FFHQ e MidJourney).\n"
            "• Teste crucial de robustez e interpretabilidade para verificar se detectores generalizam para técnicas generativas contemporâneas."
        )

    fig.text(0.065, 0.140, obs_text, fontsize=7.8, color=COLOR_TEXT_MAIN, linespacing=1.45, zorder=5)

    # Rodapé
    fig.text(0.04, 0.025, f"Documento gerado automaticamente — Base {ds['short_name']}", 
             fontsize=8.0, color="#64748B", zorder=5)
    fig.text(0.96, 0.025, f"Página {page_num} de {total_pages}", fontsize=8.0, color="#64748B", ha="right", zorder=5)

    return fig


def main():
    os.makedirs("figures/amostras_bases", exist_ok=True)
    pdf_out = "figures/amostras_bases/amostras_mffi_celeb_df40.pdf"

    print("Obtendo metadados das amostras...")
    datasets = get_samples_metadata()

    print(f"Gerando PDF em: {pdf_out}")
    with PdfPages(pdf_out) as pdf:
        print("Criando Página 1: Visão Geral Comparativa...")
        fig1 = create_overview_page(datasets)
        pdf.savefig(fig1, dpi=300)
        plt.close(fig1)

        for idx, ds in enumerate(datasets, start=2):
            print(f"Criando Página {idx}: {ds['short_name']}...")
            fig_ds = create_detailed_page(ds, page_num=idx, total_pages=4)
            pdf.savefig(fig_ds, dpi=300)
            plt.close(fig_ds)

    print("PDF concluído com sucesso!")
    print(f"Tamanho do arquivo: {os.path.getsize(pdf_out) / 1024:.1f} KB")

if __name__ == "__main__":
    main()
