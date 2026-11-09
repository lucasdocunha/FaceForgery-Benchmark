"""Extrai amostras pareadas em resolução total para cada um dos 8 principais desafios
do conjunto difícil (test_d / Test-Hard) e gera a figura consolidada.
"""

from pathlib import Path
from PIL import Image
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import shutil

test_clean_dir = Path("/media/ssd2/lucas.ocunha/datasets/phase1/testset")
test_hard_dir = Path("/media/ssd2/lucas.ocunha/datasets/phase1/test_d")
out_dir = Path("figures/samples_clean_vs_hard")
out_dir.mkdir(parents=True, exist_ok=True)

# 8 Desafios Canônicos identificados no MFFI Test-D
challenges = [
    {
        "id": "challenge_1_blur",
        "title": "1. Desfoque / Blur",
        "type": "Gaussian Blur",
        "class": "Fake",
        "fname": "747c3e721133667b2ff157f24ebfacc0.jpg",
        "desc": "Queda drástica de nitidez e perda de texturas finas.",
    },
    {
        "id": "challenge_2_noise",
        "title": "2. Ruído / Granulação",
        "type": "Heavy Noise",
        "class": "Real",
        "fname": "1507f38819c5a6207f637094de4856f3.jpg",
        "desc": "Ruído estocástico de alta frequência sobre a face.",
    },
    {
        "id": "challenge_3_jpeg",
        "title": "3. Compressão JPEG",
        "type": "JPEG Blocking",
        "class": "Fake",
        "fname": "42783ee08a200df8dff484808e072610.jpg",
        "desc": "Artefatos de macroblocos 8x8 e ringing.",
    },
    {
        "id": "challenge_4_downsampling",
        "title": "4. Sub-amostragem",
        "type": "Downsampling",
        "class": "Fake",
        "fname": "edb2154bde51a28c8405c0e073f0b8f8.jpg",
        "desc": "Redução para 224x224 com perda de resolução.",
    },
    {
        "id": "challenge_5_overexposure",
        "title": "5. Superexposição",
        "type": "Overexposure",
        "class": "Real",
        "fname": "53a8428742889ff038434aaf0a8b581c.jpg",
        "desc": "Brilho extremo com estouro de brancos.",
    },
    {
        "id": "challenge_6_underexposure",
        "title": "6. Subexposição",
        "type": "Underexposure",
        "class": "Fake",
        "fname": "1bd15fe6958fff3519836c6c8adb27f5.jpg",
        "desc": "Escurecimento severo e perda de detalhes em sombras.",
    },
    {
        "id": "challenge_7_colorshift",
        "title": "7. Distorção Cromática",
        "type": "Color Shift",
        "class": "Fake",
        "fname": "b8480e3e3ff9af2ccc5cb3b10b863987.jpg",
        "desc": "Forte desvio de matiz e desbalanceamento RGB.",
    },
    {
        "id": "challenge_8_lowcontrast",
        "title": "8. Baixo Contraste",
        "type": "Low Contrast",
        "class": "Real",
        "fname": "c0a323fc8f4af9721d860380ec02447f.jpg",
        "desc": "Compressão de faixa dinâmica com aspecto lavado.",
    },
]

clean_imgs = []
hard_imgs = []

print("Copiando imagens originais para o diretório git...")
for c in challenges:
    fn = c["fname"]
    src_clean = test_clean_dir / fn
    src_hard = test_hard_dir / fn

    dst_clean = out_dir / f"{c['id']}_clean_{c['class']}_{fn}"
    dst_hard = out_dir / f"{c['id']}_hard_{c['class']}_{fn}"

    shutil.copy2(src_clean, dst_clean)
    shutil.copy2(src_hard, dst_hard)

    img_c = Image.open(src_clean).convert("RGB")
    img_h = Image.open(src_hard).convert("RGB")

    clean_imgs.append(img_c)
    # Se downsampled, reamostra para visualização uniforme no grid
    if img_h.size != (512, 512):
        img_h_disp = img_h.resize((512, 512), Image.NEAREST)
    else:
        img_h_disp = img_h
    hard_imgs.append(img_h_disp)

    print(f" -> {c['title']} ({c['class']}): {fn} | Clean: {img_c.size} -> Hard: {img_h.size}")

# Montagem do Grid Comparativo (2 Linhas x 8 Colunas)
N = len(challenges)
fig, axes = plt.subplots(2, N, figsize=(3.2 * N, 7.2), facecolor="white")

for j, c in enumerate(challenges):
    # Linha 0: Clean
    axes[0, j].imshow(clean_imgs[j])
    axes[0, j].set_xticks([])
    axes[0, j].set_yticks([])
    axes[0, j].set_title(f"{c['title']}\n({c['class']})", fontsize=11, fontweight="bold", pad=8)
    for spine in axes[0, j].spines.values():
        spine.set_color("#2e7d32")  # verde
        spine.set_linewidth(2.5)

    # Linha 1: Hard
    axes[1, j].imshow(hard_imgs[j])
    axes[1, j].set_xticks([])
    axes[1, j].set_yticks([])
    for spine in axes[1, j].spines.values():
        spine.set_color("#c62828")  # vermelho
        spine.set_linewidth(2.5)

axes[0, 0].set_ylabel("Limpo\n(testset)", fontsize=13, fontweight="bold", labelpad=10)
axes[1, 0].set_ylabel("Difícil / OOD\n(test_d)", fontsize=13, fontweight="bold", labelpad=10)

plt.suptitle("Benchmark MFFI: Catálogo dos 8 Desafios de Degradação (Clean vs. Test-D)", fontsize=16, fontweight="bold", y=0.98)
plt.tight_layout()
grid_path = out_dir / "comparison_all_8_challenges.png"
plt.savefig(grid_path, dpi=300, bbox_inches="tight", facecolor="white")
plt.close()
print(f"\nGrid comparativo salvo em: {grid_path}")
