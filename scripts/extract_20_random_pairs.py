"""Seleciona 20 pares aleatórios (10 Reais, 10 Fakes) entre o split limpo (testset)
e o split degradado/difícil (test_d) em resolução nativa total e gera visualização.
"""

from pathlib import Path
from PIL import Image
import numpy as np
import pandas as pd
import shutil
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

test_clean_dir = Path("/media/ssd2/lucas.ocunha/datasets/phase1/testset")
test_hard_dir = Path("/media/ssd2/lucas.ocunha/datasets/phase1/test_d")

out_dir = Path("figures/samples_clean_vs_hard/random_20_pairs")
out_dir.mkdir(parents=True, exist_ok=True)

df = pd.read_csv("data/raw/test.csv")
df.columns = df.columns.str.strip()

# Arquivos já utilizados anteriormente para garantir amostras 100% inéditas
already_used = {
    "0d58d1f5e27411624cb613f8c85775f0.jpg",
    "3e0f41a3d9221fc357223b9d03d09a24.jpg",
    "86c162d665b1c92cb6c1889e18e388d2.jpg",
    "7e8ead963286595d2c4e12e1a3bc89a3.jpg",
    "747c3e721133667b2ff157f24ebfacc0.jpg",
    "1507f38819c5a6207f637094de4856f3.jpg",
    "42783ee08a200df8dff484808e072610.jpg",
    "edb2154bde51a28c8405c0e073f0b8f8.jpg",
    "53a8428742889ff038434aaf0a8b581c.jpg",
    "1bd15fe6958fff3519836c6c8adb27f5.jpg",
    "b8480e3e3ff9af2ccc5cb3b10b863987.jpg",
    "c0a323fc8f4af9721d860380ec02447f.jpg",
}

df_cand = df[~df["img_name"].isin(already_used)].copy()

# Amostragem estratificada aleatória: 10 Reais (0) e 10 Fakes (1)
rng = 42
sample_real = df_cand[df_cand["target"] == 0].sample(n=10, random_state=rng)
sample_fake = df_cand[df_cand["target"] == 1].sample(n=10, random_state=rng)

# Intercala Real e Fake para distribuição homogênea
selected_rows = []
for r, f in zip(sample_real.to_dict("records"), sample_fake.to_dict("records")):
    selected_rows.append(r)
    selected_rows.append(f)

print(f"Total selecionado: {len(selected_rows)} pares aleatórios.")

records_info = []
clean_imgs_disp = []
hard_imgs_disp = []

for idx, row in enumerate(selected_rows, start=1):
    fn = row["img_name"]
    lbl = row["target"]
    lbl_str = "Real" if lbl == 0 else "Fake"

    p_c = test_clean_dir / fn
    p_h = test_hard_dir / fn

    dst_c = out_dir / f"pair_{idx:02d}_{lbl_str}_clean_{fn}"
    dst_h = out_dir / f"pair_{idx:02d}_{lbl_str}_hard_{fn}"

    shutil.copy2(p_c, dst_c)
    shutil.copy2(p_h, dst_h)

    with Image.open(p_c) as ic, Image.open(p_h) as ih:
        s_c = ic.size
        s_h = ih.size
        c_rgb = np.array(ic.convert("RGB"), dtype=np.float32)
        h_rgb = np.array(ih.convert("RGB"), dtype=np.float32)

    if s_h != (512, 512):
        ih_disp = ih.resize((512, 512), Image.NEAREST)
        h_eval = np.array(ih.resize(s_c, Image.BILINEAR), dtype=np.float32)
    else:
        ih_disp = ih
        h_eval = h_rgb

    clean_imgs_disp.append(ic)
    hard_imgs_disp.append(ih_disp)

    diff = np.abs(c_rgb - h_eval)
    mean_diff = float(diff.mean())
    c_gray = 0.299 * c_rgb[..., 0] + 0.587 * c_rgb[..., 1] + 0.114 * c_rgb[..., 2]
    h_gray = 0.299 * h_eval[..., 0] + 0.587 * h_eval[..., 1] + 0.114 * h_eval[..., 2]
    b_shift = float(h_gray.mean() - c_gray.mean())

    records_info.append({
        "pair_idx": idx,
        "class": lbl_str,
        "filename": fn,
        "clean_size": s_c,
        "hard_size": s_h,
        "mean_pixel_diff": round(mean_diff, 2),
        "b_shift": round(b_shift, 2),
        "dst_clean": dst_c.name,
        "dst_hard": dst_h.name,
    })
    print(f"Par {idx:02d} [{lbl_str}]: {fn} | Clean: {s_c} -> Hard: {s_h} | Diff: {mean_diff:.2f}")

# Salva manifesto CSV com os metadados dos 20 pares
pd.DataFrame(records_info).to_csv(out_dir / "manifest_20_random_pairs.csv", index=False)

# Gera Grid Visual 1: Primeiros 10 Pares (Pares 1 a 10)
fig, axes = plt.subplots(2, 10, figsize=(26, 6.2), facecolor="white")
for j in range(10):
    axes[0, j].imshow(clean_imgs_disp[j])
    axes[0, j].set_xticks([])
    axes[0, j].set_yticks([])
    axes[0, j].set_title(f"Par {j+1:02d} ({records_info[j]['class']})", fontsize=10, fontweight="bold")
    for s in axes[0, j].spines.values():
        s.set_color("#2e7d32")
        s.set_linewidth(2)

    axes[1, j].imshow(hard_imgs_disp[j])
    axes[1, j].set_xticks([])
    axes[1, j].set_yticks([])
    for s in axes[1, j].spines.values():
        s.set_color("#c62828")
        s.set_linewidth(2)

axes[0, 0].set_ylabel("Limpo\n(testset)", fontsize=11, fontweight="bold")
axes[1, 0].set_ylabel("Difícil / OOD\n(test_d)", fontsize=11, fontweight="bold")
plt.suptitle("Benchmark MFFI: 10 Amostras Aleatórias Pareadas (Parte 1: Pares 01 a 10)", fontsize=14, fontweight="bold", y=0.98)
plt.tight_layout()
p1 = out_dir / "comparison_random_pairs_part1.jpg"
plt.savefig(p1, dpi=250, bbox_inches="tight")
plt.close()

# Gera Grid Visual 2: Segundos 10 Pares (Pares 11 a 20)
fig, axes = plt.subplots(2, 10, figsize=(26, 6.2), facecolor="white")
for j in range(10):
    idx = j + 10
    axes[0, j].imshow(clean_imgs_disp[idx])
    axes[0, j].set_xticks([])
    axes[0, j].set_yticks([])
    axes[0, j].set_title(f"Par {idx+1:02d} ({records_info[idx]['class']})", fontsize=10, fontweight="bold")
    for s in axes[0, j].spines.values():
        s.set_color("#2e7d32")
        s.set_linewidth(2)

    axes[1, j].imshow(hard_imgs_disp[idx])
    axes[1, j].set_xticks([])
    axes[1, j].set_yticks([])
    for s in axes[1, j].spines.values():
        s.set_color("#c62828")
        s.set_linewidth(2)

axes[0, 0].set_ylabel("Limpo\n(testset)", fontsize=11, fontweight="bold")
axes[1, 0].set_ylabel("Difícil / OOD\n(test_d)", fontsize=11, fontweight="bold")
plt.suptitle("Benchmark MFFI: 10 Amostras Aleatórias Pareadas (Parte 2: Pares 11 a 20)", fontsize=14, fontweight="bold", y=0.98)
plt.tight_layout()
p2 = out_dir / "comparison_random_pairs_part2.jpg"
plt.savefig(p2, dpi=250, bbox_inches="tight")
plt.close()

print(f"\nFinalizado com sucesso! Imagens e grids salvos em: {out_dir}")
