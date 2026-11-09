"""Script para inspecionar e catalogar os diferentes tipos de desafios de degradação
presentes no conjunto difícil (test_d) do benchmark MFFI.
"""

from pathlib import Path
from PIL import Image
import numpy as np
import pandas as pd

test_clean_dir = Path("/media/ssd2/lucas.ocunha/datasets/phase1/testset")
test_hard_dir = Path("/media/ssd2/lucas.ocunha/datasets/phase1/test_d")

df = pd.read_csv("data/raw/test.csv")
df.columns = df.columns.str.strip()

print(f"Total de imagens no teste: {len(df)}")

records = []
N_SCAN = 4000

for idx in range(min(N_SCAN, len(df))):
    fname = df.iloc[idx]["img_name"]
    lbl = int(df.iloc[idx]["target"])
    p_c = test_clean_dir / fname
    p_h = test_hard_dir / fname
    if not p_c.exists() or not p_h.exists():
        continue

    with Image.open(p_c) as ic, Image.open(p_h) as ih:
        s_c = ic.size
        s_h = ih.size
        c_rgb = np.array(ic.convert("RGB"), dtype=np.float32)
        h_rgb = np.array(ih.convert("RGB"), dtype=np.float32)

    downsampled = (s_h != s_c)
    if downsampled:
        ih_resized = Image.fromarray(h_rgb.astype(np.uint8)).resize(s_c, Image.BILINEAR)
        h_eval = np.array(ih_resized, dtype=np.float32)
    else:
        h_eval = h_rgb

    # Diferença pixel a pixel
    diff = np.abs(c_rgb - h_eval)
    mean_diff = float(diff.mean())
    max_diff = float(diff.max())

    # Luminância
    c_gray = 0.299 * c_rgb[..., 0] + 0.587 * c_rgb[..., 1] + 0.114 * c_rgb[..., 2]
    h_gray = 0.299 * h_eval[..., 0] + 0.587 * h_eval[..., 1] + 0.114 * h_eval[..., 2]

    # Laplaciano 2D rápido via fatiamento numpy
    c_lap = (
        c_gray[1:-1, 2:] + c_gray[1:-1, :-2] + c_gray[2:, 1:-1] + c_gray[:-2, 1:-1]
        - 4 * c_gray[1:-1, 1:-1]
    )
    h_lap = (
        h_gray[1:-1, 2:] + h_gray[1:-1, :-2] + h_gray[2:, 1:-1] + h_gray[:-2, 1:-1]
        - 4 * h_gray[1:-1, 1:-1]
    )
    var_c = float(c_lap.var())
    var_h = float(h_lap.var())
    var_ratio = var_h / (var_c + 1e-5)

    # Brilho e contraste
    b_c = float(c_gray.mean())
    b_h = float(h_gray.mean())
    b_shift = b_h - b_c
    c_std = float(c_gray.std())
    h_std = float(h_gray.std())
    c_ratio = h_std / (c_std + 1e-5)

    # Desvio por canal de cor
    ch_diffs = [float(np.abs(c_rgb[..., i] - h_eval[..., i]).mean()) for i in range(3)]
    color_shift = float(np.std(ch_diffs))

    # Medida de artefato de bloco JPEG 8x8
    diff_h = np.abs(h_gray[1:-1, 2:] - h_gray[1:-1, :-2])
    cols = np.arange(diff_h.shape[1])
    grid_mask = (cols % 8 == 0)
    grid_diff = float(diff_h[:, grid_mask].mean()) if grid_mask.any() else 0.0
    non_grid_diff = float(diff_h[:, ~grid_mask].mean()) if (~grid_mask).any() else 1e-5
    blockiness = grid_diff / (non_grid_diff + 1e-5)

    records.append({
        "fname": fname,
        "label": lbl,
        "downsampled": downsampled,
        "s_h": s_h,
        "mean_diff": mean_diff,
        "max_diff": max_diff,
        "var_ratio": var_ratio,
        "b_shift": b_shift,
        "c_ratio": c_ratio,
        "color_shift": color_shift,
        "blockiness": blockiness,
        "var_c": var_c,
        "var_h": var_h,
    })

res = pd.DataFrame(records)
print(f"Analisadas {len(res)} imagens com sucesso.")
print(f"Downsampled (224x224): {res['downsampled'].sum()} de {len(res)}")
print("\nEstatísticas:")
print(res[["mean_diff", "var_ratio", "b_shift", "c_ratio", "color_shift", "blockiness"]].describe().round(3))

res.to_csv("scratch_degradations.csv", index=False)
