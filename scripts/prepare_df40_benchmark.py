"""Gera o manifesto de teste equilibrado (Real vs Fake) para o benchmark DF40."""

from __future__ import annotations

import random
from pathlib import Path
import pandas as pd

def collect_images(folder: Path, sample_size: int, seed: int = 42) -> list[Path]:
    imgs = []
    for ext in ("*.png", "*.jpg", "*.jpeg"):
        imgs.extend(folder.glob(f"**/{ext}"))
    imgs = [p for p in imgs if not p.name.startswith("._") and "__MACOSX" not in str(p)]
    random.seed(seed)
    if len(imgs) > sample_size:
        return random.sample(imgs, sample_size)
    return imgs

def main():
    root = Path("/media/ssd2/lucas.ocunha/datasets/df40_extracted")
    out_dir = Path("data/df40")
    out_dir.mkdir(parents=True, exist_ok=True)
    out_csv = out_dir / "test.csv"

    records = []

    # 1. REAL SAMPLES (target = 0)
    real_configs = [
        ("whichfaceisreal/real", "whichfaceisreal", 800),
        ("CollabDiff/real", "CollabDiff", 800),
        ("MidJourney/real", "MidJourney", 800),
        ("starganv2/real", "starganv2", 800),
        ("styleclip/real", "styleclip", 800),
        ("heygen_new/real", "heygen", 1000),
    ]

    for rel_path, name, n in real_configs:
        folder = root / rel_path
        if folder.exists():
            imgs = collect_images(folder, n)
            for img in imgs:
                records.append({
                    "img_path": str(img),
                    "target": 0,
                    "method": f"Real_{name}",
                    "paradigm": "real"
                })
            print(f"Coletadas {len(imgs)} imagens REAIS de {name}")

    # 2. FAKE SAMPLES (target = 1)
    # Categorias do DF40
    fake_configs = [
        # Difusão
        ("DiT", "DiT", "diffusion", 250),
        ("SiT", "SiT", "diffusion", 250),
        ("ddim", "ddim", "diffusion", 250),
        ("sd2.1", "sd2.1", "diffusion", 250),
        ("pixart", "pixart", "diffusion", 250),
        ("RDDM", "RDDM", "diffusion", 250),
        ("CollabDiff/fake", "CollabDiff", "diffusion", 250),
        # GANs
        ("StyleGAN2", "StyleGAN2", "gan", 250),
        ("StyleGAN3", "StyleGAN3", "gan", 250),
        ("StyleGANXL", "StyleGANXL", "gan", 250),
        ("VQGAN", "VQGAN", "gan", 250),
        ("stargan/fake", "stargan", "gan", 200),
        ("starganv2/fake", "starganv2", "gan", 200),
        # Face Swapping
        ("deepfacelab/fake", "deepfacelab", "face_swap", 250),
        ("faceswap", "faceswap", "face_swap", 250),
        ("mobileswap", "mobileswap", "face_swap", 250),
        ("blendface", "blendface", "face_swap", 250),
        ("uniface", "uniface", "face_swap", 250),
        # Edição / T2I
        ("MidJourney/fake", "MidJourney", "editing_t2i", 250),
        ("whichfaceisreal/fake", "whichfaceisreal", "editing_t2i", 250),
        ("styleclip/fake", "styleclip", "editing_t2i", 250),
        ("e4e", "e4e", "editing_t2i", 250),
        # Avatares Comerciais & Vídeo
        ("heygen_new/fake", "heygen", "commercial_avatar", 250),
    ]

    for rel_path, name, paradigm, n in fake_configs:
        folder = root / rel_path
        if folder.exists():
            imgs = collect_images(folder, n)
            for img in imgs:
                records.append({
                    "img_path": str(img),
                    "target": 1,
                    "method": name,
                    "paradigm": paradigm
                })
            print(f"Coletadas {len(imgs)} imagens FAKE de {name} ({paradigm})")

    # Amostra de cdf/frames (reenactment e talking face: Wav2Lip, SadTalker, etc.)
    cdf_folder = root / "cdf" / "frames"
    if cdf_folder.exists():
        imgs = collect_images(cdf_folder, 500)
        for img in imgs:
            records.append({
                "img_path": str(img),
                "target": 1,
                "method": "cdf_reenactment",
                "paradigm": "talking_reenactment"
            })
        print(f"Coletadas {len(imgs)} imagens FAKE de cdf_reenactment (talking_reenactment)")

    df = pd.DataFrame(records)
    df = df.sample(frac=1.0, random_state=42).reset_index(drop=True)
    df.to_csv(out_csv, index=False)

    print("\n--- RESUMO DO DATASET DF40 ---")
    print(f"Total de imagens: {len(df)}")
    print(df["target"].value_counts().rename({0: "Real", 1: "Fake"}))
    print("\nDistribuição por Paradigma:")
    print(df["paradigm"].value_counts())
    print(f"\nManifesto salvo em: {out_csv}")

if __name__ == "__main__":
    main()
