#!/usr/bin/env python3
"""
Avaliação Exaustiva de TODAS as Imagens Fakes do DF-40 (650k+ Amostras)
Calcula a taxa de acerto real por técnica e agregada por grupo/paradigma.
"""

from __future__ import annotations
import os
import sys
import time
from pathlib import Path

ROOT_DIR = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT_DIR))

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from PIL import Image, ImageFile
ImageFile.LOAD_TRUNCATED_IMAGES = True
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms

from src.forensics.srm import extract_srm_residuals
from src.models.registry import get_model_spec
from src.pipelines.config import load_config


class FastImageDataset(Dataset):
    def __init__(self, paths: list[Path], img_size: int = 224):
        self.paths = paths
        self.transform = transforms.Compose([
            transforms.Resize((img_size, img_size)),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
        ])

    def __len__(self):
        return len(self.paths)

    def __getitem__(self, idx):
        p = self.paths[idx]
        try:
            with Image.open(p) as img:
                return self.transform(img.convert("RGB"))
        except Exception:
            return torch.zeros(3, 224, 224)


def collect_images_for_folder(folder: Path) -> list[Path]:
    imgs = []
    for ext in ("*.png", "*.jpg", "*.jpeg"):
        imgs.extend(folder.glob(f"**/{ext}"))
    return [p for p in imgs if not p.name.startswith("._") and "__MACOSX" not in str(p)]


def main():
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    print(f"==============================================================================")
    print(f"🚀 INFERÊNCIA EM TODAS AS IMAGENS FAKES DO DF-40 | DISPOSITIVO: {device}")
    print(f"==============================================================================")

    # 1. Carregar Modelo Campeão: DINO (ConvNeXt-B) SRM
    spec = get_model_spec("dino")
    cfg = load_config(ROOT_DIR / "configs/dino.yaml", {"fourier_mode": "srm", "regime": "finetune"})
    model = spec.build(cfg).to(device).eval()
    ckpt_path = Path("/media/ssd2/lucas.ocunha/models-tcc/dino/srm/finetune_robust/seed_42/weights/best.pth")
    state = torch.load(ckpt_path, map_location=device)
    model.load_state_dict(state)
    threshold = 0.1865392  # Threshold calibrado na validação Youden (85k TP, 96.6% Acc)

    print(f"Modelo carregado: DINO ConvNeXt-B SRM (seed 42) | Threshold: {threshold:.4f}\n")

    root = Path("/media/ssd2/lucas.ocunha/datasets/df40_extracted")
    
    # 2. Definição canônica das 24 técnicas organizadas por paradigma
    techniques = [
        # Difusão
        ("DiT", "DiT", "Difusão"),
        ("SiT", "SiT", "Difusão"),
        ("ddim", "ddim", "Difusão"),
        ("sd2.1", "sd2.1", "Difusão"),
        ("pixart", "pixart", "Difusão"),
        ("RDDM", "RDDM", "Difusão"),
        ("CollabDiff/fake", "CollabDiff", "Difusão"),
        # Síntese GAN
        ("StyleGAN2", "StyleGAN2", "Síntese GAN"),
        ("StyleGAN3", "StyleGAN3", "Síntese GAN"),
        ("StyleGANXL", "StyleGANXL", "Síntese GAN"),
        ("VQGAN", "VQGAN", "Síntese GAN"),
        ("stargan/fake", "stargan", "Síntese GAN"),
        ("starganv2/fake", "starganv2", "Síntese GAN"),
        # Troca Facial
        ("deepfacelab/fake", "deepfacelab", "Troca Facial"),
        ("faceswap", "faceswap", "Troca Facial"),
        ("mobileswap", "mobileswap", "Troca Facial"),
        ("blendface", "blendface", "Troca Facial"),
        ("uniface", "uniface", "Troca Facial"),
        # Edição / T2I
        ("MidJourney/fake", "MidJourney", "Edição / T2I"),
        ("whichfaceisreal/fake", "whichfaceisreal", "Edição / T2I"),
        ("styleclip/fake", "styleclip", "Edição / T2I"),
        ("e4e", "e4e", "Edição / T2I"),
        # Reencenação & Avatares
        ("heygen_new/fake", "heygen", "Avatares & Reencenação"),
        ("cdf/frames", "cdf_reenactment", "Avatares & Reencenação"),
    ]

    results = []
    total_imgs_all = 0
    total_detected_all = 0
    start_total_time = time.time()

    print(f"{'Paradigma':22s} | {'Técnica':18s} | {'Total Imgs':10s} | {'Detectados':10s} | {'% Acerto':8s} | {'Tempo':6s}")
    print("-" * 88)

    for rel_path, name, paradigm in techniques:
        folder = root / rel_path
        if not folder.exists():
            print(f"⚠️  Pasta não encontrada: {folder}. Pulando...")
            continue

        imgs = collect_images_for_folder(folder)
        n_imgs = len(imgs)
        if n_imgs == 0:
            continue

        ds = FastImageDataset(imgs)
        loader = DataLoader(ds, batch_size=128, shuffle=False, num_workers=6, pin_memory=True)

        t0 = time.time()
        n_detected_thresh = 0
        n_detected_05 = 0
        sum_probs = 0.0

        with torch.inference_mode():
            for x in loader:
                x = x.to(device, non_blocking=True)
                res = extract_srm_residuals(x)
                inp = torch.cat([x, res], dim=1)
                with torch.amp.autocast("cuda"):
                    logits = model(inp)
                probs = torch.softmax(logits.float(), dim=-1)[:, 1]
                n_detected_thresh += (probs >= threshold).sum().item()
                n_detected_05 += (probs >= 0.50).sum().item()
                sum_probs += probs.sum().item()

        dt = time.time() - t0
        acc_thresh = (n_detected_thresh / n_imgs) * 100
        acc_05 = (n_detected_05 / n_imgs) * 100
        mean_prob = sum_probs / n_imgs

        total_imgs_all += n_imgs
        total_detected_all += n_detected_thresh

        print(f"{paradigm:22s} | {name:18s} | {n_imgs:10d} | {n_detected_thresh:10d} | {acc_thresh:7.2f}% | {dt:5.1f}s", flush=True)

        results.append({
            "paradigma": paradigm,
            "tecnica": name,
            "n_amostras_total": n_imgs,
            "acertos_threshold_val": n_detected_thresh,
            "taxa_acerto_pct": round(acc_thresh, 2),
            "taxa_acerto_05_pct": round(acc_05, 2),
            "prob_fake_media": round(mean_prob, 4),
            "tempo_s": round(dt, 1),
        })

    total_time_min = (time.time() - start_total_time) / 60.0

    df_res = pd.DataFrame(results)
    out_dir = ROOT_DIR / "tables"
    out_dir.mkdir(parents=True, exist_ok=True)
    out_csv = out_dir / "df40_all_fakes_accuracy_breakdown.csv"
    df_res.to_csv(out_csv, index=False)

    print("\n" + "=" * 88)
    print(f"📊 RESUMO CONSOLIDADO POR GRUPO / PARADIGMA (TOTAL GERAL: {total_imgs_all:,} IMAGENS FAKES)")
    print(f"Tempo total de inferência: {total_time_min:.1f} minutos ({total_imgs_all / (total_time_min * 60):.1f} imgs/s)")
    print("=" * 88)

    group_summary = df_res.groupby("paradigma").agg(
        total_imagens=("n_amostras_total", "sum"),
        total_acertos=("acertos_threshold_val", "sum"),
        media_simples_tecnicas=("taxa_acerto_pct", "mean"),
    )
    group_summary["taxa_acerto_ponderada_pct"] = (group_summary["total_acertos"] / group_summary["total_imagens"]) * 100
    group_summary = group_summary.round(2).reset_index()

    print(f"{'Grupo / Paradigma':25s} | {'Total Imagens':13s} | {'Total Acertos':13s} | {'% Acerto Real':14s} | {'Média Simples':13s}")
    print("-" * 88)
    for _, r in group_summary.iterrows():
        print(f"{r['paradigma']:25s} | {int(r['total_imagens']):13,d} | {int(r['total_acertos']):13,d} | {r['taxa_acerto_ponderada_pct']:13.2f}% | {r['media_simples_tecnicas']:12.2f}%")

    taxa_global_ponderada = (total_detected_all / total_imgs_all) * 100
    print("-" * 88)
    print(f"{'TOTAL GERAL DF-40':25s} | {total_imgs_all:13,d} | {total_detected_all:13,d} | {taxa_global_ponderada:13.2f}% | {df_res['taxa_acerto_pct'].mean():12.2f}%")
    print("=" * 88)
    print(f"Arquivo CSV detalhado salvo em: {out_csv}\n")


if __name__ == "__main__":
    main()
