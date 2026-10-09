#!/usr/bin/env python3
"""
Inferência Exaustiva de Modelos no DF-40 (650.549 Imagens Fakes).
Suporta qualquer família de modelo e qualquer representação espectral/forense:
- none (RGB Puro)
- srm (RGB + SRM 6 canais)
- srm_only (SRM 3 canais)
- concat (RGB + FFT Mag 4 canais)
- concat_frequency (RGB + FFT Multi-Banda 7 canais)
"""

from __future__ import annotations
import argparse
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


class GenericDF40Dataset(Dataset):
    def __init__(self, paths: list[Path], img_size: int, fourier_mode: str):
        self.paths = paths
        self.img_size = img_size
        self.fourier_mode = fourier_mode
        self.base_transform = transforms.Compose([
            transforms.Resize((img_size, img_size)),
            transforms.ToTensor(),
        ])
        self.normalize = transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
        self.freq_norm = transforms.Normalize([0.5], [0.5])

    def __len__(self):
        return len(self.paths)

    def __getitem__(self, idx: int):
        p = self.paths[idx]
        try:
            with Image.open(p) as img:
                img_rgb = img.convert("RGB")
        except Exception:
            img_rgb = Image.new("RGB", (self.img_size, self.img_size))

        img_tensor = self.base_transform(img_rgb)
        rgb_norm = self.normalize(img_tensor.clone())

        if self.fourier_mode in ("none", "srm", "srm_only"):
            # Para SRM e none, o residual pode ser extraído em lote ou aqui
            return rgb_norm

        elif self.fourier_mode == "concat":
            gray = 0.2989 * img_tensor[0] + 0.5870 * img_tensor[1] + 0.1140 * img_tensor[2]
            fshift = np.fft.fftshift(np.fft.fft2(gray.numpy()))
            mag = np.log1p(np.abs(fshift))
            span = mag.max() - mag.min()
            norm_mag = np.zeros_like(mag) if span < 1e-8 else (mag - mag.min()) / span
            t_mag = self.freq_norm(torch.from_numpy(norm_mag).float().unsqueeze(0))
            return torch.cat([rgb_norm, t_mag], dim=0)

        elif self.fourier_mode == "concat_frequency":
            gray = 0.2989 * img_tensor[0] + 0.5870 * img_tensor[1] + 0.1140 * img_tensor[2]
            fshift = np.fft.fftshift(np.fft.fft2(gray.numpy()))
            abs_f = np.abs(fshift)

            def safe_norm(arr):
                sp = arr.max() - arr.min()
                return np.zeros_like(arr) if sp < 1e-8 else (arr - arr.min()) / sp

            h, w = fshift.shape
            y, x = np.ogrid[:h, :w]
            r2 = (min(h, w) * 0.12) ** 2
            d2 = (y - h // 2) ** 2 + (x - w // 2) ** 2

            mag = safe_norm(np.log1p(abs_f))
            phase = safe_norm((np.angle(fshift) + np.pi) / (2 * np.pi))
            high = safe_norm(np.log1p(abs_f * (d2 >= r2)))
            low = safe_norm(np.log1p(abs_f * (d2 < r2)))

            t_mag = self.freq_norm(torch.from_numpy(mag).float().unsqueeze(0))
            t_phase = self.freq_norm(torch.from_numpy(phase).float().unsqueeze(0))
            t_high = self.freq_norm(torch.from_numpy(high).float().unsqueeze(0))
            t_low = self.freq_norm(torch.from_numpy(low).float().unsqueeze(0))

            return torch.cat([rgb_norm, t_mag, t_phase, t_high, t_low], dim=0)

        return rgb_norm


def collect_images_for_folder(folder: Path) -> list[Path]:
    imgs = []
    for ext in ("*.png", "*.jpg", "*.jpeg"):
        imgs.extend(folder.glob(f"**/{ext}"))
    return [p for p in imgs if not p.name.startswith("._") and "__MACOSX" not in str(p)]


def main():
    parser = argparse.ArgumentParser(description="Inferência Exaustiva DF-40")
    parser.add_argument("--model-name", type=str, required=True, help="Nome identificador do modelo")
    parser.add_argument("--family", type=str, required=True, help="Família (dino, clip, resnet, etc)")
    parser.add_argument("--mode", type=str, default="none", help="Modo (none, srm, srm_only, concat, concat_frequency)")
    parser.add_argument("--checkpoint", type=str, required=True, help="Caminho do best.pth")
    parser.add_argument("--batch-size", type=int, default=128, help="Batch size")
    parser.add_argument("--num-workers", type=int, default=6, help="Workers DataLoader")
    parser.add_argument("--threshold", type=float, default=None, help="Threshold fixo (ou auto de metrics_val.csv)")
    parser.add_argument("--output-dir", type=str, default="tables/df40_all_fakes_models", help="Diretório de saída")
    args = parser.parse_args()

    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    print(f"==============================================================================")
    print(f"🚀 INFERÊNCIA EXAUSTIVA DF-40: {args.model_name.upper()} | MODO: {args.mode.upper()}")
    print(f"Dispositivo: {device} | Checkpoint: {args.checkpoint}")
    print(f"==============================================================================")

    ckpt_path = Path(args.checkpoint)
    if not ckpt_path.exists():
        print(f"❌ Checkpoint não encontrado: {ckpt_path}")
        sys.exit(1)

    # Descobrir threshold se não informado
    threshold = args.threshold
    if threshold is None:
        val_csv = ckpt_path.parent.parent / "results" / "metrics_val.csv"
        if val_csv.exists():
            try:
                df_val = pd.read_csv(val_csv)
                if "threshold" in df_val.columns:
                    threshold = float(df_val["threshold"].iloc[0])
                elif "best_threshold" in df_val.columns:
                    threshold = float(df_val["best_threshold"].iloc[0])
            except Exception:
                pass
    if threshold is None:
        threshold = 0.50
    print(f"Threshold adotado: {threshold:.4f}\n")

    # Configuração e Construção do Modelo
    config_file = ROOT_DIR / "configs" / f"{args.family}.yaml"
    cfg = load_config(config_file, {"fourier_mode": args.mode, "regime": "finetune"})
    img_size = cfg.image_size

    spec = get_model_spec(args.family)
    model = spec.build(cfg).to(device).eval()

    state = torch.load(ckpt_path, map_location=device)
    if "model_state_dict" in state:
        state = state["model_state_dict"]
    model.load_state_dict(state)

    root = Path("/media/ssd2/lucas.ocunha/datasets/df40_extracted")
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
    t_start = time.time()

    print(f"{'Paradigma':22s} | {'Técnica':18s} | {'Total Imgs':10s} | {'Detectados':10s} | {'% Acerto':8s} | {'Tempo':6s}")
    print("-" * 88)

    for rel_path, name, paradigm in techniques:
        folder = root / rel_path
        if not folder.exists():
            continue

        imgs = collect_images_for_folder(folder)
        n_imgs = len(imgs)
        if n_imgs == 0:
            continue

        ds = GenericDF40Dataset(imgs, img_size=img_size, fourier_mode=args.mode)
        loader = DataLoader(ds, batch_size=args.batch_size, shuffle=False, num_workers=args.num_workers, pin_memory=True)

        t0 = time.time()
        n_detected_thresh = 0
        n_detected_05 = 0
        sum_probs = 0.0

        with torch.inference_mode():
            for x in loader:
                x = x.to(device, non_blocking=True)
                if args.mode == "srm":
                    res = extract_srm_residuals(x)
                    inp = torch.cat([x, res], dim=1)
                elif args.mode == "srm_only":
                    inp = extract_srm_residuals(x)
                else:
                    inp = x

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
            "model_name": args.model_name,
            "family": args.family,
            "mode": args.mode,
            "paradigma": paradigm,
            "tecnica": name,
            "n_amostras_total": n_imgs,
            "acertos_threshold_val": n_detected_thresh,
            "taxa_acerto_pct": round(acc_thresh, 2),
            "taxa_acerto_05_pct": round(acc_05, 2),
            "prob_fake_media": round(mean_prob, 4),
            "tempo_s": round(dt, 1),
        })

    total_time_min = (time.time() - t_start) / 60.0
    df_res = pd.DataFrame(results)

    out_dir = ROOT_DIR / args.output_dir
    out_dir.mkdir(parents=True, exist_ok=True)
    out_csv = out_dir / f"df40_all_fakes_{args.model_name}.csv"
    df_res.to_csv(out_csv, index=False)

    print("\n" + "=" * 88)
    print(f"📊 RESUMO DO MODELO: {args.model_name.upper()} (TOTAL GERAL: {total_imgs_all:,} IMAGENS FAKES)")
    print(f"Tempo total de inferência: {total_time_min:.1f} minutos ({total_imgs_all / (total_time_min * 60):.1f} imgs/s)")
    print(f"Total Acertos: {total_detected_all:,} | Taxa Geral Ponderada: {(total_detected_all / total_imgs_all)*100:.2f}%")
    print(f"CSV salvo em: {out_csv}")
    print("=" * 88 + "\n")


if __name__ == "__main__":
    main()
