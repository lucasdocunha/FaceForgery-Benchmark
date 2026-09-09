#!/usr/bin/env python3
"""
Treina todos os 6 modelos (fourier_mode=none, seed=987) com RandomizedRobustAugment.
Executa sequencialmente na GPU 0.
"""

from __future__ import annotations

import io
import os
import random
import sys
import time
from pathlib import Path

os.environ["CUDA_VISIBLE_DEVICES"] = "0"

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from PIL import Image, ImageEnhance, ImageFilter
from torch.utils.data import DataLoader, WeightedRandomSampler
from torchvision import transforms

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.data.data import ImageDataset
from src.data.paths import models_root, phase1_split_root
from src.models.registry import get_model_spec
from src.pipelines.checkpoints import _save_results
from src.pipelines.config import load_config
from src.pipelines.evaluation import evaluate_classifier
from src.pipelines.training import Trainer, seed_everything


class RandomizedRobustAugment:
    def __init__(self, image_size: int = 224):
        self.image_size = image_size
        self.crop  = transforms.RandomResizedCrop(image_size, scale=(0.8, 1.0))
        self.hflip = transforms.RandomHorizontalFlip(p=0.5)
        self.normalize = transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])

    def __call__(self, img: Image.Image) -> torch.Tensor:
        img = self.crop(img)
        img = self.hflip(img)
        if random.random() < 0.5:
            img = img.rotate(random.uniform(-30.0, 30.0), resample=Image.BILINEAR)
        if random.random() < 0.5:
            img = ImageEnhance.Contrast(img).enhance(random.uniform(0.4, 1.8))
        if random.random() < 0.5:
            img = ImageEnhance.Brightness(img).enhance(random.uniform(0.5, 1.8))
        if random.random() < 0.3:
            img = ImageEnhance.Sharpness(img).enhance(random.uniform(0.2, 2.0))
        if random.random() < 0.3:
            img = img.filter(ImageFilter.GaussianBlur(random.uniform(0.5, 2.0)))
        if random.random() < 0.5:
            q = random.randint(25, 90)
            buf = io.BytesIO()
            img.save(buf, format="JPEG", quality=q)
            buf.seek(0)
            img = Image.open(buf).convert("RGB")
        tensor = transforms.ToTensor()(img)
        if random.random() < 0.5:
            sigma = random.uniform(0.01, 0.08)
            tensor = (tensor + torch.randn_like(tensor) * sigma).clamp(0.0, 1.0)
        return self.normalize(tensor)


def clean_transform(image_size: int = 224) -> transforms.Compose:
    return transforms.Compose([
        transforms.Resize((image_size, image_size)),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
    ])


def _balanced_sampler(dataset, seed: int) -> WeightedRandomSampler:
    labels = dataset.df.iloc[:, 1].astype(int).to_numpy()
    counts = np.bincount(labels, minlength=2)
    class_weights = np.divide(1.0, counts, out=np.zeros(2, dtype=float), where=counts > 0)
    sample_weights = torch.as_tensor(class_weights[labels], dtype=torch.double)
    generator = torch.Generator().manual_seed(seed)
    return WeightedRandomSampler(sample_weights, len(sample_weights), replacement=True, generator=generator)


# Configuração por família: (config_yaml, batch_size, epochs)
MODEL_CONFIGS = [
    ("clip",      "configs/clip.yaml",      16, 20),
    ("dino",      "configs/dino.yaml",      16, 20),
    ("vit",       "configs/vit.yaml",       32, 25),
    ("resnet",    "configs/resnet.yaml",    32, 25),
    ("mobilenet", "configs/mobilenet.yaml", 32, 20),
    ("xception",  "configs/xception.yaml",  32, 20),
]

SEED         = 987
FOURIER_MODE = "none"
REGIME       = "finetune"
NUM_WORKERS  = 4
EARLY_STOP   = 8
TEST_D_DIR   = Path("/media/ssd2/lucas.ocunha/datasets/phase1/test_d")


def eval_test_d(model, family, config, output_dir, batch_size):
    """Avalia model no test_d e salva resultados."""
    from dataclasses import dataclass
    device = next(model.parameters()).device
    common = dict(batch_size=batch_size, num_workers=NUM_WORKERS,
                  pin_memory=True, persistent_workers=NUM_WORKERS > 0)
    ds = ImageDataset(
        Path("data/raw/test.csv"), TEST_D_DIR,
        transform=clean_transform(config.image_size),
        fourier=FOURIER_MODE, spatial_size=(config.image_size, config.image_size),
    )
    loader = DataLoader(ds, shuffle=False, **common)
    val_threshold = float(pd.read_csv(output_dir / "results" / "metrics_val.csv").iloc[0]["threshold"])
    metrics = evaluate_classifier(
        model, loader, nn.CrossEntropyLoss(), device,
        threshold=val_threshold, use_amp=True, desc=f"{family} — test_d",
    )

    @dataclass
    class RunShim:
        run_dir: Path; model_family: str; fourier_mode: str
        regime: str; seed: int; threshold: float

    shim = RunShim(output_dir, family, FOURIER_MODE, REGIME, SEED, val_threshold)
    return _save_results(shim, "test_d", metrics), val_threshold


def train_one(family, config_yaml, batch_size, epochs):
    t0 = time.time()
    print(f"\n{'='*70}\n🚀  {family.upper()} | seed={SEED} | robust_aug=True\n{'='*70}", flush=True)

    seed_everything(SEED)
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    config = load_config(config_yaml, {
        "model_family": family, "fourier_mode": FOURIER_MODE,
        "regime": REGIME, "seed": SEED,
        "epochs": epochs, "batch_size": batch_size,
        "early_stop_patience": EARLY_STOP, "multi_gpu": False,
    })

    spec  = get_model_spec(family)
    model = spec.build(config)
    spec.unfreeze_for_finetune(model, config.unfreeze_last_n)

    sz = config.image_size
    common = dict(batch_size=batch_size, num_workers=NUM_WORKERS,
                  pin_memory=True, persistent_workers=NUM_WORKERS > 0)

    train_ds = ImageDataset(Path("data/raw/train.csv"), phase1_split_root("train"),
                            transform=RandomizedRobustAugment(sz),
                            fourier=FOURIER_MODE, spatial_size=(sz, sz))
    val_ds   = ImageDataset(Path("data/raw/val.csv"),   phase1_split_root("val"),
                            transform=clean_transform(sz),
                            fourier=FOURIER_MODE, spatial_size=(sz, sz))
    test_ds  = ImageDataset(Path("data/raw/test.csv"),  phase1_split_root("test"),
                            transform=clean_transform(sz),
                            fourier=FOURIER_MODE, spatial_size=(sz, sz))

    train_loader = DataLoader(train_ds, sampler=_balanced_sampler(train_ds, SEED), **common)
    val_loader   = DataLoader(val_ds,   shuffle=False, **common)
    test_loader  = DataLoader(test_ds,  shuffle=False, **common)

    output_dir = models_root() / family / FOURIER_MODE / REGIME / f"seed_{SEED}"
    output_dir.mkdir(parents=True, exist_ok=True)
    print(f"Dir: {output_dir}", flush=True)

    trainer = Trainer(model, train_loader, val_loader, test_loader,
                      config, output_dir, spec, device=device)
    test_m = trainer.fit()

    # Recarrega best weights para test_d
    best = output_dir / "weights" / "best.pth"
    model.load_state_dict(torch.load(best, map_location=device, weights_only=True))
    model.eval()

    row_d, _ = eval_test_d(model, family, config, output_dir, batch_size)
    val_auc  = float(pd.read_csv(output_dir / "results" / "metrics_val.csv").iloc[0]["auc"])

    elapsed = (time.time() - t0) / 60
    result = {
        "model_family": family, "fourier_mode": FOURIER_MODE,
        "regime": REGIME, "seed": SEED, "robust_aug": True,
        "val_auc":    round(val_auc, 4),
        "test_auc":   round(test_m["auc"], 4),
        "test_acc":   round(test_m["acc"], 4),
        "test_f1":    round(test_m["f1"], 4),
        "test_d_auc": round(row_d["auc"], 4),
        "test_d_acc": round(row_d["acc"], 4),
        "test_d_f1":  round(row_d["f1"], 4),
        "delta_auc":  round(row_d["auc"] - test_m["auc"], 4),
        "elapsed_min": round(elapsed, 1),
    }
    print(f"\n✅ {family.upper()} | test={result['test_auc']*100:.2f}% | test_d={result['test_d_auc']*100:.2f}% | ΔAUC={result['delta_auc']*100:+.2f}% | {elapsed:.1f}min", flush=True)
    return result


def main():
    print(f"\n{'█'*70}\n  ROBUSTO — TODOS OS MODELOS | fourier=none | seed={SEED}\n{'█'*70}", flush=True)

    rows, failed = [], []
    for family, config_yaml, batch_size, epochs in MODEL_CONFIGS:
        output_dir  = models_root() / family / FOURIER_MODE / REGIME / f"seed_{SEED}"
        best_weights = output_dir / "weights" / "best.pth"

        # CLIP seed 987 já foi treinado — só reavaliar test_d se necessário
        if best_weights.exists():
            print(f"\n⏭️  {family.upper()} — seed {SEED} já existe, pulando treino...", flush=True)
            try:
                seed_everything(SEED)
                device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
                config = load_config(config_yaml, {
                    "model_family": family, "fourier_mode": FOURIER_MODE,
                    "regime": REGIME, "seed": SEED, "batch_size": batch_size, "multi_gpu": False,
                })
                spec  = get_model_spec(family)
                model = spec.build(config)
                model.load_state_dict(torch.load(best_weights, map_location=device, weights_only=True))
                model = model.to(device)
                model.eval()
                row_d, _ = eval_test_d(model, family, config, output_dir, batch_size)
                test_auc = float(pd.read_csv(output_dir / "results" / "metrics_test.csv").iloc[0]["auc"])
                test_acc = float(pd.read_csv(output_dir / "results" / "metrics_test.csv").iloc[0]["acc"])
                test_f1  = float(pd.read_csv(output_dir / "results" / "metrics_test.csv").iloc[0]["f1"])
                val_auc  = float(pd.read_csv(output_dir / "results" / "metrics_val.csv").iloc[0]["auc"])
                rows.append({
                    "model_family": family, "fourier_mode": FOURIER_MODE,
                    "regime": REGIME, "seed": SEED, "robust_aug": True,
                    "val_auc": round(val_auc, 4), "test_auc": round(test_auc, 4),
                    "test_acc": round(test_acc, 4), "test_f1": round(test_f1, 4),
                    "test_d_auc": round(row_d["auc"], 4), "test_d_acc": round(row_d["acc"], 4),
                    "test_d_f1": round(row_d["f1"], 4),
                    "delta_auc": round(row_d["auc"] - test_auc, 4), "elapsed_min": 0,
                })
                print(f"   test={test_auc*100:.2f}% | test_d={row_d['auc']*100:.2f}% | ΔAUC={(row_d['auc']-test_auc)*100:+.2f}%", flush=True)
            except Exception as e:
                print(f"   Erro ao reavaliar: {e}", flush=True)
            continue

        try:
            rows.append(train_one(family, config_yaml, batch_size, epochs))
        except Exception as e:
            print(f"\n❌  {family.upper()} FALHOU: {e}", flush=True)
            failed.append(family)

    df = pd.DataFrame(rows)
    out = Path("/home/lucas.ocunha/tcc/tables/robust_all_models_seed987.csv")
    df.to_csv(out, index=False)

    print(f"\n\n{'█'*70}\n  RESUMO FINAL\n{'█'*70}")
    print(df[["model_family", "test_auc", "test_d_auc", "delta_auc", "elapsed_min"]].to_string(index=False))
    if failed:
        print(f"\n❌ Falharam: {failed}")
    print(f"\n✅ Salvo em {out}", flush=True)


if __name__ == "__main__":
    main()
