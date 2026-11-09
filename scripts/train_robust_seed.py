#!/usr/bin/env python3
"""Train the best model (CLIP ViT-B/16 none finetune) on seed 987 with randomized robust augmentations.

Pre-processing applied dynamically with random parameters per image:
- Gaussian Noise (random sigma)
- JPEG Compression (random quality factor)
- Random Rotation (random angle)
- Random Contrast & Sharpness
- Random Superexposition & Brightness
- Random Gaussian Blur
- Random Resized Crop & Horizontal Flip
"""

from __future__ import annotations

import io
import os
import random
import time
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from PIL import Image, ImageEnhance, ImageFilter
from torch.utils.data import DataLoader, WeightedRandomSampler
from torchvision import transforms

# Force GPU 0 (GPU 1 is in use)
os.environ["CUDA_VISIBLE_DEVICES"] = "0"

import sys
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.data.data import ImageDataset
from src.data.paths import data_root, models_root, phase1_split_root
from src.models.registry import get_model_spec
from src.pipelines.checkpoints import _save_results
from src.pipelines.config import load_config
from src.pipelines.evaluation import evaluate_classifier
from src.pipelines.training import Trainer, seed_everything


class RandomizedRobustAugment:
    """Dynamic, randomized image preprocessing pipeline.

    Every operation has a random probability of applying and dynamic random parameters.
    """

    def __init__(self, image_size: int = 224):
        self.image_size = image_size
        self.crop = transforms.RandomResizedCrop(image_size, scale=(0.8, 1.0))
        self.hflip = transforms.RandomHorizontalFlip(p=0.5)
        self.normalize = transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])

    def __call__(self, img: Image.Image) -> torch.Tensor:
        # Spatial crop and flip
        img = self.crop(img)
        img = self.hflip(img)

        # 1. Random Rotation (-30 to +30 deg)
        if random.random() < 0.5:
            angle = random.uniform(-30.0, 30.0)
            img = img.rotate(angle, resample=Image.BILINEAR)

        # 2. Random Contrast Adjustment (factor 0.4 to 1.8)
        if random.random() < 0.5:
            c = random.uniform(0.4, 1.8)
            img = ImageEnhance.Contrast(img).enhance(c)

        # 3. Random Superexposition / Brightness (factor 0.5 to 1.8)
        if random.random() < 0.5:
            b = random.uniform(0.5, 1.8)
            img = ImageEnhance.Brightness(img).enhance(b)

        # 4. Random Sharpness Adjustment (factor 0.2 to 2.0)
        if random.random() < 0.3:
            s = random.uniform(0.2, 2.0)
            img = ImageEnhance.Sharpness(img).enhance(s)

        # 5. Random Gaussian Blur (radius 0.5 to 2.0)
        if random.random() < 0.3:
            r = random.uniform(0.5, 2.0)
            img = img.filter(ImageFilter.GaussianBlur(r))

        # 6. Random JPEG Compression (quality 25 to 90)
        if random.random() < 0.5:
            q = random.randint(25, 90)
            buf = io.BytesIO()
            img.save(buf, format="JPEG", quality=q)
            buf.seek(0)
            img = Image.open(buf).convert("RGB")

        # Convert to Tensor
        tensor = transforms.ToTensor()(img)

        # 7. Random Gaussian Noise (sigma 0.01 to 0.08)
        if random.random() < 0.5:
            sigma = random.uniform(0.01, 0.08)
            tensor = (tensor + torch.randn_like(tensor) * sigma).clamp(0.0, 1.0)

        # Normalize with ImageNet stats
        return self.normalize(tensor)


def clean_transform(image_size: int = 224):
    return transforms.Compose([
        transforms.Resize((image_size, image_size)),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
    ])


def _balanced_sampler(dataset: ImageDataset, seed: int) -> WeightedRandomSampler:
    labels = dataset.df.iloc[:, 1].astype(int).to_numpy()
    counts = np.bincount(labels, minlength=2)
    class_weights = np.divide(1.0, counts, out=np.zeros(2, dtype=float), where=counts > 0)
    sample_weights = torch.as_tensor(class_weights[labels], dtype=torch.double)
    generator = torch.Generator().manual_seed(seed)
    return WeightedRandomSampler(sample_weights, len(sample_weights), replacement=True, generator=generator)


def main():
    seed = 987
    model_family = "clip"
    fourier_mode = "none"
    regime = "finetune"
    batch_size = 32
    epochs = 20
    early_stop_patience = 8
    num_workers = 4

    print("=" * 75)
    print(f"🚀 INICIANDO TREINAMENTO ROBUSTO COM PRÉ-PROCESSAMENTO ALEATÓRIO")
    print(f"Modelo: {model_family} (ViT-B/16)")
    print(f"Modo: {fourier_mode} | Regime: {regime} | Seed: {seed}")
    print(f"Aumentações: Ruído Gaussiano, Compressão JPEG, Rotação, Contraste, Superexposição, Blur")
    print(f"GPU: {torch.cuda.get_device_name(0)} | Batch size: {batch_size} | Épocas máx: {epochs}")
    print("=" * 75, flush=True)

    seed_everything(seed)
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    # Load base config for CLIP
    config_path = Path("configs/clip.yaml")
    config = load_config(config_path, {
        "model_family": model_family,
        "fourier_mode": fourier_mode,
        "regime": regime,
        "seed": seed,
        "epochs": epochs,
        "batch_size": batch_size,
        "early_stop_patience": early_stop_patience,
        "multi_gpu": False,
    })

    # Build model & unfreeze for fine-tuning
    spec = get_model_spec(model_family)
    model = spec.build(config)
    spec.unfreeze_for_finetune(model, config.unfreeze_last_n)

    # Build datasets and loaders
    train_dataset = ImageDataset(
        Path("data/raw/train.csv"),
        phase1_split_root("train"),
        transform=RandomizedRobustAugment(config.image_size),
        fourier=fourier_mode,
        spatial_size=(config.image_size, config.image_size),
    )
    val_dataset = ImageDataset(
        Path("data/raw/val.csv"),
        phase1_split_root("val"),
        transform=clean_transform(config.image_size),
        fourier=fourier_mode,
        spatial_size=(config.image_size, config.image_size),
    )
    test_dataset = ImageDataset(
        Path("data/raw/test.csv"),
        phase1_split_root("test"),
        transform=clean_transform(config.image_size),
        fourier=fourier_mode,
        spatial_size=(config.image_size, config.image_size),
    )

    common_loader_args = {
        "batch_size": batch_size,
        "num_workers": num_workers,
        "pin_memory": True,
        "persistent_workers": num_workers > 0,
    }

    train_sampler = _balanced_sampler(train_dataset, seed)
    train_loader = DataLoader(train_dataset, sampler=train_sampler, **common_loader_args)
    val_loader = DataLoader(val_dataset, shuffle=False, **common_loader_args)
    test_loader = DataLoader(test_dataset, shuffle=False, **common_loader_args)

    output_dir = models_root() / model_family / fourier_mode / regime / f"seed_{seed}"
    output_dir.mkdir(parents=True, exist_ok=True)
    print(f"Diretório de saída: {output_dir}\n", flush=True)

    # Train
    trainer = Trainer(
        model, train_loader, val_loader, test_loader, config, output_dir, spec, device=device
    )
    print("Iniciando fit()...", flush=True)
    test_metrics = trainer.fit()

    print("\n" + "=" * 75)
    print(" Treinamento e avaliação em Val/Test concluídos!")
    print(f"Test AUC: {test_metrics['auc']*100:.2f}% | Test ACC: {test_metrics['acc']*100:.2f}% | Test F1: {test_metrics['f1']*100:.2f}%")
    print("=" * 75, flush=True)

    # Evaluate on test_d (difficult test set)
    print("\n🔍 Avaliando modelo no Teste Difícil (test_d)...", flush=True)
    test_d_images_dir = Path("/media/ssd2/lucas.ocunha/datasets/phase1/test_d")
    test_d_dataset = ImageDataset(
        Path("data/raw/test.csv"),
        test_d_images_dir,
        transform=clean_transform(config.image_size),
        fourier=fourier_mode,
        spatial_size=(config.image_size, config.image_size),
    )
    test_d_loader = DataLoader(test_d_dataset, shuffle=False, **common_loader_args)

    # Load best model weights
    best_weights_path = output_dir / "weights" / "best.pth"
    model.load_state_dict(torch.load(best_weights_path, map_location=device, weights_only=True))
    model.eval()

    # Load threshold from val
    metrics_val_path = output_dir / "results" / "metrics_val.csv"
    val_threshold = float(pd.read_csv(metrics_val_path).iloc[0]["threshold"])

    test_d_metrics = evaluate_classifier(
        model, test_d_loader, nn.CrossEntropyLoss(), device,
        threshold=val_threshold, use_amp=True, desc="Test Difficult (test_d)"
    )

    # Mock TrainedRun for saving
    from dataclasses import dataclass
    @dataclass
    class RunShim:
        run_dir: Path
        model_family: str
        fourier_mode: str
        regime: str
        seed: int
        threshold: float

    run_shim = RunShim(output_dir, model_family, fourier_mode, regime, seed, val_threshold)
    row_d = _save_results(run_shim, "test_d", test_d_metrics)

    print("\n" + "=" * 75)
    print(f"🏆 RESULTADOS FINAIS - SEED {seed} (COM PRÉ-PROCESSAMENTO ROBUSTO):")
    print(f"  • Val AUC:     {float(pd.read_csv(metrics_val_path).iloc[0]['auc'])*100:.2f}%")
    print(f"  • Test AUC:    {test_metrics['auc']*100:.2f}% | Test ACC: {test_metrics['acc']*100:.2f}%")
    print(f"  • Test_d AUC:  {row_d['auc']*100:.2f}% | Test_d ACC: {row_d['acc']*100:.2f}%")
    delta_auc = (row_d['auc'] - test_metrics['auc']) * 100
    print(f"  • Δ AUC (Robustez): {delta_auc:.2f}%")
    print("=" * 75, flush=True)


if __name__ == "__main__":
    main()
