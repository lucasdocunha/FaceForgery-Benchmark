#!/usr/bin/env python3
"""Treinamento multi-seed com pré-processamento robusto para o benchmark de forjamento facial.

Aplica RandomizedRobustAugment (ruído, compressão JPEG, rotação, contraste, superexposição, etc.)
e treina sequencialmente as seeds fornecidas (padrão: 42, 123, 2024, 7, 2025).
Ao término de cada seed, avalia imediatamente no conjunto Test Difícil (test_d) e salva
os outputs completos (.npz e .csv).

Compatível com clusters Slurm (CISIA) e servidores locais.
"""

from __future__ import annotations

import argparse
import os
import sys
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Sequence

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, WeightedRandomSampler

# Garantir que o repositório esteja no PYTHONPATH
ROOT_DIR = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT_DIR))

from src.data.augmentations import RandomizedRobustAugment, clean_transform
from src.data.data import ImageDataset
from src.data.paths import data_root, models_root, output_root, phase1_split_root
from src.models.registry import get_model_spec
from src.pipelines.checkpoints import _save_results
from src.pipelines.config import load_config
from src.pipelines.evaluation import evaluate_classifier
from src.pipelines.training import Trainer, seed_everything

DEFAULT_SEEDS = (42, 123, 2024, 7, 2025)
VALID_FAMILIES = ("resnet", "xception", "mobilenet", "vit", "clip", "dino")


@dataclass
class RunShim:
    """Shim compatível com a assinatura de checkpoints._save_results."""
    run_dir: Path
    model_family: str
    fourier_mode: str
    regime: str
    seed: int
    threshold: float


def _balanced_sampler(dataset: ImageDataset, seed: int) -> WeightedRandomSampler:
    labels = dataset.df.iloc[:, 1].astype(int).to_numpy()
    counts = np.bincount(labels, minlength=2)
    class_weights = np.divide(1.0, counts, out=np.zeros(2, dtype=float), where=counts > 0)
    sample_weights = torch.as_tensor(class_weights[labels], dtype=torch.double)
    generator = torch.Generator().manual_seed(seed)
    return WeightedRandomSampler(sample_weights, len(sample_weights), replacement=True, generator=generator)


def is_seed_completed(run_dir: Path) -> bool:
    """Verifica se uma seed já concluiu treino e avaliação no test_d."""
    results_dir = run_dir / "results"
    return (
        (results_dir / "metrics_test.csv").exists()
        and (results_dir / "metrics_test_d.csv").exists()
        and (results_dir / "outputs_test.npz").exists()
        and (results_dir / "outputs_test_d.npz").exists()
    )


def eval_test_d(
    model: nn.Module,
    family: str,
    fourier_mode: str,
    regime: str,
    seed: int,
    output_dir: Path,
    image_size: int,
    batch_size: int,
    num_workers: int,
    device: torch.device,
    test_d_images_dir: Path,
    test_d_csv: Path,
) -> dict:
    """Executa a avaliação no split test_d e grava os artefatos oficiais."""
    val_metrics_csv = output_dir / "results" / "metrics_val.csv"
    val_threshold = float(pd.read_csv(val_metrics_csv).iloc[0]["threshold"])

    test_d_dataset = ImageDataset(
        test_d_csv,
        test_d_images_dir,
        transform=clean_transform(image_size),
        fourier=fourier_mode,
        spatial_size=(image_size, image_size),
    )
    test_d_loader = DataLoader(
        test_d_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=device.type == "cuda",
        persistent_workers=num_workers > 0,
    )

    test_d_metrics = evaluate_classifier(
        model,
        test_d_loader,
        nn.CrossEntropyLoss(),
        device,
        threshold=val_threshold,
        use_amp=device.type == "cuda",
        desc=f"{family}/{regime}/seed_{seed} (test_d)",
    )

    shim = RunShim(output_dir, family, fourier_mode, regime, seed, val_threshold)
    return _save_results(shim, "test_d", test_d_metrics)


def train_single_seed(
    family: str,
    seed: int,
    regime: str = "finetune_robust",
    fourier_mode: str = "none",
    epochs: int | None = None,
    batch_size: int | None = None,
    num_workers: int = 4,
    early_stop_patience: int = 8,
    data_limit: int | None = None,
    device: torch.device | None = None,
    test_d_images_dir: Path | None = None,
    test_d_csv: Path | None = None,
    force: bool = False,
) -> dict:
    t_start = time.time()
    device = device or torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    output_dir = models_root() / family / fourier_mode / regime / f"seed_{seed}"
    if not force and is_seed_completed(output_dir):
        print(f"⏭️  Seed {seed} já concluída em {output_dir}. Carregando resultados existentes...", flush=True)
        m_test = pd.read_csv(output_dir / "results" / "metrics_test.csv").iloc[0].to_dict()
        m_test_d = pd.read_csv(output_dir / "results" / "metrics_test_d.csv").iloc[0].to_dict()
        m_val = pd.read_csv(output_dir / "results" / "metrics_val.csv").iloc[0].to_dict()
        delta_auc = m_test_d["auc"] - m_test["auc"]
        return {
            "model_family": family,
            "seed": seed,
            "regime": regime,
            "val_auc": m_val["auc"],
            "test_auc": m_test["auc"],
            "test_acc": m_test["acc"],
            "test_f1": m_test["f1"],
            "test_d_auc": m_test_d["auc"],
            "test_d_acc": m_test_d["acc"],
            "test_d_f1": m_test_d["f1"],
            "delta_auc": delta_auc,
            "elapsed_min": 0.0,
            "cached": True,
        }

    output_dir.mkdir(parents=True, exist_ok=True)
    print(f"\n{'='*70}")
    print(f"🚀 INICIANDO TREINO ROBUSTO: {family.upper()} | Seed: {seed} | Regime: {regime}")
    print(f"Diretório de saída: {output_dir}")
    print(f"{'='*70}", flush=True)

    seed_everything(seed)

    config_path = ROOT_DIR / "configs" / f"{family}.yaml"
    overrides = {
        "model_family": family,
        "fourier_mode": fourier_mode,
        "regime": "finetune" if "finetune" in regime else "scratch",
        "seed": seed,
        "early_stop_patience": early_stop_patience,
        "multi_gpu": False,
    }
    if epochs is not None:
        overrides["epochs"] = epochs
    if batch_size is not None:
        overrides["batch_size"] = batch_size
    if data_limit is not None:
        overrides["data_limit"] = data_limit

    config = load_config(config_path, overrides)
    bs = config.batch_size
    img_size = config.image_size

    # Construir modelo e descongelar para fine-tuning
    spec = get_model_spec(family)
    model = spec.build(config)
    if "finetune" in regime:
        spec.unfreeze_for_finetune(model, config.unfreeze_last_n)
    model = model.to(device)

    # Configurar datasets e loaders
    raw_dir = data_root() / config.data_split_dir
    limit = np.inf if config.data_limit is None else config.data_limit

    train_ds = ImageDataset(
        raw_dir / "train.csv",
        phase1_split_root("train"),
        transform=RandomizedRobustAugment(img_size),
        data_limit=limit,
        fourier=fourier_mode,
        spatial_size=(img_size, img_size),
    )
    val_ds = ImageDataset(
        raw_dir / "val.csv",
        phase1_split_root("val"),
        transform=clean_transform(img_size),
        data_limit=limit,
        fourier=fourier_mode,
        spatial_size=(img_size, img_size),
    )
    test_ds = ImageDataset(
        raw_dir / "test.csv",
        phase1_split_root("test"),
        transform=clean_transform(img_size),
        data_limit=limit,
        fourier=fourier_mode,
        spatial_size=(img_size, img_size),
    )

    common_loader = {
        "batch_size": bs,
        "num_workers": num_workers,
        "pin_memory": device.type == "cuda",
        "persistent_workers": num_workers > 0,
    }

    train_sampler = _balanced_sampler(train_ds, seed)
    train_loader = DataLoader(train_ds, sampler=train_sampler, **common_loader)
    val_loader = DataLoader(val_ds, shuffle=False, **common_loader)
    test_loader = DataLoader(test_ds, shuffle=False, **common_loader)

    # Executar treinamento com Early Stopping
    trainer = Trainer(
        model, train_loader, val_loader, test_loader, config, output_dir, spec, device=device
    )
    test_metrics = trainer.fit()

    # Recarregar pesos ótimos salvos pelo early stopping para avaliação no test_d
    best_weights_path = output_dir / "weights" / "best.pth"
    model.load_state_dict(torch.load(best_weights_path, map_location=device, weights_only=True))
    model.eval()

    # Resolver diretórios do split test_d
    resolved_test_d_images = test_d_images_dir or phase1_split_root("test_d")
    resolved_test_d_csv = test_d_csv or (raw_dir / "test.csv")

    test_d_metrics = eval_test_d(
        model,
        family,
        fourier_mode,
        regime,
        seed,
        output_dir,
        img_size,
        bs,
        num_workers,
        device,
        resolved_test_d_images,
        resolved_test_d_csv,
    )

    elapsed_min = (time.time() - t_start) / 60
    val_metrics = pd.read_csv(output_dir / "results" / "metrics_val.csv").iloc[0].to_dict()
    delta_auc = test_d_metrics["auc"] - test_metrics["auc"]

    # Liberar memória da GPU
    del model, trainer, train_loader, val_loader, test_loader
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    result = {
        "model_family": family,
        "seed": seed,
        "regime": regime,
        "val_auc": round(val_metrics["auc"], 4),
        "test_auc": round(test_metrics["auc"], 4),
        "test_acc": round(test_metrics["acc"], 4),
        "test_f1": round(test_metrics["f1"], 4),
        "test_d_auc": round(test_d_metrics["auc"], 4),
        "test_d_acc": round(test_d_metrics["acc"], 4),
        "test_d_f1": round(test_d_metrics["f1"], 4),
        "delta_auc": round(delta_auc, 4),
        "elapsed_min": round(elapsed_min, 1),
        "cached": False,
    }

    print(f"\n✅ Concluída Seed {seed} ({family.upper()}):")
    print(f"   • Val AUC:    {result['val_auc']*100:.2f}%")
    print(f"   • Test AUC:   {result['test_auc']*100:.2f}% | ACC: {result['test_acc']*100:.2f}%")
    print(f"   • Test_d AUC: {result['test_d_auc']*100:.2f}% | ACC: {result['test_d_acc']*100:.2f}%")
    print(f"   • ΔAUC:       {result['delta_auc']*100:+.2f}% | Tempo: {result['elapsed_min']} min", flush=True)

    return result


def main(argv: Sequence[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description="Treino multi-seed com pré-processamento robusto")
    parser.add_argument("--family", required=True, choices=VALID_FAMILIES, help="Família do modelo")
    parser.add_argument("--seeds", default="42,123,2024,7,2025", help="Lista de seeds separadas por vírgula")
    parser.add_argument("--regime", default="finetune_robust", help="Identificador do regime (padrão: finetune_robust)")
    parser.add_argument("--fourier-mode", default="none", help="Modo de Fourier (padrão: none)")
    parser.add_argument("--epochs", type=int, default=None, help="Override do número de épocas")
    parser.add_argument("--batch-size", type=int, default=None, help="Override do batch size")
    parser.add_argument("--num-workers", type=int, default=8, help="Workers do DataLoader (padrão: 8)")
    parser.add_argument("--early-stop-patience", type=int, default=8, help="Paciência do early stopping")
    parser.add_argument("--data-limit", type=int, default=None, help="Limite de amostras para depuração/dry-run")
    parser.add_argument("--test-d-images-dir", type=Path, default=None, help="Diretório das imagens do test_d")
    parser.add_argument("--test-d-csv", type=Path, default=None, help="CSV com anotações do test_d")
    parser.add_argument("--force", action="store_true", help="Reexecutar mesmo que os resultados já existam")
    args = parser.parse_args(argv)

    seeds = [int(s.strip()) for s in args.seeds.split(",") if s.strip()]

    print("\n" + "█" * 75)
    print(f"  BENCHMARK ROBUSTO CISIA — {args.family.upper()}")
    print(f"  Seeds ({len(seeds)}): {seeds} | Regime: {args.regime}")
    print("█" * 75, flush=True)

    results = []
    for seed in seeds:
        res = train_single_seed(
            family=args.family,
            seed=seed,
            regime=args.regime,
            fourier_mode=args.fourier_mode,
            epochs=args.epochs,
            batch_size=args.batch_size,
            num_workers=args.num_workers,
            early_stop_patience=args.early_stop_patience,
            data_limit=args.data_limit,
            test_d_images_dir=args.test_d_images_dir,
            test_d_csv=args.test_d_csv,
            force=args.force,
        )
        results.append(res)

    df_results = pd.DataFrame(results)

    # Salvar sumário na pasta do modelo
    summary_dir = models_root() / args.family / args.fourier_mode / args.regime
    summary_dir.mkdir(parents=True, exist_ok=True)
    summary_file = summary_dir / f"summary_seeds_{args.family}.csv"
    df_results.to_csv(summary_file, index=False)

    # Salvar também na pasta global de tabelas do TCC
    global_tables_dir = output_root() / "tables"
    global_tables_dir.mkdir(parents=True, exist_ok=True)
    global_summary = global_tables_dir / f"robust_summary_{args.family}.csv"
    df_results.to_csv(global_summary, index=False)

    print("\n\n" + "█" * 75)
    print(f"  RESUMO FINAL — {args.family.upper()} ({args.regime})")
    print("█" * 75)
    print(df_results[["seed", "val_auc", "test_auc", "test_d_auc", "delta_auc", "elapsed_min"]].to_string(index=False))

    mean_test = df_results["test_auc"].mean() * 100
    std_test = df_results["test_auc"].std() * 100
    mean_testd = df_results["test_d_auc"].mean() * 100
    std_testd = df_results["test_d_auc"].std() * 100
    mean_delta = df_results["delta_auc"].mean() * 100
    std_delta = df_results["delta_auc"].std() * 100

    print(f"\n📊 MÉDIA DAS {len(seeds)} SEEDS:")
    print(f"   • Test AUC:   {mean_test:.2f}% ± {std_test:.2f}%")
    print(f"   • Test_d AUC: {mean_testd:.2f}% ± {std_testd:.2f}%")
    print(f"   • ΔAUC:       {mean_delta:+.2f}% ± {std_delta:.2f}%")
    print(f"\n✅ Relatórios salvos em:")
    print(f"   - {summary_file}")
    print(f"   - {global_summary}\n")


if __name__ == "__main__":
    main()
