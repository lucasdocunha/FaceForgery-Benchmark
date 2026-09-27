#!/usr/bin/env python3
"""Campanha de Treinamento e Avaliação Forense: Modelos com SRM e DTCWT nas 5 Sementes Canônicas.

Executa o treinamento de fine-tuning com aumento robusto estocástico e avalia imediatamente em:
1. Teste Limpo (test)
2. Teste Difícil Corrompido (test_d)
3. Cross-Dataset DF-40 (40 geradores modernos)
4. Cross-Dataset Celeb-DF v2 (avaliação por frames e agregação por vídeo)

Distribuição Multi-GPU recomendada:
- GPU 0: python scripts/run_forensics_campaign.py --gpu 0 --mode srm
- GPU 1: python scripts/run_forensics_campaign.py --gpu 1 --mode dtcwt
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
from PIL import ImageFile
ImageFile.LOAD_TRUNCATED_IMAGES = True
from sklearn.metrics import roc_auc_score, accuracy_score, f1_score
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, WeightedRandomSampler

ROOT_DIR = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT_DIR))
os.environ.setdefault("TCC_SKIP_UNREADABLE", "1")

from src.data.augmentations import RandomizedRobustAugment, clean_transform
from src.data.data import ImageDataset
from src.data.paths import data_root, models_root, output_root, phase1_split_root
from src.models.registry import get_model_spec
from src.pipelines.checkpoints import _save_results
from src.pipelines.config import load_config
from src.pipelines.evaluation import evaluate_classifier
from src.pipelines.training import Trainer, seed_everything

DEFAULT_SEEDS = (42, 123, 2024, 7, 2025)
VALID_FAMILIES = ("mobilenet", "resnet", "xception", "vit", "clip", "dino")


@dataclass
class RunShim:
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
    results_dir = run_dir / "results"
    return (
        (results_dir / "metrics_test.csv").exists()
        and (results_dir / "metrics_test_d.csv").exists()
        and (results_dir / "metrics_df40.csv").exists()
        and (results_dir / "metrics_celeb_df.csv").exists()
    )


def eval_test_d(
    model: nn.Module, family: str, fourier_mode: str, regime: str, seed: int,
    output_dir: Path, image_size: int, batch_size: int, num_workers: int,
    device: torch.device, test_d_images_dir: Path, test_d_csv: Path, val_threshold: float,
) -> dict:
    print(f"   -> Avaliando no Teste Difícil (test_d)...", flush=True)
    test_d_dataset = ImageDataset(
        test_d_csv, test_d_images_dir,
        transform=clean_transform(image_size),
        fourier=fourier_mode,
        spatial_size=(image_size, image_size),
        allow_skip_unreadable=True,
    )
    test_d_loader = DataLoader(
        test_d_dataset, batch_size=batch_size, shuffle=False,
        num_workers=num_workers, pin_memory=device.type == "cuda", persistent_workers=num_workers > 0,
    )
    metrics = evaluate_classifier(
        model, test_d_loader, nn.CrossEntropyLoss(), device,
        threshold=val_threshold, use_amp=device.type == "cuda",
        desc=f"{family}/{fourier_mode}/seed_{seed} (test_d)",
    )
    shim = RunShim(output_dir, family, fourier_mode, regime, seed, val_threshold)
    return _save_results(shim, "test_d", metrics)


def eval_df40(
    model: nn.Module, family: str, fourier_mode: str, regime: str, seed: int,
    output_dir: Path, image_size: int, batch_size: int, num_workers: int,
    device: torch.device, df40_csv: Path, val_threshold: float,
) -> dict | None:
    if not df40_csv.exists():
        print(f"   ⚠️  Arquivo DF-40 não encontrado em {df40_csv}. Pulando...", flush=True)
        return None
    print(f"   -> Avaliando no Cross-Dataset DF-40...", flush=True)
    df40_dataset = ImageDataset(
        df40_csv, Path(""), transform=clean_transform(image_size),
        fourier=fourier_mode, spatial_size=(image_size, image_size),
        allow_skip_unreadable=True,
    )
    df40_loader = DataLoader(
        df40_dataset, batch_size=batch_size, shuffle=False,
        num_workers=num_workers, pin_memory=device.type == "cuda", persistent_workers=num_workers > 0,
    )
    metrics = evaluate_classifier(
        model, df40_loader, nn.CrossEntropyLoss(), device,
        threshold=val_threshold, use_amp=device.type == "cuda",
        desc=f"{family}/{fourier_mode}/seed_{seed} (df40)",
    )
    shim = RunShim(output_dir, family, fourier_mode, regime, seed, val_threshold)
    return _save_results(shim, "df40", metrics)


def eval_celeb_df(
    model: nn.Module, family: str, fourier_mode: str, regime: str, seed: int,
    output_dir: Path, image_size: int, batch_size: int, num_workers: int,
    device: torch.device, celeb_csv: Path, crops_dir: Path, val_threshold: float,
) -> dict | None:
    if not celeb_csv.exists() or not crops_dir.exists():
        print(f"   ⚠️  Celeb-DF não encontrado. Pulando...", flush=True)
        return None
    print(f"   -> Avaliando no Cross-Dataset Celeb-DF v2 (Frame & Vídeo)...", flush=True)
    
    celeb_dataset = ImageDataset(
        celeb_csv, crops_dir, transform=clean_transform(image_size),
        fourier=fourier_mode, spatial_size=(image_size, image_size),
        allow_skip_unreadable=True,
    )
    celeb_loader = DataLoader(
        celeb_dataset, batch_size=batch_size, shuffle=False,
        num_workers=num_workers, pin_memory=device.type == "cuda", persistent_workers=num_workers > 0,
    )
    
    y_trues, p_fakes = [], []
    model.eval()
    with torch.no_grad():
        for batch in celeb_loader:
            x, y = batch[0].to(device), batch[1]
            logits = model(x)
            probs = torch.softmax(logits, dim=-1)[:, 1].cpu().numpy()
            y_trues.extend(y.numpy().tolist())
            p_fakes.extend(probs.tolist())

    y_trues = np.array(y_trues)
    p_fakes = np.array(p_fakes)

    # Frame metrics
    frame_auc = float(roc_auc_score(y_trues, p_fakes))
    frame_preds = (p_fakes >= val_threshold).astype(int)
    frame_acc = float(accuracy_score(y_trues, frame_preds))
    frame_f1 = float(f1_score(y_trues, frame_preds, zero_division=0))

    # Video metrics (agregação por média de probabilidade por vídeo)
    df_meta = pd.read_csv(celeb_csv)
    df_meta["prob_fake"] = p_fakes
    video_df = df_meta.groupby("video_id").agg({"target": "first", "prob_fake": "mean"}).reset_index()
    
    vid_y = video_df["target"].to_numpy().astype(int)
    vid_p = video_df["prob_fake"].to_numpy()
    vid_preds = (vid_p >= val_threshold).astype(int)

    video_auc = float(roc_auc_score(vid_y, vid_p))
    video_acc = float(accuracy_score(vid_y, vid_preds))
    video_f1 = float(f1_score(vid_y, vid_preds, zero_division=0))

    celeb_metrics = {
        "frame_auc": frame_auc,
        "frame_acc": frame_acc,
        "frame_f1": frame_f1,
        "video_auc": video_auc,
        "video_acc": video_acc,
        "video_f1": video_f1,
        "threshold": val_threshold,
    }
    
    # Salvar resultados
    res_dir = output_dir / "results"
    res_dir.mkdir(parents=True, exist_ok=True)
    pd.DataFrame([celeb_metrics]).to_csv(res_dir / "metrics_celeb_df.csv", index=False)
    np.savez_compressed(res_dir / "outputs_celeb_df.npz", y_true=y_trues, p_fake=p_fakes, video_y=vid_y, video_p=vid_p)

    return celeb_metrics


def is_lock_active(lock_file: Path) -> bool:
    if not lock_file.exists():
        return False
    try:
        content = lock_file.read_text().strip()
        for part in content.split(","):
            if part.startswith("pid="):
                pid = int(part.split("=")[1])
                os.kill(pid, 0)
                return True
    except (OSError, ValueError):
        try:
            lock_file.unlink(missing_ok=True)
        except Exception:
            pass
        return False
    return False


def train_single_seed(
    family: str,
    seed: int,
    regime: str = "finetune_robust",
    fourier_mode: str = "srm",
    epochs: int | None = None,
    batch_size: int = 64,
    num_workers: int = 8,
    early_stop_patience: int = 5,
    data_limit: int | None = None,
    device: torch.device | None = None,
    force: bool = False,
) -> dict:
    t_start = time.time()
    device = device or torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    output_dir = models_root() / family / fourier_mode / regime / f"seed_{seed}"
    if not force and is_seed_completed(output_dir):
        print(f"⏭️  Seed {seed} ({family.upper()} / {fourier_mode}) já concluída em {output_dir}. Pulando...", flush=True)
        m_test = pd.read_csv(output_dir / "results" / "metrics_test.csv").iloc[0].to_dict()
        m_test_d = pd.read_csv(output_dir / "results" / "metrics_test_d.csv").iloc[0].to_dict()
        m_df40 = pd.read_csv(output_dir / "results" / "metrics_df40.csv").iloc[0].to_dict()
        m_celeb = pd.read_csv(output_dir / "results" / "metrics_celeb_df.csv").iloc[0].to_dict()
        return {
            "model_family": family, "fourier_mode": fourier_mode, "seed": seed,
            "test_auc": m_test["auc"], "test_d_auc": m_test_d["auc"],
            "df40_auc": m_df40["auc"], "celeb_video_auc": m_celeb["video_auc"],
            "elapsed_min": 0.0, "cached": True,
        }

    lock_file = output_dir / ".training_lock"
    if not force and is_lock_active(lock_file):
        print(f"🔒 Seed {seed} ({family.upper()} / {fourier_mode}) está sendo executada por outro processo ativo. Pulando...", flush=True)
        return None

    output_dir.mkdir(parents=True, exist_ok=True)
    lock_file.write_text(f"pid={os.getpid()},device={device},time={time.time()}")
    print(f"\n{'='*75}")
    print(f"🚀 INICIANDO TREINO FORENSE: {family.upper()} | Modo: {fourier_mode.upper()} | Seed: {seed}")
    print(f"   Dispositivo: {device} | Diretório: {output_dir}")
    print(f"{'='*75}", flush=True)

    seed_everything(seed)

    config_path = ROOT_DIR / "configs" / f"{family}.yaml"
    overrides = {
        "model_family": family,
        "fourier_mode": fourier_mode,
        "regime": "finetune" if "finetune" in regime else "scratch",
        "seed": seed,
        "batch_size": batch_size,
        "num_workers": num_workers,
        "early_stop_patience": early_stop_patience,
        "multi_gpu": False,
    }
    if epochs is not None:
        overrides["epochs"] = epochs
    if data_limit is not None:
        overrides["data_limit"] = data_limit

    config = load_config(config_path, overrides)
    bs = config.batch_size
    img_size = config.image_size

    # Construir modelo com canais adaptados
    spec = get_model_spec(family)
    model = spec.build(config)
    if "finetune" in regime:
        spec.unfreeze_for_finetune(model, config.unfreeze_last_n)
    model = model.to(device)

    # Configurar datasets e loaders
    raw_dir = data_root() / config.data_split_dir
    limit = np.inf if config.data_limit is None else config.data_limit

    train_ds = ImageDataset(
        raw_dir / "train.csv", phase1_split_root("train"),
        transform=RandomizedRobustAugment(img_size),
        data_limit=limit, fourier=fourier_mode, spatial_size=(img_size, img_size),
        allow_skip_unreadable=True,
    )
    val_ds = ImageDataset(
        raw_dir / "val.csv", phase1_split_root("val"),
        transform=clean_transform(img_size),
        data_limit=limit, fourier=fourier_mode, spatial_size=(img_size, img_size),
        allow_skip_unreadable=True,
    )
    test_ds = ImageDataset(
        raw_dir / "test.csv", phase1_split_root("test"),
        transform=clean_transform(img_size),
        data_limit=limit, fourier=fourier_mode, spatial_size=(img_size, img_size),
        allow_skip_unreadable=True,
    )

    common_loader = {
        "batch_size": bs, "num_workers": num_workers,
        "pin_memory": device.type == "cuda", "persistent_workers": num_workers > 0,
    }
    train_sampler = _balanced_sampler(train_ds, seed)
    train_loader = DataLoader(train_ds, sampler=train_sampler, **common_loader)
    val_loader = DataLoader(val_ds, shuffle=False, **common_loader)
    test_loader = DataLoader(test_ds, shuffle=False, **common_loader)

    # Treinamento com Early Stopping
    trainer = Trainer(model, train_loader, val_loader, test_loader, config, output_dir, spec, device=device)
    test_metrics = trainer.fit()

    # Recarrega melhor modelo salvo
    best_weights_path = output_dir / "weights" / "best.pth"
    if best_weights_path.exists():
        state = torch.load(best_weights_path, map_location=device, weights_only=True)
        model.load_state_dict(state)

    val_metrics_csv = output_dir / "results" / "metrics_val.csv"
    val_threshold = float(pd.read_csv(val_metrics_csv).iloc[0]["threshold"]) if val_metrics_csv.exists() else 0.5

    # 1. Avaliação Test_d
    test_d_csv = raw_dir / "test.csv"
    test_d_images_dir = phase1_split_root("test_d")
    test_d_metrics = eval_test_d(
        model, family, fourier_mode, regime, seed, output_dir, img_size,
        bs, num_workers, device, test_d_images_dir, test_d_csv, val_threshold,
    )

    # 2. Avaliação DF-40
    df40_csv = data_root() / "df40" / "test.csv"
    df40_metrics = eval_df40(
        model, family, fourier_mode, regime, seed, output_dir, img_size,
        bs, num_workers, device, df40_csv, val_threshold,
    )

    # 3. Avaliação Celeb-DF v2
    celeb_csv = data_root() / "celeb_df" / "test.csv"
    crops_dir = Path(os.environ.get("TCC_CELEB_CROPS_DIR", "/datasets/Images/celeb_df_crops" if Path("/datasets/Images/celeb_df_crops").exists() else "/media/ssd2/lucas.ocunha/datasets/celeb_df_crops"))
    celeb_metrics = eval_celeb_df(
        model, family, fourier_mode, regime, seed, output_dir, img_size,
        bs, num_workers, device, celeb_csv, crops_dir, val_threshold,
    )

    elapsed_min = round((time.time() - t_start) / 60.0, 1)

    result = {
        "model_family": family,
        "fourier_mode": fourier_mode,
        "seed": seed,
        "test_auc": round(test_metrics["auc"], 4),
        "test_acc": round(test_metrics["acc"], 4),
        "test_d_auc": round(test_d_metrics["auc"], 4),
        "test_d_acc": round(test_d_metrics["acc"], 4),
        "df40_auc": round(df40_metrics["auc"], 4) if df40_metrics else None,
        "celeb_video_auc": round(celeb_metrics["video_auc"], 4) if celeb_metrics else None,
        "elapsed_min": elapsed_min,
        "cached": False,
    }

    print(f"\n✅ Concluída Seed {seed} ({family.upper()} / {fourier_mode}):")
    print(f"   • Test AUC:        {result['test_auc']*100:.2f}%")
    print(f"   • Test_d AUC:      {result['test_d_auc']*100:.2f}%")
    if result["df40_auc"]:
        print(f"   • DF-40 AUC:       {result['df40_auc']*100:.2f}%")
    if result["celeb_video_auc"]:
        print(f"   • Celeb-DF Video:  {result['celeb_video_auc']*100:.2f}%")
    print(f"   • Tempo:           {elapsed_min} min", flush=True)
    if lock_file.exists():
        try:
            lock_file.unlink()
        except Exception:
            pass

    return result


def main(argv: Sequence[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description="Campanha completa de treinamento SRM e DTCWT nas 5 sementes")
    parser.add_argument("--gpu", type=int, default=0, choices=[0, 1], help="Índice da GPU (0 ou 1)")
    parser.add_argument("--mode", type=str, default=None, choices=["srm", "dtcwt"], help="Modo forense (srm ou dtcwt)")
    parser.add_argument("--families", default="mobilenet,resnet,xception,vit,clip,dino", help="Famílias separadas por vírgula (ordem da mais leve à mais pesada)")
    parser.add_argument("--seeds", default="42,123,2024,7,2025", help="Seeds separadas por vírgula")
    parser.add_argument("--epochs", type=int, default=15, help="Número de épocas (padrão: 15)")
    parser.add_argument("--batch-size", type=int, default=64, help="Batch size (padrão: 64)")
    parser.add_argument("--num-workers", type=int, default=8, help="DataLoader workers (padrão: 8)")
    parser.add_argument("--early-stop-patience", type=int, default=5, help="Paciência early stopping (padrão: 5)")
    parser.add_argument("--data-limit", type=int, default=None, help="Limite de amostras para depuração rápida")
    parser.add_argument("--force", action="store_true", help="Forçar retreino")
    args = parser.parse_args(argv)

    # Configuração automática por GPU se não especificado
    mode = args.mode or ("srm" if args.gpu == 0 else "dtcwt")
    device = torch.device(f"cuda:{args.gpu}")
    families = [f.strip() for f in args.families.split(",") if f.strip()]
    seeds = [int(s.strip()) for s in args.seeds.split(",") if s.strip()]

    print("█" * 80)
    print(f"  CAMPANHA DE TREINAMENTO FORENSE: MODO {mode.upper()}")
    print(f"  GPU: {device} | Famílias: {families}")
    print(f"  Sementes ({len(seeds)}): {seeds} | Épocas: {args.epochs} | Batch: {args.batch_size}")
    print("█" * 80, flush=True)

    all_results = []
    for seed in seeds:
        print("\n" + "=" * 80)
        print(f"  🌟 INICIANDO CICLO DA SEED {seed} PARA TODOS OS MODELOS ({mode.upper()})")
        print(f"  Ordem: {families}")
        print("=" * 80, flush=True)

        seed_results = []
        for fam in families:
            res = train_single_seed(
                family=fam,
                seed=seed,
                regime="finetune_robust",
                fourier_mode=mode,
                epochs=args.epochs,
                batch_size=args.batch_size,
                num_workers=args.num_workers,
                early_stop_patience=args.early_stop_patience,
                data_limit=args.data_limit,
                device=device,
                force=args.force,
            )
            if res is not None:
                seed_results.append(res)
                all_results.append(res)

            # Grava progresso incremental a cada modelo finalizado
            df_current = pd.DataFrame(all_results)
            progress_table = output_root() / "tables" / f"campaign_results_{mode}_progress.csv"
            progress_table.parent.mkdir(parents=True, exist_ok=True)
            df_current.to_csv(progress_table, index=False)

        # Salva o resumo de todos os modelos para esta seed específica
        df_seed = pd.DataFrame(seed_results)
        seed_table = output_root() / "tables" / f"campaign_results_{mode}_seed_{seed}.csv"
        df_seed.to_csv(seed_table, index=False)
        print(f"\n[✓] Ciclo completo da Seed {seed} finalizado! Tabela salva em: {seed_table}\n", flush=True)

    df_all = pd.DataFrame(all_results)
    out_table = output_root() / "tables" / f"campaign_results_{mode}_5seeds.csv"
    out_table.parent.mkdir(parents=True, exist_ok=True)
    df_all.to_csv(out_table, index=False)
    print(f"\n[✓] Campanha finalizada com sucesso! Relatório completo salvo em: {out_table}", flush=True)


if __name__ == "__main__":
    main()
