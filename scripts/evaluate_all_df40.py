"""Avaliação em lote de todos os modelos no benchmark DF40 (DeepFake-40)."""

from __future__ import annotations

import argparse
import gc
import logging
import time
from pathlib import Path
from typing import Sequence

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torchvision import transforms

from src.data.data import ImageDataset
from src.data.paths import models_root
from src.pipelines.checkpoints import (
    TrainedRun,
    config_from_run,
    discover_trained_runs,
    load_model_from_run,
)
from src.pipelines.evaluation import (
    amp_context,
    binary_metrics,
    probabilities_from_logits,
    sanitize_inputs,
    sanitize_logits,
)

logging.basicConfig(level=logging.INFO, format="[%(asctime)s] %(message)s")
logger = logging.getLogger(__name__)


def _transform(image_size: int = 224):
    return transforms.Compose([
        transforms.Resize((image_size, image_size)),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
    ])


def evaluate_run_on_cached(
    model: nn.Module,
    data_source,
    criterion: nn.Module,
    device: torch.device,
    threshold: float = 0.5,
    use_amp: bool = True,
) -> dict:
    model.eval()
    losses, y_true, all_logits = [], [], []

    with torch.no_grad():
        for batch in data_source:
            x, y = batch[0], batch[1]
            x = sanitize_inputs(x.to(device, non_blocking=True))
            y = y.to(device, non_blocking=True)

            with amp_context(device, enabled=use_amp):
                out = model(x)
            out = sanitize_logits(out)
            loss = criterion(out, y)

            losses.append(float(torch.nan_to_num(loss.detach()).item()))
            y_true.extend(y.cpu().numpy().tolist())
            all_logits.append(out.detach().cpu().numpy().astype(np.float32))

    if all_logits:
        logits = np.concatenate(all_logits)
        probs = probabilities_from_logits(logits)
    else:
        probs = np.array([], dtype=np.float32)

    y_true_np = np.array(y_true, dtype=np.int64)
    metrics = binary_metrics(y_true_np, probs, threshold=threshold)
    metrics["loss"] = float(np.mean(losses)) if losses else float("nan")
    return metrics


def save_run_metrics(run: TrainedRun, metrics: dict) -> Path:
    results_dir = run.run_dir / "results"
    results_dir.mkdir(parents=True, exist_ok=True)
    out_csv = results_dir / "metrics_df40.csv"

    row = {
        "run_name": run.run_dir.name,
        "auc": metrics.get("auc", float("nan")),
        "acc": metrics.get("acc", float("nan")),
        "f1": metrics.get("f1", float("nan")),
        "precision": metrics.get("precision", float("nan")),
        "recall": metrics.get("recall", float("nan")),
        "specificity": metrics.get("specificity", float("nan")),
        "loss": metrics.get("loss", float("nan")),
        "threshold": metrics.get("threshold", 0.5),
        "model_family": run.model_family,
        "fourier_mode": run.fourier_mode,
        "regime": run.regime,
        "seed": run.seed,
    }
    pd.DataFrame([row]).to_csv(out_csv, index=False)
    return out_csv


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--families", type=str, default="resnet,mobilenet,xception,vit,clip,dino")
    parser.add_argument("--device", type=str, default="cuda:0")
    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument("--num-workers", type=int, default=4)
    parser.add_argument("--skip-existing", action="store_true", default=True)
    parser.add_argument("--manifest-csv", type=Path, default=Path("data/df40/test.csv"))
    parser.add_argument("--modes", type=str, default=None, help="Comma-separated fourier modes to evaluate")
    args = parser.parse_args()

    device = torch.device(args.device)
    target_families = set(f.strip().lower() for f in args.families.split(","))

    root = models_root()
    all_runs = discover_trained_runs(root)
    runs = [r for r in all_runs if r.model_family.lower() in target_families]

    if args.modes:
        target_modes = set(m.strip().lower() for m in args.modes.split(","))
        runs = [r for r in runs if r.fourier_mode.lower() in target_modes]

    logger.info(f"[{args.device}] Encontrados {len(runs)} runs para famílias: {target_families}")

    if args.skip_existing:
        runs = [r for r in runs if not (r.run_dir / "results" / "metrics_df40.csv").exists()]
        logger.info(f"[{args.device}] Restam {len(runs)} runs após pular existentes.")

    if not runs:
        logger.info(f"[{args.device}] Nenhum run para avaliar.")
        return

    # Agrupar por (fourier_mode, in_channels) para cache eficiente
    groups: dict[tuple[str, int], list[TrainedRun]] = {}
    for r in runs:
        cfg = config_from_run(r)
        key = (r.fourier_mode, cfg.in_channels)
        groups.setdefault(key, []).append(r)

    criterion = nn.CrossEntropyLoss()
    total_evaluated = 0
    t_start = time.time()

    for (fourier_mode, in_channels), group_runs in groups.items():
        pending_runs = [r for r in group_runs if not (r.run_dir / "results" / "metrics_df40.csv").exists()]
        if not pending_runs:
            logger.info(f"[{args.device}] Todos os runs para {fourier_mode} já foram concluídos.")
            continue

        logger.info(f"\n[{args.device}] === Modo {fourier_mode} (in_channels={in_channels}) ({len(pending_runs)} runs pendentes) ===")

        ds = ImageDataset(
            args.manifest_csv,
            Path(""),
            transform=_transform(224),
            fourier=fourier_mode,
            spatial_size=(224, 224),
            in_channels=in_channels,
        )
        loader = DataLoader(
            ds,
            batch_size=args.batch_size,
            shuffle=False,
            num_workers=args.num_workers,
            pin_memory=True,
        )

        should_cache = (in_channels <= 3)
        if should_cache:
            t_load = time.time()
            data_source = [(x.clone(), y.clone()) for x, y, _ in loader]
            logger.info(f"[{args.device}] Cache carregado: {len(data_source)} batches em {time.time() - t_load:.1f}s")
            del loader
            del ds
            gc.collect()
        else:
            logger.info(f"[{args.device}] in_channels={in_channels} > 3: streaming direto via DataLoader para economizar memória.")
            data_source = loader

        for idx, run in enumerate(pending_runs, start=1):
            if (run.run_dir / "results" / "metrics_df40.csv").exists():
                logger.info(f"[{args.device}] Pulando {run.run_dir.name} (já existe)")
                continue

            t0 = time.time()
            try:
                model = load_model_from_run(run, device)
                res = evaluate_run_on_cached(
                    model,
                    data_source,
                    criterion,
                    device,
                    threshold=run.threshold,
                    use_amp=True,
                )
                save_path = save_run_metrics(run, res)
                dt = time.time() - t0
                total_evaluated += 1

                logger.info(
                    f"[{args.device}][{idx}/{len(pending_runs)}] {run.model_family}/{run.fourier_mode}/{run.regime}/s{run.seed} "
                    f"-> AUC: {res['auc']*100:.2f}% | ACC: {res['acc']*100:.2f}% | F1: {res['f1']*100:.2f}% ({dt:.1f}s)"
                )
            except Exception as e:
                logger.error(f"[{args.device}] Erro em {run.run_dir.name}: {e}")
            finally:
                if 'model' in locals():
                    del model
                torch.cuda.empty_cache()

        del data_source
        if not should_cache:
            del loader
            del ds
        gc.collect()

    dt_total = time.time() - t_start
    logger.info(f"[{args.device}] Concluídos {total_evaluated} modelos em {dt_total/60:.2f} minutos!")


if __name__ == "__main__":
    main()
