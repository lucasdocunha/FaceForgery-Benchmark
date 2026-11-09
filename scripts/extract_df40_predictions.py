"""Extração e armazenamento de predições (outputs_df40.npz) para ensembles rápidos no DF40."""

from __future__ import annotations

import argparse
import gc
import logging
import time
from pathlib import Path

import numpy as np
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


def evaluate_and_get_probs(
    model: nn.Module,
    cached_batches: list[tuple[torch.Tensor, torch.Tensor]],
    device: torch.device,
    use_amp: bool = True,
) -> tuple[np.ndarray, np.ndarray]:
    model.eval()
    all_logits, y_true = [], []

    with torch.no_grad():
        for x, y in cached_batches:
            x = sanitize_inputs(x.to(device, non_blocking=True))
            y = y.to(device, non_blocking=True)

            with amp_context(device, enabled=use_amp):
                out = model(x)
            out = sanitize_logits(out)

            y_true.extend(y.cpu().numpy().tolist())
            all_logits.append(out.detach().cpu().numpy().astype(np.float32))

    logits = np.concatenate(all_logits)
    probs = probabilities_from_logits(logits)
    return probs, np.array(y_true, dtype=np.int64)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--modes", type=str, default="none", help="Modos de Fourier a extrair: none, concat, etc.")
    parser.add_argument("--families", type=str, default="clip,dino,vit,resnet,mobilenet,xception")
    parser.add_argument("--device", type=str, default="cuda:0")
    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument("--num-workers", type=int, default=4)
    parser.add_argument("--manifest-csv", type=Path, default=Path("data/df40/test.csv"))
    parser.add_argument("--force", action="store_true", default=False)
    args = parser.parse_args()

    device = torch.device(args.device)
    target_modes = set(m.strip().lower() for m in args.modes.split(","))
    target_families = set(f.strip().lower() for f in args.families.split(","))

    root = models_root()
    all_runs = discover_trained_runs(root)
    runs = [
        r for r in all_runs
        if r.model_family.lower() in target_families
        and r.fourier_mode.lower() in target_modes
        and r.regime == "finetune"
    ]

    logger.info(f"[{args.device}] Encontrados {len(runs)} runs para extração.")

    if not args.force:
        pending_runs = [r for r in runs if not (r.run_dir / "results" / "outputs_df40.npz").exists()]
        logger.info(f"[{args.device}] Restam {len(pending_runs)} runs pendentes após pular existentes.")
    else:
        pending_runs = runs

    if not pending_runs:
        logger.info(f"[{args.device}] Nenhum run pendente para extração.")
        return

    # Agrupar por fourier_mode
    groups: dict[str, list[TrainedRun]] = {}
    for r in pending_runs:
        groups.setdefault(r.fourier_mode, []).append(r)

    for fourier_mode, group_runs in groups.items():
        cfg = config_from_run(group_runs[0])
        in_channels = cfg.in_channels
        logger.info(f"\n[{args.device}] === Carregando cache para {fourier_mode} (in_channels={in_channels}) ({len(group_runs)} runs) ===")

        t0_load = time.time()
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

        cached_batches = [(x.clone(), y.clone()) for x, y, _ in loader]
        logger.info(f"[{args.device}] Cache carregado: {len(cached_batches)} batches em {time.time() - t0_load:.1f}s")
        del loader
        del ds
        gc.collect()

        for idx, run in enumerate(group_runs, start=1):
            out_npz = run.run_dir / "results" / "outputs_df40.npz"
            if out_npz.exists() and not args.force:
                logger.info(f"[{args.device}] Pulando {run.run_dir.name} (já existe)")
                continue

            t0 = time.time()
            try:
                model = load_model_from_run(run, device)
                probs, y_true = evaluate_and_get_probs(model, cached_batches, device, use_amp=True)

                out_npz.parent.mkdir(parents=True, exist_ok=True)
                np.savez_compressed(out_npz, probs=probs, y_true=y_true)

                m = binary_metrics(y_true, probs, threshold=run.threshold)
                dt = time.time() - t0
                logger.info(
                    f"[{args.device}][{idx}/{len(group_runs)}] Salvo {run.model_family}/{run.fourier_mode}/s{run.seed} "
                    f"-> AUC: {m['auc']*100:.2f}% | ACC: {m['acc']*100:.2f}% ({dt:.1f}s)"
                )
            except Exception as e:
                logger.error(f"[{args.device}] Erro em {run.run_dir.name}: {e}")
            finally:
                if "model" in locals():
                    del model
                torch.cuda.empty_cache()

        del cached_batches
        gc.collect()

    logger.info(f"[{args.device}] Extração de predições concluída com sucesso!")


if __name__ == "__main__":
    main()
