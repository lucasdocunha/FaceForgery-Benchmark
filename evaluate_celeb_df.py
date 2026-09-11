"""Avaliação em lote de modelos treinados no benchmark Celeb-DF v2.

Avalia todos os modelos descobertos em models_root no conjunto de teste Celeb-DF,
computando métricas tanto em nível de frame quanto em nível de vídeo (agregação),
gerando tabelas consolidadas em Markdown, CSV e LaTeX no mesmo padrão do repositório.
"""

from __future__ import annotations

import argparse
import logging
from pathlib import Path
from typing import Sequence

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torchvision import transforms

from src.data.data import ImageDataset
from src.data.paths import models_root, output_root
from src.pipelines.checkpoints import (
    TrainedRun,
    config_from_run,
    discover_trained_runs,
    load_model_from_run,
)
from src.pipelines.evaluation import binary_metrics, evaluate_classifier

logging.basicConfig(level=logging.INFO, format="[%(asctime)s] %(message)s")
logger = logging.getLogger(__name__)

METRICS = ("auc", "acc", "f1", "precision", "recall", "specificity", "loss")
GROUPS = ("model_family", "fourier_mode", "regime")


def _transform(config):
    return transforms.Compose([
        transforms.Resize((config.image_size, config.image_size)),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
    ])


def evaluate_run_on_celeb_df(
    run: TrainedRun,
    manifest_csv: Path,
    images_dir: Path,
    device: torch.device,
    batch_size: int = 64,
    num_workers: int = 4,
    limit: int | None = None,
) -> tuple[dict, dict]:
    """Executa a inferência para um modelo e calcula métricas por frame e por vídeo."""
    config = config_from_run(run)
    model = load_model_from_run(run, device)

    # Prepara dataset
    dataset = ImageDataset(
        manifest_csv,
        images_dir,
        transform=_transform(config),
        data_limit=np.inf if limit is None else limit,
        fourier=run.fourier_mode,
        spatial_size=(config.image_size, config.image_size),
        in_channels=config.in_channels,
    )

    loader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=device.type == "cuda",
        persistent_workers=num_workers > 0,
    )

    criterion = nn.CrossEntropyLoss()
    use_amp = device.type == "cuda"
    desc = f"{run.model_family}/{run.fourier_mode}/{run.regime}/s{run.seed}"

    frame_metrics = evaluate_classifier(
        model,
        loader,
        criterion,
        device,
        threshold=run.threshold,
        use_amp=use_amp,
        desc=desc,
    )

    # Carrega metadados originais do CSV para agregação por vídeo
    manifest_df = pd.read_csv(manifest_csv)
    if limit is not None:
        manifest_df = manifest_df.head(limit)
    manifest_df["prob_pos"] = frame_metrics["probs"]
    manifest_df["y_true"] = frame_metrics["y_true"]

    # --- 2. Métricas por Vídeo ---
    video_group = manifest_df.groupby("video_id")
    video_probs = video_group["prob_pos"].mean().values
    video_trues = video_group["target"].first().values
    video_metrics = binary_metrics(
        video_trues,
        video_probs,
        threshold=run.threshold,
        loss=None,
    )

    # Salva predições e métricas no diretório do run
    results_dir = run.run_dir / "results"
    results_dir.mkdir(parents=True, exist_ok=True)

    frame_row = {
        "model_family": run.model_family,
        "fourier_mode": run.fourier_mode,
        "regime": run.regime,
        "seed": run.seed,
        "split": "celeb_df",
        "threshold": run.threshold,
        **{m: frame_metrics[m] for m in METRICS if m in frame_metrics},
    }
    pd.DataFrame([frame_row]).to_csv(results_dir / "metrics_celeb_df.csv", index=False)

    video_row = {
        "model_family": run.model_family,
        "fourier_mode": run.fourier_mode,
        "regime": run.regime,
        "seed": run.seed,
        "split": "celeb_df_video",
        "threshold": run.threshold,
        **{m: video_metrics[m] for m in METRICS if m in video_metrics},
    }
    pd.DataFrame([video_row]).to_csv(results_dir / "metrics_celeb_df_video.csv", index=False)

    del model, loader, dataset
    if device.type == "cuda":
        torch.cuda.empty_cache()

    return frame_row, video_row


def aggregate_and_save(rows: list[dict], split_name: str, out_dir: Path) -> pd.DataFrame:
    raw = pd.DataFrame(rows)
    cols = list(GROUPS) + [f"{m}_{s}" for m in METRICS for s in ("mean", "std")]
    if raw.empty:
        agg = pd.DataFrame(columns=cols)
    else:
        agg = raw.groupby(list(GROUPS), dropna=False)[list(METRICS)].agg(["mean", "std"]).reset_index()
        agg.columns = ["_".join(c).rstrip("_") for c in agg.columns]
        agg["split"] = split_name
        cols_with_split = list(GROUPS) + ["split"] + [f"{m}_{s}" for m in METRICS for s in ("mean", "std")]
        agg = agg.reindex(columns=cols_with_split)

    out_dir.mkdir(parents=True, exist_ok=True)
    agg.to_csv(out_dir / f"results_{split_name}.csv", index=False)
    (out_dir / f"results_{split_name}.md").write_text(agg.to_markdown(index=False), encoding="utf-8")

    # Gera LaTeX
    paper = agg[["model_family", "fourier_mode", "regime"]].copy()
    for m in ("auc", "acc", "f1"):
        if f"{m}_mean" in agg.columns:
            paper[m.upper()] = agg.apply(
                lambda r: f"{r[f'{m}_mean']:.3f} $\\pm$ {r[f'{m}_std']:.3f}"
                if pd.notna(r.get(f"{m}_std")) else f"{r[f'{m}_mean']:.3f}",
                axis=1,
            )
    tex = paper.rename(columns={"model_family": "Model", "fourier_mode": "Mode", "regime": "Regime"}).to_latex(
        index=False, escape=False, caption=f"Celeb-DF Cross-Dataset Benchmark ({split_name})", label=f"tab:celebdf_{split_name}"
    )
    (out_dir / f"results_{split_name}.tex").write_text(tex, encoding="utf-8")
    return agg


def main():
    parser = argparse.ArgumentParser(description="Inferência em lote no Celeb-DF para todos os modelos")
    parser.add_argument("--models-root", type=Path, default=None)
    parser.add_argument("--manifest-csv", type=Path, default=Path("data/celeb_df/test.csv"))
    parser.add_argument("--crops-dir", type=Path, default=Path("/media/ssd2/lucas.ocunha/datasets/celeb_df_crops"))
    parser.add_argument("--output-dir", type=Path, default=Path("results/tables"))
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--num-workers", type=int, default=4)
    parser.add_argument("--device", default=None)
    parser.add_argument("--only-family", default=None)
    parser.add_argument("--only-mode", default=None)
    parser.add_argument("--regime", default=None, choices=("finetune", "scratch", "all"))
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument("--skip-existing", action="store_true")
    args = parser.parse_args()

    root = args.models_root or models_root()
    device = torch.device(args.device or ("cuda" if torch.cuda.is_available() else "cpu"))

    logger.info(f"Descobrindo modelos em: {root}")
    runs = discover_trained_runs(root, only_model_family=args.only_family)

    if args.only_mode:
        runs = [r for r in runs if r.fourier_mode == args.only_mode]
    if args.regime and args.regime != "all":
        runs = [r for r in runs if r.regime == args.regime]

    logger.info(f"Total de runs selecionados para avaliação: {len(runs)}")

    frame_rows = []
    video_rows = []

    for idx, run in enumerate(runs, start=1):
        f_path = run.run_dir / "results" / "metrics_celeb_df.csv"
        v_path = run.run_dir / "results" / "metrics_celeb_df_video.csv"

        if args.skip_existing and f_path.exists() and v_path.exists():
            f_row = pd.read_csv(f_path).iloc[0].to_dict()
            v_row = pd.read_csv(v_path).iloc[0].to_dict()
            frame_rows.append(f_row)
            video_rows.append(v_row)
            logger.info(f"[{idx}/{len(runs)}] [CACHED] {run.model_family}/{run.fourier_mode}/{run.regime}/s{run.seed}: Frame-AUC={f_row.get('auc', 0)*100:.2f}%, Video-AUC={v_row.get('auc', 0)*100:.2f}%")
            continue

        logger.info(f"[{idx}/{len(runs)}] Avaliando {run.model_family}/{run.fourier_mode}/{run.regime}/s{run.seed}...")
        try:
            f_row, v_row = evaluate_run_on_celeb_df(
                run,
                args.manifest_csv,
                args.crops_dir,
                device=device,
                batch_size=args.batch_size,
                num_workers=args.num_workers,
                limit=args.limit,
            )
            frame_rows.append(f_row)
            video_rows.append(v_row)
            logger.info(
                f"[{idx}/{len(runs)}] OK -> Frame AUC={f_row['auc']*100:.2f}%, Video AUC={v_row['auc']*100:.2f}%, Video ACC={v_row['acc']*100:.2f}%"
            )
        except Exception as e:
            logger.error(f"Erro ao avaliar {run.run_dir}: {e}", exc_info=True)

    # Agrega e gera as tabelas
    logger.info("Gerando tabelas consolidadas...")
    agg_frame = aggregate_and_save(frame_rows, "celeb_df_frame", args.output_dir)
    agg_video = aggregate_and_save(video_rows, "celeb_df_video", args.output_dir)

    print("\n" + "=" * 80)
    print("RESUMO DO BENCHMARK NO CELEB-DF (NÍVEL DE VÍDEO):")
    print("=" * 80)
    print(agg_video[["model_family", "fourier_mode", "regime", "auc_mean", "auc_std", "acc_mean", "acc_std", "f1_mean"]].to_markdown(index=False))


if __name__ == "__main__":
    main()
