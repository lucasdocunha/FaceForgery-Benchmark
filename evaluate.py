"""CLI entry point for evaluating trained models across dataset splits."""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Sequence

from src.data.paths import data_root, models_root
from src.pipelines.checkpoints import evaluate_trained_runs


def main(argv: Sequence[str] | None = None) -> None:
    parser = argparse.ArgumentParser(
        description="Evaluate trained models across dataset partitions (val, test, test_d) and save predictions."
    )
    parser.add_argument(
        "--models-root",
        type=Path,
        default=None,
        help="Root directory containing trained model checkpoints (defaults to TCC_MODELS_ROOT or ./models).",
    )
    parser.add_argument(
        "--data-dir",
        type=Path,
        default=None,
        help="Directory containing dataset CSVs (defaults to TCC_DATA_ROOT/raw or data/raw).",
    )
    parser.add_argument(
        "--splits",
        type=str,
        default="val,test",
        help="Comma-separated dataset splits to evaluate (e.g. 'val,test' or 'val,test,test_d').",
    )
    parser.add_argument(
        "--test-d-csv",
        type=Path,
        default=None,
        help="Optional path to degraded test set CSV (e.g. MFFI test_d.csv).",
    )
    parser.add_argument(
        "--test-d-images-dir",
        type=Path,
        default=None,
        help="Optional directory containing degraded test set images.",
    )
    parser.add_argument(
        "--only-model-family",
        type=str,
        default=None,
        help="Optional filter to only evaluate a specific model family (e.g. 'resnet', 'vit', 'dinov3').",
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=32,
        help="Batch size for evaluation dataloaders (default: 32).",
    )
    parser.add_argument(
        "--num-workers",
        type=int,
        default=0,
        help="Number of workers for data loading (default: 0).",
    )
    parser.add_argument(
        "--limit-per-split",
        type=int,
        default=None,
        help="Optional sample limit per split for quick smoke evaluation.",
    )

    args = parser.parse_args(argv)

    root = args.models_root or models_root()
    data_directory = args.data_dir or (data_root() / "raw")
    splits = tuple(s.strip() for s in args.splits.split(",") if s.strip())

    frame = evaluate_trained_runs(
        models_root=root,
        data_dir=data_directory,
        splits=splits,
        test_d_csv=args.test_d_csv,
        test_d_images_dir=args.test_d_images_dir,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        only_model_family=args.only_model_family,
        limit_per_split=args.limit_per_split,
    )
    print(f"Evaluation completed. Evaluated and saved metrics for {len(frame)} run splits.")


if __name__ == "__main__":
    main()

