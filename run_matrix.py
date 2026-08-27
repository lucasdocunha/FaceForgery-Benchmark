"""CLI entry point for running the complete benchmark training matrix across GPUs."""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Sequence

from src.data.data import ALL_FOURIER_MODES
from src.pipelines.config import load_config
from src.utils.multiprocess import run_tasks_on_gpus
from train import train_from_config

FAMILIES = ("resnet", "xception", "mobilenet", "vit", "clip", "dino")

CONFIG_DIR = Path(__file__).resolve().parent / "configs"


def build_tasks(
    regime: str,
    only: Sequence[str] | None = None,
    num_workers: int | None = None,
) -> list[dict]:
    """Build task list for all combinations of family x Fourier mode x seed."""
    families = tuple(only) if only else FAMILIES
    unknown = [family for family in families if family not in FAMILIES]
    if unknown:
        raise ValueError(
            f"Unknown model families: {', '.join(unknown)}. Valid families: {', '.join(FAMILIES)}"
        )

    tasks = []
    for family in families:
        path = CONFIG_DIR / f"{family}.yaml"
        config = load_config(path)
        for mode in ALL_FOURIER_MODES:
            for seed in config.seeds:
                kwargs = {
                    "config_path": str(path),
                    "fourier": mode,
                    "regime": regime,
                    "seed": seed,
                }
                if num_workers is not None:
                    kwargs["num_workers"] = num_workers
                tasks.append(
                    {
                        "fn": train_from_config,
                        "name": f"{family}/{mode}/{regime}/seed_{seed}",
                        "kwargs": kwargs,
                    }
                )
    return tasks


def main(argv: Sequence[str] | None = None) -> None:
    parser = argparse.ArgumentParser(
        description="Execute full benchmark matrix (6 families x 7 Fourier modes x 3 seeds) across available GPUs."
    )
    parser.add_argument(
        "--regime",
        required=True,
        choices=("scratch", "finetune"),
        help="Training regime: 'scratch' (from random init) or 'finetune' (pretrained weights).",
    )
    parser.add_argument(
        "--only",
        type=str,
        default=None,
        help="Comma-separated subset of model families to run (e.g. 'resnet,vit').",
    )
    parser.add_argument(
        "--gpus",
        type=str,
        default=None,
        help="Comma-separated list of GPU indices to distribute tasks on (e.g. '0,1,2,3').",
    )
    parser.add_argument(
        "--workers-per-gpu",
        type=int,
        default=1,
        help="Number of concurrent worker processes per GPU (default: 1).",
    )
    parser.add_argument(
        "--num-workers",
        type=int,
        default=None,
        help="DataLoader num_workers per training task (e.g. 2, 1, or 0).",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Print list of tasks to execute without launching training.",
    )

    args = parser.parse_args(argv)
    only_families = [x.strip() for x in args.only.split(",")] if args.only else None
    tasks = build_tasks(args.regime, only_families, num_workers=args.num_workers)

    if args.dry_run:
        print(f"Dry run: {len(tasks)} tasks scheduled:")
        for t in tasks:
            print(f"  - {t['name']}")
        return

    gpu_ids = [int(x.strip()) for x in args.gpus.split(",") if x.strip()] if args.gpus else None
    run_tasks_on_gpus(tasks, gpus=gpu_ids, workers_per_gpu=args.workers_per_gpu)


if __name__ == "__main__":
    main()

