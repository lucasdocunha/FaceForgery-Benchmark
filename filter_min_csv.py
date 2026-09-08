"""Utility to filter raw CSVs based on images present in a local minimal dataset directory."""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Sequence
import pandas as pd


def filter_splits(raw_dir: Path, min_dir: Path, out_dir: Path) -> None:
    if not min_dir.exists():
        print(f"Directory {min_dir} does not exist. No CSV filtering performed.")
        return

    out_dir.mkdir(parents=True, exist_ok=True)
    splits = ["train", "val", "test"]

    for split in splits:
        csv_path = raw_dir / f"{split}.csv"
        img_dir = min_dir / split

        if not csv_path.exists():
            print(f"Skipping {split}: {csv_path} not found.")
            continue
        if not img_dir.exists():
            print(f"Skipping {split}: {img_dir} directory not found.")
            continue

        df = pd.read_csv(csv_path)
        images_in_folder = set(p.name for p in img_dir.iterdir() if p.is_file())

        before = len(df)
        df_filtered = df[df["img_name"].isin(images_in_folder)].reset_index(drop=True)
        after = len(df_filtered)

        out_path = out_dir / f"{split}.csv"
        df_filtered.to_csv(out_path, index=False)
        print(f"[{split}] {before} → {after} rows | saved to {out_path}")


def main(argv: Sequence[str] | None = None) -> None:
    parser = argparse.ArgumentParser(
        description="Filter raw CSV split manifests based on files present in a minimal image dataset directory."
    )
    parser.add_argument(
        "--raw-dir",
        type=Path,
        default=Path("data/raw"),
        help="Directory containing source CSVs (default: data/raw).",
    )
    parser.add_argument(
        "--min-dir",
        type=Path,
        default=Path("min_dataset"),
        help="Directory containing subset image splits (default: min_dataset).",
    )
    parser.add_argument(
        "--out-dir",
        type=Path,
        default=Path("data/raw_min"),
        help="Output directory for filtered CSVs (default: data/raw_min).",
    )

    args = parser.parse_args(argv)
    filter_splits(args.raw_dir, args.min_dir, args.out_dir)


if __name__ == "__main__":
    main()

