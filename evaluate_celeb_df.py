"""Strict Celeb-DF evaluation of one declared historical checkpoint.

Replaces the old positional batch evaluator. An old metrics CSV is never reused
by filename. First produce a canonical manifest and a source-validation
calibration with research_cli.py. Class 1 must be declared from the checkpoint's
training semantics, NOT inferred from target AUC. Explicit shell orchestration
can call this entry point for a frozen roster.
"""

from __future__ import annotations
import argparse
from pathlib import Path
from research_cli import main as research_main


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--manifest-csv", type=Path, required=True)
    parser.add_argument("--crops-dir", type=Path, required=True)
    parser.add_argument("--calibration", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--class-one", choices=["fake", "real"], required=True)
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--num-workers", type=int, default=0)
    args = parser.parse_args(argv)
    return research_main(
        [
            "evaluate-legacy",
            "--checkpoint",
            str(args.checkpoint),
            "--manifest",
            str(args.manifest_csv),
            "--root",
            str(args.crops_dir),
            "--calibration",
            str(args.calibration),
            "--output",
            str(args.output_dir),
            "--class-one",
            args.class_one,
            "--device",
            args.device,
            "--batch-size",
            str(args.batch_size),
            "--workers",
            str(args.num_workers),
        ]
    )


if __name__ == "__main__":
    raise SystemExit(main())
