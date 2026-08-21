from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd

from src.data.paths import models_root, output_root
from src.pipelines.checkpoints import discover_trained_runs

METRICS = ("auc", "acc", "f1", "precision", "recall", "specificity", "loss")
GROUPS = ("model_family", "fourier_mode", "regime", "split")


def collect_metrics(root) -> pd.DataFrame:
    rows = []
    for run in discover_trained_runs(root):
        for path in sorted((run.run_dir / "results").glob("metrics_*.csv")):
            split = path.stem.removeprefix("metrics_")
            if split not in {"val", "test", "test_d"}:
                continue
            frame = pd.read_csv(path)
            for _, row in frame.iterrows():
                rows.append({
                    "model_family": run.model_family, "fourier_mode": run.fourier_mode,
                    "regime": run.regime, "split": split, "seed": run.seed,
                    **{metric: row.get(metric) for metric in METRICS},
                })
    return pd.DataFrame(rows)


def aggregate_metrics(raw: pd.DataFrame) -> pd.DataFrame:
    columns = list(GROUPS) + [f"{metric}_{stat}" for metric in METRICS for stat in ("mean", "std")]
    if raw.empty:
        return pd.DataFrame(columns=columns)
    aggregate = raw.groupby(list(GROUPS), dropna=False)[list(METRICS)].agg(["mean", "std"]).reset_index()
    aggregate.columns = ["_".join(column).rstrip("_") for column in aggregate.columns]
    return aggregate[columns]


def _paper_latex(aggregate: pd.DataFrame) -> str:
    sections = []
    for split in ("val", "test", "test_d"):
        frame = aggregate[aggregate["split"] == split].copy()
        if frame.empty:
            continue
        paper = frame[["model_family", "fourier_mode", "regime"]].copy()
        for metric in ("auc", "acc", "f1"):
            paper[metric.upper()] = frame.apply(
                lambda row: f"{row[f'{metric}_mean']:.3f} $\\pm$ {row[f'{metric}_std']:.3f}"
                if pd.notna(row[f"{metric}_std"]) else f"{row[f'{metric}_mean']:.3f}", axis=1,
            )
        table = paper.rename(columns={
            "model_family": "Model", "fourier_mode": "Mode", "regime": "Regime",
        }).to_latex(index=False, escape=False, caption=f"Results for {split}", label=f"tab:results_{split}")
        sections.append(table)
    return "\n\n".join(sections) + ("\n" if sections else "")


def make_tables(root=None, out=None):
    aggregate = aggregate_metrics(collect_metrics(root or models_root()))
    destination = Path(out or output_root() / "results" / "tables")
    destination.mkdir(parents=True, exist_ok=True)
    aggregate.to_csv(destination / "results_full.csv", index=False)
    (destination / "results_full.md").write_text(aggregate.to_markdown(index=False), encoding="utf-8")
    (destination / "results_paper.tex").write_text(_paper_latex(aggregate), encoding="utf-8")
    return aggregate


def main(argv=None):
    parser = argparse.ArgumentParser(description="Aggregate run metrics over seeds")
    parser.add_argument("--models-root")
    parser.add_argument("--output-dir")
    args = parser.parse_args(argv)
    print(make_tables(args.models_root, args.output_dir).to_string(index=False))


if __name__ == "__main__":
    main()
