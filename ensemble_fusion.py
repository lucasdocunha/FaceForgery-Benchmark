"""
Selects top-N models by validation AUC and fuses their predictions via soft voting.

Usage:
    python ensemble_fusion.py [--top-n 10] [--splits test,test_d] [--models-dir models]
"""

from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path

import numpy as np
import pandas as pd

from src.pipelines.evaluation import best_threshold, binary_metrics, clean_probabilities

logger = logging.getLogger(__name__)

THRESHOLD_METRIC = "accuracy"


def load_metrics_index(models_dir: Path) -> pd.DataFrame:
    rows = []
    seen_paths: set[str] = set()

    for p in sorted(models_dir.glob("**/results/metrics_val.csv")):
        try:
            df = pd.read_csv(p)
            if df.empty:
                continue
            row = df.iloc[0].to_dict()
            key = str(row.get("predictions_path", p))
            if key not in seen_paths:
                seen_paths.add(key)
                rows.append(row)
        except Exception as e:
            logger.warning("Could not read %s: %s", p, e)

    summary_csv = models_dir / "all_metrics_by_split.csv"
    if summary_csv.exists():
        try:
            df_summary = pd.read_csv(summary_csv)
            val_rows = df_summary[df_summary["split"] == "val"]
            for _, row in val_rows.iterrows():
                key = str(row.get("predictions_path", ""))
                if key and key not in seen_paths:
                    seen_paths.add(key)
                    rows.append(row.to_dict())
        except Exception as e:
            logger.warning("Could not read %s: %s", summary_csv, e)

    if not rows:
        raise RuntimeError(f"No val metrics found under {models_dir}")

    df = pd.DataFrame(rows)
    df["auc"] = pd.to_numeric(df["auc"], errors="coerce")
    df = df.dropna(subset=["auc"])
    return df


def select_top_n(metrics_df: pd.DataFrame, n: int) -> pd.DataFrame:
    ranked = metrics_df.sort_values("auc", ascending=False).reset_index(drop=True)
    top = ranked.head(n).copy()
    top.insert(0, "rank", range(1, len(top) + 1))
    return top


def print_summary_table(top_n_df: pd.DataFrame) -> None:
    print("\n=== Top Models Selected (by Val AUC) ===")
    header = f"{'Rank':>4}  {'Val AUC':>8}  {'Family':<12}  {'Architecture':<22}  {'Technique':<18}  Predictions Path"
    print(header)
    print("-" * len(header))
    for _, row in top_n_df.iterrows():
        print(
            f"{int(row['rank']):>4}  {float(row['auc']):>8.4f}  "
            f"{str(row.get('model_family', '?')):<12}  "
            f"{str(row.get('architecture', '?')):<22}  "
            f"{str(row.get('technique', '?')):<18}  "
            f"{row.get('predictions_path', '?')}"
        )
    print()


def load_predictions(predictions_path: str, models_dir: Path) -> pd.DataFrame:
    p = Path(predictions_path)
    if not p.is_absolute():
        # Paths in CSVs are project-root-relative (e.g. models/xception/.../predictions_val.csv)
        p = Path.cwd() / p
    if not p.exists():
        raise FileNotFoundError(f"Predictions not found: {p}")
    df = pd.read_csv(p)
    missing = {"id", "y_true", "prob_pos"} - set(df.columns)
    if missing:
        raise ValueError(f"Missing columns {missing} in {p}")
    return df


def fuse_predictions(dfs: list[pd.DataFrame]) -> pd.DataFrame:
    prob_matrix = np.stack(
        [clean_probabilities(df["prob_pos"].values) for df in dfs], axis=1
    )
    fused_probs = clean_probabilities(prob_matrix.mean(axis=1))
    return pd.DataFrame(
        {
            "id": dfs[0]["id"].values,
            "y_true": dfs[0]["y_true"].values,
            "prob_pos": fused_probs,
        }
    )


def save_results(
    split: str,
    fused_df: pd.DataFrame,
    metrics: dict,
    threshold: float,
    tag: str,
    out_dir: Path,
) -> dict:
    out_dir.mkdir(parents=True, exist_ok=True)

    y_pred = (clean_probabilities(fused_df["prob_pos"].values) >= threshold).astype(int)
    pred_df = pd.DataFrame(
        {
            "id": fused_df["id"].values,
            "y_true": fused_df["y_true"].values,
            "y_pred": y_pred,
            "prob_pos": fused_df["prob_pos"].values,
            "correct": (fused_df["y_true"].values.astype(int) == y_pred).astype(int),
        }
    )
    pred_df.to_csv(out_dir / f"predictions_{split}.csv", index=False)

    metrics_path = out_dir / f"metrics_{split}.csv"
    row = {
        "model_family": "ensemble",
        "architecture": "soft_vote",
        "technique": tag,
        "split": split,
        "auc": float(metrics["auc"]),
        "acc": float(metrics["acc"]),
        "f1": float(metrics["f1"]),
        "precision": float(metrics["precision"]),
        "threshold": float(threshold),
        "threshold_source": f"best_{THRESHOLD_METRIC}_on_val",
        "n_samples": int(len(fused_df)),
        "run_dir": str(out_dir.parent),
        "outputs_path": "",
        "metrics_path": str(metrics_path),
        "predictions_path": str(out_dir / f"predictions_{split}.csv"),
        "loss": 0.0,
        "recall": float(metrics["recall"]),
        "specificity": float(metrics["specificity"]),
        "tp": int(metrics["tp"]),
        "fp": int(metrics["fp"]),
        "fn": int(metrics["fn"]),
        "tn": int(metrics["tn"]),
    }
    pd.DataFrame([row]).to_csv(metrics_path, index=False)

    logger.info(
        "[%s]  AUC=%.4f  ACC=%.4f  F1=%.4f  Recall=%.4f  Spec=%.4f  n=%d",
        split,
        row["auc"],
        row["acc"],
        row["f1"],
        row["recall"],
        row["specificity"],
        row["n_samples"],
    )
    return row


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Ensemble fusion of top-N models by validation AUC."
    )
    parser.add_argument("--models-dir", type=Path, default=Path("models"))
    parser.add_argument("--top-n", type=int, default=10)
    parser.add_argument(
        "--splits",
        type=str,
        default="test,test_d",
        help="Comma-separated evaluation splits (default: test,test_d)",
    )
    parser.add_argument(
        "--output-tag",
        type=str,
        default=None,
        help="Override output directory tag (default: top{n}_val_auc)",
    )
    args = parser.parse_args()

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    )

    n = args.top_n
    tag = args.output_tag or f"top{n}_val_auc"
    out_dir = args.models_dir / "ensemble" / tag / "results"
    eval_splits = [s.strip() for s in args.splits.split(",") if s.strip()]

    # Step 1: Build val metrics index
    logger.info("Loading val metrics from %s ...", args.models_dir)
    metrics_df = load_metrics_index(args.models_dir)
    logger.info("Found %d models with val metrics.", len(metrics_df))

    # Step 2: Select top N
    top_n_df = select_top_n(metrics_df, n)
    print_summary_table(top_n_df)

    # Step 3: Load val predictions for top-N and fuse
    val_dfs: list[pd.DataFrame] = []
    included_rows: list[dict] = []
    for _, row in top_n_df.iterrows():
        try:
            df = load_predictions(str(row["predictions_path"]), args.models_dir)
            val_dfs.append(df)
            included_rows.append(row.to_dict())
        except (FileNotFoundError, ValueError) as e:
            logger.warning("Skipping model for val fusion: %s", e)

    if len(val_dfs) < 2:
        logger.error("Need at least 2 models for fusion. Got %d. Aborting.", len(val_dfs))
        sys.exit(1)

    if len(val_dfs) < n:
        logger.warning(
            "Only %d of %d selected models had loadable val predictions.", len(val_dfs), n
        )

    logger.info("Fusing val predictions from %d models ...", len(val_dfs))
    fused_val = fuse_predictions(val_dfs)

    # Step 4: Optimal threshold on fused val probs
    threshold, val_score = best_threshold(
        fused_val["y_true"].values.astype(int),
        fused_val["prob_pos"].values,
        metric=THRESHOLD_METRIC,
    )
    logger.info(
        "Optimal threshold on val: %.4f  (%s=%.4f)", threshold, THRESHOLD_METRIC, val_score
    )

    # Step 5 & 6: Evaluate and save for each test split
    summary_rows = []
    for split in eval_splits:
        split_dfs: list[pd.DataFrame] = []
        for row in included_rows:
            val_path = str(row["predictions_path"])
            split_path = val_path.replace("predictions_val.csv", f"predictions_{split}.csv")
            try:
                df = load_predictions(split_path, args.models_dir)
                split_dfs.append(df)
            except (FileNotFoundError, ValueError) as e:
                logger.warning("Skipping model for split=%s: %s", split, e)

        if not split_dfs:
            logger.warning("No predictions loaded for split=%s. Skipping.", split)
            continue

        logger.info("Fusing %s predictions from %d models ...", split, len(split_dfs))
        fused_split = fuse_predictions(split_dfs)
        metrics = binary_metrics(
            fused_split["y_true"].values.astype(int),
            fused_split["prob_pos"].values,
            threshold=threshold,
        )
        row = save_results(split, fused_split, metrics, threshold, tag, out_dir)
        summary_rows.append(row)

    # Final summary
    print("\n=== Ensemble Evaluation Results ===")
    print(f"Models fused: {len(val_dfs)}  |  Threshold (val {THRESHOLD_METRIC}): {threshold:.4f}")
    for row in summary_rows:
        print(
            f"  [{row['split']}]  AUC={row['auc']:.4f}  ACC={row['acc']:.4f}  "
            f"F1={row['f1']:.4f}  Recall={row['recall']:.4f}  "
            f"Specificity={row['specificity']:.4f}"
        )
    print(f"\nResults saved to: {out_dir}")


if __name__ == "__main__":
    main()
