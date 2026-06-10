"""
Ensemble fusion utilities.

Subcommands
-----------
top-n   Fuse top-N models by validation AUC (original behaviour).
search  Combinatorial search: sample a fraction of all size-2..max-k subsets.
"""

from __future__ import annotations

import argparse
import itertools
import logging
import math
import os
import random
import sys
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path

from tqdm import tqdm

import numpy as np
import pandas as pd

from src.pipelines.evaluation import best_threshold, binary_metrics, clean_probabilities

logger = logging.getLogger(__name__)

THRESHOLD_METRIC = "accuracy"


# ---------------------------------------------------------------------------
# Shared utilities
# ---------------------------------------------------------------------------

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


# ---------------------------------------------------------------------------
# top-n subcommand
# ---------------------------------------------------------------------------

def main_top_n(args: argparse.Namespace) -> None:
    n = args.top_n
    tag = args.output_tag or f"top{n}_val_auc"
    out_dir = args.models_dir / "ensemble" / tag / "results"
    eval_splits = [s.strip() for s in args.splits.split(",") if s.strip()]

    logger.info("Loading val metrics from %s ...", args.models_dir)
    metrics_df = load_metrics_index(args.models_dir)
    logger.info("Found %d models with val metrics.", len(metrics_df))

    top_n_df = select_top_n(metrics_df, n)
    print_summary_table(top_n_df)

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

    threshold, val_score = best_threshold(
        fused_val["y_true"].values.astype(int),
        fused_val["prob_pos"].values,
        metric=THRESHOLD_METRIC,
    )
    logger.info(
        "Optimal threshold on val: %.4f  (%s=%.4f)", threshold, THRESHOLD_METRIC, val_score
    )

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

    print("\n=== Ensemble Evaluation Results ===")
    print(f"Models fused: {len(val_dfs)}  |  Threshold (val {THRESHOLD_METRIC}): {threshold:.4f}")
    for row in summary_rows:
        print(
            f"  [{row['split']}]  AUC={row['auc']:.4f}  ACC={row['acc']:.4f}  "
            f"F1={row['f1']:.4f}  Recall={row['recall']:.4f}  "
            f"Specificity={row['specificity']:.4f}"
        )
    print(f"\nResults saved to: {out_dir}")


# ---------------------------------------------------------------------------
# search subcommand
# ---------------------------------------------------------------------------

def _model_label(row: dict) -> str:
    family = row.get("model_family", "?")
    arch = row.get("architecture", "?")
    technique = row.get("technique", "?")
    return f"{family}/{arch}/{technique}"


def generate_and_sample_combinations(
    n_models: int,
    min_k: int,
    max_k: int,
    fraction: float,
    seed: int,
    max_combos: int,
) -> list[tuple[int, ...]]:
    rng = random.Random(seed)

    max_k = min(max_k, n_models)
    if min_k > max_k:
        raise ValueError(f"min_k={min_k} > available models={n_models}")

    size_counts = [(k, math.comb(n_models, k)) for k in range(min_k, max_k + 1)]
    total = sum(c for _, c in size_counts)
    target = min(max(1, int(total * fraction)), max_combos)

    logger.info(
        "Combinations (k=%d..%d, N=%d): total=%d  sampling %d (%.2f%%)",
        min_k, max_k, n_models, total, target, 100.0 * target / max(total, 1),
    )

    # For small spaces: enumerate all then sample
    if total <= max(target * 4, 200_000):
        all_combos: list[tuple[int, ...]] = []
        for k, _ in size_counts:
            all_combos.extend(itertools.combinations(range(n_models), k))
        return rng.sample(all_combos, min(target, len(all_combos)))

    # For large spaces: generate random combos with dedup
    sizes = [k for k, _ in size_counts]
    weights = [float(c) for _, c in size_counts]
    seen: set[tuple[int, ...]] = set()
    result: list[tuple[int, ...]] = []
    max_attempts = target * 50

    for _ in range(max_attempts):
        if len(result) >= target:
            break
        k = rng.choices(sizes, weights=weights)[0]
        combo = tuple(sorted(rng.sample(range(n_models), k)))
        if combo not in seen:
            seen.add(combo)
            result.append(combo)

    if len(result) < target:
        logger.warning(
            "Generated only %d unique combinations (target: %d). "
            "Consider reducing --sample-fraction or --max-combos.",
            len(result), target,
        )
    return result


def _eval_combo_worker(args: tuple) -> tuple[int, str, dict | None]:
    """Top-level wrapper for ProcessPoolExecutor (must be picklable)."""
    combo_idx, model_rows, eval_splits, models_dir_str = args
    metrics = _eval_combo(model_rows, eval_splits, Path(models_dir_str))
    model_ids = "|".join(_model_label(r) for r in model_rows)
    return combo_idx, model_ids, metrics


def _eval_combo(
    model_rows: list[dict],
    eval_splits: list[str],
    models_dir: Path,
) -> dict | None:
    """Load val preds, fuse, threshold, evaluate all splits. Returns metrics dict or None."""
    val_dfs: list[pd.DataFrame] = []
    for row in model_rows:
        try:
            df = load_predictions(str(row["predictions_path"]), models_dir)
            val_dfs.append(df)
        except (FileNotFoundError, ValueError):
            return None

    if len(val_dfs) < 2:
        return None

    fused_val = fuse_predictions(val_dfs)
    threshold, _ = best_threshold(
        fused_val["y_true"].values.astype(int),
        fused_val["prob_pos"].values,
        metric=THRESHOLD_METRIC,
    )
    val_m = binary_metrics(
        fused_val["y_true"].values.astype(int),
        fused_val["prob_pos"].values,
        threshold=threshold,
    )

    result: dict = {
        "threshold": float(threshold),
        "val_auc": float(val_m["auc"]),
        "val_acc": float(val_m["acc"]),
        "val_f1": float(val_m["f1"]),
        "val_recall": float(val_m["recall"]),
        "val_specificity": float(val_m["specificity"]),
        "val_precision": float(val_m["precision"]),
    }

    for split in eval_splits:
        split_dfs: list[pd.DataFrame] = []
        for row in model_rows:
            val_path = str(row["predictions_path"])
            split_path = val_path.replace("predictions_val.csv", f"predictions_{split}.csv")
            try:
                df = load_predictions(split_path, models_dir)
                split_dfs.append(df)
            except (FileNotFoundError, ValueError):
                pass

        if not split_dfs:
            continue

        fused_split = fuse_predictions(split_dfs)
        split_m = binary_metrics(
            fused_split["y_true"].values.astype(int),
            fused_split["prob_pos"].values,
            threshold=threshold,
        )
        result[f"{split}_auc"] = float(split_m["auc"])
        result[f"{split}_acc"] = float(split_m["acc"])
        result[f"{split}_f1"] = float(split_m["f1"])
        result[f"{split}_recall"] = float(split_m["recall"])
        result[f"{split}_specificity"] = float(split_m["specificity"])
        result[f"{split}_precision"] = float(split_m["precision"])

    return result


def main_search(args: argparse.Namespace) -> None:
    tag = args.output_tag or f"search_{args.strategy}_k{args.min_k}_{args.max_k}"
    out_base = args.models_dir / "ensemble" / tag
    out_base.mkdir(parents=True, exist_ok=True)
    eval_splits = [s.strip() for s in args.splits.split(",") if s.strip()]

    logger.info("Loading val metrics from %s ...", args.models_dir)
    metrics_df = load_metrics_index(args.models_dir)
    logger.info("Found %d models with val metrics.", len(metrics_df))

    if args.strategy == "top_auc":
        pool_df = metrics_df.sort_values("auc", ascending=False).head(args.top_m).reset_index(drop=True)
        logger.info("top_auc strategy: using top-%d models (val AUC %.4f – %.4f)",
                    len(pool_df), pool_df["auc"].min(), pool_df["auc"].max())
    else:
        pool_df = metrics_df.reset_index(drop=True)

    pool = pool_df.to_dict("records")
    n = len(pool)

    if n < args.min_k:
        logger.error("Not enough models (%d) for min_k=%d. Aborting.", n, args.min_k)
        sys.exit(1)

    combos = generate_and_sample_combinations(
        n_models=n,
        min_k=args.min_k,
        max_k=args.max_k,
        fraction=args.sample_fraction,
        seed=args.seed,
        max_combos=args.max_combos,
    )
    n_workers = os.cpu_count() if args.workers == -1 else args.workers
    logger.info("Running %d combinations (workers=%d) ...", len(combos), n_workers)

    summary_rows: list[dict] = []
    best_auc = 0.0

    if n_workers == 1:
        with tqdm(total=len(combos), desc="search", unit="combo") as pbar:
            for i, combo_indices in enumerate(combos):
                model_rows = [pool[idx] for idx in combo_indices]
                labels = " | ".join(_model_label(r) for r in model_rows)
                pbar.set_description(f"k={len(combo_indices)}")
                pbar.set_postfix_str(f"{labels[:80]}  best={best_auc:.4f}")

                metrics = _eval_combo(model_rows, eval_splits, args.models_dir)
                if metrics is not None:
                    best_auc = max(best_auc, metrics["val_auc"])
                    model_ids = "|".join(_model_label(r) for r in model_rows)
                    summary_rows.append({
                        "combo_id": i,
                        "n_models": len(combo_indices),
                        "model_ids": model_ids,
                        **metrics,
                    })
                pbar.update(1)
    else:
        worker_args = [
            (i, [pool[idx] for idx in combo], eval_splits, str(args.models_dir))
            for i, combo in enumerate(combos)
        ]
        with ProcessPoolExecutor(max_workers=n_workers) as executor:
            futures = {executor.submit(_eval_combo_worker, arg): arg for arg in worker_args}
            with tqdm(total=len(combos), desc=f"search ({n_workers}w)", unit="combo") as pbar:
                for future in as_completed(futures):
                    combo_idx, model_ids, metrics = future.result()
                    if metrics is not None:
                        best_auc = max(best_auc, metrics["val_auc"])
                        pbar.set_postfix_str(
                            f"best={best_auc:.4f}  last={model_ids[:50]}"
                        )
                        _, model_rows_arg, _, _ = futures[future]
                        summary_rows.append({
                            "combo_id": combo_idx,
                            "n_models": len(model_rows_arg),
                            "model_ids": model_ids,
                            **metrics,
                        })
                    pbar.update(1)

    if not summary_rows:
        logger.error("No combinations produced valid results. Aborting.")
        sys.exit(1)

    summary_df = pd.DataFrame(summary_rows).sort_values("val_auc", ascending=False).reset_index(drop=True)
    summary_path = out_base / "summary.csv"
    summary_df.to_csv(summary_path, index=False)
    logger.info("Saved summary to %s", summary_path)

    _print_search_results(summary_df, eval_splits, len(combos))


def _print_search_results(df: pd.DataFrame, eval_splits: list[str], total_tried: int) -> None:
    print(f"\n=== Combination Search Results ({total_tried} combos tried, {len(df)} valid) ===")
    top10 = df.head(10)

    print("\n--- Top 10 by Val AUC ---")
    _print_combo_table(top10, eval_splits)

    first_test = next((s for s in eval_splits if f"{s}_auc" in df.columns), None)
    if first_test:
        top10_test = df.sort_values(f"{first_test}_auc", ascending=False).head(10)
        print(f"\n--- Top 10 by {first_test} AUC ---")
        _print_combo_table(top10_test, eval_splits)


def _print_combo_table(df: pd.DataFrame, eval_splits: list[str]) -> None:
    split_headers = "  ".join(f"{s.upper()+' AUC':>10}" for s in eval_splits if f"{s}_auc" in df.columns)
    header = f"{'#':>4}  {'k':>3}  {'Val AUC':>8}  {split_headers}  Models"
    print(header)
    print("-" * min(len(header) + 40, 120))
    for rank, (_, row) in enumerate(df.iterrows(), 1):
        split_vals = "  ".join(
            f"{row[f'{s}_auc']:>10.4f}" for s in eval_splits if f"{s}_auc" in df.columns
        )
        models_short = row["model_ids"][:60] + ("..." if len(row["model_ids"]) > 60 else "")
        print(f"{rank:>4}  {int(row['n_models']):>3}  {row['val_auc']:>8.4f}  {split_vals}  {models_short}")


# ---------------------------------------------------------------------------
# Reeval subcommand
# ---------------------------------------------------------------------------

def main_reeval(args: argparse.Namespace) -> None:
    summary_csv = args.summary_csv
    if summary_csv is None:
        candidates = sorted(args.models_dir.glob("ensemble/**/summary.csv"))
        if not candidates:
            logger.error("No summary.csv found under %s/ensemble/", args.models_dir)
            sys.exit(1)
        summary_csv = candidates[-1]
        logger.info("Auto-selected summary: %s", summary_csv)

    summary_csv = Path(summary_csv)
    if not summary_csv.exists():
        logger.error("Summary CSV not found: %s", summary_csv)
        sys.exit(1)

    eval_splits = [s.strip() for s in args.splits.split(",") if s.strip()]

    logger.info("Loading val metrics index from %s ...", args.models_dir)
    metrics_df = load_metrics_index(args.models_dir)
    label_to_row: dict[str, dict] = {_model_label(r): r for r in metrics_df.to_dict("records")}

    summary_df = pd.read_csv(summary_csv)
    summary_df = summary_df.sort_values("val_auc", ascending=False).head(args.top_n).reset_index(drop=True)

    print(f"\n=== Re-evaluating Top-{args.top_n} Combos on {eval_splits} ===")
    result_rows: list[dict] = []

    for rank, (_, combo_row) in enumerate(summary_df.iterrows(), 1):
        model_ids_str = str(combo_row["model_ids"])
        labels = [lbl.strip() for lbl in model_ids_str.split("|")]

        model_rows = []
        missing = []
        for lbl in labels:
            if lbl in label_to_row:
                model_rows.append(label_to_row[lbl])
            else:
                missing.append(lbl)

        if missing:
            logger.warning("Combo %d: could not find rows for %s — skipping", rank, missing)
            continue

        print(f"\n[{rank}] val_auc={float(combo_row['val_auc']):.4f}  models: {model_ids_str}")
        metrics = _eval_combo(model_rows, eval_splits, args.models_dir)
        if metrics is None:
            print("  => FAILED (could not load predictions)")
            continue

        for split in eval_splits:
            if f"{split}_auc" in metrics:
                print(f"  [{split}]  AUC={metrics[f'{split}_auc']:.4f}  "
                      f"ACC={metrics[f'{split}_acc']:.4f}  "
                      f"F1={metrics[f'{split}_f1']:.4f}  "
                      f"Recall={metrics[f'{split}_recall']:.4f}  "
                      f"Specificity={metrics[f'{split}_specificity']:.4f}")

        row = {"rank": rank, "model_ids": model_ids_str, "n_models": len(labels),
               "val_auc": metrics["val_auc"], "threshold": metrics["threshold"]}
        for split in eval_splits:
            for metric in ("auc", "acc", "f1", "recall", "specificity", "precision"):
                key = f"{split}_{metric}"
                if key in metrics:
                    row[key] = metrics[key]
        result_rows.append(row)

    if result_rows:
        out_path = summary_csv.parent / f"reeval_top{args.top_n}.csv"
        pd.DataFrame(result_rows).to_csv(out_path, index=False)
        print(f"\nSaved results to {out_path}")


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Ensemble fusion of trained models.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    sub = parser.add_subparsers(dest="subcommand", metavar="subcommand")

    # -- top-n ---------------------------------------------------------------
    p_top = sub.add_parser(
        "top-n",
        help="Fuse top-N models by validation AUC.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    p_top.add_argument("--models-dir", type=Path, default=Path("models"))
    p_top.add_argument("--top-n", type=int, default=10)
    p_top.add_argument("--splits", type=str, default="test,test_d",
                       help="Comma-separated evaluation splits")
    p_top.add_argument("--output-tag", type=str, default=None)

    # -- search --------------------------------------------------------------
    p_search = sub.add_parser(
        "search",
        help="Combinatorial search over model subsets.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    p_search.add_argument("--models-dir", type=Path, default=Path("models"))
    p_search.add_argument("--strategy", choices=["random", "top_auc"], default="random",
                          help="random: sample from all models; top_auc: restrict to top-M first")
    p_search.add_argument("--top-m", type=int, default=20,
                          help="Pool size for top_auc strategy")
    p_search.add_argument("--min-k", type=int, default=2,
                          help="Minimum number of models per combination")
    p_search.add_argument("--max-k", type=int, default=10,
                          help="Maximum number of models per combination")
    p_search.add_argument("--sample-fraction", type=float, default=0.05,
                          help="Fraction of total combinations to try")
    p_search.add_argument("--max-combos", type=int, default=5000,
                          help="Hard cap on number of combinations to evaluate")
    p_search.add_argument("--seed", type=int, default=42)
    p_search.add_argument("--splits", type=str, default="test,test_d",
                          help="Comma-separated evaluation splits")
    p_search.add_argument("--output-tag", type=str, default=None)
    p_search.add_argument("--workers", type=int, default=1,
                          help="Processos paralelos (1=sequencial, -1=todos os CPUs)")

    # -- reeval --------------------------------------------------------------
    p_reeval = sub.add_parser(
        "reeval",
        help="Re-evaluate top-N combos from a search summary on test/test_d.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    p_reeval.add_argument("--summary-csv", type=Path, default=None,
                          help="Path to summary.csv from a previous search run (auto-detected if omitted)")
    p_reeval.add_argument("--top-n", type=int, default=3,
                          help="Number of top combos (by val AUC) to evaluate")
    p_reeval.add_argument("--splits", type=str, default="test,test_d",
                          help="Comma-separated evaluation splits")
    p_reeval.add_argument("--models-dir", type=Path, default=Path("models"))

    return parser


def main() -> None:
    parser = _build_parser()
    args = parser.parse_args()

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    )

    if args.subcommand == "top-n":
        main_top_n(args)
    elif args.subcommand == "search":
        main_search(args)
    elif args.subcommand == "reeval":
        main_reeval(args)
    else:
        parser.print_help()
        sys.exit(1)


if __name__ == "__main__":
    main()
