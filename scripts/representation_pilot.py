"""Bounded min-data G/H comparison on already verified train/validation caches."""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import numpy as np
import torch

from src.experimental.feature_training import fit_metric_run, seed_cpu
from src.experimental.features import open_cache
from src.experimental.graph_training import fit_graph_run
from src.robustness.provenance import source_identity, write_json
from src.robustness.statistics import choose_threshold, complementarity, grouped_auc_interval, summary


def summarize_pilot(output, *, draws=1000):
    """Retain all results and add paired, conditional development comparisons."""
    import pandas as pd

    output = Path(output)
    pilot = json.loads((output / "pilot.json").read_text())
    runs = {(row["family"], row["kind"], row["label_fraction"]): row
            for row in pilot["runs"] if row["status"] == "complete"}
    if not runs:
        report = {**pilot, "paired_comparisons": [], "analysis_software": source_identity()}
        write_json(output / "pilot_summary.json", report)
        return report
    predictions = {key: pd.read_csv(Path(row["run_dir"]) / "val_predictions.csv") for key, row in runs.items()}
    first_run = next(iter(runs.values()))
    first_artifact = json.loads((Path(first_run["run_dir"]) / "artifact.json").read_text())
    cache_path = first_artifact["config"]["caches"]["val"]
    val = open_cache(cache_path)
    checkpoint = val.frame.copy()
    checkpoint["p_fake"] = torch.from_numpy(np.array(val.logits)).softmax(1)[:, 1].numpy()
    comparisons = []
    pairs = [(("metric", "mlp", 1.0), ("metric", "supcon", 1.0)),
             (("metric", "raw_centroid", 1.0), ("metric", "supcon", 1.0))]
    for fraction in sorted({key[2] for key in runs if key[0] == "graph"}):
        pairs.extend([(("graph", "mlp", fraction), ("graph", kind, fraction)) for kind in ("gcn", "gat", "sage")])
    for reference, candidate in pairs:
        if reference not in runs or candidate not in runs:
            continue
        a, b = predictions[reference], predictions[candidate]
        metrics, _ = complementarity(a, b, reference_threshold=runs[reference]["metrics"]["threshold"],
                                    other_threshold=runs[candidate]["metrics"]["threshold"])
        comparisons.append({
            "reference": list(reference), "candidate": list(candidate),
            "auc_difference": grouped_auc_interval(a, b, draws=draws, seed=pilot["seed"]),
            "complementarity": metrics,
        })
    enriched = []
    for key, row in runs.items():
        root = Path(row["run_dir"])
        record = json.loads((root / "artifact.json").read_text())
        telemetry = dict(row["telemetry"])
        loss = None
        if (root / "history.json").exists():
            history = json.loads((root / "history.json").read_text())
            fit = json.loads((root / "telemetry.json").read_text())
            loss = history[fit["best_epoch"]]["train_loss"]
        elif row["family"] == "graph" and row["kind"] in {"linear", "correct_smooth"}:
            linear = np.load(root / "linear.npz")
            observed = np.load(root / "observed_labels.npy")
            selected = observed >= 0
            reference = np.load(root / "reference.npy", mmap_mode="r")
            decision = (reference[selected] @ linear["coef"].T + linear["intercept"]).ravel()
            loss = float(np.logaddexp(0, -(2 * observed[selected] - 1) * decision).mean())
            # Old pilot telemetry treated sklearn fits as nonparametric; derive
            # exact fitted coefficient counts and iteration budget from artifacts.
            telemetry.update(trainable_parameters=int(linear["coef"].size + linear["intercept"].size),
                             epochs_completed=int(record["config"]["training"].get("baseline_epochs", 30)))
        complement, _ = complementarity(checkpoint, predictions[key],
                                        reference_threshold=pilot["checkpoint_baseline"]["threshold"],
                                        other_threshold=row["metrics"]["threshold"])
        aligned = predictions[key].set_index("sample_id").loc[checkpoint.sample_id, "p_fake"].to_numpy()
        complement["score_pearson"] = float(np.corrcoef(checkpoint.p_fake, aligned)[0, 1])
        enriched.append({**row, "telemetry": telemetry, "selected_training_loss": loss,
                         "checkpoint_complementarity": complement})
    enriched_by_key = {(row["family"], row["kind"], row["label_fraction"]): row for row in enriched}
    report = {**pilot, "runs": [enriched_by_key.get((row["family"], row["kind"], row["label_fraction"]), row)
                               for row in pilot["runs"]], "paired_comparisons": comparisons,
              "analysis_software": source_identity(),
              "uncertainty_limitations": "Image-only supplied groups; conditional on fitted and validation-selected checkpoints, not independent confirmation or across-seed uncertainty.",
              "rss_scope": "Process lifetime high-water mark; not an isolated per-model peak."}
    write_json(output / "pilot_summary.json", report)
    return report


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--cache-index", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--epochs", type=int, default=10)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--fractions", type=float, nargs="+", default=[.05, .10])
    args = parser.parse_args(argv)
    if not 1 <= args.epochs <= 30:
        parser.error("A pilot must use 1 to 30 epochs")
    if args.output.exists():
        parser.error("Use a fresh output directory")
    caches = json.loads(args.cache_index.read_text())
    if set(caches) != {"train", "val"}:
        parser.error("Pilot cache index must contain exactly train and val")
    train, val = open_cache(caches["train"]), open_cache(caches["val"])
    if max(len(train.frame), len(val.frame)) > 10000:
        parser.error("This bounded pilot is for at most 10000 rows per split")
    seed_cpu(args.seed)
    args.output.mkdir(parents=True)
    record = {
        "scope": "min-data source-validation development, not final benchmark results",
        "limitations": ["HF backbone already uses full MFFI training labels and full-val model selection",
                        "validation is reused for selection and reported development metrics",
                        "one seed; no Test, Test-D, DF-40 or Celeb-DF evaluation"],
        "software": source_identity(), "seed": args.seed,
        "train_cache": train.identity, "val_cache": val.identity,
        "train_n": len(train.frame), "val_n": len(val.frame), "runs": [],
    }
    probabilities = torch.from_numpy(np.array(val.logits)).softmax(1)[:, 1].numpy()
    threshold = choose_threshold(val.frame.label, probabilities, policy="youden")
    record["checkpoint_baseline"] = summary(val.frame.label, probabilities, threshold)
    record["threshold_policy"] = "source-validation Youden, fixed fake-positive polarity"
    write_json(args.output / "pilot.json", record)
    schedule = [("metric", name, 1.0) for name in ("raw_centroid", "raw_knn", "linear", "mlp", "supcon")]
    schedule += [("graph", kind, fraction) for fraction in args.fractions
                 for kind in ("linear", "mlp", "knn", "lp", "correct_smooth", "gcn", "gat", "sage")]
    failed = False
    for family, name, fraction in schedule:
        kind = name.removeprefix("raw_")
        run = args.output / f"{family}-{name}-fraction{fraction:g}-seed{args.seed}"
        config = {
            "family": family, "run_dir": str(run), "caches": caches, "seed": args.seed,
            "model": {"kind": kind, "label_fraction": fraction, "standardize": not name.startswith("raw_"),
                      "embedding_dim": 128, "hidden_dim": 64 if family == "metric" else 32,
                      "projection_dim": 128, "hardness": 0, "temperature": .1,
                      "backend": "native", "protocol": "inductive", "fanouts": [10, 5],
                      "tsne": name == "supcon", "diagnostic_samples": 1000,
                      "graph": {"k": 10, "backend": "exact", "exact_limit": 10000,
                                "policy": "directed", "dynamic": False, "iterations": 50, "alpha": .8}},
            "training": {"device": "cpu", "cpu_threads": 2, "epochs": args.epochs,
                         "batch_size": 64 if family == "metric" else 32,
                         "baseline_epochs": 30, "lr": .001, "weight_decay": .0001,
                         "early_stop_patience": 5},
        }
        started = time.perf_counter()
        result = {"family": family, "kind": name, "label_fraction": fraction, "run_dir": str(run)}
        try:
            (fit_metric_run if family == "metric" else fit_graph_run)(config)
            result.update(status="complete", metrics=json.loads((run / "metrics.json").read_text())["frame"],
                          telemetry=json.loads((run / "pipeline_telemetry.json").read_text()),
                          label_count=json.loads((run / "label_mask.json").read_text())["selected_total"])
        except Exception as error:
            failed = True
            result.update(status="failed", error=f"{type(error).__name__}: {error}")
        result["wall_seconds"] = time.perf_counter() - started
        record["runs"].append(result)
        write_json(args.output / "pilot.json", record)
        print(json.dumps(result), flush=True)
    summarize_pilot(args.output)
    return int(failed)


if __name__ == "__main__":
    raise SystemExit(main())
