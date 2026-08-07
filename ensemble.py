from __future__ import annotations

import argparse
import itertools
from concurrent.futures import ProcessPoolExecutor
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd

from src.data.paths import models_root, output_root
from src.pipelines.checkpoints import TrainedRun, discover_trained_runs
from src.pipelines.ensemble_strategies import STRATEGIES, mean
from src.pipelines.evaluation import safe_auc

SPLITS = ("val", "test", "test_d")


@dataclass(frozen=True)
class Candidate:
    name: str
    model_family: str
    fourier_mode: str
    regime: str
    seeds: tuple[int, ...]
    arrays: dict[str, dict[str, np.ndarray]]
    val_auc: float


def _read_output(run: TrainedRun, split: str) -> dict[str, np.ndarray]:
    path = run.run_dir / "results" / f"outputs_{split}.npz"
    with np.load(path) as data:
        return {key: data[key] for key in ("ids", "y_true", "probs")}


def _aggregate_runs(runs: list[TrainedRun]) -> Candidate:
    arrays = {}
    for split in SPLITS:
        values = [_read_output(run, split) for run in runs]
        reference = values[0]
        for value in values[1:]:
            if not np.array_equal(reference["ids"], value["ids"]) or not np.array_equal(reference["y_true"], value["y_true"]):
                raise ValueError(f"Seed outputs are not aligned for {runs[0].model_family}/{runs[0].fourier_mode}/{runs[0].regime}")
        arrays[split] = {
            "ids": reference["ids"], "y_true": reference["y_true"],
            "probs": np.mean([value["probs"] for value in values], axis=0),
        }
    first = runs[0]
    return Candidate(
        name=f"{first.model_family}/{first.fourier_mode}/{first.regime}",
        model_family=first.model_family, fourier_mode=first.fourier_mode, regime=first.regime,
        seeds=tuple(sorted(run.seed for run in runs)), arrays=arrays,
        val_auc=safe_auc(arrays["val"]["y_true"], arrays["val"]["probs"]),
    )


def load_candidates(root: str | Path, pool: str) -> list[Candidate]:
    grouped: dict[tuple[str, str, str], list[TrainedRun]] = {}
    for run in discover_trained_runs(root):
        if all((run.run_dir / "results" / f"outputs_{split}.npz").exists() for split in SPLITS):
            grouped.setdefault((run.model_family, run.fourier_mode, run.regime), []).append(run)
    candidates = [_aggregate_runs(runs) for runs in grouped.values()]
    if pool == "best-mode":
        best: dict[tuple[str, str], Candidate] = {}
        for candidate in candidates:
            key = (candidate.model_family, candidate.regime)
            if key not in best or candidate.val_auc > best[key].val_auc:
                best[key] = candidate
        candidates = list(best.values())
    elif pool != "all":
        raise ValueError("pool must be best-mode or all")
    return sorted(candidates, key=lambda item: item.name)


_SEARCH_PREDICTIONS = None
_SEARCH_LABELS = None


def _init_search_worker(predictions, labels):
    global _SEARCH_PREDICTIONS, _SEARCH_LABELS
    _SEARCH_PREDICTIONS, _SEARCH_LABELS = predictions, labels


def _score_subset(subset) -> tuple[tuple[int, ...], float]:
    return subset, safe_auc(
        _SEARCH_LABELS, mean([_SEARCH_PREDICTIONS[index] for index in subset]),
    )


def exhaustive_subset_search(predictions, labels, max_workers=None) -> tuple[int, ...]:
    count = len(predictions)
    if count == 0:
        raise ValueError("No predictions to search")
    combinations = [combo for size in range(1, count + 1) for combo in itertools.combinations(range(count), size)]
    with ProcessPoolExecutor(max_workers=max_workers, initializer=_init_search_worker,
                             initargs=(predictions, labels)) as executor:
        scored = list(executor.map(_score_subset, combinations,
                                   chunksize=max(1, len(combinations) // 100)))
    return max(scored, key=lambda item: (item[1], -len(item[0])))[0]


def greedy_subset_search(predictions, labels) -> tuple[int, ...]:
    remaining, selected, best_score = set(range(len(predictions))), [], -float("inf")
    while remaining:
        scored = [
            (safe_auc(labels, mean([predictions[index] for index in selected + [candidate]])), candidate)
            for candidate in remaining
        ]
        score, candidate = max(scored)
        if selected and score <= best_score:
            break
        selected.append(candidate)
        remaining.remove(candidate)
        best_score = score
    return tuple(selected)


def search_subset(predictions, labels, exhaustive=True, max_workers=None):
    return (exhaustive_subset_search(predictions, labels, max_workers)
            if exhaustive else greedy_subset_search(predictions, labels))


def run(root, pool="best-mode", strategy="search", output_dir=None, max_workers=None):
    candidates = load_candidates(root, pool)
    if not candidates:
        raise ValueError("No candidates with aligned val/test/test_d outputs")
    val_predictions = [candidate.arrays["val"]["probs"] for candidate in candidates]
    val_labels = candidates[0].arrays["val"]["y_true"]
    selected = (search_subset(val_predictions, val_labels, pool == "best-mode", max_workers)
                if strategy == "search" else tuple(range(len(candidates))))
    selected_candidates = [candidates[index] for index in selected]
    combine = "mean" if strategy == "search" else strategy
    val_auc_weights = [candidate.val_auc for candidate in selected_candidates]
    output = Path(output_dir or output_root())
    output.mkdir(parents=True, exist_ok=True)
    report = []
    for split in SPLITS:
        arrays = [candidate.arrays[split] for candidate in selected_candidates]
        reference = arrays[0]
        if any(not np.array_equal(reference["ids"], value["ids"]) for value in arrays[1:]):
            raise ValueError("Candidate prediction IDs are not aligned")
        split_predictions = [value["probs"] for value in arrays]
        if combine == "stacking":
            probabilities = STRATEGIES[combine](
                [candidate.arrays["val"]["probs"] for candidate in selected_candidates],
                val_labels, split_predictions,
            )
        elif combine == "weighted":
            probabilities = STRATEGIES[combine](split_predictions, weights=val_auc_weights)
        else:
            probabilities = STRATEGIES[combine](split_predictions)
        predictions = (probabilities >= .5).astype(int)
        pd.DataFrame({
            "id": reference["ids"], "y_true": reference["y_true"],
            "prob_pos": probabilities, "y_pred": predictions,
        }).to_csv(output / f"ensemble_predictions_{split}.csv", index=False)
        report.append({
            "split": split, "strategy": combine, "pool": pool,
            "members": len(selected_candidates),
            "member_names": ";".join(candidate.name for candidate in selected_candidates),
            "auc": safe_auc(reference["y_true"], probabilities),
        })
    frame = pd.DataFrame(report)
    frame.to_csv(output / "ensemble_report.csv", index=False)
    return frame


def main(argv=None):
    parser = argparse.ArgumentParser(description="Select ensembles on val and report held-out splits")
    parser.add_argument("--strategy", default="search", choices=tuple(STRATEGIES) + ("search",))
    parser.add_argument("--pool", default="best-mode", choices=("best-mode", "all"))
    parser.add_argument("--models-root", default=None)
    parser.add_argument("--output-dir", default=None)
    parser.add_argument("--max-workers", type=int, default=None)
    args = parser.parse_args(argv)
    print(run(args.models_root or models_root(), args.pool, args.strategy,
              args.output_dir, args.max_workers).to_string(index=False))


if __name__ == "__main__":
    main()
