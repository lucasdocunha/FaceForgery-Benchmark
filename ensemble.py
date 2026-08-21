from __future__ import annotations

import argparse
import itertools
import multiprocessing as mp
from concurrent.futures import ProcessPoolExecutor
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd

from src.data.paths import models_root, output_root
from src.pipelines.checkpoints import TrainedRun, discover_trained_runs
from src.pipelines.ensemble_strategies import STRATEGIES, mean
from src.pipelines.evaluation import safe_auc

# Splits pedidos por padrão. Os held-out que não tiverem sido avaliados são
# descartados em resolve_splits() em vez de zerarem o pool de candidatos:
# Trainer grava val/test e `evaluate.py` só produz test_d quando recebe
# --test-d-csv/--test-d-images-dir.
DEFAULT_SPLITS = ("val", "test", "test_d")

# A seleção de subconjunto/pesos acontece aqui; os demais splits são só relatório.
SELECTION_SPLIT = "val"

# Acima disso a busca exaustiva (2^N-1 subconjuntos) deixa de ser viável e caímos
# no greedy incremental. 12 é o tamanho do pool --pool best-mode (6 famílias x 2 regimes).
EXHAUSTIVE_MAX_CANDIDATES = 12


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


def _has_output(run: TrainedRun, split: str) -> bool:
    return (run.run_dir / "results" / f"outputs_{split}.npz").exists()


def resolve_splits(runs: list[TrainedRun], requested: tuple[str, ...]) -> tuple[str, ...]:
    """Reduz os splits pedidos aos que existem para todos os runs com val.

    Descartar um held-out não avaliado é melhor que exigi-lo: exigir fazia
    `python ensemble.py` falhar com "No candidates" logo após o fluxo
    documentado (train + evaluate --splits val,test).
    """
    if SELECTION_SPLIT not in requested:
        raise ValueError(f"splits precisa incluir '{SELECTION_SPLIT}': a seleção acontece nele")
    with_val = [run for run in runs if _has_output(run, SELECTION_SPLIT)]
    if not with_val:
        raise ValueError(
            f"Nenhum run tem outputs_{SELECTION_SPLIT}.npz em results/. "
            "Rode train.py e evaluate.py antes do ensemble."
        )
    available = tuple(
        split for split in requested
        if split == SELECTION_SPLIT or all(_has_output(run, split) for run in with_val)
    )
    dropped = [split for split in requested if split not in available]
    if dropped:
        print(f"[ensemble] splits sem outputs para todos os runs, ignorados: {', '.join(dropped)}")
    if len(available) < 2:
        raise ValueError(
            "É preciso pelo menos um split held-out além do val. "
            "Rode evaluate.py para test e/ou test_d."
        )
    return available


def _aggregate_runs(runs: list[TrainedRun], splits: tuple[str, ...]) -> Candidate:
    arrays = {}
    for split in splits:
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


def load_candidates(root: str | Path, pool: str,
                    splits: tuple[str, ...] = DEFAULT_SPLITS) -> list[Candidate]:
    runs = discover_trained_runs(root)
    effective = resolve_splits(runs, splits)
    grouped: dict[tuple[str, str, str], list[TrainedRun]] = {}
    for run in runs:
        if all(_has_output(run, split) for split in effective):
            grouped.setdefault((run.model_family, run.fourier_mode, run.regime), []).append(run)
    candidates = [_aggregate_runs(group, effective) for group in grouped.values()]
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
    # "spawn" explícito: no Linux o default é fork, e forkar um processo que já
    # tem threads do torch/BLAS pode travar o filho (CPython avisa desde 3.12).
    # run_tasks_on_gpus já usa spawn pelo mesmo motivo.
    with ProcessPoolExecutor(max_workers=max_workers, mp_context=mp.get_context("spawn"),
                             initializer=_init_search_worker,
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


def run(root, pool="best-mode", strategy="search", output_dir=None, max_workers=None,
        splits=DEFAULT_SPLITS, subset="search"):
    # `--strategy search` é apelido histórico do combinador `mean` (e --subset já
    # é "search" por padrão), para não quebrar os comandos do README.
    if strategy == "search":
        strategy = "mean"
    candidates = load_candidates(root, pool, splits)
    if not candidates:
        raise ValueError(
            f"Nenhum candidato com outputs alinhados em {', '.join(splits)} sob {root}"
        )
    report_splits = tuple(candidates[0].arrays)
    val_predictions = [candidate.arrays[SELECTION_SPLIT]["probs"] for candidate in candidates]
    val_labels = candidates[0].arrays[SELECTION_SPLIT]["y_true"]
    if subset == "search":
        # Exaustivo enumera 2^N-1 subconjuntos: viável só para pools pequenos,
        # independente de --pool (que agora só define o tamanho do pool).
        exhaustive = len(candidates) <= EXHAUSTIVE_MAX_CANDIDATES
        print(f"[ensemble] busca {'exaustiva' if exhaustive else 'greedy'} "
              f"sobre {len(candidates)} candidatos")
        selected = search_subset(val_predictions, val_labels, exhaustive, max_workers)
    else:
        selected = tuple(range(len(candidates)))
    selected_candidates = [candidates[index] for index in selected]
    combine = strategy
    val_auc_weights = [candidate.val_auc for candidate in selected_candidates]
    output = Path(output_dir or output_root() / "results" / "ensemble")
    output.mkdir(parents=True, exist_ok=True)
    report = []
    for split in report_splits:
        arrays = [candidate.arrays[split] for candidate in selected_candidates]
        reference = arrays[0]
        if any(not np.array_equal(reference["ids"], value["ids"]) for value in arrays[1:]):
            raise ValueError("Candidate prediction IDs are not aligned")
        split_predictions = [value["probs"] for value in arrays]
        if combine == "stacking":
            probabilities = STRATEGIES[combine](
                [candidate.arrays[SELECTION_SPLIT]["probs"] for candidate in selected_candidates],
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
            "split": split, "strategy": combine, "subset": subset, "pool": pool,
            "members": len(selected_candidates),
            "member_names": ";".join(candidate.name for candidate in selected_candidates),
            "auc": safe_auc(reference["y_true"], probabilities),
        })
    frame = pd.DataFrame(report)
    frame.to_csv(output / "ensemble_report.csv", index=False)
    return frame


def main(argv=None):
    parser = argparse.ArgumentParser(description="Select ensembles on val and report held-out splits")
    parser.add_argument("--strategy", default="search", choices=tuple(STRATEGIES) + ("search",),
                        help="combinador das probabilidades ('search' = apelido de mean + --subset search)")
    parser.add_argument("--subset", default="search", choices=("search", "all"),
                        help="buscar o melhor subconjunto no val ou usar o pool inteiro")
    parser.add_argument("--splits", default=",".join(DEFAULT_SPLITS),
                        help="splits desejados; os held-out sem outputs são ignorados")
    parser.add_argument("--pool", default="best-mode", choices=("best-mode", "all"))
    parser.add_argument("--models-root", default=None)
    parser.add_argument("--output-dir", default=None)
    parser.add_argument("--max-workers", type=int, default=None)
    args = parser.parse_args(argv)
    splits = tuple(part.strip() for part in args.splits.split(",") if part.strip())
    print(run(args.models_root or models_root(), args.pool, args.strategy,
              args.output_dir, args.max_workers, splits, args.subset).to_string(index=False))


if __name__ == "__main__":
    main()
