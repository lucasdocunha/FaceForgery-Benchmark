import numpy as np

from ensemble import greedy_subset_search, load_candidates, run, search_subset
from src.pipelines.ensemble_strategies import (
    geometric_mean, majority_vote, max_prob, mean, stacking_logreg, weighted_by_val_auc,
)


def _mock_run(root, family, mode, regime, seed, offset):
    run_dir = root / family / mode / regime / f"seed_{seed}"
    (run_dir / "weights").mkdir(parents=True)
    (run_dir / "weights" / "best.pth").touch()
    (run_dir / "results").mkdir()
    labels = np.array([0, 1, 0, 1])
    ids = np.arange(4)
    base = np.clip(np.array([.1, .9, .2, .8]) + offset, 0, 1)
    for split in ("val", "test", "test_d"):
        np.savez_compressed(run_dir / "results" / f"outputs_{split}.npz",
                            ids=ids, y_true=labels, probs=base)


def test_all_pure_strategies_and_searches():
    predictions = np.array([[.1, .8, .2, .9], [.2, .7, .4, .8]])
    labels = np.array([0, 1, 0, 1])
    for result in (
        mean(predictions), weighted_by_val_auc(predictions, [.8, .9]), majority_vote(predictions),
        max_prob(predictions), geometric_mean(predictions), stacking_logreg(predictions, labels),
    ):
        assert result.shape == (4,) and np.isfinite(result).all()
    assert search_subset(predictions, labels, exhaustive=True, max_workers=1)
    assert greedy_subset_search(predictions, labels)


def test_pooling_aggregates_seeds_and_run_writes_all_splits(tmp_path):
    root = tmp_path / "models"
    for seed, offset in ((42, 0), (123, .01)):
        _mock_run(root, "resnet", "none", "scratch", seed, offset)
        _mock_run(root, "resnet", "phase", "scratch", seed, .2 + offset)
        _mock_run(root, "resnet", "none", "finetune", seed, .02 + offset)
    assert len(load_candidates(root, "all")) == 3
    best = load_candidates(root, "best-mode")
    assert len(best) == 2 and all(candidate.seeds == (42, 123) for candidate in best)
    report = run(root, "best-mode", "search", tmp_path / "ensemble", max_workers=1)
    assert set(report["split"]) == {"val", "test", "test_d"}
    assert (tmp_path / "ensemble" / "ensemble_report.csv").exists()
