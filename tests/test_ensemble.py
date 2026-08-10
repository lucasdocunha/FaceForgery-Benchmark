import numpy as np
import pytest

from ensemble import greedy_subset_search, load_candidates, run, search_subset
from src.pipelines.ensemble_strategies import (
    geometric_mean, majority_vote, max_prob, mean, stacking_logreg, weighted_by_val_auc,
)


def _mock_run(root, family, mode, regime, seed, offset, splits=("val", "test", "test_d")):
    run_dir = root / family / mode / regime / f"seed_{seed}"
    (run_dir / "weights").mkdir(parents=True)
    (run_dir / "weights" / "best.pth").touch()
    (run_dir / "results").mkdir()
    labels = np.array([0, 1, 0, 1])
    ids = np.arange(4)
    base = np.clip(np.array([.1, .9, .2, .8]) + offset, 0, 1)
    for split in splits:
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


def test_runs_on_val_and_test_only_without_test_d(tmp_path):
    """O fluxo documentado (train + evaluate --splits val,test) não produz test_d.

    Exigir test_d fazia `python ensemble.py` abortar com "No candidates"; agora o
    split ausente é só ignorado no relatório.
    """
    root = tmp_path / "models"
    for seed, offset in ((42, 0), (123, .01)):
        _mock_run(root, "resnet", "none", "scratch", seed, offset, splits=("val", "test"))
        _mock_run(root, "resnet", "phase", "scratch", seed, .2 + offset, splits=("val", "test"))
    report = run(root, "best-mode", "search", tmp_path / "ensemble", max_workers=1)
    assert set(report["split"]) == {"val", "test"}
    assert (tmp_path / "ensemble" / "ensemble_predictions_test.csv").exists()
    assert not (tmp_path / "ensemble" / "ensemble_predictions_test_d.csv").exists()


def test_val_only_tree_is_rejected(tmp_path):
    """Sem nenhum held-out não há nada para relatar: falhar é melhor que reportar val."""
    root = tmp_path / "models"
    _mock_run(root, "resnet", "none", "scratch", 42, 0, splits=("val",))
    with pytest.raises(ValueError, match="held-out"):
        run(root, "all", "mean", tmp_path / "ensemble", max_workers=1)


def test_subset_search_composes_with_a_non_mean_combiner(tmp_path):
    """--strategy weighted --subset search: antes 'search' forçava o combinador a mean."""
    root = tmp_path / "models"
    for mode, offset in (("none", 0), ("phase", .2), ("magnitude", .05)):
        _mock_run(root, "resnet", mode, "scratch", 42, offset, splits=("val", "test"))
    report = run(root, "all", "weighted", tmp_path / "ensemble", max_workers=1, subset="search")
    assert set(report["strategy"]) == {"weighted"}
    assert set(report["subset"]) == {"search"}
