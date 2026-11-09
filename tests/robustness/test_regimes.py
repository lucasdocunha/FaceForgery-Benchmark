from types import SimpleNamespace
import json
import pytest
from src.models._regime import uses_pretraining


@pytest.mark.parametrize(
    "family", ["resnet", "mobilenet", "xception", "dino", "clip", "vit"]
)
@pytest.mark.parametrize(
    "regime,expected",
    [
        ("scratch", False),
        ("scratch_robust", False),
        ("finetune", True),
        ("finetune_robust", True),
    ],
)
def test_robust_suffix_does_not_change_new_initialization(family, regime, expected):
    assert (
        uses_pretraining(
            SimpleNamespace(regime=regime, initialization_contract="regime-v2"), family
        )
        is expected
    )


@pytest.mark.parametrize(
    "family", ["resnet", "mobilenet", "xception", "dino", "clip", "vit"]
)
def test_unversioned_historical_constructor_is_not_reinterpreted(family):
    cfg = SimpleNamespace(regime="finetune_robust", initialization_contract="legacy-v1")
    assert uses_pretraining(cfg, family) is (family in {"clip", "vit"})


def test_unknown_regime_rejected():
    with pytest.raises(ValueError):
        uses_pretraining(SimpleNamespace(regime="fine_tunish"), "resnet")


def test_saved_config_preserves_legacy_build_identity(tmp_path):
    from src.pipelines.checkpoints import TrainedRun, config_from_run

    root = tmp_path / "run"
    (root / "results").mkdir(parents=True)
    (root / "results/run_config.json").write_text(
        json.dumps(
            {
                "model_family": "xception",
                "regime": "finetune_robust",
                "fourier_mode": "none",
            }
        )
    )
    run = TrainedRun(
        "xception", "none", "finetune_robust", 42, root, root / "best.pth", {}
    )
    cfg = config_from_run(run)
    assert cfg.initialization_contract == "legacy-v1"
    assert not uses_pretraining(cfg, "xception")


def test_current_config_marks_new_contract():
    from src.pipelines.config import TrainingConfig

    cfg = TrainingConfig(regime="finetune_robust")
    assert cfg.to_dict()["initialization_contract"] == "regime-v2"
