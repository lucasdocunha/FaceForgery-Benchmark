import json
from pathlib import Path

import pytest
import torch

from src.models.registry import get_model_spec
from src.pipelines.checkpoints import (
    config_from_run, discover_trained_runs, evaluate_trained_runs, run_from_checkpoint,
)
from src.pipelines.config import RUN_CONFIG_FILENAME, TrainingConfig


def test_discovers_all_six_families_with_regime_and_seed(tmp_path):
    for family in ("resnet","xception","mobilenet","vit","clip","dino"):
        path=tmp_path/family/"none"/"scratch"/"seed_42"/"weights";path.mkdir(parents=True);(path/"best.pth").touch()
    runs=discover_trained_runs(tmp_path)
    assert {r.model_family for r in runs}=={"resnet","xception","mobilenet","vit","clip","dino"}
    assert all(r.seed==42 and r.regime=="scratch" for r in runs)
    assert run_from_checkpoint(runs[0].weights_path).model_family in {r.model_family for r in runs}


def _seed_run(models_root: Path, config: TrainingConfig) -> Path:
    """Grava um run no layout novo com pesos reais e run_config.json, sem treinar."""
    run_dir = (models_root / config.model_family / config.fourier_mode /
               config.regime / f"seed_{config.seed}")
    (run_dir / "weights").mkdir(parents=True)
    (run_dir / "results").mkdir()
    model = get_model_spec(config.model_family).build(config)
    torch.save(model.state_dict(), run_dir / "weights" / "best.pth")
    (run_dir / "results" / RUN_CONFIG_FILENAME).write_text(
        json.dumps(config.to_dict()), encoding="utf-8",
    )
    return run_dir


def test_run_config_survives_repeated_evaluation(tiny_phase1_dataset, tmp_path, monkeypatch):
    """Reavaliar não pode destruir a config de que a reconstrução do modelo depende.

    ``evaluate.py`` reescreve ``metrics_{split}.csv``; quando a config morava lá,
    a primeira reavaliação a apagava e a segunda quebrava com state_dict
    incompatível (ou, pior, reconstruía silenciosamente outra arquitetura).
    Valores não-default de propósito: com os defaults o bug fica invisível.
    """
    repo = Path(__file__).resolve().parents[1]
    monkeypatch.setenv("TCC_DATA_ROOT", str(repo / "data"))
    models = tmp_path / "models"
    config = TrainingConfig(model_family="resnet", architecture="resnet34",
                            image_size=32, dropout=.4, raw_min=True, seed=7)
    _seed_run(models, config)

    for _ in range(2):
        summary = evaluate_trained_runs(
            models, repo / "data" / "raw_min", splits=("val", "test"),
            batch_size=2, num_workers=0, limit_per_split=4,
        )
        assert len(summary) == 2
        reloaded = config_from_run(discover_trained_runs(models)[0])
        assert (reloaded.architecture, reloaded.image_size, reloaded.dropout) == \
               ("resnet34", 32, .4)


def test_missing_run_config_raises_instead_of_defaulting(tmp_path):
    """Sem run_config.json, falhar alto é melhor que reconstruir a arquitetura errada."""
    path = tmp_path / "resnet" / "none" / "scratch" / "seed_42" / "weights"
    path.mkdir(parents=True)
    (path / "best.pth").touch()
    run = discover_trained_runs(tmp_path)[0]
    with pytest.raises(FileNotFoundError, match=RUN_CONFIG_FILENAME):
        config_from_run(run)
