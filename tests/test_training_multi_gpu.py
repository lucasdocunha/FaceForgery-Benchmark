import inspect
import json

import torch

from src.pipelines.checkpoints import discover_trained_runs, load_model_from_run
from src.pipelines.config import RUN_CONFIG_FILENAME, TrainingConfig
from src.pipelines.training import Trainer, maybe_data_parallel, model_state_dict, unwrap_model


def test_generic_training_exposes_multi_gpu_support():
    """multi_gpu é decidido pela config, não por um parâmetro do Trainer."""
    assert "multi_gpu" not in inspect.signature(Trainer).parameters
    assert "enabled" in inspect.signature(maybe_data_parallel).parameters


def test_unwrap_model_returns_original_module_for_plain_model():
    model = torch.nn.Linear(2, 2)
    assert unwrap_model(model) is model


def test_maybe_data_parallel_disabled_returns_original_model():
    model = torch.nn.Linear(2, 2)
    assert maybe_data_parallel(model, torch.device("cpu"), enabled=False) is model


def test_model_state_dict_removes_data_parallel_prefixes():
    """Checkpoints salvos em multi-GPU precisam ser recarregáveis em single-GPU/CPU.

    Sem isso todo peso sai com o prefixo `module.` e o load_state_dict do
    evaluate/heatmap falha nos servidores multi-GPU.
    """
    model = torch.nn.DataParallel(torch.nn.Linear(2, 2))
    keys = model_state_dict(model).keys()
    assert keys
    assert all(not key.startswith("module.") for key in keys)


def test_load_model_from_run_accepts_a_data_parallel_checkpoint(tmp_path):
    """Contrapartida na leitura: pesos com prefixo `module.` ainda carregam.

    Cobre checkpoints gravados por versões/caminhos que não passaram por
    model_state_dict, para que o prefixo não vire um erro de state_dict.
    """
    config = TrainingConfig(model_family="resnet", architecture="resnet18", image_size=32)
    run_dir = tmp_path / "resnet" / "none" / "scratch" / "seed_42"
    (run_dir / "weights").mkdir(parents=True)
    (run_dir / "results").mkdir()
    (run_dir / "results" / RUN_CONFIG_FILENAME).write_text(
        json.dumps(config.to_dict()), encoding="utf-8",
    )

    from src.models.registry import get_model_spec
    plain = get_model_spec("resnet").build(config)
    wrapped = {f"module.{key}": value for key, value in plain.state_dict().items()}
    torch.save(wrapped, run_dir / "weights" / "best.pth")

    loaded = load_model_from_run(discover_trained_runs(tmp_path)[0])
    with torch.no_grad():
        assert loaded(torch.rand(1, 3, 32, 32)).shape == (1, 2)
