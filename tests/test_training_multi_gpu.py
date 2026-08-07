import inspect
from src.pipelines.training import Trainer, maybe_data_parallel
def test_generic_training_exposes_multi_gpu_support():
    assert "multi_gpu" not in inspect.signature(Trainer).parameters
    assert "enabled" in inspect.signature(maybe_data_parallel).parameters
