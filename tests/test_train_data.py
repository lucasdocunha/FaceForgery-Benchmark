from pathlib import Path

from torch.utils.data import WeightedRandomSampler

from src.pipelines.config import TrainingConfig
from train import _transform, build_loaders


def test_spectral_modes_disable_spatial_augmentation():
    config = TrainingConfig(fourier_mode="phase", augment=True, image_size=32)
    names = {type(operation).__name__ for operation in _transform(config, True).transforms}
    assert "RandomHorizontalFlip" not in names and "ColorJitter" not in names


def test_training_uses_raw_min_and_balanced_sampler(tiny_phase1_dataset, monkeypatch):
    repo = Path(__file__).resolve().parents[1]
    monkeypatch.setenv("TCC_DATA_ROOT", str(repo / "data"))
    config = TrainingConfig(raw_min=True, image_size=32, data_limit=8, batch_size=4, num_workers=0)
    loaders = build_loaders(config)
    assert isinstance(loaders["train"].sampler, WeightedRandomSampler)
    assert len(loaders["train"].dataset) == 8
