from __future__ import annotations

from dataclasses import asdict, dataclass, fields
from pathlib import Path
from typing import Any

import yaml

FOURIER_CHANNELS = {"none": 3, "magnitude": 1, "phase": 1, "complex": 2,
                    "concat": 4, "frequency_3": 1, "concat_frequency": 6}


@dataclass
class TrainingConfig:
    model_family: str = "resnet"
    architecture: str = "resnet18"
    fourier_mode: str = "none"
    regime: str = "scratch"
    seed: int = 42
    seeds: tuple[int, ...] = (42, 123, 2024)
    epochs: int = 50
    batch_size: int = 32
    num_workers: int = 4
    image_size: int = 224
    data_limit: int | None = None
    lr_head: float = 1e-3
    lr_backbone: float = 1e-4
    weight_decay: float = 1e-4
    early_stop_patience: int = 8
    dropout: float = 0.2
    augment: bool = True
    threshold_strategy: str = "accuracy"
    mixup_alpha: float = 0.0
    cutmix_alpha: float = 0.0
    unfreeze_last_n: int = 2
    multi_gpu: bool = True
    allow_pretrained: bool = True
    patch_size: int = 16
    hidden_size: int = 256
    num_hidden_layers: int = 6
    num_attention_heads: int = 8
    projection_dim: int = 128
    variant: str = "large"
    dino_version: str = "v3"
    model_size: str = "base"

    @property
    def in_channels(self) -> int:
        return FOURIER_CHANNELS[self.fourier_mode]

    def to_dict(self) -> dict[str, Any]:
        result = asdict(self)
        result["seeds"] = list(self.seeds)
        result["in_channels"] = self.in_channels
        return result

    def validate(self) -> None:
        if self.fourier_mode not in FOURIER_CHANNELS:
            raise ValueError(f"Unknown Fourier mode: {self.fourier_mode}")
        if self.regime not in {"scratch", "finetune"}:
            raise ValueError("regime must be scratch or finetune")


def _read_yaml(path: Path) -> dict[str, Any]:
    with path.open(encoding="utf-8") as handle:
        return yaml.safe_load(handle) or {}


def load_config(path: str | Path, overrides: dict[str, Any] | None = None) -> TrainingConfig:
    path = Path(path)
    base = path.parent / "base.yaml"
    values = _read_yaml(base) if base.exists() and base.resolve() != path.resolve() else {}
    values.update(_read_yaml(path))
    values.update({k: v for k, v in (overrides or {}).items() if v is not None})
    allowed = {field.name for field in fields(TrainingConfig)}
    unknown = set(values) - allowed
    if unknown:
        raise ValueError(f"Unknown config fields: {', '.join(sorted(unknown))}")
    if "seeds" in values:
        values["seeds"] = tuple(int(seed) for seed in values["seeds"])
    config = TrainingConfig(**values)
    config.validate()
    return config
