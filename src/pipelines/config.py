from __future__ import annotations

from dataclasses import asdict, dataclass, fields
from pathlib import Path
from typing import Any

import yaml

FOURIER_CHANNELS = {
    "none": 3, "magnitude": 1, "phase": 1, "complex": 2,
    "concat": 4, "frequency_3": 1, "concat_frequency": 6,
}

# Artefato gravado por Trainer.fit() em <run_dir>/results/ e lido por
# checkpoints.config_from_run(): fonte única da config usada para reconstruir o modelo.
RUN_CONFIG_FILENAME = "run_config.json"


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
    raw_min: bool = False
    lr_head: float = 1e-3
    lr_backbone: float = 1e-4
    weight_decay: float = 1e-4
    early_stop_patience: int = 8
    scheduler_patience: int = 3
    max_grad_norm: float | None = 1.0
    dropout: float = 0.2
    augment: bool = True
    train_backbone: bool = True
    use_weighted_sampler: bool = True
    use_class_weights: bool = False
    label_smoothing: float = 0.0
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
    model_size: str = "base"

    @property
    def in_channels(self) -> int:
        return FOURIER_CHANNELS[self.fourier_mode]

    @property
    def data_split_dir(self) -> str:
        return "raw_min" if self.raw_min else "raw"

    def to_dict(self) -> dict[str, Any]:
        result = asdict(self)
        result["seeds"] = list(self.seeds)
        result["in_channels"] = self.in_channels
        return result

    def validate(self) -> None:
        if self.model_family not in {"resnet", "xception", "mobilenet", "vit", "clip", "dino", "hybrid"}:
            raise ValueError(f"Unknown model family: {self.model_family}")
        if self.fourier_mode not in FOURIER_CHANNELS:
            raise ValueError(f"Unknown Fourier mode: {self.fourier_mode}")
        if self.regime not in {"scratch", "finetune"}:
            raise ValueError("regime must be scratch or finetune")
        if self.epochs < 1 or self.batch_size < 1 or self.num_workers < 0:
            raise ValueError("epochs/batch_size must be positive and num_workers non-negative")
        if self.early_stop_patience < 1:
            raise ValueError("early_stop_patience must be positive")
        if not 0 <= self.label_smoothing < 1:
            raise ValueError("label_smoothing must be in [0, 1)")
        if self.image_size % self.patch_size != 0 and self.model_family in {"vit", "clip"}:
            raise ValueError("image_size must be divisible by patch_size")


def _read_yaml(path: Path) -> dict[str, Any]:
    with path.open(encoding="utf-8") as handle:
        value = yaml.safe_load(handle) or {}
    if not isinstance(value, dict):
        raise ValueError(f"YAML root must be a mapping: {path}")
    return value


def load_config(path: str | Path, overrides: dict[str, Any] | None = None) -> TrainingConfig:
    path = Path(path)
    base = path.parent / "base.yaml"
    values = _read_yaml(base) if base.exists() and base.resolve() != path.resolve() else {}
    values.update(_read_yaml(path))
    values.update({key: value for key, value in (overrides or {}).items() if value is not None})
    allowed = {field.name for field in fields(TrainingConfig)}
    unknown = set(values) - allowed
    if unknown:
        raise ValueError(f"Unknown config fields: {', '.join(sorted(unknown))}")
    if "seeds" in values:
        values["seeds"] = tuple(int(seed) for seed in values["seeds"])
    config = TrainingConfig(**values)
    config.validate()
    return config
