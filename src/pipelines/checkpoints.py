from __future__ import annotations

import ast
import json
from dataclasses import dataclass, fields
from pathlib import Path
from typing import Iterable

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torchvision import transforms

from src.data.data import ALL_FOURIER_MODES, ImageDataset
from src.data.paths import phase1_split_root
from src.models.registry import MODEL_REGISTRY
from src.pipelines.config import RUN_CONFIG_FILENAME, TrainingConfig
from src.pipelines.evaluation import evaluate_classifier

SUPPORTED_FAMILIES = frozenset(MODEL_REGISTRY)


@dataclass(frozen=True)
class TrainedRun:
    model_family: str
    fourier_mode: str
    regime: str
    seed: int
    run_dir: Path
    weights_path: Path
    metadata: dict
    threshold: float = .5


@dataclass(frozen=True)
class SplitSpec:
    name: str
    csv_path: Path
    images_dir: Path


def _read_metadata(run_dir: Path) -> dict:
    """Métricas do run, usadas só para recuperar o threshold escolhido no val.

    Não serve para reconstruir a config: ``evaluate.py`` reescreve estes arquivos.
    Para isso existe ``_read_run_config``.
    """
    for name in ("metrics_val.csv", "metrics_test.csv", "metrics_summary.csv"):
        path = run_dir / "results" / name
        if path.exists():
            frame = pd.read_csv(path)
            return frame.iloc[0].to_dict() if not frame.empty else {}
    return {}


def discover_trained_runs(root: str | Path, only_model_family: str | None = None) -> list[TrainedRun]:
    root = Path(root)
    runs = []
    for weights in sorted(root.glob("*/*/*/seed_*/weights/best.pth")):
        family, mode, regime, seed_part = weights.relative_to(root).parts[:4]
        if family not in SUPPORTED_FAMILIES or mode not in ALL_FOURIER_MODES or regime not in {"scratch", "finetune"}:
            continue
        if only_model_family and family != only_model_family:
            continue
        try:
            seed = int(seed_part.removeprefix("seed_"))
        except ValueError:
            continue
        run_dir = weights.parent.parent
        metadata = _read_metadata(run_dir)
        threshold = float(np.clip(float(metadata.get("threshold", .5)), 0, 1))
        runs.append(TrainedRun(family, mode, regime, seed, run_dir, weights, metadata, threshold))
    mode_order = {mode: index for index, mode in enumerate(ALL_FOURIER_MODES)}
    return sorted(runs, key=lambda run: (run.model_family, run.regime, mode_order[run.fourier_mode], run.seed))


def run_from_checkpoint(checkpoint: str | Path) -> TrainedRun:
    checkpoint = Path(checkpoint).resolve()
    if checkpoint.name != "best.pth" or checkpoint.parent.name != "weights":
        raise ValueError("Checkpoint must be a new-layout weights/best.pth file")
    run_dir = checkpoint.parent.parent
    seed_part, regime, mode, family = run_dir.name, run_dir.parent.name, run_dir.parent.parent.name, run_dir.parent.parent.parent.name
    if family not in SUPPORTED_FAMILIES or mode not in ALL_FOURIER_MODES or regime not in {"scratch", "finetune"} or not seed_part.startswith("seed_"):
        raise ValueError("Checkpoint path does not match family/mode/regime/seed_N/weights/best.pth")
    metadata = _read_metadata(run_dir)
    return TrainedRun(family, mode, regime, int(seed_part[5:]), run_dir, checkpoint,
                      metadata, float(metadata.get("threshold", .5)))


def _coerce(field_name: str, value):
    if field_name == "seeds":
        if isinstance(value, str):
            value = ast.literal_eval(value)
        return tuple(int(seed) for seed in value)
    field = TrainingConfig.__dataclass_fields__[field_name]
    if field.type in (bool, "bool") and isinstance(value, str):
        return value.strip().lower() in {"1", "true", "yes"}
    return value


def _read_run_config(run: TrainedRun) -> dict:
    """Lê a config gravada por ``Trainer.fit`` em ``results/run_config.json``.

    Falha alto em vez de cair nos defaults de ``TrainingConfig``: reconstruir um
    checkpoint com a arquitetura errada só aparece bem depois, como state_dict
    incompatível ou heatmap renderizado na resolução errada.
    """
    path = run.run_dir / "results" / RUN_CONFIG_FILENAME
    if not path.exists():
        raise FileNotFoundError(
            f"Run sem {RUN_CONFIG_FILENAME}: {run.run_dir}. Ele é gravado por Trainer.fit(); "
            "runs anteriores a esse artefato precisam ser retreinados."
        )
    with path.open(encoding="utf-8") as handle:
        return json.load(handle)


def config_from_run(run: TrainedRun) -> TrainingConfig:
    allowed = {field.name for field in fields(TrainingConfig)}
    values = {
        key: _coerce(key, value) for key, value in _read_run_config(run).items()
        if key in allowed and not (isinstance(value, float) and np.isnan(value))
    }
    values.update(model_family=run.model_family, fourier_mode=run.fourier_mode,
                  regime=run.regime, seed=run.seed)
    config = TrainingConfig(**values)
    config.validate()
    return config


def build_model_from_run(run: TrainedRun) -> nn.Module:
    return MODEL_REGISTRY[run.model_family].build(config_from_run(run))


def load_model_from_run(run: TrainedRun, device: torch.device | str = "cpu") -> nn.Module:
    device = torch.device(device)
    model = build_model_from_run(run)
    state = torch.load(run.weights_path, map_location=device, weights_only=True)
    if state and all(str(key).startswith("module.") for key in state):
        state = {str(key).removeprefix("module."): value for key, value in state.items()}
    model.load_state_dict(state)
    return model.to(device).eval()


def parse_splits(value: str | Iterable[str]) -> tuple[str, ...]:
    parts = value.split(",") if isinstance(value, str) else value
    result = tuple(str(part).strip() for part in parts if str(part).strip())
    if not result:
        raise ValueError("At least one split is required")
    return result


def build_split_specs(data_dir, splits, test_d_csv=None, test_d_images_dir=None) -> list[SplitSpec]:
    specs = []
    for split in parse_splits(splits):
        if split in {"val", "test"}:
            specs.append(SplitSpec(split, Path(data_dir) / f"{split}.csv", phase1_split_root(split)))
        elif split == "test_d" and test_d_csv and test_d_images_dir:
            specs.append(SplitSpec(split, Path(test_d_csv), Path(test_d_images_dir)))
        elif split == "test_d":
            raise ValueError("test_d requires --test-d-csv and --test-d-images-dir")
        else:
            raise ValueError(f"Unknown split: {split}")
    return specs


def _transform(config: TrainingConfig):
    return transforms.Compose([
        transforms.Resize((config.image_size, config.image_size)), transforms.ToTensor(),
        transforms.Normalize([.485, .456, .406], [.229, .224, .225]),
    ])


def _save_results(run: TrainedRun, split: str, metrics: dict) -> dict:
    result_dir = run.run_dir / "results"
    result_dir.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        result_dir / f"outputs_{split}.npz", logits=metrics["logits"], probs=metrics["probs"],
        y_true=metrics["y_true"], y_pred=metrics["y_pred"], ids=metrics["ids"], threshold=run.threshold,
    )
    row = {
        "model_family": run.model_family, "fourier_mode": run.fourier_mode,
        "regime": run.regime, "seed": run.seed, "split": split, "threshold": run.threshold,
        **{key: metrics[key] for key in ("loss", "acc", "precision", "recall", "f1", "auc", "specificity", "tp", "fp", "fn", "tn")},
    }
    pd.DataFrame([row]).to_csv(result_dir / f"metrics_{split}.csv", index=False)
    pd.DataFrame({
        "id": metrics["ids"], "y_true": metrics["y_true"], "y_pred": metrics["y_pred"],
        "prob_pos": metrics["probs"],
    }).to_csv(result_dir / f"predictions_{split}.csv", index=False)
    return row


def evaluate_trained_runs(models_root, data_dir, splits=("val", "test", "test_d"),
                          test_d_csv=None, test_d_images_dir=None, output_csv=None,
                          batch_size=32, num_workers=0, device=None,
                          only_model_family=None, limit_per_split=None) -> pd.DataFrame:
    device = torch.device(device or ("cuda" if torch.cuda.is_available() else "cpu"))
    specs = build_split_specs(data_dir, splits, test_d_csv, test_d_images_dir)
    rows = []
    for run in discover_trained_runs(models_root, only_model_family):
        model, config = load_model_from_run(run, device), config_from_run(run)
        for split in specs:
            dataset = ImageDataset(
                split.csv_path, split.images_dir, transform=_transform(config),
                data_limit=np.inf if limit_per_split is None else limit_per_split,
                fourier=run.fourier_mode, spatial_size=(config.image_size, config.image_size),
            )
            loader = DataLoader(dataset, batch_size=batch_size, shuffle=False, num_workers=num_workers,
                                pin_memory=device.type == "cuda", persistent_workers=num_workers > 0)
            metrics = evaluate_classifier(model, loader, nn.CrossEntropyLoss(), device,
                                          threshold=run.threshold, use_amp=device.type == "cuda",
                                          desc=f"{run.model_family}/{run.fourier_mode}/{split.name}")
            rows.append(_save_results(run, split.name, metrics))
    frame = pd.DataFrame(rows)
    output = Path(output_csv or Path(models_root) / "all_metrics_by_split.csv")
    output.parent.mkdir(parents=True, exist_ok=True)
    frame.to_csv(output, index=False)
    return frame
