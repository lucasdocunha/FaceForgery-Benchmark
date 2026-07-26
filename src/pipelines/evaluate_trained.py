from __future__ import annotations

import argparse
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, Sequence

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torchvision import transforms

from src.data import ALL_FOURIER_MODES, ImageDataset
from src.data.paths import phase1_split_root
from src.models.clip import CLIPVisionClassifier, CLIPVisionClassifierWithBackbone
from src.models.dino import DINOVisionClassifier
from src.models.mobilenet import mobilenet
from src.models.resnet import resnet
from src.models.vit import VisionTransformerClassifier
from src.models.xception import xception
from src.pipelines.evaluation import evaluate_classifier

SUPPORTED_FAMILIES = {"vit"}
FOURIER_MODES = set(ALL_FOURIER_MODES[:-1])
FOURIER_CHANNELS = {
    "none": 3,
    "magnitude": 1,
    "phase": 1,
    "complex": 2,
    "concat": 4,
    "frequency_3": 1,
    #"concat_frequency": 6,
}


@dataclass(frozen=True)
class TrainedRun:
    model_family: str
    architecture: str
    technique: str
    run_dir: Path
    weights_path: Path
    metrics_path: Path | None
    metadata: dict
    threshold: float
    threshold_source: str


@dataclass(frozen=True)
class SplitSpec:
    name: str
    csv_path: Path
    images_dir: Path


def _as_path(path: str | Path) -> Path:
    return Path(path).expanduser()


def _is_missing(value) -> bool:
    try:
        return pd.isna(value)
    except TypeError:
        return False


def _metadata_value(metadata: dict, key: str, default=None):
    value = metadata.get(key, default)
    if _is_missing(value):
        return default
    return value


def _metadata_int(metadata: dict, key: str, default: int) -> int:
    return int(_metadata_value(metadata, key, default))


def _metadata_float(metadata: dict, key: str, default: float) -> float:
    return float(_metadata_value(metadata, key, default))


def _metadata_bool(metadata: dict, key: str, default: bool = False) -> bool:
    value = _metadata_value(metadata, key, default)
    if isinstance(value, str):
        return value.strip().lower() in {"1", "true", "yes", "y"}
    return bool(value)


def _read_metrics(metrics_path: Path) -> dict:
    if not metrics_path.exists():
        return {}
    metrics = pd.read_csv(metrics_path)
    if metrics.empty:
        return {}
    return metrics.iloc[0].to_dict()


def _normalize_technique(value: str | None) -> str | None:
    if value is None:
        return None
    raw = str(value)
    for mode in ALL_FOURIER_MODES:
        if raw == mode or raw.startswith(f"{mode}_limit"):
            return mode
    return raw if raw in FOURIER_MODES else None


def _architecture_from_path(family: str, parts: Sequence[str], weights_path: Path) -> str:
    if family == "mobilenet" and len(parts) >= 2:
        return parts[1]
    if family in {"vit", "clip", "dino"} and len(parts) >= 2:
        return parts[1]
    if family == "xception":
        return "xception"
    if family == "resnet":
        return "resnet"
    return weights_path.stem.removeprefix("best_")


def _technique_from_path(family: str, parts: Sequence[str]) -> str:
    if family in {"mobilenet", "vit", "clip", "dino"} and len(parts) >= 3:
        return _normalize_technique(parts[2]) or "none"
    if family in {"resnet", "xception"} and len(parts) >= 2:
        return _normalize_technique(parts[1]) or "none"
    return "none"


def _parse_dino_arch(architecture: str) -> tuple[str, str]:
    arch = architecture.lower()
    version = "v3" if "v3" in arch else "v2"
    for size in ("tiny", "small", "large", "base"):
        if size in arch:
            return version, size
    return version, "base"


def _technique_from_metadata(metadata: dict) -> str | None:
    for key in ("fourier", "input_mode", "technique"):
        technique = _normalize_technique(_metadata_value(metadata, key))
        if technique is not None:
            return technique
    return None


def _threshold_from_metadata(metadata: dict) -> tuple[float, str]:
    value = _metadata_value(metadata, "threshold")
    if value is None:
        return 0.5, "default_0.5"
    return float(np.clip(float(value), 0.0, 1.0)), "metrics_summary"


def discover_trained_runs(
    models_root: str | Path,
    only_model_family: str | None = None,
) -> list[TrainedRun]:
    models_root = _as_path(models_root)
    runs: list[TrainedRun] = []

    for weights_path in sorted(models_root.glob("**/weights/best_*.pth")):
        run_dir = weights_path.parent.parent
        try:
            parts = run_dir.relative_to(models_root).parts
        except ValueError:
            continue
        if not parts:
            continue

        family = parts[0]
        if family not in SUPPORTED_FAMILIES:
            continue
        if only_model_family is not None and family != only_model_family:
            continue

        metrics_path = run_dir / "results" / "metrics_summary.csv"
        metadata = _read_metrics(metrics_path)
        architecture = str(
            _metadata_value(
                metadata,
                "architecture",
                _metadata_value(metadata, "model", _architecture_from_path(family, parts, weights_path)),
            )
        )
        technique = _technique_from_metadata(metadata) or _technique_from_path(family, parts)
        threshold, threshold_source = _threshold_from_metadata(metadata)
        runs.append(
            TrainedRun(
                model_family=family,
                architecture=architecture,
                technique=technique,
                run_dir=run_dir,
                weights_path=weights_path,
                metrics_path=metrics_path if metrics_path.exists() else None,
                metadata=metadata,
                threshold=threshold,
                threshold_source=threshold_source,
            )
        )

    mode_order = {mode: idx for idx, mode in enumerate(ALL_FOURIER_MODES)}
    return sorted(
        runs,
        key=lambda run: (
            run.model_family,
            run.architecture,
            mode_order.get(run.technique, len(mode_order)),
            str(run.run_dir),
        ),
    )


def _in_channels(run: TrainedRun) -> int:
    return _metadata_int(
        run.metadata,
        "in_channels",
        FOURIER_CHANNELS.get(run.technique, 3),
    )


def build_model_from_run(run: TrainedRun) -> nn.Module:
    metadata = run.metadata
    in_channels = _in_channels(run)
    dropout = _metadata_float(metadata, "dropout", 0.2)

    if run.model_family == "resnet":
        architecture = str(_metadata_value(metadata, "architecture", run.architecture))
        if architecture == "resnet":
            architecture = "resnet18"
        return resnet(
            num_classes=2,
            pretrained=False,
            architecture=architecture,
            dropout=dropout,
            in_channels=in_channels,
        )

    if run.model_family == "mobilenet":
        architecture = str(_metadata_value(metadata, "architecture", run.architecture))
        variant = str(_metadata_value(metadata, "variant", "large" if "large" in architecture else "small"))
        variant = "large" if "large" in variant else "small"
        dropout_value = None if _is_missing(metadata.get("dropout")) else dropout
        return mobilenet(
            num_classes=2,
            in_channels=in_channels,
            pretrained=False,
            variant=variant,
            dropout=dropout_value,
        )

    if run.model_family == "xception":
        return xception(
            pretrained=False,
            in_channels=in_channels,
            num_classes=2,
            dropout=dropout,
        )

    if run.model_family == "vit":
        return VisionTransformerClassifier(
            num_classes=2,
            image_size=_metadata_int(metadata, "image_size", 128),
            patch_size=_metadata_int(metadata, "patch_size", 16),
            hidden_size=_metadata_int(metadata, "hidden_size", 128),
            num_hidden_layers=_metadata_int(metadata, "num_hidden_layers", 3),
            num_attention_heads=_metadata_int(metadata, "num_attention_heads", 4),
            dropout=dropout,
            in_channels=in_channels,
            use_conv_stem=_metadata_bool(metadata, "use_conv_stem", False),
            pooling=str(_metadata_value(metadata, "pooling", "cls")),
            drop_path_rate=_metadata_float(metadata, "drop_path_rate", 0.0),
        )

    if run.model_family == "clip":
        timm_model = _metadata_value(metadata, "timm_model")
        if timm_model and not _is_missing(timm_model):
            # Infer projection_dim from the checkpoint to ignore stale metadata values.
            _sd = torch.load(run.weights_path, map_location="cpu", weights_only=True)
            projection_dim = int(_sd["visual_proj.weight"].shape[0])
            return CLIPVisionClassifierWithBackbone(
                timm_model_name=str(timm_model),
                projection_dim=projection_dim,
                num_classes=2,
                dropout=dropout,
            )
        return CLIPVisionClassifier(
            num_classes=2,
            dropout=dropout,
            image_size=_metadata_int(metadata, "image_size", 224),
            patch_size=_metadata_int(metadata, "patch_size", 16),
            hidden_size=_metadata_int(metadata, "hidden_size", 256),
            projection_dim=_metadata_int(metadata, "projection_dim", 128),
            num_hidden_layers=_metadata_int(metadata, "num_hidden_layers", 6),
            num_attention_heads=_metadata_int(metadata, "num_attention_heads", 8),
        )

    if run.model_family == "dino":
        architecture = str(_metadata_value(metadata, "architecture", _metadata_value(metadata, "model", run.architecture)))
        dino_version, model_size = _parse_dino_arch(architecture)
        return DINOVisionClassifier(
            num_classes=2,
            dino_version=dino_version,
            model_size=model_size,
            dropout=dropout,
            pretrained=False,
            freeze_backbone=False,
        )

    raise ValueError(f"Unsupported model family: {run.model_family}")


def load_model_from_run(run: TrainedRun, device: torch.device) -> nn.Module:
    model = build_model_from_run(run)
    state_dict = torch.load(run.weights_path, map_location=device, weights_only=True)
    try:
        model.load_state_dict(state_dict)
    except RuntimeError:
        if isinstance(state_dict, dict) and all(str(key).startswith("module.") for key in state_dict):
            stripped = {str(key).removeprefix("module."): value for key, value in state_dict.items()}
            model.load_state_dict(stripped)
        else:
            raise
    model.to(device)
    model.eval()
    return model


def parse_splits(raw_splits: str | Iterable[str]) -> tuple[str, ...]:
    if isinstance(raw_splits, str):
        values = raw_splits.split(",")
    else:
        values = list(raw_splits)
    splits = tuple(split.strip() for split in values if split and split.strip())
    if not splits:
        raise ValueError("At least one split must be provided.")
    return splits


def build_split_specs(
    data_dir: str | Path,
    splits: Iterable[str],
    test_d_csv: str | Path | None = None,
    test_d_images_dir: str | Path | None = None,
) -> list[SplitSpec]:
    data_dir = _as_path(data_dir)
    specs: list[SplitSpec] = []
    for split in parse_splits(splits):
        if split in {"val", "test"}:
            specs.append(
                SplitSpec(
                    name=split,
                    csv_path=data_dir / f"{split}.csv",
                    images_dir=phase1_split_root(split),
                )
            )
        elif split == "test_d":
            if test_d_csv is None or test_d_images_dir is None:
                raise ValueError("test_d requires --test-d-csv and --test-d-images-dir.")
            specs.append(
                SplitSpec(
                    name=split,
                    csv_path=_as_path(test_d_csv),
                    images_dir=_as_path(test_d_images_dir),
                )
            )
        else:
            raise ValueError("splits must contain only 'val', 'test', or 'test_d'.")
    return specs


def _eval_transform(run: TrainedRun) -> transforms.Compose:
    default_size = 299 if run.model_family == "xception" else 224
    image_size = _metadata_int(run.metadata, "image_size", default_size)
    if run.model_family in {"xception", "clip"}:
        mean = [0.5, 0.5, 0.5]
        std = [0.5, 0.5, 0.5]
    else:
        mean = [0.485, 0.456, 0.406]
        std = [0.229, 0.224, 0.225]
    return transforms.Compose(
        [
            transforms.Resize((image_size, image_size)),
            transforms.ToTensor(),
            transforms.Normalize(mean=mean, std=std),
        ]
    )


def _dataset_for_split(
    run: TrainedRun,
    split: SplitSpec,
    limit_per_split: int | None,
) -> ImageDataset:
    limit = np.inf if limit_per_split is None else int(limit_per_split)
    default_size = 299 if run.model_family == "xception" else 224
    spatial_size = (
        (_metadata_int(run.metadata, "image_size", default_size),) * 2
        if run.technique != "none"
        else None
    )
    return ImageDataset(
        file_csv=split.csv_path,
        images_dir=split.images_dir,
        transform=_eval_transform(run),
        data_limit=limit,
        fourier=run.technique,  # type: ignore[arg-type]
        spatial_size=spatial_size,
    )


def _loader_for_dataset(
    dataset: ImageDataset,
    batch_size: int,
    num_workers: int,
    device: torch.device,
) -> DataLoader:
    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=device.type == "cuda",
        persistent_workers=num_workers > 0,
    )


def _metric_row(
    run: TrainedRun,
    split_name: str,
    metrics: dict,
    outputs_path: Path,
    metrics_path: Path,
    predictions_path: Path,
) -> dict:
    n_samples = int(len(metrics["y_true"]))
    return {
        "model_family": run.model_family,
        "architecture": run.architecture,
        "technique": run.technique,
        "split": split_name,
        "auc": float(metrics["auc"]),
        "acc": float(metrics["acc"]),
        "f1": float(metrics["f1"]),
        "precision": float(metrics["precision"]),
        "threshold": float(run.threshold),
        "threshold_source": run.threshold_source,
        "n_samples": n_samples,
        "run_dir": str(run.run_dir),
        "outputs_path": str(outputs_path),
        "metrics_path": str(metrics_path),
        "predictions_path": str(predictions_path),
    }


def save_split_results(run: TrainedRun, split_name: str, metrics: dict) -> dict:
    results_dir = run.run_dir / "results"
    results_dir.mkdir(parents=True, exist_ok=True)
    outputs_path = results_dir / f"outputs_{split_name}.npz"
    metrics_path = results_dir / f"metrics_{split_name}.csv"
    predictions_path = results_dir / f"predictions_{split_name}.csv"

    np.savez_compressed(
        outputs_path,
        logits=metrics["logits"],
        ids=metrics["ids"],
        probs=metrics["probs"],
        y_true=metrics["y_true"],
        y_pred=metrics["y_pred"],
        threshold=float(run.threshold),
    )

    row = _metric_row(run, split_name, metrics, outputs_path, metrics_path, predictions_path)
    metrics_extra = dict(row)
    metrics_extra.update(
        {
            "loss": float(metrics.get("loss", 0.0)),
            "recall": float(metrics.get("recall", 0.0)),
            "specificity": float(metrics.get("specificity", 0.0)),
            "tp": int(metrics.get("tp", 0)),
            "fp": int(metrics.get("fp", 0)),
            "fn": int(metrics.get("fn", 0)),
            "tn": int(metrics.get("tn", 0)),
        }
    )
    pd.DataFrame([metrics_extra]).to_csv(metrics_path, index=False)

    predictions = pd.DataFrame(
        {
            "id": metrics["ids"],
            "y_true": metrics["y_true"],
            "y_pred": metrics["y_pred"],
            "prob_pos": metrics["probs"],
        }
    )
    predictions["correct"] = (predictions["y_true"] == predictions["y_pred"]).astype(int)
    predictions.to_csv(predictions_path, index=False)
    return row


def _write_skipped(models_root: Path, skipped_rows: list[dict]) -> None:
    if skipped_rows:
        pd.DataFrame(skipped_rows).to_csv(models_root / "skipped_evaluations.csv", index=False)


def evaluate_trained_runs(
    models_root: str | Path,
    data_dir: str | Path,
    test_d_csv: str | Path | None = None,
    test_d_images_dir: str | Path | None = None,
    output_csv: str | Path | None = None,
    batch_size: int = 32,
    num_workers: int = 4,
    device: torch.device | str | None = None,
    only_model_family: str | None = None,
    splits: Iterable[str] = ("val", "test", "test_d"),
    limit_per_split: int | None = None,
) -> pd.DataFrame:
    models_root = _as_path(models_root)
    output_csv = _as_path(output_csv) if output_csv is not None else models_root / "all_metrics_by_split.csv"
    output_csv.parent.mkdir(parents=True, exist_ok=True)
    device = (
        torch.device(device)
        if isinstance(device, str)
        else device
        if device is not None
        else torch.device("cuda" if torch.cuda.is_available() else "cpu")
    )
    split_specs = build_split_specs(data_dir, splits, test_d_csv, test_d_images_dir)
    runs = discover_trained_runs(models_root, only_model_family=only_model_family)

    summary_rows: list[dict] = []
    skipped_rows: list[dict] = []
    criterion = nn.CrossEntropyLoss()

    for run in runs:
        try:
            model = load_model_from_run(run, device)
        except Exception as exc:  # pragma: no cover - exercised by server data shape.
            skipped_rows.append(
                {
                    "model_family": run.model_family,
                    "architecture": run.architecture,
                    "technique": run.technique,
                    "split": "all",
                    "run_dir": str(run.run_dir),
                    "reason": f"model_load_failed: {exc}",
                }
            )
            continue

        for split in split_specs:
            try:
                dataset = _dataset_for_split(run, split, limit_per_split)
                loader = _loader_for_dataset(dataset, batch_size, num_workers, device)
                metrics = evaluate_classifier(
                    model,
                    loader,
                    criterion,
                    device,
                    threshold=run.threshold,
                    use_amp=device.type == "cuda",
                    desc=f"{run.model_family}/{run.technique}/{split.name}",
                )
                summary_rows.append(save_split_results(run, split.name, metrics))
            except Exception as exc:  # pragma: no cover - depends on server files.
                skipped_rows.append(
                    {
                        "model_family": run.model_family,
                        "architecture": run.architecture,
                        "technique": run.technique,
                        "split": split.name,
                        "run_dir": str(run.run_dir),
                        "reason": str(exc),
                    }
                )

    summary = pd.DataFrame(
        summary_rows,
        columns=[
            "model_family",
            "architecture",
            "technique",
            "split",
            "auc",
            "acc",
            "f1",
            "precision",
            "threshold",
            "threshold_source",
            "n_samples",
            "run_dir",
            "outputs_path",
            "metrics_path",
            "predictions_path",
        ],
    )
    summary.to_csv(output_csv, index=False)
    _write_skipped(models_root, skipped_rows)
    return summary


def list_runs(models_root: str | Path, only_model_family: str | None = None) -> pd.DataFrame:
    runs = discover_trained_runs(models_root, only_model_family=only_model_family)
    return pd.DataFrame(
        [
            {
                "model_family": run.model_family,
                "architecture": run.architecture,
                "technique": run.technique,
                "threshold": run.threshold,
                "threshold_source": run.threshold_source,
                "run_dir": str(run.run_dir),
                "weights_path": str(run.weights_path),
            }
            for run in runs
        ]
    )


def _parse_limit(value: int | None) -> int | None:
    if value is None:
        return None
    return max(0, int(value))


def main(argv: Sequence[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description="Evaluate trained vision checkpoints by split.")
    parser.add_argument("--models-root", type=Path, default=Path("models"))
    parser.add_argument("--data-dir", type=Path, default=Path("data/raw"))
    parser.add_argument("--test-d-csv", type=Path, default='/home/lucas.ocunha/tcc/data/raw/test.csv')
    parser.add_argument("--test-d-images-dir", type=Path, default='/media/ssd2/lucas.ocunha/datasets/phase1/test_d')
    parser.add_argument("--output-csv", type=Path, default=None)
    parser.add_argument("--batch-size", type=int, default=124)
    parser.add_argument("--num-workers", type=int, default=16)
    parser.add_argument("--device", type=str, default=None)
    parser.add_argument("--only-model-family", type=str, default=None)
    parser.add_argument("--splits", type=str, default="val,test,test_d")
    parser.add_argument("--limit-per-split", type=int, default=None)
    parser.add_argument("--list-runs", action="store_true")
    args = parser.parse_args(argv)

    if args.list_runs:
        runs = list_runs(args.models_root, only_model_family=args.only_model_family)
        if runs.empty:
            print("No trained runs found.")
        else:
            print(runs.to_string(index=False))
        return

    summary = evaluate_trained_runs(
        models_root=args.models_root,
        data_dir=args.data_dir,
        test_d_csv=args.test_d_csv,
        test_d_images_dir=args.test_d_images_dir,
        output_csv=args.output_csv,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        device=args.device,
        only_model_family=args.only_model_family,
        splits=("test"),
        limit_per_split=_parse_limit(args.limit_per_split),
    )
    print(f"Saved {len(summary)} evaluation rows.")


if __name__ == "__main__":
    main()
