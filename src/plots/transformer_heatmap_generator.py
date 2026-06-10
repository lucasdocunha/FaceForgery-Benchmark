from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd
import torch

from src.data import ImageDataset
from src.data.paths import phase1_split_root
from src.pipelines.evaluate_trained import (
    TrainedRun,
    _eval_transform,
    _is_missing,
    _metadata_value,
    _read_metrics,
    load_model_from_run,
    build_model_from_run,
)
from src.plots.heatmap import gradcam_heatmap, save_heatmap_overlay
from src.plots.resnet_heatmap_generator import (
    _heatmap_filename,
    _prediction_group,
    select_heatmap_examples,
)

SUPPORTED_FAMILIES = ("clip", "vit", "dino", "xception", "mobilenet", "resnet")


def _detect_weights(model_dir: Path) -> Path:
    candidates = sorted((model_dir / "weights").glob("best_*.pth"))
    if not candidates:
        raise FileNotFoundError(f"No best_*.pth found in {model_dir / 'weights'}")
    return candidates[0]


def _run_from_dir(model_dir: Path, family: str) -> TrainedRun:
    metrics_path = model_dir / "results" / "metrics_summary.csv"
    metadata = _read_metrics(metrics_path)

    weights_path = _detect_weights(model_dir)

    architecture = str(
        _metadata_value(
            metadata,
            "architecture",
            _metadata_value(metadata, "model", family),
        )
    )

    technique = "none"
    for key in ("fourier", "input_mode", "technique"):
        raw = _metadata_value(metadata, key)
        if raw is not None and not _is_missing(raw):
            technique = str(raw)
            break

    threshold_raw = _metadata_value(metadata, "threshold")
    if threshold_raw is None or _is_missing(threshold_raw):
        threshold, threshold_source = 0.5, "default_0.5"
    else:
        threshold = float(np.clip(float(threshold_raw), 0.0, 1.0))
        threshold_source = "metrics_summary"

    return TrainedRun(
        model_family=family,
        architecture=architecture,
        technique=technique,
        run_dir=model_dir,
        weights_path=weights_path,
        metrics_path=metrics_path if metrics_path.exists() else None,
        metadata=metadata,
        threshold=threshold,
        threshold_source=threshold_source,
    )


def generate_transformer_heatmaps(
    model_dir: str | Path,
    family: str,
    test_csv: str | Path,
    images_dir: str | Path,
    output_dir: str | Path | None = None,
    ids: list[int] | None = None,
    per_group: int = 2,
    device: torch.device | None = None,
) -> pd.DataFrame:
    model_dir = Path(model_dir)
    output_dir = (
        Path(output_dir)
        if output_dir is not None
        else model_dir / "plots" / "heatmaps"
    )
    output_dir.mkdir(parents=True, exist_ok=True)
    device = device or torch.device("cuda" if torch.cuda.is_available() else "cpu")

    run = _run_from_dir(model_dir, family)
    model = load_model_from_run(run, device)

    predictions_path = model_dir / "results" / "predictions.csv"
    if not predictions_path.exists():
        raise FileNotFoundError(f"Missing predictions file: {predictions_path}")
    predictions = pd.read_csv(predictions_path)

    if ids is None:
        selected = select_heatmap_examples(predictions, per_group=per_group)
    else:
        id_set = {int(v) for v in ids}
        selected = predictions[predictions["id"].isin(id_set)].copy()
        selected["heatmap_group"] = selected.apply(_prediction_group, axis=1)

    dataset = ImageDataset(
        file_csv=Path(test_csv),
        images_dir=Path(images_dir),
        transform=_eval_transform(run),
        fourier=run.technique,  # type: ignore[arg-type]
        spatial_size=None,
    )
    test_rows = pd.read_csv(test_csv)
    test_rows.columns = test_rows.columns.str.strip()

    manifest_rows = []
    for _, row in selected.iterrows():
        sample_id = int(row["id"])
        if sample_id < 0 or sample_id >= len(dataset):
            raise IndexError(f"Prediction id {sample_id} is outside the test dataset.")
        img_name = str(test_rows.iloc[sample_id]["img_name"])
        image_path = Path(images_dir) / img_name
        if not image_path.exists():
            raise FileNotFoundError(f"Image missing: {image_path}")

        image_tensor, label, _ = dataset[sample_id]
        target_class = int(row["y_pred"])
        heatmap = gradcam_heatmap(model, image_tensor, target_class=target_class)
        if not torch.isfinite(heatmap).all():
            raise ValueError(f"Heatmap for id {sample_id} has non-finite values.")
        if float(heatmap.min()) < 0.0 or float(heatmap.max()) > 1.0:
            raise ValueError(f"Heatmap for id {sample_id} is outside [0, 1].")

        heatmap_path = save_heatmap_overlay(
            image_tensor,
            heatmap,
            output_dir / _heatmap_filename(row),
        )
        manifest_rows.append(
            {
                "id": sample_id,
                "img_name": img_name,
                "y_true": int(row["y_true"]),
                "y_pred": int(row["y_pred"]),
                "prob_pos": float(row["prob_pos"]),
                "correct": int(row["correct"]),
                "heatmap_group": row["heatmap_group"],
                "heatmap_path": str(heatmap_path),
                "label_from_dataset": int(label),
            }
        )

    manifest = pd.DataFrame(manifest_rows)
    manifest.to_csv(output_dir / "manifest.csv", index=False)
    return manifest


def _parse_ids(raw_ids: str | None) -> list[int] | None:
    if raw_ids is None or raw_ids.strip() == "":
        return None
    return [int(v.strip()) for v in raw_ids.split(",") if v.strip()]


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Generate Grad-CAM heatmaps for transformer models (CLIP, ViT, DINO)."
    )
    parser.add_argument(
        "--model-dir",
        type=Path,
        required=True,
        help="Path to the model run directory (contains results/ and weights/).",
    )
    parser.add_argument(
        "--family",
        type=str,
        required=True,
        choices=list(SUPPORTED_FAMILIES),
        help="Model family.",
    )
    parser.add_argument("--csv", type=Path, default=Path("data/raw/test.csv"))
    parser.add_argument("--images-dir", type=Path, default=None)
    parser.add_argument("--output-dir", type=Path, default=None)
    parser.add_argument(
        "--ids",
        type=str,
        default=None,
        help="Comma-separated prediction ids to visualize.",
    )
    parser.add_argument("--per-group", type=int, default=2)
    args = parser.parse_args()

    images_dir = args.images_dir or phase1_split_root("test")
    manifest = generate_transformer_heatmaps(
        model_dir=args.model_dir,
        family=args.family,
        test_csv=args.csv,
        images_dir=images_dir,
        output_dir=args.output_dir,
        ids=_parse_ids(args.ids),
        per_group=args.per_group,
    )
    print(f"Saved {len(manifest)} heatmaps.")


if __name__ == "__main__":
    main()
