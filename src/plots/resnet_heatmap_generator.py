from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd
import torch
from torchvision import transforms

from src.data import ImageDataset
from src.data.paths import phase1_split_root
from src.models.resnet import resnet
from src.plots.heatmap import gradcam_heatmap, save_heatmap_overlay


def read_resnet_metadata(model_dir: str | Path) -> dict:
    metrics_path = Path(model_dir) / "results" / "metrics_summary.csv"
    if not metrics_path.exists():
        raise FileNotFoundError(f"Missing metrics file: {metrics_path}")

    metrics = pd.read_csv(metrics_path)
    if metrics.empty:
        raise ValueError(f"Metrics file is empty: {metrics_path}")
    row = metrics.iloc[0]
    return {
        "architecture": str(row.get("architecture", "resnet18")),
        "fourier": str(row.get("fourier", "none")),
        "in_channels": int(row.get("in_channels", 3)),
        "image_size": int(row.get("image_size", 224)),
    }


def select_heatmap_examples(
    predictions: pd.DataFrame,
    per_group: int = 2,
) -> pd.DataFrame:
    groups = [
        ("true_positive", (predictions["y_true"] == 1) & (predictions["y_pred"] == 1), False),
        ("true_negative", (predictions["y_true"] == 0) & (predictions["y_pred"] == 0), True),
        ("false_positive", (predictions["y_true"] == 0) & (predictions["y_pred"] == 1), False),
        ("false_negative", (predictions["y_true"] == 1) & (predictions["y_pred"] == 0), True),
    ]
    selected = []
    for group_name, mask, ascending in groups:
        group = predictions.loc[mask].copy()
        if group.empty:
            continue
        group = group.sort_values("prob_pos", ascending=ascending).head(per_group)
        group["heatmap_group"] = group_name
        selected.append(group)

    if not selected:
        return predictions.head(0).copy()
    return pd.concat(selected, ignore_index=True)


def _prediction_group(row: pd.Series) -> str:
    y_true = int(row["y_true"])
    y_pred = int(row["y_pred"])
    if y_true == 1 and y_pred == 1:
        return "true_positive"
    if y_true == 0 and y_pred == 0:
        return "true_negative"
    if y_true == 0 and y_pred == 1:
        return "false_positive"
    return "false_negative"


def load_resnet_checkpoint(
    model_dir: str | Path,
    metadata: dict,
    device: torch.device,
) -> torch.nn.Module:
    checkpoint_path = Path(model_dir) / "weights" / "best_resnet.pth"
    if not checkpoint_path.exists():
        raise FileNotFoundError(f"Missing checkpoint: {checkpoint_path}")

    model = resnet(
        num_classes=2,
        pretrained=False,
        architecture=metadata["architecture"],
        in_channels=metadata["in_channels"],
    )
    state_dict = torch.load(checkpoint_path, map_location=device, weights_only=True)
    model.load_state_dict(state_dict)
    model.to(device)
    model.eval()
    return model


def _eval_transform(image_size: int) -> transforms.Compose:
    return transforms.Compose(
        [
            transforms.Resize((image_size, image_size)),
            transforms.ToTensor(),
            transforms.Normalize(
                mean=[0.485, 0.456, 0.406],
                std=[0.229, 0.224, 0.225],
            ),
        ]
    )


def _heatmap_filename(row: pd.Series) -> str:
    return (
        f"id_{int(row['id']):06d}_"
        f"true{int(row['y_true'])}_"
        f"pred{int(row['y_pred'])}_"
        f"prob{float(row['prob_pos']):.2f}.png"
    )


def generate_resnet_heatmaps(
    model_dir: str | Path,
    test_csv: str | Path,
    images_dir: str | Path,
    output_dir: str | Path | None = None,
    ids: list[int] | None = None,
    per_group: int = 2,
    device: torch.device | None = None,
) -> pd.DataFrame:
    model_dir = Path(model_dir)
    output_dir = Path(output_dir) if output_dir is not None else model_dir / "plots" / "heatmaps"
    output_dir.mkdir(parents=True, exist_ok=True)
    device = device or torch.device("cuda" if torch.cuda.is_available() else "cpu")

    metadata = read_resnet_metadata(model_dir)
    if metadata["fourier"] != "none":
        raise ValueError("This generator targets the RGB ResNet run with fourier='none'.")
    model = load_resnet_checkpoint(model_dir, metadata, device)

    predictions_path = model_dir / "results" / "predictions.csv"
    if not predictions_path.exists():
        raise FileNotFoundError(f"Missing predictions file: {predictions_path}")
    predictions = pd.read_csv(predictions_path)
    if ids is None:
        selected = select_heatmap_examples(predictions, per_group=per_group)
    else:
        id_set = {int(value) for value in ids}
        selected = predictions[predictions["id"].isin(id_set)].copy()
        selected["heatmap_group"] = selected.apply(_prediction_group, axis=1)

    dataset = ImageDataset(
        file_csv=Path(test_csv),
        images_dir=Path(images_dir),
        transform=_eval_transform(metadata["image_size"]),
        fourier="none",
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
            raise FileNotFoundError(
                f"Selected image for id {sample_id} is missing: {image_path}"
            )

        image_tensor, label, _ = dataset[sample_id]
        if image_tensor.shape[0] != metadata["in_channels"]:
            raise ValueError(
                f"Expected {metadata['in_channels']} channels, got {image_tensor.shape[0]}."
            )

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
    return [int(value.strip()) for value in raw_ids.split(",") if value.strip()]


def main() -> None:
    parser = argparse.ArgumentParser(description="Generate Grad-CAM heatmaps for ResNet.")
    parser.add_argument("--model-dir", type=Path, default=Path("models/resnet"))
    parser.add_argument("--csv", type=Path, default=Path("data/raw/test.csv"))
    parser.add_argument("--images-dir", type=Path, default=None)
    parser.add_argument("--output-dir", type=Path, default=None)
    parser.add_argument("--ids", type=str, default=None, help="Comma-separated prediction ids.")
    parser.add_argument("--per-group", type=int, default=2)
    args = parser.parse_args()

    images_dir = args.images_dir or phase1_split_root("test")
    manifest = generate_resnet_heatmaps(
        model_dir=args.model_dir,
        test_csv=args.csv,
        images_dir=images_dir,
        output_dir=args.output_dir,
        ids=_parse_ids(args.ids),
        per_group=args.per_group,
    )
    print(f"Saved {len(manifest)} heatmaps.")


if __name__ == "__main__":
    main()
