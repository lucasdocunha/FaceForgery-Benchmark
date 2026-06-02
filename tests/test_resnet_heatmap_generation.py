from pathlib import Path

import numpy as np
import pandas as pd
import pytest
import torch
from PIL import Image


def test_select_heatmap_examples_balances_prediction_groups():
    from src.plots.resnet_heatmap_generator import select_heatmap_examples

    predictions = pd.DataFrame(
        {
            "id": [0, 1, 2, 3, 4, 5],
            "y_true": [1, 1, 0, 0, 1, 0],
            "y_pred": [1, 1, 0, 0, 0, 1],
            "prob_pos": [0.91, 0.83, 0.12, 0.32, 0.21, 0.98],
            "correct": [1, 1, 1, 1, 0, 0],
        }
    )

    selected = select_heatmap_examples(predictions, per_group=1)

    assert selected["id"].tolist() == [0, 2, 5, 4]
    assert selected["heatmap_group"].tolist() == [
        "true_positive",
        "true_negative",
        "false_positive",
        "false_negative",
    ]


def test_generate_resnet_heatmaps_uses_existing_gradcam_pipeline(tmp_path):
    from src.models.resnet import resnet
    from src.plots.resnet_heatmap_generator import generate_resnet_heatmaps

    model_dir = tmp_path / "models" / "resnet"
    weights_dir = model_dir / "weights"
    results_dir = model_dir / "results"
    weights_dir.mkdir(parents=True)
    results_dir.mkdir()

    pd.DataFrame(
        [
            {
                "architecture": "resnet18",
                "fourier": "none",
                "in_channels": 3,
                "image_size": 64,
            }
        ]
    ).to_csv(results_dir / "metrics_summary.csv", index=False)
    pd.DataFrame(
        {
            "id": [0],
            "y_true": [1],
            "y_pred": [1],
            "prob_pos": [0.87],
            "correct": [1],
        }
    ).to_csv(results_dir / "predictions.csv", index=False)

    model = resnet(
        num_classes=2,
        pretrained=False,
        architecture="resnet18",
        in_channels=3,
    )
    torch.save(model.state_dict(), weights_dir / "best_resnet.pth")

    images_dir = tmp_path / "images"
    images_dir.mkdir()
    image_array = np.full((80, 80, 3), 127, dtype=np.uint8)
    image_array[16:48, 24:56, 0] = 255
    Image.fromarray(image_array, mode="RGB").save(images_dir / "sample.jpg")
    pd.DataFrame({"img_name": ["sample.jpg"], "target": [1]}).to_csv(
        tmp_path / "test.csv",
        index=False,
    )

    manifest = generate_resnet_heatmaps(
        model_dir=model_dir,
        test_csv=tmp_path / "test.csv",
        images_dir=images_dir,
        output_dir=model_dir / "plots" / "heatmaps",
        ids=[0],
        device=torch.device("cpu"),
    )

    assert len(manifest) == 1
    assert manifest.loc[0, "heatmap_group"] == "true_positive"
    heatmap_path = Path(manifest.loc[0, "heatmap_path"])
    assert heatmap_path.exists()
    assert heatmap_path.suffix == ".png"
    assert (model_dir / "plots" / "heatmaps" / "manifest.csv").exists()


def test_generate_resnet_heatmaps_fails_when_selected_image_is_missing(tmp_path):
    from src.models.resnet import resnet
    from src.plots.resnet_heatmap_generator import generate_resnet_heatmaps

    model_dir = tmp_path / "models" / "resnet"
    weights_dir = model_dir / "weights"
    results_dir = model_dir / "results"
    weights_dir.mkdir(parents=True)
    results_dir.mkdir()

    pd.DataFrame(
        [
            {
                "architecture": "resnet18",
                "fourier": "none",
                "in_channels": 3,
                "image_size": 64,
            }
        ]
    ).to_csv(results_dir / "metrics_summary.csv", index=False)
    pd.DataFrame(
        {
            "id": [0],
            "y_true": [1],
            "y_pred": [1],
            "prob_pos": [0.87],
            "correct": [1],
        }
    ).to_csv(results_dir / "predictions.csv", index=False)

    model = resnet(
        num_classes=2,
        pretrained=False,
        architecture="resnet18",
        in_channels=3,
    )
    torch.save(model.state_dict(), weights_dir / "best_resnet.pth")

    images_dir = tmp_path / "images"
    images_dir.mkdir()
    pd.DataFrame({"img_name": ["missing.jpg"], "target": [1]}).to_csv(
        tmp_path / "test.csv",
        index=False,
    )

    with pytest.raises(FileNotFoundError, match="Selected image for id 0 is missing"):
        generate_resnet_heatmaps(
            model_dir=model_dir,
            test_csv=tmp_path / "test.csv",
            images_dir=images_dir,
            output_dir=model_dir / "plots" / "heatmaps",
            ids=[0],
            device=torch.device("cpu"),
        )
