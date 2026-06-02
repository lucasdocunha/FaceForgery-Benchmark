from pathlib import Path

import numpy as np
import pandas as pd
import pytest
import torch
from PIL import Image


def _write_image_dataset(root: Path, csv_path: Path, count: int) -> None:
    root.mkdir(parents=True, exist_ok=True)
    rows = []
    for idx in range(count):
        name = f"sample_{idx:03d}.jpg"
        value = 40 + idx * 15
        arr = np.full((80, 80, 3), value, dtype=np.uint8)
        arr[:, :, 0] = (arr[:, :, 0] + idx * 9) % 255
        arr[::5, :, 1] = 255 - arr[::5, :, 1]
        Image.fromarray(arr, mode="RGB").save(root / name)
        rows.append({"img_name": name, "target": idx % 2})
    pd.DataFrame(rows).to_csv(csv_path, index=False)


def test_discover_trained_runs_detects_techniques_and_filters_family(tmp_path):
    from src.pipelines.evaluate_trained import discover_trained_runs

    models_root = tmp_path / "models"
    for path in (
        models_root / "resnet" / "weights" / "best_resnet.pth",
        models_root / "resnet" / "concat" / "weights" / "best_resnet.pth",
        models_root
        / "mobilenet"
        / "mobilenetv3_large"
        / "none"
        / "weights"
        / "best_mobilenetv3_large.pth",
    ):
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(b"checkpoint placeholder")

    runs = discover_trained_runs(models_root)
    filtered = discover_trained_runs(models_root, only_model_family="resnet")

    assert [(run.model_family, run.architecture, run.technique) for run in runs] == [
        ("mobilenet", "mobilenetv3_large", "none"),
        ("resnet", "resnet", "none"),
        ("resnet", "resnet", "concat"),
    ]
    assert [run.model_family for run in filtered] == ["resnet", "resnet"]


def test_build_model_from_run_supports_current_model_families(tmp_path):
    from src.pipelines.evaluate_trained import TrainedRun, build_model_from_run

    cases = [
        ("resnet", "resnet18", {"in_channels": 3, "image_size": 64}),
        ("mobilenet", "mobilenetv3_small", {"in_channels": 3, "image_size": 64}),
        ("xception", "xception", {"in_channels": 3, "image_size": 128}),
        (
            "vit",
            "vit_scratch",
            {
                "in_channels": 3,
                "image_size": 64,
                "patch_size": 16,
                "hidden_size": 32,
                "num_hidden_layers": 1,
                "num_attention_heads": 4,
            },
        ),
        (
            "clip",
            "clip_vit_scratch",
            {
                "image_size": 64,
                "patch_size": 16,
                "hidden_size": 32,
                "projection_dim": 16,
                "num_hidden_layers": 1,
                "num_attention_heads": 4,
            },
        ),
    ]

    for family, architecture, metadata in cases:
        weights_path = tmp_path / family / "weights" / f"best_{family}.pth"
        run = TrainedRun(
            model_family=family,
            architecture=architecture,
            technique="none",
            run_dir=weights_path.parent.parent,
            weights_path=weights_path,
            metrics_path=None,
            metadata=metadata,
            threshold=0.5,
            threshold_source="default_0.5",
        )

        model = build_model_from_run(run)

        assert isinstance(model, torch.nn.Module)


def test_build_split_specs_requires_test_d_paths(tmp_path):
    from src.pipelines.evaluate_trained import build_split_specs

    with pytest.raises(ValueError, match="test_d requires"):
        build_split_specs(
            data_dir=tmp_path / "data",
            splits=("test_d",),
            test_d_csv=None,
            test_d_images_dir=None,
        )


def test_evaluate_trained_runs_writes_outputs_predictions_metrics_and_summary(
    tmp_path,
    monkeypatch,
):
    from src.models.resnet import resnet
    from src.pipelines.evaluate_trained import evaluate_trained_runs

    models_root = tmp_path / "models"
    run_dir = models_root / "resnet"
    weights_dir = run_dir / "weights"
    results_dir = run_dir / "results"
    weights_dir.mkdir(parents=True)
    results_dir.mkdir()

    model = resnet(
        num_classes=2,
        pretrained=False,
        architecture="resnet18",
        in_channels=3,
    )
    torch.save(model.state_dict(), weights_dir / "best_resnet.pth")
    pd.DataFrame(
        [
            {
                "architecture": "resnet18",
                "fourier": "none",
                "in_channels": 3,
                "image_size": 64,
                "threshold": 0.4,
            }
        ]
    ).to_csv(results_dir / "metrics_summary.csv", index=False)

    data_dir = tmp_path / "data" / "raw"
    data_dir.mkdir(parents=True)
    dataset_root = tmp_path / "phase1"
    _write_image_dataset(dataset_root / "testset", data_dir / "test.csv", count=6)
    _write_image_dataset(dataset_root / "valset", data_dir / "val.csv", count=6)
    test_d_csv = tmp_path / "test_d.csv"
    test_d_images = tmp_path / "test_d_images"
    _write_image_dataset(test_d_images, test_d_csv, count=6)
    monkeypatch.setenv("TCC_DATASET_ROOT", str(dataset_root))

    summary = evaluate_trained_runs(
        models_root=models_root,
        data_dir=data_dir,
        test_d_csv=test_d_csv,
        test_d_images_dir=test_d_images,
        output_csv=models_root / "resnet_50_test_and_test_d_metrics.csv",
        only_model_family="resnet",
        splits=("test", "test_d"),
        limit_per_split=3,
        batch_size=2,
        num_workers=0,
        device=torch.device("cpu"),
    )

    assert summary["split"].tolist() == ["test", "test_d"]
    assert summary["n_samples"].tolist() == [3, 3]
    assert summary["threshold_source"].tolist() == ["metrics_summary", "metrics_summary"]
    assert (models_root / "resnet_50_test_and_test_d_metrics.csv").exists()

    for split in ("test", "test_d"):
        outputs_path = results_dir / f"outputs_{split}.npz"
        metrics_path = results_dir / f"metrics_{split}.csv"
        predictions_path = results_dir / f"predictions_{split}.csv"
        assert outputs_path.exists()
        assert metrics_path.exists()
        assert predictions_path.exists()

        outputs = np.load(outputs_path)
        predictions = pd.read_csv(predictions_path)
        metrics = pd.read_csv(metrics_path)

        assert outputs["logits"].shape == (3, 2)
        assert len(predictions) == 3
        assert len(metrics) == 1
        assert {"auc", "acc", "f1", "precision"}.issubset(metrics.columns)
