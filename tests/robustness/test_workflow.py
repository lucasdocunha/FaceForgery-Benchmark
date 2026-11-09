import json
from pathlib import Path
import numpy as np
import pandas as pd
from PIL import Image
import pytest
import torch
import yaml
from src.robustness.manifests import save_manifest, load_manifest
from src.robustness.imaging import CanonicalDataset
from src.robustness.experiments import read_config, plan
from src.robustness.engine import train, load_pilot, verify_complete
from src.robustness.inference import evaluate


def fixture_data(tmp_path):
    paths = {}
    for split in ["train", "val", "test"]:
        root = tmp_path / split
        root.mkdir()
        rows = []
        for i in range(8):
            name = f"{i}.png"
            rng = np.random.default_rng(100 + i)
            Image.fromarray(rng.integers(0, 256, (32, 32, 3), dtype=np.uint8)).save(
                root / name
            )
            rows.append(
                {
                    "img_name": name,
                    "label": i % 2,
                    "sample_id": f"{split}:{i}",
                    "group_id": f"{split}:g{i}",
                    "dataset": "synthetic",
                    "split": split,
                }
            )
        path = tmp_path / f"{split}.csv"
        save_manifest(pd.DataFrame(rows), path, {})
        paths[split] = (path, root)
    return paths


def config_file(tmp_path, paths):
    cfg = {
        "name": "synthetic-contract-test",
        "purpose": "synthetic_smoke",
        "model": {"kind": "rgb", "backbone": "resnet18", "pretrained": False},
        "data": {
            "train_manifest": str(paths["train"][0]),
            "val_manifest": str(paths["val"][0]),
            "train_root": str(paths["train"][1]),
            "val_root": str(paths["val"][1]),
        },
        "training": {
            "epochs": 1,
            "batch_size": 4,
            "workers": 0,
            "image_size": 32,
            "recipe": "basic_v1",
        },
        "output_root": str(tmp_path / "runs"),
    }
    path = tmp_path / "config.yaml"
    path.write_text(yaml.safe_dump(cfg))
    return path


def test_missing_image_never_substituted(tmp_path):
    paths = fixture_data(tmp_path)
    f, _ = load_manifest(paths["test"][0])
    (paths["test"][1] / "0.png").unlink()
    ds = CanonicalDataset(f, paths["test"][1], 32)
    with pytest.raises(FileNotFoundError):
        ds[0]


def test_augmentation_seed_independent_of_access_order(tmp_path):
    paths = fixture_data(tmp_path)
    f, _ = load_manifest(paths["train"][0])
    ds = CanonicalDataset(
        f, paths["train"][1], 32, recipe="robust_v1", seed=42, training=True
    )
    a = ds[0]["image"].clone()
    ds[5]
    assert torch.equal(a, ds[0]["image"])


def test_planning_never_loads_test_or_creates_run(tmp_path):
    paths = fixture_data(tmp_path)
    cfg = read_config(config_file(tmp_path, paths))
    result = plan(cfg)
    assert not Path(result["run_dir"]).exists() and result["state"] == "planned"
    assert result["config"]["training"]["recipe"] == "basic_v1"


def test_end_to_end_synthetic_train_load_evaluate(tmp_path):
    torch.set_num_threads(1)
    paths = fixture_data(tmp_path)
    cfg = read_config(config_file(tmp_path, paths))
    result = plan(cfg)
    root = Path(result["run_dir"])
    state = train(cfg, device="cpu")
    assert state["state"] == "complete"
    verify_complete(root)
    model, record = load_pilot(root)
    report = evaluate(
        model,
        *paths["test"],
        tmp_path / "evaluation",
        checkpoint_path=root / "best.pt",
        calibration_path=root / "calibration.json",
        image_size=32,
        batch_size=4,
    )
    assert report["frame"]["n"] == 8
    assert (
        json.loads((tmp_path / "evaluation/status.json").read_text())["state"]
        == "complete"
    )
    with pytest.raises(FileExistsError):
        train(cfg)
    assert train(cfg, resume=True)["state"] == "complete"
    (root / "best.pt").write_bytes(b"changed")
    with pytest.raises(ValueError):
        verify_complete(root)
