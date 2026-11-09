import copy
import json
from pathlib import Path

import numpy as np
import pandas as pd
from PIL import Image
import pytest
import torch

from src.experimental import runtime
from src.experimental.reconstruction import fit, load_model, normalize_config, predict
from src.robustness.manifests import save_manifest


@pytest.fixture(autouse=True)
def limited_threads():
    previous = torch.get_num_threads()
    torch.set_num_threads(1)
    yield
    torch.set_num_threads(previous)


@pytest.fixture
def source_config(tmp_path):
    data = {}
    for split in ("train", "val"):
        root = tmp_path / split
        root.mkdir()
        rows = []
        for index in range(6):
            identity = f"{split}-{index}"
            pixels = np.full((32, 32, 3), 60 + index * 20 + (3 if split == "val" else 0), dtype=np.uint8)
            pixels[::3, :, 0] += 20
            Image.fromarray(pixels).save(root / f"{identity}.png")
            rows.append({"sample_id": identity, "img_name": f"{identity}.png", "label": index % 2,
                         "group_id": identity, "dataset": "min-synthetic", "split": split})
        manifest = tmp_path / f"{split}.csv"
        save_manifest(pd.DataFrame(rows), manifest, {})
        data[split + "_manifest"], data[split + "_root"] = str(manifest), str(root)
    return {"task": "ae", "name": "reconstruction-smoke", "seed": 42, "output_dir": str(tmp_path / "ae"), "data": data,
            "model": {"ae": {"kind": "cae", "image_size": 32, "width": 4, "latent_dim": 8}},
            "training": {"epochs": 1, "batch_size": 2, "workers": 0, "grad_accum_steps": 2, "recipe": "none", "loss": {"lambda_ssim": 0.0}}}


@pytest.mark.parametrize("kind", ["cae", "vae", "gated"])
def test_fit_every_autoencoder_and_audited_prediction(source_config, kind, tmp_path):
    source_config["model"]["ae"]["kind"] = kind
    if kind == "vae":
        source_config["training"]["loss"]["beta"] = {"kind": "cyclic", "maximum": 1e-4, "cycle_steps": 4}
    result = fit(source_config)
    root = Path(result["run_dir"])
    assert result["epochs_completed"] == 1
    record = json.loads((root / "run.json").read_text())
    assert record["config"]["provenance"]["n_train"] == 3
    assert record["config"]["provenance"]["n_train_fake"] == 0
    assert record["selection_key"] == "loss"
    calibration = json.loads((root / "calibration.json").read_text())
    assert calibration["policy"] == "youden"
    loaded = load_model(root)
    assert loaded(torch.rand(2, 3, 32, 32)).shape == (2, 2)
    frame = predict(root, source_config["data"]["val_manifest"], source_config["data"]["val_root"])
    assert len(frame) == 6
    report = predict(root, source_config["data"]["val_manifest"], source_config["data"]["val_root"], output_dir=tmp_path / "evaluation")
    assert report["frame"]["auc"] is not None
    assert fit(source_config, resume=True)["already_complete"]


@pytest.mark.parametrize("task,frozen", [("residual", True), ("residual", False), ("latent", True), ("latent", False)])
def test_detector_fit_reload_and_real_only_finetuning(source_config, task, frozen, tmp_path):
    source_config["model"]["ae"]["kind"] = "vae" if task == "latent" else "cae"
    fit(source_config)
    cfg = copy.deepcopy(source_config)
    cfg.update(task=task, output_dir=str(tmp_path / "detector"))
    cfg["model"].update(ae_run=source_config["output_dir"], freeze_ae=frozen, width=4, hidden_dim=8, gradient=task == "residual")
    result = fit(cfg)
    assert result["validation"]["auc"] is not None
    record = json.loads((Path(cfg["output_dir"]) / "run.json").read_text())
    assert record["selection_key"] == "auc"
    model = load_model(cfg["output_dir"])
    before = model(torch.rand(2, 3, 32, 32))
    assert torch.isfinite(before).all()
    # Standalone detector checkpoint contains the complete AE and needs no source run.
    Path(source_config["output_dir"]).rename(tmp_path / "hidden-source-ae")
    assert load_model(cfg["output_dir"])(torch.rand(2, 3, 32, 32)).shape == (2, 2)


def test_vae_resume_matches_uninterrupted_with_beta_and_accumulation(source_config, tmp_path, monkeypatch):
    source_config["model"]["ae"]["kind"] = "vae"
    source_config["training"].update(epochs=2, recipe="basic_v1")
    source_config["training"]["loss"]["beta"] = {"kind": "linear", "maximum": 1e-3, "warmup_steps": 3}
    original = runtime.atomic_torch
    def interrupt(path, value):
        original(path, value)
        if Path(path).name == "last.pt" and value["epoch"] == 0:
            raise RuntimeError("planned interruption")
    with monkeypatch.context() as context:
        context.setattr(runtime, "atomic_torch", interrupt)
        with pytest.raises(RuntimeError, match="planned interruption"):
            fit(source_config)
    fit(source_config, resume=True)
    resumed = torch.load(Path(source_config["output_dir"]) / "last.pt", weights_only=True)
    baseline = copy.deepcopy(source_config)
    baseline["output_dir"] = str(tmp_path / "uninterrupted")
    fit(baseline)
    uninterrupted = torch.load(Path(baseline["output_dir"]) / "last.pt", weights_only=True)
    assert resumed["global_step"] == uninterrupted["global_step"] == 2
    assert all(torch.equal(value, uninterrupted["state_dict"][key]) for key, value in resumed["state_dict"].items())
    assert resumed["history"] == uninterrupted["history"]


def test_reject_source_split_abuse_and_incomplete_runs(source_config, tmp_path):
    invalid = copy.deepcopy(source_config)
    invalid["data"]["train_manifest"] = invalid["data"]["val_manifest"]
    with pytest.raises(ValueError, match="source train"):
        fit(invalid)
    invalid = copy.deepcopy(source_config)
    invalid["model"]["ae"]["skip_divisor"] = 2
    with pytest.raises(ValueError, match="Skips"):
        normalize_config(invalid)
    invalid = copy.deepcopy(source_config)
    invalid["task"] = "residual"
    with pytest.raises(ValueError, match="requires ae_run"):
        fit(invalid)
    fit(source_config)
    checkpoint = Path(source_config["output_dir"]) / "best.pt"
    checkpoint.write_bytes(b"modified")
    with pytest.raises(ValueError, match="artifact"):
        load_model(source_config["output_dir"])
