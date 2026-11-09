"""Complete reconstruction artifacts, exact epoch resumes and audited inference."""

import copy
import json
from pathlib import Path
import shutil

import numpy as np
import pandas as pd
from PIL import Image
import pytest
import torch
import yaml

from src.experimental.reconstruction import combine, describe_run, fit, load_model, normalize_config, predict
from src.robustness.manifests import save_manifest
from src.robustness.provenance import digest_file


@pytest.fixture(autouse=True)
def one_thread():
    previous = torch.get_num_threads()
    torch.set_num_threads(1)
    yield
    torch.set_num_threads(previous)


@pytest.fixture
def population(tmp_path):
    paths = {}
    for split in ("train", "val", "test"):
        root = tmp_path / split
        root.mkdir()
        rows = []
        for i in range(8):
            path = root / f"{i}.png"
            rng = np.random.default_rng(100 * (1 + len(paths)) + i)
            Image.fromarray(rng.integers(0, 256, (32, 32, 3), dtype=np.uint8)).save(path)
            rows.append(dict(img_name=path.name, label=i % 2, sample_id=f"{split}:{i}", group_id=f"{split}:g{i}", dataset="min-synthetic", split=split, sha256=digest_file(path)))
        manifest = tmp_path / f"{split}.csv"
        save_manifest(pd.DataFrame(rows), manifest, {})
        paths[split] = (manifest, root)
    return paths


def configuration(tmp_path, population, *, kind="cae", task="ae", name="run"):
    return {"task": task, "name": name, "seed": 42, "output_dir": str(tmp_path / name),
            "data": {f"{split}_{what}": str(population[split][index]) for split in ("train", "val") for index, what in enumerate(("manifest", "root"))},
            "model": {"ae": {"kind": kind, "image_size": 32, "width": 4, "latent_dim": 8}, "width": 4, "dropout": 0.1},
            "training": {"epochs": 1, "batch_size": 3, "grad_accum_steps": 2, "amp": False, "loss": {"lambda_ssim": 0.1}}}


@pytest.mark.parametrize("kind", ["cae", "vae", "gated"])
def test_genuine_pretraining_roundtrip_and_certified_inference(tmp_path, population, kind):
    cfg = configuration(tmp_path, population, kind=kind)
    result = fit(cfg)
    run = Path(cfg["output_dir"])
    record = json.loads((run / "run.json").read_text())
    assert record["config"]["provenance"]["n_train"] == 4
    assert record["config"]["provenance"]["n_train_fake"] == 0
    assert result["validation"]["n"] == 8
    assert fit(cfg, resume=True)["already_complete"]
    first = predict(run, *population["val"])
    second = predict(run, *population["val"])
    assert np.array_equal(first.p_fake, second.p_fake)
    info = describe_run(run)
    assert info["bundle_sha256"] == info["input_contract"]["bundle_sha256"]
    report = predict(run, *population["test"], output_dir=tmp_path / "evaluation", bootstrap_draws=20)
    assert report["frame"]["n"] == 8 and report["frame"]["auc_interval"]
    sidecar = json.loads((run / "validation_predictions.csv.json").read_text())
    assert sidecar["input_contract"] == info["input_contract"]
    if kind == "cae":
        from src.experimental.reconstruction.pilot import summarize
        reference_root = tmp_path / "hf"
        (reference_root / "reference").mkdir(parents=True)
        reference = first[["img_name", "label", "p_fake"]].rename(columns={"img_name": "sample_id"})
        reference["group_id"] = reference.sample_id
        reference.to_csv(reference_root / "reference" / "predictions_val.csv", index=False)
        # Report root must contain runs, not unrelated evaluation directories.
        campaign = tmp_path / "campaign"
        campaign.mkdir()
        (campaign / "run").symlink_to(run, target_is_directory=True)
        pilot = summarize(campaign, reference_root=reference_root, draws=20)
        assert len(pilot["metrics"]) == len(pilot["correlations"]) == 1
        assert pilot["correlations"][0]["pearson"] == pytest.approx(1)
    with pytest.raises(FileExistsError):
        fit(cfg)
    (run / "best.pt").write_bytes(b"changed")
    with pytest.raises(ValueError, match="modified"):
        describe_run(run)


def test_all_detector_modes_and_self_contained_spatial_latent_mean(tmp_path, population):
    ae_cfg = configuration(tmp_path, population, kind="vae", name="autoencoder")
    fit(ae_cfg)
    initial = load_model(ae_cfg["output_dir"]).autoencoder.state_dict()
    runs = {}
    for task, mode in (("residual", "frozen"), ("residual", "recon_finetune"), ("residual", "end_to_end"), ("latent", "frozen"), ("latent", "end_to_end")):
        name = task + "-" + mode
        cfg = configuration(tmp_path, population, kind="vae", task=task, name=name)
        cfg["model"].update(ae_run=ae_cfg["output_dir"], ae_mode=mode, gradient=True)
        cfg["training"].update(lr_ae=1e-5, lr_backbone=1e-4, lr_head=1e-3)
        fit(cfg)
        loaded = load_model(cfg["output_dir"])
        changed = any(not torch.equal(value, loaded.autoencoder.state_dict()[key]) for key, value in initial.items())
        assert changed == (mode != "frozen")
        assert torch.isfinite(loaded(torch.rand(2, 3, 32, 32))).all()
        runs[name] = cfg["output_dir"]
    spatial, latent = runs["residual-frozen"], runs["latent-frozen"]
    p = predict(spatial, *population["val"])
    q = predict(latent, *population["val"])
    combined = tmp_path / "combined"
    mean_config = {"task": "mean_ensemble", "name": "spatial-latent-mean", "seed": 42,
                   "output_dir": str(combined), "model": {"spatial_run": spatial, "latent_run": latent}}
    fit(mean_config)
    assert fit(mean_config, resume=True)["already_complete"]
    for run in [ae_cfg["output_dir"], *runs.values()]:
        shutil.rmtree(run)
    actual = predict(combined, *population["val"])
    np.testing.assert_allclose(actual.p_fake, (p.p_fake + q.p_fake) / 2, rtol=1e-6, atol=1e-7)
    assert describe_run(combined)["research_run"]["condition"]["task"] == "mean_ensemble"
    report = predict(combined, *population["test"], output_dir=tmp_path / "combined-eval")
    assert report["frame"]["n"] == 8


@pytest.mark.parametrize("schedule", ["linear", "cyclic"])
def test_vae_beta_and_randomness_resume_exactly_after_completed_epoch(tmp_path, population, monkeypatch, schedule):
    import src.experimental.runtime as runtime
    full = configuration(tmp_path, population, kind="vae", name="continuous")
    full["training"].update(epochs=3, batch_size=2, grad_accum_steps=2, early_stop_patience=5, recipe="robust_v1")
    full["training"]["loss"].update(kl_reduction="mean_per_dim", beta={"kind": schedule, "maximum": 0.01, "warmup_steps": 2, "cycle_steps": 2})
    fit(full)
    interrupted = copy.deepcopy(full)
    interrupted["output_dir"] = str(tmp_path / "resumed")
    original = runtime.fit_model

    def fail_after_first_epoch(*args, **kwargs):
        validation, calls = kwargs["validate"], 0

        def validate(model):
            nonlocal calls
            calls += 1
            if calls == 2:
                raise RuntimeError("synthetic interruption after one checkpoint")
            return validation(model)

        return original(*args, **{**kwargs, "validate": validate})

    monkeypatch.setattr(runtime, "fit_model", fail_after_first_epoch)
    with pytest.raises(RuntimeError, match="synthetic interruption"):
        fit(interrupted)
    monkeypatch.setattr(runtime, "fit_model", original)
    fit(interrupted, resume=True)
    a = torch.load(Path(full["output_dir"]) / "last.pt", weights_only=True)
    b = torch.load(Path(interrupted["output_dir"]) / "last.pt", weights_only=True)
    assert a["global_step"] == b["global_step"] == 3
    assert a["history"] == b["history"]
    for key, value in a["state_dict"].items():
        torch.testing.assert_close(value, b["state_dict"][key], rtol=0, atol=0)
    interrupted["training"]["lr_ae"] = 0.9
    with pytest.raises(ValueError, match="Resume configuration"):
        fit(interrupted, resume=True)


def test_config_rejects_inconsistent_modes_and_nongenuine_sources(tmp_path, population):
    cfg = configuration(tmp_path, population, task="residual")
    cfg["model"].update(freeze_ae=True, ae_mode="end_to_end")
    with pytest.raises(ValueError, match="conflicts"):
        normalize_config(cfg)
    cfg["model"].pop("freeze_ae")
    cfg["training"]["reconstruction_weight"] = 0
    with pytest.raises(ValueError, match="genuine-only reconstruction"):
        normalize_config(cfg)
    cfg = configuration(tmp_path, population)
    cfg["data"]["val_manifest"] = str(population["test"][0])
    with pytest.raises(ValueError, match="source train and validation"):
        fit(cfg)


def test_server_configs_preserve_benchmark_resolution_and_explicit_kl_scale():
    folder = Path(__file__).resolve().parents[2] / "configs" / "experimental" / "reconstruction"
    files = sorted(folder.glob("*.yaml"))
    assert len(files) >= 10
    for path in files:
        cfg = normalize_config(yaml.safe_load(path.read_text()))
        if cfg["task"] == "mean_ensemble":
            continue
        if path.name.startswith("server_"):
            assert cfg["model"]["ae"]["image_size"] == 224
        assert cfg["training"]["loss"]["kl_reduction"] == "mean_per_dim"


def test_pilot_comparison_has_matching_architecture_init_and_budget(tmp_path, population):
    from argparse import Namespace
    from src.experimental.reconstruction.pilot import campaign_configs
    args = Namespace(train_manifest=population["train"][0], val_manifest=population["val"][0], train_root=population["train"][1], val_root=population["val"][1],
                     output_dir=tmp_path / "campaign", seed=42, batch_size=2, accumulation=1, image_size=32, width=4, latent_dim=8, ae_epochs=3,
                     detector_epochs=2, workers=0, lpips_state=None, resnet_weights=tmp_path / "resnet.pth")
    configs = campaign_configs(args)
    arms = [copy.deepcopy(configs["residual-" + mode]) for mode in ("x_only", "residual_only", "full")]
    for cfg in arms:
        cfg.pop("name")
        cfg.pop("output_dir")
        cfg["model"].pop("input_mode")
    assert arms[0] == arms[1] == arms[2]


def test_reconstruction_finetune_does_not_apply_adamw_to_all_fake_window(tmp_path, population, monkeypatch):
    import src.experimental.runtime as runtime
    ae_cfg = configuration(tmp_path, population)
    fit(ae_cfg)
    cfg = configuration(tmp_path, population, task="residual", name="finetune")
    cfg["model"].update(ae_run=ae_cfg["output_dir"], ae_mode="recon_finetune")

    def inspect_window(model, loader, optimizer, **kwargs):
        before = {name: value.clone() for name, value in model.autoencoder.state_dict().items()}
        batch = {"image": torch.rand(2, 3, 32, 32), "clean": torch.rand(2, 3, 32, 32), "label": torch.ones(2, dtype=torch.long)}
        loss, _, _ = kwargs["loss_step"](model, batch, {"epoch": 0, "global_step": 0})
        loss.backward()
        assert all(parameter.grad is None for parameter in model.autoencoder.parameters())
        optimizer.step()
        assert all(torch.equal(value, model.autoencoder.state_dict()[name]) for name, value in before.items())
        raise RuntimeError("inspection complete")

    monkeypatch.setattr(runtime, "fit_model", inspect_window)
    with pytest.raises(RuntimeError, match="inspection complete"):
        fit(cfg)
