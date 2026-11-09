"""Tiny generated-data tests of interruption/resume and controlled plan identity."""

import copy
import json
from argparse import Namespace
from pathlib import Path
import pytest
import torch
from src.robustness import engine
from src.robustness.commands import expand, profile
from src.robustness.experiments import read_config, plan
from src.robustness.identity import evaluation_identity
from .test_workflow import fixture_data, config_file


def test_completed_epoch_resume_matches_uninterrupted_weights(tmp_path, monkeypatch):
    torch.set_num_threads(1)
    paths = fixture_data(tmp_path)
    cfg = read_config(config_file(tmp_path, paths))
    cfg["training"].update(epochs=2, patience=3)
    run_dir = Path(plan(cfg)["run_dir"])
    original = engine.atomic_torch

    def interrupt_after_saved_epoch(path, value):
        original(path, value)
        if Path(path).name == "last.pt" and value["epoch"] == 0:
            raise RuntimeError("simulated interruption after a complete epoch")

    with monkeypatch.context() as m:
        m.setattr(engine, "atomic_torch", interrupt_after_saved_epoch)
        with pytest.raises(RuntimeError, match="simulated interruption"):
            engine.train(cfg)
    assert (run_dir / "last.pt").exists() and not (run_dir / ".running.lock").exists()
    resumed = engine.train(cfg, resume=True)
    assert resumed["state"] == "complete" and resumed["epochs_completed"] == 2
    baseline = copy.deepcopy(cfg)
    baseline["output_root"] = str(tmp_path / "uninterrupted")
    baseline_dir = Path(plan(baseline)["run_dir"])
    engine.train(baseline)
    a = torch.load(run_dir / "last.pt", map_location="cpu", weights_only=True)[
        "state_dict"
    ]
    b = torch.load(baseline_dir / "last.pt", map_location="cpu", weights_only=True)[
        "state_dict"
    ]
    assert a.keys() == b.keys()
    assert all(torch.equal(a[key], b[key]) for key in a)
    measured = profile(
        Namespace(
            run=run_dir,
            output=tmp_path / "timing.json",
            device="cpu",
            batch_size=1,
            warmup=1,
            iterations=3,
        )
    )
    assert measured["median_batch_ms"] > 0 and "EXCLUDES" in measured["scope"]


def test_ablation_expansion_keeps_nonexperimental_settings(tmp_path):
    paths = fixture_data(tmp_path)
    template = config_file(tmp_path, paths)
    result = expand(
        Namespace(
            template=template,
            plan="configs/research/ablation-plan.json",
            output=tmp_path / "plan",
            variants=["rgb_basic", "rgb_robust"],
            seeds=[42, 123],
        )
    )
    assert result["state"] == "planned_not_executed" and len(result["jobs"]) == 4
    configs = [read_config(row["config"]) for row in result["jobs"]]
    assert {c["training"]["seed"] for c in configs} == {42, 123}
    assert all(c["training"]["epochs"] == 1 for c in configs)
    assert all(not c["training"]["paired_training"] for c in configs)
    assert {c["training"]["recipe"] for c in configs} == {"basic_v1", "robust_v1"}


def test_seed_identity_excludes_only_seed_and_nonscientific_paths(tmp_path):
    paths = fixture_data(tmp_path)
    cfg = read_config(config_file(tmp_path, paths))
    a = plan(cfg)
    other = copy.deepcopy(cfg)
    other["training"]["seed"] = 123
    other["output_root"] = "elsewhere"
    b = plan(other)
    assert (
        evaluation_identity(a)["condition_sha256"]
        == evaluation_identity(b)["condition_sha256"]
    )
    other["training"]["lr"] *= 2
    c = plan(other)
    assert (
        evaluation_identity(a)["condition_sha256"]
        != evaluation_identity(c)["condition_sha256"]
    )


def test_model_parameters_rejected_during_planning(tmp_path):
    import yaml

    paths = fixture_data(tmp_path)
    file = config_file(tmp_path, paths)
    cfg = yaml.safe_load(file.read_text())
    cfg["model"]["width"] = 7
    file.write_text(yaml.safe_dump(cfg))
    with pytest.raises(ValueError, match="width"):
        read_config(file)
