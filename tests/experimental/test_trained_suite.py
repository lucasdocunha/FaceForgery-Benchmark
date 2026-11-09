"""Actual trained artifacts through shared calibration and synthetic targets."""

import copy
import json
from pathlib import Path

import numpy as np
from PIL import Image
import pytest
import yaml

from src.experimental import orchestration
from src.experimental.features import open_cache
from src.robustness.artifacts import save_predictions
from src.robustness.manifests import load_manifest
from src.robustness.provenance import digest, digest_file, write_json
from src.robustness.suite import run_suite_config
from tests.experimental.test_orchestration import cached_population, suite_configuration
from tests.experimental.test_moe import fixture_experts
from tests.experimental.test_reconstruction_training import source_config
from tests.experimental.test_vlm import tiny_vlm
from tests.robustness.test_evaluation_suite import four_targets, population


@pytest.mark.parametrize("family,kind", [("metric", "centroid"), ("graph", "gcn")])
def test_trained_cached_scorer_recalibration_and_all_four_targets(tmp_path, family, kind):
    train_manifest, train_root = population(tmp_path, "training", dataset="source", split="train")
    val_manifest, val_root = population(tmp_path, "validation", dataset="source-val", split="val")
    # Distinct image content is required even when logical IDs are already disjoint.
    for path in Path(val_root).glob("*.png"):
        with Image.open(path) as image:
            pixels = np.array(image)
        pixels[0, 0, 0] ^= 1
        Image.fromarray(pixels).save(path)
    train_cache = cached_population(tmp_path / "train-cache", train_manifest, train_root)
    val_cache = cached_population(tmp_path / "val-cache", val_manifest, val_root)
    targets = four_targets(tmp_path)
    caches = {target.name: cached_population(tmp_path / (target.name + "-cache"), target.manifest, target.root)
              for target in targets}
    run = tmp_path / "trained-cached"
    config = tmp_path / "train.yaml"
    config.write_text(yaml.safe_dump({"family": family, "name": "synthetic-cache-roundtrip", "seed": 42,
        "run_dir": str(run), "caches": {"train": train_cache, "val": val_cache},
        "model": {"kind": kind, "label_fraction": .5, "hidden_dim": 8, "projection_dim": 8,
                  "backend": "native", "graph": {"k": 2, "backend": "exact", "iterations": 3}, "fanouts": [2, 2]},
        "training": {"epochs": 1, "batch_size": 4, "baseline_epochs": 1, "cpu_threads": 2}}))
    assert orchestration.train_experiment(config, execute=True)["state"] == "complete"
    calibration_dir = tmp_path / "recalibration"
    calibration = orchestration.calibrate_experiment(family, run, val_manifest, val_root, calibration_dir,
                                                      options={"cache": val_cache}, execute=True)
    assert calibration["input_contract_sha256"] == digest(orchestration.describe_run(family, run)["input_contract"])
    path = suite_configuration(tmp_path, family, run, targets, options={"target_caches": caches},
                               calibration=calibration_dir / "calibration.json")
    assert run_suite_config(path)["state"] == "planned_not_executed"
    result = run_suite_config(path, execute=True)
    assert result["state"] == "complete" and len(result["results"]) == 4
    assert result["results"]["celeb_df_v2"]["primary_unit"] == "video"
    assert result["results"]["celeb_df_v2"]["primary"]["n"] == 8
    assert "per_paradigm" in result["results"]["df40"]


@pytest.mark.parametrize("task,kind", [("ae", "cae"), ("ae", "vae"), ("ae", "gated"),
                                       ("residual", "cae"), ("latent", "vae")])
def test_trained_reconstruction_variants_all_four_targets(tmp_path, source_config, task, kind):
    cfg = copy.deepcopy(source_config)
    cfg["model"]["ae"]["kind"] = kind
    config = tmp_path / "train.yaml"
    config.write_text(yaml.safe_dump(cfg))
    orchestration.train_experiment(config, family="reconstruction", execute=True)
    if task != "ae":
        cfg.update(task=task, output_dir=str(tmp_path / "detector"))
        cfg["model"].update(ae_run=source_config["output_dir"], width=4, hidden_dim=8)
        config.write_text(yaml.safe_dump(cfg))
        orchestration.train_experiment(config, family="reconstruction", execute=True)
    run = Path(cfg["output_dir"])
    calibration_dir = tmp_path / "recalibration"
    orchestration.calibrate_experiment("reconstruction", run, cfg["data"]["val_manifest"],
        cfg["data"]["val_root"], calibration_dir, batch_size=4, execute=True)
    targets = four_targets(tmp_path)
    path = suite_configuration(tmp_path, "reconstruction", run, targets, calibration=calibration_dir / "calibration.json")
    assert run_suite_config(path)["state"] == "planned_not_executed"
    result = run_suite_config(path, execute=True)
    assert result["state"] == "complete" and len(result["results"]) == 4
    assert result["results"]["celeb_df_v2"]["primary_unit"] == "video"
    # Published target scores are reusable frozen MoE sources without sidecar editing.
    from src.experimental.moe.data import open_expert
    expert = open_expert({"name": "reconstruction", "role": "reconstruction", "kind": "predictions",
                         "path": str(tmp_path / "reconstruction-suite-output/test/predictions.csv")})
    assert len(expert.frame) == 16


def test_trained_spatial_latent_mean_all_four_targets_without_component_runs(tmp_path, source_config):
    cfg = copy.deepcopy(source_config)
    cfg["model"]["ae"]["kind"] = "vae"
    config = tmp_path / "train.yaml"
    config.write_text(yaml.safe_dump(cfg))
    orchestration.train_experiment(config, family="reconstruction", execute=True)
    component_runs = {}
    for task in ("residual", "latent"):
        detector = copy.deepcopy(cfg)
        detector.update(task=task, output_dir=str(tmp_path / task))
        detector["model"].update(ae_run=cfg["output_dir"], width=4, hidden_dim=8)
        config.write_text(yaml.safe_dump(detector))
        orchestration.train_experiment(config, family="reconstruction", execute=True)
        component_runs[task] = detector["output_dir"]
    run = tmp_path / "mean-run"
    config.write_text(yaml.safe_dump({"task": "mean_ensemble", "name": "fixed-mean-roundtrip", "seed": 42,
        "output_dir": str(run), "model": {"spatial_run": component_runs["residual"],
                                           "latent_run": component_runs["latent"]}}))
    assert orchestration.train_experiment(config, family="reconstruction", execute=True)["state"] == "complete"
    for path in (cfg["output_dir"], *component_runs.values()):
        original = Path(path)
        original.rename(original.with_name("hidden-" + original.name))
    calibration = tmp_path / "mean-recalibration"
    orchestration.calibrate_experiment("reconstruction", run, cfg["data"]["val_manifest"],
        cfg["data"]["val_root"], calibration, batch_size=4, execute=True)
    path = suite_configuration(tmp_path, "reconstruction", run, four_targets(tmp_path),
                               calibration=calibration / "calibration.json")
    assert run_suite_config(path)["state"] == "planned_not_executed"
    result = run_suite_config(path, execute=True)
    assert result["state"] == "complete" and len(result["results"]) == 4
    assert result["results"]["celeb_df_v2"]["primary"]["n"] == 8


def test_actual_tiny_vlm_train_recalibrate_and_all_four_targets(tmp_path, tiny_vlm, monkeypatch):
    _, _, model_config = tiny_vlm
    monkeypatch.setenv("TCC_PRETRAINED_ROOT", str(Path(model_config["pretrained_path"]).parent))
    data = {}
    for split in ("train", "val"):
        manifest, root = population(tmp_path, "source-" + split, dataset="source-" + split, split=split)
        data[split + "_manifest"], data[split + "_root"] = str(manifest), str(root)
    data.update(train_limit=2, val_limit=2)
    run = tmp_path / "vlm-run"
    config = tmp_path / "vlm-train.yaml"
    config.write_text(yaml.safe_dump({"family": "vlm", "task": "vlm", "name": "tiny-vlm-roundtrip", "seed": 42,
        "scope": "synthetic-structural", "run_dir": str(run), "data": data, "model": model_config,
        "training": {"epochs": 1, "batch_size": 1, "grad_accum_steps": 2, "lr": .01, "amp": False}}))
    assert orchestration.train_experiment(config, execute=True)["state"] == "complete"
    calibration_dir = tmp_path / "vlm-recalibration"
    orchestration.calibrate_experiment("vlm", run, data["val_manifest"], data["val_root"],
                                      calibration_dir, batch_size=2, use_amp=False, execute=True)
    targets = four_targets(tmp_path)
    path = suite_configuration(tmp_path, "vlm", run, targets, calibration=calibration_dir / "calibration.json")
    spec = yaml.safe_load(path.read_text())
    spec["inference"].update(batch_size=2, use_amp=False)
    path.write_text(yaml.safe_dump(spec))
    assert run_suite_config(path)["state"] == "planned_not_executed"
    result = run_suite_config(path, execute=True)
    assert result["state"] == "complete" and len(result["results"]) == 4
    assert result["results"]["celeb_df_v2"]["primary"]["n"] == 8


def target_experts(root, target, source_specs):
    """Simulated scores from fixed expert identities for an independent fixture."""
    root.mkdir()
    frame, certificate = load_manifest(target.manifest)
    frame["image_sha256"] = [digest_file(Path(target.root) / name) for name in frame.img_name]
    specs = []
    for source in source_specs:
        spec = copy.deepcopy(source)
        if source["kind"] == "feature_cache":
            original = open_cache(source["path"])
            path = root / source["name"]
            path.mkdir()
            features = np.random.default_rng(7).normal(0, .1, (len(frame), original.features.shape[1])).astype(np.float16)
            differences = frame.label.to_numpy() * 1.2 - .6
            np.save(path / "features.npy", features)
            np.save(path / "logits.npy", np.column_stack((-differences / 2, differences / 2)).astype(np.float32))
            frame.to_csv(path / "rows.csv", index=False)
            metadata = copy.deepcopy(original.metadata)
            metadata["key"]["manifest_sha256"] = certificate["manifest_sha256"]
            metadata.update(identity=digest(metadata["key"]), manifest=certificate,
                sample_ids_sha256=digest(frame.sample_id.astype(str).tolist()),
                files={name: digest_file(path / name) for name in ("features.npy", "logits.npy", "rows.csv")})
            write_json(path / "cache.json", metadata)
        else:
            path = root / "reconstruction.csv"
            sidecar = json.loads(Path(source["path"] + ".json").read_text())
            predictions = frame.assign(p_fake=.15 + .7 * frame.label, residual_score=np.linspace(0, 1, len(frame)))
            save_predictions(path, predictions, manifest_record=certificate, model_sha256=sidecar["model_sha256"],
                             metadata={key: sidecar[key] for key in ("input_contract", "input_contract_sha256")})
        spec["path"] = str(path)
        specs.append(spec)
    return specs


def test_trained_router_and_fusion_baselines_all_four_targets(tmp_path):
    _, source_specs, source_root = fixture_experts(tmp_path)
    run = tmp_path / "moe-run"
    config = tmp_path / "moe-train.yaml"
    config.write_text(yaml.safe_dump({"family": "moe", "task": "moe", "name": "moe-roundtrip", "seed": 42,
        "scope": "synthetic-structural", "run_dir": str(run), "data": {"experts": source_specs},
        "model": {"hidden_dim": 8, "train_routing": "top_k", "train_top_k": 2, "inference_routing": "soft"},
        "training": {"epochs": 1, "batch_size": 8, "amp": False, "lr": .01}}))
    assert orchestration.train_experiment(config, execute=True)["state"] == "complete"
    source_calibration = tmp_path / "moe-recalibration"
    orchestration.calibrate_experiment("moe", run, run / "val_select.csv", source_root, source_calibration,
                                      options={"sources": source_specs}, execute=True)
    targets = four_targets(tmp_path)
    caches = {target.name: target_experts(tmp_path / (target.name + "-experts"), target, source_specs)
              for target in targets}
    for method in ("router", "mean", "geometric", "logistic", "expert_srm"):
        calibration = (source_calibration / "calibration.json" if method == "router"
                       else run / "comparisons" / method / "calibration.json")
        path = suite_configuration(tmp_path, "moe", run, targets,
                                   options={"target_caches": caches, "method": method}, calibration=calibration)
        spec = yaml.safe_load(path.read_text())
        spec["output"] = str(tmp_path / (method + "-suite"))
        path.write_text(yaml.safe_dump(spec))
        assert run_suite_config(path)["state"] == "planned_not_executed"
        result = run_suite_config(path, execute=True)
        assert result["state"] == "complete" and len(result["results"]) == 4
        assert result["results"]["celeb_df_v2"]["primary"]["n"] == 8
