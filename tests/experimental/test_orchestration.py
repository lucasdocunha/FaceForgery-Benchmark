"""Shared CLI boundaries using certified synthetic populations and fake loaders."""

import json
from pathlib import Path
from types import SimpleNamespace

import pytest
import numpy as np
import yaml

from research_cli import main
from src.experimental import orchestration
from src.experimental.configuration import read_document
from src.robustness.inference import calibrate, predict, prediction_contract
from src.robustness.manifests import load_manifest
from src.robustness.provenance import digest, digest_file, write_json
from src.robustness.suite import run_suite_config
from tests.robustness.test_evaluation_suite import BrightnessDetector, four_targets, population
from tests.experimental.test_sbi import sbi_source


def test_json_configuration_preserves_scientific_numeric_types(tmp_path):
    path = tmp_path / "training.json"
    path.write_text(json.dumps({"training": {"lr_backbone": 1e-5, "lr_head": 1e-4,
                                              "amp": True, "optional": None},
                                "output_dir": "run"}))
    config = read_document(path)
    assert config["training"] == {"lr_backbone": 1e-5, "lr_head": 1e-4,
                                  "amp": True, "optional": None}
    assert isinstance(config["training"]["lr_backbone"], float)
    assert config["output_dir"] == str(tmp_path / "run")


def cached_population(cache, manifest, root):
    """Simulated frozen extractor, not a real model or a measured feature result."""
    cache.mkdir()
    frame, certificate = load_manifest(manifest)
    ids = frame.sample_id.astype(str).tolist()
    values = np.random.default_rng(42).normal(0, .05, (len(frame), 8))
    values[:, 0] += frame.label.to_numpy() * 2 - 1
    np.save(cache / "features.npy", values.astype(np.float16))
    np.save(cache / "logits.npy", np.column_stack((-values[:, 0], values[:, 0])).astype(np.float32))
    frame["image_sha256"] = [digest_file(Path(root) / name) for name in frame.img_name]
    frame.to_csv(cache / "rows.csv", index=False)
    extraction = {"mode": "srm", "image_size": 32, "srm_on_normalized": True}
    key = {"checkpoint_sha256": "synthetic-extractor", "run_config_sha256": "synthetic-extractor-config",
           "manifest_sha256": certificate["manifest_sha256"], "extraction": extraction,
           "image_inventory_sha256": digest(frame.image_sha256.tolist()), "commit": "fixture-commit"}
    write_json(cache / "cache.json", {"schema": "faceforgery-features-v1", "status": "complete",
        "key": key, "identity": digest(key), "sample_ids_sha256": digest(ids),
        "checkpoint": {"sha256": "synthetic-extractor"}, "extraction": extraction, "manifest": certificate,
        "files": {name: digest_file(cache / name) for name in ("features.npy", "logits.npy", "rows.csv")}})
    return str(cache)


def suite_configuration(tmp_path, family, run, targets, *, options=None, calibration=None):
    path = tmp_path / (family + "-suite.yaml")
    path.write_text(yaml.safe_dump({"model": {"type": "experimental", "family": family, "path": str(run),
                                               **({"options": options} if options else {})},
        "calibration": str(calibration or Path(run) / "calibration.json"), "scope": "synthetic",
        "output": str(tmp_path / (family + "-suite-output")),
        "targets": [{"name": target.name, "manifest": str(target.manifest), "root": str(target.root),
                     "primary_unit": target.primary_unit, "breakdown": target.breakdown}
                    for target in targets], "inference": {"bootstrap_draws": 0, "plots": False, "workers": 0}}))
    return path


def fake_adapter(tmp_path, monkeypatch, family, *, image_size=32):
    run = tmp_path / "trained"
    run.mkdir()
    checkpoint = run / "artifact.json"
    checkpoint.write_text(json.dumps({"scope": "synthetic boundary fixture", "family": family}))
    condition = {"model": "synthetic boundary fixture", "family": family}
    contract = {**prediction_contract(32), "image_size": image_size,
                "family": family, "bundle_sha256": digest_file(checkpoint)}
    description = {"checkpoint_path": checkpoint, "input_contract": contract, "image_size": image_size,
                   "research_run": {"name": "fixture", "seed": 42, "condition": condition,
                                    "condition_sha256": digest(condition)}}
    calls = {"loads": [], "validated": []}

    def external(frame, root, **kwargs):
        assert kwargs["positive_class"] == "fake"
        return predict(BrightnessDetector(), frame, root, image_size=32, batch_size=4)

    def callbacks(run_dir, caches, protocol=None):
        result = {}
        for name in caches:
            def check(frame, root, key=name):
                assert len(frame) > 0 and Path(root).is_dir()
                calls["validated"].append(key)

            def callback(frame, root, **kwargs):
                return external(frame, root, **kwargs)

            callback.validate_population = check
            result[name] = callback
        return result

    def loaded_model(*args, **kwargs):
        calls["loads"].append("tensor")
        return BrightnessDetector()

    def loaded_predictor(*args, **kwargs):
        calls["loads"].append("external")
        return external

    def checked_sources(run_dir, sources, frame=None, root=None):
        assert sources and len(frame) > 0 and Path(root).is_dir()
        if isinstance(sources[0], dict) and "role" in sources[0]:
            assert [source["name"] for source in sources] == ["srm", "rgb", "reconstruction"]
            assert [source["kind"] for source in sources] == ["feature_cache", "feature_cache", "predictions"]
            assert all(Path(source["path"]).is_absolute() for source in sources)
        calls["validated"].append("moe")

    module = SimpleNamespace(describe_run=lambda *a, **kw: description,
                             load_model=loaded_model, load_predictor=loaded_predictor,
                             feature_cache_predictor=callbacks, validate_sources=checked_sources)
    monkeypatch.setattr(orchestration, "import_module", lambda name: module)
    return run, description, calls


def validation(tmp_path, description):
    manifest, root = population(tmp_path, "source_val", dataset="source", split="val")
    frame, certificate = load_manifest(manifest)
    output = tmp_path / "source_calibration.json"
    calibrate(predict(BrightnessDetector(), frame, root, image_size=32), manifest_record=certificate,
              output=output, model_sha256=digest_file(description["checkpoint_path"]),
              input_contract=description["input_contract"], policy="youden")
    return manifest, root, output


@pytest.mark.parametrize("family", sorted(orchestration.FAMILIES))
def test_every_server_template_synthetic_dry_run(tmp_path, monkeypatch, family, capsys):
    targets = four_targets(tmp_path)
    run, description, calls = fake_adapter(tmp_path, monkeypatch, family,
                                           image_size=None if family in {"vlm", "moe"} else 32)
    _, _, calibration = validation(tmp_path, description)
    variables = {"TCC_EVAL_MODEL": run, "TCC_EVAL_CALIBRATION": calibration,
                 "TCC_OUTPUT_ROOT": tmp_path / "never-created"}
    prefixes = ["TEST", "TEST_D", "DF40", "CELEB"]
    for target, prefix in zip(targets, prefixes):
        variables[f"TCC_EVAL_{prefix}_MANIFEST"] = target.manifest
        variables[f"TCC_EVAL_{prefix}_ROOT"] = target.root
        variables[f"TCC_EVAL_{prefix}_FEATURE_CACHE"] = tmp_path / (target.name + "-features")
        variables[f"TCC_EVAL_{prefix}_SRM_CACHE"] = tmp_path / (target.name + "-srm")
        variables[f"TCC_EVAL_{prefix}_RGB_CACHE"] = tmp_path / (target.name + "-rgb")
        variables[f"TCC_EVAL_{prefix}_RECON_PREDICTIONS"] = tmp_path / (target.name + "-reconstruction.csv")
    for key, value in variables.items():
        monkeypatch.setenv(key, str(value))
    template = Path(__file__).resolve().parents[2] / "configs/experimental/suites" / (family + ".yaml")
    spec = yaml.safe_load(template.read_text())
    spec["scope"] = "synthetic"
    path = tmp_path / "suite.yaml"
    path.write_text(yaml.safe_dump(spec))
    capsys.readouterr()
    assert main(["experimental", "evaluate", "--config", str(path)]) == 0
    result = json.loads(capsys.readouterr().out)
    assert result["state"] == "planned_not_executed"
    assert len(result["targets"]) == 4 and not calls["loads"]
    assert result["targets"]["df40"]["real_reference_policy"] == "pooled_all_real"
    assert result["targets"]["celeb_df_v2"]["primary_unit"] == "video"
    assert not (tmp_path / "never-created").exists()
    assert len(calls["validated"]) == (4 if family in {"metric", "graph", "moe"} else 0)


@pytest.mark.parametrize("family", ["reconstruction", "vlm", "metric", "moe"])
def test_input_bound_calibration_cli_for_tensor_and_external_scorers(tmp_path, monkeypatch, family, capsys):
    run, description, calls = fake_adapter(tmp_path, monkeypatch, family,
                                           image_size=None if family in {"vlm", "moe"} else 32)
    manifest, root, _ = validation(tmp_path, description)
    output = tmp_path / "new-calibration"
    arguments = ["experimental", "calibrate", "--family", family, "--run", str(run),
                 "--manifest", str(manifest), "--root", str(root), "--output", str(output)]
    if family in {"metric", "moe"}:
        options = tmp_path / "options.yaml"
        options.write_text(yaml.safe_dump({"cache": "cache"} if family == "metric"
                                         else {"sources": [
                                             {"name": "srm", "role": "srm", "kind": "feature_cache", "path": "srm-cache"},
                                             {"name": "rgb", "role": "rgb", "kind": "feature_cache", "path": "rgb-cache"},
                                             {"name": "reconstruction", "role": "reconstruction", "kind": "predictions", "path": "scores.csv"},
                                         ]}))
        arguments += ["--options", str(options)]
    assert main(arguments) == 0 and not output.exists() and not calls["loads"]
    capsys.readouterr()
    assert main(arguments + ["--execute"]) == 0
    result = json.loads(capsys.readouterr().out)
    assert result["selection_split"] == "val"
    assert result["input_contract_sha256"] == digest(description["input_contract"])
    assert result["model_sha256"] == digest_file(description["checkpoint_path"])
    assert json.loads((output / "status.json").read_text())["state"] == "complete"


def test_calibration_rejects_fusion_fit_population_before_loading(tmp_path, monkeypatch):
    run, description, calls = fake_adapter(tmp_path, monkeypatch, "moe", image_size=None)
    manifest, root, _ = validation(tmp_path, description)
    description["calibration_manifest_sha256"] = "a" * 64
    with pytest.raises(ValueError, match="held-out source selection population"):
        orchestration.calibrate_experiment("moe", run, manifest, root, tmp_path / "never-written",
                                            options={"sources": ["fixture.csv"]}, execute=True)
    assert not calls["loads"] and not (tmp_path / "never-written").exists()


def test_source_metadata_and_cache_path_shorthands_keep_distinct_semantics(tmp_path):
    path = tmp_path / "options.yaml"
    path.write_text(yaml.safe_dump({
        "sources": [{"name": "dino_srm", "role": "srm", "kind": "feature_cache", "path": "dino/val"}],
        "target_caches": {"test": "test-cache", "test_d": "degraded-cache"},
    }))
    resolved = read_document(path)
    assert resolved["sources"] == [{"name": "dino_srm", "role": "srm", "kind": "feature_cache",
                                    "path": str(tmp_path / "dino/val")}]
    assert resolved["target_caches"] == {"test": str(tmp_path / "test-cache"),
                                          "test_d": str(tmp_path / "degraded-cache")}


def test_portable_training_seed_override_and_unset_variable_rejection(tmp_path, monkeypatch, capsys):
    monkeypatch.setenv("TCC_TRAIN_MANIFEST", str(tmp_path / "source.csv"))
    path = tmp_path / "train.json"
    path.write_text(json.dumps({"family": "reconstruction", "run_dir": "runs/seed7", "seed": 42,
                                "data": {"train_manifest": "${TCC_TRAIN_MANIFEST}"}, "model": {}, "training": {}}))
    target_run = tmp_path / "override" / "seed7"
    assert main(["experimental", "train", "--config", str(path), "--seed", "7", "--output", str(target_run)]) == 0
    result = json.loads(capsys.readouterr().out)
    assert result["state"] == "planned_not_executed" and result["config"]["seed"] == 7
    assert result["run_dir"] == str(target_run)
    assert result["config"]["data"]["train_manifest"] == str(tmp_path / "source.csv")
    monkeypatch.delenv("TCC_TRAIN_MANIFEST")
    with pytest.raises(ValueError, match="Unset environment"):
        read_document(path)


def test_bad_calibration_never_loads_experimental_model(tmp_path, monkeypatch):
    targets = four_targets(tmp_path)
    run, description, calls = fake_adapter(tmp_path, monkeypatch, "reconstruction")
    _, _, calibration = validation(tmp_path, description)
    value = json.loads(calibration.read_text())
    value["frame_threshold"] = 1.5
    calibration.write_text(json.dumps(value))
    path = tmp_path / "suite.yaml"
    path.write_text(yaml.safe_dump({"model": {"type": "experimental", "family": "reconstruction", "path": str(run)},
                                    "calibration": str(calibration), "scope": "synthetic", "output": str(tmp_path / "never-created"),
                                    "targets": [{"name": target.name, "manifest": str(target.manifest), "root": str(target.root),
                                                 "primary_unit": target.primary_unit, "breakdown": target.breakdown}
                                                for target in targets]}))
    with pytest.raises(ValueError, match="finite frozen validation threshold"):
        run_suite_config(path, execute=True)
    assert not calls["loads"] and not (tmp_path / "never-created").exists()


def test_trained_sbi_run_shared_cli_and_all_four_targets(tmp_path, sbi_source, capsys):
    config = tmp_path / "sbi.yaml"
    config.write_text(yaml.safe_dump(sbi_source))
    assert main(["experimental", "train", "--family", "sbi", "--config", str(config), "--execute"]) == 0
    training = json.loads(capsys.readouterr().out)
    run = Path(training["run_dir"])
    targets = four_targets(tmp_path)
    calibration_dir = tmp_path / "source-calibration"
    calibration = orchestration.calibrate_experiment("sbi", run, sbi_source["data"]["val_manifest"],
        sbi_source["data"]["val_root"], calibration_dir, batch_size=4, execute=True)
    assert "bundle_sha256" in calibration["input_contract"]
    path = suite_configuration(tmp_path, "sbi", run, targets, calibration=calibration_dir / "calibration.json")
    result = run_suite_config(path, execute=True)
    assert result["state"] == "complete" and len(result["results"]) == 4
    assert result["results"]["celeb_df_v2"]["primary_unit"] == "video"
