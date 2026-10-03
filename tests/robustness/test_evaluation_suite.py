"""Generated images and reviewed synthetic manifests, never benchmark results."""

from argparse import Namespace
import json
from pathlib import Path

import numpy as np
import pandas as pd
from PIL import Image
import pytest
import torch
import yaml

from src.robustness.artifacts import load_predictions
from src.robustness.commands import aggregate_seeds
from src.robustness.df40 import main as prepare_df40_cli
from src.robustness.inference import calibrate, evaluate_predictions, predict, prediction_contract
from src.robustness.manifests import load_manifest, save_manifest
from src.robustness.provenance import SCHEMA, digest, digest_file
from src.robustness.statistics import aggregate_videos
from src.robustness.suite import TargetSpec, evaluate_suite, legacy_contract, run_suite_config


class BrightnessDetector(torch.nn.Module):
    def forward(self, image):
        score = image.mean((1, 2, 3)) * 4 - 2
        return torch.stack([torch.zeros_like(score), score], dim=1)


def population(tmp_path, name, *, dataset="synthetic", split="smoke_test", video=False):
    root = tmp_path / name
    root.mkdir()
    rows = []
    for index in range(16):
        label = index % 2
        filename = f"{index}.png"
        Image.fromarray(np.full((25, 31, 3), 40 + label * 150 + index, dtype=np.uint8)).save(root / filename)
        row = {"img_name": filename, "label": label, "sample_id": f"{dataset}:{index}",
               "group_id": f"{dataset}:g{index}", "dataset": dataset, "split": split}
        if video:
            # Eight videos, unequal frame counts, with one consistent label/group.
            pair = index // 2
            video_number = (pair // 2) * 2 + label
            row.update(video_id=f"v{video_number}", group_id=f"v{video_number}")
        rows.append(row)
    if video:
        rows.pop()
        (root / "15.png").unlink()
    frame = pd.DataFrame(rows)
    manifest = tmp_path / (name + ".csv")
    save_manifest(frame, manifest, {"group_unit": "video" if video else "image-only synthetic"})
    return manifest, root


def calibrated_fixture(tmp_path, contract=None):
    manifest, root = population(tmp_path, "source_val", dataset="source", split="val")
    frame, certificate = load_manifest(manifest)
    model = BrightnessDetector()
    checkpoint = tmp_path / "weights.pt"
    torch.save(model.state_dict(), checkpoint)
    contract = contract or {**prediction_contract(32), "bundle_sha256": digest_file(checkpoint)}
    path = tmp_path / "calibration.json"
    calibration = calibrate(predict(model, frame, root, image_size=32),
                             manifest_record=certificate, output=path,
                             model_sha256=digest_file(checkpoint), policy="youden", input_contract=contract)
    return model, checkpoint, path, contract, calibration


def four_targets(tmp_path):
    test = population(tmp_path, "clean", dataset="paired")
    degraded = population(tmp_path, "degraded", dataset="paired")
    video = population(tmp_path, "videos", dataset="synthetic-celeb", video=True)
    root = tmp_path / "df40_images"
    root.mkdir()
    source_rows = []
    old_prefix = tmp_path / "old_server_prefix"
    for index in range(16):
        filename = f"image_{index}.png"
        label = index % 2
        Image.fromarray(np.full((32, 32, 3), 45 + label * 150, dtype=np.uint8)).save(root / filename)
        source_rows.append({"img_path": str(old_prefix / filename), "target": label,
                            "method": "synthetic_generator" if label else "Real_fixture",
                            "paradigm": "diffusion" if label else "real", "source_domain": "fixture"})
    source = tmp_path / "absolute_df40.csv"
    pd.DataFrame(source_rows).to_csv(source, index=False)
    df40 = tmp_path / "certified_df40.csv"
    prepare_df40_cli(["--source", str(source), "--images-root", str(root), "--output", str(df40),
                      "--path-prefix", str(old_prefix), "--source-domain-column", "source_domain",
                      "--convention", "fake-is-1", "--acknowledge-reviewed-subset"])
    return [TargetSpec("test", *test), TargetSpec("test_d", *degraded),
            TargetSpec("df40", df40, root, breakdown=True),
            TargetSpec("celeb_df_v2", *video, primary_unit="video")]


def test_all_four_targets_tensor_and_external_adapter(tmp_path, monkeypatch):
    targets = four_targets(tmp_path)
    model, checkpoint, calibration_path, contract, calibration = calibrated_fixture(tmp_path)
    result = evaluate_suite(model, targets, tmp_path / "suite", checkpoint_path=checkpoint,
                            calibration_path=calibration_path, image_size=32, input_contract=contract,
                            batch_size=4, bootstrap_draws=50, scope="synthetic")
    assert result["state"] == "complete" and len(result["results"]) == 4
    assert result["paired_test_d"]["interval"]["estimate"] == 0
    assert result["results"]["df40"]["per_generator"]["macro_auc"] == 1
    assert result["results"]["df40"]["per_paradigm"]["macro_auc"] == 1
    report = result["results"]["celeb_df_v2"]
    assert report["primary_unit"] == "video"
    assert report["primary"]["n"] == 8
    assert report["primary"]["threshold"] == calibration["frame_threshold"]
    assert "transferred without target tuning" in report["primary"]["threshold_policy"]
    frame, _ = load_predictions(tmp_path / "suite" / "celeb_df_v2" / "predictions.csv")
    videos, _ = load_predictions(tmp_path / "suite" / "celeb_df_v2" / "video_predictions.csv")
    assert videos.p_fake.to_numpy() == pytest.approx(aggregate_videos(frame).p_fake.to_numpy())
    assert videos.n_frames.nunique() == 2
    status = json.loads((tmp_path / "suite" / "celeb_df_v2" / "status.json").read_text())
    assert "video_scores.png" in status["artifacts"]
    assert "frame_roc.csv" in status["artifacts"]
    assert "metrics.csv" in status["artifacts"]
    metrics = pd.read_csv(tmp_path / "suite" / "celeb_df_v2" / "metrics.csv")
    assert set(metrics.unit) == {"frame", "video"}
    assert metrics.loc[metrics.unit.eq("video"), "n"].iloc[0] == 8
    assert all(digest_file(tmp_path / "suite" / "celeb_df_v2" / name) == checksum for name, checksum in status["artifacts"].items())

    calls = []

    def external(frame, root, **kwargs):
        calls.append(root)
        return predict(model, frame, root, **kwargs).sample(frac=1, random_state=8).drop(columns=["img_name", "dataset", "split"])

    second = evaluate_suite(None, targets, tmp_path / "adapter_suite", checkpoint_path=checkpoint,
                            calibration_path=calibration_path, image_size=32, input_contract=contract,
                            predict_fn=external, bootstrap_draws=0, plots=False, scope="synthetic")
    assert len(calls) == 4
    assert second["results"]["celeb_df_v2"]["primary"]["auc"] == report["primary"]["auc"]

    monkeypatch.setenv("TCC_DATASET_ROOT", str(tmp_path))
    monkeypatch.setenv("TCC_SKIP_UNREADABLE", "1")
    (Path(targets[0].root) / "0.png").unlink()
    with pytest.raises(FileNotFoundError):
        evaluate_suite(model, targets, tmp_path / "failed", checkpoint_path=checkpoint,
                       calibration_path=calibration_path, image_size=32, input_contract=contract,
                       bootstrap_draws=0, plots=False, scope="synthetic")
    assert json.loads((tmp_path / "failed" / "status.json").read_text())["state"] == "failed"


def test_suite_pairing_dry_run_and_contract_failures(tmp_path):
    targets = four_targets(tmp_path)
    _, checkpoint, calibration, contract, _ = calibrated_fixture(tmp_path)
    planned = evaluate_suite(None, targets, tmp_path / "dry", checkpoint_path=checkpoint,
                              calibration_path=calibration, image_size=32, input_contract=contract,
                              dry_run=True, scope="synthetic")
    assert planned["state"] == "planned_not_executed" and not (tmp_path / "dry").exists()
    with pytest.raises(ValueError, match="Inference setting"):
        evaluate_suite(None, targets, tmp_path / "bad_size", checkpoint_path=checkpoint,
                       calibration_path=calibration, image_size=64, input_contract=contract,
                       dry_run=True, scope="synthetic")
    with pytest.raises(ValueError, match="bundle_sha256"):
        evaluate_suite(None, targets, tmp_path / "bad_bundle", checkpoint_path=checkpoint,
                       calibration_path=calibration, image_size=32,
                       input_contract={key: value for key, value in contract.items() if key != "bundle_sha256"},
                       dry_run=True, scope="synthetic")
    with pytest.raises(ValueError, match="different input"):
        evaluate_suite(None, targets, tmp_path / "bad", checkpoint_path=checkpoint,
                       calibration_path=calibration, image_size=64, input_contract={**contract, "image_size": 64},
                       dry_run=True, scope="synthetic")
    frame, _ = load_manifest(targets[1].manifest)
    frame.loc[0, "group_id"] = "wrong"
    changed = tmp_path / "changed.csv"
    save_manifest(frame, changed, {})
    targets[1] = TargetSpec("test_d", changed, targets[1].root)
    with pytest.raises(ValueError, match="group identity"):
        evaluate_suite(None, targets, tmp_path / "bad_pair", checkpoint_path=checkpoint,
                       calibration_path=calibration, image_size=32, input_contract=contract, dry_run=True, scope="synthetic")


def test_seed_summary_uses_video_auc_and_exact_seed_population(tmp_path):
    # Unequal frame counts give frame AUC 0.96 but video AUC 0.75.
    rows = []
    for video, label, scores in [("r0", 0, [0.8]), ("r1", 0, [0.1] * 4),
                                 ("f0", 1, [0.7]), ("f1", 1, [0.9] * 4)]:
        for index, score in enumerate(scores):
            rows.append({"sample_id": f"{video}:{index}", "img_name": f"{video}_{index}.png",
                         "group_id": video, "video_id": video, "label": label,
                         "dataset": "synthetic_video", "split": "smoke_test", "p_fake": score})
    scores = pd.DataFrame(rows)
    manifest = tmp_path / "videos.csv"
    save_manifest(scores.drop(columns="p_fake"), manifest, {"group_unit": "video"})
    frame, certificate = load_manifest(manifest)
    condition = {"method": "synthetic-controlled-seed"}
    outputs = []
    for seed in [42, 123]:
        checkpoint = tmp_path / f"weights_{seed}.pt"
        checkpoint.write_bytes(str(seed).encode())
        contract = {"checkpoint_class1": "fake", "bundle_sha256": digest_file(checkpoint)}
        calibration_path = tmp_path / f"cal_{seed}.json"
        val = frame.assign(split="val", p_fake=frame.label * 0.8 + 0.1)
        calibrate(val, manifest_record={**certificate, "split": "val"}, output=calibration_path,
                  model_sha256=digest_file(checkpoint), input_contract=contract)
        predictions = scores
        output = tmp_path / f"eval_{seed}"
        report = evaluate_predictions(predictions, manifest, output, checkpoint_path=checkpoint,
                                      calibration_path=calibration_path, input_contract=contract,
                                      research_run={"name": "controlled", "seed": seed,
                                                    "condition": condition, "condition_sha256": digest(condition)},
                                      primary_unit="video")
        assert report["frame"]["auc"] == pytest.approx(0.96)
        assert report["video"]["auc"] == pytest.approx(0.75)
        outputs.append(str(output))
    args = Namespace(seeds=[42, 123], evaluations=outputs, output=str(tmp_path / "seeds.json"))
    result = aggregate_seeds(args)
    assert result["results"][0]["unit"] == "video"
    assert result["results"][0]["mean_auc"] == pytest.approx(0.75)
    args.seeds = [42, 123, 2025]
    args.output = str(tmp_path / "missing_seed.json")
    with pytest.raises(ValueError, match="Missing/extra/duplicate"):
        aggregate_seeds(args)


def test_legacy_cli_config_dry_run_never_loads_network(tmp_path, monkeypatch, capsys):
    from src.pipelines import checkpoints

    targets = four_targets(tmp_path)
    checkpoint = tmp_path / "models" / "resnet" / "none" / "finetune" / "seed_42" / "weights" / "best.pth"
    checkpoint.parent.mkdir(parents=True)
    checkpoint.write_bytes(b"not a model; dry run must never load it")
    results = checkpoint.parent.parent / "results"
    results.mkdir()
    (results / "run_config.json").write_text(json.dumps({"image_size": 32, "allow_pretrained": False}))
    run = checkpoints.run_from_checkpoint(checkpoint)
    config = checkpoints.config_from_run(run)
    contract = legacy_contract(run, config)
    calibration = tmp_path / "legacy_calibration.json"
    calibration.write_text(json.dumps({"schema": SCHEMA, "frame_threshold": 0.5,
                                      "model_sha256": digest_file(checkpoint), "selection_split": "val",
                                      "checkpoint_class1": "fake", "input_contract": contract,
                                      "input_contract_sha256": digest(contract)}))
    path = tmp_path / "suite.yaml"
    path.write_text(yaml.safe_dump({"model": {"type": "legacy", "path": str(checkpoint)},
        "calibration": str(calibration), "output": str(tmp_path / "never_created"), "scope": "synthetic",
        "targets": [{"name": t.name, "manifest": str(t.manifest), "root": str(t.root)} for t in targets]}))
    monkeypatch.setattr(checkpoints, "load_model_from_run", lambda *_: pytest.fail("dry run loaded network"))
    assert run_suite_config(path)["state"] == "planned_not_executed"
    from research_cli import main

    capsys.readouterr()
    assert main(["evaluate-suite", "--config", str(path), "--output", str(tmp_path / "cli_dry")]) == 0
    output = json.loads(capsys.readouterr().out)
    assert output["state"] == "planned_not_executed" and len(output["targets"]) == 4
    assert not (tmp_path / "cli_dry").exists()
