import json
from pathlib import Path

import numpy as np
import pandas as pd
import pytest
import torch
from PIL import Image

from src.experimental.moe.data import open_sources, split_validation
from src.experimental.moe.inference import describe_run, load_predictor, validate_calibration_population, validate_sources
from src.experimental.moe.models import FrozenLateFusion, load_balancing_loss, simple_fusion
from src.experimental.moe.training import fit
from src.robustness.artifacts import save_predictions
from src.robustness.inference import calibrate
from src.robustness.manifests import load_manifest, save_manifest
from src.robustness.provenance import digest, digest_file, write_json


def fixture_experts(tmp_path, n=48):
    images = tmp_path / "images"
    images.mkdir()
    names = [f"sample_{i}.png" for i in range(n)]
    hashes = []
    for i, name in enumerate(names):
        Image.new("RGB", (4, 4), (i, 20, 80)).save(images / name)
        hashes.append(digest_file(images / name))
    frame = pd.DataFrame({"img_name": names, "label": np.arange(n) % 2,
        "sample_id": [f"id_{i:03d}" for i in range(n)], "group_id": [f"group_{i:03d}" for i in range(n)],
        "dataset": "synthetic", "split": "val", "image_sha256": hashes})
    manifest = tmp_path / "val.csv"
    certificate = save_manifest(frame, manifest, {})
    specs = []
    rng = np.random.default_rng(42)
    for index, role in enumerate(("srm", "rgb")):
        cache = tmp_path / role
        cache.mkdir()
        features = rng.normal(size=(n, 5 + index)).astype(np.float16)
        difference = (2 * frame.label.to_numpy() - 1) * 0.6 + rng.normal(size=n)
        logits = np.stack([-difference / 2, difference / 2], -1).astype(np.float32)
        order = np.arange(n) if index == 0 else np.arange(n)[::-1]
        np.save(cache / "features.npy", features[order])
        np.save(cache / "logits.npy", logits[order])
        frame.iloc[order].to_csv(cache / "rows.csv", index=False)
        extraction = {"mode": "srm" if index == 0 else "none", "checkpoint_class1": "fake", "image_size": 224}
        checkpoint = digest(["synthetic-checkpoint", role])
        key = {"checkpoint_sha256": checkpoint, "run_config_sha256": digest(role), "extraction": extraction,
               "code_sha256": {"synthetic-extractor": digest("fixture")}, "packages": {}, "manifest_sha256": certificate["manifest_sha256"]}
        write_json(cache / "cache.json", {"schema": "faceforgery-features-v1", "status": "complete", "key": key,
            "identity": digest(key), "checkpoint": {"sha256": checkpoint}, "feature_dim": features.shape[1],
            "extraction": extraction, "manifest": certificate, "sample_ids_sha256": digest(frame.iloc[order].sample_id.tolist()),
            "files": {name: digest_file(cache / name) for name in ("features.npy", "logits.npy", "rows.csv")}})
        specs.append({"name": role, "role": role, "kind": "feature_cache", "path": str(cache)})
    predicted = frame.copy()
    predicted["p_fake"] = np.clip(0.2 + 0.6 * frame.label + rng.normal(scale=.2, size=n), .01, .99)
    predicted["residual_score"] = rng.random(n)
    checkpoint = digest("synthetic-reconstruction")
    contract = {"kind": "synthetic-reconstruction", "checkpoint_class1": "fake", "score": "sigmoid fixed residual detector"}
    path, calibration = tmp_path / "reconstruction.csv", tmp_path / "reconstruction-calibration.json"
    save_predictions(path, predicted, manifest_record=certificate, model_sha256=checkpoint,
                     metadata={"input_contract": contract, "input_contract_sha256": digest(contract)})
    calibrate(predicted, manifest_record=certificate, output=calibration, model_sha256=checkpoint, input_contract=contract)
    specs.append({"name": "reconstruction", "role": "reconstruction", "kind": "predictions", "path": str(path),
                  "calibration": str(calibration), "feature_columns": ["residual_score"]})
    return frame, specs, images


@pytest.mark.parametrize("routing", ["soft", "top_k"])
def test_routing_classification_gradient_and_frozen_inputs(routing):
    torch.manual_seed(4)
    model = FrozenLateFusion(4, 3, train_routing=routing, train_top_k=2, expert_dropout=.8,
                            inference_routing="top_k", inference_top_k=1)
    x = torch.randn(12, 4, requires_grad=True)
    p = torch.tensor([[.1, .6, .9]] * 12, requires_grad=True)
    out = model(x, p)
    assert torch.allclose(out["weights"].sum(-1), torch.ones(12))
    assert (out["weights"] > 0).sum(-1).min() >= 2
    torch.nn.functional.binary_cross_entropy(out["p_fake"], torch.ones(12)).backward()
    assert any(parameter.grad is not None and parameter.grad.abs().sum() > 0 for parameter in model.parameters())
    assert x.grad is None and p.grad is None
    model.eval()
    sparse = model(x, p)
    assert torch.equal((sparse["weights"] > 0).sum(-1), torch.ones(12, dtype=torch.long))
    assert torch.allclose(sparse["p_fake"], (sparse["weights"] * p).sum(-1))
    assert torch.equal(model(x, p)["weights"], sparse["weights"])
    with pytest.raises(ValueError, match="zero classification gradient"):
        FrozenLateFusion(4, 3, train_routing="top_k", train_top_k=1)


def test_temperature_balancing_and_simple_fusions():
    p = torch.tensor([[.2, .8], [.4, .6]])
    assert torch.allclose(simple_fusion(p), torch.tensor([.5, .5]))
    assert torch.allclose(simple_fusion(p, "geometric"), p.prod(-1).sqrt())
    assert torch.allclose(simple_fusion(p, "geometric_binary"), torch.tensor([.5, .5]))
    uniform = torch.full((6, 3), 1 / 3, requires_grad=True)
    assignment = torch.eye(3).repeat(2, 1)
    assert load_balancing_loss(uniform, assignment).item() == pytest.approx(1)
    collapsed = torch.tensor([[1., 0., 0.]] * 6, requires_grad=True)
    assert load_balancing_loss(collapsed, torch.tensor([[1., 0., 0.]] * 6)).item() == pytest.approx(3)
    cold = FrozenLateFusion(2, 3, temperature=.5, expert_dropout=0)
    hot = FrozenLateFusion(2, 3, temperature=2., expert_dropout=0)
    hot.load_state_dict(cold.state_dict())
    scores, x = torch.tensor([[.1, .5, .9]] * 8), torch.randn(8, 2)
    assert cold(x, scores)["entropy"].mean() < hot(x, scores)["entropy"].mean()
    with pytest.raises(ValueError):
        simple_fusion(torch.tensor([[float("nan"), .2]]))


def test_key_alignment_hash_split_and_leakage_rejection(tmp_path):
    frame, specs, _ = fixture_experts(tmp_path)
    aligned = open_sources(specs)
    x, p = aligned.block(np.arange(len(frame)))
    assert x.shape == (len(frame), 15) and p.shape == (len(frame), 3)
    fit_ids, select_ids, record = split_validation(aligned.frame)
    shuffled = aligned.frame.sample(frac=1, random_state=7).reset_index(drop=True)
    again_fit, again_select, again_record = split_validation(shuffled)
    assert record == again_record
    assert set(frame.iloc[fit_ids].sample_id) == set(shuffled.iloc[again_fit].sample_id)
    assert set(frame.iloc[select_ids].group_id).isdisjoint(frame.iloc[fit_ids].group_id)
    for split in ("train", "test", "test_d"):
        with pytest.raises(ValueError, match="source validation"):
            split_validation(frame.assign(split=split))
    crossing = aligned.frame.copy()
    crossing.loc[fit_ids[0], "source_id"] = "shared-source"
    crossing.loc[select_ids[0], "source_id"] = "shared-source"
    crossing["source_id"] = crossing.source_id.fillna("")
    with pytest.raises(ValueError, match="overlap in source_id"):
        split_validation(crossing)
    wrong = frame.copy()
    wrong.loc[0, "label"] = 1 - wrong.loc[0, "label"]
    with pytest.raises(ValueError, match="alignment mismatch: label"):
        open_sources(specs, frame=wrong)
    expected = [expert.contract for expert in aligned.experts]
    expected[0] = {**expected[0], "checkpoint_sha256": digest("different")}
    with pytest.raises(ValueError, match="differs from fitted router"):
        open_sources(specs, expected_contracts=expected)


def test_three_expert_fit_reload_calibration_and_baselines(tmp_path):
    frame, specs, images = fixture_experts(tmp_path)
    config = {"task": "moe", "name": "synthetic-three-expert", "scope": "synthetic-structural", "seed": 42,
              "output_dir": str(tmp_path / "fit"), "data": {"experts": specs},
              "model": {"hidden_dim": 8, "train_routing": "top_k", "train_top_k": 2,
                        "inference_routing": "soft", "expert_dropout": .1},
              "training": {"device": "cpu", "epochs": 2, "batch_size": 8, "amp": False, "lr": .01}}
    run = fit(config)
    report = json.loads((run / "comparison.json").read_text())
    assert {"router", "mean", "geometric", "logistic", "expert_srm", "expert_rgb", "expert_reconstruction"} <= set(report["comparisons"])
    description = describe_run(run)
    selection, certificate = load_manifest(run / "val_select.csv")
    validate_calibration_population(run, selection, certificate)
    with pytest.raises(ValueError, match="val_select"):
        validate_calibration_population(run, frame, {"manifest_sha256": digest("full-val")})
    predicted = load_predictor(run, specs)(selection.drop(columns=["image_sha256", "fusion_subset"]), images, device="cpu")
    saved = pd.read_csv(run / "validation_predictions.csv").sort_values("sample_id")
    assert np.allclose(predicted.p_fake, saved.p_fake, atol=1e-6)
    calibration = json.loads((run / "calibration.json").read_text())
    assert calibration["model_sha256"] == digest_file(description["checkpoint_path"])
    assert calibration["input_contract"] == description["input_contract"]
    assert report["counts"]["val_fit"] + report["counts"]["val_select"] == len(frame)
    assert len(validate_sources(run, specs, frame=selection, root=images).frame) == len(selection)
    assert sum(report["routing"]["mean_weight"].values()) == pytest.approx(1)
    for method in ("mean", "geometric", "logistic", "expert_srm"):
        method_description = describe_run(run, method)
        method_calibration = json.loads((run / "comparisons" / method / "calibration.json").read_text())
        assert method_calibration["input_contract"] == method_description["input_contract"]
        assert method_calibration["model_sha256"] == digest_file(method_description["checkpoint_path"])
        result = load_predictor(run, specs, method=method)(selection, images)
        expected = pd.read_csv(run / "comparisons" / method / "validation_predictions.csv").sort_values("sample_id")
        assert np.allclose(result.p_fake, expected.p_fake, atol=1e-6)
    (images / frame.img_name.iloc[0]).write_bytes(b"changed")
    with pytest.raises(ValueError, match="image bytes"):
        load_predictor(run, specs)(frame, images)
    with (run / "stacker.pt").open("ab") as stream:
        stream.write(b"tampered")
    with pytest.raises(ValueError, match="dependency missing or changed"):
        describe_run(run)


def test_completed_epoch_resume_keeps_source_population_identity(tmp_path, monkeypatch):
    from src.experimental.moe import training

    _, specs, _ = fixture_experts(tmp_path)
    config = {"task": "moe", "name": "resume-fixture", "seed": 7,
        "output_dir": str(tmp_path / "resume"), "data": {"experts": specs},
        "training": {"device": "cpu", "epochs": 2, "batch_size": 8, "amp": False}}
    original = training.fit_model

    def stop_after_epoch(*args, **kwargs):
        original(*args, **{**kwargs, "epochs": 1})
        raise RuntimeError("simulated interruption after completed epoch")

    monkeypatch.setattr(training, "fit_model", stop_after_epoch)
    with pytest.raises(RuntimeError, match="simulated interruption"):
        fit(config)
    monkeypatch.setattr(training, "fit_model", original)
    config["training"]["resume"] = True
    run = fit(config)
    history = json.loads((run / "history.json").read_text())
    assert [row["epoch"] for row in history] == [0, 1]
    record = json.loads((run / "run.json").read_text())
    assert "resume" not in record["config"]["training"]
