import json
from argparse import Namespace
from pathlib import Path
import numpy as np
import pandas as pd
import pytest
from src.robustness.artifacts import save_predictions, load_predictions, import_legacy
from src.robustness.inference import calibrate
from src.robustness.manifests import save_manifest, load_manifest
from src.robustness.provenance import digest_file, write_json
from src.robustness.analysis import compare_files, shared_failures
from src.robustness.commands import expand, export_xai


def data(tmp_path, split="val"):
    frame = pd.DataFrame(
        {
            "img_name": [f"{i}.png" for i in range(8)],
            "label": [i % 2 for i in range(8)],
            "sample_id": [f"s{i}" for i in range(8)],
            "group_id": [f"g{i}" for i in range(8)],
            "dataset": "synthetic",
            "split": split,
        }
    )
    path = tmp_path / f"{split}.csv"
    save_manifest(frame, path, {})
    frame, cert = load_manifest(path)
    frame["p_fake"] = frame.label * 0.8 + 0.1
    return frame, cert, path


def test_prediction_integrity_and_calibration_binding(tmp_path):
    frame, cert, _ = data(tmp_path)
    path = tmp_path / "preds.csv"
    save_predictions(path, frame, manifest_record=cert, model_sha256="model")
    cal = calibrate(
        frame, manifest_record=cert, output=tmp_path / "cal.json", model_sha256="model"
    )
    assert len(load_predictions(path, calibration=cal)[0]) == 8
    wrong = {**cal, "model_sha256": "wrong"}
    with pytest.raises(ValueError, match="different checkpoint"):
        load_predictions(path, calibration=wrong)
    path.write_text(path.read_text() + "\n")
    with pytest.raises(ValueError, match="bytes"):
        load_predictions(path)


def test_orientation_bound_to_calibration(tmp_path):
    frame, cert, _ = data(tmp_path)
    path = tmp_path / "preds.csv"
    save_predictions(
        path,
        frame,
        manifest_record=cert,
        model_sha256="model",
        checkpoint_class1="real",
    )
    cal = calibrate(
        frame,
        manifest_record=cert,
        output=tmp_path / "cal.json",
        model_sha256="model",
        checkpoint_class1="fake",
    )
    with pytest.raises(ValueError, match="orientation"):
        load_predictions(path, calibration=cal)


def test_no_target_threshold_tuning(tmp_path):
    frame, cert, _ = data(tmp_path, split="test")
    with pytest.raises(ValueError, match="source validation"):
        calibrate(
            frame, manifest_record=cert, output=tmp_path / "cal.json", model_sha256="m"
        )


def test_legacy_import_reorders_and_requires_explicit_semantics(tmp_path):
    frame, cert, manifest = data(tmp_path)
    checkpoint = tmp_path / "weights.pth"
    checkpoint.write_bytes(b"test-artifact-not-loaded")
    old = pd.DataFrame(
        {"id": range(8), "y_true": 1 - frame.label, "prob_pos": frame.p_fake}
    )
    source = tmp_path / "old.csv"
    old.sample(frac=1, random_state=4).to_csv(source, index=False)
    with pytest.raises(ValueError, match="acknowledgement"):
        import_legacy(
            source,
            manifest,
            tmp_path / "new.csv",
            checkpoint=checkpoint,
            source_labels="real-is-1",
            checkpoint_class1="fake",
        )
    import_legacy(
        source,
        manifest,
        tmp_path / "new.csv",
        checkpoint=checkpoint,
        source_labels="real-is-1",
        checkpoint_class1="fake",
        acknowledge_row_order=True,
    )
    new, record = load_predictions(tmp_path / "new.csv")
    assert np.allclose(new.p_fake, frame.p_fake)
    assert record["model_sha256"] == digest_file(checkpoint)
    assert record["origin"].startswith("existing prediction")


def test_legacy_missing_rows_fail(tmp_path):
    f, _, manifest = data(tmp_path)
    weights = tmp_path / "w"
    weights.write_bytes(b"w")
    source = tmp_path / "p.csv"
    pd.DataFrame({"id": [0, 1], "y_true": [0, 1], "prob_pos": [0.1, 0.8]}).to_csv(
        source, index=False
    )
    with pytest.raises(ValueError, match="Missing"):
        import_legacy(
            source,
            manifest,
            tmp_path / "out.csv",
            checkpoint=weights,
            source_labels="fake-is-1",
            checkpoint_class1="fake",
            acknowledge_row_order=True,
        )


def test_comparison_and_empty_intersection_are_artifact_only(tmp_path):
    frame, cert, _ = data(tmp_path)
    files = []
    calibrations = []
    for number in [0, 1]:
        path = tmp_path / f"p{number}.csv"
        cal = tmp_path / f"c{number}.json"
        save_predictions(path, frame, manifest_record=cert, model_sha256=str(number))
        calibrate(frame, manifest_record=cert, output=cal, model_sha256=str(number))
        files.append(path)
        calibrations.append(cal)
    report = compare_files(*files, tmp_path / "comparison", *calibrations, draws=50)
    assert report["paired_auc_interval"]["estimate"] == 0
    result = shared_failures(files, calibrations, tmp_path / "shared")
    assert result["n_shared_failures"] == 0
    assert pd.read_csv(tmp_path / "shared/shared_failures.csv").empty


def test_xai_bridge_keeps_manifest_order(tmp_path):
    frame, cert, manifest = data(tmp_path)
    path = tmp_path / "p.csv"
    cal = tmp_path / "c.json"
    save_predictions(path, frame, manifest_record=cert, model_sha256="m")
    calibrate(frame, manifest_record=cert, output=cal, model_sha256="m")
    export_xai(
        Namespace(
            manifest=manifest,
            predictions=path,
            calibration=cal,
            output=tmp_path / "xai",
        )
    )
    mapping = pd.read_csv(tmp_path / "xai/identity_map.csv")
    assert mapping.sample_id.tolist() == frame.sample_id.tolist()
    assert len(pd.read_csv(tmp_path / "xai/predictions.csv")) == len(frame)
