"""Explicit inference artifacts. No training or threshold selection on target data."""

from __future__ import annotations
import json
from contextlib import nullcontext
from pathlib import Path
import pandas as pd
import torch
from torch.utils.data import DataLoader
from .imaging import CanonicalDataset, PREPROCESSING
from .legacy_encoding import encode_legacy_tensor
from .manifests import load_manifest
from .provenance import SCHEMA, digest, digest_file, source_identity, write_json, write_csv
from .artifacts import save_predictions
from .statistics import (
    aggregate_videos,
    choose_threshold,
    generator_metrics,
    summary,
    checked_predictions,
    grouped_auc_interval,
    subgroup_metrics,
)


def prediction_contract(image_size, mode=None, in_channels=None, positive_class="fake"):
    """Content identity of the default tensor predictor's input and score semantics."""
    helpers = {}
    if mode is not None:
        root = Path(__file__).resolve().parents[2]
        paths = ["src/data/data.py", "src/robustness/legacy_encoding.py"]
        if mode == "srm":
            paths.append("src/forensics/srm.py")
        elif mode == "dtcwt":
            paths.append("src/forensics/dtcwt_module.py")
        helpers = {name: digest_file(root / name) for name in paths}
    return {
        "image_size": int(image_size),
        "resize": "PIL RGB square bilinear",
        "representation": mode or "model-internal",
        "in_channels": in_channels,
        "checkpoint_class1": positive_class,
        "score": "FP32 softmax of two logits; declared fake class",
        "legacy_helpers_sha256": helpers,
    }


def inference_precision(device="cpu", use_amp=True):
    device = torch.device(device)
    effective = "float32"
    if device.type == "cuda" and use_amp:
        effective = "bfloat16" if torch.cuda.is_bf16_supported() else "float16"
    return {"device_type": device.type, "amp_requested": bool(use_amp), "effective": effective}


def bind_population(predictions, frame):
    """Check IDs and labels, then restore annotations from the certified population."""
    p = checked_predictions(predictions)
    expected = frame.sort_values("sample_id").reset_index(drop=True)
    if not p.sample_id.equals(expected.sample_id.astype(str)) or len(p) != len(expected):
        raise ValueError("Incomplete inference population or changed sample identity")
    for column in ("label", "group_id"):
        if not p[column].equals(expected[column]):
            raise ValueError(f"Prediction {column} differs from certified population")
    for column in expected:
        if column in p and not p[column].astype(str).equals(expected[column].astype(str)):
            raise ValueError(f"Prediction annotation changed: {column}")
    result = expected.copy()
    for column in p:
        if column not in result:
            result[column] = p[column]
    return checked_predictions(result)


def predict(
    model,
    frame,
    root,
    *,
    image_size: int,
    device="cpu",
    batch_size=32,
    workers=0,
    mode=None,
    in_channels=None,
    positive_class="fake",
    hash_images=True,
    use_amp=True,
):
    if positive_class not in {"fake", "real"}:
        raise ValueError("Declare checkpoint class-1 meaning")
    if batch_size < 1 or workers < 0:
        raise ValueError("Invalid loader settings")
    device = torch.device(device)
    ds = CanonicalDataset(frame, root, image_size, hash_images=hash_images)
    loader = DataLoader(ds, batch_size=batch_size, num_workers=workers, shuffle=False)
    model = model.to(device).eval()
    precision = inference_precision(device, use_amp)
    rows = []
    with torch.inference_mode():
        for batch in loader:
            raw = batch["image"]
            inputs = (
                encode_legacy_tensor(raw, mode, in_channels).to(device)
                if mode is not None
                else raw.to(device)
            )
            context = (
                torch.amp.autocast("cuda", dtype=getattr(torch, precision["effective"]))
                if precision["effective"] != "float32" else nullcontext()
            )
            with context:
                logits = model(inputs)
            if (
                not isinstance(logits, torch.Tensor)
                or logits.ndim != 2
                or logits.shape != (len(raw), 2)
                or not torch.isfinite(logits).all()
            ):
                raise ValueError(
                    "Expected finite two-class logits; no silent numerical replacement"
                )
            p = (
                logits.float()
                .softmax(-1)[:, 1 if positive_class == "fake" else 0]
                .cpu()
                .numpy()
            )
            for j, position in enumerate(batch["index"].tolist()):
                row = frame.iloc[position].to_dict()
                if row["sample_id"] != batch["sample_id"][j]:
                    raise ValueError("Loader changed sample identity")
                row.update(p_fake=float(p[j]), image_sha256=batch["image_sha256"][j])
                rows.append(row)
    result = checked_predictions(pd.DataFrame(rows))
    if set(result.sample_id) != set(frame.sample_id) or len(result) != len(frame):
        raise ValueError("Incomplete inference population")
    return result


def calibrate(
    predictions, *, manifest_record, output, model_sha256, checkpoint_class1="fake",
    policy="balanced_accuracy", input_contract=None,
):
    if checkpoint_class1 not in {"fake", "real"}:
        raise ValueError("Invalid score orientation")
    if input_contract is not None and input_contract.get("checkpoint_class1", checkpoint_class1) != checkpoint_class1:
        raise ValueError("Calibration input contract has a different score orientation")
    if manifest_record["split"] != "val":
        raise ValueError("Threshold fitting requires source validation split=val")
    output = Path(output)
    if output.exists():
        raise FileExistsError("Calibration is frozen; use a new artifact path")
    p = checked_predictions(predictions)
    if "rows" in manifest_record and len(p) != manifest_record["rows"]:
        raise ValueError("Calibration population differs from source certificate")
    for column in ("dataset", "split"):
        if column in p and set(p[column]) != {manifest_record[column]}:
            raise ValueError("Calibration predictions differ from source certificate")
    record = {
        "schema": SCHEMA,
        "model_sha256": model_sha256,
        "source_manifest_sha256": manifest_record["manifest_sha256"],
        "source_dataset": manifest_record["dataset"],
        "selection_split": "val",
        "label_convention": "fake-is-1",
        "checkpoint_class1": checkpoint_class1,
        "frame_threshold": choose_threshold(p.label, p.p_fake, policy),
        "policy": policy,
        "method": f"maximum validation {policy}; smallest threshold tie break",
    }
    if input_contract is not None:
        record["input_contract"] = input_contract
        record["input_contract_sha256"] = digest(input_contract)
    if "video_id" in p and p.video_id.astype(str).str.strip().ne("").all():
        videos = aggregate_videos(p)
        record["video_threshold"] = choose_threshold(videos.label, videos.p_fake, policy)
        record["video_aggregation"] = "mean frame fake probability"
    write_json(output, record)
    return record


def evaluation_report(predictions, calibration, checkpoint_hash, *, primary_unit=None,
                      breakdown=False, real_reference_policy="source_matched",
                      bootstrap_draws=0, bootstrap_seed=42, confidence=0.95):
    if (
        calibration.get("schema") != SCHEMA
        or calibration.get("model_sha256") != checkpoint_hash
        or calibration.get("selection_split") != "val"
    ):
        raise ValueError(
            "Calibration does not belong to this checkpoint/source-validation protocol"
        )
    p = checked_predictions(predictions)
    result = {
        "frame": summary(p.label, p.p_fake, float(calibration["frame_threshold"])),
        "calibration": calibration,
        "label_convention": "fake-is-1",
    }
    result["frame"]["threshold_policy"] = f"frozen source-validation {calibration.get('policy', 'balanced_accuracy')}"
    if "video_id" in p and p.video_id.astype(str).str.strip().ne("").all():
        videos = aggregate_videos(p)
        t = calibration.get("video_threshold", calibration["frame_threshold"])
        result["video"] = summary(videos.label, videos.p_fake, float(t))
        result["video"]["threshold_policy"] = (
            "source-video validation"
            if "video_threshold" in calibration
            else "source-frame validation threshold transferred without target tuning"
        )
        result["video"]["aggregation"] = (
            "mean fake probability; one observation per video"
        )
    if {"generator", "source_domain"} <= set(p) and not breakdown:
        result["per_generator"] = generator_metrics(
            p, float(calibration["frame_threshold"])
        )
    if breakdown:
        for column in ("generator", "paradigm"):
            if column in p:
                result["per_" + column] = subgroup_metrics(
                    p, float(calibration["frame_threshold"]), column=column,
                    real_reference_policy=real_reference_policy,
                    draws=bootstrap_draws, seed=bootstrap_seed, confidence=confidence,
                )
    primary_unit = primary_unit or ("video" if "video" in result else "frame")
    if primary_unit not in {"frame", "video"} or primary_unit not in result:
        raise ValueError("Primary observation unit is unavailable")
    result["primary_unit"] = primary_unit
    result["primary"] = result[primary_unit]
    if bootstrap_draws:
        for unit in ("frame", "video"):
            if unit not in result:
                continue
            population = aggregate_videos(p) if unit == "video" else p
            result[unit]["auc_interval"] = grouped_auc_interval(
                population, draws=bootstrap_draws, seed=bootstrap_seed, confidence=confidence
            )
    return result


def evaluate(
    model,
    manifest,
    root,
    output,
    *,
    checkpoint_path,
    calibration_path,
    image_size,
    mode=None,
    in_channels=None,
    positive_class="fake",
    research_run=None,
    predict_fn=None,
    input_contract=None,
    primary_unit=None,
    breakdown=False,
    real_reference_policy="source_matched",
    bootstrap_draws=0,
    bootstrap_seed=42,
    confidence=0.95,
    plots=False,
    target_name=None,
    **kwargs,
):
    output = Path(output)
    if output.exists():
        raise FileExistsError(
            "Use a new evaluation output directory; stale caches are not reused"
        )
    frame, record = load_manifest(manifest)
    checksum = digest_file(checkpoint_path)
    calibration = json.loads(Path(calibration_path).read_text())
    contract = input_contract or prediction_contract(image_size, mode, in_channels, positive_class)
    validate_input_contract(calibration, contract)
    validate_prediction_settings(contract, image_size, mode, in_channels, positive_class)
    if calibration.get("checkpoint_class1", "fake") != positive_class:
        raise ValueError(
            "Calibration score orientation differs from this checkpoint interpretation"
        )
    if (
        calibration.get("model_sha256") != checksum
        or calibration.get("selection_split") != "val"
    ):
        raise ValueError("Unmatched or non-validation calibration")
    output.mkdir(parents=True)
    write_json(
        output / "status.json",
        {"state": "running", "manifest_sha256": record["manifest_sha256"]},
    )
    try:
        arguments = dict(
            image_size=image_size,
            mode=mode,
            in_channels=in_channels,
            positive_class=positive_class,
            **kwargs,
        )
        p = predict_fn(frame, root, **arguments) if predict_fn is not None else predict(model, frame, root, **arguments)
        return _publish_predictions(
            bind_population(p, frame), record, output, calibration, checksum,
            calibration_sha256=digest_file(calibration_path), contract=contract,
            research_run=research_run, primary_unit=primary_unit, breakdown=breakdown,
            real_reference_policy=real_reference_policy, bootstrap_draws=bootstrap_draws,
            bootstrap_seed=bootstrap_seed, confidence=confidence, plots=plots,
            target_name=target_name,
            origin="external prediction adapter" if predict_fn is not None else "fresh image inference",
            precision=inference_precision(kwargs.get("device", "cpu"), kwargs.get("use_amp", True)),
        )
    except Exception as error:
        write_json(output / "status.json", {"state": "failed", "error": f"{type(error).__name__}: {error}"})
        raise


def validate_input_contract(calibration, contract):
    if "input_contract_sha256" in calibration:
        if digest(calibration.get("input_contract")) != calibration["input_contract_sha256"]:
            raise ValueError("Calibration input contract changed")
        if digest(contract) != calibration["input_contract_sha256"]:
            raise ValueError("Calibration belongs to a different input or score contract")


def validate_prediction_settings(contract, image_size, mode, in_channels, positive_class):
    """Reject tensor settings that disagree with an explicitly recorded contract."""
    expected = {"image_size": int(image_size), "checkpoint_class1": positive_class}
    if mode is not None:
        expected.update(representation=mode, in_channels=in_channels)
    for key, value in expected.items():
        if key in contract and contract[key] != value:
            raise ValueError(f"Inference setting differs from input contract: {key}")


def evaluate_predictions(predictions, manifest, output, *, checkpoint_path,
                         calibration_path, input_contract, research_run=None, **report_kwargs):
    """Report certified externally computed scores through the same artifact writer."""
    output = Path(output)
    if output.exists():
        raise FileExistsError("Use a new evaluation output directory")
    frame, record = load_manifest(manifest)
    calibration = json.loads(Path(calibration_path).read_text())
    checksum = digest_file(checkpoint_path)
    validate_input_contract(calibration, input_contract)
    if calibration.get("checkpoint_class1", "fake") != input_contract.get("checkpoint_class1", "fake"):
        raise ValueError("Calibration score orientation differs from predictor")
    output.mkdir(parents=True)
    try:
        return _publish_predictions(
            bind_population(predictions, frame), record, output, calibration, checksum,
            calibration_sha256=digest_file(calibration_path), contract=input_contract,
            research_run=research_run, origin="explicit external predictions", precision=None,
            **report_kwargs,
        )
    except Exception as error:
        write_json(output / "status.json", {"state": "failed", "error": f"{type(error).__name__}: {error}"})
        raise


def _publish_predictions(p, record, output, calibration, checksum, *, calibration_sha256,
                         contract, research_run=None, origin, precision, primary_unit=None,
                         breakdown=False, real_reference_policy="source_matched",
                         bootstrap_draws=0, bootstrap_seed=42, confidence=0.95,
                         plots=False, target_name=None):
    report = evaluation_report(
        p, calibration, checksum, primary_unit=primary_unit, breakdown=breakdown,
        real_reference_policy=real_reference_policy, bootstrap_draws=bootstrap_draws,
        bootstrap_seed=bootstrap_seed, confidence=confidence,
    )
    report.update(
        schema=SCHEMA,
        manifest=record,
        checkpoint_sha256=checksum,
        checkpoint_class1=calibration.get("checkpoint_class1", "fake"),
        preprocessing=(PREPROCESSING if contract.get("representation") == "model-internal"
                       else "PIL RGB bilinear; exact predictor preprocessing bound by input_contract"),
        representation=contract.get("representation", "external"),
        input_contract=contract,
        input_contract_sha256=digest(contract),
        calibration_sha256=calibration_sha256,
        input_contract_binding="verified" if "input_contract_sha256" in calibration else "historical calibration without input contract",
        inference_precision=precision,
        target_name=target_name or record["dataset"],
        software=source_identity(),
    )
    if research_run is not None:
        report["research_run"] = research_run
    save_predictions(
        output / "predictions.csv",
        p,
        manifest_record=record,
        model_sha256=checksum,
        checkpoint_class1=calibration.get("checkpoint_class1", "fake"),
        metadata={
            "origin": origin,
            "preprocessing": report["preprocessing"],
            "input_contract_sha256": digest(contract),
        },
    )
    files = ["predictions.csv", "predictions.csv.json", "metrics.json", "metrics.csv"]
    units = [unit for unit in ("frame", "video") if unit in report]
    write_csv(output / "metrics.csv", pd.DataFrame([_metric_row(report[unit], unit=unit) for unit in units]))
    for name in ("per_generator", "per_paradigm"):
        if name in report:
            write_csv(output / f"{name}.csv", pd.DataFrame([_metric_row(row) for row in report[name]["groups"]]))
            files.append(f"{name}.csv")
    if "video" in report:
        videos = aggregate_videos(p)
        save_predictions(
            output / "video_predictions.csv", videos, manifest_record=record,
            model_sha256=checksum, checkpoint_class1=calibration.get("checkpoint_class1", "fake"),
            metadata={"origin": "mean frame p_fake", "unit": "video"},
        )
        files.extend(["video_predictions.csv", "video_predictions.csv.json"])
    if plots:
        from .plots import write_forensic_plots

        for unit in ("frame", "video"):
            if unit in report:
                population = aggregate_videos(p) if unit == "video" else p
                files.extend(write_forensic_plots(population, report[unit], output,
                                                 title=f"{report['target_name']} / {unit}", unit=unit))
    write_json(output / "metrics.json", report)
    write_json(
        output / "status.json",
        {
            "state": "complete",
            "expected": record["rows"],
            "observed": len(p),
            "artifacts": {n: digest_file(output / n) for n in files},
        },
    )
    return report


def _metric_row(metrics, **labels):
    row = {**labels, **{key: value for key, value in metrics.items()
                       if value is None or isinstance(value, (str, int, float, bool))}}
    row.update({"auc_ci_" + key: value for key, value in metrics.get("auc_interval", {}).items()
                if value is None or isinstance(value, (str, int, float, bool))})
    return row
