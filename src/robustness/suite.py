"""One calibrated evaluation protocol for four target kinds and score adapters."""

from __future__ import annotations

from dataclasses import asdict, dataclass
import json
import os
from pathlib import Path
import re

from .artifacts import load_predictions
from .inference import (evaluate, prediction_contract, validate_calibration,
                        validate_input_contract, validate_prediction_settings)
from .manifests import load_manifest
from .provenance import digest, digest_file, source_identity, write_json
from .statistics import aggregate_videos, align, grouped_auc_interval

KINDS = {"test", "test_d", "df40", "celeb_df_v2"}


@dataclass(frozen=True)
class TargetSpec:
    name: str
    manifest: str | Path
    root: str | Path
    primary_unit: str = "frame"
    breakdown: bool = False
    real_reference_policy: str = "source_matched"
    kind: str | None = None


def validate_targets(targets):
    if not targets or len({target.name for target in targets}) != len(targets):
        raise ValueError("Distinct nonempty evaluation targets required")
    populations, records = {}, {}
    kinds = {}
    for target in targets:
        if not re.fullmatch(r"[A-Za-z0-9][A-Za-z0-9_.-]*", target.name):
            raise ValueError("Target name must be a safe directory component")
        kind = target.kind or target.name
        if kind not in KINDS:
            raise ValueError(f"Unsupported target kind: {kind}")
        if kind in kinds:
            raise ValueError("One manifest per target kind; use a separate suite for another population")
        kinds[kind] = target.name
        if target.primary_unit not in {"frame", "video"}:
            raise ValueError("Primary unit must be frame or video")
        if target.real_reference_policy not in {"source_matched", "pooled_all_real"}:
            raise ValueError("Declare the real-reference policy")
        if not Path(target.root).is_dir():
            raise FileNotFoundError(f"Target image root does not exist: {target.root}")
        frame, record = load_manifest(target.manifest)
        if record["split"] in {"train", "val"}:
            raise ValueError("Target suite cannot use training or source validation manifests")
        if kind == "df40":
            if not {"generator", "paradigm"} <= set(frame):
                raise ValueError("DF40 requires generator and paradigm annotations")
            if not target.breakdown:
                raise ValueError("DF40 requires subgroup reporting")
            if target.real_reference_policy == "source_matched" and "source_domain" not in frame:
                raise ValueError("DF40 source-matched policy requires source_domain")
        if kind == "celeb_df_v2":
            if target.primary_unit != "video":
                raise ValueError("Celeb-DF primary observation must be video")
            aggregate_videos(frame.assign(p_fake=0.5))
            coverage = record.get("coverage")
            if coverage and not coverage.get("complete"):
                raise ValueError("Incomplete Celeb-DF extraction coverage")
        populations[target.name], records[target.name] = frame, record
    if {"test", "test_d"} <= set(kinds):
        align(populations[kinds["test"]].assign(p_fake=0.5),
              populations[kinds["test_d"]].assign(p_fake=0.5))
    return populations, records, kinds


def evaluate_suite(model, targets, output, *, checkpoint_path, calibration_path,
                   image_size, predict_fn=None, input_contract=None, research_run=None,
                   mode=None, in_channels=None, positive_class="fake", device="cpu",
                   batch_size=32, workers=0, use_amp=True, bootstrap_draws=1000,
                   bootstrap_seed=42, confidence=0.95, plots=True, dry_run=False,
                   scope="benchmark", allow_unbound_calibration=False):
    """Reuse identical reports/artifacts for tensor, VLM, cached and graph scores.

    predict_fn(frame, root, **inference_kwargs) must return canonical p_fake rows.
    Dry runs validate metadata only, without loading a model or decoding images.
    """
    _, records, kinds = validate_targets(targets)
    if isinstance(predict_fn, dict):
        if set(predict_fn) != {target.name for target in targets} or not all(callable(fn) for fn in predict_fn.values()):
            raise ValueError("Prediction callbacks must match the exact suite target names")
    if image_size is None and model is not None and predict_fn is None:
        raise ValueError("Tensor inference requires an explicit image size")
    if scope not in {"benchmark", "pilot", "synthetic"}:
        raise ValueError("Declare benchmark, pilot or synthetic suite scope")
    if scope == "benchmark" and any(record["split"] != "test" and record["split"] != "test_d" for record in records.values()):
        raise ValueError("Benchmark scope cannot claim smoke or proxy manifests")
    calibration = json.loads(Path(calibration_path).read_text())
    checksum = digest_file(checkpoint_path)
    validate_calibration(calibration, checksum)
    if calibration.get("checkpoint_class1", "fake") != positive_class:
        raise ValueError("Calibration score orientation differs from the suite")
    contract = input_contract or prediction_contract(image_size, mode, in_channels, positive_class)
    if not allow_unbound_calibration and "input_contract_sha256" not in calibration:
        raise ValueError("Suite requires an input-bound calibration; explicitly allow historical unbound calibration if needed")
    if not allow_unbound_calibration and not re.fullmatch(r"[0-9a-f]{64}", str(contract.get("bundle_sha256", ""))):
        raise ValueError("Suite input contract must bind the exact model bundle_sha256")
    validate_input_contract(calibration, contract)
    validate_prediction_settings(contract, image_size, mode, in_channels, positive_class)
    if batch_size < 1 or workers < 0 or bootstrap_draws < 0 or (bootstrap_draws and bootstrap_draws < 20) or not 0 < confidence < 1:
        raise ValueError("Invalid suite inference or bootstrap settings")
    plan = {
        "schema": "faceforgery-evaluation-suite-v1", "scope": scope,
        "checkpoint_sha256": checksum, "calibration_sha256": digest_file(calibration_path),
        "input_contract": contract, "input_contract_sha256": digest(contract),
        "targets": {target.name: {**asdict(target), "manifest": str(target.manifest),
                                   "root": str(target.root), "certificate": records[target.name]}
                    for target in targets},
        "settings": {"device": device, "batch_size": batch_size, "workers": workers,
                     "amp": use_amp, "bootstrap_draws": bootstrap_draws,
                     "bootstrap_seed": bootstrap_seed, "confidence": confidence},
    }
    output = Path(output)
    if output.exists():
        raise FileExistsError("Use a fresh suite output directory")
    if dry_run:
        return {**plan, "state": "planned_not_executed"}
    output.mkdir(parents=True)
    write_json(output / "suite.json", plan)
    write_json(output / "status.json", {"state": "running"})
    reports = {}
    try:
        for target in targets:
            reports[target.name] = evaluate(
                model, target.manifest, target.root, output / target.name,
                checkpoint_path=checkpoint_path, calibration_path=calibration_path,
                image_size=image_size,
                predict_fn=predict_fn[target.name] if isinstance(predict_fn, dict) else predict_fn,
                input_contract=contract,
                research_run=research_run, mode=mode, in_channels=in_channels,
                positive_class=positive_class, device=device, batch_size=batch_size,
                workers=workers, use_amp=use_amp, primary_unit=target.primary_unit,
                breakdown=target.breakdown, real_reference_policy=target.real_reference_policy,
                bootstrap_draws=bootstrap_draws, bootstrap_seed=bootstrap_seed,
                confidence=confidence, plots=plots, target_name=target.name,
            )
        result = {**plan, "state": "complete", "results": reports}
        if {"test", "test_d"} <= set(kinds):
            clean, _ = load_predictions(output / kinds["test"] / "predictions.csv", calibration=calibration)
            degraded, _ = load_predictions(output / kinds["test_d"] / "predictions.csv", calibration=calibration)
            align(clean, degraded)
            result["paired_test_d"] = {"quantity": "test_d_minus_test_auc",
                                      "estimate": reports[kinds["test_d"]]["frame"]["auc"] - reports[kinds["test"]]["frame"]["auc"]}
            if bootstrap_draws:
                result["paired_test_d"]["interval"] = grouped_auc_interval(
                    clean, degraded, draws=bootstrap_draws, seed=bootstrap_seed, confidence=confidence
                )
        write_json(output / "suite_metrics.json", result)
        files = ["suite.json", "suite_metrics.json"] + [f"{target.name}/status.json" for target in targets]
        write_json(output / "status.json", {"state": "complete", "targets": list(reports),
                   "artifacts": {name: digest_file(output / name) for name in files}})
        return result
    except Exception as error:
        write_json(output / "status.json", {"state": "failed", "completed_targets": list(reports),
                   "error": f"{type(error).__name__}: {error}"})
        raise


def legacy_contract(run, config):
    base = prediction_contract(config.image_size, run.fourier_mode, config.in_channels)
    config_hash = digest_file(run.run_dir / "results" / "run_config.json")
    return {**base, "run_config_sha256": config_hash,
            "bundle_sha256": digest({"checkpoint": digest_file(run.weights_path),
                                     "run_config": config_hash, "effective_config": config.to_dict()})}


def legacy_identity(run, config):
    settings = config.to_dict()
    settings.pop("seed")
    settings.pop("seeds", None)
    condition = {
        "effective_training_without_seed": settings,
        "input_contract": prediction_contract(config.image_size, run.fourier_mode, config.in_channels),
        "historical_training_data": "not recorded; not certified by current manifests",
        "evaluation_code_sha256": source_identity()["code_sha256"],
    }
    return {"name": f"{run.model_family}_{run.fourier_mode}_{run.regime}", "seed": run.seed,
            "condition": condition, "condition_sha256": digest(condition),
            "run_id": digest([digest_file(run.weights_path), digest_file(run.run_dir / "results" / "run_config.json")]),
            "historical_provenance_status": "training manifest hashes unavailable"}


def pilot_contract(root, record):
    root = Path(root)
    return {**prediction_contract(record["config"]["training"]["image_size"]),
            "run_config_sha256": digest_file(root / "run.json"),
            "bundle_sha256": digest({"checkpoint": digest_file(root / "best.pt"),
                                     "run_config": digest_file(root / "run.json")})}


def read_suite_config(path):
    from src.experimental.configuration import read_document

    path = Path(path).resolve()
    value = read_document(path)
    allowed = {"model", "calibration", "targets", "output", "scope", "inference"}
    if not isinstance(value, dict) or set(value) - allowed or not {"model", "calibration", "targets", "output"} <= set(value):
        raise ValueError("Invalid suite config fields")

    def location(value):
        expanded = os.path.expandvars(str(value))
        if "$" in expanded:
            raise ValueError(f"Unresolved environment variable in suite path: {value}")
        result = Path(expanded).expanduser()
        return str(result if result.is_absolute() else path.parent / result)

    value["calibration"], value["output"] = location(value["calibration"]), location(value["output"])
    model = value["model"]
    if not isinstance(model, dict) or model.get("type") not in {"legacy", "stack_b", "experimental"}:
        raise ValueError("Suite model must declare legacy, stack_b or experimental and its path")
    allowed_model = {"type", "path", "family", "options"} if model["type"] == "experimental" else {"type", "path"}
    required_model = {"type", "path", "family"} if model["type"] == "experimental" else {"type", "path"}
    if set(model) - allowed_model or not required_model <= set(model):
        raise ValueError("Invalid suite model fields")
    if "options" in model and not isinstance(model["options"], dict):
        raise ValueError("Experimental model options must be a mapping")
    model["path"] = location(model["path"])
    if not isinstance(value["targets"], list) or not value["targets"]:
        raise ValueError("Suite requires a nonempty list of targets")
    targets = []
    for target in value["targets"]:
        target = dict(target)
        target["manifest"], target["root"] = location(target["manifest"]), location(target["root"])
        kind = target.get("kind", target["name"])
        target.setdefault("primary_unit", "video" if kind == "celeb_df_v2" else "frame")
        target.setdefault("breakdown", kind == "df40")
        try:
            targets.append(TargetSpec(**target))
        except TypeError as error:
            raise ValueError("Invalid target spec fields") from error
    value["targets"] = targets
    return value


def run_suite_config(path, *, device=None, workers=None, output=None, execute=False):
    """Lazy model rebuild: dry runs never instantiate a network."""
    import torch

    spec = read_suite_config(path)
    if output is not None:
        spec["output"] = str(Path(output).expanduser().resolve())
    settings = dict(spec.get("inference", {}))
    allowed = {"device", "batch_size", "workers", "use_amp", "bootstrap_draws", "bootstrap_seed", "confidence", "plots", "allow_unbound_calibration"}
    if set(settings) - allowed:
        raise ValueError("Unknown suite inference settings")
    settings["device"] = device or settings.get("device", "cpu")
    if workers is not None:
        settings["workers"] = workers
    selected = spec["model"]
    model, predict_fn = None, None
    if selected["type"] == "legacy":
        from src.pipelines.checkpoints import run_from_checkpoint, config_from_run, load_model_from_run

        run = run_from_checkpoint(selected["path"])
        config = config_from_run(run)
        checkpoint = run.weights_path
        contract = legacy_contract(run, config)
        identity = legacy_identity(run, config)
        image_size, mode, channels = config.image_size, run.fourier_mode, config.in_channels
    elif selected["type"] == "stack_b":
        from .engine import load_pilot
        from .identity import evaluation_identity

        root = Path(selected["path"])
        record = json.loads((root / "run.json").read_text())
        checkpoint = root / "best.pt"
        contract = pilot_contract(root, record)
        identity = evaluation_identity(record)
        image_size, mode, channels = record["config"]["training"]["image_size"], None, None
    else:
        from src.experimental.orchestration import describe_run, load_evaluator, validate_options

        validate_options(selected["family"], selected.get("options"),
                         target_names=[target.name for target in spec["targets"]])
        description = describe_run(selected["family"], selected["path"], options=selected.get("options"))
        checkpoint, contract = description["checkpoint_path"], description["input_contract"]
        identity = description["research_run"]
        image_size, mode, channels = description["image_size"], None, None
        if settings.get("allow_unbound_calibration"):
            raise ValueError("Experimental suites require input-bound calibration")
    calibration = json.loads(Path(spec["calibration"]).read_text())
    positive_class = calibration.get("checkpoint_class1", "fake")
    if selected["type"] == "experimental" and positive_class != "fake":
        raise ValueError("Experimental suite calibration must use fake-is-1 scores")
    if selected["type"] != "experimental":
        contract["checkpoint_class1"] = positive_class
    arguments = dict(checkpoint_path=checkpoint,
        calibration_path=spec["calibration"], image_size=image_size, input_contract=contract,
        research_run=identity, mode=mode, in_channels=channels, positive_class=positive_class,
        scope=spec.get("scope", "benchmark"), **settings,
    )
    planned = evaluate_suite(None, spec["targets"], spec["output"], dry_run=True, **arguments)
    if selected["type"] == "experimental" and selected["family"] in {"metric", "graph", "moe"}:
        from src.experimental.orchestration import preflight_cached

        populations = {target.name: (load_manifest(target.manifest)[0], target.root) for target in spec["targets"]}
        predict_fn = preflight_cached(selected["family"], selected["path"], selected.get("options"), populations)
    if not execute:
        return planned
    if selected["type"] == "legacy":
        model = load_model_from_run(run, torch.device(settings["device"]))
    elif selected["type"] == "stack_b":
        model, _ = load_pilot(root, settings["device"])
    else:
        if predict_fn is None:
            model, predict_fn = load_evaluator(
                selected["family"], selected["path"], device=settings["device"],
                options=selected.get("options"), target_names=[target.name for target in spec["targets"]],
            )
    return evaluate_suite(model, spec["targets"], spec["output"], predict_fn=predict_fn,
                          dry_run=False, **arguments)
