"""One CLI contract for trained image, VLM and cached experimental scorers."""

from __future__ import annotations

import copy
from importlib import import_module
from pathlib import Path
import re

from src.robustness.artifacts import save_predictions
from src.robustness.inference import bind_population, calibrate, predict
from src.robustness.manifests import load_manifest
from src.robustness.provenance import digest, digest_file, write_json

from .configuration import read_document


FAMILIES = {"reconstruction", "sbi", "metric", "graph", "vlm", "moe"}
DESCRIPTION_MODULES = {
    "reconstruction": "src.experimental.reconstruction.training",
    "sbi": "src.experimental.sbi",
    "metric": "src.experimental.feature_training",
    "graph": "src.experimental.feature_training",
    "vlm": "src.experimental.vlm.inference",
    "moe": "src.experimental.moe.inference",
}


def _family(value):
    if not isinstance(value, str) or value not in FAMILIES:
        raise ValueError(f"Unknown experimental family: {value}")
    return value


def describe_run(family, run_dir, *, options=None):
    """Verify immutable artifacts without rebuilding any network."""
    family = _family(family)
    options = options or {}
    module = import_module(DESCRIPTION_MODULES[family])
    arguments = {}
    if family in {"metric", "graph"} and "protocol" in options:
        arguments["protocol"] = options["protocol"]
    if family == "moe" and "method" in options:
        arguments["method"] = options["method"]
    description = dict(module.describe_run(run_dir, **arguments))
    required = {"checkpoint_path", "input_contract", "research_run", "image_size"}
    if not required <= set(description):
        raise ValueError("Experimental run description is incomplete")
    checkpoint = Path(description["checkpoint_path"])
    if not checkpoint.is_file():
        raise FileNotFoundError(f"Missing experimental checkpoint inventory: {checkpoint}")
    contract = description["input_contract"]
    if not isinstance(contract, dict) or not re.fullmatch(r"[0-9a-f]{64}", str(contract.get("bundle_sha256", ""))):
        raise ValueError("Experimental input contract must bind an exact bundle_sha256")
    if contract.get("checkpoint_class1", "fake") != "fake":
        raise ValueError("Experimental scorers must declare class 1 as fake")
    if contract.get("family", family) != family:
        raise ValueError("Configured family differs from the trained artifact")
    size = description["image_size"]
    if size is not None and (not isinstance(size, int) or isinstance(size, bool) or size < 1):
        raise ValueError("Invalid experimental image size")
    identity = description["research_run"]
    if not isinstance(identity, dict) or not {"name", "seed", "condition", "condition_sha256"} <= set(identity):
        raise ValueError("Experimental run needs controlled seed-condition metadata")
    if digest(identity["condition"]) != identity["condition_sha256"]:
        raise ValueError("Experimental condition changed")
    description["checkpoint_path"] = checkpoint
    return description


def validate_options(family, options, *, target_names=None):
    family = _family(family)
    if options is None:
        options = {}
    if not isinstance(options, dict):
        raise ValueError("Experimental evaluation options must be a mapping")
    allowed = ({"cache", "target_caches", "protocol"} if family in {"metric", "graph"}
               else {"sources", "target_caches", "method"} if family == "moe" else set())
    if set(options) - allowed:
        raise ValueError("Unknown experimental evaluation options")
    if family in {"metric", "graph", "moe"}:
        caches = options.get("target_caches")
        if target_names is not None:
            if not isinstance(caches, dict) or set(caches) != set(target_names):
                raise ValueError("Target cache names differ from suite targets")
        elif caches is not None:
            if not isinstance(caches, dict) or set(caches) != {"source_val"}:
                raise ValueError("Calibration requires one explicit source_val cache")
        elif ("sources" if family == "moe" else "cache") not in options:
            raise ValueError("Cached calibration requires explicit validation cache sources")
    return options


def load_evaluator(family, run_dir, *, device="cpu", options=None, target_names=None):
    """Return a tensor model or named prediction callbacks for explicit caches."""
    family = _family(family)
    options = validate_options(family, options, target_names=target_names)
    if family in {"reconstruction", "sbi"}:
        if options:
            raise ValueError("Image-model evaluation does not accept cache options")
        module = import_module(DESCRIPTION_MODULES[family])
        return module.load_model(run_dir, device), None
    if family == "vlm":
        if options:
            raise ValueError("VLM inference uses its immutable processor and score contract")
        module = import_module(DESCRIPTION_MODULES[family])
        return None, module.load_predictor(run_dir, device)
    if family in {"metric", "graph"}:
        if set(options) - {"cache", "target_caches", "protocol"}:
            raise ValueError("Unknown feature-scorer evaluation options")
        caches = options.get("target_caches")
        if caches is None and "cache" in options:
            caches = {"source_val": options["cache"]}
        if not isinstance(caches, dict) or not caches:
            raise ValueError("Cached scorers require explicit cache paths per target")
        if target_names is not None and set(caches) != set(target_names):
            raise ValueError("Target cache names differ from suite targets")
        module = import_module("src.experimental.feature_training")
        callbacks = module.feature_cache_predictor(run_dir, caches, protocol=options.get("protocol"))
        return None, callbacks
    if set(options) - {"sources", "target_caches", "method"}:
        raise ValueError("Unknown MoE evaluation options")
    module = import_module(DESCRIPTION_MODULES[family])
    if "target_caches" in options:
        caches = options["target_caches"]
        if not isinstance(caches, dict) or not caches:
            raise ValueError("MoE requires explicit expert caches per target")
        if target_names is not None and set(caches) != set(target_names):
            raise ValueError("Target expert-cache names differ from suite targets")
        return None, {name: module.load_predictor(run_dir, sources, device=device, method=options.get("method", "router"))
                      for name, sources in caches.items()}
    if "sources" not in options:
        raise ValueError("MoE calibration requires explicit validation expert sources")
    return None, module.load_predictor(run_dir, options["sources"], device=device, method=options.get("method", "router"))


def preflight_cached(family, run_dir, options, populations, *, calibration=False):
    """Check cached target identities and image bytes without creating a network."""
    if family not in {"metric", "graph", "moe"}:
        return None
    options = validate_options(family, options, target_names=None if calibration else populations)
    if family in {"metric", "graph"}:
        _, callbacks = load_evaluator(family, run_dir, options=options,
                                      target_names=None if calibration else populations)
        for name, (frame, root) in populations.items():
            callbacks[name].validate_population(frame, root)
        return callbacks
    module = import_module(DESCRIPTION_MODULES[family])
    for name, (frame, root) in populations.items():
        sources = options["target_caches"][name] if "target_caches" in options else options["sources"]
        module.validate_sources(run_dir, sources, frame=frame, root=root)
    return None


def train_experiment(path, *, family=None, device=None, seed=None, output=None, resume=False, execute=False):
    config = read_document(path)
    declared = config.pop("family", None)
    family = _family(family or declared)
    if declared is not None and declared != family:
        raise ValueError("CLI family differs from config family")
    if "run_dir" in config:
        if "output_dir" in config and config["run_dir"] != config["output_dir"]:
            raise ValueError("Conflicting experimental output locations")
        config["output_dir"] = config.pop("run_dir")
    if "output_dir" not in config:
        raise ValueError("Declare experimental run_dir or output_dir")
    if output is not None:
        config["output_dir"] = str(Path(output).expanduser().resolve())
    config = copy.deepcopy(config)
    if seed is not None:
        if seed < 0:
            raise ValueError("Experimental seed must be nonnegative")
        config["seed"] = seed
    selected_device = device or config.get("training", {}).get("device", "cpu")
    if family not in {"reconstruction", "sbi"}:
        config.setdefault("training", {})["device"] = selected_device
        config["training"]["resume"] = bool(resume or config["training"].get("resume", False))
    plan = {"state": "planned_not_executed", "family": family, "run_dir": config["output_dir"],
            "config": config, "config_sha256": digest(config), "device": selected_device,
            "selection": "source train and validation only; benchmark targets evaluated separately"}
    if not execute:
        return plan
    if family in {"reconstruction", "sbi"}:
        module = import_module(DESCRIPTION_MODULES[family])
        result = module.fit(config, device=selected_device, resume=resume)
    elif family == "metric":
        result = import_module("src.experimental.feature_training").fit_metric_run(config)
    elif family == "graph":
        result = import_module("src.experimental.graph_training").fit_graph_run(config)
    else:
        result = import_module(f"src.experimental.{family}.training").fit(config)
    return {"state": "complete", "family": family, "run_dir": str(config["output_dir"]),
            "result": result if isinstance(result, dict) else {"run_dir": str(result)}}


def calibrate_experiment(family, run_dir, manifest, root, output, *, options=None,
                         device="cpu", batch_size=32, workers=0, use_amp=True,
                         policy="youden", execute=False):
    """Freeze source-validation thresholds with exact run and predictor binding."""
    frame, certificate = load_manifest(manifest)
    if certificate["split"] != "val":
        raise ValueError("Experimental calibration accepts source validation only")
    output = Path(output)
    if output.exists():
        raise FileExistsError("Use a fresh experimental calibration directory")
    if not Path(root).is_dir():
        raise FileNotFoundError(f"Source image root does not exist: {root}")
    options = validate_options(family, options)
    description = describe_run(family, run_dir, options=options)
    for key, observed in (("calibration_manifest_sha256", certificate["manifest_sha256"]),
                          ("calibration_sample_ids_sha256", digest(sorted(frame.sample_id.astype(str))))):
        if key in description and description[key] != observed:
            raise ValueError("Calibration population differs from the held-out source selection population")
    contract, checkpoint = description["input_contract"], description["checkpoint_path"]
    if batch_size < 1 or workers < 0 or policy not in {"youden", "balanced_accuracy"}:
        raise ValueError("Invalid experimental calibration settings")
    plan = {"state": "planned_not_executed", "family": family, "manifest": certificate,
            "checkpoint_sha256": digest_file(checkpoint), "input_contract": contract,
            "input_contract_sha256": digest(contract), "threshold_policy": policy,
            "output": str(output), "selection_split": "val"}
    callback = preflight_cached(family, run_dir, options, {"source_val": (frame, root)}, calibration=True)
    if not execute:
        return plan
    model = None
    if callback is None:
        model, callback = load_evaluator(family, run_dir, device=device, options=options)
    if isinstance(callback, dict):
        if set(callback) != {"source_val"}:
            raise ValueError("Calibration requires one explicit source_val cache")
        callback = callback["source_val"]
    arguments = {"image_size": description["image_size"], "mode": None, "in_channels": None,
                 "positive_class": "fake", "device": device, "batch_size": batch_size,
                 "workers": workers, "use_amp": use_amp}
    predictions = (callback(frame, root, **arguments) if callback is not None
                   else predict(model, frame, root, **arguments))
    predictions = bind_population(predictions, frame)
    output.mkdir(parents=True)
    checksum = digest_file(checkpoint)
    save_predictions(output / "validation_predictions.csv", predictions, manifest_record=certificate,
                     model_sha256=checksum, metadata={"origin": "shared experimental calibration", "input_contract": contract})
    result = calibrate(predictions, manifest_record=certificate, output=output / "calibration.json",
                       model_sha256=checksum, policy=policy, input_contract=contract)
    files = ("validation_predictions.csv", "validation_predictions.csv.json", "calibration.json")
    write_json(output / "status.json", {"state": "complete", "artifacts": {name: digest_file(output / name) for name in files}})
    return result


def cache_features(args):
    frame, certificate = load_manifest(args.manifest)
    if not Path(args.root).is_dir():
        raise FileNotFoundError(f"Feature image root does not exist: {args.root}")
    plan = {"state": "planned_not_executed", "checkpoint_sha256": digest_file(args.checkpoint),
            "manifest": certificate, "rows": len(frame), "cache_root": str(Path(args.output).resolve())}
    if not args.execute:
        return plan
    from .features import extract_features, open_cache

    output = extract_features(args.checkpoint, args.manifest, args.root, args.output,
                              device=args.device, batch_size=args.batch_size, workers=args.workers,
                              use_amp=not args.no_amp)
    store = open_cache(output)
    return {**plan, "state": "complete", "cache": str(output), "identity": store.identity}


def register(commands):
    parser = commands.add_parser("experimental", help="Shared experimental train, cache, calibrate and evaluate interface")
    actions = parser.add_subparsers(dest="experimental_action", required=True)
    train = actions.add_parser("train", help="Review configuration; append --execute to train")
    train.add_argument("--config", required=True)
    train.add_argument("--family", choices=sorted(FAMILIES))
    train.add_argument("--device")
    train.add_argument("--seed", type=int, help="Override the configured realization seed")
    train.add_argument("--output", help="Override the run directory; use a distinct directory per seed")
    train.add_argument("--resume", action="store_true")
    train.add_argument("--execute", action="store_true")
    cache = actions.add_parser("cache", help="Build provenance-bound frozen legacy features")
    for key in ("checkpoint", "manifest", "root", "output"):
        cache.add_argument("--" + key, required=True)
    cache.add_argument("--device", default="cpu")
    cache.add_argument("--batch-size", type=int, default=8)
    cache.add_argument("--workers", type=int, default=0)
    cache.add_argument("--no-amp", action="store_true")
    cache.add_argument("--execute", action="store_true")
    calibration = actions.add_parser("calibrate", help="Freeze input-bound validation thresholds for any experimental run")
    calibration.add_argument("--family", choices=sorted(FAMILIES), required=True)
    for key in ("run", "manifest", "root", "output"):
        calibration.add_argument("--" + key, required=True)
    calibration.add_argument("--options", help="JSON/YAML with explicit source feature cache or expert sources")
    calibration.add_argument("--device", default="cpu")
    calibration.add_argument("--batch-size", type=int, default=32)
    calibration.add_argument("--workers", type=int, default=0)
    calibration.add_argument("--no-amp", action="store_true")
    calibration.add_argument("--threshold-policy", choices=["youden", "balanced_accuracy"], default="youden")
    calibration.add_argument("--execute", action="store_true")
    evaluation = actions.add_parser("evaluate", help="Run the shared four-target suite")
    evaluation.add_argument("--config", required=True)
    evaluation.add_argument("--device")
    evaluation.add_argument("--workers", type=int)
    evaluation.add_argument("--output")
    evaluation.add_argument("--execute", action="store_true")


def run(args):
    if args.experimental_action == "train":
        return train_experiment(args.config, family=args.family, device=args.device,
                                seed=args.seed, output=args.output, resume=args.resume, execute=args.execute)
    if args.experimental_action == "cache":
        return cache_features(args)
    if args.experimental_action == "calibrate":
        options = read_document(args.options) if args.options else None
        return calibrate_experiment(args.family, args.run, args.manifest, args.root, args.output,
                                    options=options, device=args.device, batch_size=args.batch_size,
                                    workers=args.workers, use_amp=not args.no_amp,
                                    policy=args.threshold_policy, execute=args.execute)
    from src.robustness.suite import run_suite_config

    return run_suite_config(args.config, device=args.device, workers=args.workers,
                            output=args.output, execute=args.execute)
