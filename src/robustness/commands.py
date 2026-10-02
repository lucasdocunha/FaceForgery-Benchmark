"""Additional CLI actions for existing artifacts and bounded experiment planning."""

from __future__ import annotations
import copy
import json
import time
from pathlib import Path
import numpy as np
import pandas as pd
import yaml
from .artifacts import load_predictions, import_legacy
from .experiments import read_config
from .provenance import digest_file, source_identity, write_json, write_csv
from .statistics import checked_predictions, generator_metrics


def register(commands):
    p = commands.add_parser(
        "import-predictions",
        help="Certify existing legacy exports without inference or training",
    )
    for key in ["source", "manifest", "output", "checkpoint"]:
        p.add_argument("--" + key, required=True)
    p.add_argument("--source-labels", required=True, choices=["fake-is-1", "real-is-1"])
    p.add_argument("--class-one", required=True, choices=["fake", "real"])
    p.add_argument("--acknowledge-row-order", action="store_true")
    p = commands.add_parser(
        "calibrate-export",
        help="Freeze thresholds from certified source-validation predictions",
    )
    p.add_argument("--predictions", required=True)
    p.add_argument("--output", required=True)
    p.add_argument("--threshold-policy", choices=["balanced_accuracy", "youden"], default="balanced_accuracy")
    p.add_argument("--input-contract", help="JSON score/preprocessing contract, including checkpoint bundle identity")
    p = commands.add_parser(
        "expand-plan", help="Write explicit ablation configs; never submit or run jobs"
    )
    p.add_argument("--template", required=True)
    p.add_argument("--output", required=True)
    p.add_argument("--plan", default="configs/research/ablation-plan.json")
    p.add_argument("--seeds", type=int, nargs="+", default=[42])
    p.add_argument("--variants", nargs="+")
    p = commands.add_parser(
        "profile",
        help="Measure model/encoding compute only; excludes I/O and face extraction",
    )
    p.add_argument("--run", required=True)
    p.add_argument("--output", required=True)
    p.add_argument("--device", default="cpu")
    p.add_argument("--batch-size", type=int, default=1)
    p.add_argument("--warmup", type=int, default=10)
    p.add_argument("--iterations", type=int, default=50)
    p = commands.add_parser(
        "export-xai",
        help="Explicit identity bridge to the retained legacy XAI preparation format",
    )
    p.add_argument("--manifest", required=True)
    p.add_argument("--predictions", required=True)
    p.add_argument("--calibration", required=True)
    p.add_argument("--output", required=True)
    p = commands.add_parser("report-generators")
    p.add_argument("--predictions", required=True)
    p.add_argument("--calibration", required=True)
    p.add_argument("--output", required=True)
    p = commands.add_parser(
        "aggregate-seeds",
        help="Require exactly the declared seed population per dataset/variant",
    )
    p.add_argument("--evaluations", nargs="+", required=True)
    p.add_argument("--seeds", type=int, nargs="+", required=True)
    p.add_argument("--output", required=True)


def run(args):
    if args.command == "import-predictions":
        return import_legacy(
            args.source,
            args.manifest,
            args.output,
            checkpoint=args.checkpoint,
            source_labels=args.source_labels,
            checkpoint_class1=args.class_one,
            acknowledge_row_order=args.acknowledge_row_order,
        )
    if args.command == "calibrate-export":
        from .inference import calibrate

        frame, record = load_predictions(args.predictions)
        return calibrate(
            frame,
            manifest_record=record,
            output=args.output,
            model_sha256=record["model_sha256"],
            checkpoint_class1=record["checkpoint_class1"],
            policy=getattr(args, "threshold_policy", "balanced_accuracy"),
            input_contract=(json.loads(Path(args.input_contract).read_text())
                            if getattr(args, "input_contract", None) else record.get("input_contract")),
        )
    if args.command == "expand-plan":
        return expand(args)
    if args.command == "profile":
        return profile(args)
    if args.command == "export-xai":
        return export_xai(args)
    if args.command == "report-generators":
        calibration = json.loads(Path(args.calibration).read_text())
        frame, record = load_predictions(args.predictions, calibration=calibration)
        result = generator_metrics(frame, calibration["frame_threshold"])
        result["prediction_certificate"] = record
        write_json(args.output, result)
        return result
    return aggregate_seeds(args)


def expand(args):
    cfg = read_config(args.template)
    spec = json.loads(Path(args.plan).read_text())
    choices = {v["name"]: v for v in spec["variants"]}
    names = args.variants or list(choices)
    if len(set(args.seeds)) != len(args.seeds) or not args.seeds or min(args.seeds) < 0:
        raise ValueError("Declare distinct nonnegative seeds")
    if len(set(names)) != len(names) or set(names) - set(choices):
        raise ValueError("Unknown or duplicate ablation variants")
    root = Path(args.output)
    if root.exists():
        raise FileExistsError("Plan directory exists; use a new frozen location")
    root.mkdir(parents=True)
    rows = []
    for name in names:
        variant = choices[name]
        for seed in args.seeds:
            item = copy.deepcopy(cfg)
            item["name"] = name
            item["training"]["seed"] = seed
            item["model"]["kind"] = variant["kind"]
            for k, v in variant.items():
                if k not in {"name", "kind"}:
                    item["training"][k] = v
            path = root / f"{name}_seed{seed}.yaml"
            path.write_text(yaml.safe_dump(item, sort_keys=False))
            read_config(path)
            rows.append(
                {
                    "variant": name,
                    "seed": seed,
                    "config": str(path),
                    "config_sha256": digest_file(path),
                    "review_command": f"python research_cli.py train --config {path}",
                    "execution_requires": "append --execute --device cuda:0 only after review",
                }
            )
    result = {
        "state": "planned_not_executed",
        "source_plan_sha256": digest_file(args.plan),
        "jobs": rows,
        "selection": "source validation only; no selection by target test performance",
    }
    write_json(root / "plan.json", result)
    return result


def profile(args):
    import torch
    from .engine import load_pilot

    if min(args.batch_size, args.iterations) < 1 or args.warmup < 0:
        raise ValueError("Invalid timing budget")
    output = Path(args.output)
    if output.exists():
        raise FileExistsError("Use a new timing record")
    model, record = load_pilot(args.run, args.device)
    device = torch.device(args.device)
    size = record["config"]["training"]["image_size"]
    x = torch.zeros(args.batch_size, 3, size, size, device=device)

    def sync():
        if device.type == "cuda":
            torch.cuda.synchronize(device)

    with torch.inference_mode():
        for _ in range(args.warmup):
            model(x)
        sync()
        if device.type == "cuda":
            torch.cuda.reset_peak_memory_stats(device)
        timings = []
        for _ in range(args.iterations):
            sync()
            start = time.perf_counter()
            model(x)
            sync()
            timings.append((time.perf_counter() - start) * 1000)
    result = {
        "checkpoint_sha256": digest_file(Path(args.run) / "best.pt"),
        "device": str(device),
        "device_name": torch.cuda.get_device_name(device)
        if device.type == "cuda"
        else "CPU",
        "batch_size": args.batch_size,
        "input_shape": list(x.shape),
        "warmup": args.warmup,
        "iterations": args.iterations,
        "median_batch_ms": float(np.median(timings)),
        "p95_batch_ms": float(np.quantile(timings, 0.95)),
        "images_per_second_at_median": float(
            args.batch_size * 1000 / np.median(timings)
        ),
        "parameters": sum(p.numel() for p in model.parameters()),
        "peak_cuda_allocated_bytes": torch.cuda.max_memory_allocated(device)
        if device.type == "cuda"
        else None,
        "scope": "model + internal RGB/FFT encoding on resident synthetic tensors; EXCLUDES data transfer, I/O, decoding, face extraction and training",
        "raw_batch_ms": timings,
        "software": source_identity(),
    }
    write_json(output, result)
    return result


def export_xai(args):
    from .manifests import load_manifest

    manifest, certificate = load_manifest(args.manifest)
    calibration = json.loads(Path(args.calibration).read_text())
    frame, record = load_predictions(args.predictions, calibration=calibration)
    if certificate["manifest_sha256"] != record["manifest_sha256"]:
        raise ValueError("Wrong XAI manifest")
    frame = frame.set_index("sample_id").loc[manifest.sample_id].reset_index()
    if not np.array_equal(frame.label.to_numpy(), manifest.label.to_numpy()):
        raise ValueError("XAI labels disagree")
    root = Path(args.output)
    if root.exists():
        raise FileExistsError("Use a new XAI export directory")
    root.mkdir(parents=True)
    write_csv(root / "manifest.csv", manifest[["img_name", "label"]])
    write_csv(
        root / "predictions.csv",
        pd.DataFrame(
            {
                "id": np.arange(len(frame)),
                "y_true": frame.label,
                "prob_pos": frame.p_fake,
                "y_pred": (frame.p_fake >= calibration["frame_threshold"]).astype(int),
            }
        ),
    )
    write_csv(
        root / "identity_map.csv",
        pd.DataFrame(
            {
                "id": np.arange(len(frame)),
                "sample_id": manifest.sample_id,
                "group_id": manifest.group_id,
            }
        ),
    )
    result = {
        "prediction_certificate": record,
        "canonical_manifest_sha256": certificate["manifest_sha256"],
        "exported_manifest_sha256": digest_file(root / "manifest.csv"),
        "warning": "Compatibility bridge only, not executed attributions. Legacy XAI still requires its model config and full val/test roster. Use this exported manifest order, never an unrelated CSV.",
    }
    write_json(root / "provenance.json", result)
    return result


def aggregate_seeds(args):
    expected = set(args.seeds)
    if len(expected) != len(args.seeds) or not expected:
        raise ValueError("Distinct expected seeds required")
    rows = []
    for name in args.evaluations:
        root = Path(name)
        status = json.loads((root / "status.json").read_text())
        if status.get("state") != "complete":
            raise ValueError(f"Incomplete evaluation {root}")
        for artifact, checksum in status["artifacts"].items():
            if (
                Path(artifact).name != artifact
                or digest_file(root / artifact) != checksum
            ):
                raise ValueError("Modified evaluation artifacts")
        report = json.loads((root / "metrics.json").read_text())
        identity = report.get("research_run")
        if not identity or "condition_sha256" not in identity:
            raise ValueError(
                "Seed aggregation requires complete pilot condition metadata; do not infer settings from filenames"
            )
        from .provenance import digest

        if digest(identity["condition"]) != identity["condition_sha256"]:
            raise ValueError("Experimental-condition metadata changed")
        rows.append(
            {
                "variant": identity["name"],
                "dataset": report["manifest"]["dataset"],
                "split": report["manifest"]["split"],
                "seed": identity["seed"],
                "unit": report.get("primary_unit", "frame"),
                "target": report.get("target_name", report["manifest"]["dataset"]),
                "auc": report[report.get("primary_unit", "frame")]["auc"],
                "eer": report[report.get("primary_unit", "frame")].get("eer"),
                "f1": report[report.get("primary_unit", "frame")].get("f1"),
                "accuracy": report[report.get("primary_unit", "frame")].get("accuracy"),
                "threshold_policy": report["calibration"].get("policy", "balanced_accuracy"),
                "model_sha256": report["checkpoint_sha256"],
                "condition_sha256": identity["condition_sha256"],
                "manifest_sha256": report["manifest"]["manifest_sha256"],
            }
        )
    frame = pd.DataFrame(rows)
    results = []
    for key, group in frame.groupby(["variant", "dataset", "split", "unit", "target"], sort=True):
        if group.seed.duplicated().any() or set(group.seed) != expected:
            raise ValueError(f"Missing/extra/duplicate seeds for {key}")
        if group.manifest_sha256.nunique() != 1:
            raise ValueError("Seed runs evaluated on different populations")
        if group.condition_sha256.nunique() != 1:
            raise ValueError(
                "Seed runs differ in model, training, source data or software conditions"
            )
        if group.threshold_policy.nunique() != 1:
            raise ValueError("Seed runs use different threshold policies")
        if group.auc.isna().any():
            raise ValueError("Undefined AUC in seed table")
        results.append(
            {
                "variant": key[0],
                "dataset": key[1],
                "split": key[2],
                "unit": key[3],
                "target": key[4],
                "n_seeds": len(group),
                "mean_auc": float(group.auc.mean()),
                "sample_std_auc": float(group.auc.std(ddof=1))
                if len(group) > 1
                else None,
                "manifest_sha256": group.manifest_sha256.iloc[0],
                **{f"mean_{metric}": float(group[metric].mean()) if group[metric].notna().all() else None
                   for metric in ("eer", "f1", "accuracy")},
            }
        )
    result = {
        "expected_seeds": sorted(expected),
        "results": results,
        "runs": rows,
        "uncertainty": "Sample standard deviation across these independent seed runs; not a confidence interval and not image bootstrap uncertainty.",
    }
    if Path(args.output).exists():
        raise FileExistsError("Use a new seed report")
    write_json(args.output, result)
    return result
