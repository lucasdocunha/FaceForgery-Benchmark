"""Auditable research entry point. Training is a dry run unless --execute is given."""

from __future__ import annotations
import argparse
import json
from pathlib import Path
from PIL import ImageFile

ImageFile.LOAD_TRUNCATED_IMAGES = True


def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__)
    commands = p.add_subparsers(dest="command", required=True)
    cv = commands.add_parser(
        "convert-manifest",
        help="Convert explicitly declared labels into a new canonical manifest",
    )
    cv.add_argument("--source", required=True)
    cv.add_argument("--output", required=True)
    cv.add_argument("--dataset", required=True)
    cv.add_argument("--split", required=True)
    cv.add_argument("--label-column", required=True)
    cv.add_argument("--convention", required=True, choices=["fake-is-1", "real-is-1"])
    cv.add_argument("--group-column")
    au = commands.add_parser("audit-images")
    au.add_argument("--manifest", required=True)
    au.add_argument("--root", required=True)
    au.add_argument("--output", required=True)
    au.add_argument("--hash-images", action="store_true")
    pl = commands.add_parser("plan")
    pl.add_argument("--config", required=True)
    pl.add_argument("--output")
    tr = commands.add_parser("train")
    tr.add_argument("--config", required=True)
    tr.add_argument("--device", default="cpu")
    tr.add_argument("--execute", action="store_true")
    tr.add_argument("--resume", action="store_true")
    ev = commands.add_parser("evaluate")
    ev.add_argument("--run", required=True)
    ev.add_argument("--manifest", required=True)
    ev.add_argument("--root", required=True)
    ev.add_argument("--output", required=True)
    ev.add_argument("--device", default="cpu")
    ev.add_argument("--batch-size", type=int, default=32)
    ev.add_argument("--workers", type=int, default=0)
    suite = commands.add_parser("evaluate-suite", help="Audited four-target suite; dry run unless --execute")
    suite.add_argument("--config", required=True)
    suite.add_argument("--device")
    suite.add_argument("--workers", type=int)
    suite.add_argument("--output", help="Override the fresh suite output directory, for job-local scratch")
    suite.add_argument("--execute", action="store_true")
    cr = commands.add_parser("calibrate-run", help="Freeze input-bound source-validation calibration for a Stack B run")
    for key in ("run", "manifest", "root", "output"):
        cr.add_argument("--" + key, required=True)
    cr.add_argument("--device", default="cpu")
    cr.add_argument("--batch-size", type=int, default=32)
    cr.add_argument("--workers", type=int, default=0)
    cr.add_argument("--threshold-policy", choices=["balanced_accuracy", "youden"], default="balanced_accuracy")
    le = commands.add_parser(
        "evaluate-legacy",
        help="Evaluate existing checkpoint after source calibration; model class-1 must be explicit",
    )
    le.add_argument("--checkpoint", required=True)
    le.add_argument("--manifest", required=True)
    le.add_argument("--root", required=True)
    le.add_argument("--calibration", required=True)
    le.add_argument("--output", required=True)
    le.add_argument("--class-one", required=True, choices=["fake", "real"])
    le.add_argument("--device", default="cpu")
    le.add_argument("--batch-size", type=int, default=32)
    le.add_argument("--workers", type=int, default=0)
    ca = commands.add_parser(
        "calibrate-legacy",
        help="Predict source validation to freeze checkpoint-specific thresholds",
    )
    ca.add_argument("--checkpoint", required=True)
    ca.add_argument("--manifest", required=True)
    ca.add_argument("--root", required=True)
    ca.add_argument("--output", required=True)
    ca.add_argument("--class-one", required=True, choices=["fake", "real"])
    ca.add_argument("--device", default="cpu")
    ca.add_argument("--batch-size", type=int, default=32)
    ca.add_argument("--workers", type=int, default=0)
    ca.add_argument("--threshold-policy", choices=["balanced_accuracy", "youden"], default="balanced_accuracy")
    co = commands.add_parser("compare")
    co.add_argument("--reference", required=True)
    co.add_argument("--other", required=True)
    co.add_argument("--reference-calibration", required=True)
    co.add_argument("--other-calibration", required=True)
    co.add_argument("--output", required=True)
    co.add_argument("--draws", type=int, default=1000)
    co.add_argument("--seed", type=int, default=42)
    sh = commands.add_parser("shared-failures")
    sh.add_argument("--predictions", nargs="+", required=True)
    sh.add_argument("--calibrations", nargs="+", required=True)
    sh.add_argument("--output", required=True)
    iv = commands.add_parser("inventory")
    iv.add_argument("--models-root", required=True)
    iv.add_argument("--output", required=True)
    from src.robustness.commands import register, run as extra_command
    from src.experimental.orchestration import register as register_experimental

    register(commands)
    register_experimental(commands)
    args = p.parse_args(argv)
    from src.robustness.provenance import write_json

    if args.command == "experimental":
        from src.experimental.orchestration import run as run_experimental

        result = run_experimental(args)
    elif args.command in {
        "import-predictions",
        "calibrate-export",
        "expand-plan",
        "profile",
        "export-xai",
        "report-generators",
        "aggregate-seeds",
    }:
        result = extra_command(args)
    elif args.command == "convert-manifest":
        from src.robustness.manifests import convert_manifest

        result = convert_manifest(
            args.source,
            args.output,
            dataset=args.dataset,
            split=args.split,
            label_column=args.label_column,
            convention=args.convention,
            group_column=args.group_column,
        )
    elif args.command == "audit-images":
        from src.robustness.manifests import audit_images

        result = audit_images(
            args.manifest, args.root, output=args.output, hash_images=args.hash_images
        )
        print(json.dumps(result, indent=2))
        return 0 if result["complete"] else 1
    elif args.command in {"plan", "train"}:
        from src.robustness.experiments import read_config, plan

        config = read_config(args.config)
        if args.command == "train" and args.execute:
            from src.robustness.engine import train

            result = train(config, device=args.device, resume=args.resume)
        else:
            result = plan(config)
            if args.command == "plan" and args.output:
                write_json(args.output, result)
    elif args.command == "evaluate-suite":
        from src.robustness.suite import run_suite_config

        result = run_suite_config(args.config, device=args.device, workers=args.workers,
                                  output=args.output, execute=args.execute)
    elif args.command == "calibrate-run":
        from src.robustness.engine import load_pilot
        from src.robustness.inference import predict, calibrate
        from src.robustness.manifests import load_manifest
        from src.robustness.artifacts import save_predictions
        from src.robustness.provenance import digest_file
        from src.robustness.suite import pilot_contract

        frame, record = load_manifest(args.manifest)
        if record["split"] != "val":
            raise ValueError("Calibration accepts source validation only")
        output = Path(args.output)
        if output.exists():
            raise FileExistsError("Use a fresh calibration directory")
        model, run_record = load_pilot(args.run, args.device)
        contract = pilot_contract(args.run, run_record)
        predictions = predict(model, frame, args.root, image_size=contract["image_size"],
                              device=args.device, batch_size=args.batch_size, workers=args.workers)
        output.mkdir(parents=True)
        checksum = digest_file(Path(args.run) / "best.pt")
        save_predictions(output / "validation_predictions.csv", predictions,
                         manifest_record=record, model_sha256=checksum,
                         metadata={"input_contract": contract})
        result = calibrate(predictions, manifest_record=record, output=output / "calibration.json",
                           model_sha256=checksum, policy=args.threshold_policy, input_contract=contract)
    elif args.command == "evaluate":
        from src.robustness.engine import load_pilot
        from src.robustness.identity import evaluation_identity
        from src.robustness.inference import evaluate
        from src.robustness.suite import pilot_contract

        model, record = load_pilot(args.run, args.device)
        result = evaluate(
            model,
            args.manifest,
            args.root,
            args.output,
            checkpoint_path=Path(args.run) / "best.pt",
            calibration_path=Path(args.run) / "calibration.json",
            image_size=record["config"]["training"]["image_size"],
            device=args.device,
            batch_size=args.batch_size,
            workers=args.workers,
            research_run=evaluation_identity(record),
            input_contract=pilot_contract(args.run, record),
        )
    elif args.command in {"evaluate-legacy", "calibrate-legacy"}:
        import torch
        from src.pipelines.checkpoints import (
            run_from_checkpoint,
            config_from_run,
            load_model_from_run,
        )
        from src.robustness.inference import evaluate, predict, calibrate
        from src.robustness.manifests import load_manifest
        from src.robustness.provenance import digest_file
        from src.robustness.suite import legacy_contract, legacy_identity

        run = run_from_checkpoint(args.checkpoint)
        config = config_from_run(run)
        contract = legacy_contract(run, config)
        contract["checkpoint_class1"] = args.class_one
        if args.command == "calibrate-legacy":
            frame, record = load_manifest(args.manifest)
            if record["split"] != "val":
                raise ValueError("Calibration accepts source validation only")
            out = Path(args.output)
            if out.exists():
                raise FileExistsError("Use a new calibration directory")
            model = load_model_from_run(run, torch.device(args.device))
            predictions = predict(
                model,
                frame,
                args.root,
                image_size=config.image_size,
                mode=run.fourier_mode,
                in_channels=config.in_channels,
                positive_class=args.class_one,
                device=args.device,
                batch_size=args.batch_size,
                workers=args.workers,
            )
            out.mkdir(parents=True)
            from src.robustness.artifacts import save_predictions

            save_predictions(
                out / "validation_predictions.csv",
                predictions,
                manifest_record=record,
                model_sha256=digest_file(args.checkpoint),
                checkpoint_class1=args.class_one,
                metadata={"input_contract": contract},
            )
            result = calibrate(
                predictions,
                manifest_record=record,
                output=out / "calibration.json",
                model_sha256=digest_file(args.checkpoint),
                checkpoint_class1=args.class_one,
                policy=args.threshold_policy,
                input_contract=contract,
            )
        else:
            model = load_model_from_run(run, torch.device(args.device))
            result = evaluate(
                model,
                args.manifest,
                args.root,
                args.output,
                checkpoint_path=args.checkpoint,
                calibration_path=args.calibration,
                image_size=config.image_size,
                mode=run.fourier_mode,
                in_channels=config.in_channels,
                positive_class=args.class_one,
                device=args.device,
                batch_size=args.batch_size,
                workers=args.workers,
                input_contract=contract,
                research_run=legacy_identity(run, config),
            )
    elif args.command == "compare":
        from src.robustness.analysis import compare_files

        result = compare_files(
            args.reference,
            args.other,
            args.output,
            args.reference_calibration,
            args.other_calibration,
            draws=args.draws,
            seed=args.seed,
        )
    elif args.command == "shared-failures":
        from src.robustness.analysis import shared_failures

        result = shared_failures(args.predictions, args.calibrations, args.output)
    else:
        from src.robustness.analysis import inventory_runs

        result = inventory_runs(args.models_root, args.output)
    print(json.dumps(result, indent=2, ensure_ascii=False, allow_nan=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
