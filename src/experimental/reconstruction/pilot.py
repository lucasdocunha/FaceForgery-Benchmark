"""Bounded source-only A-D campaign; target splits are never accepted or read."""

from __future__ import annotations

import argparse
import copy
import gc
import json
import math
from pathlib import Path

import numpy as np
import pandas as pd
import torch

from src.robustness.manifests import load_manifest
from src.robustness.provenance import digest_file, write_json
from src.robustness.statistics import choose_threshold, complementarity, grouped_auc_interval, summary

from .training import combine, describe_run, fit, normalize_config


def campaign_configs(args):
    train, certificate = load_manifest(args.train_manifest)
    if certificate["split"] != "train":
        raise ValueError("The reconstruction pilot requires a certified train population")
    real_count = int(train.label.eq(0).sum())
    ae_updates = math.ceil(math.ceil(real_count / args.batch_size) / args.accumulation)
    root = Path(args.output_dir)
    base = {"task": "ae", "name": "", "seed": args.seed, "output_dir": "",
            "data": {"train_manifest": str(args.train_manifest), "val_manifest": str(args.val_manifest),
                     "train_root": str(args.train_root), "val_root": str(args.val_root)},
            "model": {"ae": {"image_size": args.image_size, "width": args.width, "latent_dim": args.latent_dim},
                      "dropout": 0.0, "gradient": True, "width": args.width},
            "training": {"epochs": args.ae_epochs, "batch_size": args.batch_size, "workers": args.workers,
                         "grad_accum_steps": args.accumulation, "amp": True, "early_stop_patience": max(args.ae_epochs, args.detector_epochs) + 1,
                         "lr_ae": 0.001, "lr_backbone": 0.0001, "lr_head": 0.001, "plots": False,
                         "loss": {"lambda_ssim": 0.1, "kl_reduction": "mean_per_dim"}}}
    result = {}
    for kind in ("cae", "vae", "gated"):
        cfg = copy.deepcopy(base)
        cfg["model"]["ae"]["kind"] = kind
        cfg["training"]["loss"]["beta"] = {"kind": "linear", "maximum": 0.001 if kind == "vae" else 0.0,
                                                "warmup_steps": max(1, round(0.3 * ae_updates * args.ae_epochs))}
        result["ae-" + kind] = cfg
    if args.lpips_state:
        cfg = copy.deepcopy(result["ae-vae"])
        cfg["training"]["epochs"] = 1
        cfg["training"]["loss"].update(lambda_lpips=0.1, lpips_state_path=str(args.lpips_state),
                                          beta={"kind": "cyclic", "maximum": 0.001, "cycle_steps": max(2, ae_updates // 2)})
        result["ae-vae-lpips-cyclic-smoke"] = cfg
    for mode in ("x_only", "residual_only", "full"):
        cfg = copy.deepcopy(base)
        cfg["task"] = "residual"
        cfg["model"].update(ae_run=str(root / "ae-cae"), ae_mode="frozen", input_mode=mode,
                               backbone="resnet18", backbone_weights=str(args.resnet_weights))
        cfg["model"]["ae"]["kind"] = "cae"
        cfg["training"]["epochs"] = args.detector_epochs
        result["residual-" + mode] = cfg
    for mode in ("recon_finetune", "end_to_end"):
        cfg = copy.deepcopy(result["residual-full"])
        cfg["model"]["ae_mode"] = mode
        cfg["training"].update(epochs=1, lr_ae=0.00001)
        result["residual-" + mode + "-smoke"] = cfg
    for mode in ("frozen", "recon_finetune", "end_to_end"):
        cfg = copy.deepcopy(base)
        cfg["task"] = "latent"
        cfg["model"].update(ae_run=str(root / "ae-vae"), ae_mode=mode)
        cfg["model"]["ae"]["kind"] = "vae"
        cfg["training"].update(epochs=args.detector_epochs if mode == "frozen" else 1, lr_ae=0.00001)
        cfg["training"]["loss"]["beta"] = {"kind": "linear", "maximum": 0.001, "warmup_steps": ae_updates}
        result["latent" if mode == "frozen" else "latent-" + mode + "-smoke"] = cfg
    for name, cfg in result.items():
        cfg.update(name="min-development-" + name, output_dir=str(root / name))
        result[name] = normalize_config(cfg)
    return result


def run_campaign(args):
    configurations = campaign_configs(args)
    root = Path(args.output_dir)
    root.mkdir(parents=True, exist_ok=True)
    write_json(root / "campaign.json", {"purpose": "pilot/min-dataset/development", "target_access": "none",
               "configs": configurations, "comparison": "x_only/residual_only/full share init, seed, train population, backbone dimensions and optimizer budget"})
    for name, cfg in configurations.items():
        if args.phase == "ae" and cfg["task"] != "ae":
            continue
        if args.phase == "detectors" and cfg["task"] == "ae":
            continue
        print(json.dumps({"state": "starting", "run": name}), flush=True)
        result = fit(cfg, device=args.device, resume=args.resume and (Path(cfg["output_dir"]) / "last.pt").exists())
        print(json.dumps({"state": "complete", "run": name, "validation": result["validation"],
                          "runtime_seconds": result.get("runtime_seconds"), "peak_rss_kib": result.get("peak_rss_kib")}), flush=True)
        gc.collect()
        if str(args.device).startswith("cuda"):
            torch.cuda.empty_cache()
    if args.phase != "ae":
        verify_controlled_arms(root)
        combined = root / "spatial-latent-mean"
        if not combined.exists():
            combine(root / "residual-full", root / "latent", combined, device=args.device)
        else:
            describe_run(combined)


def verify_controlled_arms(root):
    root = Path(root)
    controls = {}
    for mode in ("x_only", "residual_only", "full"):
        run = root / ("residual-" + mode)
        record = describe_run(run)["run_record"]
        telemetry = json.loads((run / "telemetry.json").read_text())
        controls[mode] = {"initial_model_state_sha256": record["config"]["provenance"]["initial_model_state_sha256"],
                          "global_step": telemetry["global_step"], "epochs_completed": telemetry["epochs_completed"],
                          "seed": record["seed"]}
    if any(len({values[key] for values in controls.values()}) != 1 for key in next(iter(controls.values()))):
        raise ValueError("Residual input arms did not preserve initial model state, seed and update budget")
    write_json(root / "matched_controls.json", {"purpose": "pilot/min-dataset/development", "matched": True, "arms": controls})
    return controls


def summarize(root, *, reference_root=None, draws=200):
    """Validation-only aligned metrics, bootstrap uncertainty and HF complementarity."""
    root = Path(root)
    rows, populations, correlations = [], {}, []
    for run in sorted(root.iterdir()):
        if not run.is_dir() or not (run / "status.json").exists():
            continue
        describe_run(run)
        predictions = pd.read_csv(run / "validation_predictions.csv", keep_default_na=False)
        if set(predictions.split) != {"val"}:
            raise ValueError("Pilot summaries accept only source validation")
        metrics = json.loads((run / "validation_metrics.json").read_text())
        calibration = json.loads((run / "calibration.json").read_text())
        telemetry = json.loads((run / "telemetry.json").read_text())
        interval = grouped_auc_interval(predictions, draws=draws)
        rows.append({"run": run.name, "purpose": "pilot/min-dataset/development", "auc": metrics["auc"],
                     "auc_lower": interval["lower"], "auc_upper": interval["upper"], "eer": metrics["eer"],
                     "f1": metrics["f1"], "accuracy": metrics["accuracy"], "threshold": calibration["frame_threshold"],
                     "score_std": float(predictions.p_fake.std()), **telemetry})
        populations[run.name] = (predictions, calibration)
        if reference_root:
            for reference_path in sorted(Path(reference_root).glob("*/predictions_val.csv")):
                reference = pd.read_csv(reference_path, keep_default_na=False)
                # Historical verification used image basenames as IDs; resolve by
                # the declared filename, then bind to the canonical population.
                reference = reference.rename(columns={"sample_id": "img_name", "label": "reference_label", "p_fake": "reference_score"})
                columns = ["img_name", "reference_label", "reference_score"]
                joined = predictions.merge(reference[columns], on="img_name", how="outer", validate="one_to_one", indicator=True)
                if not joined["_merge"].eq("both").all() or not joined.label.eq(joined.reference_label).all():
                    raise ValueError("HF validation population or labels differ")
                original = predictions.set_index("sample_id").loc[joined.sample_id].reset_index()
                original["p_fake"] = joined.reference_score.to_numpy()
                threshold = choose_threshold(original.label, original.p_fake)
                counts, _ = complementarity(original, predictions, reference_threshold=threshold,
                                            other_threshold=calibration["frame_threshold"])
                merged = joined.reference_score.to_numpy()
                own = joined.p_fake.to_numpy()
                correlation = float(np.corrcoef(own, merged)[0, 1]) if own.std() and merged.std() else None
                baseline_auc = summary(joined.label, merged)["auc"]
                mean_auc = summary(joined.label, (own + merged) / 2)["auc"]
                correlations.append({"run": run.name, "reference": reference_path.parent.name,
                                     "reference_prediction_sha256": digest_file(reference_path), "pearson": correlation,
                                     "reference_auc": baseline_auc, "fixed_mean_auc": mean_auc,
                                     "fixed_mean_auc_delta": mean_auc - baseline_auc, **counts})
    if not rows:
        raise ValueError("No completed reconstruction runs to summarize")
    intervals = {}
    if {"residual-x_only", "residual-full"} <= set(populations):
        intervals["full_minus_x_only"] = grouped_auc_interval(populations["residual-x_only"][0], populations["residual-full"][0], draws=draws)
    report = {"purpose": "pilot/min-dataset/development", "split": "val", "target_access": "none",
              "bootstrap_draws": draws, "metrics": rows, "correlations": correlations, "paired_control_intervals": intervals,
              "limitations": ["One seed; 202 independent genuine training images in min dataset.",
                              "HF checkpoints saw min-train and used full validation containing min-val for selection.",
                              "Filename groups lack identity/video grouping; bootstrap cannot remove identity dependence.",
                              "Fixed-mean deltas are validation development evidence, without learned fusion or target evaluation."]}
    write_json(root / "pilot_summary.json", report)
    pd.DataFrame(rows).to_csv(root / "pilot_metrics.csv", index=False)
    pd.DataFrame(correlations).to_csv(root / "pilot_correlations.csv", index=False)
    return report


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--phase", choices=("ae", "detectors", "all", "report"), default="all")
    parser.add_argument("--train-manifest", type=Path)
    parser.add_argument("--val-manifest", type=Path)
    parser.add_argument("--train-root", type=Path)
    parser.add_argument("--val-root", type=Path)
    parser.add_argument("--output-dir", required=True, type=Path)
    parser.add_argument("--resnet-weights", type=Path)
    parser.add_argument("--lpips-state", type=Path)
    parser.add_argument("--reference-root", type=Path)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--image-size", type=int, default=128)
    parser.add_argument("--width", type=int, default=8)
    parser.add_argument("--latent-dim", type=int, default=32)
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--accumulation", type=int, default=1)
    parser.add_argument("--workers", type=int, default=0)
    parser.add_argument("--ae-epochs", type=int, default=3)
    parser.add_argument("--detector-epochs", type=int, default=2)
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--bootstrap-draws", type=int, default=200)
    parser.add_argument("--resume", action="store_true")
    args = parser.parse_args(argv)
    if args.phase != "report":
        if any(getattr(args, name) is None for name in ("train_manifest", "val_manifest", "train_root", "val_root", "resnet_weights")):
            parser.error("Training needs source manifests/roots and offline resnet weights")
        run_campaign(args)
    else:
        result = summarize(args.output_dir, reference_root=args.reference_root, draws=args.bootstrap_draws)
        print(json.dumps({"runs": len(result["metrics"]), "summary": str(args.output_dir / "pilot_summary.json")}), flush=True)


if __name__ == "__main__":
    main()
