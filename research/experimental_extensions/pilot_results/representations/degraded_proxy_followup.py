"""Replay fixed source fits and score the predeclared degraded validation proxy.

Run with PYTHONPATH pointing to the integrated repository. This evidence driver
does not open test populations, choose proxy thresholds or search configurations.
"""

import argparse
import copy
import json
from pathlib import Path
import resource
import time

import numpy as np
import pandas as pd
import torch

from src.experimental.feature_training import (
    feature_cache_predictor, fit_metric_run, read_model_artifact,
    predict_metric_run, representation_key, seed_cpu,
)
from src.experimental.features import extract_features, open_cache
from src.experimental.graph_training import fit_graph_run, predict_graph_run
from src.robustness.artifacts import load_predictions, save_predictions
from src.robustness.manifests import load_manifest
from src.robustness.provenance import digest_file, source_identity, write_json
from src.robustness.statistics import align, grouped_auc_interval, summary


RECIPE_SHA256 = "9c99d6ba69c622db9a2dd842d819d038f49b12cd965bc607cdc44102448ec660"
METRICS = ("n", "n_real", "n_fake", "auc", "f1", "accuracy", "balanced_accuracy",
           "eer", "threshold", "tn", "fp", "fn", "tp")


def read_json(path):
    return json.loads(Path(path).read_text())


def precise_predictions(path):
    # First certify bytes and identities, then preserve exact threshold ties.
    frame, certificate = load_predictions(path)
    precise = pd.read_csv(path, usecols=["sample_id", "p_fake"],
                          dtype={"sample_id": "string"}, float_precision="round_trip")
    frame["p_fake"] = precise.set_index("sample_id").loc[frame.sample_id, "p_fake"].to_numpy()
    return frame, certificate


def provenance():
    return {"software": source_identity(), "driver_sha256": digest_file(__file__),
            "scope": "source-only development follow-up; not independent benchmark evidence",
            "seed_scope": "downstream fit and label selection, fixed HF-supervised DINO-SRM extractor"}


def replay(args):
    previous = read_json(args.previous / "pilot.json")
    if len(previous["runs"]) != 21 or previous["seed"] != 42:
        raise ValueError("This follow-up requires the original 21 seed-42 candidates")
    args.output.mkdir(parents=True, exist_ok=False)
    report = {**provenance(), "previous_pilot": str(args.previous),
              "previous_pilot_sha256": digest_file(args.previous / "pilot.json"),
              "seed": previous["seed"], "checkpoint_baseline": previous["checkpoint_baseline"],
              "config_changes": ["fresh run_dir", "disable diagnostic t-SNE only"], "runs": []}
    started = time.perf_counter()
    for old_row in previous["runs"]:
        old_run = Path(old_row["run_dir"])
        row = {key: old_row[key] for key in ("family", "kind", "label_fraction")}
        row.update(previous_status=old_row["status"], previous_run=str(old_run))
        if old_row["status"] != "complete":
            row.update(status="retained_previous_failure", previous_result=old_row)
            report["runs"].append(row)
            write_json(args.output / "replay.json", report)
            continue
        row["previous_artifact_sha256"] = digest_file(old_run / "artifact.json")
        try:
            read_model_artifact(old_run)
            raise AssertionError("Expected strict implementation rejection of the previous bundle")
        except ValueError as error:
            if "implementation changed" not in str(error):
                raise
            row["previous_reload"] = {"status": "rejected", "error": str(error)}
        config = copy.deepcopy(read_json(old_run / "artifact.json")["config"])
        if set(config["caches"]) != {"train", "val"} or config["seed"] != 42:
            raise ValueError("Unexpected source cache roles or seed")
        for split, cache_path in config["caches"].items():
            cache = open_cache(cache_path)
            if set(cache.frame.split) != {split} or len(cache.frame) != 1000:
                raise ValueError("Only the fixed 1,000-row source train/val caches are allowed")
        run = args.output / old_run.name
        config["run_dir"] = str(run)
        config["model"]["tsne"] = False
        row["run_dir"] = str(run)
        run_started = time.perf_counter()
        try:
            (fit_metric_run if row["family"] == "metric" else fit_graph_run)(config)
            old_prediction, _ = precise_predictions(old_run / "val_predictions.csv")
            new_prediction, _ = precise_predictions(run / "val_predictions.csv")
            a, b = align(old_prediction, new_prediction)
            difference = float(np.max(np.abs(a.p_fake.to_numpy() - b.p_fake.to_numpy())))
            threshold = read_json(old_run / "calibration.json")["frame_threshold"]
            fresh_threshold = read_json(run / "calibration.json")["frame_threshold"]
            np.testing.assert_allclose(a.p_fake, b.p_fake, rtol=0, atol=1e-12)
            if fresh_threshold != threshold:
                raise AssertionError("Replay changed the original frozen validation threshold")
            if digest_file(old_run / "label_mask.json") != digest_file(run / "label_mask.json"):
                raise AssertionError("Replay changed the original training label mask")
            row.update(status="complete", frozen_threshold=threshold,
                       max_absolute_clean_score_difference=difference,
                       threshold_unchanged=True, label_mask_unchanged=True,
                       artifact_sha256=digest_file(run / "artifact.json"),
                       implementation=read_model_artifact(run)["implementation"],
                       label_count=read_json(run / "label_mask.json")["selected_total"],
                       telemetry=read_json(run / "pipeline_telemetry.json"))
        except Exception as error:
            row.update(status="failed", error=f"{type(error).__name__}: {error}")
        row["wall_seconds"] = time.perf_counter() - run_started
        report["runs"].append(row)
        write_json(args.output / "replay.json", report)
        print(json.dumps({key: row[key] for key in ("family", "kind", "label_fraction", "status", "wall_seconds")}), flush=True)
    report.update(wall_seconds=time.perf_counter() - started,
                  peak_rss_kib=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss)
    write_json(args.output / "replay.json", report)
    return int(any(row["status"] != "complete" for row in report["runs"]))


def extract(args):
    if args.output.exists():
        raise FileExistsError("Use a fresh extraction record")
    record = {**provenance(), "status": "started", "manifest": str(args.manifest),
              "root": str(args.root), "source_index": str(args.cache_index)}
    write_json(args.output, record)
    started = time.perf_counter()
    try:
        index = read_json(args.cache_index)
        if set(index) != {"train", "val"}:
            raise ValueError("Exactly the original source train/val cache index is required")
        clean = open_cache(index["val"])
        frame, certificate = load_manifest(args.manifest)
        if certificate.get("recipe_sha256") != RECIPE_SHA256 or set(frame.split) != {"pilot"}:
            raise ValueError("Unexpected fixed degraded proxy recipe or split")
        if certificate["original_manifest"]["manifest_sha256"] != clean.metadata["manifest"]["manifest_sha256"]:
            raise ValueError("Proxy does not derive from the original source validation cache")
        checkpoint = Path(clean.metadata["checkpoint"]["path"])
        if digest_file(checkpoint) != clean.metadata["key"]["checkpoint_sha256"]:
            raise ValueError("Original verified extractor checkpoint changed")
        seed_cpu(42)
        torch.cuda.reset_peak_memory_stats()
        path = extract_features(checkpoint, args.manifest, args.root, args.cache_root,
                                device="cuda:0", batch_size=8, workers=0, use_amp=True)
        target = open_cache(path)
        if representation_key(clean.metadata) != representation_key(target.metadata):
            raise ValueError("Target extractor differs from the original verified extractor")
        record.update(status="complete", cache=str(path), cache_identity=target.identity,
                      cache_json_sha256=digest_file(path / "cache.json"),
                      clean_cache=index["val"], clean_cache_identity=clean.identity,
                      representation_key=representation_key(target.metadata),
                      certificate=certificate, source_commit=clean.metadata["key"]["commit"],
                      target_commit=target.metadata["key"]["commit"],
                      peak_vram_bytes=torch.cuda.max_memory_allocated())
    except Exception as error:
        record.update(status="failed", error=f"{type(error).__name__}: {error}")
        raise
    finally:
        record.update(wall_seconds=time.perf_counter() - started,
                      peak_rss_kib=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss)
        write_json(args.output, record)
        print(json.dumps(record), flush=True)
    return 0


def verify_replay(args):
    """Retain the strict audit and check numerical drift without another fit."""
    if args.output.exists():
        raise FileExistsError("Use a fresh verification record")
    report = read_json(args.replay)
    report.update(initial_replay=str(args.replay), initial_replay_sha256=digest_file(args.replay),
                  verification={**provenance(), "score_atol": 1e-6, "score_rtol": 0,
                                "policy": "float32 tolerance; exact thresholds, masks, decisions and selected epochs; AUC tolerance 1e-12"})
    seed_cpu(42)
    for row in report["runs"]:
        row["initial_audit"] = {"status": row["status"], "error": row.pop("error", None)}
        old_run, run = Path(row["previous_run"]), Path(row["run_dir"])
        old_artifact = read_json(old_run / "artifact.json")
        artifact = read_model_artifact(run)
        old_config, config = copy.deepcopy(old_artifact["config"]), copy.deepcopy(artifact["config"])
        for value in (old_config, config):
            value.pop("run_dir")
            value["model"]["tsne"] = False
        if old_config != config:
            raise AssertionError("A non-diagnostic fit setting changed")
        a, _ = precise_predictions(old_run / "val_predictions.csv")
        b, _ = precise_predictions(run / "val_predictions.csv")
        a, b = align(a, b)
        np.testing.assert_allclose(a.p_fake, b.p_fake, rtol=0, atol=1e-6)
        threshold = read_json(old_run / "calibration.json")["frame_threshold"]
        if threshold != read_json(run / "calibration.json")["frame_threshold"]:
            raise AssertionError("The original frozen validation threshold changed")
        np.testing.assert_array_equal(a.p_fake.to_numpy() >= threshold, b.p_fake.to_numpy() >= threshold)
        if digest_file(old_run / "label_mask.json") != digest_file(run / "label_mask.json"):
            raise AssertionError("Training label mask changed")
        old_metrics, metrics = summary(a.label, a.p_fake, threshold), summary(b.label, b.p_fake, threshold)
        original_metrics = read_json(old_run / "metrics.json")["frame"]
        for field in ("tn", "fp", "fn", "tp"):
            if not original_metrics[field] == old_metrics[field] == metrics[field]:
                raise AssertionError("Confusion counts differ from the original calibration metrics")
        if abs(old_metrics["auc"] - metrics["auc"]) > 1e-12:
            raise AssertionError("Clean AUC changed")
        old_telemetry = read_json(old_run / "pipeline_telemetry.json")
        telemetry = read_json(run / "pipeline_telemetry.json")
        if old_telemetry["best_epoch"] != telemetry["best_epoch"]:
            raise AssertionError("Selected training epoch changed")
        val = open_cache(config["caches"]["val"])
        reload_scores = (predict_metric_run if row["family"] == "metric" else predict_graph_run)(run, val.features)
        saved = b.set_index("sample_id").loc[val.frame.sample_id, "p_fake"].to_numpy()
        np.testing.assert_allclose(reload_scores, saved, rtol=0, atol=1e-6)
        state_difference = None
        if (old_run / "best.pt").exists():
            old_state = torch.load(old_run / "best.pt", map_location="cpu", weights_only=True)["state_dict"]
            state = torch.load(run / "best.pt", map_location="cpu", weights_only=True)["state_dict"]
            state_difference = max(float((old_state[key] - state[key]).abs().max()) for key in old_state)
        row.update(status="complete", frozen_threshold=threshold,
                   max_absolute_clean_score_difference=float(np.max(np.abs(a.p_fake.to_numpy() - b.p_fake.to_numpy()))),
                   max_absolute_reload_difference=float(np.max(np.abs(reload_scores - saved))),
                   max_absolute_selected_weight_difference=state_difference,
                   clean_auc_difference=metrics["auc"] - old_metrics["auc"],
                   threshold_unchanged=True, label_mask_unchanged=True,
                   decisions_unchanged=True, confusion_counts_unchanged=True, best_epoch_unchanged=True,
                   artifact_sha256=digest_file(run / "artifact.json"), implementation=artifact["implementation"],
                   label_count=read_json(run / "label_mask.json")["selected_total"], telemetry=telemetry)
    write_json(args.output, report)
    print(json.dumps({"verified": len(report["runs"]), "score_atol": 1e-6,
                      "max_absolute_clean_score_difference": max(row["max_absolute_clean_score_difference"] for row in report["runs"]),
                      "max_absolute_reload_difference": max(row["max_absolute_reload_difference"] for row in report["runs"])}), flush=True)
    return 0


def score(args):
    replay_record, extraction = read_json(args.replay), read_json(args.extraction)
    if extraction["status"] != "complete" or any(row["status"] != "complete" for row in replay_record["runs"]):
        raise ValueError("Extraction and every fixed replay must complete before scoring")
    args.output.mkdir(parents=True, exist_ok=False)
    seed_cpu(42)
    clean, target = open_cache(extraction["clean_cache"]), open_cache(extraction["cache"])
    report = {**provenance(), "replay": str(args.replay), "replay_sha256": digest_file(args.replay),
              "extraction": str(args.extraction), "extraction_sha256": digest_file(args.extraction),
              "recipe_sha256": RECIPE_SHA256, "seed": 42, "rows": len(target.frame),
              "threshold_policy": "original clean-validation Youden thresholds; never refit on the proxy",
              "limitations": ["HF extractor used full MFFI training labels and full validation for selection",
                              "same min-val images after one deterministic degradation draw",
                              "image-only groups; identity/video dependence unresolved",
                              "conditional intervals, not independent confirmation or across-seed uncertainty",
                              "one downstream seed; no upstream-seed independence claim"],
              "runs": [], "paired_comparisons": []}
    frames = {}
    schedule = [{"family": "checkpoint", "kind": "dino_srm_logits", "label_fraction": 1.0,
                 "frozen_threshold": replay_record["checkpoint_baseline"]["threshold"]}, *replay_record["runs"]]
    started = time.perf_counter()
    for row in schedule:
        key = (row["family"], row["kind"], row["label_fraction"])
        name = f"{key[0]}-{key[1]}-fraction{key[2]:g}"
        threshold = row["frozen_threshold"]
        if row["family"] == "checkpoint":
            a, b = clean.frame.copy(), target.frame.copy()
            a["p_fake"] = torch.from_numpy(np.array(clean.logits)).softmax(1)[:, 1].numpy()
            b["p_fake"] = torch.from_numpy(np.array(target.logits)).softmax(1)[:, 1].numpy()
            model_sha256 = target.metadata["key"]["checkpoint_sha256"]
        else:
            run = Path(row["run_dir"])
            a, _ = precise_predictions(run / "val_predictions.csv")
            callback = feature_cache_predictor(run, {"degraded": extraction["cache"]})["degraded"]
            unlabeled = target.frame.drop(columns="label")
            predicted = callback(unlabeled, extraction["root"])
            b = target.frame.copy()
            b["p_fake"] = predicted.set_index("sample_id").loc[b.sample_id, "p_fake"].to_numpy()
            model_sha256 = digest_file(run / "artifact.json")
        align(a, b)
        frames[key] = {"clean": a, "degraded": b}
        destination = args.output / f"{name}-degraded.csv"
        certificate = save_predictions(destination, b, manifest_record=target.metadata["manifest"],
                                       model_sha256=model_sha256,
                                       metadata={"frozen_threshold": threshold,
                                                 "source_calibration": "original clean-validation Youden"})
        result = {key: row[key] for key in ("family", "kind", "label_fraction")}
        result.update(frozen_threshold=threshold, label_count=row.get("label_count"),
                      predictions_sha256=certificate["predictions_sha256"], model_sha256=model_sha256)
        for split, prediction in frames[key].items():
            metrics = summary(prediction.label, prediction.p_fake, threshold)
            result[split] = {field: metrics[field] for field in METRICS}
        result["degraded_minus_clean_auc"] = result["degraded"]["auc"] - result["clean"]["auc"]
        report["runs"].append(result)
        write_json(args.output / "summary.json", report)
        print(json.dumps(result), flush=True)
    pairs = [(("metric", reference, 1.0), ("metric", "supcon", 1.0))
             for reference in ("linear", "mlp", "raw_centroid")]
    pairs += [(("graph", "mlp", fraction), ("graph", candidate, fraction))
              for fraction in (.05, .10) for candidate in ("gcn", "gat", "sage")]
    for reference, candidate in pairs:
        contrast = {"reference": list(reference), "candidate": list(candidate)}
        for split in ("clean", "degraded"):
            contrast[split] = grouped_auc_interval(frames[reference][split], frames[candidate][split],
                                                   draws=args.draws, seed=42)
        report["paired_comparisons"].append(contrast)
        write_json(args.output / "summary.json", report)
    report.update(wall_seconds=time.perf_counter() - started,
                  peak_rss_kib=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss)
    write_json(args.output / "summary.json", report)
    flat = [{**{key: row[key] for key in ("family", "kind", "label_fraction", "label_count", "frozen_threshold")},
             **{f"{split}_{field}": value for split in ("clean", "degraded") for field, value in row[split].items()},
             "degraded_minus_clean_auc": row["degraded_minus_clean_auc"]} for row in report["runs"]]
    pd.DataFrame(flat).to_csv(args.output / "summary.csv", index=False)
    return 0


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest="command", required=True)
    replay_parser = commands.add_parser("replay")
    replay_parser.add_argument("--previous", type=Path, required=True)
    replay_parser.add_argument("--output", type=Path, required=True)
    extract_parser = commands.add_parser("extract")
    for flag in ("cache-index", "manifest", "root", "cache-root", "output"):
        extract_parser.add_argument(f"--{flag}", type=Path, required=True)
    verify_parser = commands.add_parser("verify-replay")
    for flag in ("replay", "output"):
        verify_parser.add_argument(f"--{flag}", type=Path, required=True)
    score_parser = commands.add_parser("score")
    for flag in ("replay", "extraction", "output"):
        score_parser.add_argument(f"--{flag}", type=Path, required=True)
    score_parser.add_argument("--draws", type=int, default=1000)
    args = parser.parse_args()
    return {"replay": replay, "extract": extract, "verify-replay": verify_replay, "score": score}[args.command](args)


if __name__ == "__main__":
    raise SystemExit(main())
