"""Prepare all protocols from existing prediction files; never load model weights.

The legacy CSV IDs are row positions, not filenames. Using them requires explicit
acknowledgement and byte-pinned original manifests. The execution stage verifies
every selected image's fresh probability against its frozen exported prediction.
"""
from __future__ import annotations
import json
import os
import tempfile
from pathlib import Path
import numpy as np
import pandas as pd
import yaml
from .contracts import (FAMILIES, SEEDS, PURE_FREQUENCY, HF_REPO, HF_REVISION,
    align_predictions, curate_validation, digest, freeze, manifest, matched_controls,
    metrics, raw_predictions, safe_name, sample64, select_frequency, sha256,
    shared_errors, strata, threshold)

METHOD_NAMES = ("gradcam", "integrated_gradients", "gradient_shap", "kernel_shap",
    "lime", "saliency", "input_x_gradient", "smoothgrad", "occlusion",
    "feature_ablation", "shapley_sampling", "attention_rollout")


def model_id(family, mode, regime, seed):
    return safe_name(f"{family}.{mode}.{regime}.seed_{seed}")


def relative_run(family, mode, regime, seed):
    return f"{safe_name(family)}/{safe_name(mode)}/{safe_name(regime)}/seed_{int(seed)}"


def load_settings(path):
    value = yaml.safe_load(Path(path).read_text())
    if not isinstance(value, dict) or value.get("version") != 1:
        raise ValueError("Expected version: 1 XAI configuration")
    required = {"version", "seed", "regime", "primary_seed", "families", "methods", "settings", "task2"}
    if set(value) - required or required - set(value):
        raise ValueError(f"Configuration fields must be exactly {sorted(required)}")
    if value["regime"] not in {"scratch", "finetune"}:
        raise ValueError("Invalid regime")
    if value["primary_seed"] not in SEEDS:
        raise ValueError("primary_seed must belong to the declared three-seed roster")
    families, methods = value["families"], value["methods"]
    if not families or len(set(families)) != len(families) or not set(families).issubset(FAMILIES):
        raise ValueError("Invalid or duplicate architecture roster")
    if "resnet" not in families:
        raise ValueError("Reference architecture resnet must be in the primary roster")
    if not methods or len(set(methods)) != len(methods) or not set(methods).issubset(METHOD_NAMES):
        raise ValueError("Invalid or duplicate methods")
    settings = value["settings"]
    if settings.get("baseline") not in {"encoded_zero", "blurred_rgb"}:
        raise ValueError("Explicit baseline must be encoded_zero or blurred_rgb")
    if int(settings.get("replicates", 1)) < 1 or int(settings.get("patch_size", 32)) < 1:
        raise ValueError("Invalid replication or feature-group settings")
    if not isinstance(value["task2"], dict) or not 0 <= float(value["task2"].get("auc_floor", .60)) <= 1:
        raise ValueError("Invalid Task 2 selection floor")
    return value


def _freeze_csv(path, frame):
    """Ancillary CSV export; the JSON plan is the authoritative cohort contract."""
    path = Path(path); path.parent.mkdir(parents=True, exist_ok=True)
    data = frame.to_csv(index=False, lineterminator="\n").encode()
    with tempfile.NamedTemporaryFile(dir=path.parent, delete=False) as f:
        tmp = Path(f.name); f.write(data)
    try:
        try:
            os.link(tmp, path)
        except FileExistsError:
            if path.read_bytes() != data:
                raise ValueError(f"Frozen CSV differs: {path}")
    finally:
        tmp.unlink(missing_ok=True)


def _validation_selection(root, settings, inputs):
    rows, missing = [], []
    for family in settings["families"]:
        for mode in ("none", *PURE_FREQUENCY):
            for seed in SEEDS:
                rel = relative_run(family, mode, settings["regime"], seed)
                path = root / rel / "results/metrics_val.csv"
                if not path.exists():
                    missing.append(str(path.relative_to(root))); continue
                frame = pd.read_csv(path)
                if len(frame) != 1 or "auc" not in frame:
                    raise ValueError(f"Expected one validation metric row: {path}")
                row = frame.iloc[0].to_dict()
                expected = {"model_family": family, "fourier_mode": mode,
                            "regime": settings["regime"], "seed": seed, "split": "val"}
                for key, value in expected.items():
                    if key in row and row[key] != value:
                        raise ValueError(f"Validation metadata/path mismatch in {path}: {key}")
                rows.append({**expected, "auc": float(row["auc"])})
                inputs[str(path.relative_to(root))] = sha256(path)
    if not rows:
        raise ValueError("Task 2 needs existing validation metrics for matched RGB/frequency runs")
    selection = select_frequency(pd.DataFrame(rows), float(settings["task2"].get("auc_floor", .60)))
    selection["missing_validation_files"] = missing
    return selection


def prepare(config_path, models_root, val_manifest, test_manifest, output,
            *, acknowledge_legacy_row_ids=False, curate_identical_validation=False,
            include_degraded=False):
    if not acknowledge_legacy_row_ids:
        raise ValueError("Legacy predictions require --acknowledge-legacy-row-ids and the exact original manifests")
    config = load_settings(config_path)
    root, output = Path(models_root).resolve(), Path(output)
    val, test = manifest(val_manifest), manifest(test_manifest)
    source_hashes = {}; selected = None
    if config["task2"].get("enabled", True):
        selected = _validation_selection(root, config, source_hashes)
    descriptors = {}
    for family in config["families"]:
        for seed in SEEDS:
            ident = model_id(family, "none", config["regime"], seed)
            descriptors[ident] = {"family": family, "mode": "none", "regime": config["regime"], "seed": seed}
    if selected:
        pair = selected["selected"]
        ident = model_id(pair["model_family"], pair["fourier_mode"], pair["regime"], config["primary_seed"])
        descriptors[ident] = {"family": pair["model_family"], "mode": pair["fourier_mode"],
                              "regime": pair["regime"], "seed": config["primary_seed"]}
    validation = {}
    for ident, run in descriptors.items():
        rel = relative_run(run["family"], run["mode"], run["regime"], run["seed"])
        run["relative_dir"] = rel
        cfg = root / rel / "results/run_config.json"
        if not cfg.exists():
            raise FileNotFoundError(f"The baseline loader requires original run_config.json: {cfg}")
        run["config_sha256"] = sha256(cfg)
        weights = root / rel / "weights/best.pth"
        run["checkpoint_sha256"] = sha256(weights) if weights.exists() else None
        path = root / rel / "results/predictions_val.csv"
        validation[ident] = raw_predictions(path)
        source_hashes[str(path.relative_to(root))] = sha256(path)
    cleaned, derived_val, validation_audit = curate_validation(validation, val,
        allow_identical_duplicates=curate_identical_validation)
    for ident, table in cleaned.items():
        descriptors[ident]["threshold"] = threshold(table.y_true, table.prob_pos)
    tables, numerical = {}, []
    for ident, run in descriptors.items():
        path = root / run["relative_dir"] / "results/predictions_test.csv"
        table = align_predictions(raw_predictions(path), test)
        table["y_pred"] = (table.prob_pos >= run["threshold"]).astype(int)
        tables[ident] = table
        source_hashes[str(path.relative_to(root))] = sha256(path)
        numerical.append({"model": ident, "split": "test", **metrics(table.y_true, table.prob_pos, run["threshold"])})
        vt = cleaned[ident]
        numerical.append({"model": ident, "split": "val_derived", **metrics(vt.y_true, vt.prob_pos, run["threshold"])})
    primary = [model_id(f, "none", config["regime"], config["primary_seed"]) for f in config["families"]]
    rgb_all_seeds = {k: v for k, v in tables.items() if descriptors[k]["mode"] == "none"}
    primary_tables = {k: tables[k] for k in primary}
    reference = model_id("resnet", "none", config["regime"], config["primary_seed"])
    common = sample64(primary_tables, reference, int(config["seed"]))
    hard = shared_errors(primary_tables)
    controls = matched_controls(primary_tables, hard, int(config["seed"]))
    sensitivity = shared_errors(rgb_all_seeds)
    task2_roster = []
    if selected:
        pair = selected["selected"]
        task2_roster = [model_id(pair["model_family"], mode, pair["regime"], config["primary_seed"])
                        for mode in ("none", pair["fourier_mode"])]
    frames = {"task1": common, "task2": common if task2_roster else common.iloc[:0],
              "task3": hard, "task3_controls": controls}
    task_rosters = {"task1": primary, "task2": task2_roster, "task3": primary, "task3_controls": primary}
    cohorts = {}
    indexed = {k: v.set_index("id") for k, v in tables.items()}
    for task, frame in frames.items():
        records = []
        for row in frame.to_dict("records"):
            ident = int(row["id"])
            record = {"id": ident, "img_name": row["img_name"], "y_true": int(row["y_true"])}
            if "reference_stratum" in row:
                record["reference_stratum"] = row["reference_stratum"]
            record["predictions"] = {model: {"prob_pos": float(indexed[model].loc[ident, "prob_pos"]),
                                                "y_pred": int(indexed[model].loc[ident, "y_pred"])}
                                       for model in task_rosters[task]}
            records.append(record)
        cohorts[task] = {"roster": task_rosters[task], "samples": records}
    summary = {"evidence": "reanalysis of exported predictions, not new image inference",
               "test_images": len(test), "derived_validation_images": len(derived_val),
               "primary_roster": primary, "all_seed_roster": sorted(rgb_all_seeds),
               "shared_error_count_primary": len(hard), "shared_error_count_all_seeds": len(sensitivity),
               "shared_error_rate_primary": len(hard) / len(test),
               "shared_error_rate_all_seeds": len(sensitivity) / len(test)}
    if include_degraded:
        degraded = {}
        for ident in rgb_all_seeds:
            run = descriptors[ident]
            path = root / run["relative_dir"] / "results/predictions_test_d.csv"
            table = align_predictions(raw_predictions(path), test)
            table["y_pred"] = (table.prob_pos >= run["threshold"]).astype(int)
            degraded[ident] = table
            source_hashes[str(path.relative_to(root))] = sha256(path)
            numerical.append({"model": ident, "split": "test_d", **metrics(table.y_true, table.prob_pos, run["threshold"])})
        summary["degraded_shared_error_count_primary"] = len(shared_errors({k: degraded[k] for k in primary}))
        summary["degraded_shared_error_count_all_seeds"] = len(shared_errors(degraded))
        summary["degraded_pairing_assumption"] = "test_d numeric IDs follow the original clean test manifest; no byte-level verification asserted"
    plan = {"schema_version": 1, "config": config, "reference_model": reference,
            "provenance": {"hf_repository": HF_REPO, "hf_revision": HF_REVISION,
                "release_pin_is_user_expected_not_verified_by_local_paths": True,
                "original_validation_manifest_sha256": sha256(val_manifest),
                "original_test_manifest_sha256": sha256(test_manifest),
                "prediction_id_semantics": "explicitly acknowledged original CSV row positions",
                "input_artifact_sha256": source_hashes},
            "validation_audit": validation_audit, "models": descriptors,
            "selection": selected, "cohorts": cohorts, "prediction_summary": summary,
            "execution_status": "not_executed"}
    plan["plan_id"] = digest(plan)
    freeze(output / "plan.json", plan)
    freeze(output / "prediction_summary.json", summary)
    _freeze_csv(output / "prediction_metrics.csv", pd.DataFrame(numerical))
    for task, frame in frames.items():
        _freeze_csv(output / task / "cohort.csv", frame)
    _freeze_csv(output / "task3/all_seed_intersection.csv", sensitivity)
    # Preserve each model's own stratum on the common cohort, not the reference's label.
    outcome_rows = []
    for ident in primary:
        current = indexed[ident].loc[common.id].reset_index()
        for row, group in zip(current.to_dict("records"), strata(current)):
            outcome_rows.append({"model": ident, "id": int(row["id"]), "stratum": str(group),
                                 "prob_pos": float(row["prob_pos"]), "y_pred": int(row["y_pred"])})
    _freeze_csv(output / "task1/model_outcomes.csv", pd.DataFrame(outcome_rows))
    return plan


def read_plan(path):
    value = json.loads(Path(path).read_text())
    identity = value.pop("plan_id")
    if digest(value) != identity or value.get("schema_version") != 1:
        raise ValueError("Plan hash/schema mismatch")
    value["plan_id"] = identity
    return value
