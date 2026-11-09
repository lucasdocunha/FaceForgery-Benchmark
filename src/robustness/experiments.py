"""Frozen pilot plans: training and source validation only, no test-driven search."""

from __future__ import annotations
import json
from pathlib import Path
import yaml
from .manifests import load_manifest, assert_disjoint
from .provenance import SCHEMA, digest, source_identity, write_json

TOP = {"name", "purpose", "model", "data", "training", "output_root"}
MODEL = {"kind", "backbone", "pretrained", "width", "branch_dropout"}
DATA = {"train_manifest", "val_manifest", "train_root", "val_root"}
TRAIN = {
    "seed",
    "epochs",
    "batch_size",
    "workers",
    "image_size",
    "lr",
    "weight_decay",
    "patience",
    "recipe",
    "paired_training",
    "consistency_weight",
    "aux_weight",
    "teacher_run",
    "distillation_weight",
    "temperature",
}
DEFAULTS = {
    "seed": 42,
    "epochs": 10,
    "batch_size": 32,
    "workers": 0,
    "image_size": 224,
    "lr": 0.0001,
    "weight_decay": 0.0001,
    "patience": 4,
    "recipe": "basic_v1",
    "paired_training": False,
    "consistency_weight": 0.0,
    "aux_weight": 0.1,
    "teacher_run": None,
    "distillation_weight": 0.0,
    "temperature": 2.0,
}
KINDS = {"rgb", "spectral", "early_concat", "late_mean", "adaptive", "spatial_control"}


def read_config(path):
    raw = yaml.safe_load(Path(path).read_text(encoding="utf-8"))
    if (
        not isinstance(raw, dict)
        or set(raw) - TOP
        or not {"name", "model", "data"} <= set(raw)
    ):
        raise ValueError("Invalid/unknown research config fields")
    for key, allowed in [("model", MODEL), ("data", DATA), ("training", TRAIN)]:
        if not isinstance(raw.get(key, {}), dict) or set(raw.get(key, {})) - allowed:
            raise ValueError(f"Invalid/unknown {key} fields")
    cfg = {
        "name": raw["name"],
        "purpose": raw.get("purpose", "pilot"),
        "model": {"width": 32, "branch_dropout": 0.2, **raw["model"]},
        "data": raw["data"],
        "training": {**DEFAULTS, **raw.get("training", {})},
        "output_root": raw.get("output_root", "outputs/research"),
    }
    t, m = cfg["training"], cfg["model"]
    if (
        not isinstance(m["width"], int)
        or isinstance(m["width"], bool)
        or m["width"] < 4
        or m["width"] % 4
    ):
        raise ValueError("Auxiliary width must be a positive multiple of four")
    if not 0 <= float(m["branch_dropout"]) < 1:
        raise ValueError("branch_dropout must be in [0,1)")
    if (
        not {"kind", "backbone", "pretrained"} <= set(m)
        or m["kind"] not in KINDS
        or set(cfg["data"]) != DATA
    ):
        raise ValueError("Declare all model and source train/val inputs explicitly")
    if not isinstance(m["pretrained"], bool) or not isinstance(
        t["paired_training"], bool
    ):
        raise ValueError("Use actual YAML booleans")
    for key in ["seed", "epochs", "batch_size", "workers", "image_size", "patience"]:
        if not isinstance(t[key], int) or isinstance(t[key], bool):
            raise ValueError(f"{key} must be integer")
    if (
        t["seed"] < 0
        or min(t["epochs"], t["batch_size"], t["patience"]) < 1
        or t["workers"] < 0
        or t["image_size"] < 16
    ):
        raise ValueError("Invalid budgets")
    if t["recipe"] not in {"none", "basic_v1", "robust_v1"} or cfg["purpose"] not in {
        "pilot",
        "confirmatory",
        "synthetic_smoke",
    }:
        raise ValueError("Unknown recipe/purpose")
    import math

    for key in [
        "lr",
        "weight_decay",
        "consistency_weight",
        "aux_weight",
        "distillation_weight",
        "temperature",
    ]:
        if not math.isfinite(float(t[key])) or float(t[key]) < 0:
            raise ValueError(f"Invalid {key}")
    if t["lr"] <= 0 or t["temperature"] <= 0:
        raise ValueError("lr and temperature must be positive")
    if t["consistency_weight"] and not t["paired_training"]:
        raise ValueError("Consistency requires matched paired training")
    if t["paired_training"] and t["recipe"] != "robust_v1":
        raise ValueError("Paired training requires the declared corruption recipe")
    if bool(t["teacher_run"]) != bool(t["distillation_weight"]):
        raise ValueError(
            "Teacher run and positive distillation weight must be supplied together"
        )
    if (
        cfg["purpose"] != "synthetic_smoke"
        and not m["pretrained"]
        and m["kind"] != "spectral"
    ):
        raise ValueError(
            "Production pilot uses pretrained RGB; scratch is limited to explicit synthetic smoke tests"
        )
    return cfg


def plan(config: dict):
    data = config["data"]
    train, tc = load_manifest(data["train_manifest"])
    val, vc = load_manifest(data["val_manifest"])
    checks = assert_disjoint(train, val)
    identity = {
        "schema": SCHEMA,
        "config": config,
        "train_manifest_sha256": tc["manifest_sha256"],
        "val_manifest_sha256": vc["manifest_sha256"],
        "software": source_identity(),
    }
    from .imaging import RECIPE, PREPROCESSING

    identity["preprocessing"] = PREPROCESSING
    identity["augmentation_definition"] = (
        RECIPE
        if config["training"]["recipe"] == "robust_v1"
        else {"name": config["training"]["recipe"]}
    )
    if config["training"]["teacher_run"]:
        from .provenance import digest_file

        teacher = Path(config["training"]["teacher_run"])
        identity["teacher_checkpoint_sha256"] = digest_file(teacher / "best.pt")
        identity["teacher_run_sha256"] = digest_file(teacher / "run.json")
    run_id = digest(identity)
    return {
        **identity,
        "run_id": run_id,
        "run_dir": str(Path(config["output_root"]) / run_id[:16]),
        "split_audit": checks,
        "train_images": len(train),
        "val_images": len(val),
        "state": "planned",
        "research_status": "unexecuted; this is not a performance result",
    }
