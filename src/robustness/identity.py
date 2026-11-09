"""Distinguish a seed realization from its controlled experimental condition."""

from __future__ import annotations
import copy
from .provenance import digest


def evaluation_identity(record: dict) -> dict:
    config = record["config"]
    training = copy.deepcopy(config["training"])
    seed = training.pop("seed")
    teacher = training.pop("teacher_run", None)
    condition = {
        "model": config["model"],
        "training_without_seed": training,
        "teacher_present": bool(teacher),
        "train_manifest_sha256": record["train_manifest_sha256"],
        "val_manifest_sha256": record["val_manifest_sha256"],
        "preprocessing": record["preprocessing"],
        "augmentation_definition": record["augmentation_definition"],
        "source_code_sha256": record["software"]["code_sha256"],
        "packages": record["software"]["packages"],
    }
    return {
        "name": config["name"],
        "seed": seed,
        "run_id": record["run_id"],
        "condition": condition,
        "condition_sha256": digest(condition),
        "teacher_checkpoint_sha256": record.get("teacher_checkpoint_sha256"),
        "note": "Teacher bytes may differ across seeds; teacher selection policy still needs protocol review.",
    }
