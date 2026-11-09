"""Bind prediction bytes to population, checkpoint, and score semantics.

Certificates are integrity/provenance records, not signatures or proof of label
truth. Importing an old positional export requires an explicit row-order
acknowledgement; model orientation is never selected by maximizing target AUC.
"""

from __future__ import annotations
import json
from pathlib import Path
import numpy as np
import pandas as pd
from .manifests import binary, labels, load_manifest, read_csv
from .provenance import SCHEMA, digest_file, write_csv, write_json
from .statistics import checked_predictions, probabilities


def save_predictions(
    path,
    frame,
    *,
    manifest_record,
    model_sha256,
    checkpoint_class1="fake",
    metadata=None,
):
    if checkpoint_class1 not in {"fake", "real"}:
        raise ValueError("Declare checkpoint class-1 meaning")
    frame = checked_predictions(frame)
    path = Path(path)
    write_csv(path, frame)
    record = {
        **(metadata or {}),
        "schema": SCHEMA,
        "label_convention": "fake-is-1",
        "score": "p_fake",
        "checkpoint_class1": checkpoint_class1,
        "model_sha256": model_sha256,
        "manifest_sha256": manifest_record["manifest_sha256"],
        "dataset": manifest_record["dataset"],
        "split": manifest_record["split"],
        "rows": len(frame),
        "predictions_sha256": digest_file(path),
    }
    write_json(str(path) + ".json", record)
    return record


def load_predictions(path, *, calibration=None):
    path = Path(path)
    record = json.loads(Path(str(path) + ".json").read_text(encoding="utf-8"))
    required = {
        "schema",
        "label_convention",
        "score",
        "checkpoint_class1",
        "model_sha256",
        "manifest_sha256",
        "dataset",
        "split",
        "rows",
        "predictions_sha256",
    }
    if (
        not required <= set(record)
        or record["schema"] != SCHEMA
        or record["label_convention"] != "fake-is-1"
        or record["score"] != "p_fake"
    ):
        raise ValueError("Uncertified prediction schema or semantics")
    if record["checkpoint_class1"] not in {"fake", "real"} or record[
        "predictions_sha256"
    ] != digest_file(path):
        raise ValueError(
            "Prediction bytes or class orientation differ from their certificate"
        )
    frame = checked_predictions(read_csv(path))
    if len(frame) != record["rows"]:
        raise ValueError("Prediction count differs from certificate")
    for key in ("dataset", "split"):
        if key in frame and set(frame[key].astype(str)) != {record[key]}:
            raise ValueError(f"Prediction {key} differs from certificate")
    if calibration is not None:
        if (
            calibration.get("schema") != SCHEMA
            or calibration.get("selection_split") != "val"
        ):
            raise ValueError("Calibration must be frozen on source validation")
        if (
            calibration.get("model_sha256") != record["model_sha256"]
            or calibration.get("checkpoint_class1", "fake")
            != record["checkpoint_class1"]
        ):
            raise ValueError(
                "Calibration belongs to a different checkpoint or score orientation"
            )
    return frame, record


def import_legacy(
    source,
    manifest,
    output,
    *,
    checkpoint,
    source_labels,
    checkpoint_class1,
    acknowledge_row_order=False,
):
    if not acknowledge_row_order:
        raise ValueError(
            "Legacy numeric IDs require explicit acknowledgement of ORIGINAL manifest row order"
        )
    output = Path(output)
    if output.exists() or Path(str(output) + ".json").exists():
        raise FileExistsError("Use a fresh imported prediction artifact")
    frame, certificate = load_manifest(manifest)
    old = pd.read_csv(source)
    if not {"id", "y_true", "prob_pos"} <= set(old):
        raise ValueError("Legacy export requires id, y_true, prob_pos")
    ids = pd.to_numeric(old.id, errors="raise").to_numpy(dtype=float)
    if (
        not np.isfinite(ids).all()
        or (ids != np.floor(ids)).any()
        or set(ids) != set(range(len(frame)))
        or len(ids) != len(frame)
    ):
        raise ValueError(
            "Missing, duplicate, or invalid legacy IDs; no implicit curation"
        )
    old = old.assign(id=ids.astype(np.int64)).set_index("id").loc[np.arange(len(frame))]
    y = labels(old.y_true.to_numpy(), source_labels)
    if not np.array_equal(y, frame.label.to_numpy()):
        raise ValueError("Declared source labels disagree with canonical manifest")
    p = probabilities(old.prob_pos)
    if checkpoint_class1 not in {"fake", "real"}:
        raise ValueError("Explicit checkpoint class-1 is required")
    out = frame.copy()
    out["p_fake"] = p if checkpoint_class1 == "fake" else 1 - p
    return save_predictions(
        output,
        out,
        manifest_record=certificate,
        model_sha256=digest_file(checkpoint),
        checkpoint_class1=checkpoint_class1,
        metadata={
            "origin": "existing prediction export; NOT fresh image inference",
            "source_export_sha256": digest_file(source),
            "source_label_convention": source_labels,
            "identity_assumption": "caller acknowledged original manifest row order; not image-verified",
        },
    )
