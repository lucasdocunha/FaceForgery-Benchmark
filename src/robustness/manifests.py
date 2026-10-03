"""Explicit label conversion and stable sample/group identity across datasets.

Canonical label convention is 0=real, 1=fake. Never choose a label direction by AUC.
A manifest has a JSON sidecar at <manifest.csv>.json binding its exact bytes.
"""

from __future__ import annotations
import json
from pathlib import Path
import numpy as np
import pandas as pd
from .provenance import (
    SCHEMA,
    contained,
    digest,
    digest_file,
    relative_path,
    write_csv,
    write_json,
)

REQUIRED = {"sample_id", "img_name", "label", "group_id", "dataset", "split"}
STRING_COLUMNS = [
    "sample_id",
    "img_name",
    "group_id",
    "video_id",
    "dataset",
    "split",
    "generator",
    "source_domain",
    "source_id",
    "sha256",
]


def binary(values) -> np.ndarray:
    raw = pd.to_numeric(pd.Series(values), errors="raise").to_numpy(dtype=float)
    if not np.isfinite(raw).all() or not np.isin(raw, [0.0, 1.0]).all():
        raise ValueError("Labels must be finite exact 0/1 values")
    return raw.astype(np.int64)


def labels(values, convention: str) -> np.ndarray:
    if convention not in {"fake-is-1", "real-is-1"}:
        raise ValueError(
            "Declare fake-is-1 or real-is-1; automatic inversion is forbidden"
        )
    y = binary(values)
    return y if convention == "fake-is-1" else 1 - y


def read_csv(path: str | Path) -> pd.DataFrame:
    df = pd.read_csv(
        path, dtype={k: "string" for k in STRING_COLUMNS}, keep_default_na=False
    )
    df.columns = df.columns.str.strip()
    return df


def validate(frame: pd.DataFrame, *, require_both: bool = True) -> pd.DataFrame:
    missing = REQUIRED - set(frame.columns)
    if missing:
        raise ValueError(f"Missing canonical manifest columns: {sorted(missing)}")
    frame = frame.copy().reset_index(drop=True)
    if frame.empty:
        raise ValueError("Empty manifest")
    for column in REQUIRED - {"label"}:
        if (
            frame[column].isna().any()
            or frame[column].astype(str).str.strip().eq("").any()
        ):
            raise ValueError(f"Missing {column}")
        frame[column] = frame[column].astype(str)
    frame["img_name"] = frame.img_name.map(relative_path)
    frame["label"] = binary(frame.label)
    if frame.sample_id.duplicated().any() or frame.img_name.duplicated().any():
        raise ValueError("Duplicate sample identity or image path")
    if frame.dataset.nunique() != 1 or frame.split.nunique() != 1:
        raise ValueError(
            "One dataset and split per manifest; combine through an explicit protocol"
        )
    if require_both and frame.label.nunique() != 2:
        raise ValueError("This operation requires both classes")
    if "video_id" in frame and frame.video_id.astype(str).str.strip().ne("").any():
        if frame.video_id.astype(str).str.strip().eq("").any():
            raise ValueError("Partial video identity coverage")
        if frame.groupby("video_id", sort=False).label.nunique().gt(1).any():
            raise ValueError("Conflicting labels within a video")
    return frame


def save_manifest(frame: pd.DataFrame, output: str | Path, metadata: dict) -> dict:
    output = Path(output)
    sidecar = Path(str(output) + ".json")
    if output.exists() or sidecar.exists():
        raise FileExistsError(f"Use a new versioned manifest path: {output}")
    frame = validate(frame, require_both=False)
    write_csv(output, frame)
    record = {
        **metadata,
        "schema": SCHEMA,
        "label_convention": "fake-is-1",
        "manifest_sha256": digest_file(output),
        "rows": len(frame),
        "dataset": frame.dataset.iloc[0],
        "split": frame.split.iloc[0],
        "class_counts": {str(k): int(v) for k, v in frame.label.value_counts().items()},
    }
    write_json(sidecar, record)
    return record


def load_manifest(path: str | Path, *, require_both: bool = True):
    path = Path(path)
    sidecar = Path(str(path) + ".json")
    if not sidecar.is_file():
        candidates = [
            path.parent.parent / "manifests" / path.name,
            path.parent / "manifests" / path.name,
            path.parent / (path.stem + ".csv"),
        ]
        for cand in candidates:
            cand_sidecar = Path(str(cand) + ".json")
            if cand.is_file() and cand_sidecar.is_file():
                path = cand
                sidecar = cand_sidecar
                break
        if not sidecar.is_file() and path.is_file():
            raw = read_csv(path)
            label_col = "target" if "target" in raw.columns else ("label" if "label" in raw.columns else None)
            if label_col:
                split_name = "train" if "train" in path.name else ("val" if "val" in path.name else "test")
                manifest_dir = path.parent.parent / "manifests"
                manifest_dir.mkdir(parents=True, exist_ok=True)
                converted_path = manifest_dir / path.name
                if not converted_path.exists() or not Path(str(converted_path) + ".json").exists():
                    if converted_path.exists():
                        converted_path.unlink()
                    convert_manifest(path, converted_path, dataset="mffi", split=split_name, label_column=label_col, convention="fake-is-1")
                path = converted_path
                sidecar = Path(str(path) + ".json")

    record = json.loads(Path(str(path) + ".json").read_text(encoding="utf-8"))
    if record.get("schema") != SCHEMA or record.get("label_convention") != "fake-is-1":
        raise ValueError(
            "Uncertified label convention; explicitly convert the original manifest"
        )
    if record.get("manifest_sha256") != digest_file(path):
        raise ValueError("Manifest bytes differ from their sidecar certificate")
    frame = validate(read_csv(path), require_both=require_both)
    if (
        len(frame) != record["rows"]
        or frame.dataset.iloc[0] != record["dataset"]
        or frame.split.iloc[0] != record["split"]
    ):
        raise ValueError("Manifest identity differs from certificate")
    return frame, record


def convert_manifest(
    source: str | Path,
    output: str | Path,
    *,
    dataset: str,
    split: str,
    label_column: str,
    convention: str,
    group_column: str | None = None,
) -> dict:
    if not dataset.strip() or not split.strip():
        raise ValueError("Dataset and split names are required")
    old = read_csv(source)
    if "img_name" not in old or label_column not in old:
        raise ValueError("Name img_name and the explicit label column are required")
    names = old.img_name.map(relative_path)
    result = pd.DataFrame(
        {"img_name": names, "label": labels(old[label_column], convention)}
    )
    result["dataset"], result["split"] = dataset, split
    # Do not include split: clean/degraded counterparts may share one logical identity.
    result["sample_id"] = [digest([dataset, name]) for name in names]
    if group_column:
        if (
            group_column not in old
            or old[group_column].isna().any()
            or old[group_column].astype(str).str.strip().eq("").any()
        ):
            raise ValueError("Explicit group column is absent or incomplete")
        result["group_id"] = [digest([dataset, str(g)]) for g in old[group_column]]
    else:
        result["group_id"] = result.sample_id
    for key in ("video_id", "generator", "source_domain", "source_id", "sha256"):
        if key in old:
            result[key] = old[key].astype(str)
    ordered = ["img_name", "label", "sample_id", "group_id", "dataset", "split"]
    ordered += [k for k in result if k not in ordered]
    return save_manifest(
        result[ordered],
        output,
        {
            "source_sha256": digest_file(source),
            "source_label_column": label_column,
            "source_label_convention": convention,
            "group_column": group_column,
            "group_unit": group_column
            or "image-only (identity/source dependence unresolved)",
            "warning": "Conversion certifies declared semantics, not visual ground truth or dataset independence.",
        },
    )


def assert_disjoint(train: pd.DataFrame, val: pd.DataFrame) -> dict:
    train, val = validate(train), validate(val)
    if set(train.split) != {"train"} or set(val.split) != {"val"}:
        raise ValueError(
            "Training requires split=train and selection requires source split=val"
        )
    checks = {}
    for key in ("sample_id", "group_id", "source_id", "sha256"):
        if key not in train or key not in val:
            checks[key] = "not available; not certified"
            continue
        a = set(train[key].dropna().astype(str)) - {"", "unknown"}
        b = set(val[key].dropna().astype(str)) - {"", "unknown"}
        overlap = a & b
        if overlap:
            raise ValueError(
                f"Train/validation overlap in {key}: {len(overlap)} values"
            )
        checks[key] = "no overlap among provided values"
    return checks


def audit_images(
    manifest: str | Path,
    root: str | Path,
    *,
    output: str | Path,
    hash_images: bool = False,
) -> dict:
    from PIL import Image

    frame, certificate = load_manifest(manifest, require_both=False)
    errors, hashes = [], []
    for row in frame.itertuples():
        try:
            path = contained(root, row.img_name)
            with Image.open(path) as image:
                image.verify()
            checksum = digest_file(path) if hash_images else None
            if (
                checksum
                and "sha256" in frame
                and str(row.sha256)
                and str(row.sha256) != checksum
            ):
                raise ValueError("Image hash differs from manifest")
            if checksum:
                hashes.append({"sample_id": row.sample_id, "sha256": checksum})
        except Exception as error:
            errors.append(
                {
                    "sample_id": row.sample_id,
                    "error": f"{type(error).__name__}: {error}",
                }
            )
    result = {
        "manifest_sha256": certificate["manifest_sha256"],
        "expected": len(frame),
        "valid": len(frame) - len(errors),
        "errors": errors,
        "image_hashes": hashes,
        "complete": not errors,
        "duplicate_content_groups": [],
    }
    if hashes:
        table = pd.DataFrame(hashes)
        result["duplicate_content_groups"] = [
            g.sample_id.tolist() for _, g in table.groupby("sha256") if len(g) > 1
        ]
    write_json(output, result)
    return result
