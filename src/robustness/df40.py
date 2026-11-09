"""Certify a reviewed local DF40 subset without claiming official full coverage.

The imported upstream script samples selected folders and stores absolute paths.
This adapter preserves the resulting row order and explicit labels; it never
infers class or source-domain labels from a directory name or a model score.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path, PurePosixPath

import pandas as pd
from PIL import Image

from .manifests import labels, read_csv, save_manifest
from .provenance import contained, digest, digest_file, relative_path


def prepare(
    source,
    images_root,
    output,
    *,
    path_prefix=None,
    group_column=None,
    source_domain_column=None,
    convention="fake-is-1",
    acknowledge_reviewed_subset=False,
):
    if not acknowledge_reviewed_subset:
        raise ValueError(
            "Review source labels and subset provenance before acknowledging this custom subset"
        )
    frame = read_csv(source)
    if not {"img_path", "target", "method"} <= set(frame):
        raise ValueError("Expected the upstream subset columns img_path, target, method")
    if frame.empty:
        raise ValueError("Empty local DF40 subset")
    for column in [group_column, source_domain_column]:
        if column and (
            column not in frame
            or frame[column].isna().any()
            or frame[column].astype(str).str.strip().eq("").any()
        ):
            raise ValueError(f"Missing or incomplete declared annotation: {column}")
    root = Path(images_root).resolve()
    prefix = PurePosixPath(str(path_prefix or images_root))
    converted = []
    checksums = []
    for value in frame.img_path.astype(str):
        path = PurePosixPath(value)
        if path.is_absolute():
            try:
                name = str(path.relative_to(prefix))
            except ValueError as error:
                raise ValueError(f"Absolute image path is outside --path-prefix: {value}") from error
        else:
            name = value
        name = relative_path(name)
        actual = contained(root, name)
        with Image.open(actual) as image:
            image.verify()
        converted.append(name)
        checksums.append(digest_file(actual))
    dataset = "df40-local-subset"
    out = pd.DataFrame(
        {
            "img_name": converted,
            "label": labels(frame.target, convention),
            "sample_id": [digest([dataset, name]) for name in converted],
            "dataset": dataset,
            "split": "test",
            "generator": frame.method.astype(str),
            "sha256": checksums,
        }
    )
    out["group_id"] = (
        [digest([dataset, str(value)]) for value in frame[group_column]]
        if group_column
        else out.sample_id
    )
    if source_domain_column:
        out["source_domain"] = frame[source_domain_column].astype(str)
    if "video_id" in frame:
        out["video_id"] = frame.video_id.astype(str)
    if "source_id" in frame:
        out["source_id"] = frame.source_id.astype(str)
    if "paradigm" in frame:
        out["paradigm"] = frame.paradigm.astype(str)
    columns = ["img_name", "label", "sample_id", "group_id", "dataset", "split"]
    out = out[columns + [column for column in out if column not in columns]]
    return save_manifest(
        out,
        output,
        {
            "source_sha256": digest_file(source),
            "source_label_convention": convention,
            "origin": "reviewed local subset export; not the complete official DF40 benchmark",
            "official_coverage": "not established",
            "row_order": "preserved from the supplied local subset manifest",
            "group_unit": group_column or "image-only; video/identity dependence unresolved",
            "source_domain_annotation": source_domain_column or "not supplied; generator-level source-matched AUC unavailable",
            "source_method_counts": {
                str(key): int(value) for key, value in frame.method.value_counts().items()
            },
            "review_acknowledged": True,
            "warning": "Acknowledgement and hashes do not prove source label truth, source independence, or official protocol comparability.",
        },
    )


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", required=True)
    parser.add_argument("--images-root", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--path-prefix")
    parser.add_argument("--group-column")
    parser.add_argument("--source-domain-column")
    parser.add_argument("--convention", required=True, choices=["fake-is-1", "real-is-1"])
    parser.add_argument("--acknowledge-reviewed-subset", action="store_true")
    args = parser.parse_args(argv)
    result = prepare(
        args.source,
        args.images_root,
        args.output,
        path_prefix=args.path_prefix,
        group_column=args.group_column,
        source_domain_column=args.source_domain_column,
        convention=args.convention,
        acknowledge_reviewed_subset=args.acknowledge_reviewed_subset,
    )
    print(json.dumps(result, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
