"""Offline, genuine-only MediaPipe detection; no training-time downloads."""
from __future__ import annotations

import argparse
from collections import Counter
from importlib.metadata import version
from pathlib import Path
import time

import numpy as np

from src.robustness.manifests import load_manifest
from src.robustness.provenance import contained, digest_file, write_json


def precompute(manifest, root, asset, output):
    output = Path(output)
    if output.exists():
        raise FileExistsError("Use a new landmark cache path")
    if not Path(asset).is_file():
        raise FileNotFoundError("Stage the MediaPipe face_landmarker.task asset explicitly")
    try:
        import mediapipe as mp
        from mediapipe.tasks.python import BaseOptions
        from mediapipe.tasks.python.vision import FaceLandmarker, FaceLandmarkerOptions, RunningMode
    except ImportError as exc:
        raise ImportError("Landmark preprocessing requires requirements-experimental-landmarks.txt") from exc
    frame, certificate = load_manifest(manifest, require_both=False)
    if certificate["split"] not in {"train", "val"}:
        raise ValueError("SBI landmark preparation only accepts source train/val")
    frame = frame[frame.label.eq(0)]
    if frame.empty:
        raise ValueError("No genuine faces to prepare")
    options = FaceLandmarkerOptions(
        base_options=BaseOptions(model_asset_path=str(asset), delegate=BaseOptions.Delegate.CPU),
        running_mode=RunningMode.IMAGE, num_faces=2,
        min_face_detection_confidence=.5, min_face_presence_confidence=.5)
    records, started = [], time.perf_counter()
    with FaceLandmarker.create_from_options(options) as detector:
        for row in frame.itertuples(index=False):
            path = contained(root, row.img_name)
            record = {"sample_id": row.sample_id, "img_name": row.img_name,
                      "image_sha256": digest_file(path), "label": 0}
            try:
                image = mp.Image.create_from_file(str(path))
                faces = detector.detect(image).face_landmarks
                record.update(width=image.width, height=image.height, faces=len(faces))
                if len(faces) != 1:
                    record["status"] = "no_face" if not faces else "multiple_faces"
                else:
                    points = np.asarray([[p.x, p.y, p.z] for p in faces[0]])
                    valid = (np.isfinite(points).all() and (points[:, :2] >= -.2).all()
                             and (points[:, :2] <= 1.2).all())
                    record.update(status="ok" if valid else "invalid_geometry", landmarks=points.tolist())
            except Exception as exc:
                record.update(status="error", error=f"{type(exc).__name__}: {exc}")
            records.append(record)
    result = {"schema": "faceforgery-landmarks-v1", "source": "mediapipe_mesh",
              "coordinates": "normalized_xy", "manifest_sha256": certificate["manifest_sha256"],
              "asset_sha256": digest_file(asset), "package": version("mediapipe"),
              "status_counts": dict(Counter(r["status"] for r in records)),
              "seconds": time.perf_counter() - started, "records": records}
    write_json(output, result)
    return {k: v for k, v in result.items() if k != "records"}


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    for key in ("manifest", "root", "asset", "output"):
        parser.add_argument("--" + key, required=True)
    args = parser.parse_args()
    import json
    print(json.dumps(precompute(**vars(args)), indent=2))
