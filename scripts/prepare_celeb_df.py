"""Create a NEW certified Celeb-DF-v2 manifest: 0=real, 1=fake.

The official list has the opposite convention (1=real). Folder semantics and
labels are cross-checked before conversion. Full relative video paths provide
collision-safe identity. Original lists, crops and results are never overwritten.
"""

from __future__ import annotations
import argparse
import hashlib
from pathlib import Path, PurePosixPath
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import cv2
import numpy as np
import pandas as pd
from PIL import Image
from src.robustness.manifests import save_manifest
from src.robustness.provenance import (
    contained,
    digest,
    digest_file,
    relative_path,
    write_json,
)

FOLDERS = {"Celeb-real": 1, "YouTube-real": 1, "Celeb-synthesis": 0}


def parse_testing_list(list_path: Path, dataset_root: Path):
    entries = []
    seen = set()
    for line_number, text in enumerate(
        Path(list_path).read_text(encoding="utf-8-sig").splitlines(), 1
    ):
        if not text.strip():
            continue
        parts = text.strip().split(maxsplit=1)
        if len(parts) != 2 or parts[0] not in {"0", "1"}:
            raise ValueError(f"Malformed test list at line {line_number}")
        original_label = int(parts[0])
        rel = relative_path(parts[1])
        folder = PurePosixPath(rel).parts[0]
        if folder not in FOLDERS or FOLDERS[folder] != original_label:
            raise ValueError(f"Label/folder mismatch at line {line_number}: {rel}")
        if rel in seen:
            raise ValueError(f"Duplicate video: {rel}")
        seen.add(rel)
        entries.append((1 - original_label, contained(dataset_root, rel), rel))
    if not entries:
        raise ValueError("Empty official test list")
    return entries


def crop_face(image, cascade, target_size: int, policy: str):
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    faces = cascade.detectMultiScale(
        gray, scaleFactor=1.1, minNeighbors=5, minSize=(64, 64)
    )
    h, w = image.shape[:2]
    if len(faces):
        x, y, fw, fh = max(faces, key=lambda f: int(f[2]) * int(f[3]))
        cx, cy = x + fw / 2, y + fh / 2
        side = min(max(fw, fh) * 1.4, min(h, w))
        x0 = int(max(0, min(w - side, cx - side / 2)))
        y0 = int(max(0, min(h - side, cy - side / 2)))
        image = image[y0 : y0 + int(side), x0 : x0 + int(side)]
        status = "detected"
    elif policy == "center":
        side = min(h, w)
        x0 = (w - side) // 2
        y0 = (h - side) // 2
        image = image[y0 : y0 + side, x0 : x0 + side]
        status = "explicit_center_fallback"
    else:
        raise ValueError(
            "No detected face; center fallback was not explicitly authorized"
        )
    out = Image.fromarray(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
    return out.resize((target_size, target_size), Image.Resampling.LANCZOS), status


def prepare(
    dataset_root,
    output_dir,
    manifest_out,
    *,
    frames_per_video=15,
    target_size=512,
    no_face="error",
    limit_videos=None,
):
    dataset_root, output_dir, manifest_out = map(
        Path, (dataset_root, output_dir, manifest_out)
    )
    if frames_per_video < 1 or target_size < 16 or no_face not in {"error", "center"}:
        raise ValueError("Invalid extraction settings")
    if (
        output_dir.exists()
        or manifest_out.exists()
        or Path(str(manifest_out) + ".json").exists()
    ):
        raise FileExistsError(
            "Use fresh versioned crop and manifest locations; no cache reuse"
        )
    test_list = dataset_root / "List_of_testing_videos.txt"
    entries = parse_testing_list(test_list, dataset_root)
    expected = len(entries)
    if limit_videos is not None:
        if limit_videos < 1:
            raise ValueError("limit-videos must be positive")
        entries = entries[:limit_videos]
    output_dir.mkdir(parents=True)
    cascade_path = Path(cv2.data.haarcascades) / "haarcascade_frontalface_default.xml"
    detector = cv2.CascadeClassifier(str(cascade_path))
    if detector.empty():
        raise RuntimeError("Face detector could not be loaded")
    records, failures, coverage = [], [], []
    for label, path, rel in entries:
        cap = cv2.VideoCapture(str(path))
        try:
            if not cap.isOpened():
                raise IOError("Could not open video")
            count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
            if count <= 0:
                raise ValueError("No readable frame count")
            indices = np.linspace(
                0, count - 1, min(count, frames_per_video), dtype=int
            ).tolist()
            video_id = "celeb-df-v2:" + rel
            for index in indices:
                cap.set(cv2.CAP_PROP_POS_FRAMES, index)
                ok, image = cap.read()
                if not ok or image is None:
                    raise IOError(f"Unreadable frame {index}")
                crop, status = crop_face(image, detector, target_size, no_face)
                name = str(
                    PurePosixPath(rel).with_suffix("") / f"frame_{index:07d}.png"
                )
                target = contained(output_dir, name)
                target.parent.mkdir(parents=True, exist_ok=True)
                crop.save(target, format="PNG")
                records.append(
                    {
                        "img_name": name,
                        "label": label,
                        "sample_id": digest(["celeb-df-v2", rel, index]),
                        "group_id": video_id,
                        "dataset": "celeb-df-v2",
                        "split": "test",
                        "video_id": video_id,
                        "frame_idx": index,
                        "source_video": rel,
                        "source_label": 1 - label,
                        "face_status": status,
                        "sha256": digest_file(target),
                    }
                )
            coverage.append(
                {"video_id": video_id, "frames": len(indices), "status": "complete"}
            )
        except Exception as error:
            failures.append(
                {"source_video": rel, "error": f"{type(error).__name__}: {error}"}
            )
        finally:
            cap.release()
    audit = {
        "expected_official_videos": expected,
        "selected_videos": len(entries),
        "completed_videos": len(coverage),
        "failures": failures,
        "video_coverage": coverage,
        "complete": not failures,
        "subset": len(entries) != expected,
        "test_list_sha256": digest_file(test_list),
    }
    write_json(output_dir / "extraction_audit.json", audit)
    if failures:
        raise RuntimeError(
            f"{len(failures)} videos failed. No certified evaluation manifest was published; see extraction_audit.json"
        )
    frame = pd.DataFrame(records)
    if len(entries) != expected:
        frame["split"] = "smoke_test"
    return save_manifest(
        frame,
        manifest_out,
        {
            "source_label_convention": "real-is-1",
            "label_conversion": "canonical_label = 1 - official_label",
            "test_list_sha256": digest_file(test_list),
            "group_unit": "video (cross-video identity dependence unresolved)",
            "coverage": audit,
            "preprocessing": {
                "face_detector": "OpenCV Haar frontalface_default",
                "detector_sha256": digest_file(cascade_path),
                "context_padding": 0.2,
                "no_face": no_face,
                "target_size": target_size,
                "crop_format": "PNG lossless",
                "frames_per_video": frames_per_video,
            },
            "note": "This certified manifest replaces neither legacy CSVs nor legacy metrics. PNG changes preprocessing relative to old JPEG crops.",
        },
    )


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--dataset-root", type=Path, required=True)
    p.add_argument("--output-dir", type=Path, required=True)
    p.add_argument("--manifest-out", type=Path, required=True)
    p.add_argument("--frames-per-video", type=int, default=15)
    p.add_argument("--target-size", type=int, default=512)
    p.add_argument("--no-face", choices=["error", "center"], default="error")
    p.add_argument("--limit-videos", type=int)
    args = p.parse_args()
    result = prepare(
        args.dataset_root,
        args.output_dir,
        args.manifest_out,
        frames_per_video=args.frames_per_video,
        target_size=args.target_size,
        no_face=args.no_face,
        limit_videos=args.limit_videos,
    )
    print(__import__("json").dumps(result, indent=2))


if __name__ == "__main__":
    main()
