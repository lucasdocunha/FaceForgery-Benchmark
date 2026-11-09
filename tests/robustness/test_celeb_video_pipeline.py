"""Generated video fixtures only; no real faces or benchmark inference."""
from pathlib import Path

import cv2
import numpy as np
import pytest

from scripts.prepare_celeb_df import prepare
from src.robustness.manifests import load_manifest


def video_fixture(tmp_path):
    dataset = tmp_path / "videos"
    entries = []
    for label, folder, value in [(1, "Celeb-real", 30), (0, "Celeb-synthesis", 180)]:
        directory = dataset / folder
        directory.mkdir(parents=True)
        path = directory / "same_stem.avi"
        writer = cv2.VideoWriter(str(path), cv2.VideoWriter_fourcc(*"MJPG"), 5, (32, 32))
        if not writer.isOpened():
            pytest.fail("The CPU test environment cannot create its synthetic video fixture")
        for number in range(3):
            writer.write(np.full((32, 32, 3), value + number, dtype=np.uint8))
        writer.release()
        entries.append(f"{label} {folder}/same_stem.avi")
    (dataset / "List_of_testing_videos.txt").write_text("\n".join(entries)+"\n")
    return dataset


def test_celeb_video_extraction_is_collision_safe_and_explicit(tmp_path):
    dataset = video_fixture(tmp_path)
    output = tmp_path / "crops"
    manifest = tmp_path / "manifest.csv"
    report = prepare(dataset, output, manifest, frames_per_video=2, target_size=16, no_face="center")
    frame, _ = load_manifest(manifest)
    assert report["coverage"]["complete"]
    assert report["coverage"]["completed_videos"] == 2
    assert len(frame) == 4 and frame.video_id.nunique() == 2
    assert frame.img_name.nunique() == 4
    assert frame.groupby("video_id").label.first().sort_index().tolist() == [0, 1]
    assert frame.face_status.eq("explicit_center_fallback").all()
    assert frame.sha256.str.len().eq(64).all()


def test_celeb_detection_failure_never_publishes_complete_manifest(tmp_path):
    dataset = video_fixture(tmp_path)
    manifest = tmp_path / "manifest.csv"
    with pytest.raises(RuntimeError, match="No certified"):
        prepare(dataset, tmp_path / "crops", manifest, frames_per_video=1, target_size=16)
    assert not manifest.exists()
