import json
import random

import numpy as np
import pandas as pd
from PIL import Image
import pytest
import torch

from src.robustness.degraded_proxy import prepare
from src.robustness.manifests import load_manifest, save_manifest
from src.robustness.provenance import digest, digest_file
from src.robustness.statistics import align


def source(tmp_path, split="val"):
    root = tmp_path / "source"
    root.mkdir()
    rows = []
    for index in range(4):
        name = f"image-{index}.jpg"
        pixels = np.random.default_rng(index).integers(0, 256, (40, 50, 3), dtype=np.uint8)
        Image.fromarray(pixels).save(root / name)
        rows.append({"img_name": name, "sample_id": str(index), "group_id": str(index),
                     "label": index % 2, "dataset": "fixed-source", "split": split})
    manifest = tmp_path / "source.csv"
    save_manifest(pd.DataFrame(rows), manifest, {})
    return manifest, root


def test_order_independent_proxy_preserves_identity_and_rng(tmp_path):
    manifest, root = source(tmp_path)
    before = {path.name: digest_file(path) for path in root.iterdir()}
    python_state, torch_state = random.getstate(), torch.get_rng_state()
    first = prepare(manifest, root, tmp_path / "proxy-a", tmp_path / "clean-a.csv", tmp_path / "proxy-a.csv",
                    image_size=32)
    assert random.getstate() == python_state and torch.equal(torch.get_rng_state(), torch_state)
    original, _ = load_manifest(manifest)
    shuffled = tmp_path / "shuffled.csv"
    save_manifest(original.iloc[::-1], shuffled, {})
    prepare(shuffled, root, tmp_path / "proxy-b", tmp_path / "clean-b.csv", tmp_path / "proxy-b.csv", image_size=32)
    clean, _ = load_manifest(tmp_path / "clean-a.csv")
    degraded, certificate = load_manifest(tmp_path / "proxy-a.csv")
    assert set(clean.split) == set(degraded.split) == {"pilot"}
    for column in ("sample_id", "group_id", "label", "dataset", "img_name"):
        assert original[column].equals(clean[column]) and original[column].equals(degraded[column])
    align(clean.assign(p_fake=.5), degraded.assign(p_fake=.5))
    for name, checksum in before.items():
        assert digest_file(root / name) == checksum
        assert digest_file(tmp_path / "proxy-a" / name) == digest_file(tmp_path / "proxy-b" / name)
        with Image.open(tmp_path / "proxy-a" / name) as image:
            assert image.format == "PNG" and image.size == (32, 32)
    artifact = json.loads((tmp_path / "proxy-a/artifact.json").read_text())
    assert certificate["recipe_sha256"] == digest(artifact["recipe"]) == first["recipe_sha256"]
    assert certificate["image_inventory_sha256"] == digest(artifact["images"])
    assert len(artifact["images"]) == 4


def test_reject_training_and_overwriting_proxy(tmp_path):
    manifest, root = source(tmp_path, split="train")
    with pytest.raises(ValueError, match="validation or frozen test"):
        prepare(manifest, root, tmp_path / "proxy", tmp_path / "clean.csv", tmp_path / "proxy.csv")
    assert not (tmp_path / "proxy").exists()
    valid = tmp_path / "valid"
    valid.mkdir()
    manifest, root = source(valid)
    prepare(manifest, root, tmp_path / "proxy", tmp_path / "clean.csv", tmp_path / "proxy.csv", image_size=32)
    with pytest.raises(FileExistsError, match="fresh proxy"):
        prepare(manifest, root, tmp_path / "proxy", tmp_path / "clean.csv", tmp_path / "proxy.csv")
