import json

import numpy as np
import pandas as pd
from PIL import Image
import pytest

from src.experimental.sbi.data import SBIDataset
from src.robustness.heldout_sbi_proxy import prepare
from src.robustness.manifests import load_manifest, save_manifest
from src.robustness.provenance import digest_file, write_json


def fixture(tmp_path, split="val"):
    root = tmp_path / "source"
    root.mkdir()
    rows, records = [], []
    for index in range(3):
        name = f"{index}.png"
        Image.fromarray(np.random.default_rng(index).integers(0, 256, (40, 40, 3), dtype=np.uint8)).save(root / name)
        rows.append({"sample_id": str(index), "img_name": name, "label": int(index == 2),
                     "group_id": "source-" + str(index), "dataset": "source", "split": split})
        if index != 2:
            records.append({"sample_id": str(index), "img_name": name, "label": 0,
                            "image_sha256": digest_file(root / name), "status": "ok" if index == 0 else "no_face",
                            "landmarks": [[.2, .2], [.8, .2], [.8, .8], [.2, .8]]})
    manifest = tmp_path / "source.csv"
    certificate = save_manifest(pd.DataFrame(rows), manifest, {})
    landmarks = tmp_path / "landmarks.json"
    write_json(landmarks, {"schema": "faceforgery-landmarks-v1", "coordinates": "normalized_xy",
                          "source": "synthetic-test-only", "manifest_sha256": certificate["manifest_sha256"],
                          "records": records})
    return manifest, root, landmarks


def test_materializer_matches_recipe_and_preserves_heldout_provenance(tmp_path):
    manifest, root, landmarks = fixture(tmp_path)
    source, source_cert = load_manifest(manifest)
    result = prepare(manifest, root, landmarks, tmp_path / "proxy", tmp_path / "proxy.csv", image_size=32)
    rows, cert = load_manifest(tmp_path / "proxy.csv")
    assert result["n_source_real"] == 2 and result["n_accepted_sources"] == 1 and result["rows"] == 2
    assert set(rows.label) == {0, 1} and set(rows.group_id) == {"0"} and set(rows.split) == {"pilot"}
    assert set(rows.sample_id).isdisjoint(source.sample_id)
    assert cert["original_manifest"] == source_cert and cert["recipe"]["post_augment"] is True
    assert cert["failures"]["landmark"] == {"1": "no_face"}
    adapter = source[source.label.eq(0)].assign(split="train")
    dataset = SBIDataset(adapter, root, landmarks, manifest_sha256=source_cert["manifest_sha256"],
                         image_size=32, failure_policy="exclude", post_augment=True)
    from torchvision.transforms.functional import to_pil_image
    for label in (0, 1):
        expected = np.asarray(to_pil_image(dataset[label]["image"]))
        with Image.open(tmp_path / "proxy" / rows[rows.label.eq(label)].img_name.iloc[0]) as image:
            assert np.array_equal(expected, np.asarray(image))
    assert json.loads((tmp_path / "proxy/artifact.json").read_text())["state"] == "complete"
    assert (tmp_path / "proxy/qa.png").is_file()


def test_training_population_cannot_be_called_heldout(tmp_path):
    manifest, root, landmarks = fixture(tmp_path, split="train")
    with pytest.raises(ValueError, match="source validation"):
        prepare(manifest, root, landmarks, tmp_path / "proxy", tmp_path / "proxy.csv")
    assert not (tmp_path / "proxy").exists()
