from pathlib import Path

import numpy as np
import pandas as pd
import pytest
from PIL import Image

from src.robustness.df40 import prepare
from src.robustness.manifests import load_manifest


def fixture(tmp_path):
    root = tmp_path / "images"
    root.mkdir()
    for name in ["real.png", "fake.png"]:
        Image.fromarray(np.zeros((16, 16, 3), dtype=np.uint8)).save(root / name)
    source = tmp_path / "original.csv"
    pd.DataFrame(
        {
            "img_path": ["/old/location/real.png", "/old/location/fake.png"],
            "target": [0, 1],
            "method": ["real_source", "generator_a"],
            "source_domain": ["source_a", "source_a"],
            "identity": ["real_id", "fake_id"],
        }
    ).to_csv(source, index=False)
    return source, root


def test_df40_preserves_order_and_records_custom_scope(tmp_path):
    source, root = fixture(tmp_path)
    output = tmp_path / "canonical.csv"
    record = prepare(
        source, root, output, path_prefix="/old/location", group_column="identity",
        source_domain_column="source_domain", acknowledge_reviewed_subset=True,
    )
    frame, _ = load_manifest(output)
    assert frame.label.tolist() == [0, 1]
    assert frame.img_name.tolist() == ["real.png", "fake.png"]
    assert frame.sha256.str.len().eq(64).all()
    assert record["official_coverage"] == "not established"
    assert record["dataset"] == "df40-local-subset"


def test_df40_requires_provenance_review(tmp_path):
    source, root = fixture(tmp_path)
    with pytest.raises(ValueError, match="Review source labels"):
        prepare(source, root, tmp_path / "out.csv")


def test_df40_absolute_paths_cannot_escape_prefix(tmp_path):
    source, root = fixture(tmp_path)
    with pytest.raises(ValueError, match="outside --path-prefix"):
        prepare(source, root, tmp_path / "out.csv", path_prefix="/wrong",
                acknowledge_reviewed_subset=True)


def test_df40_missing_image_fails_before_certification(tmp_path):
    source, root = fixture(tmp_path)
    (root / "fake.png").unlink()
    output = tmp_path / "out.csv"
    with pytest.raises(FileNotFoundError):
        prepare(source, root, output, path_prefix="/old/location",
                acknowledge_reviewed_subset=True)
    assert not output.exists()
