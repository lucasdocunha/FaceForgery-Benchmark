import json
import numpy as np
import pandas as pd
import pytest
import torch
from torch import nn
from src.robustness.explain import attribute, sample_cohort, load_cohort, overlay_bytes
from src.robustness.artifacts import save_predictions
from src.robustness.provenance import SCHEMA, write_json


class LinearImage(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv = nn.Conv2d(3, 2, 1, bias=False)

    def forward(self, x):
        return self.conv(x).mean((-2, -1))


@pytest.mark.parametrize("method", ["integrated_gradients", "occlusion", "gradcam"])
def test_primary_methods_produce_finite_maps(method):
    model = LinearImage().eval().requires_grad_(False)
    x = torch.rand(1, 3, 8, 8)
    attr, meta = attribute(
        model, x, method, baseline=torch.zeros_like(x), steps=8, layer="conv"
    )
    assert torch.isfinite(attr).all() and attr.shape[-2:] == (8, 8)
    assert meta["target"] == "fake_minus_real_logit_margin"
    assert overlay_bytes(x[0], attr[0]).startswith(b"\x89PNG")
    if method == "integrated_gradients":
        assert abs(meta["completeness_residual"]) < 1e-5


def test_cam_requires_declared_layer():
    with pytest.raises(ValueError, match="explicit"):
        attribute(
            LinearImage(),
            torch.rand(1, 3, 8, 8),
            "gradcam",
            baseline=torch.zeros(1, 3, 8, 8),
        )


def test_fixed_cohort_exact_strata_and_integrity(tmp_path):
    frame = pd.DataFrame(
        {
            "sample_id": [f"s{i:03}" for i in range(80)],
            "group_id": [f"g{i}" for i in range(80)],
            "label": ([1] * 40 + [0] * 40),
            "p_fake": ([0.9] * 20 + [0.1] * 20 + [0.1] * 20 + [0.9] * 20),
        }
    )
    record = {"manifest_sha256": "source", "dataset": "synthetic", "split": "test"}
    predictions = tmp_path / "predictions.csv"
    save_predictions(predictions, frame, manifest_record=record, model_sha256="m")
    calibration = tmp_path / "cal.json"
    write_json(
        calibration,
        {
            "schema": SCHEMA,
            "selection_split": "val",
            "model_sha256": "m",
            "checkpoint_class1": "fake",
            "frame_threshold": 0.5,
        },
    )
    first = tmp_path / "first.csv"
    second = tmp_path / "second.csv"
    sample_cohort(predictions, calibration, first)
    sample_cohort(predictions, calibration, second)
    a, _ = load_cohort(first)
    b, _ = load_cohort(second)
    assert a.equals(b) and len(a) == 64 and a.sample_id.nunique() == 64
    assert set(a.reference_stratum.value_counts()) == {16}
    first.write_text(first.read_text() + "\n")
    with pytest.raises(ValueError, match="modified"):
        load_cohort(first)
