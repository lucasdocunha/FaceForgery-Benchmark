"""Hand-computed metric and frozen input-contract checks."""

import json

import numpy as np
import pandas as pd
import pytest

from src.robustness.inference import calibrate, prediction_contract, validate_input_contract
from src.robustness.statistics import choose_threshold, equal_error_point, summary, subgroup_metrics


@pytest.mark.parametrize("labels,scores,expected", [
    ([0, 0, 1, 1], [0.1, 0.2, 0.8, 0.9], 0),
    ([0, 0, 1, 1], [0.8, 0.9, 0.1, 0.2], 1),
    ([0, 0, 1, 1], [0.5, 0.5, 0.5, 0.5], 0.5),
    ([0, 0, 1, 1], [0.1, 0.6, 0.4, 0.9], 0.5),
])
def test_eer_known_roc_crossings(labels, scores, expected):
    point = equal_error_point(labels, scores)
    assert point["eer"] == pytest.approx(expected)
    assert np.isfinite(point["eer_threshold"])
    assert point["eer_threshold"] <= np.nextafter(1.0, 2.0)


def test_eer_interpolated_toy_threshold():
    # FPR/FNR go from (0, 1/2) at .9 to (1, 1/2) at .6.
    point = equal_error_point([1, 0, 1], [0.9, 0.6, 0.3])
    assert point == pytest.approx({"eer": 0.5, "eer_threshold": 0.75})


def test_single_class_and_f1_confusion_contract():
    result = summary([0, 0, 1, 1], [0.1, 0.8, 0.6, 0.4], 0.5)
    assert result["f1"] == 0.5
    assert result["confusion_matrix"] == [[1, 1], [1, 1]]
    assert result["confusion_matrix_normalized"] == [[0.5, 0.5], [0.5, 0.5]]
    one = summary([0, 0], [0.1, 0.2])
    assert one["auc"] is one["eer"] is one["eer_threshold"] is None
    assert one["f1"] == 0
    assert one["confusion_matrix_normalized"][1] == [None, None]
    json.dumps(one, allow_nan=False)


def test_youden_is_same_optimizer_with_ties_and_endpoints():
    for scores in ([0, 0.5, 0.5, 1], [0.2] * 4, [1, 0, 1, 0]):
        assert choose_threshold([0, 1, 0, 1], scores, "youden") == choose_threshold([0, 1, 0, 1], scores)
    with pytest.raises(ValueError, match="policy"):
        choose_threshold([0, 1], [0.1, 0.9], "test_optimal")


def test_frozen_input_contract_rejects_resize_and_tampering(tmp_path):
    frame = pd.DataFrame({"sample_id": ["a", "b"], "group_id": ["a", "b"],
                          "label": [0, 1], "p_fake": [0.1, 0.9]})
    contract = prediction_contract(32, "srm", 6)
    calibration = calibrate(frame, manifest_record={"split": "val", "dataset": "synthetic",
                            "manifest_sha256": "source", "rows": 2},
                            output=tmp_path / "calibration.json", model_sha256="weights",
                            policy="youden", input_contract=contract)
    validate_input_contract(calibration, contract)
    assert calibration["policy"] == "youden"
    with pytest.raises(ValueError, match="different input"):
        validate_input_contract(calibration, prediction_contract(64, "srm", 6))
    calibration["input_contract"]["image_size"] = 64
    with pytest.raises(ValueError, match="changed"):
        validate_input_contract(calibration, contract)


def test_df40_real_reference_policies_are_explicit():
    frame = pd.DataFrame({"sample_id": ["a", "b", "c", "d"], "group_id": list("abcd"),
                          "label": [0, 0, 1, 1], "p_fake": [0.1, 0.95, 0.8, 0.7],
                          "generator": ["real", "real", "g", "g"], "paradigm": ["real", "real", "diffusion", "diffusion"],
                          "source_domain": ["A", "B", "A", "A"]})
    matched = subgroup_metrics(frame, 0.5)
    pooled = subgroup_metrics(frame, 0.5, real_reference_policy="pooled_all_real")
    assert matched["macro_auc"] == 1
    assert pooled["macro_auc"] == 0.5
    assert subgroup_metrics(frame, 0.5, column="paradigm")["groups"][0]["paradigm"] == "diffusion"
    with pytest.raises(ValueError, match="source_domain"):
        subgroup_metrics(frame.drop(columns="source_domain"), 0.5)
