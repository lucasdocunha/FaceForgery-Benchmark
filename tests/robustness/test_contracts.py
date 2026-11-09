import json
from pathlib import Path
import numpy as np
import pandas as pd
import pytest
from sklearn.metrics import roc_auc_score, balanced_accuracy_score
from src.robustness.manifests import (
    labels,
    validate,
    save_manifest,
    load_manifest,
    convert_manifest,
    assert_disjoint,
)
from src.robustness.provenance import contained, digest_file
from src.robustness.statistics import (
    summary,
    align,
    choose_threshold,
    grouped_auc_interval,
    complementarity,
    aggregate_videos,
    generator_metrics,
)


def manifest(n=8, split="train"):
    return pd.DataFrame(
        {
            "img_name": [f"{split}/{i}.png" for i in range(n)],
            "label": [i % 2 for i in range(n)],
            "sample_id": [f"{split}:{i:03}" for i in range(n)],
            "group_id": [f"{split}:g{i}" for i in range(n)],
            "dataset": "synthetic",
            "split": split,
        }
    )


def predictions():
    f = manifest(split="test")
    f["p_fake"] = [0.1, 0.8, 0.3, 0.6, 0.7, 0.4, 0.2, 0.9]
    return f


def test_label_conversion_is_explicit():
    assert labels([1, 0], "real-is-1").tolist() == [0, 1]
    with pytest.raises(ValueError):
        labels([0, 1], "auto")
    with pytest.raises(ValueError):
        labels([0.9, 0], "real-is-1")
    with pytest.raises(ValueError):
        labels([np.nan, 0], "fake-is-1")


def test_label_inversion_changes_auc_not_model():
    y = np.array([0, 0, 1, 1])
    p = np.array([0.1, 0.7, 0.6, 0.9])
    assert roc_auc_score(1 - y, p) == pytest.approx(1 - roc_auc_score(y, p))


@pytest.mark.parametrize(
    "bad", ["../secret", "/absolute", "a/../../b", "a\\b", "C:drive"]
)
def test_path_escape_rejected(tmp_path, bad):
    with pytest.raises(ValueError):
        contained(tmp_path, bad)


def test_symlink_escape_rejected(tmp_path):
    (tmp_path / "link").symlink_to("/tmp")
    with pytest.raises(ValueError):
        contained(tmp_path, "link/file")


def test_manifest_certificate_and_no_overwrite(tmp_path):
    p = tmp_path / "train.csv"
    save_manifest(manifest(), p, {})
    assert len(load_manifest(p)[0]) == 8
    with pytest.raises(FileExistsError):
        save_manifest(manifest(), p, {})
    p.write_text(p.read_text() + "\n")
    with pytest.raises(ValueError, match="bytes"):
        load_manifest(p)


def test_split_group_leakage_detected():
    a, b = manifest(), manifest(split="val")
    assert_disjoint(a, b)
    b.loc[0, "group_id"] = a.loc[0, "group_id"]
    with pytest.raises(ValueError, match="overlap"):
        assert_disjoint(a, b)


def test_test_cannot_be_selection_split():
    with pytest.raises(ValueError):
        assert_disjoint(manifest(), manifest(split="test"))


def test_converter_preserves_original_and_identity(tmp_path):
    source = tmp_path / "old.csv"
    pd.DataFrame(
        {
            "img_name": ["Celeb-real/id.png", "Celeb-synthesis/id.png"],
            "target": [1, 0],
            "vid": ["real/id", "fake/id"],
        }
    ).to_csv(source, index=False)
    before = digest_file(source)
    dest = tmp_path / "new.csv"
    convert_manifest(
        source,
        dest,
        dataset="celeb",
        split="test",
        label_column="target",
        convention="real-is-1",
        group_column="vid",
    )
    f, _ = load_manifest(dest)
    assert (
        f.label.tolist() == [0, 1]
        and f.sample_id.nunique() == 2
        and digest_file(source) == before
    )


def test_population_alignment_is_keyed():
    f = predictions()
    a, b = align(f, f.sample(frac=1, random_state=9))
    assert a.sample_id.equals(b.sample_id)
    with pytest.raises(ValueError):
        align(f, f.iloc[:-1])
    g = f.copy()
    g.loc[0, "label"] = 1
    with pytest.raises(ValueError):
        align(f, g)


@pytest.mark.parametrize("value", [np.nan, np.inf, -0.1, 1.1])
def test_bad_probabilities_fail(value):
    f = predictions()
    f.loc[0, "p_fake"] = value
    with pytest.raises(ValueError):
        align(f, f)


def test_one_class_auc_is_undefined_not_zero():
    assert summary([1, 1], [0.3, 0.7])["auc"] is None


def test_threshold_matches_brute_force_and_ties():
    f = predictions()
    y, p = f.label.to_numpy(), f.p_fake.to_numpy()
    candidates = np.unique(np.r_[0, 0.5, 1, p, np.nextafter(1.0, 2.0)])
    expected = max(candidates, key=lambda t: (balanced_accuracy_score(y, p >= t), -t))
    assert choose_threshold(y, p) == expected


def test_paired_bootstrap_zero_difference():
    f = predictions()
    r = grouped_auc_interval(f, f, draws=50)
    assert r["estimate"] == r["lower"] == r["upper"] == 0
    assert r == grouped_auc_interval(f, f, draws=50)


def test_bootstrap_rejects_single_cluster():
    f = predictions()
    f["group_id"] = "one"
    with pytest.raises(ValueError):
        grouped_auc_interval(f, draws=50)


def test_complementarity_counts_and_oracle_label():
    a = predictions()
    b = a.copy()
    b["p_fake"] = b.label * 0.8 + 0.1
    report, cases = complementarity(a, b, reference_threshold=0.5, other_threshold=0.5)
    assert report["repaired_by_other"] == 2 and report["both_wrong"] == 0
    assert report["oracle_accuracy_upper_bound_not_deployable"] == 1
    assert len(cases) == len(a)


def test_video_aggregation_and_conflicts():
    f = predictions()
    f["video_id"] = f.sample_id
    assert len(aggregate_videos(f)) == len(f)
    f.loc[1, "video_id"] = f.loc[0, "video_id"]
    with pytest.raises(ValueError):
        aggregate_videos(f)


def test_generator_requires_real_reference():
    f = predictions()
    f["generator"] = np.where(f.label == 1, "fakegen", "real")
    f["source_domain"] = "source"
    report = generator_metrics(f, 0.5)
    assert report["auc_defined_groups"] == 1
    f.loc[f.label.eq(1), "source_domain"] = "different"
    report = generator_metrics(f, 0.5)
    assert report["auc_defined_groups"] == 0 and report["macro_auc"] is None


def test_celeb_parser_inverts_verified_source_labels_and_namespaces(tmp_path):
    from scripts.prepare_celeb_df import parse_testing_list

    file = tmp_path / "List_of_testing_videos.txt"
    file.write_text("1 Celeb-real/id0.mp4\n0 Celeb-synthesis/id0.mp4\n")
    entries = parse_testing_list(file, tmp_path)
    assert [e[0] for e in entries] == [0, 1] and entries[0][2] != entries[1][2]
    file.write_text("0 Celeb-real/id0.mp4\n")
    with pytest.raises(ValueError, match="mismatch"):
        parse_testing_list(file, tmp_path)
