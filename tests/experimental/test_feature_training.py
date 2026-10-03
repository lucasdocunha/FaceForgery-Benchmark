import json
from pathlib import Path

import numpy as np
import pandas as pd
import pytest
import torch

from src.experimental.feature_training import (
    describe_run, feature_cache_predictor, fit_metric_run, predict_metric_run,
)
from src.experimental.features import open_cache
from src.experimental.graph_training import (
    _predict_gnn, disjoint_ego_batch, fit_graph_run, predict_graph_run,
)
from src.experimental.graphs import GraphClassifier, cosine_neighbors, nested_label_mask
from src.robustness.manifests import save_manifest
from src.robustness.provenance import digest, digest_file, write_json


def write_cache(root, split, *, labels=None, features=None):
    root.mkdir(parents=True)
    n = 80 if split == "train" else 24
    y = np.tile([0, 1], n // 2) if labels is None else labels
    rng = np.random.default_rng(42 if split == "train" else 7)
    values = rng.normal(0, .15, (n, 8)) if features is None else features.copy()
    if features is None:
        values[:, 0] += 2 * y - 1
        values[:, 1] += 1 - 2 * y
    ids = [f"{split}-{i}" for i in range(n)]
    for name in ids:
        (root / f"{name}.png").write_bytes(name.encode())
    frame = pd.DataFrame({
        "sample_id": ids, "group_id": ids, "label": y,
        "img_name": [f"{name}.png" for name in ids],
        "image_sha256": [digest_file(root / f"{name}.png") for name in ids],
        "dataset": "feature-fixture", "split": split,
    })
    manifest = root / "manifest.csv"
    save_manifest(frame.drop(columns="image_sha256"), manifest, {})
    certificate = json.loads(Path(str(manifest) + ".json").read_text())
    np.save(root / "features.npy", values.astype(np.float16))
    np.save(root / "logits.npy", np.column_stack((-values[:, 0], values[:, 0])).astype(np.float32))
    frame.to_csv(root / "rows.csv", index=False)
    extraction = {"mode": "srm", "image_size": 32, "srm_on_normalized": True}
    key = {"checkpoint_sha256": "fixture-checkpoint", "run_config_sha256": "fixture-config",
           "manifest_sha256": certificate["manifest_sha256"], "extraction": extraction,
           "image_inventory_sha256": digest(ids), "commit": "fixture-commit"}
    metadata = {
        "schema": "faceforgery-features-v1", "status": "complete", "key": key,
        "identity": digest(key), "sample_ids_sha256": digest(ids),
        "checkpoint": {"sha256": "fixture-checkpoint"}, "extraction": extraction,
        "manifest": certificate,
        "files": {name: digest_file(root / name) for name in ("features.npy", "logits.npy", "rows.csv")},
    }
    write_json(root / "cache.json", metadata)
    return str(root)


@pytest.fixture
def caches(tmp_path):
    return {split: write_cache(tmp_path / split, split) for split in ("train", "val")}


def configuration(tmp_path, caches, family, kind, **model):
    return {
        "family": family, "run_dir": str(tmp_path / f"{family}-{kind}"), "caches": caches,
        "seed": 42,
        "model": {"kind": kind, "hidden_dim": 16, "label_fraction": .25,
                  "backend": "native", "projection_dim": 16, "embedding_dim": 128,
                  "graph": {"k": 3, "backend": "exact", "iterations": 5},
                  "fanouts": [3, 2], **model},
        "training": {"epochs": 2, "batch_size": 8, "baseline_epochs": 3,
                     "device": "cpu", "lr": .01, "cpu_threads": 2},
    }


@pytest.mark.parametrize("kind", ["linear", "mlp", "centroid", "knn", "supcon"])
def test_metric_cache_fit_reload_and_certified_prediction(tmp_path, caches, kind):
    config = configuration(tmp_path, caches, "metric", kind)
    run = fit_metric_run(config)
    val = open_cache(caches["val"])
    expected = pd.read_csv(run / "val_predictions.csv").set_index("sample_id").loc[val.frame.sample_id, "p_fake"].to_numpy()
    actual = predict_metric_run(run, val.features)
    np.testing.assert_allclose(actual, expected, atol=1e-7)
    assert np.isfinite(actual).all() and np.ptp(actual) > .01
    assert json.loads((run / "metrics.json").read_text())["frame"]["auc"] > .9
    callback = feature_cache_predictor(run, {"fixture": caches["val"]})["fixture"]
    changed = val.frame.iloc[::-1].copy()
    changed["label"] = 1 - changed.label
    predictions = callback(changed, caches["val"])
    np.testing.assert_allclose(predictions.p_fake, actual[::-1])
    description = describe_run(run)
    calibration = json.loads((run / "calibration.json").read_text())
    assert calibration["input_contract"] == description["input_contract"]
    assert calibration["policy"] == "youden"
    with pytest.raises(ValueError, match="populations"):
        callback(changed.iloc[:-1], caches["val"])


@pytest.mark.parametrize("kind", ["linear", "mlp", "knn", "lp", "correct_smooth", "gcn", "gat", "sage"])
def test_graph_cache_fit_reload_and_same_mask(tmp_path, caches, kind):
    config = configuration(tmp_path, caches, "graph", kind)
    run = fit_graph_run(config)
    val = open_cache(caches["val"])
    expected = pd.read_csv(run / "val_predictions.csv").set_index("sample_id").loc[val.frame.sample_id, "p_fake"].to_numpy()
    actual = predict_graph_run(run, val.features)
    np.testing.assert_allclose(actual, expected, atol=1e-7)
    singleton = np.array([predict_graph_run(run, val.features[i:i + 1])[0] for i in (0, 1, 4)])
    np.testing.assert_allclose(singleton, actual[[0, 1, 4]], atol=2e-7)
    assert np.isfinite(actual).all() and np.ptp(actual) > .001
    observed = np.load(run / "observed_labels.npy")
    mask = nested_label_mask(open_cache(caches["train"]).frame.label, open_cache(caches["train"]).frame.sample_id, .25)
    assert np.array_equal(observed >= 0, mask)
    assert set(json.loads((run / "baseline_metrics.json").read_text())["methods"]) == {"linear", "knn", "lp", "correct_smooth"}
    with pytest.raises(ValueError, match="protocol"):
        predict_graph_run(run, val.features, protocol="transductive")


@pytest.mark.parametrize("kind", ["lp", "correct_smooth", "sage"])
def test_transductive_calibration_is_separate_and_query_labels_are_unused(tmp_path, caches, kind):
    config = configuration(tmp_path, caches, "graph", kind, protocol="transductive")
    run = fit_graph_run(config)
    assert describe_run(run)["input_contract"]["protocol"] == "transductive"
    callback = feature_cache_predictor(run, {"val": caches["val"]})["val"]
    val = open_cache(caches["val"])
    original = callback(val.frame, caches["val"]).p_fake
    changed = val.frame.copy()
    changed["label"] = 1 - changed.label
    np.testing.assert_array_equal(original, callback(changed, caches["val"]).p_fake)
    with pytest.raises(ValueError, match="protocol"):
        describe_run(run, protocol="inductive")


@pytest.mark.parametrize("family,kind", [("metric", "supcon"), ("graph", "sage"), ("graph", "correct_smooth")])
def test_hidden_training_labels_do_not_affect_fixed_mask_predictions(tmp_path, caches, family, kind):
    first = fit_metric_run if family == "metric" else fit_graph_run
    config = configuration(tmp_path, caches, family, kind)
    run = first(config)
    train = open_cache(caches["train"])
    mask = np.load(run / "label_mask.npy")
    changed = train.frame.label.to_numpy().copy()
    changed[~mask] = 1 - changed[~mask]
    second_cache = write_cache(tmp_path / "hidden-changed", "train", labels=changed, features=train.features)
    other = {**config, "run_dir": str(tmp_path / "second"),
             "caches": {**caches, "train": second_cache}, "label_mask": str(run / "label_mask.json")}
    second = first(other)
    a = pd.read_csv(run / "val_predictions.csv").p_fake.to_numpy()
    b = pd.read_csv(second / "val_predictions.csv").p_fake.to_numpy()
    np.testing.assert_array_equal(a, b)


@pytest.mark.parametrize("backend", ["native", "pyg"])
@pytest.mark.parametrize("kind", ["gcn", "gat", "sage"])
def test_directed_ego_graphs_are_invariant_to_query_batch_composition(backend, kind):
    if backend == "pyg":
        pytest.importorskip("torch_geometric")
    torch.manual_seed(42)
    x = np.random.default_rng(42).normal(size=(15, 8)).astype(np.float32)
    queries = np.random.default_rng(7).normal(size=(4, 8)).astype(np.float32)
    source_neighbors, _ = cosine_neighbors(x, k=3)
    query_neighbors, _ = cosine_neighbors(x, queries, k=3)
    model = GraphClassifier(8, kind=kind, backend=backend, hidden_dim=8, projection_dim=8)
    joint = _predict_gnn(model, x, queries, source_neighbors, query_neighbors, fanouts=(3, 2), batch_size=4)
    single = _predict_gnn(model, x, queries, source_neighbors, query_neighbors, fanouts=(3, 2), batch_size=1)
    np.testing.assert_allclose(joint, single, atol=1e-7)


def test_dynamic_graph_selected_projection_round_trip(tmp_path, caches):
    config = configuration(tmp_path, caches, "graph", "sage",
                           graph={"k": 3, "backend": "exact", "iterations": 5, "dynamic": True, "rebuild_interval": 1})
    run = fit_graph_run(config)
    assert (run / "graph_space.npy").is_file()
    val = open_cache(caches["val"])
    actual = predict_graph_run(run, val.features)
    expected = pd.read_csv(run / "val_predictions.csv").set_index("sample_id").loc[val.frame.sample_id, "p_fake"]
    np.testing.assert_allclose(actual, expected, atol=1e-7)


def test_grouped_views_have_nested_label_budget():
    y = np.tile([0, 1], 20)
    groups = np.repeat([f"source-{i}" for i in range(20)], 2)
    ids = [f"view-{i}" for i in range(40)]
    small = nested_label_mask(y, ids, .05, group_ids=groups)
    large = nested_label_mask(y, ids, .1, group_ids=groups)
    assert np.all(~small | large)
    assert all(small[2 * i] == small[2 * i + 1] for i in range(20))


@pytest.mark.parametrize("score", ["knn", "linear"])
def test_projected_metric_scoring_rules(tmp_path, caches, score):
    config = configuration(tmp_path, caches, "metric", "supcon", score=score)
    run = fit_metric_run(config)
    scores = predict_metric_run(run, open_cache(caches["val"]).features)
    assert np.isfinite(scores).all() and np.ptp(scores) > .01


def test_graph_uses_portable_metric_projection_with_identical_mask(tmp_path, caches):
    metric = fit_metric_run(configuration(tmp_path, caches, "metric", "supcon"))
    config = configuration(tmp_path, caches, "graph", "sage")
    config["projection_run"] = str(metric)
    run = fit_graph_run(config)
    assert np.load(run / "reference.npy").shape[1] == 128
    val = open_cache(caches["val"])
    p = predict_graph_run(run, val.features)
    # Prediction requires only the graph bundle, not the original projection directory.
    metric.rename(tmp_path / "relocated-metric")
    np.testing.assert_array_equal(p, predict_graph_run(run, val.features))
    config["projection_run"] = str(tmp_path / "relocated-metric")
    config["model"]["label_fraction"] = .5
    config["run_dir"] = str(tmp_path / "different-mask")
    with pytest.raises(ValueError, match="exact same graph label mask"):
        fit_graph_run(config)


def test_callback_rejects_changed_images_and_artifact_mutation(tmp_path, caches):
    run = fit_metric_run(configuration(tmp_path, caches, "metric", "centroid"))
    callback = feature_cache_predictor(run, {"val": caches["val"]})["val"]
    frame = open_cache(caches["val"]).frame
    callback.validate_population(frame, caches["val"])
    (Path(caches["val"]) / frame.iloc[0].img_name).write_bytes(b"replaced-image")
    with pytest.raises(ValueError, match="image content"):
        callback(frame, caches["val"])
    np.save(run / "centroids.npy", np.ones((2, 8)))
    with pytest.raises(ValueError, match="dependency"):
        predict_metric_run(run, open_cache(caches["val"]).features)


@pytest.mark.parametrize("family,kind", [("metric", "supcon"), ("graph", "sage")])
def test_completed_resume_preserves_frozen_calibration_and_rejects_changes(tmp_path, caches, family, kind):
    fit = fit_metric_run if family == "metric" else fit_graph_run
    config = configuration(tmp_path, caches, family, kind)
    run = fit(config)
    before = digest_file(run / "calibration.json"), digest_file(run / "artifact.json")
    config["training"]["resume"] = True
    assert fit(config) == run
    assert before == (digest_file(run / "calibration.json"), digest_file(run / "artifact.json"))
    config["training"]["lr"] = .02
    with pytest.raises(ValueError, match="Resume config"):
        fit(config)


def test_full_scale_reference_uses_bounded_ego_batches():
    reference = np.broadcast_to(np.ones((1, 8), dtype=np.float32), (524429, 8))
    neighbors = np.broadcast_to(np.arange(1, 11), (len(reference), 10))
    queries = np.ones((4, 8), dtype=np.float32)
    x, edges, roots = disjoint_ego_batch(reference, queries, neighbors[:4], neighbors, fanouts=(5, 3))
    assert x.shape[0] <= 4 * (1 + 5 + 5 * 3)
    assert edges.shape[1] <= 4 * (5 + 5 * 3)
    assert len(roots) == 4
