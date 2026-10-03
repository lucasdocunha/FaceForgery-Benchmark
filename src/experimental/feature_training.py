"""Cached-feature training and portable prediction artifacts for experiment G."""

from __future__ import annotations

import json
import random
import resource
import time
from functools import wraps
from pathlib import Path

import numpy as np
import torch
from torch import nn
from torch.utils.data import DataLoader, Dataset, Subset

from src.robustness.artifacts import save_predictions
from src.robustness.inference import calibrate, evaluation_report
from src.robustness.manifests import assert_disjoint
from src.robustness.provenance import contained, digest, digest_file, source_identity, write_json
from src.robustness.statistics import summary
from .metric import (
    BalancedBatchSampler, ProjectionHead, centroid_score, embedding_diagnostics,
    supervised_contrastive_loss, unit_rows,
)


def seed_cpu(seed, threads=2):
    if not 1 <= int(threads) <= 2:
        raise ValueError("Cached-feature CPU thread budget is one or two")
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.set_num_threads(int(threads))


def open_training_caches(config):
    from .features import open_cache

    caches = config.get("caches", {})
    data = config.get("data", {})
    train = open_cache(caches.get("train", data.get("train_cache")))
    val = open_cache(caches.get("val", data.get("val_cache")))
    assert_disjoint(train.frame, val.frame)
    if train.features.shape[1] != val.features.shape[1]:
        raise ValueError("Train and validation cache dimensions differ")
    if representation_key(train.metadata) != representation_key(val.metadata):
        raise ValueError("Train and validation feature representation identities differ")
    if set(train.frame.image_sha256) & set(val.frame.image_sha256):
        raise ValueError("Train/validation overlap in cached image content")
    return train, val


def representation_key(metadata):
    """Cache dependencies that must agree across different split populations."""
    return {key: value for key, value in metadata["key"].items()
            if key not in {"manifest_sha256", "image_inventory_sha256"}}


def label_budget(train, config, root):
    """Select once, persist keyed masks, and expose only selected label values."""
    from .graphs import nested_label_mask

    cfg = config.get("model", {})
    seed = int(config.get("seed", 42))
    fraction = float(cfg.get("label_fraction", 1.0 if config.get("family") != "graph" else .05))
    ids = train.frame.sample_id.astype(str).to_numpy()
    population = digest(sorted(ids.tolist()))
    source = config.get("data", {}).get("label_mask") or config.get("label_mask")
    group_column = "source_id" if "source_id" in train.frame else "group_id"
    if source:
        supplied = json.loads(Path(source).read_text())
        if supplied.get("population_sha256") != population:
            raise ValueError("Label-mask population differs from training cache")
        selected_ids = supplied["selected_ids"]
        if len(selected_ids) != len(set(selected_ids)) or not set(selected_ids) <= set(ids):
            raise ValueError("Label mask has duplicate or unknown sample IDs")
        mask = np.isin(ids, selected_ids)
        fraction = float(supplied["fraction"])
        if "label_fraction" in cfg and fraction != float(cfg["label_fraction"]):
            raise ValueError("Requested label fraction differs from supplied mask")
    else:
        mask = nested_label_mask(train.frame.label.to_numpy(), ids, fraction, seed,
                                 group_ids=train.frame[group_column])
    observed = np.full(len(ids), -1, dtype=np.int64)
    observed[mask] = train.frame.loc[mask, "label"].to_numpy(dtype=np.int64)
    if set(observed[mask]) != {0, 1}:
        raise ValueError("Both classes are required in the selected training labels")
    selected_labels = dict(zip(ids[mask].tolist(), observed[mask].tolist()))
    if source and supplied.get("selected_labels") != selected_labels:
        raise ValueError("Selected labels differ from the frozen label mask")
    for _, rows in train.frame.groupby(group_column, sort=False):
        if len(set(mask[rows.index.to_numpy()])) != 1:
            raise ValueError("All views in a source group must share the label mask")
    record = {
        "schema": "faceforgery-label-mask-v1", "seed": seed, "fraction": fraction,
        "population_sha256": population, "selected_ids": sorted(selected_labels),
        "selected_labels": selected_labels, "group_column": group_column,
        "selected_class_counts": {str(c): int((observed == c).sum()) for c in (0, 1)},
        "selected_total": int(mask.sum()), "population": len(mask),
        "interpretation": "low-label downstream adaptation; HF-supervised backbones already use all MFFI labels",
    }
    np.save(Path(root) / "observed_labels.npy", observed)
    np.save(Path(root) / "label_mask.npy", mask)
    write_json(Path(root) / "label_mask.json", record)
    return mask, observed


def fit_standardizer(features, block_size=1024):
    if len(features) < 1 or features.ndim != 2:
        raise ValueError("Nonempty feature matrix required")
    total = np.zeros(features.shape[1], dtype=np.float64)
    squares = total.copy()
    for start in range(0, len(features), block_size):
        x = np.asarray(features[start:start + block_size], dtype=np.float64)
        if not np.isfinite(x).all():
            raise ValueError("Nonfinite training features")
        total += x.sum(0)
        squares += np.square(x).sum(0)
    mean = total / len(features)
    std = np.sqrt(np.maximum(squares / len(features) - mean * mean, 0))
    return mean.astype(np.float32), np.maximum(std, 1e-6).astype(np.float32)


def standardized_block(features, mean, std):
    x = np.asarray(features, dtype=np.float32)
    if x.ndim != 2 or x.shape[1] != len(mean) or not np.isfinite(x).all():
        raise ValueError("Invalid query features or feature dimension mismatch")
    return (x - np.asarray(mean, dtype=np.float32)) / np.asarray(std, dtype=np.float32)


def measured_feature_run(function):
    @wraps(function)
    def measured(config):
        destination = Path(config.get("run_dir", config.get("output_dir")))
        if (destination / "artifact.json").exists() and config.get("training", {}).get("resume", False):
            prior = read_model_artifact(destination)

            def identity(value):
                value = {**value, "training": dict(value.get("training", {}))}
                value.pop("resume", None)
                value["training"].pop("resume", None)
                return digest(value)

            if identity(prior["config"]) != identity(config):
                raise ValueError("Resume config differs from the completed feature artifact")
            train, val = open_training_caches(config)
            if (prior["train_cache"], prior["val_cache"]) != (train.identity, val.identity):
                raise ValueError("Resume caches differ from the fitted feature artifact")
            if (destination / "calibration.json").exists():
                return destination
        started = time.perf_counter()
        root = function(config)
        record = read_model_artifact(root)
        telemetry = Path(root) / "telemetry.json"
        fit = json.loads(telemetry.read_text()) if telemetry.exists() else {}
        write_json(Path(root) / "pipeline_telemetry.json", {
            "runtime_seconds": time.perf_counter() - started,
            "peak_rss_kib": resource.getrusage(resource.RUSAGE_SELF).ru_maxrss,
            "peak_vram_bytes": fit.get("peak_vram_bytes", 0),
            "trainable_parameters": fit.get("trainable_parameters", 0),
            "epochs_completed": fit.get("epochs_completed", 0),
            "global_step": fit.get("global_step", 0),
            "seed": config.get("seed", 42), "scope": "cached-feature development",
        })
        if not (Path(root) / "run.json").exists():
            write_json(Path(root) / "run.json", {"config": config, "software": record["software"],
                                                "selection": "fixed nonparametric or shallow baseline"})
        return root
    return measured


def write_transformed_features(path, features, mean, std, *, model=None, device="cpu", block_size=512):
    dim = features.shape[1]
    if model is not None:
        model.eval()
        with torch.inference_mode():
            sample = model(torch.from_numpy(standardized_block(features[:1], mean, std)).to(device))
        dim = sample.shape[1]
    result = np.lib.format.open_memmap(path, mode="w+", dtype="float32", shape=(len(features), dim))
    for start in range(0, len(features), block_size):
        x = standardized_block(features[start:start + block_size], mean, std)
        if model is not None:
            with torch.inference_mode():
                x = model(torch.from_numpy(x).to(device)).float().cpu().numpy()
        result[start:start + len(x)] = unit_rows(x)
    result.flush()
    return np.load(path, mmap_mode="r")


def streaming_centroids(features, labels, block_size=1024):
    labels = np.asarray(labels)
    if len(features) != len(labels) or not np.isin(labels, [-1, 0, 1]).all() or set(labels[labels >= 0]) != {0, 1}:
        raise ValueError("Training centroids need both classes and aligned labels")
    sums = np.zeros((2, features.shape[1]), dtype=np.float64)
    counts = np.zeros(2, dtype=np.int64)
    for start in range(0, len(features), block_size):
        x = unit_rows(features[start:start + block_size])
        y = labels[start:start + block_size]
        for c in (0, 1):
            sums[c] += x[y == c].sum(0)
            counts[c] += (y == c).sum()
    return unit_rows(sums / counts[:, None])


class FeatureDataset(Dataset):
    def __init__(self, features, labels, mean, std):
        self.features, self.labels, self.mean, self.std = features, np.asarray(labels), mean, std

    def __len__(self):
        return len(self.features)

    def __getitem__(self, index):
        x = standardized_block(self.features[index:index + 1], self.mean, self.std)[0]
        return torch.from_numpy(x), torch.tensor(int(self.labels[index]), dtype=torch.long)


def classifier(input_dim, kind, hidden_dim=64):
    if kind == "linear":
        return nn.Linear(input_dim, 2)
    if kind == "mlp":
        return nn.Sequential(nn.Linear(input_dim, hidden_dim), nn.ReLU(), nn.Linear(hidden_dim, 2))
    raise ValueError("Classifier kind must be linear or mlp")


def predict_network(model, features, mean, std, *, device="cpu", batch_size=512):
    model.eval()
    outputs = []
    with torch.inference_mode():
        for start in range(0, len(features), batch_size):
            x = standardized_block(features[start:start + batch_size], mean, std)
            outputs.append(model(torch.from_numpy(x).to(device)).float().cpu().numpy())
    if not outputs:
        raise ValueError("Empty query feature population")
    return np.concatenate(outputs)


def train_model(model, loader, loss_step, validate, root, config):
    from .runtime import fit_model

    training = config.get("training", {})
    device = training.get("device", "cpu")
    optimizer = torch.optim.AdamW(
        model.parameters(), lr=float(training.get("lr", 1e-3)),
        weight_decay=float(training.get("weight_decay", 1e-4)),
    )
    fit_model(
        model, loader, optimizer, loss_step=loss_step, validate=validate,
        run_dir=root, config=config, epochs=int(training.get("epochs", 10)),
        device=device, grad_accum_steps=int(training.get("grad_accum_steps", 1)),
        selection_key="auc", selection_mode="max", resume=training.get("resume", False),
        max_grad_norm=float(training.get("max_grad_norm", 1.0)),
    )
    checkpoint = torch.load(Path(root) / "best.pt", map_location="cpu", weights_only=False)
    model.load_state_dict(checkpoint["state_dict"])
    model.to(device).eval()
    return model


def write_model_artifact(root, record, files):
    root = Path(root)
    record = {
        "schema": "faceforgery-feature-model-v1", "label_convention": "fake-is-1",
        **record,
        "files": {name: digest_file(root / name) for name in files},
        "implementation": {p.name: digest_file(p) for p in (
            Path(__file__), Path(__file__).with_name("metric.py"),
            Path(__file__).with_name("graphs.py"), Path(__file__).with_name("graph_training.py"),
        ) if p.exists()},
        "software": source_identity(),
    }
    write_json(root / "artifact.json", record)
    return record


def read_model_artifact(root):
    root = Path(root)
    record = json.loads((root / "artifact.json").read_text())
    if record.get("schema") != "faceforgery-feature-model-v1" or record.get("label_convention") != "fake-is-1":
        raise ValueError("Unsupported feature-model artifact")
    for name, checksum in record["files"].items():
        if Path(name).name != name or digest_file(root / name) != checksum:
            raise ValueError("Feature-model dependency missing or changed")
    return record


def describe_run(run_dir, protocol=None):
    root = Path(run_dir)
    record = read_model_artifact(root)
    checkpoint = root / "artifact.json"
    declared = record.get("inference_protocol", "inductive")
    if protocol is not None and protocol != declared:
        raise ValueError("Inference protocol differs from validation calibration; fit a separate run")
    protocol = declared
    return {
        "checkpoint_path": checkpoint,
        "input_contract": {"family": record["task"], "bundle_sha256": digest_file(checkpoint),
                           "representation_key": record["representation_key"], "protocol": protocol},
        "research_run": {"schema": "faceforgery-feature-model-v1", "config": record["config"],
                         "software": record["software"], "training_source": record["train_cache"]},
        "image_size": record["representation_key"].get("extraction", {}).get("image_size", 224),
    }


def save_validation(root, cache, probabilities, *, extra=None):
    root = Path(root)
    frame = cache.frame.copy()
    frame["p_fake"] = np.asarray(probabilities)
    checkpoint_hash = digest_file(root / "artifact.json")
    save_predictions(
        root / "val_predictions.csv", frame, manifest_record=cache.metadata["manifest"],
        model_sha256=checkpoint_hash,
        metadata={"origin": "frozen feature cache", "cache_identity": cache.identity},
    )
    calibration = calibrate(
        frame, manifest_record=cache.metadata["manifest"], output=root / "calibration.json",
        model_sha256=checkpoint_hash,
        policy="youden", input_contract=describe_run(root)["input_contract"],
    )
    report = evaluation_report(frame, calibration, checkpoint_hash)
    report.update(scope="development", cache_identity=cache.identity, **(extra or {}))
    write_json(root / "metrics.json", report)
    return report


@measured_feature_run
def fit_metric_run(config):
    training, model_config = config.get("training", {}), config.get("model", {})
    seed = int(config.get("seed", 42))
    seed_cpu(seed, training.get("cpu_threads", 2))
    kind = model_config.get("kind", "supcon")
    if kind not in {"supcon", "linear", "mlp", "centroid", "knn"}:
        raise ValueError("Unknown metric model kind")
    root = Path(config.get("run_dir", config.get("output_dir")))
    if root.exists() and not training.get("resume", False):
        raise FileExistsError("Use a fresh experiment directory")
    root.mkdir(parents=True, exist_ok=True)
    train, val = open_training_caches(config)
    mean, std = fit_standardizer(train.features) if model_config.get("standardize", True) else (
        np.zeros(train.features.shape[1], dtype=np.float32), np.ones(train.features.shape[1], dtype=np.float32))
    np.savez(root / "normalizer.npz", mean=mean, std=std)
    mask, observed = label_budget(train, config, root)
    selected = np.flatnonzero(mask)
    labels = observed[selected]
    val_labels = val.frame.label.to_numpy(dtype=np.int64)
    device = training.get("device", "cpu")
    files = ["normalizer.npz", "label_mask.json", "label_mask.npy", "observed_labels.npy"]
    model = None
    if kind in {"supcon", "linear", "mlp"}:
        if kind == "supcon":
            model = ProjectionHead(train.features.shape[1], model_config.get("embedding_dim", 128), model_config.get("hidden_dim", 256))
        else:
            model = classifier(train.features.shape[1], kind, model_config.get("hidden_dim", 64))
        model.to(device)
        dataset = Subset(FeatureDataset(train.features, observed, mean, std), selected.tolist())
        if kind == "supcon":
            sampler = BalancedBatchSampler(labels, training.get("batch_size", 64), seed)
            loader = DataLoader(dataset, batch_sampler=sampler, num_workers=0)
        else:
            loader = DataLoader(dataset, batch_size=training.get("batch_size", 64), shuffle=True, num_workers=0)

        def loss_step(current, batch, state):
            x, y = (t.to(device) for t in batch)
            if kind == "supcon":
                loss = supervised_contrastive_loss(
                    current(x), y, temperature=model_config.get("temperature", 0.1),
                    hardness=model_config.get("hardness", 0.0),
                )
            else:
                loss = nn.functional.cross_entropy(current(x), y)
            return loss, {"objective": float(loss.detach())}, len(y)

        def validate(current):
            if kind == "supcon":
                reference = write_transformed_features(root / "selection_features.npy", train.features, mean, std, model=current, device=device)
                centers = streaming_centroids(reference, observed)
                embedded = predict_network(current, val.features, mean, std, device=device)
                p = centroid_score(embedded, centers)
            else:
                logits = predict_network(current, val.features, mean, std, device=device)
                p = torch.from_numpy(logits).softmax(1)[:, 1].numpy()
            return summary(val_labels, p)

        model = train_model(model, loader, loss_step, validate, root, config)
        files.append("best.pt")
    if kind in {"supcon", "centroid", "knn"}:
        reference = write_transformed_features(
            root / "reference.npy", train.features, mean, std,
            model=model if kind == "supcon" else None, device=device,
        )
        np.save(root / "centroids.npy", streaming_centroids(reference, observed))
        np.save(root / "reference_labels.npy", observed)
        files.extend(["reference.npy", "centroids.npy", "reference_labels.npy"])
        if kind == "supcon" and model_config.get("score") == "linear":
            from .graph_training import _linear_fit
            coef, intercept = _linear_fit(reference, observed, seed, int(training.get("baseline_epochs", 30)))
            np.savez(root / "probe.npz", coef=coef, intercept=intercept)
            files.append("probe.npz")
    record = write_model_artifact(root, {
        "task": "metric", "kind": kind, "config": config, "input_dim": len(mean),
        "train_cache": train.identity, "val_cache": val.identity,
        "representation_key": representation_key(train.metadata),
        "train_manifest": train.metadata["manifest"], "validation_manifest": val.metadata["manifest"],
        "label_mask_sha256": digest_file(root / "label_mask.json"),
        "training_label_provenance": train.metadata.get("label_provenance", "backbone label exposure must be checked from checkpoint provenance"),
        "score_semantics": "fake-class probability or fixed monotone centroid distance score",
    }, files)
    p = predict_metric_run(root, val.features)
    save_validation(root, val, p, extra={"method": kind})
    if kind == "supcon":
        embeddings = predict_network(model, val.features, mean, std, device=device)
        diagnostics = embedding_diagnostics(
            embeddings, val_labels, seed=seed,
            max_samples=int(model_config.get("diagnostic_samples", 1000)),
            tsne=bool(model_config.get("tsne", False)),
        )
        centers = np.load(root / "centroids.npy")
        diagnostics["centroid_distance_auc"] = {
            distance: summary(val_labels, centroid_score(embeddings, centers, distance=distance))["auc"]
            for distance in ("cosine", "euclidean", "squared_euclidean")
        }
        write_json(root / "embedding_diagnostics.json", diagnostics)
        if "tsne" in diagnostics:
            import matplotlib
            matplotlib.use("Agg")
            import matplotlib.pyplot as plt
            points = np.asarray(diagnostics["tsne"])
            selected_labels = val_labels[diagnostics["sample_indices"]]
            figure, axis = plt.subplots(figsize=(6, 5))
            for label, name in ((0, "Real"), (1, "Fake")):
                subset = points[selected_labels == label]
                axis.scatter(subset[:, 0], subset[:, 1], s=8, alpha=.65, label=name)
            axis.set(xlabel="t-SNE 1", ylabel="t-SNE 2", title="Source-validation diagnostic, not model selection")
            axis.legend()
            figure.tight_layout()
            figure.savefig(root / "embedding_tsne.png", dpi=150)
            plt.close(figure)
    selection_file = root / "selection_features.npy"
    if selection_file.exists():
        selection_file.unlink()
    return root


def predict_metric_run(run_dir, query_features):
    root = Path(run_dir)
    record = read_model_artifact(root)
    if record["task"] != "metric":
        raise ValueError("Expected a metric artifact")
    normalizer = np.load(root / "normalizer.npz")
    mean, std = normalizer["mean"], normalizer["std"]
    kind, model_config = record["kind"], record["config"].get("model", {})
    if kind in {"linear", "mlp", "supcon"}:
        model = (
            ProjectionHead(record["input_dim"], model_config.get("embedding_dim", 128), model_config.get("hidden_dim", 256))
            if kind == "supcon" else classifier(record["input_dim"], kind, model_config.get("hidden_dim", 64))
        )
        state = torch.load(root / "best.pt", map_location="cpu", weights_only=False)
        model.load_state_dict(state["state_dict"])
        values = predict_network(model, query_features, mean, std)
        if kind != "supcon":
            return torch.from_numpy(values).softmax(1)[:, 1].numpy()
    else:
        values = standardized_block(query_features, mean, std)
    if kind == "knn" or (kind == "supcon" and model_config.get("score") == "knn"):
        from .graphs import cosine_neighbors
        reference = np.load(root / "reference.npy", mmap_mode="r")
        observed = np.load(root / "reference_labels.npy", mmap_mode="r")
        selected = np.flatnonzero(observed >= 0)
        indices, _ = cosine_neighbors(
            reference[selected], values, k=min(int(model_config.get("k", 10)), len(selected)),
            backend=model_config.get("neighbor_backend", "exact"),
            exact_limit=int(model_config.get("exact_limit", 10000)),
        )
        return observed[selected][indices].mean(1)
    if kind == "supcon" and model_config.get("score") == "linear":
        from .graph_training import _linear_probabilities
        probe = np.load(root / "probe.npz")
        return _linear_probabilities(values, probe["coef"], probe["intercept"])[:, 1]
    return centroid_score(values, np.load(root / "centroids.npy"), distance=model_config.get("distance", "cosine"))


def feature_cache_predictor(run_dir, caches, protocol=None):
    """Build suite callbacks with certified ID alignment and no query-label access."""
    from .features import open_cache
    from .graph_training import predict_graph_run

    record = read_model_artifact(run_dir)
    declared_protocol = record.get("inference_protocol", "inductive")
    if protocol is not None and protocol != declared_protocol:
        raise ValueError("Inference protocol differs from the validation-calibrated run")

    def make_callback(path):
        store = open_cache(path)
        if representation_key(store.metadata) != record["representation_key"]:
            raise ValueError("Target cache uses a different feature representation")

        def validate_population(frame, root):
            ids = frame.sample_id.astype(str)
            cached_ids = store.frame.sample_id.astype(str)
            if ids.duplicated().any() or set(ids) != set(cached_ids) or len(ids) != len(cached_ids):
                raise ValueError("Target cache and evaluation manifest populations differ")
            aligned = store.frame.set_index(cached_ids).loc[ids]
            for column in ("img_name", "group_id", "source_id", "dataset", "split", "video_id", "image_sha256"):
                if column in frame and column in aligned:
                    if frame[column].astype(str).tolist() != aligned[column].astype(str).tolist():
                        raise ValueError(f"Target cache identity differs in {column}")
            for row, cached_hash in zip(frame.itertuples(index=False), aligned.image_sha256):
                if digest_file(contained(root, row.img_name)) != cached_hash:
                    raise ValueError("Target image content differs from its cached representation")
            return aligned

        def predict(frame, root, **kwargs):
            if kwargs.get("positive_class", "fake") != "fake":
                raise ValueError("Feature predictions require declared fake class 1")
            aligned = validate_population(frame, root)
            ids = frame.sample_id.astype(str)
            cached_ids = store.frame.sample_id.astype(str)
            # The complete population is scored in cache order; labels are never passed.
            if record["task"] == "metric":
                probabilities = predict_metric_run(run_dir, store.features)
            else:
                probabilities = predict_graph_run(run_dir, store.features, protocol=declared_protocol)
            by_id = dict(zip(cached_ids, probabilities))
            result = frame.copy()
            result["p_fake"] = ids.map(by_id).to_numpy()
            result["image_sha256"] = aligned.image_sha256.to_numpy()
            return result
        predict.validate_population = validate_population
        return predict
    return {name: make_callback(path) for name, path in caches.items()}
