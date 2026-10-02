"""Cached-feature training and portable prediction artifacts for experiment G."""

from __future__ import annotations

import json
import random
from pathlib import Path

import numpy as np
import torch
from torch import nn
from torch.utils.data import DataLoader, Dataset

from src.robustness.artifacts import save_predictions
from src.robustness.inference import calibrate, evaluation_report
from src.robustness.manifests import assert_disjoint
from src.robustness.provenance import digest_file, write_json
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

    train = open_cache(config["data"]["train_cache"])
    val = open_cache(config["data"]["val_cache"])
    assert_disjoint(train.frame, val.frame)
    if train.features.shape[1] != val.features.shape[1]:
        raise ValueError("Train and validation cache dimensions differ")
    if train.metadata["checkpoint"]["sha256"] != val.metadata["checkpoint"]["sha256"]:
        raise ValueError("Train and validation caches use different checkpoints")
    if train.metadata["extraction"] != val.metadata["extraction"]:
        raise ValueError("Train and validation feature extraction differs")
    return train, val


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
    return (x - mean) / std


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
    if len(features) != len(labels) or set(np.unique(labels)) != {0, 1}:
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
            Path(__file__).with_name("graphs.py"),
        ) if p.exists()},
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
    )
    report = evaluation_report(frame, calibration, checkpoint_hash)
    report.update(scope="development", cache_identity=cache.identity, **(extra or {}))
    write_json(root / "metrics.json", report)
    return report


def fit_metric_run(config):
    training, model_config = config.get("training", {}), config.get("model", {})
    seed = int(config.get("seed", 42))
    seed_cpu(seed, training.get("cpu_threads", 2))
    kind = model_config.get("kind", "supcon")
    if kind not in {"supcon", "linear", "mlp", "centroid", "knn"}:
        raise ValueError("Unknown metric model kind")
    root = Path(config["output_dir"])
    if root.exists() and not training.get("resume", False):
        raise FileExistsError("Use a fresh experiment directory")
    root.mkdir(parents=True, exist_ok=True)
    train, val = open_training_caches(config)
    mean, std = fit_standardizer(train.features)
    np.savez(root / "normalizer.npz", mean=mean, std=std)
    labels = train.frame.label.to_numpy(dtype=np.int64)
    val_labels = val.frame.label.to_numpy(dtype=np.int64)
    device = training.get("device", "cpu")
    files = ["normalizer.npz"]
    model = None
    if kind in {"supcon", "linear", "mlp"}:
        if kind == "supcon":
            model = ProjectionHead(train.features.shape[1], model_config.get("embedding_dim", 128), model_config.get("hidden_dim", 256))
        else:
            model = classifier(train.features.shape[1], kind, model_config.get("hidden_dim", 64))
        model.to(device)
        dataset = FeatureDataset(train.features, labels, mean, std)
        if kind == "supcon":
            sampler = BalancedBatchSampler(labels, training.get("batch_size", 64), seed)
            loader = DataLoader(dataset, batch_sampler=sampler, num_workers=0)
        else:
            loader = DataLoader(dataset, batch_size=training.get("batch_size", 64), shuffle=True, num_workers=0)

        def loss_step(current, batch, state):
            x, y = (t.to(device) for t in batch)
            if kind == "supcon":
                sampler.set_epoch(state["epoch"])
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
                centers = streaming_centroids(reference, labels)
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
        np.save(root / "centroids.npy", streaming_centroids(reference, labels))
        np.save(root / "reference_labels.npy", labels)
        files.extend(["reference.npy", "centroids.npy", "reference_labels.npy"])
    record = write_model_artifact(root, {
        "task": "metric", "kind": kind, "config": config, "input_dim": len(mean),
        "train_cache": train.identity, "val_cache": val.identity,
        "training_label_provenance": train.metadata.get("label_provenance", "backbone label exposure must be checked from checkpoint provenance"),
        "score_semantics": "fake-class probability or fixed monotone centroid distance score",
    }, files)
    p = predict_metric_run(root, val.features)
    save_validation(root, val, p, extra={"method": kind})
    if kind == "supcon":
        embeddings = predict_network(model, val.features, mean, std, device=device)
        write_json(root / "embedding_diagnostics.json", embedding_diagnostics(
            embeddings, val_labels, seed=seed,
            max_samples=int(model_config.get("diagnostic_samples", 1000)),
            tsne=bool(model_config.get("tsne", False)),
        ))
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
        indices, _ = cosine_neighbors(
            reference, values, k=int(model_config.get("k", 10)),
            backend=model_config.get("neighbor_backend", "exact"),
            exact_limit=int(model_config.get("exact_limit", 10000)),
        )
        return np.load(root / "reference_labels.npy", mmap_mode="r")[indices].mean(1)
    return centroid_score(values, np.load(root / "centroids.npy"), distance=model_config.get("distance", "cosine"))
