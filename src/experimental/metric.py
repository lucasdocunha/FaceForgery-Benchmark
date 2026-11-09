"""Supervised metric learning on verified, frozen feature caches.

The binary class convention is always real=0, fake=1. Hard negatives are a
supervised adaptation of Robinson et al., not their unsupervised estimator.
"""

from __future__ import annotations

import numpy as np
import torch
from torch import nn
from torch.nn import functional as F
from torch.utils.data import Sampler


class ProjectionHead(nn.Module):
    def __init__(self, input_dim: int, embedding_dim: int = 128, hidden_dim: int = 256):
        super().__init__()
        if input_dim < 1 or embedding_dim not in (128, 256) or hidden_dim < 1:
            raise ValueError("Positive dimensions and embedding_dim=128 or 256 required")
        self.network = nn.Sequential(
            nn.Linear(input_dim, hidden_dim), nn.ReLU(), nn.Linear(hidden_dim, embedding_dim)
        )

    def forward(self, x):
        return F.normalize(self.network(x).float(), p=2, dim=-1, eps=1e-12)


def supervised_contrastive_loss(embeddings, labels, *, temperature=0.1, hardness=0.0):
    """Average positive log probabilities; opposite-class negatives only are mined.

    Hard-negative weights are detached and have mean one per anchor. Therefore
    hardness=0 is exactly ordinary SupCon, with no unknown-class correction.
    Every anchor needs another positive and at least one negative.
    """
    if temperature <= 0 or not np.isfinite(temperature) or not 0 <= hardness <= 20:
        raise ValueError("temperature must be positive and hardness in [0,20]")
    if embeddings.ndim != 2 or labels.ndim != 1 or len(embeddings) != len(labels):
        raise ValueError("Expected embeddings [B,D] and labels [B]")
    if len(labels) < 4 or not torch.isin(labels, labels.new_tensor([0, 1])).all():
        raise ValueError("SupCon requires a binary batch with two examples per class")
    if not torch.isfinite(embeddings).all():
        raise ValueError("Nonfinite embeddings")
    if (embeddings.float().norm(dim=1) <= 1e-12).any():
        raise ValueError("Zero embeddings do not define cosine similarity")
    z = F.normalize(embeddings.float(), dim=1)
    cosine = z @ z.T
    same = labels[:, None] == labels[None, :]
    diagonal = torch.eye(len(z), dtype=torch.bool, device=z.device)
    positive, negative = same & ~diagonal, ~same
    if (positive.sum(1) == 0).any() or (negative.sum(1) == 0).any():
        raise ValueError("Every anchor requires a positive and an opposite-class negative")
    similarity = cosine / temperature
    log_weights = torch.zeros_like(similarity)
    if hardness:
        hard_logits = (cosine.detach() * hardness).masked_fill(~negative, -torch.inf)
        log_normalizer = torch.logsumexp(hard_logits, dim=1, keepdim=True)
        log_mean_weights = hard_logits - log_normalizer + negative.sum(1, keepdim=True).log()
        log_weights = torch.where(negative, log_mean_weights, log_weights)
    denominator = torch.logsumexp(
        (similarity + log_weights).masked_fill(diagonal, -torch.inf), dim=1
    )
    mean_positive = (similarity * positive).sum(1) / positive.sum(1)
    return (denominator - mean_positive).mean()


class BalancedBatchSampler(Sampler):
    """Deterministic balanced batches without duplicate rows inside a batch."""

    def __init__(self, labels, batch_size=64, seed=42, batches=None):
        labels = np.asarray(labels)
        if batch_size < 4 or batch_size % 2 or not np.isin(labels, [0, 1]).all():
            raise ValueError("Balanced batches require even batch_size>=4 and binary labels")
        self.groups = [np.flatnonzero(labels == c) for c in (0, 1)]
        self.half = min(batch_size // 2, *(len(g) for g in self.groups))
        if self.half < 2:
            raise ValueError("Need at least two distinct source rows in each class")
        self.seed, self.epoch = int(seed), 0
        self.batches = int(batches or max(1, int(np.ceil(len(labels) / (2 * self.half)))))

    def __len__(self):
        return self.batches

    def set_epoch(self, epoch):
        self.epoch = int(epoch)

    def __iter__(self):
        rng = np.random.default_rng(self.seed + self.epoch)
        for _ in range(self.batches):
            indices = np.concatenate([rng.choice(g, self.half, replace=False) for g in self.groups])
            yield rng.permutation(indices).tolist()


def unit_rows(values):
    values = np.asarray(values, dtype=np.float32)
    if values.ndim != 2 or not np.isfinite(values).all():
        raise ValueError("Expected finite feature matrix")
    norms = np.linalg.norm(values, axis=1, keepdims=True)
    if np.any(norms <= 1e-12):
        raise ValueError("Zero feature vectors do not define cosine distance")
    return values / norms


def class_centroids(features, labels):
    z, labels = unit_rows(features), np.asarray(labels)
    if len(z) != len(labels) or set(np.unique(labels)) != {0, 1}:
        raise ValueError("Aligned real and fake training samples required")
    return unit_rows(np.stack([z[labels == c].mean(0) for c in (0, 1)]))


def centroid_score(features, centroids, *, distance="cosine"):
    z, centers = unit_rows(features), unit_rows(centroids)
    if centers.shape != (2, z.shape[1]):
        raise ValueError("Two matching class centroids required")
    if distance == "cosine":
        score = (z @ centers.T)[:, 1] - (z @ centers.T)[:, 0]
        return np.clip((score + 2) / 4, 0, 1)
    if distance in {"euclidean", "squared_euclidean"}:
        d2 = np.maximum(0, 2 - 2 * z @ centers.T)
        if distance == "euclidean":
            distances = np.sqrt(d2)
            return np.clip((distances[:, 0] - distances[:, 1] + 2) / 4, 0, 1)
        return np.clip((d2[:, 0] - d2[:, 1] + 4) / 8, 0, 1)
    raise ValueError("distance must be cosine, euclidean or squared_euclidean")


def embedding_diagnostics(features, labels, *, seed=42, max_samples=1000, tsne=False):
    """Bounded diagnostic subsample; never returns a model-selection criterion."""
    from sklearn.metrics import silhouette_score

    z, labels = unit_rows(features), np.asarray(labels)
    if len(z) != len(labels) or max_samples < 4:
        raise ValueError("Aligned labels and max_samples>=4 required")
    rng = np.random.default_rng(seed)
    indices = np.sort(rng.choice(len(z), min(len(z), max_samples), replace=False))
    x, y = z[indices], labels[indices]
    result = {"sample_indices": indices.tolist(), "selection_eligible": False}
    result["silhouette_cosine"] = (
        float(silhouette_score(x, y, metric="cosine"))
        if 1 < len(np.unique(y)) < len(y) else None
    )
    if tsne:
        from sklearn.manifold import TSNE
        result["tsne"] = TSNE(
            n_components=2, perplexity=min(30, max(2, (len(x) - 1) / 3)),
            random_state=seed, init="pca", learning_rate="auto", n_jobs=1,
        ).fit_transform(x).tolist()
    return result


def fit_metric(config):
    """Fit a projected metric model or same-feature baseline, returning its directory."""
    from .feature_training import fit_metric_run
    return fit_metric_run(config)


def predict_metric(run_dir, query_features):
    """Score unlabeled queries from a portable fitted artifact."""
    from .feature_training import predict_metric_run
    return predict_metric_run(run_dir, query_features)
