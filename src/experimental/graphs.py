"""Bounded cosine graphs, masked graph baselines, and trainable GNN layers."""

from __future__ import annotations

import hashlib
import math

import numpy as np
import torch
from torch import nn
from torch.nn import functional as F

from .metric import unit_rows


def cosine_neighbors(reference, queries=None, *, k=10, backend="exact", exact_limit=10000,
                     query_block=128, reference_block=2048, self_indices=None,
                     reference_ids=None, memory_budget_mb=1024, seed=42, hnsw_m=16,
                     ef_search=64):
    """Return [queries,k] source indices and cosine scores without dense N*N arrays.

    Exact search has an explicit size guard, not just blocked memory. FAISS is an
    optional CPU ANN dependency and never silently falls back to exact search.
    """
    if queries is None:
        queries = reference
        self_indices = np.arange(len(reference), dtype=np.int64)
    if reference.ndim != 2 or queries.ndim != 2 or reference.shape[1] != queries.shape[1]:
        raise ValueError("Reference and query feature dimensions differ")
    if not len(queries) or k < 1 or k > len(reference) - int(self_indices is not None):
        raise ValueError("Invalid k or empty graph population")
    if query_block < 1 or reference_block < 1 or memory_budget_mb <= 0:
        raise ValueError("Positive search block and memory limits required")
    if self_indices is not None:
        self_indices = np.asarray(self_indices, dtype=np.int64)
        if self_indices.shape != (len(queries),) or ((self_indices < -1) | (self_indices >= len(reference))).any():
            raise ValueError("Invalid self-exclusion indices")
    if reference_ids is None:
        tie_rank = np.arange(len(reference))
    else:
        names = np.asarray(reference_ids, dtype=str)
        if len(names) != len(reference) or len(np.unique(names)) != len(names):
            raise ValueError("Unique reference IDs required")
        tie_rank = np.argsort(np.argsort(names, kind="stable"), kind="stable")
    result_indices = np.empty((len(queries), k), dtype=np.int64)
    result_scores = np.empty((len(queries), k), dtype=np.float32)
    if backend == "exact":
        if len(reference) > exact_limit:
            raise ValueError("Exact cosine search exceeds exact_limit; configure backend=faiss for full scale")
        tile_bytes = query_block * (reference_block + k) * 24
        if tile_bytes > memory_budget_mb * 2**20:
            raise ValueError("Exact search tile exceeds configured memory budget")
        for qstart in range(0, len(queries), query_block):
            q = unit_rows(queries[qstart:qstart + query_block])
            best_scores = np.empty((len(q), 0), dtype=np.float32)
            best_indices = np.empty((len(q), 0), dtype=np.int64)
            for rstart in range(0, len(reference), reference_block):
                r = unit_rows(reference[rstart:rstart + reference_block])
                score = q @ r.T
                indices = np.broadcast_to(np.arange(rstart, rstart + len(r)), score.shape)
                if self_indices is not None:
                    score[indices == self_indices[qstart:qstart + len(q), None]] = -np.inf
                scores = np.concatenate((best_scores, score), axis=1)
                candidates = np.concatenate((best_indices, indices), axis=1)
                order = np.lexsort((tie_rank[candidates], -scores), axis=1)[:, :k]
                best_scores = np.take_along_axis(scores, order, axis=1)
                best_indices = np.take_along_axis(candidates, order, axis=1)
            if not np.isfinite(best_scores).all():
                raise ValueError("Not enough valid neighbors after self exclusion")
            result_indices[qstart:qstart + len(q)] = best_indices
            result_scores[qstart:qstart + len(q)] = np.clip(best_scores, -1, 1)
    elif backend == "faiss":
        try:
            import faiss
        except ImportError as error:
            raise ImportError("ANN graph construction requires optional dependency faiss-cpu") from error
        estimated = len(reference) * (4 * reference.shape[1] + 8 * hnsw_m)
        if estimated > memory_budget_mb * 2**20:
            raise MemoryError("Estimated FAISS index exceeds memory_budget_mb; reduce graph dimension or raise an audited budget")
        faiss.omp_set_num_threads(2)
        index = faiss.IndexHNSWFlat(reference.shape[1], int(hnsw_m), faiss.METRIC_INNER_PRODUCT)
        index.hnsw.efConstruction = max(80, int(ef_search))
        index.hnsw.efSearch = max(k + 1, int(ef_search))
        # Canonical insertion order makes ID-tie behavior independent of source row ordering.
        insertion = np.argsort(tie_rank)
        for start in range(0, len(reference), reference_block):
            index.add(unit_rows(reference[insertion[start:start + reference_block]]))
        for start in range(0, len(queries), query_block):
            score, neighbors = index.search(unit_rows(queries[start:start + query_block]), min(len(reference), k + 1))
            for offset in range(len(neighbors)):
                valid = neighbors[offset] >= 0
                ids, scores = insertion[neighbors[offset][valid]], score[offset][valid]
                if self_indices is not None:
                    keep = ids != self_indices[start + offset]
                    ids, scores = ids[keep], scores[keep]
                order = np.lexsort((tie_rank[ids], -scores))[:k]
                if len(order) != k:
                    raise RuntimeError("ANN search returned too few valid neighbors")
                result_indices[start + offset] = ids[order]
                result_scores[start + offset] = np.clip(scores[order], -1, 1)
    else:
        raise ValueError("Neighbor backend must be exact or faiss")
    return result_indices, result_scores


def audit_neighbor_recall(reference, indices, *, k=None, max_queries=128, seed=42):
    """Exact blocked audit of a fixed small query sample, even for large references."""
    k = indices.shape[1] if k is None else int(k)
    rng = np.random.default_rng(seed)
    selected = np.sort(rng.choice(len(reference), min(len(reference), max_queries), replace=False))
    exact, _ = cosine_neighbors(reference, reference[selected], k=k,
                                exact_limit=len(reference), self_indices=selected)
    recalls = [len(set(a) & set(b[:k])) / k for a, b in zip(exact, indices[selected])]
    return {"recall_at_k": float(np.mean(recalls)), "k": k, "query_count": len(selected), "seed": seed}


def nested_label_mask(labels, sample_ids, fraction, seed=42):
    labels, sample_ids = np.asarray(labels), np.asarray(sample_ids, dtype=str)
    if not 0 < fraction <= 1 or len(labels) != len(sample_ids) or set(np.unique(labels)) != {0, 1}:
        raise ValueError("A fraction in (0,1] and aligned binary source labels required")
    if len(np.unique(sample_ids)) != len(sample_ids):
        raise ValueError("Duplicate sample IDs in label-mask population")
    mask = np.zeros(len(labels), dtype=bool)
    for c in (0, 1):
        members = np.flatnonzero(labels == c)
        ordered = sorted(members, key=lambda i: hashlib.sha256(f"{seed}:{sample_ids[i]}".encode()).digest())
        count = max(1, math.ceil(fraction * len(members)))
        mask[ordered[:count]] = True
    return mask


def noisy_neighbors(neighbors, fraction, *, seed=42, oracle_labels=None):
    """Random rewiring, or an explicitly oracle-only cross-class stress fixture."""
    neighbors = np.asarray(neighbors, dtype=np.int64)
    if neighbors.ndim != 2 or not 0 <= fraction <= 1:
        raise ValueError("Invalid neighbor matrix or noise fraction")
    n, k = neighbors.shape
    result = neighbors.copy()
    rng = np.random.default_rng(seed)
    labels = None if oracle_labels is None else np.asarray(oracle_labels)
    if labels is not None and (len(labels) != n or not np.isin(labels, [0, 1]).all()):
        raise ValueError("Oracle stress labels must be aligned and binary")
    count = int(round(fraction * k))
    if count == 0:
        return result
    if labels is not None and n > 10000:
        raise ValueError("Oracle edge injection is restricted to explicit small stress fixtures")
    for row in range(n):
        slots = rng.choice(k, count, replace=False)
        blocked = {row, *result[row, np.setdiff1d(np.arange(k), slots)]}
        if labels is not None or n <= 2 * (k + 1):
            pool = np.array([i for i in range(n) if i not in blocked and (labels is None or labels[i] != labels[row])])
            if len(pool) < count:
                raise ValueError("Insufficient unique candidates for requested oracle rewiring")
            replacement = rng.choice(pool, count, replace=False)
        else:
            replacement = []
            while len(replacement) < count:
                candidate = int(rng.integers(n))
                if candidate not in blocked:
                    replacement.append(candidate)
                    blocked.add(candidate)
        result[row, slots] = replacement
    return result


def transition_matrix(neighbors, similarities=None, *, policy="directed", self_loop=True):
    from scipy import sparse

    indices = np.asarray(neighbors, dtype=np.int64)
    n, k = indices.shape
    if (indices < 0).any() or (indices >= n).any() or k < 1:
        raise ValueError("Invalid graph neighbor indices")
    values = np.ones_like(indices, dtype=np.float32) if similarities is None else (np.asarray(similarities, dtype=np.float32) + 1) / 2
    if values.shape != indices.shape or not np.isfinite(values).all() or (values < 0).any():
        raise ValueError("Graph weights must be finite and nonnegative")
    matrix = sparse.csr_matrix((values.ravel(), (np.repeat(np.arange(n), k), indices.ravel())), shape=(n, n))
    if policy == "union":
        matrix = matrix.maximum(matrix.T)
    elif policy == "mutual":
        matrix = matrix.minimum(matrix.T)
    elif policy != "directed":
        raise ValueError("Graph policy must be directed, union, or mutual")
    if self_loop:
        matrix.setdiag(1)
    matrix.eliminate_zeros()
    totals = np.asarray(matrix.sum(1)).ravel()
    return (sparse.diags(1 / np.maximum(totals, 1e-12)) @ matrix).tocsr()


def label_propagation(transition, observed_labels, *, alpha=0.8, iterations=50):
    y = np.asarray(observed_labels, dtype=np.int64)
    if not 0 <= alpha < 1 or iterations < 1 or not np.isin(y, [-1, 0, 1]).all():
        raise ValueError("Invalid propagation parameters or masked labels")
    mask = y >= 0
    if transition.shape != (len(y), len(y)) or set(y[mask]) != {0, 1}:
        raise ValueError("Propagation requires an aligned graph and both observed classes")
    seeds = np.zeros((len(y), 2), dtype=np.float32)
    seeds[np.flatnonzero(mask), y[mask]] = 1
    values = seeds.copy()
    for _ in range(iterations):
        values = alpha * (transition @ values) + (1 - alpha) * seeds
    row_sum = values.sum(1, keepdims=True)
    prior = np.bincount(y[mask], minlength=2) / mask.sum()
    return np.where(row_sum > 1e-12, values / np.maximum(row_sum, 1e-12), prior)


def correct_and_smooth(transition, base_probabilities, observed_labels, *, alpha=0.8, iterations=50, return_details=False):
    """Fixed-scale C&S: masked residual correction, then masked-label smoothing."""
    base, y = np.asarray(base_probabilities, dtype=np.float32), np.asarray(observed_labels)
    if base.shape != (len(y), 2) or not np.isfinite(base).all() or (base < 0).any() or (base > 1).any():
        raise ValueError("Finite two-class base probabilities required")
    if not np.allclose(base.sum(1), 1, atol=1e-5) or not 0 <= alpha < 1 or iterations < 1:
        raise ValueError("Invalid probabilities or propagation parameters")
    if not np.isin(y, [-1, 0, 1]).all() or set(y[y >= 0]) != {0, 1}:
        raise ValueError("Both selected label classes required; all others must be -1")
    mask = y >= 0
    target = np.zeros_like(base)
    target[np.flatnonzero(mask), y[mask]] = 1
    residual = np.zeros_like(base)
    residual[mask] = target[mask] - base[mask]
    correction = residual.copy()
    for _ in range(iterations):
        correction = alpha * (transition @ correction) + (1 - alpha) * residual
    corrected = np.clip(base + correction, 0, 1)
    corrected /= np.maximum(corrected.sum(1, keepdims=True), 1e-12)
    corrected[mask] = target[mask]
    values = corrected.copy()
    for _ in range(iterations):
        values = alpha * (transition @ values) + (1 - alpha) * corrected
        values[mask] = target[mask]
    probabilities = values / np.maximum(values.sum(1, keepdims=True), 1e-12)
    return {"probabilities": probabilities, "correction": correction} if return_details else probabilities


class NativeGraphLayer(nn.Module):
    """Sparse edge-scatter CPU/device fallback; no optional extension kernels."""
    def __init__(self, input_dim, output_dim, kind):
        super().__init__()
        self.kind = kind
        self.linear = nn.Linear(input_dim * (2 if kind == "sage" else 1), output_dim, bias=kind == "sage")
        if kind != "sage":
            self.bias = nn.Parameter(torch.zeros(output_dim))
        if kind == "gat":
            self.attention = nn.Parameter(torch.empty(2, output_dim))
            nn.init.xavier_uniform_(self.attention)

    def forward(self, x, edges):
        source, target = edges
        if self.kind == "sage":
            aggregated = torch.zeros_like(x).index_add_(0, target, x[source])
            degree = torch.bincount(target, minlength=len(x)).clamp_min(1).to(x.dtype)
            return self.linear(torch.cat((x, aggregated / degree[:, None]), dim=1))
        loops = torch.arange(len(x), device=x.device)
        source, target = torch.cat((source, loops)), torch.cat((target, loops))
        h = self.linear(x)
        if self.kind == "gcn":
            degree = torch.bincount(target, minlength=len(x)).clamp_min(1).to(x.dtype)
            weight = (degree[source] * degree[target]).rsqrt()
        elif self.kind == "gat":
            score = F.leaky_relu((h[source] * self.attention[0]).sum(1) + (h[target] * self.attention[1]).sum(1), 0.2)
            maxima = score.new_full((len(x),), -torch.inf).scatter_reduce_(0, target, score, reduce="amax", include_self=True)
            exp = (score - maxima[target]).exp()
            total = score.new_zeros(len(x)).index_add_(0, target, exp)
            weight = exp / total[target].clamp_min(1e-12)
        else:
            raise ValueError("Native graph kind must be gcn, sage, or gat")
        return torch.zeros_like(h).index_add_(0, target, h[source] * weight[:, None]) + self.bias


class GraphClassifier(nn.Module):
    def __init__(self, input_dim, *, kind="sage", hidden_dim=32, projection_dim=128,
                 backend="pyg", dropout=0.1):
        super().__init__()
        if kind not in {"gcn", "gat", "sage"} or min(input_dim, hidden_dim, projection_dim) < 1:
            raise ValueError("Invalid graph architecture or dimensions")
        self.projector = nn.Linear(input_dim, projection_dim)
        self.dropout = float(dropout)
        if not 0 <= self.dropout < 1:
            raise ValueError("Dropout must be in [0,1)")
        if backend == "pyg":
            try:
                from torch_geometric.nn import GATConv, GCNConv, SAGEConv
            except ImportError as error:
                raise ImportError("Graph backend=pyg requires torch-geometric; use backend=native explicitly for fallback") from error
            constructors = {"gcn": lambda a, b: GCNConv(a, b, cached=False),
                            "gat": lambda a, b: GATConv(a, b, heads=1, concat=False),
                            "sage": SAGEConv}
            build = constructors[kind]
        elif backend == "native":
            build = lambda a, b: NativeGraphLayer(a, b, kind)
        else:
            raise ValueError("Graph model backend must be pyg or native")
        self.first, self.second = build(projection_dim, hidden_dim), build(hidden_dim, 2)

    def embed(self, x):
        return F.normalize(self.projector(x).float(), dim=1, eps=1e-12)

    def forward(self, x, edge_index):
        h = self.embed(x)
        h = F.relu(self.first(h, edge_index))
        return self.second(F.dropout(h, p=self.dropout, training=self.training), edge_index)


def fit_graph(config):
    from .graph_training import fit_graph_run
    return fit_graph_run(config)


def predict_graph(run_dir, query_features, protocol="inductive"):
    from .graph_training import predict_graph_run
    return predict_graph_run(run_dir, query_features, protocol=protocol)
