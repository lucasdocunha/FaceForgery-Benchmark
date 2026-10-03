"""Low-label graph fitting with source-only labels and bounded ego-graph batches."""

from __future__ import annotations

import json
import tempfile
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader, Subset

from src.robustness.provenance import digest_file, write_json
from src.robustness.statistics import summary
from .feature_training import (
    FeatureDataset, classifier, fit_standardizer, label_budget, measured_feature_run, open_training_caches,
    predict_network, read_model_artifact, save_validation, seed_cpu,
    representation_key, standardized_block, train_model, write_model_artifact, write_transformed_features,
)
from .graphs import (
    GraphClassifier, audit_neighbor_recall, correct_and_smooth, cosine_neighbors,
    graph_memory_estimate, label_propagation, noisy_neighbors, transition_matrix,
)
from .metric import ProjectionHead, unit_rows


def _neighbor_options(graph):
    allowed = ("backend", "exact_limit", "query_block", "reference_block", "memory_budget_mb", "hnsw_m", "ef_search")
    return {key: graph[key] for key in allowed if key in graph}


def _graph_arrays(features, graph, *, ids=None, seed=42):
    neighbors, scores = cosine_neighbors(features, k=int(graph.get("k", 10)), reference_ids=ids, **_neighbor_options(graph))
    if graph.get("noise_fraction", 0):
        neighbors = noisy_neighbors(neighbors, graph["noise_fraction"], seed=seed)
        for start in range(0, len(features), 128):
            x = unit_rows(features[start:start + 128])
            for offset, row in enumerate(neighbors[start:start + len(x)]):
                scores[start + offset] = unit_rows(features[row]) @ x[offset]
    return neighbors, scores


def disjoint_ego_batch(reference, root_features, root_neighbors, source_neighbors,
                       *, source_roots=None, fanouts=(10, 5), rng=None):
    """Each query has its own source-only two-hop graph, so queries cannot interact."""
    if len(fanouts) != 2 or min(fanouts) < 1:
        raise ValueError("Exactly two positive fanouts required for a two-layer model")
    features, edges, roots = [], [], []
    offset = 0
    for row, query in enumerate(root_features):
        mapping = {} if source_roots is None else {int(source_roots[row]): 0}
        local = [np.asarray(query, dtype=np.float32)]
        local_edges = set()

        def node(source):
            source = int(source)
            if source not in mapping:
                mapping[source] = len(local)
                local.append(np.asarray(reference[source], dtype=np.float32))
            return mapping[source]

        def select(values, fanout):
            count = min(len(values), fanout)
            return values[:count] if rng is None else rng.choice(values, count, replace=False)

        first = select(root_neighbors[row], fanouts[0])
        for source in first:
            target_node = node(source)
            local_edges.add((target_node, 0))
            for parent in select(source_neighbors[source], fanouts[1]):
                local_edges.add((node(parent), target_node))
        roots.append(offset)
        features.extend(local)
        edges.extend((a + offset, b + offset) for a, b in sorted(local_edges) if a != b)
        offset += len(local)
    if not features:
        raise ValueError("Empty ego-graph batch")
    return (
        torch.from_numpy(np.stack(features)),
        torch.tensor(edges, dtype=torch.long).T.contiguous().reshape(2, -1),
        torch.tensor(roots, dtype=torch.long),
    )


def _predict_gnn(model, reference, query, source_neighbors, query_neighbors,
                 *, fanouts=(10, 5), batch_size=32, device="cpu", source_roots=None):
    model.eval()
    result = []
    with torch.inference_mode():
        for start in range(0, len(query), batch_size):
            x, edge_index, roots = disjoint_ego_batch(
                reference, query[start:start + batch_size], query_neighbors[start:start + batch_size],
                source_neighbors, fanouts=fanouts,
                source_roots=None if source_roots is None else source_roots[start:start + batch_size],
            )
            logits = model(x.to(device), edge_index.to(device))[roots.to(device)]
            result.append(logits.float().softmax(1)[:, 1].cpu().numpy())
    return np.concatenate(result)


def _project_space(model, features, path, device="cpu"):
    model.eval()
    result = np.lib.format.open_memmap(path, mode="w+", dtype="float32", shape=(len(features), model.projector.out_features))
    with torch.inference_mode():
        for start in range(0, len(features), 512):
            x = torch.from_numpy(np.array(features[start:start + 512], dtype=np.float32)).to(device)
            result[start:start + len(x)] = model.embed(x).float().cpu().numpy()
    result.flush()
    return np.load(path, mmap_mode="r")


def _linear_fit(features, observed, seed, epochs=30):
    from sklearn.linear_model import SGDClassifier

    selected = np.flatnonzero(observed >= 0)
    if set(observed[selected]) != {0, 1}:
        raise ValueError("Linear baseline needs both selected training classes")
    estimator = SGDClassifier(loss="log_loss", penalty="l2", alpha=1e-4, random_state=seed, average=True)
    rng = np.random.default_rng(seed)
    for epoch in range(epochs):
        order = rng.permutation(selected)
        for start in range(0, len(order), 256):
            batch = order[start:start + 256]
            estimator.partial_fit(np.asarray(features[batch]), observed[batch], classes=np.array([0, 1]))
    return estimator.coef_.astype(np.float32), estimator.intercept_.astype(np.float32)


def _linear_probabilities(features, coef, intercept):
    from scipy.special import expit

    p = np.empty(len(features), dtype=np.float32)
    for start in range(0, len(features), 1024):
        p[start:start + 1024] = expit((np.asarray(features[start:start + 1024]) @ coef.T + intercept).ravel())
    return np.column_stack((1 - p, p))


def _query_weights(scores):
    weight = np.maximum((np.asarray(scores, dtype=np.float64) + 1) / 2, 1e-6)
    return weight / weight.sum(1, keepdims=True)


def _source_baselines(reference, observed, neighbors, scores, coef, intercept, graph):
    transition = transition_matrix(neighbors, scores, policy=graph.get("policy", "directed"))
    parameters = {"alpha": float(graph.get("alpha", .8)), "iterations": int(graph.get("iterations", 50))}
    base = _linear_probabilities(reference, coef, intercept)
    lp = label_propagation(transition, observed, **parameters)
    cs = correct_and_smooth(transition, base, observed, return_details=True, **parameters)
    return lp, cs, base


def _baseline_predictions(reference, query, observed, source_lp, source_cs,
                          coef, intercept, graph, *, query_neighbors=None, query_scores=None):
    if query_neighbors is None:
        query_neighbors, query_scores = cosine_neighbors(reference, query, k=int(graph.get("k", 10)), **_neighbor_options(graph))
    weight = _query_weights(query_scores)
    base = _linear_probabilities(query, coef, intercept)
    lp = (source_lp[query_neighbors] * weight[:, :, None]).sum(1)
    lp /= np.maximum(lp.sum(1, keepdims=True), 1e-12)
    alpha = float(graph.get("alpha", .8))
    correction = alpha * (source_cs["correction"][query_neighbors] * weight[:, :, None]).sum(1)
    corrected = np.clip(base + correction, 0, 1)
    corrected /= np.maximum(corrected.sum(1, keepdims=True), 1e-12)
    smooth_neighbors = (source_cs["probabilities"][query_neighbors] * weight[:, :, None]).sum(1)
    cs = np.clip((1 - alpha) * corrected + alpha * smooth_neighbors, 0, 1)
    cs /= np.maximum(cs.sum(1, keepdims=True), 1e-12)
    labeled = np.flatnonzero(observed >= 0)
    nearest, _ = cosine_neighbors(reference[labeled], query, k=min(int(graph.get("k", 10)), len(labeled)), **_neighbor_options(graph))
    knn = observed[labeled][nearest].mean(1)
    return {"linear": base[:, 1], "knn": knn, "lp": lp[:, 1], "correct_smooth": cs[:, 1]}


def _transductive_predictions(reference, query, observed, coef, intercept, graph,
                              *, model=None, dynamic=False, fanouts=(10, 5), device="cpu"):
    """One source plus one target population; target labels never enter this API."""
    with tempfile.TemporaryDirectory(prefix="faceforgery-graph-") as temporary:
        temporary = Path(temporary)
        joint = np.lib.format.open_memmap(temporary / "joint.npy", mode="w+", dtype="float32",
                                          shape=(len(reference) + len(query), reference.shape[1]))
        for start in range(0, len(reference), 1024):
            block = reference[start:start + 1024]
            joint[start:start + len(block)] = block
        joint[len(reference):] = query
        joint.flush()
        space = _project_space(model, joint, temporary / "space.npy", device) if dynamic else joint
        neighbors, scores = _graph_arrays(space, graph)
        targets = np.arange(len(reference), len(joint))
        if model is not None:
            return _predict_gnn(model, joint, query, neighbors, neighbors[targets], source_roots=targets,
                                fanouts=fanouts, device=device)
        labels = np.concatenate((observed, np.full(len(query), -1, dtype=np.int64)))
        lp, cs, _ = _source_baselines(joint, labels, neighbors, scores, coef, intercept, graph)
        return {"lp": lp[targets, 1], "correct_smooth": cs["probabilities"][targets, 1]}


def graph_diagnostics(neighbors, scores, observed, graph):
    from scipy.sparse.csgraph import connected_components

    transition = transition_matrix(neighbors, scores, policy=graph.get("policy", "directed"))
    degree = np.diff(transition.indptr)
    labeled_edges = (observed[:, None] >= 0) & (observed[neighbors] >= 0)
    return {
        "nodes": len(neighbors), "directed_neighbor_edges": int(neighbors.size),
        "transition_nnz": int(transition.nnz), "degree_min": int(degree.min()),
        "degree_max": int(degree.max()), "degree_mean": float(degree.mean()),
        "weak_components": int(connected_components(transition, directed=True, connection="weak", return_labels=False)),
        "mean_neighbor_cosine": float(scores.mean()),
        "observed_label_edges": int(labeled_edges.sum()),
        "observed_label_homophily": float((observed[:, None] == observed[neighbors])[labeled_edges].mean()) if labeled_edges.any() else None,
        "hidden_labels_used": False,
    }


def _metric_transform(root, spec, features, output=None):
    cfg = spec["model"]
    model = ProjectionHead(spec["input_dim"], cfg.get("embedding_dim", 128), cfg.get("hidden_dim", 256))
    model.load_state_dict(torch.load(Path(root) / "metric_projection.pt", map_location="cpu", weights_only=True))
    normalizer = np.load(Path(root) / "metric_normalizer.npz")
    if output is not None:
        return write_transformed_features(output, features, normalizer["mean"], normalizer["std"], model=model)
    return predict_network(model, features, normalizer["mean"], normalizer["std"])


def _prepare_metric_projection(config, train, val, root):
    source = config.get("projection_run") or config.get("data", {}).get("projection_run")
    if not source:
        return train.features, val.features, None
    source = Path(source)
    metric = read_model_artifact(source)
    if metric["task"] != "metric" or metric["kind"] != "supcon":
        raise ValueError("Graph metric projection requires a fitted SupCon artifact")
    if metric["train_cache"] != train.identity or metric["representation_key"] != representation_key(train.metadata):
        raise ValueError("Metric projection was trained on a different source cache")
    mask = json.loads((root / "label_mask.json").read_text())
    prior = json.loads((source / "label_mask.json").read_text())
    if mask["selected_labels"] != prior["selected_labels"]:
        raise ValueError("Supervised projection must obey the exact same graph label mask")
    state = torch.load(source / "best.pt", map_location="cpu", weights_only=True)["state_dict"]
    torch.save(state, root / "metric_projection.pt")
    standardizer = np.load(source / "normalizer.npz")
    np.savez(root / "metric_normalizer.npz", mean=standardizer["mean"], std=standardizer["std"])
    spec = {"source_artifact_sha256": digest_file(source / "artifact.json"),
            "input_dim": metric["input_dim"], "model": metric["config"]["model"],
            "label_mask_sha256": metric["label_mask_sha256"]}
    return (_metric_transform(root, spec, train.features, root / "training_metric.npy"),
            _metric_transform(root, spec, val.features, root / "validation_metric.npy"), spec)


@measured_feature_run
def fit_graph_run(config):
    training, model_config = config.get("training", {}), config.get("model", {})
    graph = model_config.get("graph", {})
    protocol = model_config.get("protocol", "inductive")
    if protocol not in {"inductive", "transductive"}:
        raise ValueError("Explicit inductive or transductive inference protocol required")
    seed = int(config.get("seed", 42))
    seed_cpu(seed, training.get("cpu_threads", 2))
    kind = model_config.get("kind", "sage")
    neural_graph = kind in {"gcn", "gat", "sage"}
    if not neural_graph and kind not in {"linear", "mlp", "knn", "lp", "correct_smooth"}:
        raise ValueError("Unknown graph or baseline model kind")
    if neural_graph and graph.get("policy", "directed") != "directed":
        raise ValueError("Sampled GNNs currently require explicit directed graph policy")
    dynamic = bool(graph.get("dynamic", False))
    if dynamic and not neural_graph:
        raise ValueError("Dynamic graphs require a learned GNN projector")
    interval = int(graph.get("rebuild_interval", 5))
    if interval < 1:
        raise ValueError("Graph rebuild interval must be positive")
    root = Path(config.get("run_dir", config.get("output_dir")))
    if root.exists() and not training.get("resume", False):
        raise FileExistsError("Use a fresh graph experiment directory")
    root.mkdir(parents=True, exist_ok=True)
    train, val = open_training_caches(config)
    mask, observed = label_budget(train, {**config, "family": "graph"}, root)
    train_features, val_features, metric_projection = _prepare_metric_projection(config, train, val, root)
    mean, std = fit_standardizer(train_features) if model_config.get("standardize", True) else (
        np.zeros(train_features.shape[1], dtype=np.float32), np.ones(train_features.shape[1], dtype=np.float32))
    np.savez(root / "normalizer.npz", mean=mean, std=std)
    reference = write_transformed_features(root / "reference.npy", train_features, mean, std)
    query = write_transformed_features(root / "validation_features.npy", val_features, mean, std)
    coef, intercept = _linear_fit(reference, observed, seed, int(training.get("baseline_epochs", 30)))
    np.savez(root / "linear.npz", coef=coef, intercept=intercept)
    neighbors, scores = _graph_arrays(reference, graph, ids=train.frame.sample_id, seed=seed)
    static_query_neighbors, static_query_scores = cosine_neighbors(
        reference, query, k=int(graph.get("k", 10)), **_neighbor_options(graph))
    base_lp, base_cs, _ = _source_baselines(reference, observed, neighbors, scores, coef, intercept, graph)
    baseline_scores = _baseline_predictions(reference, query, observed, base_lp, base_cs, coef, intercept, graph,
                                             query_neighbors=static_query_neighbors, query_scores=static_query_scores)
    if protocol == "transductive":
        baseline_scores.update(_transductive_predictions(reference, query, observed, coef, intercept, graph))
    val_labels = val.frame.label.to_numpy(dtype=np.int64)
    write_json(root / "baseline_metrics.json", {
        "scope": "source-validation development", "protocol": protocol, "same_label_mask": True,
        "linear_solver": "SGD log-loss with L2 and averaged iterates",
        "methods": {name: summary(val_labels, p) for name, p in baseline_scores.items()},
    })
    device = training.get("device", "cpu")
    files = ["normalizer.npz", "reference.npy", "observed_labels.npy", "label_mask.npy", "label_mask.json", "linear.npz"]
    if metric_projection:
        files.extend(["metric_projection.pt", "metric_normalizer.npz"])
    fanouts = tuple(model_config.get("fanouts", [10, 5]))
    write_json(root / "memory_estimate.json", graph_memory_estimate(
        len(reference), reference.shape[1], int(graph.get("k", 10)),
        hnsw_m=int(graph.get("hnsw_m", 16)), batch_size=int(training.get("batch_size", 32)), fanouts=fanouts))
    if neural_graph:
        model = GraphClassifier(
            reference.shape[1], kind=kind, hidden_dim=int(model_config.get("hidden_dim", 32)),
            projection_dim=int(model_config.get("projection_dim", 128)),
            backend=model_config.get("backend", "pyg"), dropout=float(model_config.get("dropout", .1)),
        ).to(device)
        loader = DataLoader(torch.from_numpy(np.flatnonzero(mask)), batch_size=int(training.get("batch_size", 32)), shuffle=True, num_workers=0)
        state = {"epoch": None, "neighbors": neighbors, "scores": scores, "space": reference}
        if dynamic and training.get("resume", False) and interval != 1:
            raise ValueError("Resume dynamic graphs with rebuild_interval=1; wider intervals require saved graph-state replay")

        def loss_step(current, batch, step):
            epoch = step["epoch"]
            if state["epoch"] != epoch:
                if dynamic and (state["epoch"] is None or epoch % interval == 0):
                    space = _project_space(current, reference, root / "training_space.npy", device)
                    state["neighbors"], state["scores"] = _graph_arrays(space, graph, ids=train.frame.sample_id, seed=seed + epoch)
                    state["space"] = space
                state["epoch"] = epoch
            current.train()
            selected = batch.numpy()
            x, edges, roots = disjoint_ego_batch(
                reference, reference[selected], state["neighbors"][selected], state["neighbors"],
                source_roots=selected, fanouts=fanouts, rng=np.random.default_rng(seed + step["global_step"]),
            )
            logits = current(x.to(device), edges.to(device))[roots.to(device)]
            labels = torch.from_numpy(observed[selected]).to(device)
            loss = torch.nn.functional.cross_entropy(logits, labels)
            return loss, {"classification": float(loss.detach())}, len(selected)

        def validate(current):
            current.eval()
            if dynamic:
                space = _project_space(current, reference, root / "selection_space.npy", device)
                validation_space = _project_space(current, query, root / "selection_query.npy", device)
                selection_neighbors, selection_scores = _graph_arrays(space, graph, ids=train.frame.sample_id, seed=seed)
            else:
                space, validation_space = reference, query
                selection_neighbors, selection_scores = neighbors, scores
            qn = (cosine_neighbors(space, validation_space, k=int(graph.get("k", 10)), **_neighbor_options(graph))[0]
                  if dynamic else static_query_neighbors)
            p = _predict_gnn(current, reference, query, selection_neighbors, qn,
                             fanouts=fanouts, batch_size=int(training.get("inference_batch_size", 32)), device=device)
            metrics = summary(val_labels, p)
            if protocol == "transductive":
                p = _transductive_predictions(reference, query, observed, coef, intercept, graph,
                                               model=current, dynamic=dynamic, fanouts=fanouts, device=device)
                metrics = summary(val_labels, p)
            return metrics

        model = train_model(model, loader, loss_step, validate, root, config)
        if dynamic:
            space = _project_space(model, reference, root / "graph_space.npy", device)
            files.append("graph_space.npy")
        else:
            space = reference
        neighbors, scores = _graph_arrays(space, graph, ids=train.frame.sample_id, seed=seed)
        np.save(root / "neighbors.npy", neighbors)
        np.save(root / "similarities.npy", scores)
        files.append("best.pt")
    elif kind == "mlp":
        model = classifier(reference.shape[1], "mlp", int(model_config.get("hidden_dim", 32))).to(device)
        dataset = FeatureDataset(reference, observed, np.zeros(reference.shape[1]), np.ones(reference.shape[1]))
        loader = DataLoader(Subset(dataset, np.flatnonzero(mask).tolist()), batch_size=int(training.get("batch_size", 32)), shuffle=True, num_workers=0)

        def loss_step(current, batch, step):
            x, y = (v.to(device) for v in batch)
            loss = torch.nn.functional.cross_entropy(current(x), y)
            return loss, {"classification": float(loss.detach())}, len(y)

        def validate(current):
            logits = predict_network(current, query, np.zeros(reference.shape[1]), np.ones(reference.shape[1]), device=device)
            return summary(val_labels, torch.from_numpy(logits).softmax(1)[:, 1].numpy())

        train_model(model, loader, loss_step, validate, root, config)
        files.append("best.pt")
        space = reference
    else:
        space = reference
    if not neural_graph:
        np.save(root / "neighbors.npy", neighbors)
        np.save(root / "similarities.npy", scores)
    files.extend(["neighbors.npy", "similarities.npy"])
    source_lp, source_cs, _ = _source_baselines(reference, observed, neighbors, scores, coef, intercept, graph)
    np.savez(root / "propagation.npz", lp=source_lp, cs=source_cs["probabilities"], correction=source_cs["correction"])
    files.append("propagation.npz")
    if graph.get("backend", "exact") == "faiss":
        audited = _graph_arrays(space, {**graph, "noise_fraction": 0}, ids=train.frame.sample_id, seed=seed)[0] if graph.get("noise_fraction", 0) else neighbors
        recall = audit_neighbor_recall(space, audited, seed=seed, max_queries=int(graph.get("audit_queries", 128)))
        write_json(root / "ann_audit.json", recall)
        if recall["recall_at_k"] < float(graph.get("minimum_recall", .95)):
            raise RuntimeError("ANN recall below predeclared acceptance threshold")
    write_json(root / "graph_diagnostics.json", graph_diagnostics(neighbors, scores, observed, graph))
    write_model_artifact(root, {
        "task": "graph", "kind": kind, "config": config, "input_dim": reference.shape[1],
        "train_cache": train.identity, "val_cache": val.identity,
        "representation_key": representation_key(train.metadata),
        "train_manifest": train.metadata["manifest"], "validation_manifest": val.metadata["manifest"],
        "label_mask_sha256": digest_file(root / "label_mask.json"),
        "metric_projection": metric_projection,
        "inference_protocol": protocol, "dynamic": dynamic,
        "inference_graph_policy": "rebuild from selected projector" if dynamic else "frozen input-feature graph",
        "label_budget_scope": "selected downstream training labels only; supervised backbone exposure is separate",
        "fit_statistics": {
            "trainable_parameters": int(coef.size + intercept.size) if kind in {"linear", "correct_smooth"} else 0,
            "epochs_completed": int(training.get("baseline_epochs", 30)) if kind in {"linear", "correct_smooth"} else 0,
            "global_step": int(training.get("baseline_epochs", 30)) * int(np.ceil(mask.sum() / 256)) if kind in {"linear", "correct_smooth"} else 0,
            "description": "SGD logistic base fit; LP and kNN have no learned parameters",
        },
    }, files)
    p = predict_graph_run(root, val.features, protocol=protocol)
    save_validation(root, val, p, extra={"method": kind, "label_count": int(mask.sum()), "protocol": protocol})
    for name in ("validation_features.npy", "training_space.npy", "selection_space.npy", "selection_query.npy",
                 "training_metric.npy", "validation_metric.npy"):
        path = root / name
        if path.exists():
            path.unlink()
    return root


def _load_graph_model(record, root):
    cfg = record["config"]["model"]
    if record["kind"] == "mlp":
        model = classifier(record["input_dim"], "mlp", cfg.get("hidden_dim", 32))
    else:
        model = GraphClassifier(record["input_dim"], kind=record["kind"], hidden_dim=cfg.get("hidden_dim", 32),
                                projection_dim=cfg.get("projection_dim", 128), backend=cfg.get("backend", "pyg"),
                                dropout=cfg.get("dropout", .1))
    model.load_state_dict(torch.load(root / "best.pt", map_location="cpu", weights_only=False)["state_dict"])
    return model.eval()


def predict_graph_run(run_dir, query_features, protocol=None):
    root = Path(run_dir)
    record = read_model_artifact(root)
    protocol = record["inference_protocol"] if protocol is None else protocol
    if record["task"] != "graph" or protocol not in {"inductive", "transductive"}:
        raise ValueError("Graph artifact and explicit inductive/transductive protocol required")
    if protocol != record["inference_protocol"]:
        raise ValueError("Inference protocol differs from validation calibration; fit a separate run")
    cfg, kind = record["config"]["model"], record["kind"]
    graph = cfg.get("graph", {})
    normalizer = np.load(root / "normalizer.npz")
    if record.get("metric_projection"):
        query_features = _metric_transform(root, record["metric_projection"], query_features)
    query = unit_rows(standardized_block(query_features, normalizer["mean"], normalizer["std"]))
    reference = np.load(root / "reference.npy", mmap_mode="r")
    observed = np.load(root / "observed_labels.npy", mmap_mode="r")
    linear = np.load(root / "linear.npz")
    if kind == "linear":
        return _linear_probabilities(query, linear["coef"], linear["intercept"])[:, 1]
    if kind == "mlp":
        model = _load_graph_model(record, root)
        logits = predict_network(model, query, np.zeros(record["input_dim"]), np.ones(record["input_dim"]))
        return torch.from_numpy(logits).softmax(1)[:, 1].numpy()
    neural = kind in {"gcn", "gat", "sage"}
    model = _load_graph_model(record, root) if neural else None
    if protocol == "transductive" and (neural or kind in {"lp", "correct_smooth"}):
        result = _transductive_predictions(reference, query, observed, linear["coef"], linear["intercept"], graph,
                                           model=model, dynamic=record["dynamic"], fanouts=tuple(cfg.get("fanouts", [10, 5])))
        return result if neural else result[kind]
    neighbors = np.load(root / "neighbors.npy", mmap_mode="r")
    source_space = np.load(root / "graph_space.npy", mmap_mode="r") if record["dynamic"] else reference
    if record["dynamic"]:
        with torch.inference_mode():
            query_space = model.embed(torch.from_numpy(query)).numpy()
    else:
        query_space = query
    qn, qs = cosine_neighbors(source_space, query_space, k=int(graph.get("k", 10)), **_neighbor_options(graph))
    if neural:
        return _predict_gnn(model, reference, query, neighbors, qn, fanouts=tuple(cfg.get("fanouts", [10, 5])))
    propagation = np.load(root / "propagation.npz")
    predictions = _baseline_predictions(reference, query, observed, propagation["lp"],
                                        {"probabilities": propagation["cs"], "correction": propagation["correction"]},
                                        linear["coef"], linear["intercept"], graph, query_neighbors=qn, query_scores=qs)
    return predictions[kind]
