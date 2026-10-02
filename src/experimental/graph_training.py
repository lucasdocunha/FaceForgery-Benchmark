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
    FeatureDataset, classifier, fit_standardizer, open_training_caches,
    predict_network, read_model_artifact, save_validation, seed_cpu,
    standardized_block, train_model, write_model_artifact, write_transformed_features,
)
from .graphs import (
    GraphClassifier, audit_neighbor_recall, correct_and_smooth, cosine_neighbors,
    label_propagation, nested_label_mask, noisy_neighbors, transition_matrix,
)
from .metric import unit_rows


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
    weight = np.maximum((np.asarray(scores) + 1) / 2, 1e-6)
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
    alpha = float(graph.get("alpha", .8))
    correction = alpha * (source_cs["correction"][query_neighbors] * weight[:, :, None]).sum(1)
    corrected = np.clip(base + correction, 0, 1)
    corrected /= np.maximum(corrected.sum(1, keepdims=True), 1e-12)
    smooth_neighbors = (source_cs["probabilities"][query_neighbors] * weight[:, :, None]).sum(1)
    cs = (1 - alpha) * corrected + alpha * smooth_neighbors
    labeled = np.flatnonzero(observed >= 0)
    nearest, _ = cosine_neighbors(reference[labeled], query, k=min(int(graph.get("k", 10)), len(labeled)), **_neighbor_options(graph))
    knn = observed[labeled][nearest].mean(1)
    return {"linear": base[:, 1], "knn": knn, "lp": lp[:, 1], "correct_smooth": cs[:, 1]}


def fit_graph_run(config):
    training, model_config = config.get("training", {}), config.get("model", {})
    graph = model_config.get("graph", {})
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
    root = Path(config["output_dir"])
    if root.exists() and not training.get("resume", False):
        raise FileExistsError("Use a fresh graph experiment directory")
    root.mkdir(parents=True, exist_ok=True)
    train, val = open_training_caches(config)
    mean, std = fit_standardizer(train.features)
    np.savez(root / "normalizer.npz", mean=mean, std=std)
    reference = write_transformed_features(root / "reference.npy", train.features, mean, std)
    query = write_transformed_features(root / "validation_features.npy", val.features, mean, std)
    all_labels = train.frame.label.to_numpy(dtype=np.int64)
    mask = nested_label_mask(all_labels, train.frame.sample_id, float(model_config.get("label_fraction", .05)), seed)
    observed = np.where(mask, all_labels, -1)
    np.save(root / "observed_labels.npy", observed)
    np.save(root / "label_mask.npy", mask)
    write_json(root / "label_mask.json", {
        "seed": seed, "fraction": model_config.get("label_fraction", .05),
        "selected_ids": train.frame.loc[mask, "sample_id"].tolist(),
        "selected_class_counts": {str(c): int((observed == c).sum()) for c in (0, 1)},
        "selected_total": int(mask.sum()), "population": len(mask),
        "interpretation": "low-label downstream adaptation; checkpoint backbone may already use all MFFI labels",
    })
    del all_labels
    coef, intercept = _linear_fit(reference, observed, seed, int(training.get("baseline_epochs", 30)))
    np.savez(root / "linear.npz", coef=coef, intercept=intercept)
    neighbors, scores = _graph_arrays(reference, graph, ids=train.frame.sample_id, seed=seed)
    base_lp, base_cs, _ = _source_baselines(reference, observed, neighbors, scores, coef, intercept, graph)
    baseline_scores = _baseline_predictions(reference, query, observed, base_lp, base_cs, coef, intercept, graph)
    val_labels = val.frame.label.to_numpy(dtype=np.int64)
    write_json(root / "baseline_metrics.json", {
        "scope": "source-validation development", "protocol": "inductive", "same_label_mask": True,
        "linear_solver": "SGD log-loss with L2 and averaged iterates",
        "methods": {name: summary(val_labels, p) for name, p in baseline_scores.items()},
    })
    device = training.get("device", "cpu")
    files = ["normalizer.npz", "reference.npy", "observed_labels.npy", "label_mask.npy", "label_mask.json", "linear.npz"]
    fanouts = tuple(model_config.get("fanouts", [10, 5]))
    if neural_graph:
        model = GraphClassifier(
            reference.shape[1], kind=kind, hidden_dim=int(model_config.get("hidden_dim", 32)),
            projection_dim=int(model_config.get("projection_dim", 128)),
            backend=model_config.get("backend", "pyg"), dropout=float(model_config.get("dropout", .1)),
        ).to(device)
        loader = DataLoader(torch.from_numpy(np.flatnonzero(mask)), batch_size=int(training.get("batch_size", 32)), shuffle=True, num_workers=0)
        state = {"epoch": None, "neighbors": neighbors, "scores": scores, "space": reference, "best": -float("inf")}
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
            qn, _ = cosine_neighbors(space, validation_space, k=int(graph.get("k", 10)), **_neighbor_options(graph))
            p = _predict_gnn(current, reference, query, selection_neighbors, qn,
                             fanouts=fanouts, batch_size=int(training.get("inference_batch_size", 32)), device=device)
            metrics = summary(val_labels, p)
            if metrics["auc"] > state["best"]:
                state["best"] = metrics["auc"]
                np.save(root / "neighbors.npy", selection_neighbors)
                np.save(root / "similarities.npy", selection_scores)
            return metrics

        model = train_model(model, loader, loss_step, validate, root, config)
        if dynamic:
            space = _project_space(model, reference, root / "graph_space.npy", device)
            files.append("graph_space.npy")
        else:
            space = reference
        neighbors = np.load(root / "neighbors.npy", mmap_mode="r")
        scores = np.load(root / "similarities.npy", mmap_mode="r")
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
        recall = audit_neighbor_recall(space, neighbors, seed=seed, max_queries=int(graph.get("audit_queries", 128)))
        write_json(root / "ann_audit.json", recall)
        if recall["recall_at_k"] < float(graph.get("minimum_recall", .95)):
            raise RuntimeError("ANN recall below predeclared acceptance threshold")
    write_model_artifact(root, {
        "task": "graph", "kind": kind, "config": config, "input_dim": reference.shape[1],
        "train_cache": train.identity, "val_cache": val.identity,
        "inference_protocol": "inductive", "dynamic": dynamic,
        "inference_graph_policy": "rebuild from selected projector" if dynamic else "frozen input-feature graph",
        "label_budget_scope": "selected downstream training labels only; supervised backbone exposure is separate",
    }, files)
    p = predict_graph_run(root, val.features, protocol="inductive")
    save_validation(root, val, p, extra={"method": kind, "label_count": int(mask.sum()), "protocol": "inductive"})
    for name in ("validation_features.npy", "training_space.npy", "selection_space.npy", "selection_query.npy"):
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


def predict_graph_run(run_dir, query_features, protocol="inductive"):
    root = Path(run_dir)
    record = read_model_artifact(root)
    if record["task"] != "graph" or protocol not in {"inductive", "transductive"}:
        raise ValueError("Graph artifact and explicit inductive/transductive protocol required")
    cfg, kind = record["config"]["model"], record["kind"]
    graph = cfg.get("graph", {})
    normalizer = np.load(root / "normalizer.npz")
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
    if protocol == "transductive":
        with tempfile.TemporaryDirectory(prefix="faceforgery-graph-") as temporary:
            temporary = Path(temporary)
            joint = np.lib.format.open_memmap(temporary / "joint.npy", mode="w+", dtype="float32",
                                              shape=(len(reference) + len(query), reference.shape[1]))
            for start in range(0, len(reference), 1024):
                joint[start:start + 1024] = reference[start:start + 1024]
            joint[len(reference):] = query
            joint.flush()
            space = _project_space(model, joint, temporary / "space.npy") if record["dynamic"] else joint
            neighbors, scores = _graph_arrays(space, graph, seed=int(record["config"].get("seed", 42)))
            targets = np.arange(len(reference), len(joint))
            if neural:
                return _predict_gnn(model, joint, query, neighbors, neighbors[targets], source_roots=targets,
                                    fanouts=tuple(cfg.get("fanouts", [10, 5])))
            labels = np.concatenate((observed, np.full(len(query), -1, dtype=np.int64)))
            lp, cs, base = _source_baselines(joint, labels, neighbors, scores, linear["coef"], linear["intercept"], graph)
            if kind == "lp":
                return lp[targets, 1]
            if kind == "correct_smooth":
                return cs["probabilities"][targets, 1]
            # k-NN always uses the same labeled source subset, even in this protocol.
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
