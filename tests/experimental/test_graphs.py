import numpy as np
import pytest
import torch

from src.experimental.graphs import (
    GraphClassifier, NativeGraphLayer, audit_neighbor_recall, correct_and_smooth, cosine_neighbors, label_propagation,
    nested_label_mask, noisy_neighbors, transition_matrix,
)


def test_blocked_cosine_matches_oracle_and_stable_ties():
    x = np.array([[1, 0], [1, 0], [0, 1], [-1, 0]], dtype=np.float32)
    indices, score = cosine_neighbors(x, k=2, reference_ids=["z", "a", "b", "c"], query_block=2, reference_block=1)
    assert indices[2].tolist() == [1, 3]
    assert all(i not in row for i, row in enumerate(indices))
    for i, row in enumerate(indices):
        np.testing.assert_allclose(score[i], x[i] @ x[row].T)
    with pytest.raises(ValueError, match="exact_limit"):
        cosine_neighbors(x, k=2, exact_limit=3)


def test_nested_masks_are_stable_under_row_permutation():
    y = np.array([0] * 40 + [1] * 160)
    ids = np.array([f"row-{i}" for i in range(len(y))])
    five = nested_label_mask(y, ids, .05)
    ten = nested_label_mask(y, ids, .10)
    assert np.all(~five | ten)
    assert five.sum() == 10 and ten.sum() == 20
    order = np.random.default_rng(7).permutation(len(y))
    permuted = nested_label_mask(y[order], ids[order], .10)
    assert set(ids[ten]) == set(ids[order][permuted])


def test_propagation_and_correction_only_receive_observed_labels():
    neighbors = np.array([[1], [0], [3], [2]])
    transition = transition_matrix(neighbors)
    masked = np.array([0, -1, 1, -1])
    p = label_propagation(transition, masked)
    assert p[1, 0] > .99 and p[3, 1] > .99
    fixed = correct_and_smooth(transition, np.full((4, 2), .5), masked)
    np.testing.assert_allclose(fixed.sum(1), 1)
    assert fixed[1, 0] > .5 and fixed[3, 1] > .5


def test_random_noise_is_deterministic_and_removes_self_edges():
    x = np.random.default_rng(42).normal(size=(20, 8))
    edges, _ = cosine_neighbors(x, k=3)
    a = noisy_neighbors(edges, .67, seed=7)
    np.testing.assert_array_equal(a, noisy_neighbors(edges, .67, seed=7))
    assert all(i not in row and len(set(row)) == 3 for i, row in enumerate(a))


@pytest.mark.parametrize("backend", ["native", "pyg"])
@pytest.mark.parametrize("kind", ["gcn", "gat", "sage"])
def test_graph_architectures_forward_backward(backend, kind):
    if backend == "pyg":
        pytest.importorskip("torch_geometric")
    model = GraphClassifier(8, kind=kind, backend=backend, hidden_dim=8, projection_dim=16)
    edges = torch.tensor([[0, 1, 2, 3, 1, 2], [1, 0, 3, 2, 2, 1]])
    logits = model(torch.randn(4, 8), edges)
    assert logits.shape == (4, 2)
    torch.nn.functional.cross_entropy(logits, torch.tensor([0, 0, 1, 1])).backward()
    assert all(p.grad is not None and torch.isfinite(p.grad).all() for p in model.parameters())


def test_directed_gcn_normalization_hand_computed():
    layer = NativeGraphLayer(1, 1, "gcn")
    with torch.no_grad():
        layer.linear.weight.fill_(1)
        layer.bias.zero_()
    x = torch.tensor([[1.], [2.], [4.]])
    edges = torch.tensor([[1, 2], [0, 1]])
    expected = torch.tensor([[1.5], [1 + 4 / np.sqrt(2)], [4.]], dtype=torch.float32)
    torch.testing.assert_close(layer(x, edges), expected)


def test_full_scale_exact_search_refuses_quadratic_work():
    reference = np.broadcast_to(np.ones((1, 2), dtype=np.float32), (524429, 2))
    with pytest.raises(ValueError, match="exact_limit"):
        cosine_neighbors(reference, np.ones((1, 2)), k=10)


def test_optional_faiss_hnsw_recall_and_memory_guard():
    pytest.importorskip("faiss")
    x = np.random.default_rng(42).normal(size=(100, 16)).astype(np.float32)
    neighbors, _ = cosine_neighbors(x, k=5, backend="faiss", memory_budget_mb=10)
    assert audit_neighbor_recall(x, neighbors)["recall_at_k"] >= .95
    with pytest.raises(MemoryError, match="memory_budget_mb"):
        cosine_neighbors(x, k=5, backend="faiss", memory_budget_mb=.001)
