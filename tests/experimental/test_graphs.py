import numpy as np
import pytest
import torch

from src.experimental.graphs import (
    GraphClassifier, correct_and_smooth, cosine_neighbors, label_propagation,
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
