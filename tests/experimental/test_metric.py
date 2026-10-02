import numpy as np
import pytest
import torch

from src.experimental.metric import (
    BalancedBatchSampler, ProjectionHead, centroid_score, class_centroids,
    supervised_contrastive_loss,
)


@pytest.mark.parametrize("dimension", [128, 256])
def test_projection_norms_and_gradients(dimension):
    torch.manual_seed(42)
    model = ProjectionHead(8, dimension, 16)
    z = model(torch.randn(8, 8))
    torch.testing.assert_close(z.norm(dim=1), torch.ones(8))
    loss = supervised_contrastive_loss(z, torch.tensor([0, 0, 0, 0, 1, 1, 1, 1]), hardness=2)
    loss.backward()
    assert all(p.grad is not None and torch.isfinite(p.grad).all() for p in model.parameters())


def test_supcon_hand_computed_and_zero_hardness():
    z = torch.tensor([[1., 0.], [1., 0.], [0., 1.], [0., 1.]], requires_grad=True)
    y = torch.tensor([0, 0, 1, 1])
    expected = np.log(np.exp(1) + 2) - 1
    loss = supervised_contrastive_loss(z, y, temperature=1)
    assert loss.item() == pytest.approx(expected)
    torch.testing.assert_close(loss, supervised_contrastive_loss(z, y, temperature=1, hardness=0))
    # Equal negative similarities must have identical normalized mining weights.
    torch.testing.assert_close(loss, supervised_contrastive_loss(z, y, temperature=1, hardness=10))


def test_supcon_rejects_invalid_pairs_and_nonfinite():
    with pytest.raises(ValueError, match="positive"):
        supervised_contrastive_loss(torch.eye(4), torch.tensor([0, 0, 0, 1]))
    with pytest.raises(ValueError, match="Zero"):
        supervised_contrastive_loss(torch.zeros(4, 2), torch.tensor([0, 0, 1, 1]))


def test_balanced_batches_unique_and_reproducible():
    y = np.array([0] * 5 + [1] * 20)
    a, b = BalancedBatchSampler(y, 8, 42), BalancedBatchSampler(y, 8, 42)
    assert list(a) == list(b)
    for indices in a:
        assert len(set(indices)) == 8
        assert np.bincount(y[indices]).tolist() == [4, 4]
    a.set_epoch(1)
    assert list(a) != list(b)


def test_centroid_polarity_and_squared_distance_equivalence():
    x = np.array([[1, 0], [1, .1], [0, 1], [.1, 1]], dtype=np.float32)
    centers = class_centroids(x, [0, 0, 1, 1])
    cosine = centroid_score(x, centers)
    distance = centroid_score(x, centers, distance="euclidean")
    np.testing.assert_allclose(cosine, distance)
    assert max(cosine[:2]) < min(cosine[2:])
