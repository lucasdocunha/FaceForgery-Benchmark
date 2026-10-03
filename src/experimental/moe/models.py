"""Late fusion of frozen forensic scores, with a trainable feature router."""

from __future__ import annotations

import math

import torch
from torch import nn
from torch.nn import functional as F

EPSILON = 1e-6


def checked_scores(scores):
    if scores.ndim != 2 or scores.shape[1] < 2 or not torch.isfinite(scores).all():
        raise ValueError("Expected finite (batch, experts) probabilities with at least two experts")
    if ((scores < 0) | (scores > 1)).any():
        raise ValueError("Frozen expert scores must be declared p_fake in [0, 1]")
    return scores.detach().float()


def simple_fusion(scores, method="mean"):
    """Geometric matches the repository's scalar probability geometric mean."""
    scores = checked_scores(scores)
    if method == "mean":
        return scores.mean(-1)
    if method == "geometric":
        return scores.clamp_min(1e-9).log().mean(-1).exp()
    if method == "geometric_binary":
        p = scores.clamp(EPSILON, 1 - EPSILON)
        return torch.sigmoid(p.logit().mean(-1))
    raise ValueError("Unknown simple fusion method")


def load_balancing_loss(probabilities, assignments):
    """Switch-style N * sum(mean gate mass * detached assignment fraction).

    Top-k assignment fractions are divided by k; soft routing uses a top-1
    occupancy diagnostic. Uniform mass and utilization give a value of one.
    """
    if probabilities.shape != assignments.shape or probabilities.ndim != 2:
        raise ValueError("Router mass and assignment shape mismatch")
    count = assignments.sum(-1, keepdim=True)
    if (count <= 0).any():
        raise ValueError("Every row must have a routing assignment")
    load = (assignments.float() / count).mean(0).detach()
    return probabilities.shape[1] * (probabilities.mean(0) * load).sum()


class FrozenLateFusion(nn.Module):
    """Only router parameters train; inputs and expert predictions are detached.

    Renormalized sparse top-1 produces no classification gradient to its router.
    Training therefore supports soft or top-k with k >= 2. Inference may use
    top-1 after an explicitly configured soft/top-2 training regime.
    """

    def __init__(self, input_dim, num_experts, *, hidden_dim=64, temperature=1.0,
                 train_routing="soft", train_top_k=2, inference_routing="soft",
                 inference_top_k=1, expert_dropout=0.1, mean=None, std=None):
        super().__init__()
        if input_dim < 1 or num_experts < 2 or hidden_dim < 1:
            raise ValueError("Invalid router dimensions")
        if not math.isfinite(float(temperature)) or temperature <= 0:
            raise ValueError("Router temperature must be finite and positive")
        if train_routing not in {"soft", "top_k"} or inference_routing not in {"soft", "top_k"}:
            raise ValueError("Routing must be soft or top_k")
        if not 2 <= train_top_k <= num_experts:
            raise ValueError("Training top-k must be at least two; renormalized top-1 has zero classification gradient")
        if not 1 <= inference_top_k <= num_experts:
            raise ValueError("Inference top-k is outside the expert count")
        if not math.isfinite(float(expert_dropout)) or not 0 <= expert_dropout < 1:
            raise ValueError("Expert dropout must lie in [0, 1)")
        self.input_dim, self.num_experts = int(input_dim), int(num_experts)
        self.temperature, self.expert_dropout = float(temperature), float(expert_dropout)
        self.train_routing, self.train_top_k = train_routing, int(train_top_k)
        self.inference_routing, self.inference_top_k = inference_routing, int(inference_top_k)
        self.register_buffer("feature_mean", torch.zeros(input_dim) if mean is None else torch.as_tensor(mean).float())
        self.register_buffer("feature_std", torch.ones(input_dim) if std is None else torch.as_tensor(std).float())
        if (self.feature_mean.shape != (input_dim,) or self.feature_std.shape != (input_dim,)
                or not torch.isfinite(self.feature_mean).all() or not torch.isfinite(self.feature_std).all()
                or (self.feature_std <= 0).any()):
            raise ValueError("Invalid val_fit feature standardizer")
        self.router = nn.Sequential(nn.Linear(input_dim, hidden_dim), nn.ReLU(), nn.Linear(hidden_dim, num_experts))

    def forward(self, features, scores):
        scores = checked_scores(scores)
        if (features.shape != (len(scores), self.input_dim) or scores.shape[1] != self.num_experts
                or not torch.isfinite(features).all()):
            raise ValueError("Frozen router feature or expert dimension mismatch")
        x = (features.detach().float() - self.feature_mean) / self.feature_std
        logits = self.router(x).float() / self.temperature
        dense = logits.softmax(-1)
        routing = self.train_routing if self.training else self.inference_routing
        k = self.train_top_k if self.training else self.inference_top_k
        available = torch.ones_like(logits, dtype=torch.bool)
        if self.training and self.expert_dropout:
            available = torch.rand_like(logits) >= self.expert_dropout
            # At least two active experts keep the classification gradient alive.
            restore = logits.detach().topk(max(2, k if routing == "top_k" else 2), -1).indices
            insufficient = available.sum(-1) < (k if routing == "top_k" else 2)
            available[insufficient] = available[insufficient].scatter(1, restore[insufficient], True)
        masked = logits.masked_fill(~available, -torch.inf)
        if routing == "top_k":
            selected = masked.topk(k, -1).indices
            assignments = torch.zeros_like(logits, dtype=torch.bool).scatter(1, selected, True)
            weights = masked.masked_fill(~assignments, -torch.inf).softmax(-1)
            occupancy = torch.zeros_like(logits).scatter(1, logits.detach().topk(k, -1).indices, 1)
        else:
            weights = masked.softmax(-1)
            occupancy = F.one_hot(logits.detach().argmax(-1), self.num_experts).float()
        p_fake = (weights * scores).sum(-1).clamp(EPSILON, 1 - EPSILON)
        entropy = -(weights * weights.clamp_min(EPSILON).log()).sum(-1)
        return {"p_fake": p_fake, "logits": torch.stack((torch.log1p(-p_fake), p_fake.log()), -1),
                "weights": weights, "router_probabilities": dense,
                "load_balance": load_balancing_loss(dense, occupancy), "entropy": entropy}

