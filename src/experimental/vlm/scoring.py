"""Teacher-forced verbalizer scores with fake-is-1 orientation."""

from __future__ import annotations

import torch

SCORE_POLICY = "sum-label-logprob-exclude-eos-v1"
LABELS = ("Real", "Fake")


def sequence_logprob(logits, input_ids, target_mask, attention_mask=None):
    """Sum p(token at t | prefix before t) at explicitly selected positions."""
    if logits.ndim != 3 or input_ids.ndim != 2 or logits.shape[:2] != input_ids.shape:
        raise ValueError("Expected matching BTV logits and BT input IDs")
    if target_mask.shape != input_ids.shape or target_mask.dtype != torch.bool:
        raise ValueError("Target mask must be a BT boolean tensor")
    if target_mask[:, 0].any() or not target_mask.any(dim=1).all():
        raise ValueError("Every answer needs a noninitial target token")
    if attention_mask is not None:
        if attention_mask.shape != input_ids.shape or (target_mask & ~attention_mask.bool()).any():
            raise ValueError("Padding cannot be part of an answer")
    row, token_position = target_mask.nonzero(as_tuple=True)
    selected = logits[row, token_position - 1].float()
    if not torch.isfinite(selected).all():
        raise ValueError("Nonfinite answer logits")
    target_ids = input_ids[row, token_position]
    if (target_ids < 0).any() or (target_ids >= logits.shape[-1]).any():
        raise ValueError("Answer ID lies outside the model vocabulary")
    values = selected.log_softmax(-1).gather(1, target_ids[:, None]).squeeze(1)
    return torch.zeros(len(input_ids), device=logits.device, dtype=torch.float32).scatter_add(0, row, values)


def fake_probability(real_scores, fake_scores):
    if real_scores.shape != fake_scores.shape or real_scores.ndim != 1:
        raise ValueError("Expected one sequence score per image and class")
    scores = torch.stack((real_scores, fake_scores), dim=-1)
    if not torch.isfinite(scores).all():
        raise ValueError("Class sequence scores must be finite")
    return scores.softmax(-1)[:, 1]


def sft_loss_step(model, batch, step_state=None):
    """Mean per-image answer NLL; runtime accumulation weight is image count."""
    del step_state
    labels = batch["labels"]
    inputs = {key: value for key, value in batch.items() if key != "labels"}
    outputs = model(**inputs, use_cache=False, return_dict=True)
    supervised = labels.ne(-100)
    sums = sequence_logprob(outputs.logits, batch["input_ids"], supervised, batch["attention_mask"])
    loss = (-sums / supervised.sum(-1)).mean()
    return loss, {"answer_nll": float(loss.detach())}, len(labels)
