"""Processor-aware answer spans. Never infer padding from token identity."""

from __future__ import annotations

import torch

from .scoring import LABELS

DEFAULT_PROMPT = "Classify the face image as authentic or manipulated. Answer with exactly Real or Fake."


def messages(prompt):
    return [{"role": "user", "content": [{"type": "image"}, {"type": "text", "text": prompt}]}]


def _process(processor, texts, images, max_length):
    encoded = dict(processor(text=texts, images=images, padding=True, truncation=False, return_tensors="pt"))
    if encoded["input_ids"].shape[1] > max_length:
        raise ValueError("VLM sequence exceeds max_length; visual/answer truncation is forbidden")
    return {key: value for key, value in encoded.items() if isinstance(value, torch.Tensor)}


def answer_batch(processor, images, labels, *, prompt=DEFAULT_PROMPT, max_length=512):
    """Return model inputs, label-only score mask and answer-plus-EOS training mask."""
    if len(images) != len(labels) or not images or any(label not in LABELS for label in labels):
        raise ValueError("One Real/Fake continuation is required for every image")
    if processor.tokenizer.padding_side != "right":
        raise ValueError("VLM teacher forcing currently requires right padding")
    prefix = processor.apply_chat_template(messages(prompt), tokenize=False, add_generation_prompt=True)
    full = [processor.apply_chat_template(
        messages(prompt) + [{"role": "assistant", "content": [{"type": "text", "text": label}]}],
        tokenize=False, add_generation_prompt=False,
    ) for label in labels]
    prefixes = _process(processor, [prefix] * len(images), images, max_length)
    result = _process(processor, full, images, max_length)
    if hasattr(processor, "image_seq_len"):
        image_id = processor.image_token_id
        if not (result["input_ids"].eq(image_id).sum(-1) == processor.image_seq_len).all():
            raise ValueError("Bounded VLM batch requires one complete image-token sequence per image")
        if result.get("pixel_values") is None or result["pixel_values"].shape[1] != 1:
            raise ValueError("Bounded VLM batch requires exactly one image patch per image")
    score_mask = torch.zeros_like(result["input_ids"], dtype=torch.bool)
    training_labels = torch.full_like(result["input_ids"], -100)
    eos_id = processor.tokenizer.eos_token_id
    if eos_id is None:
        raise ValueError("Processor must declare the assistant end-of-utterance token")
    verbalizers = []
    for index in range(len(images)):
        n_prefix = int(prefixes["attention_mask"][index].sum())
        n_full = int(result["attention_mask"][index].sum())
        prefix_ids = prefixes["input_ids"][index, :n_prefix]
        ids = result["input_ids"][index]
        if n_prefix >= n_full or not torch.equal(prefix_ids, ids[:n_prefix]):
            raise ValueError("Assistant boundary changes prefix tokenization; cannot infer label span")
        ends = (ids[n_prefix:n_full] == eos_id).nonzero().flatten()
        if len(ends) != 1 or int(ends[0]) == 0:
            raise ValueError("Assistant answer must contain label tokens and exactly one closing marker")
        end = n_prefix + int(ends[0])
        score_mask[index, n_prefix:end] = True
        training_labels[index, n_prefix:end + 1] = ids[n_prefix:end + 1]
        verbalizers.append(ids[n_prefix:end].tolist())
    result["labels"] = training_labels
    return result, score_mask, verbalizers


class SFTCollator:
    def __init__(self, processor, *, prompt=DEFAULT_PROMPT, max_length=512):
        self.processor, self.prompt, self.max_length = processor, prompt, max_length

    def __call__(self, rows):
        batch, _, _ = answer_batch(
            self.processor, [row["image"] for row in rows], [LABELS[int(row["label"])] for row in rows],
            prompt=self.prompt, max_length=self.max_length,
        )
        return batch


def move_inputs(batch, device, dtype=None):
    return {key: value.to(device=device, dtype=dtype if dtype is not None and value.is_floating_point() else value.dtype)
            for key, value in batch.items()}
