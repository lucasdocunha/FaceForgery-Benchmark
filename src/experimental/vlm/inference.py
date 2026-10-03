"""Raw-image VLM callback for the shared audited evaluator."""

from __future__ import annotations

from contextlib import nullcontext

import torch

from src.robustness.manifests import validate
from src.robustness.statistics import checked_predictions
from .artifacts import rebuild_bundle
from .batches import DEFAULT_PROMPT, answer_batch, move_inputs
from .data import read_image
from .models import UnsupportedQuantizationError
from .scoring import LABELS, fake_probability, sequence_logprob


class VLMPredictor:
    def __init__(self, model, processor, config, device="cpu"):
        self.model, self.processor, self.config, self.device = model.eval(), processor, config, torch.device(device)

    def __call__(self, frame, root, *, batch_size=1, workers=0, positive_class="fake", hash_images=True,
                 amp=True, use_amp=None, device=None, image_size=None, mode=None, in_channels=None, **kwargs):
        del image_size, in_channels
        if use_amp is not None:
            amp = bool(use_amp)
        if device is not None and torch.device(device) != self.device:
            raise ValueError("Suite device differs from loaded VLM predictor")
        if kwargs:
            raise TypeError(f"Unsupported VLM inference options: {sorted(kwargs)}")
        if positive_class != "fake" or mode is not None:
            raise ValueError("VLM predictor uses native RGB processor and fixed fake-is-1 orientation")
        if int(batch_size) < 1 or workers != 0:
            raise ValueError("VLM predictor requires a positive batch size and workers=0")
        frame = validate(frame, require_both=False)
        probabilities, hashes, real_scores, fake_scores = [], [], [], []
        self.model.eval()
        with torch.inference_mode():
            for start in range(0, len(frame), int(batch_size)):
                rows = frame.iloc[start:start + int(batch_size)]
                loaded = [read_image(root, row, hash_images=hash_images) for row in rows.to_dict("records")]
                images = [value[0] for value in loaded]
                scores = []
                for label in LABELS:
                    batch, mask, _ = answer_batch(self.processor, images, [label] * len(images),
                        prompt=self.config.get("prompt", DEFAULT_PROMPT), max_length=int(self.config.get("max_length", 512)))
                    batch.pop("labels")
                    inputs = move_inputs(batch, self.device)
                    dtype = getattr(torch, self.config.get("dtype", "bfloat16"))
                    context = torch.amp.autocast("cuda", dtype=dtype) if amp and self.device.type == "cuda" and dtype != torch.float32 else nullcontext()
                    try:
                        with context:
                            output = self.model(**inputs, use_cache=False, return_dict=True)
                    except RuntimeError as error:
                        if self.config.get("quantization", "none") == "nf4":
                            raise UnsupportedQuantizationError(f"NF4 inference failed: {error}") from error
                        raise
                    scores.append(sequence_logprob(output.logits, inputs["input_ids"], mask.to(self.device), inputs["attention_mask"]))
                probabilities.extend(fake_probability(*scores).cpu().tolist())
                real_scores.extend(scores[0].cpu().tolist())
                fake_scores.extend(scores[1].cpu().tolist())
                hashes.extend(value[1] for value in loaded)
        result = frame.copy()
        result["p_fake"], result["image_sha256"] = probabilities, hashes
        result["vlm_logprob_real"], result["vlm_logprob_fake"] = real_scores, fake_scores
        return checked_predictions(result)


def load_predictor(run_dir, device="cpu"):
    model, processor, document = rebuild_bundle(run_dir, device)
    predictor = VLMPredictor(model, processor, document["model_config"], device)
    predictor.bundle_metadata = document
    return predictor


def describe_run(run_dir):
    """Verify bundle metadata without instantiating the VLM for a dry run."""
    from pathlib import Path
    from src.robustness.provenance import digest_file
    from .artifacts import verify_bundle
    from .training import input_contract

    root = Path(run_dir)
    document, _ = verify_bundle(root)
    return {"checkpoint_path": root / "bundle.json", "input_contract": input_contract(document),
            "research_run": {"family": "vlm", "seed": document["metadata"].get("seed", 42),
                             "scope": document["metadata"].get("scope", "development"),
                             "bundle_sha256": digest_file(root / "bundle.json")}, "image_size": None}
