"""Adapter, processor and base-file binding for portable offline rebuilds."""

from __future__ import annotations

import copy
import json
import os
from importlib.metadata import version
from pathlib import Path

from PIL import Image

from src.robustness.provenance import contained, digest_file, write_json
from .batches import DEFAULT_PROMPT, answer_batch
from .models import load_base, load_processor, optional_dependencies, resolve_base
from .scoring import LABELS, SCORE_POLICY

SCHEMA = "faceforgery-vlm-bundle-v1"


def _inventory(path):
    return {str(file.relative_to(path)): digest_file(file) for file in sorted(Path(path).rglob("*")) if file.is_file()}


def score_contract(processor, config):
    image = Image.new("RGB", (32, 32), (80, 40, 160))
    _, _, ids = answer_batch(processor, [image, image], list(LABELS),
                            prompt=config.get("prompt", DEFAULT_PROMPT), max_length=int(config.get("max_length", 512)))
    return {"policy": SCORE_POLICY, "labels": list(LABELS), "label_token_ids": ids,
            "eos_in_score": False, "eos_in_training": True, "label_convention": "fake-is-1",
            "prompt": config.get("prompt", DEFAULT_PROMPT), "max_length": int(config.get("max_length", 512))}


def save_bundle(run_dir, model, processor, model_config, *, metadata=None):
    run_dir = Path(run_dir)
    base_path = resolve_base(model_config)
    saved_config = copy.deepcopy(model_config)
    saved_config.pop("pretrained_path", None)
    adapter_path, processor_path = run_dir / "adapter", run_dir / "processor"
    if adapter_path.exists() or processor_path.exists() or (run_dir / "bundle.json").exists():
        raise FileExistsError("Use a fresh VLM bundle directory")
    run_dir.mkdir(parents=True, exist_ok=True)
    for adapter in model.peft_config.values():
        adapter.base_model_name_or_path = str(model_config.get("model_id", base_path.name))
    model.save_pretrained(adapter_path, safe_serialization=True, save_embedding_layers=False)
    processor.save_pretrained(processor_path)
    base_files = {file.name: digest_file(file) for file in sorted(base_path.iterdir())
                  if file.is_file() and (file.suffix == ".safetensors" or file.name in {"config.json", "model.safetensors.index.json"})}
    document = {
        "schema": SCHEMA, "base_subdir": base_path.name, "base_files": base_files,
        "model_config": saved_config, "adapter_files": _inventory(adapter_path),
        "processor_files": _inventory(processor_path), "score_contract": score_contract(processor, saved_config),
        "packages": {name: version(name) for name in ("torch", "transformers", "peft", "accelerate", "safetensors")},
        "metadata": metadata or {},
    }
    write_json(run_dir / "bundle.json", document)
    return run_dir / "bundle.json"


def verify_bundle(run_dir, *, base_root=None):
    run_dir = Path(run_dir)
    document = json.loads((run_dir / "bundle.json").read_text())
    if document.get("schema") != SCHEMA or document.get("score_contract", {}).get("policy") != SCORE_POLICY:
        raise ValueError("Unsupported VLM bundle or scoring contract")
    base_root = base_root or os.environ.get("TCC_PRETRAINED_ROOT")
    if not base_root:
        raise ValueError("Set TCC_PRETRAINED_ROOT to the staged base snapshots before offline VLM reload")
    base_path = contained(base_root, document["base_subdir"])
    for root, key in ((base_path, "base_files"), (run_dir / "adapter", "adapter_files"), (run_dir / "processor", "processor_files")):
        if not document.get(key):
            raise ValueError(f"Empty VLM bundle inventory: {key}")
        for name, checksum in document[key].items():
            file = contained(root, name)
            if digest_file(file) != checksum:
                raise ValueError(f"VLM bundle file differs from its immutable identity: {key}/{name}")
    return document, base_path


def rebuild_bundle(run_dir, device="cpu", *, base_root=None):
    document, base_path = verify_bundle(run_dir, base_root=base_root)
    config = {**document["model_config"], "pretrained_path": str(base_path)}
    base, _ = load_base(config, device, training=False)
    if config.get("quantization", "none") == "nf4":
        # Preserve the same frozen parameter dtypes as the training preparation.
        *_, prepare_kbit = optional_dependencies()
        base = prepare_kbit(base, use_gradient_checkpointing=False)
    *_, peft_type, get_peft, prepare_kbit = optional_dependencies()
    del get_peft, prepare_kbit
    model = peft_type.from_pretrained(base, Path(run_dir) / "adapter", is_trainable=False)
    processor = load_processor(Path(run_dir) / "processor", config)
    # Saved processor folder contains processor config, not the architecture config.
    if score_contract(processor, config) != document["score_contract"]:
        raise ValueError("Reloaded processor changed prompt, label tokens or score semantics")
    return model.eval(), processor, document
