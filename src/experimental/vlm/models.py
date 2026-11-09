"""Local, native-HF VLM loading and explicitly scoped PEFT adaptation."""

from __future__ import annotations

import json
import math
from pathlib import Path

import torch

MAX_PARAMETERS = 4_500_000_000
# Transformers 5 matches skip entries as regex prefixes or exact suffixes.
# A bare parent name does not exclude nested children such as
# model.vision_model.encoder.layers.0.self_attn.q_proj.
NF4_EXCLUSIONS = [r"(^|.*\.)(vision_model|connector|visual)(\.|$)"]


class UnsupportedQuantizationError(RuntimeError):
    pass


def optional_dependencies():
    try:
        from transformers import AutoModelForImageTextToText, AutoProcessor, BitsAndBytesConfig
        from peft import LoraConfig, PeftModel, get_peft_model, prepare_model_for_kbit_training
    except ImportError as error:
        raise ImportError("VLM experiments require Transformers 5.14.1, PEFT and Accelerate from optional dependencies") from error
    return AutoModelForImageTextToText, AutoProcessor, BitsAndBytesConfig, LoraConfig, PeftModel, get_peft_model, prepare_model_for_kbit_training


def stored_parameters(path):
    from safetensors import safe_open
    path = Path(path)
    config = json.loads((path / "config.json").read_text())
    if config.get("quantization_config"):
        raise ValueError("Use a full-precision base snapshot so the <=3B count is verifiable")
    files = sorted(path.glob("model*.safetensors"))
    if not files:
        raise FileNotFoundError(f"Missing local model safetensors: {path}")
    total, seen = 0, set()
    for file in files:
        with safe_open(file, framework="pt", device="cpu") as handle:
            for key in handle.keys():
                if key in seen:
                    raise ValueError("Duplicate base tensor across safetensor shards")
                seen.add(key)
                total += math.prod(handle.get_slice(key).get_shape())
    if total > MAX_PARAMETERS:
        raise ValueError(f"Base model has {total:,} stored parameters, exceeding the <=3B experiment cap")
    return total


def resolve_base(config):
    path = Path(config["pretrained_path"]).expanduser()
    if not path.is_dir():
        raise FileNotFoundError(f"Stage the pinned VLM snapshot locally first: {path}")
    return path


def load_processor(path, config):
    _, processor_type, *_ = optional_dependencies()
    try:
        processor = processor_type.from_pretrained(path, local_files_only=True, trust_remote_code=False, backend="pil")
    except (AttributeError, TypeError):
        processor = processor_type.from_pretrained(path, local_files_only=True, trust_remote_code=False)
    processor.tokenizer.padding_side = "right"
    overrides = config.get("processor", {})
    if hasattr(processor, "image_seq_len"):
        architecture_file = Path(path) / "config.json"
        if not architecture_file.exists():
            architecture_file = resolve_base(config) / "config.json"
        architecture = json.loads(architecture_file.read_text())
        vision = architecture["vision_config"]
        trained_edge = int(vision["image_size"])
        edge = int(overrides.get("max_image_edge", trained_edge))
        if edge != trained_edge:
            raise ValueError("VLM processor must retain the trained image patch geometry")
        processor.image_processor.size = {"longest_edge": edge}
        processor.image_processor.max_image_size = {"longest_edge": edge}
        processor.image_processor.do_image_splitting = bool(overrides.get("do_image_splitting", False))
        if processor.image_processor.do_image_splitting:
            raise ValueError("Bounded SmolVLM path requires do_image_splitting=False")
        expected_tokens = (trained_edge // int(vision["patch_size"])) ** 2 // int(architecture["scale_factor"]) ** 2
        if processor.image_seq_len != expected_tokens:
            raise ValueError("Processor image sequence length does not match the architecture")
    else:
        # Qwen's size bounds are pixel counts, not a legacy square image transform.
        if "max_pixels" in overrides:
            size = dict(processor.image_processor.size)
            size["longest_edge"] = int(overrides["max_pixels"])
            size["shortest_edge"] = int(overrides.get("min_pixels", 65_536))
            processor.image_processor.size = size
    return processor


def decoder_targets(model, projections):
    allowed = {"q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"}
    if not projections or not set(projections) <= allowed:
        raise ValueError("Declare supported decoder projection names")
    names = [name for name, layer in model.named_modules()
             if name.startswith(("model.text_model.", "model.language_model."))
             and name.rsplit(".", 1)[-1] in projections and hasattr(layer, "weight")]
    if not names or any("vision" in name or "connector" in name for name in names):
        raise ValueError("No verified text-decoder adapter targets in this VLM architecture")
    return names


def adapt_model(model, config):
    _, _, _, lora_type, _, get_peft, prepare_kbit = optional_dependencies()
    lora = config.get("lora", {})
    targets = decoder_targets(model, lora.get("projections", ["q_proj", "v_proj"]))
    model.requires_grad_(False)
    if config.get("quantization", "none") == "nf4":
        model = prepare_kbit(model, gradient_checkpointing_kwargs={"use_reentrant": False})
    elif config.get("gradient_checkpointing", True):
        model.gradient_checkpointing_enable(gradient_checkpointing_kwargs={"use_reentrant": False})
    model.config.use_cache = False
    result = get_peft(model, lora_type(
        r=int(lora.get("rank", 8)), lora_alpha=int(lora.get("alpha", 16)),
        lora_dropout=float(lora.get("dropout", 0.05)), bias="none", task_type="CAUSAL_LM",
        target_modules=targets,
    ))
    trainable = [name for name, value in result.named_parameters() if value.requires_grad]
    if not trainable or any("lora_" not in name or "vision" in name for name in trainable):
        raise ValueError("Unexpected non-decoder trainable parameters")
    return result, targets


def load_base(config, device="cpu", *, training=False):
    model_type, _, bnb_config, *_ = optional_dependencies()
    path = resolve_base(config)
    architecture = json.loads((path / "config.json").read_text())
    if config.get("model_type") and architecture["model_type"] != config["model_type"]:
        raise ValueError("Staged VLM architecture differs from the requested model_type")
    count = stored_parameters(path)
    device = torch.device(device)
    dtype_name = config.get("dtype", "bfloat16" if device.type == "cuda" else "float32")
    if dtype_name not in {"bfloat16", "float16", "float32"}:
        raise ValueError("Unsupported VLM dtype")
    dtype = getattr(torch, dtype_name)
    quantization = config.get("quantization", "none")
    if quantization not in {"none", "nf4"}:
        raise ValueError("Quantization must be explicitly none or nf4")
    kwargs = dict(local_files_only=True, trust_remote_code=False, use_safetensors=True, dtype=dtype,
                  device_map={"": str(device)}, attn_implementation=config.get("attention", "sdpa"))
    if quantization == "nf4":
        if device.type != "cuda" or not torch.cuda.is_available():
            raise UnsupportedQuantizationError("NF4 requires a verified CUDA-device bitsandbytes backend; select an explicit BF16 run for fallback")
        try:
            import bitsandbytes  # noqa: F401
        except ImportError as error:
            raise UnsupportedQuantizationError("NF4 requested but bitsandbytes is missing") from error
        kwargs["quantization_config"] = bnb_config(
            load_in_4bit=True, bnb_4bit_quant_type="nf4", bnb_4bit_use_double_quant=True,
            bnb_4bit_compute_dtype=dtype, llm_int8_skip_modules=NF4_EXCLUSIONS,
        )
    try:
        model = model_type.from_pretrained(path, **kwargs)
    except Exception as error:
        if quantization == "nf4":
            raise UnsupportedQuantizationError(f"NF4 base load failed explicitly: {type(error).__name__}: {error}") from error
        raise
    quantized = [name for name, layer in model.named_modules() if layer.__class__.__name__ == "Linear4bit"]
    if quantization == "nf4":
        if not any(name.startswith(("model.text_model.", "model.language_model.")) for name in quantized):
            raise UnsupportedQuantizationError("NF4 requested but no verified 4-bit decoder layers were constructed")
        if any("vision" in name or "connector" in name or "visual" in name for name in quantized):
            raise UnsupportedQuantizationError("NF4 unexpectedly quantized the excluded vision tower or connector")
    model.config.use_cache = False
    targets = []
    if training:
        model, targets = adapt_model(model, config)
    storage = {}
    for parameter in model.parameters():
        key = str(parameter.dtype)
        item = storage.setdefault(key, {"stored_numel": 0, "storage_bytes": 0})
        item["stored_numel"] += parameter.numel()
        item["storage_bytes"] += parameter.numel() * parameter.element_size()
    return model, {"stored_parameters": count, "adapter_targets": targets, "dtype": dtype_name,
                   "quantization": quantization, "quantized_decoder_modules": quantized,
                   "parameter_storage_by_dtype": storage,
                   "storage_count_policy": "actual parameter tensor storage after PEFT preparation; 4-bit packed counts are not original model parameter counts"}
