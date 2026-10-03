"""PEFT task adapter around the shared objective-aware training runtime."""

from __future__ import annotations

import copy
from pathlib import Path

import torch
from torch.utils.data import DataLoader

from src.experimental.runtime import fit_model
from src.robustness.artifacts import save_predictions
from src.robustness.engine import seed_all
from src.robustness.inference import calibrate, evaluation_report
from src.robustness.manifests import assert_disjoint, load_manifest, save_manifest
from src.robustness.provenance import digest, digest_file, write_json
from src.robustness.statistics import summary
from .artifacts import save_bundle
from .batches import DEFAULT_PROMPT, SFTCollator, move_inputs
from .data import RawImageDataset, balanced_subset
from .inference import VLMPredictor
from .models import load_base, load_processor, resolve_base
from .scoring import sft_loss_step


def input_contract(document):
    return {"kind": "native-vlm", "bundle_sha256": digest(document), "checkpoint_class1": "fake", "image_input": "raw-RGB-PIL",
            "processor_files_sha256": digest(document["processor_files"]),
            "processor_overrides": document["model_config"].get("processor", {}),
            "score_contract": document["score_contract"]}


def fit(config):
    """Fit certified source train/val only and return a portable run directory."""
    import json
    from peft import get_peft_model_state_dict, set_peft_model_state_dict

    config = copy.deepcopy(config)
    if config.get("task") != "vlm":
        raise ValueError("VLM fit requires task=vlm")
    root = Path(config["output_dir"])
    training, data, model_config = config.get("training", {}), config["data"], config["model"]
    device, seed = training.get("device", "cpu"), int(config.get("seed", 42))
    if int(training.get("workers", 0)) != 0:
        raise ValueError("Bounded VLM loading currently requires workers=0")
    torch.set_num_threads(min(2, int(training.get("cpu_threads", 2))))
    seed_all(seed)  # Model and adapter initialization must precede runtime reseeding.
    train, train_record = load_manifest(data["train_manifest"])
    val, val_record = load_manifest(data["val_manifest"])
    disjoint = assert_disjoint(train, val)
    train = balanced_subset(train, data.get("train_limit"), seed)
    val = balanced_subset(val, data.get("val_limit"), seed)
    resume = bool(training.get("resume", False))
    config["training"].pop("resume", None)
    root.mkdir(parents=True, exist_ok=True)
    if not resume:
        train_record = save_manifest(train, root / "selected_train.csv", {"source_manifest_sha256": train_record["manifest_sha256"]})
        val_record = save_manifest(val, root / "selected_val.csv", {"source_manifest_sha256": val_record["manifest_sha256"]})
    else:
        train, train_record = load_manifest(root / "selected_train.csv")
        val, val_record = load_manifest(root / "selected_val.csv")
    processor = load_processor(resolve_base(model_config), model_config)
    model, model_info = load_base(model_config, device, training=True)
    model.config.pad_token_id = processor.tokenizer.pad_token_id
    if hasattr(model.config, "text_config"):
        model.config.text_config.pad_token_id = processor.tokenizer.pad_token_id
    dataset = RawImageDataset(train, data["train_root"])
    loader = DataLoader(dataset, batch_size=int(training.get("batch_size", 1)), shuffle=True,
                        num_workers=0, generator=torch.Generator().manual_seed(seed),
                        collate_fn=SFTCollator(processor, prompt=model_config.get("prompt", DEFAULT_PROMPT),
                                              max_length=int(model_config.get("max_length", 512))))
    optimizer = torch.optim.AdamW([value for value in model.parameters() if value.requires_grad],
                                  lr=float(training.get("lr", 2e-4)), weight_decay=float(training.get("weight_decay", 0.0)))

    def loss_step(current, batch, state):
        return sft_loss_step(current, move_inputs(batch, device), state)

    def predict(current):
        return VLMPredictor(current, processor, model_config, device)(val, data["val_root"],
            batch_size=int(training.get("eval_batch_size", 1)), use_amp=bool(training.get("amp", True)))

    def validate(current):
        rows = predict(current)
        return summary(rows.label, rows.p_fake)

    zero_shot = validate(model) if not resume else None
    result = fit_model(model, loader, optimizer, loss_step=loss_step, validate=validate,
        run_dir=root, config=config, epochs=int(training.get("epochs", 2)), device=device,
        grad_accum_steps=int(training.get("grad_accum_steps", 8)), selection_key="auc", selection_mode="max",
        resume=resume, max_grad_norm=float(training.get("max_grad_norm", 1.0)),
        state_dict_fn=get_peft_model_state_dict, load_state_fn=set_peft_model_state_dict)
    bundle_path = save_bundle(root, model, processor, model_config, metadata={
        **model_info, "seed": seed, "train_manifest_sha256": train_record["manifest_sha256"],
        "val_manifest_sha256": val_record["manifest_sha256"], "disjointness": disjoint,
        "loss_weight_unit": "image; within-image mean answer-plus-EOS token NLL", "scope": config.get("scope", "development"),
    })
    document = json.loads(bundle_path.read_text())
    contract, checksum = input_contract(document), digest_file(bundle_path)
    predictions = predict(model)
    save_predictions(root / "validation_predictions.csv", predictions, manifest_record=val_record,
                     model_sha256=checksum, metadata={"input_contract": contract, "origin": "native VLM teacher forcing"})
    calibration = calibrate(predictions, manifest_record=val_record, output=root / "calibration.json",
                            model_sha256=checksum, policy=training.get("threshold_policy", "youden"), input_contract=contract)
    metrics = evaluation_report(predictions, calibration, checksum)
    metrics.update(scope=config.get("scope", "development"), zero_shot=zero_shot, training=result, model=model_info)
    write_json(root / "metrics.json", metrics)
    write_json(root / "status.json", {"state": "complete", "bundle_sha256": checksum,
        "artifacts": {name: digest_file(root / name) for name in
                      ("bundle.json", "calibration.json", "metrics.json", "validation_predictions.csv", "validation_predictions.csv.json")}})
    return root
