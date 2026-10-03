"""Audited source-only fit, checkpoint reconstruction and inference for A-D."""

from __future__ import annotations

import copy
from dataclasses import asdict
import json
import math
from pathlib import Path
import time

import torch
from torch.nn import functional as F
from torch.utils.data import DataLoader

from src.robustness.artifacts import save_predictions
from src.robustness.engine import atomic_torch, seed_all
from src.robustness.imaging import CanonicalDataset
from src.robustness.inference import calibrate, evaluate, predict as audited_predict, prediction_contract
from src.robustness.manifests import load_manifest
from src.robustness.provenance import contained, digest, digest_file, source_identity, write_json
from src.robustness.statistics import summary

from .losses import BetaSchedule, CompositeReconstructionLoss
from .models import AutoencoderConfig, FaceAutoencoder, LatentDetector, MeanReconstructionEnsemble, ReconstructionAnomaly, ResidualDetector


TRAINING_DEFAULTS = {
    "epochs": 5, "batch_size": 8, "workers": 0, "grad_accum_steps": 1,
    "lr_ae": 0.001, "lr_backbone": 0.0001, "lr_head": 0.001,
    "weight_decay": 0.0001, "max_grad_norm": 1.0, "amp": True,
    "early_stop_patience": 3, "scheduler_patience": 2, "recipe": "basic_v1",
    "reconstruction_weight": 1.0, "threshold_policy": "youden", "plots": False,
}
MODEL_DEFAULTS = {
    "ae_run": None, "ae_mode": "frozen", "input_mode": "full", "gradient": False,
    "backbone": "small", "backbone_weights": None, "width": 16,
    "hidden_dim": 64, "dropout": 0.1, "allow_untrained_ae": False,
}


def normalize_config(config):
    config = copy.deepcopy(config)
    required = {"task", "name", "seed", "output_dir", "data", "model", "training"}
    if set(config) != required:
        raise ValueError(f"Reconstruction config fields must be exactly {sorted(required)}")
    if config["task"] not in {"ae", "residual", "latent"}:
        raise ValueError("task must be ae, residual, or latent")
    if not isinstance(config["seed"], int) or isinstance(config["seed"], bool) or config["seed"] < 0:
        raise ValueError("seed must be a nonnegative integer")
    if not config["name"] or not config["output_dir"]:
        raise ValueError("Declare experiment name and output_dir")
    if set(config["data"]) != {"train_manifest", "val_manifest", "train_root", "val_root"}:
        raise ValueError("Declare certified source train/val manifests and image roots only")
    if set(config["model"]) - (set(MODEL_DEFAULTS) | {"ae", "freeze_ae"}):
        raise ValueError("Unknown reconstruction model field")
    if set(config["training"]) - (set(TRAINING_DEFAULTS) | {"loss"}):
        raise ValueError("Unknown reconstruction training field")
    if "freeze_ae" in config["model"]:
        legacy = config["model"].pop("freeze_ae")
        if not isinstance(legacy, bool):
            raise ValueError("Use a YAML boolean for freeze_ae")
        explicit = config["model"].get("ae_mode")
        if explicit and legacy != (explicit == "frozen"):
            raise ValueError("freeze_ae conflicts with the explicit ae_mode")
        config["model"].setdefault("ae_mode", "frozen" if legacy else "recon_finetune")
    config["model"] = {**MODEL_DEFAULTS, **config["model"]}
    config["model"]["ae"] = asdict(AutoencoderConfig(**config["model"].get("ae", {})))
    config["training"] = {**TRAINING_DEFAULTS, **config["training"]}
    t, m = config["training"], config["model"]
    for key in ("epochs", "batch_size", "workers", "grad_accum_steps", "early_stop_patience", "scheduler_patience"):
        if not isinstance(t[key], int) or isinstance(t[key], bool) or t[key] < (0 if key in {"workers", "scheduler_patience"} else 1):
            raise ValueError(f"Invalid integer training field: {key}")
    for key in ("lr_ae", "lr_backbone", "lr_head", "weight_decay", "reconstruction_weight"):
        if not math.isfinite(float(t[key])) or float(t[key]) < 0:
            raise ValueError(f"Invalid nonnegative training field: {key}")
    if min(t["lr_ae"], t["lr_backbone"], t["lr_head"]) <= 0:
        raise ValueError("Learning rates must be positive")
    for key in ("gradient", "allow_untrained_ae"):
        if not isinstance(m[key], bool):
            raise ValueError(f"Use a YAML boolean for {key}")
    if not isinstance(t["amp"], bool) or not isinstance(t["plots"], bool):
        raise ValueError("amp and plots must be booleans")
    if m["ae_mode"] not in {"frozen", "recon_finetune", "end_to_end"}:
        raise ValueError("ae_mode must be frozen, recon_finetune, or end_to_end")
    if m["input_mode"] not in {"x_only", "residual_only", "full"}:
        raise ValueError("input_mode must be x_only, residual_only, or full")
    if m["backbone"] not in {"small", "resnet18"} or not isinstance(m["width"], int) or m["width"] < 4 or not isinstance(m["hidden_dim"], int) or m["hidden_dim"] < 2:
        raise ValueError("Invalid detector backbone or dimensions")
    if not 0 <= m["dropout"] < 1:
        raise ValueError("dropout must lie in [0,1)")
    if t["max_grad_norm"] is not None and (not math.isfinite(float(t["max_grad_norm"])) or t["max_grad_norm"] <= 0):
        raise ValueError("max_grad_norm must be positive or null")
    if t["recipe"] not in {"none", "basic_v1", "robust_v1"}:
        raise ValueError("Unknown augmentation recipe")
    if t["threshold_policy"] not in {"balanced_accuracy", "youden"}:
        raise ValueError("Threshold policy must be balanced_accuracy or youden")
    loss = {"lambda_ssim": 0.1, "lambda_lpips": 0.0, "kl_reduction": "sum", "lpips_state_path": None, "lpips_net": "alex", **t.get("loss", {})}
    if set(loss) - {"lambda_ssim", "lambda_lpips", "kl_reduction", "lpips_state_path", "lpips_net", "beta"}:
        raise ValueError("Unknown reconstruction loss field")
    if loss["kl_reduction"] not in {"sum", "mean_per_dim"} or any(not math.isfinite(float(loss[key])) or loss[key] < 0 for key in ("lambda_ssim", "lambda_lpips")):
        raise ValueError("Invalid reconstruction loss scale")
    loss["beta"] = asdict(BetaSchedule(**loss.get("beta", {})))
    if m["ae"]["kind"] != "vae" and loss["beta"]["maximum"]:
        raise ValueError("KL regularization requires a VAE")
    t["loss"] = loss
    if config["task"] == "latent" and m["ae"]["kind"] != "vae":
        raise ValueError("The latent detector requires a VAE")
    if config["task"] != "ae" and m["ae_mode"] != "frozen" and t["reconstruction_weight"] <= 0:
        raise ValueError("Trainable AE requires a positive genuine-only reconstruction weight")
    return config


def build_model(model_config, *, task="ae", initialize=True):
    if task == "mean_ensemble":
        return MeanReconstructionEnsemble(
            build_model(model_config["spatial"], task="residual", initialize=False),
            build_model(model_config["latent"], task="latent", initialize=False),
        )
    cfg = {**MODEL_DEFAULTS, **model_config}
    ae = FaceAutoencoder(cfg.get("ae", {}))
    if task != "ae" and initialize:
        if cfg["ae_run"]:
            original = load_model(cfg["ae_run"], "cpu")
            if not isinstance(original, ReconstructionAnomaly):
                raise ValueError("ae_run must name a completed AE pretraining run")
            if original.autoencoder.config != ae.config:
                raise ValueError("Configured autoencoder differs from ae_run architecture")
            ae.load_state_dict(original.autoencoder.state_dict(), strict=True)
        elif not cfg["allow_untrained_ae"]:
            raise ValueError("Detector requires ae_run; untrained AE is only an explicit smoke control")
    if task == "ae":
        return ReconstructionAnomaly(ae)
    if task == "residual":
        return ResidualDetector(ae, ae_mode=cfg["ae_mode"], input_mode=cfg["input_mode"], gradient=cfg["gradient"], backbone=cfg["backbone"], backbone_weights=cfg["backbone_weights"], width=cfg["width"], dropout=cfg["dropout"], initialize=initialize)
    if task == "latent":
        return LatentDetector(ae, ae_mode=cfg["ae_mode"], hidden_dim=cfg["hidden_dim"], dropout=cfg["dropout"])
    raise ValueError("Unknown reconstruction task")


def _verify_artifacts(root):
    status = json.loads((root / "status.json").read_text())
    if status.get("state") != "complete":
        raise ValueError("Reconstruction run is not complete")
    for name, expected in status.get("artifacts", {}).items():
        if Path(name).name != name or digest_file(root / name) != expected:
            raise ValueError("Completed reconstruction artifact is missing or modified")
    if not {"run.json", "best.pt", "calibration.json", "validation_predictions.csv"} <= set(status.get("artifacts", {})):
        raise ValueError("Incomplete reconstruction certificate")
    return status


def load_model(run_dir, device="cpu"):
    """Rebuild full detector weights without pretrained assets or network access."""
    root = Path(run_dir)
    _verify_artifacts(root)
    record = json.loads((root / "run.json").read_text())
    bundle = torch.load(root / "best.pt", map_location="cpu", weights_only=True)
    if record["config_sha256"] != digest(record["config"]) or bundle["config_sha256"] != record["config_sha256"]:
        raise ValueError("Checkpoint/config identity mismatch")
    model = build_model(record["config"]["model"], task=record["config"]["task"], initialize=False)
    model.load_state_dict(bundle["state_dict"], strict=True)
    return model.to(device).eval()


def run_contract(run_dir):
    root = Path(run_dir)
    record = json.loads((root / "run.json").read_text())
    checkpoint_hash, record_hash = digest_file(root / "best.pt"), digest_file(root / "run.json")
    return {**prediction_contract(record["config"]["model"]["ae"]["image_size"]),
            "run_config_sha256": record_hash,
            "bundle_sha256": digest({"checkpoint": checkpoint_hash, "run_config": record_hash})}


def evaluation_identity(record):
    cfg = record["config"]

    def without_asset_paths(value):
        if isinstance(value, dict):
            return {k: without_asset_paths(v) for k, v in value.items() if k not in {"ae_run", "backbone_weights", "lpips_state_path"}}
        return value

    provenance = cfg["provenance"]
    condition = {"task": cfg["task"], "model": without_asset_paths(cfg["model"]),
                 "training": without_asset_paths(cfg["training"]),
                 "train_manifest_sha256": provenance["train_manifest_sha256"],
                 "val_manifest_sha256": provenance["val_manifest_sha256"],
                 "training_images_sha256": provenance.get("training_images_sha256"),
                 "validation_images_sha256": provenance.get("validation_images_sha256"),
                 "initialization_sha256": {k: v for k, v in provenance.get("initialization_sha256", {}).items() if k != "ae_checkpoint"},
                 "ae_pretraining_condition": provenance.get("ae_pretraining_condition"),
                 "component_conditions": provenance.get("component_conditions"),
                 "reconstruction_code_sha256": provenance.get("reconstruction_code_sha256"),
                 "software_code_sha256": record["software"].get("code_sha256"),
                 "packages": record["software"].get("packages")}
    return {"name": cfg["name"], "seed": cfg["seed"], "run_id": record["config_sha256"],
            "condition": condition, "condition_sha256": digest(condition)}


def describe_run(run_dir):
    """Verify an inference bundle without instantiating models or loading assets."""
    root = Path(run_dir)
    _verify_artifacts(root)
    record = json.loads((root / "run.json").read_text())
    if digest(record["config"]) != record["config_sha256"]:
        raise ValueError("Reconstruction configuration identity mismatch")
    size = record["config"]["model"]["ae"]["image_size"]
    contract = run_contract(root)
    calibration = json.loads((root / "calibration.json").read_text())
    checkpoint_hash = digest_file(root / "best.pt")
    if calibration.get("model_sha256") != checkpoint_hash or calibration.get("input_contract_sha256") != digest(contract) or calibration.get("selection_split") != "val":
        raise ValueError("Calibration model or input identity mismatch")
    return {"checkpoint_path": str(root / "best.pt"), "calibration_path": str(root / "calibration.json"),
            "bundle_sha256": contract["bundle_sha256"], "input_contract": contract,
            "research_run": evaluation_identity(record), "run_record": record, "image_size": size}


class _EpochDataset(CanonicalDataset):
    def set_epoch(self, epoch):
        self.epoch = int(epoch)


def _sources(config):
    data = config["data"]
    train, tc = load_manifest(data["train_manifest"], require_both=config["task"] != "ae")
    val, vc = load_manifest(data["val_manifest"])
    if set(train.split) != {"train"} or set(val.split) != {"val"}:
        raise ValueError("Only source train and validation splits are permitted")
    for frame, image_root in ((train, data["train_root"]), (val, data["val_root"])):
        observed = [digest_file(contained(image_root, name)) for name in frame.img_name]
        if "sha256" in frame and any(expected and expected != actual for expected, actual in zip(frame.sha256, observed)):
            raise ValueError("Source image bytes differ from the declared manifest")
        frame["sha256"] = observed
    for key in ("sample_id", "group_id", "source_id", "sha256"):
        if key in train and key in val:
            overlap = (set(train[key].astype(str)) & set(val[key].astype(str))) - {"", "unknown"}
            if overlap:
                raise ValueError(f"Train/validation overlap in {key}")
    effective = train[train.label.eq(0)].copy() if config["task"] == "ae" else train.copy()
    if effective.empty or not train.label.eq(0).any() or not val.label.eq(0).any():
        raise ValueError("Genuine images are required in source train and validation")
    return effective.reset_index(drop=True), val, tc, vc


def _asset_identity(config, *, previous=None):
    candidates = {"ae_checkpoint": str(Path(config["model"]["ae_run"]) / "best.pt") if config["model"]["ae_run"] else None,
                  "backbone_weights": config["model"]["backbone_weights"],
                  "lpips_state": config["training"]["loss"]["lpips_state_path"]}
    hashes = {}
    for name, path in candidates.items():
        if path:
            if Path(path).is_file():
                hashes[name] = digest_file(path)
            elif previous and name in previous:
                hashes[name] = previous[name]
            else:
                raise FileNotFoundError(f"Missing declared {name}: {path}")
    return hashes


def _posterior(details):
    return {"mu": details["mu"], "logvar": details["logvar"]} if details["logvar"] is not None else {}


def _fit_latent_normalizer(model, loader, device):
    model.eval()
    total = square_total = None
    count = 0
    with torch.no_grad():
        for batch in loader:
            details = model.autoencoder.reconstruct(batch["image"].to(device), sample=False)
            features = model.latent_features(details).double().cpu()
            total = features.sum(0) if total is None else total + features.sum(0)
            square_total = features.square().sum(0) if square_total is None else square_total + features.square().sum(0)
            count += len(features)
    mean = total / count
    scale = (square_total / count - mean.square()).clamp_min(0).sqrt().clamp_min(1e-4)
    model.feature_mean.copy_(mean.to(model.feature_mean))
    model.feature_scale.copy_(scale.to(model.feature_scale))


def fit(config, *, device="cpu", resume=False):
    from src.experimental.runtime import fit_model

    cfg = normalize_config(config)
    root, t, m = Path(cfg["output_dir"]), cfg["training"], cfg["model"]
    train, val, tc, vc = _sources(cfg)
    old_record = json.loads((root / "run.json").read_text()) if resume and (root / "run.json").is_file() else None
    previous_assets = old_record["config"]["provenance"]["initialization_sha256"] if old_record else None
    cfg["provenance"] = {
        "train_manifest_sha256": tc["manifest_sha256"], "val_manifest_sha256": vc["manifest_sha256"],
        "training_images_sha256": digest(sorted(zip(train.sample_id, train.sha256))),
        "validation_images_sha256": digest(sorted(zip(val.sample_id, val.sha256))),
        "effective_training_ids_sha256": digest(sorted(train.sample_id)),
        "n_train": len(train), "n_train_real": int(train.label.eq(0).sum()), "n_train_fake": int(train.label.eq(1).sum()),
        "n_val": len(val), "n_val_real": int(val.label.eq(0).sum()),
        "label_convention": "fake-is-1",
        "ae_gradient_source": ("classification on all training examples plus genuine reconstruction" if m["ae_mode"] == "end_to_end" else "frozen after genuine pretraining") if cfg["task"] != "ae" and m["ae_mode"] != "recon_finetune" else "genuine reconstruction only",
        "initialization_exposure": "generic_pretrained" if m["backbone_weights"] else "scratch",
        "training_fake_source": "none" if cfg["task"] == "ae" else "source-manifest fakes",
        "score_semantics": "bounded mean raw-RGB L1; larger is faker; not probability calibration" if cfg["task"] == "ae" else "softmax class 1 is fake",
        "initialization_sha256": _asset_identity(cfg, previous=previous_assets),
        "reconstruction_code_sha256": {p.name: digest_file(p) for p in sorted(Path(__file__).parent.glob("*.py"))},
        "training_runtime_sha256": digest_file(Path(__file__).parents[1] / "runtime.py"),
        "canonical_imaging_sha256": digest_file(Path(__file__).parents[2] / "robustness" / "imaging.py"),
    }
    if m["ae_run"]:
        if (Path(m["ae_run"]) / "status.json").is_file():
            cfg["provenance"]["ae_pretraining_condition"] = describe_run(m["ae_run"])["research_run"]["condition"]
        elif old_record:
            cfg["provenance"]["ae_pretraining_condition"] = old_record["config"]["provenance"].get("ae_pretraining_condition")
    if old_record and old_record["config_sha256"] != digest(cfg):
        raise ValueError("Resume configuration, assets, source population or reconstruction code changed")
    if resume and (root / "status.json").is_file() and json.loads((root / "status.json").read_text()).get("state") == "complete":
        _verify_artifacts(root)
        return {"run_dir": str(root), **json.loads((root / "telemetry.json").read_text()), "validation": json.loads((root / "validation_metrics.json").read_text()), "already_complete": True}
    seed_all(cfg["seed"])
    model = build_model(m, task=cfg["task"], initialize=not resume).to(device)
    criterion = CompositeReconstructionLoss(**t["loss"]).to(device)
    size = m["ae"]["image_size"]
    dataset = _EpochDataset(train, cfg["data"]["train_root"], size, training=True, recipe=t["recipe"], seed=cfg["seed"])
    loader = DataLoader(dataset, batch_size=t["batch_size"], shuffle=True, generator=torch.Generator().manual_seed(cfg["seed"]), num_workers=t["workers"], persistent_workers=False)
    if isinstance(model, LatentDetector) and not resume:
        clean = CanonicalDataset(train, cfg["data"]["train_root"], size)
        _fit_latent_normalizer(model, DataLoader(clean, batch_size=t["batch_size"], num_workers=t["workers"]), device)
    groups = [{"name": "autoencoder", "params": model.parameters(), "lr": t["lr_ae"]}] if cfg["task"] == "ae" else model.parameter_groups(t["lr_ae"], t["lr_backbone"], t["lr_head"])
    optimizer = torch.optim.AdamW(groups, weight_decay=t["weight_decay"])

    def loss_step(current, batch, state):
        x, target, labels = batch["image"].to(device), batch["clean"].to(device), batch["label"].to(device)
        details = current.forward_details(x)
        if cfg["task"] == "ae":
            if labels.any():
                raise ValueError("A fake image reached genuine-only AE training")
            parts = criterion(details["reconstruction"], target, global_step=state["global_step"], **_posterior(details))
            return parts["loss"], {k: v for k, v in parts.items() if k != "loss"}, len(labels)
        classification = F.cross_entropy(details["logits"], labels)
        total, components = classification, {"classification": classification}
        if m["ae_mode"] != "frozen":
            mask = labels.eq(0)
            if mask.any():
                parts = criterion(details["reconstruction"], target, real_mask=mask, global_step=state["global_step"], **_posterior(details))
                # Weight real-only terms across accumulation windows by genuine count.
                total = total + t["reconstruction_weight"] * parts["loss"] * mask.float().mean()
                components.update({"ae_" + k: v for k, v in parts.items()})
            # An all-fake window in recon_finetune leaves AE gradients absent,
            # preventing AdamW momentum/decay from updating it without genuine data.
        return total, components, len(labels)

    def validate(current):
        if cfg["task"] != "ae":
            p = audited_predict(current, val, cfg["data"]["val_root"], image_size=size, device=device, batch_size=t["batch_size"], workers=t["workers"], hash_images=False, use_amp=t["amp"])
            values = summary(p.label, p.p_fake)
            return {key: values[key] for key in ("auc", "f1", "accuracy", "eer", "log_loss")}
        genuine = val[val.label.eq(0)].reset_index(drop=True)
        val_loader = DataLoader(CanonicalDataset(genuine, cfg["data"]["val_root"], size), batch_size=t["batch_size"], num_workers=t["workers"])
        sums, count = {}, 0
        for batch in val_loader:
            x = batch["image"].to(device)
            details = current.forward_details(x)
            parts = criterion(details["reconstruction"], x, **_posterior(details))
            # Model selection always uses reconstruction quality, independent of beta phase.
            parts["loss"] = parts["l1"] + t["loss"]["lambda_ssim"] * parts["ssim_loss"] + t["loss"]["lambda_lpips"] * parts["lpips"]
            for key, value in parts.items():
                sums[key] = sums.get(key, 0.0) + float(value) * len(x)
            count += len(x)
        return {key: value / count for key, value in sums.items()}

    result = fit_model(model, loader, optimizer, loss_step=loss_step, validate=validate, run_dir=root, config=cfg, epochs=t["epochs"], device=device, grad_accum_steps=t["grad_accum_steps"], selection_key="loss" if cfg["task"] == "ae" else "auc", selection_mode="min" if cfg["task"] == "ae" else "max", resume=resume, max_grad_norm=t["max_grad_norm"])
    return _finalize(model, cfg, val, vc, root, result, device)


def _finalize(model, cfg, val, vc, root, result, device):
    t, size = cfg["training"], cfg["model"]["ae"]["image_size"]
    predictions = audited_predict(model, val, cfg["data"]["val_root"], image_size=size, device=device, batch_size=t["batch_size"], workers=t["workers"], hash_images=True, use_amp=t["amp"])
    checkpoint_hash = digest_file(root / "best.pt")
    contract = run_contract(root)
    save_predictions(root / "validation_predictions.csv", predictions, manifest_record=vc, model_sha256=checkpoint_hash, metadata={"score_semantics": cfg["provenance"]["score_semantics"], "input_contract": contract, "input_contract_sha256": digest(contract), "bundle_sha256": contract["bundle_sha256"], "purpose": "source-validation model development"})
    calibration_path = root / "calibration.json"
    if calibration_path.exists():
        calibration = json.loads(calibration_path.read_text())
        if calibration.get("model_sha256") != checkpoint_hash or calibration.get("source_manifest_sha256") != vc["manifest_sha256"] or calibration.get("input_contract_sha256") != digest(contract):
            raise ValueError("Existing calibration does not belong to this model/population")
    else:
        calibration = calibrate(predictions, manifest_record=vc, output=calibration_path, model_sha256=checkpoint_hash, policy=t["threshold_policy"], input_contract=contract)
    metrics = summary(predictions.label, predictions.p_fake, calibration["frame_threshold"])
    dataset_name = vc["dataset"].lower()
    pilot = dataset_name.startswith("min") or dataset_name.endswith("-min")
    metrics.update(purpose="pilot/min-dataset/development" if pilot else "source-validation development", threshold_policy=t["threshold_policy"], score_semantics=cfg["provenance"]["score_semantics"])
    write_json(root / "validation_metrics.json", metrics)
    files = ["run.json", "best.pt", "telemetry.json", "calibration.json", "validation_predictions.csv", "validation_predictions.csv.json", "validation_metrics.json"]
    files.extend(name for name in ("last.pt", "history.json") if (root / name).exists())
    if t["plots"]:
        from src.robustness.plots import write_forensic_plots
        files.extend(write_forensic_plots(predictions, metrics, root, title=cfg["name"] + " source validation", unit="frame"))
    write_json(root / "status.json", {"state": "complete", "config_sha256": digest(cfg), "artifacts": {name: digest_file(root / name) for name in files}, "evidence": "source train/validation only; targets evaluated separately"})
    return {**result, "validation": metrics}


def combine(spatial_run, latent_run, output_dir, *, device="cpu", name="spatial-latent-mean"):
    """Export a standalone fixed probability mean and source-val calibration."""
    root = Path(output_dir)
    if root.exists():
        raise FileExistsError("Use a new mean-ensemble output directory")
    spatial_info, latent_info = describe_run(spatial_run), describe_run(latent_run)
    a, b = (info["run_record"]["config"] for info in (spatial_info, latent_info))
    if a["task"] != "residual" or b["task"] != "latent":
        raise ValueError("Combine requires completed residual and latent runs")
    if a["model"]["ae"]["image_size"] != b["model"]["ae"]["image_size"]:
        raise ValueError("Spatial and latent models must have the same input resolution")
    for key in ("train_manifest_sha256", "val_manifest_sha256"):
        if a["provenance"][key] != b["provenance"][key]:
            raise ValueError("Component source populations differ")
    model = MeanReconstructionEnsemble(load_model(spatial_run, device), load_model(latent_run, device)).eval()
    cfg = {"task": "mean_ensemble", "name": name, "seed": a["seed"], "output_dir": str(root),
           "data": copy.deepcopy(a["data"]), "training": copy.deepcopy(a["training"]),
           "model": {"ae": copy.deepcopy(a["model"]["ae"]), "spatial": a["model"], "latent": b["model"]},
           "provenance": {"score_semantics": "fixed mean of spatial and latent class-1 fake probabilities",
                          "component_sha256": {"spatial": spatial_info["bundle_sha256"], "latent": latent_info["bundle_sha256"]},
                          "component_conditions": {"spatial": spatial_info["research_run"]["condition"], "latent": latent_info["research_run"]["condition"]},
                          "train_manifest_sha256": a["provenance"]["train_manifest_sha256"],
                          "val_manifest_sha256": a["provenance"]["val_manifest_sha256"]}}
    val, vc = load_manifest(cfg["data"]["val_manifest"])
    if vc["manifest_sha256"] != cfg["provenance"]["val_manifest_sha256"]:
        raise ValueError("Component source-validation manifest changed")
    root.mkdir(parents=True)
    record = {"schema": "faceforgery-experimental-v1", "config": cfg, "config_sha256": digest(cfg),
              "seed": cfg["seed"], "software": source_identity(), "selection_key": "fixed_mean", "selection_mode": "none"}
    write_json(root / "run.json", record)
    write_json(root / "status.json", {"state": "running", "config_sha256": digest(cfg)})
    atomic_torch(root / "best.pt", {"schema": record["schema"], "config_sha256": digest(cfg), "state_dict": model.state_dict()})
    started = time.perf_counter()
    write_json(root / "telemetry.json", {"trainable_parameters": 0, "epochs_completed": 0,
                                        "runtime_seconds": time.perf_counter() - started,
                                        "method": "fixed arithmetic probability mean; no learned fusion"})
    return _finalize(model, cfg, val, vc, root, {"run_dir": str(root)}, device)


def predict(run_dir, manifest, root, *, device="cpu", output_dir=None, **kwargs):
    """Use the existing audited predictor/evaluator with the frozen source threshold."""
    model = load_model(run_dir, device)
    path = Path(run_dir)
    record = json.loads((path / "run.json").read_text())
    cfg = record["config"]
    arguments = {"image_size": cfg["model"]["ae"]["image_size"], "device": device,
                 "batch_size": cfg["training"]["batch_size"], "workers": cfg["training"]["workers"],
                 "use_amp": cfg["training"]["amp"], **kwargs}
    if output_dir is not None:
        return evaluate(model, manifest, root, output_dir, checkpoint_path=path / "best.pt", calibration_path=path / "calibration.json", research_run=evaluation_identity(record), input_contract=run_contract(path), **arguments)
    frame, _ = load_manifest(manifest)
    return audited_predict(model, frame, root, **arguments)
