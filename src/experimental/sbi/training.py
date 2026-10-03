"""Matched SBI controls and HF adaptation with offline, complete run artifacts."""
from __future__ import annotations

import copy
from dataclasses import asdict
import json
from pathlib import Path

import torch
from torch import nn
from torch.nn import functional as F
from torch.utils.data import DataLoader

from src.experimental.runtime import fit_model
from src.robustness.artifacts import save_predictions
from src.robustness.engine import seed_all
from src.robustness.inference import calibrate, predict, prediction_contract
from src.robustness.legacy_encoding import encode_legacy_tensor
from src.robustness.manifests import load_manifest
from src.robustness.provenance import digest, digest_file, write_json
from src.robustness.statistics import grouped_auc_interval, summary
from .data import SBIDataset


def normalize_config(config):
    cfg = copy.deepcopy(config)
    required = {"name", "seed", "output_dir", "data", "model", "training"}
    if set(cfg) != required:
        raise ValueError(f"SBI config fields must be exactly {sorted(required)}")
    if set(cfg["data"]) != {"train_manifest", "val_manifest", "train_root", "val_root", "landmarks"}:
        raise ValueError("Declare source manifests, roots and offline landmarks")
    defaults = {"architecture": "mobilenet_v3_large", "mode": "srm", "image_size": 224,
                "initialization": "imagenet", "weights": None, "train_backbone": True}
    training = {"arm": "sbi", "epochs": 3, "batch_size": 8, "workers": 0, "grad_accum_steps": 4,
                "lr_backbone": 1e-4, "lr_head": 1e-3, "weight_decay": 1e-4, "amp": True,
                "early_stop_patience": 3, "scheduler_patience": 2, "failure_policy": "error",
                "mask": {}, "cache_images": False, "post_augment": True, "bootstrap_draws": 200}
    if set(cfg["model"]) - set(defaults) or set(cfg["training"]) - set(training):
        raise ValueError("Unknown SBI model or training option")
    cfg["model"] = {**defaults, **cfg["model"]}
    cfg["training"] = {**training, **cfg["training"]}
    m, t = cfg["model"], cfg["training"]
    if m["initialization"] not in {"imagenet", "hf_mffi", "scratch"} or m["mode"] not in {"none", "srm"}:
        raise ValueError("Declare initialization and RGB/legacy SRM representation")
    if m["initialization"] != "scratch" and not m["weights"]:
        raise ValueError("Explicit local weights are required; training never downloads")
    for key in ("epochs", "batch_size", "grad_accum_steps", "early_stop_patience"):
        if not isinstance(t[key], int) or isinstance(t[key], bool) or t[key] < 1:
            raise ValueError(f"Invalid SBI integer {key}")
    if t["workers"] < 0 or m["image_size"] < 32 or t["arm"] not in {"sbi", "mffi", "mixed"}:
        raise ValueError("Invalid SBI data settings")
    if min(t["lr_backbone"], t["lr_head"]) <= 0 or t["weight_decay"] < 0:
        raise ValueError("Invalid SBI optimizer settings")
    for obj, key in [(m, "train_backbone"), (t, "amp"), (t, "cache_images"), (t, "post_augment")]:
        if not isinstance(obj[key], bool):
            raise ValueError(f"Use an explicit boolean for {key}")
    return cfg


def _generic(spec, weights=None):
    from torchvision.models import mobilenet_v3_large, resnet18
    if spec["architecture"] not in {"mobilenet_v3_large", "resnet18"}:
        raise ValueError("Generic SBI supports mobilenet_v3_large or resnet18")
    net = {"mobilenet_v3_large": mobilenet_v3_large, "resnet18": resnet18}[spec["architecture"]](weights=None)
    if weights:
        net.load_state_dict(torch.load(weights, map_location="cpu", weights_only=True), strict=True)
    if spec["architecture"] == "mobilenet_v3_large":
        conv, parent, key = net.features[0][0], net.features[0], 0
        net.classifier[-1] = nn.Linear(net.classifier[-1].in_features, 2)
    else:
        conv, parent, key = net.conv1, net, "conv1"
        net.fc = nn.Linear(net.fc.in_features, 2)
    if spec["mode"] == "srm":
        extended = nn.Conv2d(6, conv.out_channels, conv.kernel_size, conv.stride, conv.padding, bias=conv.bias is not None)
        with torch.no_grad():
            extended.weight.zero_()
            extended.weight[:, :3].copy_(conv.weight)
            if conv.bias is not None:
                extended.bias.copy_(conv.bias)
        if isinstance(key, int):
            parent[key] = extended
        else:
            setattr(parent, key, extended)
    return net


def _hf_restore(spec):
    from src.models.registry import MODEL_REGISTRY
    from src.pipelines.config import TrainingConfig
    if spec["family"] == "clip":
        from transformers import CLIPVisionConfig, CLIPVisionModel
        from src.models.clip import CLIPClassifier
        backbone = CLIPVisionModel(CLIPVisionConfig(**spec["vision_config"]))
        backbone.set_attn_implementation("eager")
        return CLIPClassifier(backbone, spec["config"]["dropout"])
    values = {**spec["config"], "regime": "scratch", "allow_pretrained": False}
    return MODEL_REGISTRY[spec["family"]].build(TrainingConfig(**values))


class SBIClassifier(nn.Module):
    def __init__(self, network, mode, train_backbone=True):
        super().__init__()
        self.network, self.mode, self.train_backbone = network, mode, train_backbone
        for name, param in network.named_parameters():
            param.requires_grad_(train_backbone or name.startswith(("classifier.", "fc.", "head.")))

    def train(self, mode=True):
        super().train(mode)
        if not self.train_backbone:
            self.network.eval()
            for name in ("classifier", "fc", "head"):
                if hasattr(self.network, name):
                    getattr(self.network, name).train(mode)
        return self

    def forward(self, raw):
        return self.network(encode_legacy_tensor(raw, self.mode, 6 if self.mode == "srm" else 3))

    def parameter_groups(self, training):
        groups = {"head": [], "backbone": []}
        for name, param in self.network.named_parameters():
            if param.requires_grad:
                groups["head" if name.startswith(("classifier.", "fc.", "head.")) else "backbone"].append(param)
        return [{"params": values, "name": name, "lr": training["lr_"+name]} for name, values in groups.items() if values]


def _initialize(config):
    m = config["model"]
    if m["initialization"] == "hf_mffi":
        from src.pipelines.checkpoints import config_from_run, run_from_checkpoint
        run = run_from_checkpoint(m["weights"])
        if run.model_family not in {"mobilenet", "resnet", "dino", "clip"}:
            raise ValueError("HF SBI currently supports verified MobileNet/ResNet/DINO/CLIP")
        source = config_from_run(run)
        if source.image_size != m["image_size"] or source.fourier_mode != m["mode"]:
            raise ValueError("HF input representation or image size mismatch")
        spec = {"kind": "hf", "family": run.model_family, "config": asdict(source)}
        if run.model_family == "clip":
            from transformers import CLIPVisionConfig
            from src.models._regime import uses_pretraining
            if uses_pretraining(source, "clip"):
                # The legacy factory explicitly uses openai/clip-vit-base-patch16.
                vision = CLIPVisionConfig(image_size=source.image_size, patch_size=16, num_channels=source.in_channels)
            else:
                vision = CLIPVisionConfig(image_size=source.image_size, patch_size=source.patch_size,
                    num_channels=source.in_channels, hidden_size=source.hidden_size,
                    num_hidden_layers=source.num_hidden_layers, num_attention_heads=source.num_attention_heads,
                    intermediate_size=source.hidden_size*4, projection_dim=source.projection_dim)
            spec["vision_config"] = vision.to_dict()
        network = _hf_restore(spec)
        state = torch.load(run.weights_path, map_location="cpu", weights_only=True)
        if state and all(str(key).startswith("module.") for key in state):
            state = {key.removeprefix("module."): value for key, value in state.items()}
        network.load_state_dict(state, strict=True)
    else:
        spec = {"kind": "generic", "architecture": m["architecture"], "mode": m["mode"]}
        network = _generic(spec, m["weights"] if m["initialization"] == "imagenet" else None)
    return SBIClassifier(network, m["mode"], m["train_backbone"]), spec


def _rebuild(config):
    spec, m = config["network_spec"], config["model"]
    network = _generic(spec) if spec["kind"] == "generic" else _hf_restore(spec)
    return SBIClassifier(network, m["mode"], m["train_backbone"])


def _contract(root, cfg):
    contract = prediction_contract(cfg["model"]["image_size"])
    contract.update(family="sbi", representation=cfg["model"]["mode"],
                    srm_on_normalized=cfg["model"]["mode"] == "srm",
                    bundle_sha256=digest({"config_sha256": digest(cfg), "best_sha256": digest_file(root / "best.pt")}))
    return contract


def describe_run(run_dir):
    root = Path(run_dir)
    status = json.loads((root / "status.json").read_text())
    required = {"run.json", "best.pt", "calibration.json", "validation_predictions.csv"}
    if status.get("state") != "complete" or not required <= set(status.get("artifacts", {})):
        raise ValueError("SBI run is incomplete")
    for name, value in status["artifacts"].items():
        if Path(name).name != name or digest_file(root / name) != value:
            raise ValueError("SBI artifact changed")
    record = json.loads((root / "run.json").read_text())
    return {"checkpoint_path": root / "best.pt", "input_contract": _contract(root, record["config"]),
            "research_run": record, "image_size": record["config"]["model"]["image_size"]}


def load_model(run_dir, device="cpu"):
    descriptor = describe_run(run_dir)
    model = _rebuild(descriptor["research_run"]["config"])
    state = torch.load(descriptor["checkpoint_path"], map_location="cpu", weights_only=True)
    model.load_state_dict(state["state_dict"], strict=True)
    return model.to(device).eval()


def fit(config, *, device="cpu", resume=False):
    cfg = normalize_config(config)
    root, data, t = Path(cfg["output_dir"]), cfg["data"], cfg["training"]
    train, tc = load_manifest(data["train_manifest"])
    val, vc = load_manifest(data["val_manifest"])
    if set(train.split) != {"train"} or set(val.split) != {"val"}:
        raise ValueError("Only source train/val are allowed")
    for key in ("sample_id", "group_id", "source_id", "sha256"):
        if key in train and key in val and ((set(train[key]) & set(val[key])) - {"", "unknown"}):
            raise ValueError(f"Train/validation overlap in {key}")
    ds = SBIDataset(train, data["train_root"], data["landmarks"], manifest_sha256=tc["manifest_sha256"],
                    image_size=cfg["model"]["image_size"], arm=t["arm"], seed=cfg["seed"],
                    failure_policy=t["failure_policy"], mask=t["mask"], cache_images=t["cache_images"], post_augment=t["post_augment"])
    seed_all(cfg["seed"])
    if resume:
        old = json.loads((root / "run.json").read_text())["config"]
        cfg["network_spec"] = old["network_spec"]
        model = _rebuild(cfg)
        weight_hash = old["provenance"]["initialization_sha256"]
        if cfg["model"]["weights"] and Path(cfg["model"]["weights"]).exists() and digest_file(cfg["model"]["weights"]) != weight_hash:
            raise ValueError("Initialization asset changed")
    else:
        model, cfg["network_spec"] = _initialize(cfg)
        weight_hash = digest_file(cfg["model"]["weights"]) if cfg["model"]["weights"] else None
    cfg["provenance"] = {"train_manifest": tc, "val_manifest": vc, "cohort": ds.cohort,
                         "initialization_sha256": weight_hash, "label_convention": "fake-is-1",
                         "selection": "MFFI source validation including fakes; never target selection",
                         "prior_mffi_fake_exposure": cfg["model"]["initialization"] == "hf_mffi",
                         "code_sha256": {p.name: digest_file(p) for p in sorted(Path(__file__).parent.glob("*.py"))}}
    if resume and digest(old) != digest(cfg):
        raise ValueError("SBI resume identity changed")
    if resume and json.loads((root / "status.json").read_text()).get("state") == "complete":
        describe_run(root)
        return {"run_dir": str(root), "already_complete": True}
    loader = DataLoader(ds, batch_size=t["batch_size"], shuffle=True, generator=torch.Generator().manual_seed(cfg["seed"]), num_workers=t["workers"])
    optimizer = torch.optim.AdamW(model.parameter_groups(t), weight_decay=t["weight_decay"])
    def step(current, batch, state):
        labels = batch["label"].to(device)
        loss = F.cross_entropy(current(batch["image"].to(device)), labels)
        return loss, {"cross_entropy": loss}, len(labels)
    def predictions(current):
        return predict(current, val, data["val_root"], image_size=cfg["model"]["image_size"], device=device,
                       batch_size=t["batch_size"], workers=t["workers"], use_amp=t["amp"])
    def validation(current):
        p = predictions(current)
        values = summary(p.label, p.p_fake)
        return {k: values[k] for k in ("auc", "f1", "accuracy", "eer", "log_loss")}
    result = fit_model(model, loader, optimizer, loss_step=step, validate=validation, run_dir=root, config=cfg,
                       epochs=t["epochs"], device=device, resume=resume, grad_accum_steps=t["grad_accum_steps"], selection_key="auc", selection_mode="max")
    p, checksum = predictions(model), digest_file(root / "best.pt")
    save_predictions(root / "validation_predictions.csv", p, manifest_record=vc, model_sha256=checksum,
                     metadata={"purpose": "source-validation development", "cohort": ds.cohort})
    contract = _contract(root, cfg)
    calibration = calibrate(p, manifest_record=vc, output=root / "calibration.json", model_sha256=checksum,
                            policy="youden", input_contract=contract)
    metrics = summary(p.label, p.p_fake, calibration["frame_threshold"])
    metrics.update(purpose="pilot/min-dataset/development" if "min" in tc["dataset"].lower() else "source-validation development", threshold_policy="youden")
    metrics["auc_ci"] = grouped_auc_interval(p, draws=t["bootstrap_draws"], seed=cfg["seed"])
    write_json(root / "validation_metrics.json", metrics)
    write_json(root / "sbi_cohort.json", ds.cohort)
    files = ["run.json", "best.pt", "last.pt", "history.json", "telemetry.json", "calibration.json", "validation_predictions.csv", "validation_predictions.csv.json", "validation_metrics.json", "sbi_cohort.json"]
    write_json(root / "status.json", {"state": "complete", "config_sha256": digest(cfg),
                                    "artifacts": {name: digest_file(root / name) for name in files}})
    return {**result, "validation": metrics}
