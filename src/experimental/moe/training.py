"""Source-validation-only router fit and mandatory frozen fusion comparisons."""

from __future__ import annotations

import copy
import json
import math
import warnings
from pathlib import Path

import numpy as np
import torch
from filelock import FileLock
from sklearn.exceptions import ConvergenceWarning
from sklearn.linear_model import LogisticRegression
from torch.nn import functional as F
from torch.utils.data import DataLoader

from src.experimental.runtime import fit_model
from src.robustness.artifacts import save_predictions
from src.robustness.engine import atomic_torch, seed_all
from src.robustness.inference import calibrate, evaluation_report
from src.robustness.manifests import load_manifest, save_manifest
from src.robustness.provenance import digest, digest_file, source_identity, write_json
from src.robustness.statistics import summary
from .data import FusionDataset, fit_standardizer, open_sources, split_validation
from .models import FrozenLateFusion, simple_fusion

MODEL_DEFAULTS = {"hidden_dim": 64, "temperature": 1.0, "train_routing": "soft", "train_top_k": 2,
                  "inference_routing": "soft", "inference_top_k": 1, "expert_dropout": 0.1}
SCORE_POLICY = "convex weighted mean of frozen p_fake; FP32 router softmax at fixed temperature"
SCHEMA = "faceforgery-frozen-moe-v1"


def normalize_config(config):
    config = copy.deepcopy(config)
    if config.get("task") != "moe":
        raise ValueError("Frozen fusion fit requires task=moe")
    if set(config) - {"task", "name", "scope", "seed", "output_dir", "data", "model", "training"}:
        raise ValueError("Unknown MoE configuration field")
    if not config.get("name") or not config.get("output_dir"):
        raise ValueError("Declare MoE name and output_dir")
    data = config["data"]
    if set(data) - {"experts", "split_seed", "fit_fraction", "required_roles"}:
        raise ValueError("Unknown MoE data field; only certified source-val experts are accepted")
    if not isinstance(data.get("experts"), list) or len(data["experts"]) < 2:
        raise ValueError("Declare at least two frozen experts")
    required = set(data.get("required_roles", ["srm", "rgb", "reconstruction"]))
    if not required <= {expert.get("role") for expert in data["experts"]}:
        raise ValueError("Missing mandatory forensic expert role")
    model = config.get("model", {})
    if set(model) - set(MODEL_DEFAULTS):
        raise ValueError("Unknown MoE router field")
    config["model"] = {**MODEL_DEFAULTS, **model}
    training = config.setdefault("training", {})
    known = {"device", "epochs", "batch_size", "workers", "cpu_threads", "amp", "lr", "weight_decay",
             "grad_accum_steps", "max_grad_norm", "load_balance_weight", "resume", "eval_batch_size",
             "threshold_policy", "scheduler_patience", "early_stop_patience", "stacking_c", "stacking_max_iter"}
    if set(training) - known:
        raise ValueError("Unknown MoE training field")
    if int(training.get("workers", 0)) != 0 or not 1 <= int(training.get("cpu_threads", 2)) <= 2:
        raise ValueError("Cached fusion uses workers=0 and one or two CPU threads")
    if float(training.get("load_balance_weight", 0.01)) < 0:
        raise ValueError("Load-balancing weight must be nonnegative")
    if training.get("threshold_policy", "youden") not in {"youden", "balanced_accuracy"}:
        raise ValueError("Unknown validation threshold policy")
    return config


def input_contract(document, checksum):
    return {"kind": "frozen-forensic-late-fusion", "bundle_sha256": checksum,
            "checkpoint_class1": "fake", "score": SCORE_POLICY, "model": document["model"],
            "expert_contracts": document["experts"], "feature_input": "expert penultimate features plus clipped log-odds",
            "normalization": "mean/std fitted on val_fit only; frozen checkpoint buffers",
            "split": document["validation_split"]}


def baseline_contract(document, checksum, method, method_checksum):
    policies = {"mean": "arithmetic mean of frozen p_fake",
                "geometric": "exp(mean(log(clamp(p_fake, min=1e-9)))); repository scalar geometric fusion",
                "geometric_binary": "sigmoid(mean(logit(clamp(p_fake,1e-6,1-1e-6)))); normalized binary geometric fusion",
                "logistic": "sigmoid(L2 logistic stacking on frozen p_fake); coefficient/intercept fitted on val_fit only"}
    if method.startswith("expert_"):
        names = [expert["name"] for expert in document["experts"]]
        if method[len("expert_"):] not in names:
            raise ValueError("Unknown single expert comparison")
        policy = "unchanged certified p_fake from declared frozen expert"
    else:
        if method not in policies:
            raise ValueError("Unknown frozen fusion comparison")
        policy = policies[method]
    return {**input_contract(document, checksum), "method": method,
            "method_bundle_sha256": method_checksum, "score": policy}


def predict_router(model, aligned, indices, *, device="cpu", batch_size=512):
    if batch_size < 1:
        raise ValueError("Fusion inference batch size must be positive")
    model.eval()
    scores, weights = [], []
    with torch.inference_mode():
        for start in range(0, len(indices), batch_size):
            x, p = aligned.block(indices[start:start + batch_size])
            output = model(torch.from_numpy(x).to(device), torch.from_numpy(p).to(device))
            scores.extend(output["p_fake"].cpu().tolist())
            weights.append(output["weights"].cpu().numpy())
    return np.asarray(scores, dtype=np.float32), np.concatenate(weights)


def routing_report(weights, names):
    return {"mean_weight": {name: float(value) for name, value in zip(names, weights.mean(0))},
            "nonzero_fraction": {name: float(value) for name, value in zip(names, (weights > 0).mean(0))},
            "top1_fraction": {name: float((weights.argmax(-1) == i).mean()) for i, name in enumerate(names)},
            "mean_entropy": float(-(weights * np.log(np.clip(weights, 1e-6, 1))).sum(-1).mean())}


def fit(config):
    config = normalize_config(config)
    root = Path(config["output_dir"])
    root.mkdir(parents=True, exist_ok=True)
    with FileLock(str(root / ".moe-fit.lock"), timeout=0):
        return _fit(config, root)


def _fit(config, root):
    training, data = config["training"], config["data"]
    resume = bool(training.get("resume", False))
    if (root / "artifact.json").exists():
        raise FileExistsError("Completed MoE/calibration are frozen; use a new run directory")
    if not resume and any((root / name).exists() for name in ("val_fit.csv", "val_select.csv", "last.pt")):
        raise FileExistsError("Partial MoE run exists; use explicit resume or a new output")
    config = copy.deepcopy(config)
    config["training"].pop("resume", None)  # Invocation control is not training identity.
    seed, device = int(config.get("seed", 42)), training.get("device", "cpu")
    torch.set_num_threads(int(training.get("cpu_threads", 2)))
    seed_all(seed)
    aligned = open_sources(data["experts"])
    fit_indices, select_indices, split = split_validation(aligned.frame,
        seed=int(data.get("split_seed", 42)), fit_fraction=float(data.get("fit_fraction", 0.5)))
    fit_frame, select_frame = aligned.frame.iloc[fit_indices].copy(), aligned.frame.iloc[select_indices].copy()
    fit_frame["fusion_subset"], select_frame["fusion_subset"] = "val_fit", "val_select"
    source = {"expert_source_identities": [expert.identity for expert in aligned.experts], "fusion_split": split}
    certificates = {}
    for subset, frame in (("val_fit", fit_frame), ("val_select", select_frame)):
        path = root / f"{subset}.csv"
        if resume:
            old, certificate = load_manifest(path)
            if not old.equals(frame.reset_index(drop=True)) or certificate.get("expert_source_identities") != source["expert_source_identities"]:
                raise ValueError("Resumed fusion source population or frozen expert cache changed")
        else:
            certificate = save_manifest(frame, path, source)
        certificates[subset] = certificate
    mean, std = fit_standardizer(aligned, fit_indices)
    model_kwargs = {"input_dim": aligned.input_dim, "num_experts": len(aligned.experts), **config["model"]}
    model = FrozenLateFusion(**model_kwargs, mean=mean, std=std)
    loader = DataLoader(FusionDataset(aligned, fit_indices), batch_size=int(training.get("batch_size", 32)),
                        shuffle=True, num_workers=0, generator=torch.Generator().manual_seed(seed))
    optimizer = torch.optim.AdamW(model.parameters(), lr=float(training.get("lr", 1e-3)),
                                 weight_decay=float(training.get("weight_decay", 1e-4)))
    coefficient = float(training.get("load_balance_weight", 0.01))
    if not math.isfinite(coefficient):
        raise ValueError("Load balance coefficient must be finite")

    def loss_step(current, batch, state):
        del state
        x, p, y = (value.to(device) for value in batch)
        output = current(x, p)
        classification = F.nll_loss(output["logits"], y.long())
        loss = classification + coefficient * output["load_balance"]
        return loss, {"classification": classification, "load_balance": output["load_balance"],
                      "routing_entropy": output["entropy"].mean()}, len(y)

    def validate(current):
        p, _ = predict_router(current, aligned, select_indices, device=device,
                              batch_size=int(training.get("eval_batch_size", 512)))
        return summary(select_frame.label, p)

    result = fit_model(model, loader, optimizer, loss_step=loss_step, validate=validate,
        run_dir=root, config={**config, "provenance": source}, epochs=int(training.get("epochs", 10)),
        device=device, grad_accum_steps=int(training.get("grad_accum_steps", 1)), selection_key="auc", selection_mode="max",
        resume=resume, max_grad_norm=float(training.get("max_grad_norm", 1.0)))
    _, fit_scores = aligned.block(fit_indices)
    _, select_scores = aligned.block(select_indices)
    with warnings.catch_warnings():
        warnings.simplefilter("error", ConvergenceWarning)
        stacker = LogisticRegression(C=float(training.get("stacking_c", 1.0)),
            max_iter=int(training.get("stacking_max_iter", 1000)), random_state=seed, solver="lbfgs")
        stacker.fit(fit_scores, fit_frame.label)
    if stacker.classes_.tolist() != [0, 1]:
        raise ValueError("Logistic stacking class orientation mismatch")
    atomic_torch(root / "stacker.pt", {"coefficient": torch.from_numpy(stacker.coef_[0].copy()).double(),
        "intercept": torch.tensor(float(stacker.intercept_[0]), dtype=torch.float64), "classes": [0, 1]})
    names = [expert.contract["name"] for expert in aligned.experts]
    document = {"schema": SCHEMA, "label_convention": "fake-is-1", "model": model_kwargs,
        "score_policy": SCORE_POLICY, "experts": [expert.contract for expert in aligned.experts],
        "expert_source_identities": source["expert_source_identities"], "validation_split": split,
        "calibration_manifest_sha256": certificates["val_select"]["manifest_sha256"],
        "calibration_sample_ids_sha256": split["val_select_sample_ids_sha256"],
        "scope": config.get("scope", "development"), "seed": seed, "software": source_identity(),
        "expert_selection_warning": "Inherited expert training and full-source-val model selection may overlap fusion data; independent fusion fit is required for full-scale claims.",
        "implementation": {file.name: digest_file(file) for file in Path(__file__).parent.glob("*.py")},
        "files": {name: digest_file(root / name) for name in
                  ("best.pt", "stacker.pt", "val_fit.csv", "val_fit.csv.json", "val_select.csv", "val_select.csv.json")}}
    write_json(root / "artifact.json", document)
    checksum = digest_file(root / "artifact.json")
    router_scores, weights = predict_router(model, aligned, select_indices, device=device)
    variants = {"router": router_scores, "mean": simple_fusion(torch.from_numpy(select_scores), "mean").numpy(),
                "geometric": simple_fusion(torch.from_numpy(select_scores), "geometric").numpy(),
                "geometric_binary": simple_fusion(torch.from_numpy(select_scores), "geometric_binary").numpy(),
                "logistic": stacker.predict_proba(select_scores)[:, 1]}
    variants.update({f"expert_{name}": select_scores[:, index] for index, name in enumerate(names)})
    comparisons = {}
    contract = input_contract(document, checksum)
    for method, probabilities in variants.items():
        output = root if method == "router" else root / "comparisons" / method
        output.mkdir(parents=True, exist_ok=True)
        method_checksum = checksum
        method_contract = contract
        if method != "router":
            write_json(output / "method.json", {"schema": "faceforgery-fusion-baseline-v1",
                "parent_artifact_sha256": checksum, "method": method, "scope": document["scope"]})
            method_checksum = digest_file(output / "method.json")
            method_contract = baseline_contract(document, checksum, method, method_checksum)
        predicted = select_frame.copy()
        predicted["p_fake"] = probabilities
        save_predictions(output / "validation_predictions.csv", predicted,
            manifest_record=certificates["val_select"], model_sha256=method_checksum,
            metadata={"input_contract": method_contract, "input_contract_sha256": digest(method_contract),
                      "fit_subset": "val_fit", "selection_subset": "val_select", "scope": document["scope"]})
        calibration = calibrate(predicted, manifest_record=certificates["val_select"], output=output / "calibration.json",
            model_sha256=method_checksum, policy=training.get("threshold_policy", "youden"), input_contract=method_contract)
        metrics = evaluation_report(predicted, calibration, method_checksum)
        write_json(output / "metrics.json", metrics)
        comparisons[method] = metrics["frame"]
    correlations = np.corrcoef(select_scores.T)
    report = {"scope": document["scope"], "training": result, "validation_split": split,
              "counts": {"val_fit": len(fit_frame), "val_select": len(select_frame)}, "comparisons": comparisons,
              "routing": routing_report(weights, names), "expert_names": names,
              "expert_prediction_correlation": [[float(v) if math.isfinite(v) else None for v in row] for row in correlations],
              "selection_statement": "All method comparisons are development val_select scores; no test target has been accessed."}
    write_json(root / "comparison.json", report)
    write_json(root / "status.json", {"state": "complete", "artifact_sha256": checksum,
        "artifacts": {name: digest_file(root / name) for name in
                      ("artifact.json", "calibration.json", "metrics.json", "comparison.json", "validation_predictions.csv", "validation_predictions.csv.json")}})
    return root
