"""Portable cached-expert callback for the shared four-target evaluator."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import torch

from src.robustness.manifests import validate
from src.robustness.provenance import contained, digest, digest_file
from src.robustness.statistics import checked_predictions
from .data import open_sources
from .models import FrozenLateFusion, simple_fusion
from .training import SCHEMA, SCORE_POLICY, baseline_contract, input_contract, predict_router


def read_artifact(run_dir):
    root = Path(run_dir)
    document = json.loads((root / "artifact.json").read_text())
    if document.get("schema") != SCHEMA or document.get("score_policy") != SCORE_POLICY or document.get("label_convention") != "fake-is-1":
        raise ValueError("Unsupported frozen-MoE artifact or score contract")
    required = {"best.pt", "stacker.pt", "val_fit.csv", "val_fit.csv.json", "val_select.csv", "val_select.csv.json"}
    if set(document.get("files", {})) != required:
        raise ValueError("Incomplete frozen-MoE inventory")
    for name, checksum in document["files"].items():
        if digest_file(contained(root, name)) != checksum:
            raise ValueError(f"Frozen-MoE dependency missing or changed: {name}")
    for name, checksum in document.get("implementation", {}).items():
        if digest_file(contained(Path(__file__).parent, name)) != checksum:
            raise ValueError(f"MoE implementation differs from fitted score contract: {name}")
    status = json.loads((root / "status.json").read_text())
    checksum = digest_file(root / "artifact.json")
    if status.get("state") != "complete" or status.get("artifact_sha256") != checksum:
        raise ValueError("MoE run is incomplete or artifact differs from its completion certificate")
    calibration = json.loads((root / "calibration.json").read_text())
    if (calibration.get("model_sha256") != checksum or calibration.get("input_contract") != input_contract(document, checksum)
            or calibration.get("source_manifest_sha256") != document["calibration_manifest_sha256"]):
        raise ValueError("MoE frozen calibration differs from bundle or val_select population")
    return document


def describe_run(run_dir, method="router"):
    """Read and verify metadata only; never instantiate a model in dry-run."""
    root = Path(run_dir)
    document = read_artifact(root)
    checksum = digest_file(root / "artifact.json")
    checkpoint_path, contract = root / "artifact.json", input_contract(document, checksum)
    if method != "router":
        # Validate method before constructing its relative artifact path.
        baseline_contract(document, checksum, method, "pending")
        checkpoint_path = root / "comparisons" / method / "method.json"
        declared = json.loads(checkpoint_path.read_text())
        if declared.get("parent_artifact_sha256") != checksum or declared.get("method") != method:
            raise ValueError("Comparison artifact does not belong to frozen router bundle")
        contract = baseline_contract(document, checksum, method, digest_file(checkpoint_path))
        calibration = json.loads((checkpoint_path.parent / "calibration.json").read_text())
        if calibration.get("model_sha256") != digest_file(checkpoint_path) or calibration.get("input_contract") != contract:
            raise ValueError("Comparison calibration differs from its method bundle")
    condition = {"family": "moe", "method": method,
                 "expert_seed_policy": document.get("expert_seed_policy", "fixed"),
                 "expert_conditions": document.get("expert_conditions", {
                     "experts": document["experts"], "expert_source_identities": document["expert_source_identities"]}),
                 "validation_split": document["validation_split"],
                 "calibration_manifest_sha256": document["calibration_manifest_sha256"],
                 "scope": document["scope"], "implementation": document["implementation"]}
    if method == "router":
        condition.update(model=document["model"], training_policy=document["training_policy"], score_policy=SCORE_POLICY)
    elif method == "logistic":
        condition["stacking_policy"] = {key: document["training_policy"].get(key, default)
                                       for key, default in (("stacking_c", 1.0), ("stacking_max_iter", 1000))}
    else:
        condition["score_policy"] = contract["score"]
    return {"checkpoint_path": checkpoint_path, "input_contract": contract,
            "research_run": {"name": f"{document['name']}/{method}", "family": "moe", "seed": document["seed"], "scope": document["scope"],
                             "condition": condition, "condition_sha256": digest(condition),
                             "artifact_sha256": checksum, "method": method,
                             "realization": {"experts": document["experts"],
                                             "expert_source_identities": document["expert_source_identities"],
                                             "expert_seeds": document.get("expert_seed_realizations"),
                                             "fusion_seed": document["seed"]}}, "image_size": None,
            "calibration_manifest_sha256": document["calibration_manifest_sha256"],
            "calibration_sample_ids_sha256": document["calibration_sample_ids_sha256"]}


def source_specs(document, sources):
    if len(sources) != len(document["experts"]):
        raise ValueError("Query source count differs from frozen expert order")
    specs = []
    for source, contract in zip(sources, document["experts"]):
        if isinstance(source, (str, Path)):
            source = {"name": contract["name"], "role": contract["role"], "kind": contract["kind"], "path": str(source)}
            if contract["kind"] == "predictions":
                source["feature_columns"] = contract["feature_columns"]
        specs.append(source)
    return specs


def validate_sources(run_dir, sources, frame=None, root=None):
    """Verify query caches and optional image bytes without model construction."""
    document = read_artifact(run_dir)
    aligned = open_sources(source_specs(document, sources), frame=frame, expected_contracts=document["experts"])
    if root is not None:
        for row in aligned.frame.itertuples(index=False):
            if digest_file(contained(root, row.img_name)) != row.image_sha256:
                raise ValueError("Query image bytes differ from frozen expert cache")
    return aligned


class MoEPredictor:
    def __init__(self, run_dir, sources, device="cpu", method="router"):
        root = Path(run_dir)
        self.document = read_artifact(root)
        self.sources = source_specs(self.document, sources)
        describe_run(run_dir, method)  # Also bind a baseline's own frozen calibration.
        self.device, self.method = torch.device(device), method
        self.model = FrozenLateFusion(**self.document["model"])
        state = torch.load(root / "best.pt", map_location="cpu", weights_only=True)
        self.model.load_state_dict(state["state_dict"], strict=True)
        self.model.to(self.device).eval()
        self.stacker = torch.load(root / "stacker.pt", map_location="cpu", weights_only=True)
        if self.stacker.get("classes") != [0, 1]:
            raise ValueError("Stacker class orientation mismatch")
        self.bundle_metadata = self.document

    def __call__(self, frame, root, *, batch_size=512, workers=0, positive_class="fake", hash_images=True,
                 use_amp=True, amp=None, device=None, image_size=None, mode=None, in_channels=None, **kwargs):
        del use_amp, amp, image_size, in_channels
        if kwargs or workers != 0 or positive_class != "fake" or mode is not None:
            raise ValueError("Frozen MoE requires canonical fake-is-1 cached inference and workers=0")
        if device is not None and torch.device(device) != self.device:
            raise ValueError("Suite device differs from loaded MoE predictor")
        if batch_size < 1 or not hash_images:
            raise ValueError("MoE uses positive batch size and mandatory image-byte verification")
        frame = validate(frame, require_both=False)
        aligned = open_sources(self.sources, frame=frame, expected_contracts=self.document["experts"])
        for row in aligned.frame.itertuples(index=False):
            if digest_file(contained(root, row.img_name)) != row.image_sha256:
                raise ValueError("Query image bytes differ from frozen expert cache")
        indices = np.arange(len(frame))
        if self.method == "router":
            scores, weights = predict_router(self.model, aligned, indices, device=self.device, batch_size=batch_size)
        else:
            scores, weights = [], None
            for start in range(0, len(frame), batch_size):
                p = aligned.score_block(indices[start:start + batch_size])
                p = torch.from_numpy(p)
                if self.method == "logistic":
                    score = torch.sigmoid(p.double() @ self.stacker["coefficient"] + self.stacker["intercept"])
                elif self.method.startswith("expert_"):
                    names = [expert["name"] for expert in self.document["experts"]]
                    score = p[:, names.index(self.method[len("expert_"):])]
                else:
                    score = simple_fusion(p, self.method)
                scores.extend(score.tolist())
        result = aligned.frame.copy()
        result["p_fake"] = scores
        if weights is not None:
            for index, expert in enumerate(self.document["experts"]):
                result[f"route_{expert['name']}"] = weights[:, index]
        return checked_predictions(result)


def load_predictor(run_dir, sources, device="cpu", method="router"):
    return MoEPredictor(run_dir, sources, device, method)


def validate_calibration_population(run_dir, frame, manifest_record):
    document = read_artifact(run_dir)
    if (manifest_record["manifest_sha256"] != document["calibration_manifest_sha256"]
            or digest(sorted(frame.sample_id.astype(str))) != document["calibration_sample_ids_sha256"]):
        raise ValueError("MoE calibration requires the exact frozen val_select population, excluding val_fit")
