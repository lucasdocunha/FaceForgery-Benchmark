"""Certified frozen experts, key alignment and source-only fusion splits."""

from __future__ import annotations

import json
import re
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset

from src.experimental.features import open_cache
from src.robustness.artifacts import load_predictions
from src.robustness.manifests import validate
from src.robustness.provenance import digest, digest_file


@dataclass
class FrozenExpert:
    frame: pd.DataFrame
    features: np.ndarray
    scores: np.ndarray
    identity: str
    contract: dict


def open_expert(spec):
    if not isinstance(spec, dict) or set(spec) - {"name", "role", "kind", "path", "expected_identity", "checkpoint_sha256", "calibration", "feature_columns"}:
        raise ValueError("Unknown frozen expert source field")
    kind, path = spec.get("kind", "feature_cache"), Path(spec["path"])
    if not re.fullmatch(r"[A-Za-z][A-Za-z0-9_]*", str(spec.get("name", ""))) or not spec.get("role"):
        raise ValueError("Declare each frozen expert name and forensic role")
    if kind == "feature_cache":
        cache = open_cache(path, expected_identity=spec.get("expected_identity"))
        extraction, key = cache.metadata["extraction"], cache.metadata["key"]
        checkpoint_hash = cache.metadata["checkpoint"]["sha256"]
        if (extraction.get("checkpoint_class1") != "fake" or key.get("checkpoint_sha256") != checkpoint_hash
                or cache.metadata["feature_dim"] != cache.features.shape[1]):
            raise ValueError("Frozen feature checkpoint or class orientation contract mismatch")
        expected_mode = {"srm": "srm", "rgb": "none"}.get(spec["role"])
        if expected_mode is not None and extraction.get("mode") != expected_mode:
            raise ValueError("Frozen expert forensic role does not match its preprocessing")
        scores = torch.from_numpy(np.asarray(cache.logits).copy()).float().softmax(-1)[:, 1].numpy()
        contract = {"kind": kind, "checkpoint_sha256": checkpoint_hash,
                    "run_config_sha256": key["run_config_sha256"], "extraction": extraction,
                    "code_sha256": key["code_sha256"], "packages": key["packages"],
                    "feature_dim": cache.features.shape[1], "score": "FP32 softmax of original two logits; class 1 is fake"}
        frame, features, identity = cache.frame, cache.features, cache.identity
    elif kind == "predictions":
        frame, record = load_predictions(path)
        declared = record.get("input_contract")
        if not declared or declared.get("checkpoint_class1", record["checkpoint_class1"]) != record["checkpoint_class1"]:
            raise ValueError("Frozen prediction expert needs a checkpoint-bound input_contract")
        if record.get("input_contract_sha256", digest(declared)) != digest(declared):
            raise ValueError("Prediction expert input contract hash mismatch")
        checkpoint_hash = record["model_sha256"]
        scores = frame.p_fake.to_numpy(dtype=np.float32)
        columns = spec.get("feature_columns", [])
        forbidden = {"label", "target", "y", "y_true", "img_name", "sample_id", "group_id", "dataset", "split", "generator", "video_id", "source_id"}
        if any(name not in frame or name in forbidden for name in columns):
            raise ValueError("Prediction expert feature columns must be declared label-free numeric observations")
        features = frame[columns].to_numpy(dtype=np.float32) if columns else np.empty((len(frame), 0), dtype=np.float32)
        if not np.isfinite(features).all():
            raise ValueError("Nonfinite prediction expert features")
        contract = {"kind": kind, "checkpoint_sha256": checkpoint_hash, "input_contract": declared,
                    "feature_columns": list(columns), "feature_dim": features.shape[1], "score": "certified p_fake"}
        identity = digest({"predictions_sha256": record["predictions_sha256"], "contract": contract})
    else:
        raise ValueError("Frozen expert kind must be feature_cache or predictions")
    if spec.get("checkpoint_sha256", checkpoint_hash) != checkpoint_hash:
        raise ValueError("Frozen expert checkpoint differs from explicitly required identity")
    if spec.get("calibration"):
        calibration = json.loads(Path(spec["calibration"]).read_text())
        if calibration.get("model_sha256") != checkpoint_hash or calibration.get("selection_split") != "val":
            raise ValueError("Expert calibration does not belong to its frozen checkpoint")
        if kind == "predictions" and calibration.get("input_contract") != contract["input_contract"]:
            raise ValueError("Expert calibration and prediction preprocessing differ")
    frame = validate(frame, require_both=False)
    if "image_sha256" not in frame or not frame.image_sha256.astype(str).map(lambda x: bool(re.fullmatch("[0-9a-f]{64}", x))).all():
        raise ValueError("Every frozen expert row needs its verified source image SHA-256")
    contract.update(name=str(spec["name"]), role=str(spec["role"]), checkpoint_class1="fake")
    return FrozenExpert(frame, features, scores, identity, contract)


class AlignedExperts:
    """Preserve memmaps; materialize only selected minibatches of features."""

    def __init__(self, experts, frame=None):
        if len(experts) < 2 or len({e.contract["name"] for e in experts}) != len(experts):
            raise ValueError("Use at least two distinctly named experts")
        self.experts = experts
        self.frame = validate(experts[0].frame if frame is None else frame, require_both=False)
        self.positions = []
        for expert in experts:
            if frame is None and set(expert.frame.sample_id) != set(self.frame.sample_id):
                raise ValueError("Frozen expert populations differ; no implicit inner join")
            lookup = pd.Series(np.arange(len(expert.frame)), index=expert.frame.sample_id)
            positions = lookup.reindex(self.frame.sample_id)
            if positions.isna().any():
                raise ValueError("Missing requested sample IDs in frozen expert")
            positions = positions.to_numpy(dtype=np.int64)
            matched = expert.frame.iloc[positions].reset_index(drop=True)
            for name in ("sample_id", "img_name", "group_id", "label", "dataset", "split", "image_sha256", "source_id", "video_id"):
                if name in self.frame and name in matched and not self.frame[name].astype(str).equals(matched[name].astype(str)):
                    raise ValueError(f"Frozen expert sample alignment mismatch: {name}")
            if "image_sha256" not in self.frame:
                self.frame["image_sha256"] = matched.image_sha256
            self.positions.append(positions)
        self.input_dim = sum(e.features.shape[1] for e in experts) + len(experts)

    def block(self, indices):
        indices = np.asarray(indices, dtype=np.int64)
        blocks, probabilities = [], []
        for expert, positions in zip(self.experts, self.positions):
            take = positions[indices]
            blocks.append(np.asarray(expert.features[take], dtype=np.float32))
            probabilities.append(expert.scores[take])
        scores = np.stack(probabilities, -1).astype(np.float32)
        clipped = np.clip(scores, 1e-6, 1 - 1e-6)
        odds = np.log(clipped) - np.log1p(-clipped)
        return np.concatenate([*blocks, odds], -1), scores


def open_sources(specs, frame=None, expected_contracts=None):
    experts = [open_expert(spec) for spec in specs]
    if expected_contracts is not None and [expert.contract for expert in experts] != expected_contracts:
        raise ValueError("Expert checkpoint, preprocessing, feature or ordering differs from fitted router")
    return AlignedExperts(experts, frame)


def split_validation(frame, *, seed=42, fit_fraction=0.5):
    """Hash source image names within groups, without label stratification.

    All group members follow the hash of their sorted source image names.
    Shared source IDs or image bytes across the split fail closed.
    """
    frame = validate(frame)
    if set(frame.split) != {"val"} or not 0 < fit_fraction < 1:
        raise ValueError("Learned fusion fits source validation only with fraction in (0,1)")
    group_fit = {}
    for group, members in frame.groupby("group_id", sort=True):
        value = digest(["forensic-fusion-val-split-v1", int(seed), sorted(members.img_name.astype(str))])
        group_fit[group] = int(value, 16) / 2**256 < fit_fraction
    mask = frame.group_id.map(group_fit).to_numpy(dtype=bool)
    fit, select = np.flatnonzero(mask), np.flatnonzero(~mask)
    if not len(fit) or not len(select) or frame.iloc[fit].label.nunique() != 2 or frame.iloc[select].label.nunique() != 2:
        raise ValueError("Hash split requires both classes in val_fit and val_select; declare another split seed before evaluation")
    checks = {}
    for name in ("sample_id", "group_id", "source_id", "image_sha256", "sha256"):
        if name not in frame:
            checks[name] = "unavailable; not certified"
            continue
        left = set(frame.iloc[fit][name].astype(str)) - {"", "unknown"}
        right = set(frame.iloc[select][name].astype(str)) - {"", "unknown"}
        if left & right:
            raise ValueError(f"Fusion fit/selection overlap in {name}")
        checks[name] = "disjoint among provided identities"
    return fit, select, {"rule": "SHA-256 of seed and sorted source img_name per group; labels excluded",
                         "seed": int(seed), "fit_fraction": float(fit_fraction), "disjointness": checks,
                         "val_fit_sample_ids_sha256": digest(sorted(frame.iloc[fit].sample_id.astype(str))),
                         "val_select_sample_ids_sha256": digest(sorted(frame.iloc[select].sample_id.astype(str)))}


def fit_standardizer(aligned, indices, block_size=512):
    total = np.zeros(aligned.input_dim, dtype=np.float64)
    squares = total.copy()
    for start in range(0, len(indices), block_size):
        values, _ = aligned.block(indices[start:start + block_size])
        values = values.astype(np.float64)
        total += values.sum(0)
        squares += np.square(values).sum(0)
    mean = total / len(indices)
    std = np.sqrt(np.maximum(squares / len(indices) - mean * mean, 0))
    # A feature constant on val_fit must not amplify unseen validation values.
    return mean.astype(np.float32), np.where(std > 1e-6, std, 1.0).astype(np.float32)


class FusionDataset(Dataset):
    def __init__(self, aligned, indices):
        self.aligned, self.indices = aligned, np.asarray(indices)

    def __len__(self):
        return len(self.indices)

    def __getitem__(self, index):
        selected = self.indices[index]
        x, p = self.aligned.block([selected])
        return torch.from_numpy(x[0]), torch.from_numpy(p[0]), torch.tensor(int(self.aligned.frame.iloc[selected].label))
