"""Portable data/artifact paths; explicit overrides never fall back silently."""
from __future__ import annotations

import os
from pathlib import Path

_SPLIT_TO_SUBDIR = {"train": "trainset", "val": "valset", "test": "testset"}
_REPO_DATA_ROOT = Path(__file__).resolve().parents[2] / "data"


def phase1_split_root(split: str) -> Path:
    explicit = os.environ.get("TCC_DATASET_ROOT")
    candidates = ([Path(explicit).expanduser()] if explicit else
                  [_REPO_DATA_ROOT / "datasets" / "phase1", _REPO_DATA_ROOT / "datasets" / "min_dataset"])
    for base in candidates:
        for name in dict.fromkeys((_SPLIT_TO_SUBDIR.get(split, split), split)):
            path = base / name
            if path.is_dir():
                return path
    expected = candidates[0] / _SPLIT_TO_SUBDIR.get(split, split)
    if explicit:
        raise FileNotFoundError(f"Split {split!r} not found under TCC_DATASET_ROOT={explicit}; expected {expected}")
    return expected


def data_root() -> Path:
    """CSV manifests, independent of the process working directory."""
    return Path(os.environ.get("TCC_DATA_ROOT", str(_REPO_DATA_ROOT))).expanduser()


def models_root() -> Path:
    """Training work tree (job-local when launched by the CISIA runner)."""
    return Path(os.environ.get("TCC_MODELS_ROOT", str(_REPO_DATA_ROOT.parent / "models"))).expanduser()


def output_root() -> Path:
    return Path(os.environ.get("TCC_OUTPUT_ROOT", str(_REPO_DATA_ROOT.parent))).expanduser()


def pretrained_root() -> Path:
    """Read-only pretrained source; the CISIA runner resolves each family."""
    explicit = os.environ.get("TCC_PRETRAINED_ROOT")
    if explicit:
        return Path(explicit).expanduser()
    for path in (Path("/datasets/ai_models/faceforgery/pretrained"),
                 Path("/projects/models") / os.environ.get("USER", "") / "pretrained"):
        if path.is_dir():
            return path
    return models_root() / "pretrained"
