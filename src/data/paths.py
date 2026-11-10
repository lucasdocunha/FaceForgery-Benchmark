"""Portable data/artifact paths; explicit overrides never fall back silently."""
from __future__ import annotations

import os
from pathlib import Path

_SPLIT_TO_SUBDIR = {"train": "trainset", "val": "valset", "test": "testset"}
_REPO_DATA_ROOT = Path(__file__).resolve().parents[2] / "data"


def phase1_split_root(split: str) -> Path:
    explicit = os.environ.get("TCC_DATASET_ROOT")
    candidates = ([Path(explicit).expanduser()] if explicit else
                  [_REPO_DATA_ROOT / "datasets" / "phase1", _REPO_DATA_ROOT / "datasets" / "min_dataset", Path("/media/ssd2/lucas.ocunha/datasets/phase1")])
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
    explicit = os.environ.get("TCC_MODELS_ROOT") or os.environ.get("CISIA_MODELS_ROOT")
    if explicit:
        return Path(explicit).expanduser()
    user = os.environ.get("USER", "")
    for candidate in (Path("/media/ssd2/lucas.ocunha/models-tcc"),
                      Path("/projects/models") / user / "models-tcc",
                      Path("/projects/models") / user / "faceforgery"):
        if candidate.is_dir():
            return candidate
    return Path(str(_REPO_DATA_ROOT.parent / "models")).expanduser()


def _external_dataset(env: str, server_default: str) -> Path:
    """Dataset that lives outside the repo; `env` overrides the lab-server default."""
    return Path(os.environ.get(env, server_default)).expanduser()


def test_d_root() -> Path:
    """Images of the hard (degraded) MFFI test split, `test_d`."""
    return _external_dataset("TCC_TEST_D_ROOT", "/media/ssd2/lucas.ocunha/datasets/phase1/test_d")


def celeb_df_root() -> Path:
    """Face crops produced by scripts/prepare_celeb_df.py (names in data/celeb_df/test.csv)."""
    return _external_dataset("TCC_CELEB_DF_ROOT", "/media/ssd2/lucas.ocunha/datasets/celeb_df_crops")


def df40_root() -> Path:
    """Extracted DF-40 tree (holds paths_cache.json for the 650k-fake protocol)."""
    return _external_dataset("TCC_DF40_ROOT", "/media/ssd2/lucas.ocunha/datasets/df40_extracted")


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
