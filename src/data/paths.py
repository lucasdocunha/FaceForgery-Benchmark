"""Raiz das imagens no disco (mesma para raw e raw_min).

`raw_min` vs `raw` altera apenas os CSV em ``data/raw_min`` ou ``data/raw``;
os arquivos `.jpg` continuam nos splits phase1 abaixo.

Override opcional: variável de ambiente ``TCC_DATASET_ROOT``.
"""

from __future__ import annotations

import os
from pathlib import Path

_SPLIT_TO_SUBDIR = {"train": "trainset", "val": "valset", "test": "testset"}
_SPLIT_TO_SHORT_SUBDIR = {"train": "train", "val": "val", "test": "test"}

_REPO_DATA_ROOT = Path(__file__).resolve().parent.parent.parent / "data"
_DEFAULT_DATA_ROOT = _REPO_DATA_ROOT
_DEFAULT_ROOT = _REPO_DATA_ROOT / "datasets" / "phase1"
_LEGACY_ROOT = Path("/media/ssd2/lucas.ocunha/datasets/phase1")

# Raiz dos CSVs (data/raw e data/raw_min). Fixa e independente do cwd do processo,
# já que jobs (ex.: via myjobs) podem rodar com o working directory apontando para
# uma pasta de sandbox do job, não para a raiz do repositório.
# Override opcional: variável de ambiente ``TCC_DATA_ROOT``.
_DEFAULT_DATA_ROOT = Path("/users/home/lucas.ocunha/research/TCC/data")


def _safe_exists(path: Path) -> bool:
    try:
        return path.exists()
    except (OSError, PermissionError):
        return False


def phase1_split_root(split: str) -> Path:
    candidates = []
    if "TCC_DATASET_ROOT" in os.environ:
        candidates.append(Path(os.environ["TCC_DATASET_ROOT"]))
    candidates.append(_REPO_DATA_ROOT / "datasets" / "phase1")
    candidates.append(_REPO_DATA_ROOT / "datasets" / "min_dataset")
    candidates.append(Path("/home/lucas/TCC/phase1"))
    candidates.append(_LEGACY_ROOT)

    for base in candidates:
        canonical = base / _SPLIT_TO_SUBDIR.get(split, split)
        if _safe_exists(canonical):
            return canonical

        short = base / _SPLIT_TO_SHORT_SUBDIR.get(split, split)
        if _safe_exists(short):
            return short

        if _safe_exists(base / split):
            return base / split

    return _DEFAULT_ROOT / _SPLIT_TO_SUBDIR.get(split, split)


def data_root() -> Path:
    if "TCC_DATA_ROOT" in os.environ:
        return Path(os.environ["TCC_DATA_ROOT"])
    if _safe_exists(_DEFAULT_DATA_ROOT):
        return _DEFAULT_DATA_ROOT
    return _REPO_DATA_ROOT


# Disco local (/) enche rápido com checkpoints; ssd2 tem espaço de sobra.
# Override opcional: variável de ambiente ``TCC_MODELS_ROOT``.
_DEFAULT_MODELS_ROOT = Path("/media/ssd2/lucas.ocunha/models-tcc")


def models_root() -> Path:
    """Root for new-layout checkpoints, configurable per environment."""
    if "TCC_MODELS_ROOT" in os.environ:
        return Path(os.environ["TCC_MODELS_ROOT"])
    cisia_root = Path("/projects/models/lucas.ocunha")
    if _safe_exists(cisia_root):
        return cisia_root
    if _safe_exists(_DEFAULT_MODELS_ROOT):
        return _DEFAULT_MODELS_ROOT
    return _REPO_DATA_ROOT.parent / "models"


def output_root() -> Path:
    """Root for generated tables and heatmaps."""
    return Path(os.environ.get("TCC_OUTPUT_ROOT", str(_REPO_DATA_ROOT.parent)))


def pretrained_root() -> Path:
    """Diretório dedicado com uma subpasta para cada arquitetura pré-treinada.

    Estrutura:
        <pretrained_root>/clip/
        <pretrained_root>/vit/
        <pretrained_root>/dino/
        <pretrained_root>/resnet/
        <pretrained_root>/mobilenet/
        <pretrained_root>/xception/
    """
    if "TCC_PRETRAINED_ROOT" in os.environ:
        return Path(os.environ["TCC_PRETRAINED_ROOT"])
    cisia_candidate = Path("/projects/models/lucas.ocunha/pretrained")
    if _safe_exists(cisia_candidate) or _safe_exists(cisia_candidate.parent):
        return cisia_candidate
    return models_root() / "pretrained"


