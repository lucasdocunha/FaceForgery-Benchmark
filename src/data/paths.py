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

_DEFAULT_ROOT = Path("/media/ssd2/lucas.ocunha/datasets/phase1")

# Raiz dos CSVs (data/raw e data/raw_min). Fixa e independente do cwd do processo,
# já que jobs (ex.: via myjobs) podem rodar com o working directory apontando para
# uma pasta de sandbox do job, não para a raiz do repositório.
# Override opcional: variável de ambiente ``TCC_DATA_ROOT``.
_DEFAULT_DATA_ROOT = Path("/home/lucas.ocunha/research/TCC/data")


def phase1_split_root(split: str) -> Path:
    base = Path(os.environ.get("TCC_DATASET_ROOT", str(_DEFAULT_ROOT)))
    canonical = base / _SPLIT_TO_SUBDIR[split]
    if canonical.exists():
        return canonical

    short = base / _SPLIT_TO_SHORT_SUBDIR[split]
    if short.exists():
        return short

    return canonical


def data_root() -> Path:
    return Path(os.environ.get("TCC_DATA_ROOT", str(_DEFAULT_DATA_ROOT)))


# Disco local (/) enche rápido com checkpoints; ssd2 tem espaço de sobra.
# Override opcional: variável de ambiente ``TCC_MODELS_ROOT``.
_DEFAULT_MODELS_ROOT = Path("/media/ssd2/lucas.ocunha/models-tcc")


def models_root() -> Path:
    """Root for new-layout checkpoints, configurable per environment."""
    return Path(os.environ.get("TCC_MODELS_ROOT", str(_DEFAULT_MODELS_ROOT)))


def output_root() -> Path:
    """Root for generated tables and heatmaps."""
    return Path(os.environ.get("TCC_OUTPUT_ROOT", str(Path.cwd())))
