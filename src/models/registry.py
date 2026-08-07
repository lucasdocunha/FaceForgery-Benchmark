from __future__ import annotations
from dataclasses import dataclass
from typing import Callable
import torch.nn as nn
from src.models import clip, dino, mobilenet, resnet, vit, xception

@dataclass(frozen=True)
class ModelSpec:
    build: Callable[[object], nn.Module]
    freeze_backbone: Callable[[nn.Module], None]
    unfreeze_for_finetune: Callable[[nn.Module, int], None]

MODEL_REGISTRY = {name: ModelSpec(module.build, module.freeze_backbone, module.unfreeze_for_finetune)
                  for name, module in {"resnet":resnet, "xception":xception, "mobilenet":mobilenet,
                                       "vit":vit, "clip":clip, "dino":dino}.items()}

def get_model_spec(family: str) -> ModelSpec:
    try: return MODEL_REGISTRY[family]
    except KeyError as exc: raise ValueError(f"Unknown model family: {family}") from exc
