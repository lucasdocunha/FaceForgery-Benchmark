from __future__ import annotations

from dataclasses import dataclass
from typing import Callable

import torch.nn as nn

from src.models import clip, dino, hybrid, mobilenet, resnet, vit, xception

ParameterGroups = list[dict[str, object]]

_HEAD_PREFIXES = ("classifier.", "fc.", "head.", "fusion.", "freq_branch.")


def _default_parameter_groups(model: nn.Module, config: object) -> ParameterGroups:
    head, backbone = [], []
    for name, param in model.named_parameters():
        if not param.requires_grad:
            continue
        (head if name.startswith(_HEAD_PREFIXES) else backbone).append(param)
    groups: ParameterGroups = []
    if backbone:
        groups.append({"params": backbone, "lr": config.lr_backbone, "name": "backbone"})
    if head:
        groups.append({"params": head, "lr": config.lr_head, "name": "head"})
    if not groups:
        raise ValueError("Model has no trainable parameters")
    return groups


@dataclass(frozen=True)
class ModelSpec:
    build: Callable[[object], nn.Module]
    freeze_backbone: Callable[[nn.Module], None]
    unfreeze_for_finetune: Callable[[nn.Module, int], None]
    parameter_groups: Callable[[nn.Module, object], ParameterGroups] = _default_parameter_groups


MODEL_REGISTRY = {
    name: ModelSpec(module.build, module.freeze_backbone, module.unfreeze_for_finetune)
    for name, module in {
        "resnet": resnet, "xception": xception, "mobilenet": mobilenet,
        "vit": vit, "clip": clip, "dino": dino, "hybrid": hybrid,
    }.items()
}


def get_model_spec(family: str) -> ModelSpec:
    try:
        return MODEL_REGISTRY[family]
    except KeyError as exc:
        raise ValueError(f"Unknown model family: {family}") from exc
