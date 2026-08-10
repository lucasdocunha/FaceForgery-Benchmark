from __future__ import annotations

from dataclasses import dataclass
from typing import Callable

import torch.nn as nn

from src.models import clip, dino, mobilenet, resnet, vit, xception

ParameterGroups = list[dict[str, object]]


# Prefixos ancorados na raiz do modelo. Todas as famílias expõem a cabeça como
# `self.fc` (resnet/xception) ou `self.classifier` (mobilenet/vit/clip/dino).
# Ancorar é obrigatório: um teste de substring também casaria `mlp.fc1`/`mlp.fc2`
# (CLIP, ConvNeXt do DINOv3) e o squeeze-excitation `fc1`/`fc2` do MobileNetV3,
# jogando a maior parte do backbone para o lr_head.
_HEAD_PREFIXES = ("classifier.", "fc.", "head.")


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
        "vit": vit, "clip": clip, "dino": dino,
    }.items()
}


def get_model_spec(family: str) -> ModelSpec:
    try:
        return MODEL_REGISTRY[family]
    except KeyError as exc:
        raise ValueError(f"Unknown model family: {family}") from exc
