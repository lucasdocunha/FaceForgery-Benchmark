"""Localização de submódulos em backbones do ``transformers``.

Os caminhos mudaram entre as versões maiores da biblioteca:

===================  ==========================  ==========================
módulo               transformers 4.x            transformers 5.x
===================  ==========================  ==========================
``ViTModel``         ``encoder.layer``           ``layers``
``CLIPVisionModel``  ``vision_model.encoder``    ``encoder``
===================  ==========================  ==========================

Fixar um caminho quebrava todo o finetune de ViT/CLIP com ``AttributeError``
(``'ViTModel' object has no attribute 'encoder'``), e nenhum teste pegava porque
``unfreeze_for_finetune`` não era exercitado. A resolução aqui é tolerante a
versão e falha com a lista de candidatos reais quando não encontra nada.
"""

from __future__ import annotations

import torch.nn as nn

# Ordem: 5.x primeiro (é o piso em pyproject.toml), 4.x como fallback.
_ENCODER_LAYER_PATHS = (
    "layers",                       # 5.x ViTModel
    "encoder.layers",               # 5.x CLIPVisionModel
    "encoder.layer",                # 4.x ViTModel
    "vision_model.encoder.layers",  # 4.x CLIPVisionModel
)

_EMBEDDINGS_PATHS = (
    "embeddings",                   # 5.x CLIPVisionModel / ViTModel
    "vision_model.embeddings",      # 4.x CLIPVisionModel
)


def _resolve(root: nn.Module, dotted_path: str) -> nn.Module | None:
    node = root
    for part in dotted_path.split("."):
        node = getattr(node, part, None)
        if node is None:
            return None
    return node


def first_module(root: nn.Module, paths: tuple[str, ...], kind: type | None = None) -> nn.Module:
    for path in paths:
        node = _resolve(root, path)
        if node is not None and (kind is None or isinstance(node, kind)):
            return node
    found = [name for name, module in root.named_modules()
             if kind is None or isinstance(module, kind)]
    raise AttributeError(
        f"Nenhum de {paths} existe em {type(root).__name__}"
        f"{f' como {kind.__name__}' if kind else ''}. Encontrados: {found[:12] or 'nenhum'}"
    )


def encoder_layers(backbone: nn.Module) -> nn.ModuleList:
    """ModuleList com os blocos do encoder, para descongelar os últimos N."""
    layers = first_module(backbone, _ENCODER_LAYER_PATHS, nn.ModuleList)
    if len(layers) == 0:
        raise AttributeError(f"Encoder de {type(backbone).__name__} não tem blocos")
    return layers


def embeddings(backbone: nn.Module) -> nn.Module:
    """Módulo de embeddings, onde vive a projeção de patches (cirurgia de canal)."""
    return first_module(backbone, _EMBEDDINGS_PATHS)
