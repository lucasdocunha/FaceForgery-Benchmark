from __future__ import annotations

import torch
import torch.nn as nn
from torchvision import models
from src.models._channel_adapt import adapt_conv2d_channels


_VARIANTS = {
    "small": models.mobilenet_v3_small,
    "large": models.mobilenet_v3_large,
}


def _adapt_first_conv(model: nn.Module, in_channels: int, pretrained: bool) -> None:
    first_conv = model.features[0][0]
    if not isinstance(first_conv, nn.Conv2d):
        raise TypeError("Expected MobileNet first layer to be Conv2d.")
    if first_conv.in_channels == in_channels:
        return

    model.features[0][0] = adapt_conv2d_channels(first_conv, in_channels)


def mobilenet(
    num_classes: int = 2,
    in_channels: int = 3,
    pretrained: bool = False,
    variant: str = "small",
    dropout: float | None = None,
    allow_pretrained: bool = False,
) -> nn.Module:
    if variant not in _VARIANTS:
        valid = ", ".join(sorted(_VARIANTS))
        raise ValueError(f"variant must be one of: {valid}")

    if pretrained and not allow_pretrained:
        raise ValueError("External pretrained MobileNet weights are disabled for this project.")
    builder = _VARIANTS[variant]
    model = builder(weights="DEFAULT" if pretrained else None)
    _adapt_first_conv(model, in_channels, pretrained=pretrained)

    last_linear = model.classifier[-1]
    if not isinstance(last_linear, nn.Linear):
        raise TypeError("Expected MobileNet classifier to end with Linear.")
    if dropout is not None:
        for module in model.classifier.modules():
            if isinstance(module, nn.Dropout):
                module.p = dropout
    model.classifier[-1] = nn.Linear(last_linear.in_features, num_classes)
    return model


def mobilenetv3_small(
    num_classes: int = 2,
    in_channels: int = 3,
    pretrained: bool = False,
    dropout: float | None = None,
) -> nn.Module:
    return mobilenet(
        num_classes=num_classes,
        in_channels=in_channels,
        pretrained=pretrained,
        variant="small",
        dropout=dropout,
    )


def mobilenetv3_large(
    num_classes: int = 2,
    in_channels: int = 3,
    pretrained: bool = False,
    dropout: float | None = None,
) -> nn.Module:
    return mobilenet(
        num_classes=num_classes,
        in_channels=in_channels,
        pretrained=pretrained,
        variant="large",
        dropout=dropout,
    )


def freeze_classifier_only(model: nn.Module) -> None:
    for param in model.parameters():
        param.requires_grad = False
    for param in model.classifier.parameters():
        param.requires_grad = True


def unfreeze_last_blocks(model: nn.Module, last_n_blocks: int = 3) -> None:
    for block in model.features[-last_n_blocks:] if last_n_blocks > 0 else []:
        for param in block.parameters():
            param.requires_grad = True
    for param in model.classifier.parameters():
        param.requires_grad = True


def build(config) -> nn.Module:
    return mobilenet(2, config.in_channels, config.regime == "finetune",
                     config.variant, config.dropout, config.allow_pretrained)


def freeze_backbone(model: nn.Module) -> None:
    freeze_classifier_only(model)


def unfreeze_for_finetune(model: nn.Module, n: int) -> None:
    freeze_classifier_only(model)
    unfreeze_last_blocks(model, n)
