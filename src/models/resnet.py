import torch.nn as nn
from torchvision import models
from src.models._channel_adapt import adapt_conv2d_channels


_ARCHITECTURES = {
    "resnet18": models.resnet18,
    "resnet34": models.resnet34,
    "resnet50": models.resnet50,
    "resnet101": models.resnet101,
    "resnet152": models.resnet152,
}


def resnet(
    num_classes: int = 2,
    pretrained: bool = False,
    architecture: str = "resnet18",
    dropout: float = 0.2,
    in_channels: int = 3,
    allow_pretrained: bool = False,
) -> nn.Module:
    if architecture not in _ARCHITECTURES:
        valid = ", ".join(sorted(_ARCHITECTURES))
        raise ValueError(f"architecture must be one of: {valid}")

    if pretrained and not allow_pretrained:
        raise ValueError("External pretrained ResNet weights are disabled for this project.")
    builder = _ARCHITECTURES[architecture]
    model = builder(weights="DEFAULT" if pretrained else None)

    model.conv1 = adapt_conv2d_channels(model.conv1, in_channels)

    in_features = model.fc.in_features
    model.fc = nn.Sequential(
        nn.Dropout(p=dropout),
        nn.Linear(in_features, num_classes),
    )
    return model


def freeze_backbone(model: nn.Module) -> None:
    for name, param in model.named_parameters():
        param.requires_grad = name.startswith("fc.")


def unfreeze_last_blocks(model: nn.Module, train_layer3: bool = False) -> None:
    for param in model.layer4.parameters():
        param.requires_grad = True

    if train_layer3:
        for param in model.layer3.parameters():
            param.requires_grad = True

    for param in model.fc.parameters():
        param.requires_grad = True


def build(config) -> nn.Module:
    return resnet(2, config.regime == "finetune", config.architecture,
                  config.dropout, config.in_channels, config.allow_pretrained)


def unfreeze_for_finetune(model: nn.Module, n: int) -> None:
    freeze_backbone(model)
    if n > 0:
        unfreeze_last_blocks(model, train_layer3=n > 1)
