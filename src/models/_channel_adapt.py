from __future__ import annotations

import torch
import torch.nn as nn


def adapt_conv2d_channels(conv: nn.Conv2d, in_channels: int) -> nn.Conv2d:
    """Clone a Conv2d for a new input count, preserving pretrained signal."""
    if conv.in_channels == in_channels:
        return conv
    if conv.groups != 1:
        raise ValueError("Input-channel adaptation requires groups=1")
    new = nn.Conv2d(
        in_channels, conv.out_channels, conv.kernel_size, conv.stride, conv.padding,
        conv.dilation, conv.groups, conv.bias is not None, conv.padding_mode,
    ).to(device=conv.weight.device, dtype=conv.weight.dtype)
    with torch.no_grad():
        old = conv.weight
        if in_channels == 1:
            weight = old.mean(dim=1, keepdim=True)
        else:
            repeats = (in_channels + old.shape[1] - 1) // old.shape[1]
            weight = old.repeat(1, repeats, 1, 1)[:, :in_channels]
            weight.mul_(old.shape[1] / float(in_channels))
        new.weight.copy_(weight)
        if conv.bias is not None:
            new.bias.copy_(conv.bias)
    return new


def replace_conv2d(root: nn.Module, dotted_path: str, in_channels: int) -> nn.Conv2d:
    parts = dotted_path.split(".")
    parent: nn.Module = root
    for part in parts[:-1]:
        parent = parent[int(part)] if part.isdigit() else getattr(parent, part)
    leaf = parts[-1]
    old = parent[int(leaf)] if leaf.isdigit() else getattr(parent, leaf)
    if not isinstance(old, nn.Conv2d):
        raise TypeError(f"Expected Conv2d at {dotted_path}")
    new = adapt_conv2d_channels(old, in_channels)
    if leaf.isdigit():
        parent[int(leaf)] = new
    else:
        setattr(parent, leaf, new)
    return new
