from __future__ import annotations

import torch
import torch.nn as nn


def maybe_data_parallel(
    model: nn.Module,
    device: torch.device,
    enabled: bool = True,
) -> nn.Module:
    if enabled and device.type == "cuda" and torch.cuda.device_count() > 1:
        return nn.DataParallel(model)
    return model


def unwrap_model(model: nn.Module) -> nn.Module:
    if isinstance(model, nn.DataParallel):
        return model.module
    return model


def model_state_dict(model: nn.Module):
    return unwrap_model(model).state_dict()


def mixup_batch(
    x: torch.Tensor,
    y: torch.Tensor,
    alpha: float,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, float]:
    if alpha <= 0.0 or x.size(0) < 2:
        return x, y, y, 1.0

    lam = float(torch.distributions.Beta(alpha, alpha).sample().item())
    perm = torch.randperm(x.size(0), device=x.device)
    mixed_x = lam * x + (1.0 - lam) * x[perm]
    return mixed_x, y, y[perm], lam


def rand_bbox(size: tuple[int, ...], lam: float) -> tuple[int, int, int, int]:
    """Generates a random bounding box coordinates for CutMix."""
    W = size[-1]
    H = size[-2]
    cut_rat = float(torch.sqrt(torch.tensor(1.0 - lam)).item())
    cut_w = int(W * cut_rat)
    cut_h = int(H * cut_rat)
    
    cx = int(torch.randint(0, W, (1,)).item())
    cy = int(torch.randint(0, H, (1,)).item())
    
    bbx1 = int(torch.clamp(torch.tensor(cx - cut_w // 2), 0, W).item())
    bby1 = int(torch.clamp(torch.tensor(cy - cut_h // 2), 0, H).item())
    bbx2 = int(torch.clamp(torch.tensor(cx + cut_w // 2), 0, W).item())
    bby2 = int(torch.clamp(torch.tensor(cy + cut_h // 2), 0, H).item())
    
    return bbx1, bby1, bbx2, bby2


def cutmix_batch(
    x: torch.Tensor,
    y: torch.Tensor,
    alpha: float,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, float]:
    if alpha <= 0.0 or x.size(0) < 2:
        return x, y, y, 1.0

    lam = float(torch.distributions.Beta(alpha, alpha).sample().item())
    perm = torch.randperm(x.size(0), device=x.device)
    
    bbx1, bby1, bbx2, bby2 = rand_bbox(x.size(), lam)
    mixed_x = x.clone()
    mixed_x[:, :, bby1:bby2, bbx1:bbx2] = x[perm, :, bby1:bby2, bbx1:bbx2]
    
    lam = 1.0 - float((bbx2 - bbx1) * (bby2 - bby1) / (x.size(-1) * x.size(-2)))
    return mixed_x, y, y[perm], lam


def apply_mixup_or_cutmix(
    x: torch.Tensor,
    y: torch.Tensor,
    mixup_alpha: float,
    cutmix_alpha: float,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, float]:
    """Applies Mixup, CutMix, or randomly chooses between both (50/50) if both are active."""
    use_mixup = mixup_alpha > 0.0
    use_cutmix = cutmix_alpha > 0.0
    
    if use_mixup and use_cutmix:
        if float(torch.rand(1).item()) < 0.5:
            return mixup_batch(x, y, mixup_alpha)
        else:
            return cutmix_batch(x, y, cutmix_alpha)
    elif use_mixup:
        return mixup_batch(x, y, mixup_alpha)
    elif use_cutmix:
        return cutmix_batch(x, y, cutmix_alpha)
    else:
        return x, y, y, 1.0


def mixup_loss(
    criterion,
    logits: torch.Tensor,
    y_a: torch.Tensor,
    y_b: torch.Tensor,
    lam: float,
) -> torch.Tensor:
    if lam >= 1.0:
        return criterion(logits, y_a)
    return lam * criterion(logits, y_a) + (1.0 - lam) * criterion(logits, y_b)
