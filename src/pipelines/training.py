from __future__ import annotations

import torch
import torch.nn as nn
import random
from pathlib import Path
import numpy as np
import pandas as pd
from torch.optim import AdamW
from torch.optim.lr_scheduler import ReduceLROnPlateau
from src.pipelines.evaluation import best_threshold, checkpoint_score, evaluate_classifier, sanitize_inputs, sanitize_logits
from src.plots.plots import plot_confusion_matrix, plot_roc_auc


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


def seed_everything(seed: int) -> None:
    random.seed(seed); np.random.seed(seed); torch.manual_seed(seed)
    if torch.cuda.is_available(): torch.cuda.manual_seed_all(seed)


class Trainer:
    def __init__(self, model, train_loader, val_loader, test_loader, config, output_dir, model_spec=None, device=None):
        self.config, self.output_dir = config, Path(output_dir)
        self.device = device or torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model = maybe_data_parallel(model.to(self.device), self.device, config.multi_gpu)
        self.train_loader, self.val_loader, self.test_loader = train_loader, val_loader, test_loader
        self.model_spec = model_spec

    def _optimizer(self):
        model = unwrap_model(self.model)
        head_ids = {id(p) for name, p in model.named_parameters() if any(k in name for k in ("classifier", "fc", "head")) and p.requires_grad}
        head = [p for p in model.parameters() if id(p) in head_ids]
        backbone = [p for p in model.parameters() if p.requires_grad and id(p) not in head_ids]
        groups = []
        if backbone: groups.append({"params": backbone, "lr": self.config.lr_backbone})
        if head: groups.append({"params": head, "lr": self.config.lr_head})
        return AdamW(groups, weight_decay=self.config.weight_decay)

    def fit(self):
        seed_everything(self.config.seed)
        for folder in ("weights", "results", "plots"): (self.output_dir / folder).mkdir(parents=True, exist_ok=True)
        criterion, optimizer = nn.CrossEntropyLoss(), self._optimizer()
        scheduler = ReduceLROnPlateau(optimizer, mode="max", patience=max(1, self.config.early_stop_patience // 2))
        best_score, stale, history = -float("inf"), 0, []
        for epoch in range(self.config.epochs):
            self.model.train(); losses = []
            for x, y, _ in self.train_loader:
                x, y = sanitize_inputs(x.to(self.device)), y.to(self.device); optimizer.zero_grad(set_to_none=True)
                x, ya, yb, lam = apply_mixup_or_cutmix(x, y, self.config.mixup_alpha, self.config.cutmix_alpha)
                with torch.amp.autocast("cuda", enabled=self.device.type == "cuda"):
                    logits = sanitize_logits(self.model(x)); loss = mixup_loss(criterion, logits, ya, yb, lam)
                loss.backward(); torch.nn.utils.clip_grad_norm_(self.model.parameters(), 1.0); optimizer.step(); losses.append(loss.item())
            val = evaluate_classifier(self.model, self.val_loader, criterion, self.device, use_amp=self.device.type == "cuda", desc="Val")
            threshold, _ = best_threshold(val["y_true"], val["probs"], self.config.threshold_strategy)
            score = checkpoint_score(val); scheduler.step(score)
            history.append({"epoch": epoch + 1, "train_loss": float(np.mean(losses)), "val_auc": val["auc"], "val_acc": val["acc"]})
            if score > best_score:
                best_score, stale = score, 0
                torch.save(model_state_dict(self.model), self.output_dir / "weights" / "best.pth")
            else:
                stale += 1
                if stale >= self.config.early_stop_patience: break
        torch.save(model_state_dict(self.model), self.output_dir / "weights" / "final.pth")
        unwrap_model(self.model).load_state_dict(torch.load(self.output_dir / "weights" / "best.pth", map_location=self.device, weights_only=True))
        test = evaluate_classifier(self.model, self.test_loader, criterion, self.device, threshold=threshold, use_amp=self.device.type == "cuda", desc="Test")
        meta = self.config.to_dict() | {k: v for k, v in test.items() if np.isscalar(v)} | {"threshold": threshold}
        pd.DataFrame([meta]).to_csv(self.output_dir / "results" / "metrics_test.csv", index=False)
        pd.DataFrame(history).to_csv(self.output_dir / "results" / "history.csv", index=False)
        np.savez_compressed(self.output_dir / "results" / "outputs_test.npz", probs=test["probs"], y_true=test["y_true"], y_pred=test["y_pred"], ids=test["ids"])
        plot_confusion_matrix(test, str(self.output_dir), title=f"{self.config.model_family} confusion matrix")
        plot_roc_auc(test, str(self.output_dir), title=f"{self.config.model_family} ROC", family=self.config.model_family)
        return test
