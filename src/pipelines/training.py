from __future__ import annotations

import json
import random
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.optim import AdamW
from torch.optim.lr_scheduler import ReduceLROnPlateau

from src.pipelines.config import RUN_CONFIG_FILENAME
from src.pipelines.evaluation import (
    best_threshold, checkpoint_score, evaluate_classifier, sanitize_inputs, sanitize_logits,
)
from src.plots.plots import plot_confusion_matrix, plot_roc_auc, save_metrics_csv


def maybe_data_parallel(model: nn.Module, device: torch.device, enabled: bool = True) -> nn.Module:
    if enabled and device.type == "cuda" and torch.cuda.device_count() > 1:
        return nn.DataParallel(model)
    return model


def unwrap_model(model: nn.Module) -> nn.Module:
    return model.module if isinstance(model, nn.DataParallel) else model


def model_state_dict(model: nn.Module):
    return unwrap_model(model).state_dict()


def mixup_batch(x: torch.Tensor, y: torch.Tensor, alpha: float):
    if alpha <= 0.0 or x.size(0) < 2:
        return x, y, y, 1.0
    lam = float(torch.distributions.Beta(alpha, alpha).sample().item())
    perm = torch.randperm(x.size(0), device=x.device)
    return lam * x + (1.0 - lam) * x[perm], y, y[perm], lam


def rand_bbox(size: tuple[int, ...], lam: float) -> tuple[int, int, int, int]:
    width, height = size[-1], size[-2]
    ratio = float(torch.sqrt(torch.tensor(1.0 - lam)).item())
    cut_w, cut_h = int(width * ratio), int(height * ratio)
    cx, cy = int(torch.randint(0, width, (1,)).item()), int(torch.randint(0, height, (1,)).item())
    return max(cx-cut_w//2, 0), max(cy-cut_h//2, 0), min(cx+cut_w//2, width), min(cy+cut_h//2, height)


def cutmix_batch(x: torch.Tensor, y: torch.Tensor, alpha: float):
    if alpha <= 0.0 or x.size(0) < 2:
        return x, y, y, 1.0
    lam = float(torch.distributions.Beta(alpha, alpha).sample().item())
    perm = torch.randperm(x.size(0), device=x.device)
    x1, y1, x2, y2 = rand_bbox(tuple(x.size()), lam)
    mixed = x.clone()
    mixed[:, :, y1:y2, x1:x2] = x[perm, :, y1:y2, x1:x2]
    lam = 1.0 - ((x2-x1)*(y2-y1) / float(x.size(-1)*x.size(-2)))
    return mixed, y, y[perm], lam


def apply_mixup_or_cutmix(x, y, mixup_alpha: float, cutmix_alpha: float):
    if mixup_alpha > 0 and cutmix_alpha > 0:
        return mixup_batch(x, y, mixup_alpha) if torch.rand(1).item() < .5 else cutmix_batch(x, y, cutmix_alpha)
    if mixup_alpha > 0:
        return mixup_batch(x, y, mixup_alpha)
    if cutmix_alpha > 0:
        return cutmix_batch(x, y, cutmix_alpha)
    return x, y, y, 1.0


def mixup_loss(criterion, logits, y_a, y_b, lam: float):
    return criterion(logits, y_a) if lam >= 1.0 else lam*criterion(logits, y_a)+(1-lam)*criterion(logits, y_b)


def seed_everything(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def _scalar_metrics(metrics: dict) -> dict:
    keys = ("loss", "acc", "precision", "recall", "f1", "auc", "specificity", "tp", "fp", "fn", "tn")
    return {key: metrics[key] for key in keys}


class Trainer:
    def __init__(self, model, train_loader, val_loader, test_loader, config, output_dir, model_spec, device=None):
        self.config = config
        self.output_dir = Path(output_dir)
        self.device = torch.device(device or ("cuda" if torch.cuda.is_available() else "cpu"))
        self.model_spec = model_spec
        self.model = maybe_data_parallel(model.to(self.device), self.device, config.multi_gpu)
        self.train_loader, self.val_loader, self.test_loader = train_loader, val_loader, test_loader
        self.epochs_completed = 0
        self.stopped_early = False

    def _optimizer(self):
        return AdamW(
            self.model_spec.parameter_groups(unwrap_model(self.model), self.config),
            weight_decay=self.config.weight_decay,
        )

    def _criterion(self):
        weights = None
        if self.config.use_class_weights:
            dataset = self.train_loader.dataset
            if hasattr(dataset, "df"):
                labels = torch.as_tensor(dataset.df.iloc[:, 1].astype(int).to_numpy(copy=True))
            elif hasattr(dataset, "tensors") and len(dataset.tensors) > 1:
                labels = dataset.tensors[1].long().cpu()
            else:
                raise ValueError("Class weights require dataset labels")
            counts = torch.bincount(labels, minlength=2).float()
            weights = torch.where(counts > 0, counts.sum() / (2 * counts), torch.zeros_like(counts)).to(self.device)
        return nn.CrossEntropyLoss(weight=weights, label_smoothing=self.config.label_smoothing)

    def _save_run_config(self) -> None:
        """Grava a config do run como artefato próprio.

        Fonte única de verdade para reconstruir a arquitetura depois (ver
        ``checkpoints.config_from_run``). Fica fora de ``metrics_{split}.csv``
        de propósito: aquele arquivo é reescrito por ``evaluate.py``, que não
        conhece a config, e antes disso a reavaliação apagava os campos de que
        dependia para reconstruir o modelo.
        """
        path = self.output_dir / "results" / RUN_CONFIG_FILENAME
        path.write_text(json.dumps(self.config.to_dict(), indent=2, sort_keys=True), encoding="utf-8")

    def _save_split(self, split: str, metrics: dict, threshold: float) -> None:
        result_dir = self.output_dir / "results"
        row = {
            "model_family": self.config.model_family,
            "fourier_mode": self.config.fourier_mode,
            "regime": self.config.regime,
            "seed": self.config.seed,
            "split": split,
            "threshold": threshold,
            **_scalar_metrics(metrics),
        }
        pd.DataFrame([row]).to_csv(result_dir / f"metrics_{split}.csv", index=False)
        np.savez_compressed(
            result_dir / f"outputs_{split}.npz", logits=metrics["logits"], probs=metrics["probs"],
            y_true=metrics["y_true"], y_pred=metrics["y_pred"], ids=metrics["ids"], threshold=threshold,
        )
        pd.DataFrame({
            "id": metrics["ids"], "y_true": metrics["y_true"], "y_pred": metrics["y_pred"],
            "prob_pos": metrics["probs"],
        }).to_csv(result_dir / f"predictions_{split}.csv", index=False)

    def fit(self):
        seed_everything(self.config.seed)
        for folder in ("weights", "results", "plots"):
            (self.output_dir / folder).mkdir(parents=True, exist_ok=True)
        self._save_run_config()
        criterion = self._criterion()
        optimizer = self._optimizer()
        scheduler = ReduceLROnPlateau(optimizer, mode="max", patience=self.config.scheduler_patience)
        scaler = torch.amp.GradScaler("cuda", enabled=self.device.type == "cuda")
        best_score, best_threshold_value, stale, history = -float("inf"), .5, 0, []

        for epoch in range(self.config.epochs):
            self.model.train()
            losses = []
            for x, y, _ in self.train_loader:
                x, y = sanitize_inputs(x.to(self.device)), y.to(self.device)
                optimizer.zero_grad(set_to_none=True)
                x, y_a, y_b, lam = apply_mixup_or_cutmix(x, y, self.config.mixup_alpha, self.config.cutmix_alpha)
                with torch.amp.autocast("cuda", enabled=self.device.type == "cuda"):
                    logits = sanitize_logits(self.model(x))
                    loss = mixup_loss(criterion, logits, y_a, y_b, lam)
                scaler.scale(loss).backward()
                scaler.unscale_(optimizer)
                if self.config.max_grad_norm is not None:
                    torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.config.max_grad_norm)
                scaler.step(optimizer)
                scaler.update()
                losses.append(float(loss.detach().item()))

            validation = evaluate_classifier(
                self.model, self.val_loader, criterion, self.device,
                use_amp=self.device.type == "cuda", desc="Val",
            )
            threshold, threshold_score = best_threshold(
                validation["y_true"], validation["probs"], self.config.threshold_strategy,
            )
            score = checkpoint_score(validation)
            scheduler.step(score)
            self.epochs_completed = epoch + 1
            history.append({
                "epoch": epoch + 1, "train_loss": float(np.mean(losses)) if losses else 0.0,
                "val_loss": validation["loss"], "val_auc": validation["auc"], "val_acc": validation["acc"],
                "threshold": threshold, "threshold_score": threshold_score,
            })
            if score > best_score:
                best_score, best_threshold_value, stale = score, threshold, 0
                torch.save(model_state_dict(self.model), self.output_dir / "weights" / "best.pth")
            else:
                stale += 1
                if stale >= self.config.early_stop_patience:
                    self.stopped_early = True
                    break

        torch.save(model_state_dict(self.model), self.output_dir / "weights" / "final.pth")
        unwrap_model(self.model).load_state_dict(torch.load(
            self.output_dir / "weights" / "best.pth", map_location=self.device, weights_only=True,
        ))
        validation = evaluate_classifier(
            self.model, self.val_loader, criterion, self.device, threshold=best_threshold_value,
            use_amp=self.device.type == "cuda", desc="Val best",
        )
        test = evaluate_classifier(
            self.model, self.test_loader, criterion, self.device, threshold=best_threshold_value,
            use_amp=self.device.type == "cuda", desc="Test",
        )
        self._save_split("val", validation, best_threshold_value)
        self._save_split("test", test, best_threshold_value)
        save_metrics_csv(test, str(self.output_dir), extra_info={
            **self.config.to_dict(), "threshold": best_threshold_value,
        })
        pd.DataFrame(history).to_csv(self.output_dir / "results" / "history.csv", index=False)
        plot_confusion_matrix(test, str(self.output_dir), title=f"{self.config.model_family} confusion matrix")
        plot_roc_auc(test, str(self.output_dir), title=f"{self.config.model_family} ROC", family=self.config.model_family)
        return test
