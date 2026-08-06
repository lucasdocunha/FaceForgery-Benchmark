from __future__ import annotations

import logging
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, WeightedRandomSampler
from torchvision import transforms
from tqdm import tqdm

from src.data import FourierMode, ImageDataset
from src.data.paths import data_root, phase1_split_root
from src.models.vit import PretrainedViTClassifier, VisionTransformerClassifier
from src.pipelines.evaluation import (
    ThresholdMetric,
    amp_context,
    best_threshold,
    binary_metrics,
    evaluate_classifier,
    sanitize_inputs,
    sanitize_logits,
)
import math
from torch.optim.lr_scheduler import _LRScheduler

from src.pipelines.training import (
    maybe_data_parallel,
    mixup_batch,
    mixup_loss,
    apply_mixup_or_cutmix,
    model_state_dict,
    unwrap_model,
)
from src.plots import plot_confusion_matrix, plot_roc_auc, save_metrics_csv

logger = logging.getLogger(__name__)


class CosineAnnealingWithWarmup(_LRScheduler):
    """Cosine Annealing with linear warmup (SOTA)."""
    def __init__(self, optimizer, warmup_epochs: int, max_epochs: int, min_lr: float = 1e-6, last_epoch: int = -1):
        self.warmup_epochs = max(0, warmup_epochs)
        self.max_epochs = max(1, max_epochs)
        self.min_lr = min_lr
        super().__init__(optimizer, last_epoch)

    def get_lr(self) -> list[float]:
        epoch = self.last_epoch
        if self.warmup_epochs > 0 and epoch < self.warmup_epochs:
            return [
                self.min_lr + (base_lr - self.min_lr) * (epoch + 1) / self.warmup_epochs
                for base_lr in self.base_lrs
            ]
        else:
            progress = (epoch - self.warmup_epochs) / max(1, self.max_epochs - self.warmup_epochs)
            progress = min(max(progress, 0.0), 1.0)
            cos_decay = 0.5 * (1.0 + math.cos(math.pi * progress))
            return [
                self.min_lr + (base_lr - self.min_lr) * cos_decay
                for base_lr in self.base_lrs
            ]


def collate_fn(batch) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    imgs, labels, idxs = zip(*batch)
    return (
        torch.stack([img if isinstance(img, torch.Tensor) else torch.as_tensor(img) for img in imgs]),
        torch.tensor(labels, dtype=torch.long),
        torch.tensor(idxs, dtype=torch.long),
    )


def _device() -> torch.device:
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


def _transforms(image_size: int, augment: bool = True):
    mean = [0.485, 0.456, 0.406]
    std = [0.229, 0.224, 0.225]
    if augment:
        train_transform = transforms.Compose(
            [
                transforms.RandomResizedCrop(image_size, scale=(0.82, 1.0), ratio=(0.9, 1.1)),
                transforms.RandomHorizontalFlip(),
                transforms.RandomApply(
                    [
                        transforms.ColorJitter(
                            brightness=0.12,
                            contrast=0.12,
                            saturation=0.08,
                            hue=0.015,
                        )
                    ],
                    p=0.5,
                ),
                transforms.ToTensor(),
                transforms.Normalize(mean=mean, std=std),
            ]
        )
    else:
        train_transform = transforms.Compose(
            [
                transforms.Resize((image_size, image_size)),
                transforms.ToTensor(),
                transforms.Normalize(mean=mean, std=std),
            ]
        )

    eval_transform = transforms.Compose(
        [
            transforms.Resize((image_size, image_size)),
            transforms.ToTensor(),
            transforms.Normalize(mean=mean, std=std),
        ]
    )
    return train_transform, eval_transform


def _class_balance(
    csv_path: Path,
) -> tuple[torch.Tensor, WeightedRandomSampler, dict[int, int]]:
    df = pd.read_csv(csv_path)
    df.columns = df.columns.str.strip()
    class_counts = df["target"].value_counts().sort_index()
    counts = {int(k): int(v) for k, v in class_counts.to_dict().items()}
    total = len(df)
    loss_weights = torch.tensor(
        [total / (2.0 * counts.get(0, 1)), total / (2.0 * counts.get(1, 1))],
        dtype=torch.float,
    )
    sample_weights = (
        df["target"].map({0: 1.0 / counts.get(0, 1), 1: 1.0 / counts.get(1, 1)}).values
    )
    sampler = WeightedRandomSampler(
        weights=torch.tensor(sample_weights, dtype=torch.float),
        num_samples=len(df),
        replacement=True,
    )
    return loss_weights, sampler, counts


def _vit_safe_name() -> str:
    return "vit_scratch"


def run_vit(
    fourier: FourierMode = "none",
    epochs: int = 50,
    raw_min: bool = True,
    data_limit: int | float | None = None,
    output_root: str | Path | None = None,
    batch_size: int = 32,
    num_workers: int = 4,
    image_size: int = 224,
    patch_size: int = 16,
    hidden_size: int = 128,
    num_hidden_layers: int = 3,
    num_attention_heads: int = 4,
    dropout: float = 0.25,
    train_backbone: bool = True,
    learning_rate_classifier: float = 8e-4,
    learning_rate_backbone: float = 3e-4,
    weight_decay: float = 1e-4,
    early_stop_patience: int = 8,
    use_weighted_sampler: bool = True,
    use_class_weights: bool = True,
    label_smoothing: float = 0.0,
    threshold_metric: ThresholdMetric = "f1",
    augment: bool = True,
    seed: int = 42,
    max_grad_norm: float | None = 1.0,
    mixup_alpha: float = 0.2,
    multi_gpu: bool = True,
    # SOTA additions
    use_conv_stem: bool = False,
    pooling: str = "cls",
    drop_path_rate: float = 0.0,
    cutmix_alpha: float = 0.0,
    scheduler_type: str = "plateau",
    warmup_epochs: int = 5,
    pretrained: bool = False,
    unfreeze_last_n: int | None = None,
):
    if not logging.root.handlers:
        logging.basicConfig(
            level=logging.INFO,
            format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
        )

    torch.manual_seed(seed)
    np.random.seed(seed)

    pwd = Path.cwd()
    output_root = Path(output_root) if output_root is not None else pwd
    data_limit = np.inf if data_limit is None else data_limit
    data_dir = data_root() / ("raw_min" if raw_min else "raw")
    device = _device()
    pin_memory = device.type == "cuda"
    persistent_workers = num_workers > 0
    safe_model = "vit_pretrained" if pretrained else _vit_safe_name()
    run_dir = fourier if data_limit == np.inf else f"{fourier}_limit{data_limit}"
    model_dir = output_root / "models" / "vit" / safe_model / run_dir
    (model_dir / "weights").mkdir(parents=True, exist_ok=True)
    (model_dir / "results").mkdir(parents=True, exist_ok=True)
    best_path = model_dir / "weights" / "best_vit.pth"

    # Alinhamento espaço-frequência dinâmico agora suporta Data Augmentation perfeitamente em todos os modos Fourier
    effective_augment = augment
    train_transform, eval_transform = _transforms(image_size, augment=effective_augment)
    spatial_size = (image_size, image_size) if fourier != "none" else None
    train_ds = ImageDataset(
        file_csv=data_dir / "train.csv",
        images_dir=phase1_split_root("train"),
        transform=train_transform,
        data_limit=data_limit,
        fourier=fourier,
        spatial_size=spatial_size,
    )
    val_ds = ImageDataset(
        file_csv=data_dir / "val.csv",
        images_dir=phase1_split_root("val"),
        transform=eval_transform,
        data_limit=data_limit,
        fourier=fourier,
        spatial_size=spatial_size,
    )
    test_ds = ImageDataset(
        file_csv=data_dir / "test.csv",
        images_dir=phase1_split_root("test"),
        transform=eval_transform,
        data_limit=data_limit,
        fourier=fourier,
        spatial_size=spatial_size,
    )
    sample_x, _, _ = train_ds[0]
    in_channels = sample_x.shape[0]

    # Prevenção de dupla penalização (Sampler balanceado + Pesos na Perda)
    if use_weighted_sampler and use_class_weights:
        logger.info("Sampler balanceado ativo: desativando pesos na CrossEntropyLoss para evitar dupla penalização redundante.")
        use_class_weights = False

    loss_weights, sampler, class_counts = _class_balance(data_dir / "train.csv")
    if data_limit != np.inf or not use_weighted_sampler:
        sampler = None

    logger.info(
        "Starting ViT from scratch: mode=%s channels=%d train=%d val=%d test=%d classes=%s device=%s",
        fourier,
        in_channels,
        len(train_ds),
        len(val_ds),
        len(test_ds),
        class_counts,
        device,
    )

    loader_kwargs = {
        "batch_size": batch_size,
        "num_workers": num_workers,
        "pin_memory": pin_memory,
        "persistent_workers": persistent_workers,
        "collate_fn": collate_fn,
    }
    train_loader = DataLoader(
        train_ds,
        sampler=sampler,
        shuffle=sampler is None,
        **loader_kwargs,
    )
    val_loader = DataLoader(val_ds, shuffle=False, **loader_kwargs)
    test_loader = DataLoader(test_ds, shuffle=False, **loader_kwargs)

    if pretrained:
        if in_channels != 3:
            raise ValueError("Pretrained ViT requires RGB input (fourier='none').")
        model = PretrainedViTClassifier(num_classes=2, dropout=dropout)
        model.freeze_backbone()
        if unfreeze_last_n is not None:
            model.unfreeze_last_n_layers(unfreeze_last_n)
    else:
        model = VisionTransformerClassifier(
            num_classes=2,
            image_size=image_size,
            patch_size=patch_size,
            hidden_size=hidden_size,
            num_hidden_layers=num_hidden_layers,
            num_attention_heads=num_attention_heads,
            dropout=dropout,
            in_channels=in_channels,
            use_conv_stem=use_conv_stem,
            pooling=pooling,
            drop_path_rate=drop_path_rate,
        )
        if not train_backbone:
            model.freeze_backbone()
    model = model.to(device)
    model = maybe_data_parallel(model, device, enabled=multi_gpu)
    base_model = unwrap_model(model)

    criterion = nn.CrossEntropyLoss(
        weight=loss_weights.to(device) if use_class_weights else None,
        label_smoothing=label_smoothing,
    )
    classifier_params = list(base_model.classifier.parameters())
    classifier_ids = {id(p) for p in classifier_params}
    backbone_params = [
        p for p in base_model.parameters() if p.requires_grad and id(p) not in classifier_ids
    ]
    param_groups = [{"params": classifier_params, "lr": learning_rate_classifier}]
    if backbone_params:
        param_groups.append({"params": backbone_params, "lr": learning_rate_backbone})
    optimizer = torch.optim.AdamW(param_groups, weight_decay=weight_decay)
    if scheduler_type == "cosine":
        scheduler = CosineAnnealingWithWarmup(
            optimizer,
            warmup_epochs=warmup_epochs,
            max_epochs=epochs,
            min_lr=1e-6,
        )
    else:
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
            optimizer,
            mode="max",
            factor=0.5,
            patience=2,
        )

    best_score = -1.0
    best_val_auc = 0.0
    best_threshold_value = 0.5
    epochs_without_improvement = 0
    epochs_run = 0

    for epoch in tqdm(range(epochs), desc="Epochs"):
        model.train()
        train_loss = 0.0
        train_correct = 0
        train_total = 0
        for x, y, _ in tqdm(train_loader, desc=f"ViT train {epoch + 1}/{epochs}", leave=False):
            x = sanitize_inputs(x.to(device))
            y = y.to(device)
            x, y_a, y_b, lam = apply_mixup_or_cutmix(x, y, mixup_alpha, cutmix_alpha)
            optimizer.zero_grad(set_to_none=True)
            with amp_context(device):
                logits = sanitize_logits(model(x))
                loss = mixup_loss(criterion, logits, y_a, y_b, lam)
            loss.backward()
            if max_grad_norm is not None:
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_grad_norm)
            optimizer.step()

            train_loss += float(torch.nan_to_num(loss.detach()).item())
            train_correct += (logits.argmax(1) == y).sum().item()
            train_total += y.size(0)

        train_loss /= max(len(train_loader), 1)
        train_acc = train_correct / max(train_total, 1)
        val_metrics = evaluate_classifier(model, val_loader, criterion, device)
        threshold, selection_score = best_threshold(
            val_metrics["y_true"],
            val_metrics["probs"],
            metric=threshold_metric,
        )
        if scheduler_type == "cosine":
            scheduler.step()
        else:
            scheduler.step(selection_score)
        epochs_run = epoch + 1

        logger.info(
            "Epoch %d/%d | train_loss=%.4f train_acc=%.4f | val_loss=%.4f val_acc=%.4f val_auc=%.4f threshold=%.4f score=%.4f",
            epoch + 1,
            epochs,
            train_loss,
            train_acc,
            val_metrics["loss"],
            val_metrics["acc"],
            val_metrics["auc"],
            threshold,
            selection_score,
        )

        if selection_score > best_score:
            best_score = selection_score
            best_val_auc = val_metrics["auc"]
            best_threshold_value = threshold
            epochs_without_improvement = 0
            torch.save(model_state_dict(model), best_path)
        else:
            epochs_without_improvement += 1
            if epochs_without_improvement >= early_stop_patience:
                logger.info("Early stopping at epoch %d.", epoch + 1)
                break

    unwrap_model(model).load_state_dict(
        torch.load(best_path, map_location=device, weights_only=True)
    )
    test_results = evaluate_classifier(
        model,
        test_loader,
        criterion,
        device,
        threshold=best_threshold_value,
    )
    np.savez_compressed(
        model_dir / "results" / "outputs.npz",
        logits=test_results["logits"],
        ids=test_results["ids"],
        probs=test_results["probs"],
        y_true=test_results["y_true"],
        y_pred=test_results["y_pred"],
        threshold=best_threshold_value,
    )
    torch.save(model_state_dict(model), model_dir / "weights" / "vit.pth")

    plot_confusion_matrix(test_results, str(model_dir), "ViT Confusion Matrix")
    plot_roc_auc(test_results, str(model_dir), "ViT ROC-AUC Curve")
    save_metrics_csv(
        test_results,
        str(model_dir),
        extra_info={
            "model": safe_model,
            "architecture": safe_model,
            "fourier": fourier,
            "in_channels": in_channels,
            "image_size": image_size,
            "patch_size": patch_size,
            "hidden_size": hidden_size,
            "num_hidden_layers": num_hidden_layers,
            "num_attention_heads": num_attention_heads,
            "epochs_requested": epochs,
            "epochs_run": epochs_run,
            "best_val_auc": best_val_auc,
            "threshold": best_threshold_value,
            "threshold_metric": threshold_metric,
            "train_backbone": train_backbone,
            "use_weighted_sampler": use_weighted_sampler,
            "use_class_weights": use_class_weights,
            "label_smoothing": label_smoothing,
            "dropout": dropout,
            "mixup_alpha": mixup_alpha,
            "learning_rate_classifier": learning_rate_classifier,
            "learning_rate_backbone": learning_rate_backbone,
            "augment": effective_augment,
            # SOTA additions
            "use_conv_stem": use_conv_stem,
            "pooling": pooling,
            "drop_path_rate": drop_path_rate,
            "cutmix_alpha": cutmix_alpha,
            "scheduler_type": scheduler_type,
            "warmup_epochs": warmup_epochs,
        },
    )
    logger.info(
        "ViT test: acc=%.4f precision=%.4f recall=%.4f f1=%.4f auc=%.4f specificity=%.4f",
        test_results["acc"],
        test_results["precision"],
        test_results["recall"],
        test_results["f1"],
        test_results["auc"],
        test_results["specificity"],
    )
    return test_results
