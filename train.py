from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader, WeightedRandomSampler
from torchvision import transforms

from src.data.data import ImageDataset
from src.data.paths import data_root, models_root, phase1_split_root
from src.models.registry import get_model_spec
from src.pipelines.config import TrainingConfig, load_config
from src.pipelines.training import Trainer, seed_everything


def _transform(config: TrainingConfig, train: bool):
    operations = []
    if train and config.augment and config.fourier_mode == "none":
        operations.extend([
            transforms.RandomResizedCrop(config.image_size, scale=(.8, 1.0)),
            transforms.RandomHorizontalFlip(),
            transforms.ColorJitter(brightness=.15, contrast=.15, saturation=.1),
        ])
    else:
        operations.append(transforms.Resize((config.image_size, config.image_size)))
    operations.extend([
        transforms.ToTensor(),
        transforms.Normalize([.485, .456, .406], [.229, .224, .225]),
    ])
    return transforms.Compose(operations)


def _balanced_sampler(dataset: ImageDataset, seed: int) -> WeightedRandomSampler:
    labels = dataset.df.iloc[:, 1].astype(int).to_numpy()
    counts = np.bincount(labels, minlength=2)
    class_weights = np.divide(1.0, counts, out=np.zeros(2, dtype=float), where=counts > 0)
    sample_weights = torch.as_tensor(class_weights[labels], dtype=torch.double)
    generator = torch.Generator().manual_seed(seed)
    return WeightedRandomSampler(sample_weights, len(sample_weights), replacement=True, generator=generator)


def build_loaders(config: TrainingConfig):
    limit = np.inf if config.data_limit is None else config.data_limit
    split_root = data_root() / config.data_split_dir
    datasets = {
        split: ImageDataset(
            split_root / f"{split}.csv", phase1_split_root(split),
            transform=_transform(config, split == "train"), data_limit=limit,
            fourier=config.fourier_mode, spatial_size=(config.image_size, config.image_size),
        )
        for split in ("train", "val", "test")
    }
    common = {
        "batch_size": config.batch_size,
        "num_workers": config.num_workers,
        "pin_memory": torch.cuda.is_available(),
    }
    train_options = (
        {"sampler": _balanced_sampler(datasets["train"], config.seed)}
        if config.use_weighted_sampler
        else {"shuffle": True}
    )
    return {
        "train": DataLoader(
            datasets["train"],
            persistent_workers=(config.num_workers > 0),
            **train_options,
            **common,
        ),
        "val": DataLoader(
            datasets["val"],
            shuffle=False,
            persistent_workers=False,
            **common,
        ),
        "test": DataLoader(
            datasets["test"],
            shuffle=False,
            persistent_workers=False,
            **common,
        ),
    }


def train_from_config(config_path, fourier=None, regime=None, seed=None, epochs=None,
                      data_limit=None, raw_min=None, multi_gpu=None, num_workers=None):
    # multi_gpu=None (e não True) para que `multi_gpu: false` no YAML seja respeitado:
    # load_config trata todo override não-None como explícito. run_tasks_on_gpus
    # continua forçando multi_gpu=False por worker.
    config = load_config(config_path, {
        "fourier_mode": fourier, "regime": regime, "seed": seed, "epochs": epochs,
        "data_limit": data_limit, "raw_min": raw_min, "multi_gpu": multi_gpu,
        "num_workers": num_workers,
    })
    seed_everything(config.seed)
    spec = get_model_spec(config.model_family)
    model = spec.build(config)
    if not config.train_backbone:
        spec.freeze_backbone(model)
    elif config.regime == "finetune":
        spec.unfreeze_for_finetune(model, config.unfreeze_last_n)
    loaders = build_loaders(config)
    output = models_root() / config.model_family / config.fourier_mode / config.regime / f"seed_{config.seed}"
    return Trainer(model, loaders["train"], loaders["val"], loaders["test"], config, output, spec).fit()


def main(argv=None):
    parser = argparse.ArgumentParser(description="Train one configured run")
    parser.add_argument("--config", required=True, type=Path)
    parser.add_argument("--fourier")
    parser.add_argument("--regime", choices=("scratch", "finetune"))
    parser.add_argument("--seed", type=int)
    parser.add_argument("--epochs", type=int)
    parser.add_argument("--data-limit", type=int)
    parser.add_argument("--raw-min", action=argparse.BooleanOptionalAction, default=None)
    parser.add_argument("--num-workers", type=int, default=None)
    args = parser.parse_args(argv)
    train_from_config(
        args.config,
        args.fourier,
        args.regime,
        args.seed,
        args.epochs,
        args.data_limit,
        args.raw_min,
        num_workers=args.num_workers,
    )


if __name__ == "__main__":
    main()
