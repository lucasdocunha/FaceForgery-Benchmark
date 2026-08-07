from __future__ import annotations
import argparse
from pathlib import Path
import numpy as np
from torch.utils.data import DataLoader
from torchvision import transforms
from src.data.data import ImageDataset
from src.data.paths import data_root, models_root, phase1_split_root
from src.models.registry import get_model_spec
from src.pipelines.config import load_config
from src.pipelines.training import Trainer, seed_everything

def train_from_config(config_path, fourier=None, regime=None, seed=None, epochs=None, data_limit=None, multi_gpu=True):
    config = load_config(config_path, {"fourier_mode":fourier, "regime":regime, "seed":seed, "epochs":epochs, "data_limit":data_limit, "multi_gpu":multi_gpu})
    seed_everything(config.seed); spec = get_model_spec(config.model_family); model = spec.build(config)
    if config.regime == "finetune": spec.unfreeze_for_finetune(model, config.unfreeze_last_n)
    transform = transforms.Compose([transforms.Resize((config.image_size, config.image_size)), transforms.ToTensor(), transforms.Normalize([.485,.456,.406],[.229,.224,.225])])
    limit = np.inf if config.data_limit is None else config.data_limit
    loaders = {}
    for split in ("train", "val", "test"):
        ds = ImageDataset(data_root()/"raw_min"/f"{split}.csv", phase1_split_root(split), transform=transform, data_limit=limit, fourier=config.fourier_mode, spatial_size=(config.image_size, config.image_size))
        loaders[split] = DataLoader(ds, batch_size=config.batch_size, shuffle=split=="train", num_workers=config.num_workers)
    out = models_root()/config.model_family/config.fourier_mode/config.regime/f"seed_{config.seed}"
    return Trainer(model, loaders["train"], loaders["val"], loaders["test"], config, out, spec).fit()

def main(argv=None):
    p=argparse.ArgumentParser(); p.add_argument("--config",required=True); p.add_argument("--fourier"); p.add_argument("--regime",choices=("scratch","finetune")); p.add_argument("--seed",type=int); p.add_argument("--epochs",type=int); p.add_argument("--data-limit",type=int); a=p.parse_args(argv)
    train_from_config(a.config,a.fourier,a.regime,a.seed,a.epochs,a.data_limit)
if __name__ == "__main__": main()
