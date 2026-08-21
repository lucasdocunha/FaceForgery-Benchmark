from __future__ import annotations

import argparse
from pathlib import Path

import torch
from PIL import Image
from torchvision import transforms
from torchvision.utils import save_image

from src.data.data import encode_pil_image
from src.data.paths import output_root
from src.pipelines.checkpoints import config_from_run, load_model_from_run, run_from_checkpoint
from src.plots.heatmap import generate, grid, overlay


def generate_from_paths(checkpoint, image_paths, method="auto", grid_mode=False, output=None):
    run = run_from_checkpoint(checkpoint)
    config = config_from_run(run)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = load_model_from_run(run, device)
    encoded, displays = [], []
    for path in image_paths:
        image = Image.open(path).convert("RGB")
        encoded.append(encode_pil_image(image, run.fourier_mode, config.image_size))
        displays.append(transforms.ToTensor()(transforms.Resize((config.image_size, config.image_size))(image)))
    inputs = torch.stack(encoded).to(device)
    display_batch = torch.stack(displays)
    heatmaps = generate(model, run.model_family, inputs, method)
    result = grid(display_batch, heatmaps) if grid_mode else overlay(display_batch[:1], heatmaps[:1])[0]
    default_name = f"{run.model_family}_{run.fourier_mode}_{run.regime}_seed_{run.seed}.png"
    destination = Path(output or output_root() / "figures" / "heatmaps" / default_name)
    destination.parent.mkdir(parents=True, exist_ok=True)
    save_image(result, destination)
    return destination


def main(argv=None):
    parser = argparse.ArgumentParser(description="Generate Grad-CAM or attention-rollout heatmaps")
    parser.add_argument("--checkpoint", required=True, type=Path)
    parser.add_argument("--image", required=True, type=Path, nargs="+")
    parser.add_argument("--method", default="auto", choices=("auto", "gradcam", "attention"))
    parser.add_argument("--grid", action="store_true")
    parser.add_argument("--output", type=Path)
    args = parser.parse_args(argv)
    destination = generate_from_paths(args.checkpoint, args.image, args.method, args.grid, args.output)
    print(destination)


if __name__ == "__main__":
    main()
