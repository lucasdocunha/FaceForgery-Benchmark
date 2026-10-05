"""Deterministic paired image views and shared native-domain encodings.

All spectral views are derived AFTER the same image-space augmentation. No
missing image is substituted. The FFT phase matches ImageDataset, not its older
standalone visualization helper. Images and stored maps are never synthesized as
experimental results by this module.
"""

from __future__ import annotations
import hashlib
import io
import random
from pathlib import Path
import numpy as np
from PIL import Image, ImageEnhance, ImageFile, ImageFilter, ImageOps
ImageFile.LOAD_TRUNCATED_IMAGES = True
import torch
from torch.utils.data import Dataset
from .provenance import contained, digest_file
from .manifests import validate

PREPROCESSING = "bilinear-square_imagenet_centered-luminance-fft_v1"
MEAN = (0.485, 0.456, 0.406)
STD = (0.229, 0.224, 0.225)
MODES = {
    "none": 3,
    "magnitude": 1,
    "phase": 1,
    "complex": 2,
    "concat": 4,
    "frequency_3": 1,
    "concat_frequency": 7,
    "srm": 6,
    "dtcwt": 9,
}
RECIPE = {
    "name": "robust_v1",
    "jpeg_probability": 0.5,
    "jpeg_quality": [30, 90],
    "blur_probability": 0.3,
    "blur_radius": [0.5, 2.0],
    "noise_probability": 0.5,
    "noise_std": [0.01, 0.06],
    "brightness_probability": 0.5,
    "brightness_factor": [0.7, 1.3],
}


def sample_seed(seed: int, epoch: int, sample_id: str, stream: str) -> int:
    value = f"{seed}:{epoch}:{sample_id}:{stream}".encode()
    return int.from_bytes(hashlib.sha256(value).digest()[:8], "big") % (2**63 - 1)


def to_tensor(image: Image.Image) -> torch.Tensor:
    return (
        torch.from_numpy(
            np.array(image.convert("RGB"), dtype=np.float32).copy()
        ).permute(2, 0, 1)
        / 255.0
    )


def normalize_rgb(x: torch.Tensor) -> torch.Tensor:
    shape = (1, 3, 1, 1) if x.ndim == 4 else (3, 1, 1)
    return (x - x.new_tensor(MEAN).reshape(shape)) / x.new_tensor(STD).reshape(shape)


def encode_tensor(
    raw: torch.Tensor, mode: str, in_channels: int | None = None
) -> torch.Tensor:
    if mode not in MODES or raw.ndim not in (3, 4) or raw.shape[-3] != 3:
        raise ValueError("Expected RGB CHW/BCHW input and a supported representation")
    single = raw.ndim == 3
    x = raw.unsqueeze(0) if single else raw
    rgb = normalize_rgb(x)
    if mode == "none":
        out = rgb
    elif mode == "srm":
        from src.forensics.srm import SRMConv2d

        residuals = SRMConv2d(mode="residual_only").to(rgb.device)(rgb)
        out = torch.cat([rgb, residuals], dim=1)
    elif mode == "dtcwt":
        from src.forensics.dtcwt_module import extract_dtcwt_features

        # Legacy datasets normalize each image separately, including each band.
        bands = torch.cat([
            extract_dtcwt_features(item[None], mode="directional_only")
            for item in rgb
        ])
        out = torch.cat([rgb, bands], dim=1)
    else:
        gray = 0.299 * x[:, 0] + 0.587 * x[:, 1] + 0.114 * x[:, 2]
        fft = torch.fft.fftshift(torch.fft.fft2(gray), dim=(-2, -1))
        magnitude = fft.abs()

        def scale(t):
            lo = t.amin((-2, -1), keepdim=True)
            span = t.amax((-2, -1), keepdim=True) - lo
            return torch.where(
                span > 1e-8, (t - lo) / span.clamp_min(1e-8), torch.zeros_like(t)
            ).unsqueeze(1)

        mag = scale(torch.log1p(magnitude)) * 2 - 1
        phase = (torch.angle(fft) / torch.pi).unsqueeze(1)
        if mode == "magnitude":
            out = mag
        elif mode == "phase":
            out = phase
        elif mode == "complex":
            den = magnitude.amax((-2, -1), keepdim=True).clamp_min(1e-8)
            out = torch.stack([fft.real / den, fft.imag / den], dim=1)
        elif mode == "concat":
            out = torch.cat([rgb, mag], dim=1)
        else:
            h, w = x.shape[-2:]
            yy, xx = torch.meshgrid(
                torch.arange(h, device=x.device),
                torch.arange(w, device=x.device),
                indexing="ij",
            )
            mask = (yy - h // 2) ** 2 + (xx - w // 2) ** 2 >= (min(h, w) * 0.12) ** 2
            high = scale(torch.log1p(magnitude * mask)) * 2 - 1
            if mode == "frequency_3":
                out = high
            else:
                low = scale(torch.log1p(magnitude * (~mask))) * 2 - 1
                out = torch.cat([rgb, mag, phase, high, low], dim=1)
    if mode == "concat_frequency" and in_channels == 6:
        out = out[:, :6]
    elif in_channels is not None and out.shape[1] != in_channels:
        raise ValueError(
            f"Representation has {out.shape[1]} channels, checkpoint expects {in_channels}"
        )
    if not torch.isfinite(out).all():
        raise ValueError("Nonfinite encoded input")
    return out[0] if single else out


def corrupt(
    image: Image.Image, rng: random.Random, generator: torch.Generator
) -> torch.Tensor:
    """The named robust_v1 recipe is fixed independently of training seed."""
    if rng.random() < RECIPE["brightness_probability"]:
        image = ImageEnhance.Brightness(image).enhance(
            rng.uniform(*RECIPE["brightness_factor"])
        )
    if rng.random() < RECIPE["blur_probability"]:
        image = image.filter(
            ImageFilter.GaussianBlur(rng.uniform(*RECIPE["blur_radius"]))
        )
    if rng.random() < RECIPE["jpeg_probability"]:
        stream = io.BytesIO()
        image.save(stream, format="JPEG", quality=rng.randint(*RECIPE["jpeg_quality"]))
        stream.seek(0)
        with Image.open(stream) as decoded:
            image = decoded.convert("RGB").copy()
    tensor = to_tensor(image)
    if rng.random() < RECIPE["noise_probability"]:
        noise = torch.randn(tensor.shape, generator=generator) * rng.uniform(
            *RECIPE["noise_std"]
        )
        tensor = (tensor + noise).clamp(0, 1)
    return tensor


class CanonicalDataset(Dataset):
    def __init__(
        self,
        frame,
        root,
        image_size: int,
        *,
        recipe: str = "none",
        seed: int = 42,
        training: bool = False,
        hash_images: bool = False,
    ):
        self.frame = validate(frame, require_both=False)
        self.root = Path(root).resolve()
        if image_size < 16 or recipe not in {"none", "basic_v1", "robust_v1"}:
            raise ValueError("Invalid image size or recipe")
        if not training and recipe != "none":
            raise ValueError(
                "Evaluation must use a separately materialized, frozen corruption manifest"
            )
        self.size, self.recipe, self.seed = image_size, recipe, int(seed)
        self.training, self.hash_images, self.epoch = training, hash_images, 0

    def __len__(self):
        return len(self.frame)

    def __getitem__(self, index):
        row = self.frame.iloc[index]
        path = contained(self.root, row.img_name)
        with Image.open(path) as opened:
            image = opened.convert("RGB").resize(
                (self.size, self.size), Image.Resampling.BILINEAR
            )
        if self.training and self.recipe != "none":
            rng = random.Random(
                sample_seed(self.seed, self.epoch, row.sample_id, "geometry")
            )
            if rng.random() < 0.5:
                image = ImageOps.mirror(image)
        clean = to_tensor(image)
        x = clean
        if self.training and self.recipe == "robust_v1":
            value = sample_seed(self.seed, self.epoch, row.sample_id, "degradation")
            x = corrupt(
                image, random.Random(value), torch.Generator().manual_seed(value)
            )
        expected = str(row.get("sha256", ""))
        image_hash = digest_file(path) if self.hash_images or expected else ""
        if expected and expected != image_hash:
            raise ValueError(f"Image bytes changed: {row.sample_id}")
        return {
            "image": x,
            "clean": clean,
            "label": int(row.label),
            "index": index,
            "sample_id": row.sample_id,
            "image_sha256": image_hash,
        }
