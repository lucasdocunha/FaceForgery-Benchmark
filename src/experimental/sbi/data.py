"""Deterministic self-blended siblings and matched genuine/fake training arms.

This independent SBI-style recipe uses a configurable landmark hull. MediaPipe
mesh hulls cover less forehead than the original SBI dlib-81 implementation.
"""
from __future__ import annotations

import json
import random
from collections import Counter
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw, ImageEnhance, ImageFile, ImageFilter, ImageOps
ImageFile.LOAD_TRUNCATED_IMAGES = True
from torch.utils.data import Dataset

from src.robustness.imaging import corrupt, sample_seed, to_tensor
from src.robustness.provenance import contained, digest, digest_file
from .cache import LandmarkStore


def convex_hull(points):
    points = sorted(set(map(tuple, points)))
    if len(points) < 3:
        raise ValueError("A face hull needs at least three distinct points")
    def cross(o, a, b):
        return (a[0]-o[0])*(b[1]-o[1]) - (a[1]-o[1])*(b[0]-o[0])
    lower, upper = [], []
    for p in points:
        while len(lower) >= 2 and cross(lower[-2], lower[-1], p) <= 0:
            lower.pop()
        lower.append(p)
    for p in reversed(points):
        while len(upper) >= 2 and cross(upper[-2], upper[-1], p) <= 0:
            upper.pop()
        upper.append(p)
    hull = lower[:-1] + upper[:-1]
    if len(hull) < 3:
        raise ValueError("Degenerate landmark hull")
    return hull


def face_mask(landmarks, size, *, indices=None, forehead_extension=0.0):
    points = np.asarray(landmarks, dtype=float)
    if points.ndim != 2 or points.shape[1] < 2 or not np.isfinite(points).all():
        raise ValueError("Invalid landmark geometry")
    points = points[:, :2] if indices is None else points[np.asarray(indices, dtype=int), :2]
    if not 0 <= forehead_extension <= .2:
        raise ValueError("forehead_extension must be in [0,.2] face heights")
    points = points.copy()
    top, bottom = points[:, 1].min(), points[:, 1].max()
    points[points[:, 1] <= top + .25 * (bottom-top), 1] -= forehead_extension * (bottom-top)
    hull = convex_hull(np.clip(points, 0, 1) * (size-1))
    image = Image.new("L", (size, size), 0)
    ImageDraw.Draw(image).polygon(hull, fill=255)
    area = np.asarray(image).mean()/255
    if not .005 < area < .95:
        raise ValueError("Landmark hull has implausible face coverage")
    return image


def blend(image, mask, rng):
    """Create affine/color/resampling discrepancies with a softened boundary."""
    size = image.width
    source = ImageEnhance.Color(image).enhance(rng.uniform(.65, 1.4))
    source = ImageEnhance.Contrast(source).enhance(rng.uniform(.75, 1.3))
    source = ImageEnhance.Brightness(source).enhance(rng.uniform(.8, 1.2))
    if rng.random() < .5:
        reduced = max(8, int(size * rng.uniform(.35, .85)))
        source = source.resize((reduced, reduced), Image.Resampling.BILINEAR).resize(image.size, Image.Resampling.BILINEAR)
    else:
        source = ImageEnhance.Sharpness(source).enhance(rng.uniform(1.5, 3.0))
    scale, shear = rng.uniform(.97, 1.03), rng.uniform(-.02, .02)
    transform = (scale, shear, rng.uniform(-.025, .025)*size,
                 -shear, scale, rng.uniform(-.025, .025)*size)
    source = source.transform(image.size, Image.Transform.AFFINE, transform, Image.Resampling.BILINEAR)
    mask = mask.transform(image.size, Image.Transform.AFFINE, transform, Image.Resampling.BILINEAR)
    mask = mask.filter(ImageFilter.GaussianBlur(rng.uniform(.008, .04)*size))
    alpha = np.asarray(mask, dtype=np.float32)[..., None]/255 * rng.uniform(.6, 1.0)
    value = alpha*np.asarray(source, dtype=np.float32) + (1-alpha)*np.asarray(image, dtype=np.float32)
    return Image.fromarray(np.rint(value).clip(0, 255).astype(np.uint8)), alpha[..., 0]


class SBIDataset(Dataset):
    def __init__(self, frame, root, landmarks, *, manifest_sha256, image_size=224,
                 arm="sbi", seed=42, failure_policy="error", mask=None,
                 cache_images=False, post_augment=True):
        if arm not in {"sbi", "mffi", "mixed"} or failure_policy not in {"error", "exclude"}:
            raise ValueError("Declare SBI/MFFI/mixed arm and landmark failure policy")
        if set(frame.split) != {"train"} or image_size < 32:
            raise ValueError("SBI fitting requires source train and image_size >=32")
        self.records = LandmarkStore(landmarks)
        cache = self.records.metadata
        if (cache.get("schema") != "faceforgery-landmarks-v1" or cache.get("coordinates") != "normalized_xy"
                or cache.get("manifest_sha256") != manifest_sha256):
            raise ValueError("Landmark cache does not bind this source manifest")
        records = self.records.catalog.values()
        real = frame[frame.label.eq(0)].sort_values("sample_id").copy()
        if set(real.sample_id) != set(self.records):
            raise ValueError("Landmark population differs from genuine training population")
        self.failures = {r["sample_id"]: r["status"] for r in records if r["status"] != "ok"}
        if self.failures and failure_policy == "error":
            raise ValueError(f"Landmark failures: {dict(Counter(self.failures.values()))}; explicit exclusion required")
        self.real = real[~real.sample_id.isin(self.failures)].reset_index(drop=True)
        self.fake = frame[frame.label.eq(1)].sort_values("sample_id").reset_index(drop=True)
        if self.real.empty or (arm != "sbi" and self.fake.empty):
            raise ValueError("Empty effective training class")
        self.root, self.size, self.arm, self.seed = Path(root), image_size, arm, int(seed)
        self.epoch, self.post_augment = 0, bool(post_augment)
        self.images, self.masks, self.mask_config = {}, {}, mask or {}
        self.image_hashes = {}
        selected = self.real if arm == "sbi" else frame[frame.sample_id.isin(set(self.real.sample_id) | set(self.fake.sample_id))]
        for row in selected.itertuples(index=False):
            path = contained(root, row.img_name)
            checksum = digest_file(path)
            if getattr(row, "sha256", "") and row.sha256 != checksum:
                raise ValueError("Training image differs from its declared manifest hash")
            self.image_hashes[row.sample_id] = checksum
            if row.label == 0:
                record = self.records.catalog[row.sample_id]
                if record["label"] != 0 or record["image_sha256"] != checksum:
                    raise ValueError("Landmark image or label identity changed")
                if cache_images:
                    self.masks[row.sample_id] = face_mask(self.records[row.sample_id]["landmarks"], image_size, **self.mask_config)
            if cache_images:
                with Image.open(path) as image:
                    self.images[row.sample_id] = image.convert("RGB").resize((image_size,image_size),Image.Resampling.BILINEAR)
        self.cohort = {"arm": arm, "real_ids_sha256": digest(self.real.sample_id.tolist()),
                       "n_unique_real": len(self.real), "n_available_mffi_fake": len(self.fake) if arm != "sbi" else 0,
                       "examples_per_epoch": len(self), "landmark_source": cache["source"],
                       "landmark_sha256": digest_file(landmarks), "failures": self.failures,
                       "failure_policy": failure_policy, "recipe": "sbi_landmark_hull_v1",
                       "image_inventory_sha256": digest(sorted(self.image_hashes.items())),
                       "fake_exposure": "SBI only" if arm == "sbi" else "MFFI fake training images"}
        self.set_epoch(0)

    def __len__(self):
        return 2*len(self.real)

    def set_epoch(self, epoch):
        self.epoch = int(epoch)
        self.fake_order = list(range(len(self.fake)))
        random.Random(sample_seed(self.seed, self.epoch, "population", "fake-order")).shuffle(self.fake_order)

    def _image(self, row):
        if row.sample_id in self.images:
            return self.images[row.sample_id].copy()
        path = contained(self.root, row.img_name)
        if digest_file(path) != self.image_hashes[row.sample_id]:
            raise ValueError("Training image bytes changed during this run")
        with Image.open(path) as image:
            return image.convert("RGB").resize((self.size, self.size),Image.Resampling.BILINEAR)

    def __getitem__(self, index):
        if index < 0 or index >= len(self):
            raise IndexError(index)
        # Every real has one fake training slot. MFFI fake slots rotate each epoch.
        position, label = index // 2, index % 2
        row = self.real.iloc[position]
        kind = "real"
        use_sbi = self.arm == "sbi" or (self.arm == "mixed" and (position+self.epoch) % 2 == 0)
        if label and not use_sbi:
            row = self.fake.iloc[self.fake_order[position % len(self.fake)]]
            kind = "mffi"
        image = self._image(row)
        rng = random.Random(sample_seed(self.seed, self.epoch, str(row.sample_id), f"sbi-{label}"))
        if label and use_sbi:
            mask = self.masks.get(row.sample_id)
            if mask is None:
                mask = face_mask(self.records[row.sample_id]["landmarks"], self.size, **self.mask_config)
            image, _ = blend(image, mask, rng)
            kind = "sbi"
        if rng.random() < .5:
            image = ImageOps.mirror(image)
        if self.post_augment:
            import torch
            value = sample_seed(self.seed, self.epoch, str(row.sample_id), f"post-{label}")
            tensor = corrupt(image, random.Random(value), torch.Generator().manual_seed(value))
        else:
            tensor = to_tensor(image)
        return {"image": tensor, "label": label, "base_sample_id": str(row.sample_id),
                "group_id": str(row.group_id), "kind": kind}
