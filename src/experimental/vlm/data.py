"""Fail-closed raw-image loading for canonical VLM manifests."""

from __future__ import annotations

import pandas as pd
from PIL import Image, ImageFile
from torch.utils.data import Dataset

ImageFile.LOAD_TRUNCATED_IMAGES = True

from src.robustness.manifests import validate
from src.robustness.provenance import contained, digest_file


def read_image(root, row, *, hash_images=True):
    path = contained(root, str(row["img_name"]))
    checksum = digest_file(path) if hash_images else ""
    expected = str(row.get("sha256", ""))
    if expected and checksum and checksum != expected:
        raise ValueError(f"Image differs from declared SHA-256: {row['sample_id']}")
    with Image.open(path) as source:
        image = source.convert("RGB")
        image.load()
    return image, checksum


def balanced_subset(frame, limit, seed):
    if limit is None:
        return frame.reset_index(drop=True)
    if int(limit) < 2 or int(limit) % 2:
        raise ValueError("A balanced VLM subset requires a positive even sample count")
    each = int(limit) // 2
    if any(int((frame.label == label).sum()) < each for label in (0, 1)):
        raise ValueError("Not enough images per class for the requested balanced subset")
    result = pd.concat([frame[frame.label == label].sort_values("sample_id").sample(n=each, random_state=int(seed))
                        for label in (0, 1)])
    return result.sort_values("sample_id").reset_index(drop=True)


class RawImageDataset(Dataset):
    def __init__(self, frame, root):
        self.frame, self.root = validate(frame, require_both=False), root

    def __len__(self):
        return len(self.frame)

    def __getitem__(self, index):
        row = self.frame.iloc[index].to_dict()
        image, _ = read_image(self.root, row)
        return {"image": image, "label": int(row["label"])}
