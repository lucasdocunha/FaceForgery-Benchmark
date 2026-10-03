"""Small, deterministic face-mask QA contact sheet; no training or selection."""
import argparse
from pathlib import Path
import random

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from src.robustness.manifests import load_manifest
from src.robustness.provenance import write_json
from .data import SBIDataset, blend, face_mask


def mask_qa(manifest, root, landmarks, output, *, seed=42, count=4, image_size=128):
    output = Path(output)
    if output.exists():
        raise FileExistsError("Use a new QA output path")
    frame, cert = load_manifest(manifest)
    ds = SBIDataset(frame, root, landmarks, manifest_sha256=cert["manifest_sha256"],
                    image_size=image_size, arm="sbi", seed=seed, post_augment=False)
    selected = random.Random(seed).sample(range(len(ds.real)), min(count, len(ds.real)))
    fig, axes = plt.subplots(len(selected), 4, figsize=(8, 2*len(selected)), squeeze=False)
    records = []
    for index, position in enumerate(selected):
        row = ds.real.iloc[position]
        original = ds._image(row)
        mask = face_mask(ds.records[row.sample_id]["landmarks"], image_size)
        fake, alpha = blend(original, mask, random.Random(seed+position))
        difference = np.abs(np.asarray(original,dtype=float)-np.asarray(fake,dtype=float))/255
        for column, (values, title) in enumerate([(original,"Genuine"),(alpha,"Blend weight"),(fake,"SBI-style"),(difference*4,"Difference x4")]):
            axes[index,column].imshow(np.clip(values,0,1) if isinstance(values,np.ndarray) and values.dtype.kind == "f" else values,
                                      cmap="gray" if column == 1 else None)
            axes[index,column].set_title(title if index == 0 else "",fontsize=10)
            axes[index,column].axis("off")
        records.append({"sample_id":row.sample_id, "mask_fraction":float((alpha > .1).mean()), "mean_abs_change":float(difference.mean())})
    fig.suptitle("Min-train SBI mask QA, seed-selected examples", fontsize=12)
    fig.tight_layout()
    output.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output, dpi=100)
    plt.close(fig)
    write_json(output.with_suffix(".json"), {"purpose":"min-dataset mask QA, no quality claim", "seed":seed,
               "landmark_source":ds.cohort["landmark_source"], "landmark_sha256":ds.cohort["landmark_sha256"],
               "forehead_note":"MediaPipe mesh hull covers less forehead than original SBI dlib-81", "examples":records})
    return str(output)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    for key in ("manifest", "root", "landmarks", "output"):
        parser.add_argument("--"+key, required=True)
    parser.add_argument("--seed",type=int,default=42)
    print(mask_qa(**vars(parser.parse_args())))
