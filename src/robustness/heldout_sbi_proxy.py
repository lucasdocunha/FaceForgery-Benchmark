"""Materialize paired held-out real/SBI views without fitting a model."""

from __future__ import annotations

import argparse
from collections import Counter
import inspect
import json
from pathlib import Path
import random

import numpy as np
import pandas as pd
from PIL import Image, ImageDraw
from torchvision.transforms.functional import to_pil_image

from src.experimental.sbi.data import SBIDataset
from . import imaging
from .manifests import load_manifest, save_manifest
from .provenance import digest, digest_file, source_identity, write_json


def prepare(manifest, root, landmarks, output, output_manifest, *, seed=42, image_size=224):
    """One recipe draw per accepted val-real face; bootstrap unit is the pair."""
    frame, source = load_manifest(manifest, require_both=False)
    if source["split"] != "val":
        raise ValueError("Held-out SBI materialization requires certified source validation")
    if not isinstance(seed, int) or isinstance(seed, bool) or seed < 0:
        raise ValueError("Proxy seed must be a nonnegative integer")
    if not isinstance(image_size, int) or isinstance(image_size, bool) or image_size < 32:
        raise ValueError("Proxy image_size must be an integer >=32")
    output, output_manifest = Path(output).resolve(), Path(output_manifest).resolve()
    if output.exists() or output_manifest.exists() or Path(str(output_manifest) + ".json").exists():
        raise FileExistsError("Use fresh held-out SBI image and manifest paths")
    real = frame[frame.label.eq(0)].copy()
    if real.empty:
        raise ValueError("No validation real faces")
    # This alias is private to the train-only dataset constructor. The source
    # certificate remains val; no train manifest or fit invocation is produced.
    adapter = real.copy()
    adapter["split"] = "train"
    dataset = SBIDataset(adapter, root, landmarks, manifest_sha256=source["manifest_sha256"],
                         image_size=image_size, arm="sbi", seed=seed, failure_policy="exclude",
                         mask={}, cache_images=False, post_augment=True)
    recipe = {"operator": "src.experimental.sbi.data.SBIDataset.__getitem__",
              "operator_sha256": digest_file(inspect.getfile(SBIDataset)),
              "imaging_sha256": digest_file(inspect.getfile(imaging)),
              "arm": "sbi", "epoch": 0, "seed": seed, "image_size": image_size, "mask": {},
              "post_augment": True, "post_augmentation_recipe": imaging.RECIPE,
              "randomness": "unchanged SBIDataset sample_seed streams sbi-label and post-label",
              "encoding": "torchvision to_pil_image RGB8 truncation, lossless PNG",
              "constructor_split_alias": "train in memory only; original certified population is val",
              "fitting": "none; materialization and frozen-model inference only"}
    recipe_sha = digest(recipe)
    output.mkdir(parents=True)
    rows, pairs, recipe_failures = [], [], []
    for position, row in dataset.real.iterrows():
        try:
            views = [dataset[2 * position + label] for label in (0, 1)]
            images = [to_pil_image(view["image"]) for view in views]
        except Exception as error:
            recipe_failures.append({"sample_id": row.sample_id, "img_name": row.img_name,
                                    "error": f"{type(error).__name__}: {error}"})
            continue
        pair = {"source_sample_id": row.sample_id, "source_img_name": row.img_name,
                "source_sha256": dataset.records.catalog[row.sample_id]["image_sha256"], "views": []}
        for label, (view, image) in enumerate(zip(views, images)):
            if view["base_sample_id"] != row.sample_id or view["label"] != label:
                raise ValueError("SBI recipe returned a different source or label")
            kind = "real" if label == 0 else "sbi"
            name = kind + "/" + row.sample_id + ".png"
            path = output / name
            path.parent.mkdir(parents=True, exist_ok=True)
            image.save(path, format="PNG")
            identity = digest({"source_manifest_sha256": source["manifest_sha256"],
                               "source_sample_id": row.sample_id, "view": kind, "recipe_sha256": recipe_sha})
            checksum = digest_file(path)
            rows.append({"sample_id": identity, "img_name": name, "label": label,
                         "group_id": row.sample_id, "dataset": str(row.dataset) + "-heldout-sbi-proxy",
                         "split": "pilot", "source_sample_id": row.sample_id,
                         "source_img_name": row.img_name, "sha256": checksum, "view": kind})
            pair["views"].append({"sample_id": identity, "img_name": name, "sha256": checksum})
        pair["mean_absolute_pair_difference"] = float(np.abs(
            np.asarray(images[0], dtype=float) - np.asarray(images[1], dtype=float)).mean() / 255)
        pairs.append(pair)
    failures = {"landmark": dataset.failures, "recipe": recipe_failures}
    write_json(output / "failures.json", failures)
    if not rows:
        raise ValueError("No accepted held-out real/SBI pairs; failures.json records exclusions")
    certificate = save_manifest(pd.DataFrame(rows), output_manifest, {
        "scope": "pilot", "purpose": "held-out self-blend separability, distinct from MFFI-fake transfer",
        "original_manifest": source, "original_certificate_sha256": digest_file(str(manifest) + ".json"),
        "landmark_sha256": digest_file(landmarks), "landmark_metadata": dataset.records.metadata,
        "recipe": recipe, "recipe_sha256": recipe_sha,
        "group_unit": "original source-face sample ID; real/SBI pair resampled together",
        "n_source_real": len(real), "n_accepted_sources": len(pairs), "failures": failures,
        "image_inventory_sha256": digest(pairs), "warning": "Landmark-conditioned min-val proxy; no model fitting or threshold tuning."})
    selected = random.Random(seed).sample(pairs, min(4, len(pairs)))
    sheet = Image.new("RGB", (2 * image_size, len(selected) * (image_size + 20)), "white")
    draw = ImageDraw.Draw(sheet)
    for index, pair in enumerate(selected):
        y = index * (image_size + 20)
        for column, view in enumerate(pair["views"]):
            with Image.open(output / view["img_name"]) as image:
                sheet.paste(image, (column * image_size, y + 20))
            draw.text((column * image_size + 4, y + 3),
                      ("Real " if column == 0 else "SBI ") + pair["source_sample_id"][:12], fill="black")
    sheet.save(output / "qa.png")
    record = {"schema": "faceforgery-heldout-sbi-proxy-v1", "state": "complete", "scope": "pilot",
              "recipe": recipe, "recipe_sha256": recipe_sha, "source_manifest": source,
              "source_root": str(Path(root).resolve()), "root": str(output), "manifest": str(output_manifest),
              "certificate": certificate, "pairs": pairs, "failures": failures,
              "landmark_status_counts": dict(Counter(value["status"] for value in dataset.records.catalog.values())),
              "qa_pairs": selected, "qa_sha256": digest_file(output / "qa.png"), "software": source_identity()}
    write_json(output / "artifact.json", record)
    return {"state": "complete", "root": str(output), "manifest": str(output_manifest),
            "n_source_real": len(real), "n_accepted_sources": len(pairs), "rows": len(rows),
            "landmark_status_counts": record["landmark_status_counts"], "recipe_failures": len(recipe_failures),
            "recipe_sha256": recipe_sha, "artifact_sha256": digest_file(output / "artifact.json")}


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    for key in ("manifest", "root", "landmarks", "output", "output-manifest"):
        parser.add_argument("--" + key, required=True)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--image-size", type=int, default=224)
    print(json.dumps(prepare(**vars(parser.parse_args(argv))), indent=2))
    return 0


if __name__ == "__main__":
    main()
