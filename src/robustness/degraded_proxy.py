"""Materialize a fixed repository degradation recipe for paired pilot inference."""

from __future__ import annotations

import argparse
import inspect
import json
from pathlib import Path
import random

from PIL import Image
import torch
from torchvision.transforms.functional import to_pil_image

from src.data.augmentations import RandomizedRobustAugment
from .manifests import load_manifest, save_manifest
from .provenance import contained, digest, digest_file, source_identity, write_json


def prepare(manifest, root, output, clean_manifest, degraded_manifest, *, seed=42, image_size=224):
    """Keep logical identity and certify clean/degraded aliases as split=pilot."""
    if not isinstance(seed, int) or isinstance(seed, bool) or seed < 0:
        raise ValueError("Proxy seed must be a nonnegative integer")
    if not isinstance(image_size, int) or isinstance(image_size, bool) or image_size < 1:
        raise ValueError("Proxy image_size must be positive")
    frame, source = load_manifest(manifest)
    if source["split"] not in {"val", "test", "smoke_test"}:
        raise ValueError("Degraded proxy accepts source validation or frozen test populations only")
    root, output = Path(root).resolve(), Path(output).resolve()
    clean_manifest, degraded_manifest = Path(clean_manifest).resolve(), Path(degraded_manifest).resolve()
    if clean_manifest == degraded_manifest:
        raise ValueError("Clean and degraded manifests need distinct paths")
    if output.exists() or any(path.exists() or Path(str(path) + ".json").exists()
                              for path in (clean_manifest, degraded_manifest)):
        raise FileExistsError("Use fresh proxy image and manifest paths")
    if "artifact.json" in set(frame.img_name):
        raise ValueError("Proxy image path conflicts with artifact inventory")
    inputs = [{"sample_id": str(row.sample_id), "img_name": str(row.img_name),
               "source_sha256": digest_file(contained(root, row.img_name))}
              for row in frame.itertuples()]
    for row, item in zip(frame.itertuples(), inputs):
        if "sha256" in frame and str(row.sha256) != item["source_sha256"]:
            raise ValueError("Source image bytes differ from the certified image inventory")
    operator = Path(inspect.getfile(RandomizedRobustAugment)).resolve()
    recipe = {"operator": "src.data.augmentations.RandomizedRobustAugment.transform_to_tensor",
              "operator_sha256": digest_file(operator), "image_size": image_size, "seed": seed,
              "sample_seed": "int(SHA256(canonical JSON {seed,sample_id})[:16],16) modulo 2**32",
              "random_streams": ["python random", "torch CPU default generator"],
              "normalization": "none; operator returns RGB tensor in [0,1]",
              "encoding": "torchvision to_pil_image RGB8 truncation; lossless PNG with original img_name retained",
              "selection": "one draw per sample, no severity or label-based selection"}
    common = {"scope": "pilot", "original_manifest": source,
              "original_certificate_sha256": digest_file(str(manifest) + ".json"),
              "source_image_inventory_sha256": digest(inputs),
              "group_unit": source.get("group_unit", "supplied groups"),
              "warning": "Deterministic min-population proxy, not the original Test-D distribution or untouched validation."}
    output.mkdir(parents=True)
    augment = RandomizedRobustAugment(image_size=image_size)
    inventory = []
    for item in inputs:
        sample_seed = int(digest({"seed": seed, "sample_id": item["sample_id"]})[:16], 16) % (2 ** 32)
        state = random.getstate()
        try:
            with torch.random.fork_rng(devices=[]):
                random.seed(sample_seed)
                torch.random.default_generator.manual_seed(sample_seed)
                with Image.open(contained(root, item["img_name"])) as image:
                    tensor = augment.transform_to_tensor(image.convert("RGB"))
                result = to_pil_image(tensor)
        finally:
            random.setstate(state)
        destination = contained(output, item["img_name"])
        destination.parent.mkdir(parents=True, exist_ok=True)
        result.save(destination, format="PNG")
        inventory.append({**item, "sample_seed": sample_seed, "degraded_sha256": digest_file(destination)})
    clean, degraded = frame.copy(), frame.copy()
    clean["split"] = degraded["split"] = "pilot"
    clean["sha256"] = [item["source_sha256"] for item in inventory]
    degraded["sha256"] = [item["degraded_sha256"] for item in inventory]
    if "image_sha256" in frame:
        clean["image_sha256"], degraded["image_sha256"] = clean.sha256, degraded.sha256
    clean_record = save_manifest(clean, clean_manifest, {**common, "proxy_label": clean_manifest.stem,
                                                       "paired_degradation_recipe_sha256": digest(recipe)})
    degraded_record = save_manifest(degraded, degraded_manifest, {**common, "proxy_label": output.name,
        "recipe": recipe, "recipe_sha256": digest(recipe), "clean_manifest_sha256": clean_record["manifest_sha256"],
        "image_inventory_sha256": digest(inventory)})
    record = {"schema": "faceforgery-degraded-proxy-v1", "state": "complete", "scope": "pilot",
              "label": output.name, "recipe": recipe, "recipe_sha256": digest(recipe),
              "source_manifest": source, "source_certificate_sha256": common["original_certificate_sha256"],
              "source_root": str(root), "root": str(output), "images": inventory,
              "clean_manifest": str(clean_manifest), "degraded_manifest": str(degraded_manifest),
              "clean_certificate": clean_record, "degraded_certificate": degraded_record,
              "software": source_identity()}
    write_json(output / "artifact.json", record)
    return {"state": "complete", "scope": "pilot", "root": str(output), "rows": len(frame),
            "clean_manifest": str(clean_manifest), "degraded_manifest": str(degraded_manifest),
            "recipe_sha256": digest(recipe), "artifact_sha256": digest_file(output / "artifact.json")}


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    for key in ("manifest", "root", "output", "clean-manifest", "degraded-manifest"):
        parser.add_argument("--" + key, required=True)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--image-size", type=int, default=224)
    result = prepare(**vars(parser.parse_args(argv)))
    print(json.dumps(result, indent=2))
    return 0


if __name__ == "__main__":
    main()
