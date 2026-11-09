"""Immutable penultimate-feature caches for verified legacy checkpoints."""
from __future__ import annotations

import json
import shutil
import tempfile
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from filelock import FileLock
from torch.utils.data import DataLoader

from src.robustness.imaging import CanonicalDataset
from src.robustness.legacy_encoding import encode_legacy_tensor
from src.robustness.manifests import load_manifest
from src.robustness.provenance import contained, digest, digest_file, source_identity, write_json


@dataclass
class FeatureStore:
    features: np.ndarray
    logits: np.ndarray
    frame: pd.DataFrame
    identity: str
    metadata: dict


def open_cache(path, expected_identity=None):
    root = Path(path)
    meta = json.loads((root / "cache.json").read_text())
    if meta.get("schema") != "faceforgery-features-v1" or meta.get("status") != "complete":
        raise ValueError("Incomplete or unknown feature cache")
    identity = digest(meta["key"])
    if identity != meta.get("identity") or (expected_identity is not None and identity != expected_identity):
        raise ValueError("Feature cache key mismatch")
    expected_files = {"features.npy", "logits.npy", "rows.csv"}
    if set(meta.get("files", {})) != expected_files:
        raise ValueError("Feature cache inventory mismatch")
    for name, checksum in meta["files"].items():
        if digest_file(root / name) != checksum:
            raise ValueError(f"Feature cache artifact changed: {name}")
    frame = pd.read_csv(root / "rows.csv", keep_default_na=False)
    if frame.empty or not {"sample_id", "group_id", "label", "image_sha256"} <= set(frame):
        raise ValueError("Missing feature cache row identity")
    features = np.load(root / "features.npy", mmap_mode="r", allow_pickle=False)
    logits = np.load(root / "logits.npy", mmap_mode="r", allow_pickle=False)
    if (features.ndim != 2 or len(features) != len(frame) or logits.shape != (len(frame), 2)
            or features.dtype != np.float16 or logits.dtype != np.float32
            or frame.sample_id.duplicated().any()):
        raise ValueError("Feature cache shape, dtype or identity mismatch")
    for start in range(0, len(frame), 4096):
        if not np.isfinite(features[start:start + 4096]).all() or not np.isfinite(logits[start:start + 4096]).all():
            raise ValueError("Nonfinite feature cache")
    if digest(frame.sample_id.astype(str).tolist()) != meta["sample_ids_sha256"]:
        raise ValueError("Feature cache row ordering changed")
    return FeatureStore(features, logits, frame, identity, meta)


def final_linear(model):
    layers = [(name, module) for name, module in model.named_modules()
              if isinstance(module, torch.nn.Linear)
              and (name in {"classifier", "fc", "head"} or name.startswith(("classifier.", "fc.", "head.")))]
    if not layers or layers[-1][1].out_features != 2:
        raise ValueError("Expected a final root classifier Linear with two outputs")
    return layers[-1]


def extract_features(checkpoint, manifest, root, cache_root, *, device="cpu",
                     batch_size=8, workers=0, use_amp=True):
    """Cache final-linear inputs and original logits, aligned to certified IDs.

    The representation is exactly the original RGB/SRM preprocessing. A caller
    must verify any checkpoint on held-out source data before scientific reuse.
    """
    from src.pipelines.checkpoints import run_from_checkpoint, config_from_run, load_model_from_run

    if batch_size < 1 or workers < 0:
        raise ValueError("Invalid feature loader settings")
    checkpoint, manifest, cache_root = Path(checkpoint), Path(manifest), Path(cache_root)
    frame, record = load_manifest(manifest)
    run = run_from_checkpoint(checkpoint)
    config = config_from_run(run)
    if config.fourier_mode not in {"none", "srm"}:
        raise ValueError("Initial verified feature cache supports RGB and legacy SRM only")
    device = torch.device(device)
    if device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("Requested GPU is unavailable")
    amp = bool(use_amp and device.type == "cuda")
    amp_dtype = torch.bfloat16 if amp and torch.cuda.is_bf16_supported() else torch.float16
    inventory = []
    for row in frame.itertuples(index=False):
        inventory.append([str(row.sample_id), digest_file(contained(root, row.img_name))])
    repo = Path(__file__).resolve().parents[2]
    code_paths = [Path(__file__), repo / "src/data/data.py", repo / "src/forensics/srm.py",
                  repo / "src/robustness/legacy_encoding.py", repo / "src/pipelines/checkpoints.py"]
    software = source_identity()
    extraction = {"mode": config.fourier_mode, "image_size": config.image_size,
                  "channels": config.in_channels, "resize": "PIL bilinear square",
                  "normalization": {"mean": [.485, .456, .406], "std": [.229, .224, .225]},
                  "srm_on_normalized": config.fourier_mode == "srm",
                  "feature_point": "input to final root classifier Linear in eval mode",
                  "features_dtype": "float16", "logits_dtype": "float32",
                  "amp_dtype": str(amp_dtype) if amp else None,
                  "checkpoint_class1": "fake"}
    key = {"checkpoint_sha256": digest_file(checkpoint),
           "run_config_sha256": digest_file(run.run_dir / "results/run_config.json"),
           "manifest_sha256": record["manifest_sha256"],
           "image_inventory_sha256": digest(inventory), "extraction": extraction,
           "code_sha256": {str(p.relative_to(repo)): digest_file(p) for p in code_paths},
           "commit": software["commit"], "packages": software["packages"]}
    identity = digest(key)
    cache_root.mkdir(parents=True, exist_ok=True)
    output = cache_root / identity
    with FileLock(str(cache_root / f"{identity}.lock"), timeout=0):
        if output.exists():
            open_cache(output, expected_identity=identity)
            return output
        temp = Path(tempfile.mkdtemp(prefix=f".{identity}.partial-", dir=cache_root))
        hook = None
        try:
            model = load_model_from_run(run, device).eval()
            model.requires_grad_(False)
            layer_name, linear = final_linear(model)
            capture = {}

            def collect(module, args):
                capture["features"] = args[0].detach()

            hook = linear.register_forward_pre_hook(collect)
            features = np.lib.format.open_memmap(temp / "features.npy", mode="w+", dtype=np.float16,
                                                shape=(len(frame), linear.in_features))
            logits_cache = np.lib.format.open_memmap(temp / "logits.npy", mode="w+", dtype=np.float32,
                                                     shape=(len(frame), 2))
            loader = DataLoader(CanonicalDataset(frame, root, config.image_size, hash_images=True),
                                batch_size=batch_size, num_workers=workers, shuffle=False)
            observed, hashes = [], []
            offset = 0
            with torch.inference_mode():
                for batch in loader:
                    x = encode_legacy_tensor(batch["image"], config.fourier_mode, config.in_channels).to(device)
                    with torch.amp.autocast("cuda", dtype=amp_dtype, enabled=amp):
                        logits = model(x)
                    f = capture.pop("features")
                    if logits.shape != (len(x), 2) or not torch.isfinite(logits).all() or not torch.isfinite(f).all():
                        raise ValueError("Invalid extractor output")
                    with torch.amp.autocast("cuda", dtype=amp_dtype, enabled=amp):
                        rebuilt = linear(f)
                    if not torch.allclose(logits, rebuilt, atol=1e-5, rtol=1e-5):
                        raise ValueError("Chosen final-linear input does not reconstruct logits")
                    capture.clear()
                    positions = batch["index"].tolist()
                    if positions != list(range(offset, offset + len(x))):
                        raise ValueError("Feature loader changed row order")
                    features[offset:offset + len(x)] = f.cpu().to(torch.float16).numpy()
                    logits_cache[offset:offset + len(x)] = logits.float().cpu().numpy()
                    observed.extend(batch["sample_id"])
                    hashes.extend(batch["image_sha256"])
                    offset += len(x)
            if observed != frame.sample_id.astype(str).tolist() or [[i, h] for i, h in zip(observed, hashes)] != inventory:
                raise ValueError("Feature population or image bytes changed during extraction")
            features.flush()
            logits_cache.flush()
            del features, logits_cache
            rows = frame.copy()
            rows["image_sha256"] = hashes
            rows.to_csv(temp / "rows.csv", index=False)
            meta = {"schema": "faceforgery-features-v1", "status": "complete", "identity": identity,
                    "key": key, "manifest": record, "extraction": extraction,
                    "checkpoint": {"path": str(checkpoint), "sha256": key["checkpoint_sha256"],
                                   "run_config_sha256": key["run_config_sha256"]},
                    "feature_layer": layer_name, "feature_dim": linear.in_features,
                    "sample_ids_sha256": digest(observed), "software": software,
                    "files": {name: digest_file(temp / name) for name in ["features.npy", "logits.npy", "rows.csv"]}}
            write_json(temp / "cache.json", meta)
            open_cache(temp, expected_identity=identity)
            temp.rename(output)
            return output
        finally:
            if hook is not None:
                hook.remove()
            if temp.exists():
                shutil.rmtree(temp)
