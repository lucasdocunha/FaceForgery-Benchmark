"""Explicitly invoked image experiments. Preparing a plan never imports this module."""
from __future__ import annotations
import hashlib
import importlib.metadata
import json
import os
from pathlib import Path
import time
import numpy as np
from PIL import Image, ImageFilter
import torch
from torchvision import transforms
from src.data.data import ImageDataset
from src.pipelines.checkpoints import config_from_run, load_model_from_run, run_from_checkpoint
from .contracts import canonical, contained, digest, freeze, sha256
from .methods import UnsupportedMethod, explain, perturbation_curves, seeded
from .prepare import read_plan


class EvaluationEncoder(ImageDataset):
    """Reuse ICLR's FFT implementations, but never its corrupt-image substitution.

    The legacy standalone encode_pil_image helper min-max rescales phase whereas
    ImageDataset does not. This encoder follows the evaluation dataset, preserves
    six/seven-channel historical concat_frequency checkpoints, and fails on shape
    mismatches. The ICLR training/evaluation code remains untouched.
    """
    def __init__(self, mode: str, size: int, channels: int):
        self.fourier, self.size, self.channels = mode, size, channels
        self.resize = transforms.Resize((size, size))
        self.to_tensor = transforms.ToTensor()
        self.normalize = transforms.Normalize([.485, .456, .406], [.229, .224, .225])
        self.frequency_normalize = transforms.Normalize([.5], [.5])

    def encode(self, image: Image.Image):
        raw = self.to_tensor(self.resize(image.convert("RGB")))
        rgb = self.normalize(raw)
        f = self.frequency_normalize
        if self.fourier == "none":
            x = rgb
        elif self.fourier == "magnitude":
            x = f(self._fft_magnitude(raw))
        elif self.fourier == "phase":
            x = f(self._fft_phase(raw))
        elif self.fourier == "complex":
            x = self._fft_complex(raw)
        elif self.fourier == "frequency_3":
            x = f(self._fft_highpass(raw))
        elif self.fourier == "concat":
            x = torch.cat((rgb, f(self._fft_magnitude(raw))))
        elif self.fourier == "concat_frequency":
            if self.channels not in {6, 7}:
                raise ValueError("concat_frequency requires the recorded six or seven channels")
            x = torch.cat((rgb, f(self._fft_magnitude(raw)), f(self._fft_phase(raw)),
                           f(self._fft_highpass(raw)), f(self._fft_lowpass(raw))))[:self.channels]
        else:
            raise ValueError(f"Unsupported representation: {self.fourier}")
        if x.shape != (self.channels, self.size, self.size) or not torch.isfinite(x).all():
            raise ValueError("Encoded tensor differs from recorded input dimensions or is non-finite")
        return x, raw


def _save_npz(path, **arrays):
    """Atomic compressed numeric artifact; pickle/object arrays are forbidden."""
    path = Path(path); path.parent.mkdir(parents=True, exist_ok=True)
    for key, array in arrays.items():
        if np.asarray(array).dtype.hasobject:
            raise ValueError(f"Object arrays are forbidden: {key}")
    if path.exists():
        with np.load(path, allow_pickle=False) as old:
            if set(old.files) != set(arrays) or any(not np.array_equal(old[k], v) for k, v in arrays.items()):
                raise ValueError(f"Existing array artifact differs: {path}")
        return
    temporary = path.with_name(path.name + f".{os.getpid()}.partial")
    try:
        with temporary.open("wb") as f:
            np.savez_compressed(f, **arrays)
        # A competing shard must produce identical assets, never a different overwrite.
        try:
            os.link(temporary, path)
        except FileExistsError:
            _save_npz(path, **arrays)
    finally:
        temporary.unlink(missing_ok=True)


def _versions():
    versions = {}
    for name in ("torch", "torchvision", "numpy", "pandas", "captum", "timm", "transformers", "scikit-learn", "Pillow"):
        try:
            versions[name] = importlib.metadata.version(name)
        except importlib.metadata.PackageNotFoundError:
            versions[name] = "not-installed"
    return versions


def _source_hash():
    root = Path(__file__).resolve().parents[2]
    sources = {str(p.relative_to(root)): sha256(p) for p in sorted((root / "src").rglob("*.py"))}
    return digest(sources)


def cell_seed(plan, task, model, method, sample_id, replicate):
    token = f"{plan['config']['seed']}:{task}:{model}:{method}:{sample_id}:{replicate}"
    return int.from_bytes(hashlib.sha256(token.encode()).digest()[:4], "big")


def _head_randomization(model, family, mode, x, baseline, method, settings, seed, original_attr):
    heads = [(name, m) for name, m in model.named_modules() if isinstance(m, torch.nn.Linear) and m.out_features == 2]
    if not heads:
        return {}, {"randomization_status": "unsupported: no two-logit Linear head"}
    name, head = heads[-1]
    saved = {key: value.detach().clone() for key, value in head.state_dict().items()}
    try:
        with seeded(seed + 1, x.device), torch.no_grad():
            head.reset_parameters()
        shuffled, _, _ = explain(model, family, mode, x, baseline, method, settings, seed)
    finally:
        head.load_state_dict(saved, strict=True)
    a, b = original_attr.detach().cpu().numpy().ravel(), shuffled.cpu().numpy().ravel()
    correlation = float(np.corrcoef(a, b)[0, 1]) if np.std(a) > 0 and np.std(b) > 0 else None
    return {"randomized_head_attribution": shuffled.cpu().numpy()}, {
        "randomization_status": "completed", "randomized_layer": name,
        "randomized_head_correlation": correlation,
        "randomization_scope": "binary head only; not a cascading/all-parameter sanity test"}


def run(plan_path, models_root, images_root, output, task, *, model_filter=None,
        method_filter=None, shard_index=0, shards=1, device="cpu", resume=False):
    plan = read_plan(plan_path)
    if task not in plan["cohorts"] or shards < 1 or not 0 <= shard_index < shards:
        raise ValueError("Invalid task or shard configuration")
    cohort = plan["cohorts"][task]
    methods = plan["config"]["methods"]
    if model_filter and model_filter not in cohort["roster"]:
        raise ValueError("Model is not in this task's frozen roster")
    if method_filter and method_filter not in methods:
        raise ValueError("Method is not in the frozen method roster")
    out, root = Path(output), Path(models_root)
    settings = plan["config"]["settings"]
    replicas = int(settings.get("replicates", 1))
    dev = torch.device(device)
    if dev.type not in {"cpu", "cuda"}:
        raise ValueError("The validated execution contract currently supports cpu/cuda devices")
    if dev.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA requested but unavailable")
    os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")
    torch.use_deterministic_algorithms(True)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True
    if hasattr(torch.backends.cuda.matmul, "allow_tf32"):
        torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    freeze(out / "execution_contract.json", {"plan_id": plan["plan_id"], "source_sha256": _source_hash(),
        "versions": _versions(), "precision": "float32_no_autocast", "deterministic_algorithms": True})
    failures = 0; counts = {"complete": 0, "unsupported": 0, "resumed": 0, "failed": 0}
    selected_methods = [method_filter] if method_filter else methods
    for model_name in cohort["roster"]:
        if model_filter and model_name != model_filter:
            continue
        run_info = plan["models"][model_name]
        selected = [r for r in cohort["samples"] if int(r["id"]) % shards == shard_index]
        if not selected:
            continue
        run_dir = contained(root, run_info["relative_dir"])
        checkpoint, config_file = run_dir / "weights/best.pth", run_dir / "results/run_config.json"
        if sha256(config_file) != run_info["config_sha256"]:
            raise ValueError("Checkpoint configuration changed since plan freeze")
        actual_weights = sha256(checkpoint)
        if run_info["checkpoint_sha256"] and actual_weights != run_info["checkpoint_sha256"]:
            raise ValueError("Checkpoint bytes differ from frozen plan")
        freeze(out / "model_contracts" / f"{model_name}.json", {"plan_id": plan["plan_id"],
            "checkpoint_sha256": actual_weights, "config_sha256": sha256(config_file),
            "threshold": run_info["threshold"]})
        trained_run = run_from_checkpoint(checkpoint)
        cfg = config_from_run(trained_run)
        # Deliberately keep the known ICLR architecture builder and strict state load.
        # Its original pretrained cache requirements still apply; no topology guessing.
        model = load_model_from_run(trained_run, dev)
        encoder = EvaluationEncoder(trained_run.fourier_mode, cfg.image_size, cfg.in_channels)
        try:
            for row in selected:
                path = contained(images_root, row["img_name"])
                image_hash = sha256(path)
                with Image.open(path) as source:
                    image = source.convert("RGB")
                encoded, rgb = encoder.encode(image)
                if settings["baseline"] == "encoded_zero":
                    baseline = torch.zeros_like(encoded)
                else:
                    baseline, _ = encoder.encode(image.filter(ImageFilter.GaussianBlur(float(settings.get("blur_radius", 8)))))
                x, b = encoded[None].to(dev), baseline[None].to(dev)
                with torch.no_grad():
                    probability = float(torch.softmax(model(x), dim=1)[0, 1])
                expected = row["predictions"][model_name]
                tolerance = float(settings.get("prediction_atol", 1e-4))
                decision = int(probability >= run_info["threshold"])
                if abs(probability - expected["prob_pos"]) > tolerance or decision != expected["y_pred"]:
                    raise ValueError(f"Prediction/image mismatch for {model_name}, ID {row['id']}: "
                        f"fresh p={probability:.8f}, frozen p={expected['prob_pos']:.8f}. "
                        "Recheck manifest order, run provenance and preprocessing; do not silently reselect samples.")
                asset = out / "assets" / model_name / f"{row['id']}.npz"
                _save_npz(asset, encoded_input=encoded.numpy(), encoded_baseline=baseline.numpy(), rgb=rgb.numpy())
                asset_hash = sha256(asset)
                for method in selected_methods:
                    for rep in range(replicas):
                        directory = out / task / model_name / method
                        directory.mkdir(parents=True, exist_ok=True)
                        name = f"{row['id']}_r{rep}"
                        meta_path = directory / f"{name}.json"
                        identity = {"plan_id": plan["plan_id"], "image_sha256": image_hash,
                            "checkpoint_sha256": actual_weights, "asset_sha256": asset_hash,
                            "model": model_name, "method": method, "id": row["id"], "replicate": rep, "task": task}
                        if meta_path.exists():
                            old = json.loads(meta_path.read_text())
                            if not resume:
                                raise ValueError("Existing cell: pass --resume to verify and reuse it")
                            if old.get("identity") != identity:
                                raise ValueError("Resume identity mismatch")
                            if old["status"] == "complete":
                                for filename, checksum in old["artifacts"].items():
                                    if sha256(directory / filename) != checksum:
                                        raise ValueError("Resume artifact checksum mismatch")
                            elif old["status"] != "unsupported":
                                raise ValueError("Unexpected cell status")
                            counts["resumed"] += 1; continue
                        lock = directory / f"{name}.lock"
                        fd = os.open(lock, os.O_CREAT | os.O_EXCL | os.O_WRONLY)
                        os.close(fd)
                        started = time.monotonic()
                        try:
                            seed = cell_seed(plan, task, model_name, method, row["id"], rep)
                            attr, mask, metadata = explain(model, run_info["family"], run_info["mode"], x, b, method, settings, seed)
                            arrays = {"attribution": attr.cpu().numpy(), "feature_mask": mask.cpu().numpy()}
                            for key in list(metadata):
                                if isinstance(metadata[key], np.ndarray):
                                    arrays[key] = metadata.pop(key)
                            if settings.get("randomize_head", False):
                                extra_arrays, extra_meta = _head_randomization(model, run_info["family"], run_info["mode"],
                                    x, b, method, settings, seed, attr)
                                arrays.update(extra_arrays); metadata.update(extra_meta)
                            if settings.get("perturbation_controls", False):
                                metadata["perturbation_curves"] = perturbation_curves(model, x, b, attr, mask, seed,
                                    int(settings.get("curve_points", 11)))
                            array_path = directory / f"{name}.npz"
                            _save_npz(array_path, **arrays)
                            from .render import render_cell
                            picture = directory / f"{name}.png"
                            render_cell(encoded.numpy(), rgb.numpy(), attr.cpu().numpy(), run_info["mode"],
                                method, picture, label=f"ID {row['id']} | true={row['y_true']} | p(fake)={probability:.3f}")
                            record = {"status": "complete", "identity": identity,
                                "metadata": metadata, "img_name": row["img_name"], "y_true": row["y_true"],
                                "fresh_probability": probability, "threshold": run_info["threshold"],
                                "asset": str(asset.relative_to(out)), "device": str(dev),
                                "elapsed_seconds": time.monotonic() - started,
                                "artifacts": {array_path.name: sha256(array_path), picture.name: sha256(picture)}}
                            freeze(meta_path, record); counts["complete"] += 1
                        except UnsupportedMethod as error:
                            freeze(meta_path, {"status": "unsupported", "identity": identity, "reason": str(error)})
                            counts["unsupported"] += 1
                        except Exception as error:
                            failure = {"identity": identity, "status": "failed", "error": f"{type(error).__name__}: {error}"}
                            # Failed attempts are append-only and never masquerade as completed cells.
                            with (directory / f"{name}.failures.jsonl").open("ab") as f:
                                f.write(json.dumps(failure, sort_keys=True).encode() + b"\n")
                            counts["failed"] += 1; failures += 1
                        finally:
                            lock.unlink(missing_ok=True)
        finally:
            del model
            if dev.type == "cuda":
                torch.cuda.empty_cache()
    print(json.dumps(counts, indent=2))
    if failures:
        raise RuntimeError(f"{failures} attribution cells failed; see append-only failure records")
    return counts
