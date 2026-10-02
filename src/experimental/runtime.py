"""Objective-aware training with source-validation selection and epoch resume.

Loss callbacks return a mean and its weight count. Accumulation is weighted by
that count, including the final partial window. No target split is loaded here.
"""
from __future__ import annotations

import math
import resource
import time
from functools import wraps
from contextlib import nullcontext
from pathlib import Path

import torch
from filelock import FileLock

from src.robustness.engine import atomic_torch, rng_state, restore_rng, seed_all
from src.robustness.provenance import digest, digest_file, source_identity, write_json


def _plain(value):
    if isinstance(value, dict):
        return {str(k): _plain(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [_plain(v) for v in value]
    if isinstance(value, Path):
        return str(value)
    return value


def _state(model):
    return model.state_dict()


def _load(model, state):
    model.load_state_dict(state, strict=True)


def _locked_fit(function):
    @wraps(function)
    def wrapped(*args, **kwargs):
        root = Path(kwargs["run_dir"])
        root.mkdir(parents=True, exist_ok=True)
        with FileLock(str(root / ".running.lock"), timeout=0):
            return function(*args, **kwargs)
    return wrapped


@_locked_fit
def fit_model(model, train_loader, optimizer, *, loss_step, validate, run_dir,
              config, epochs, device="cpu", grad_accum_steps=1,
              selection_key="loss", selection_mode="min", resume=False,
              max_grad_norm=1.0, state_dict_fn=None, load_state_fn=None):
    """Fit a task adapter and restore its validation-selected best parameters.

    Callers build and seed models before this function, supply explicit optimizer
    groups, and move their batches inside loss_step. A model already dispatched
    by a quantization loader is left on its device. Checkpoint state callbacks
    let PEFT persist adapters without copying a frozen VLM base.
    """
    if epochs < 1 or grad_accum_steps < 1 or selection_mode not in {"min", "max"}:
        raise ValueError("Invalid epochs, accumulation or selection mode")
    if not len(train_loader):
        raise ValueError("Empty training loader")
    config = _plain(config)
    device = torch.device(device)
    if device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("Requested GPU is unavailable")
    cfg_hash = digest(config)
    root = Path(run_dir)
    root.mkdir(parents=True, exist_ok=True)
    if (root / "last.pt").exists() and not resume:
        raise FileExistsError("Training artifacts already exist; use explicit resume")
    if resume and not (root / "last.pt").is_file():
        raise FileNotFoundError("No completed-epoch checkpoint to resume")
    save_state = state_dict_fn or _state
    load_state = load_state_fn or _load
    if not getattr(model, "hf_device_map", None) and not getattr(model, "is_loaded_in_4bit", False):
        model.to(device)
    training = config.get("training", {})
    use_amp = device.type == "cuda" and bool(training.get("amp", True))
    dtype = torch.bfloat16 if use_amp and torch.cuda.is_bf16_supported() else torch.float16
    scaler = torch.amp.GradScaler("cuda", enabled=use_amp and dtype == torch.float16)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode=selection_mode, patience=int(training.get("scheduler_patience", 3)))
    history, first_epoch, global_step, stale = [], 0, 0, 0
    best = math.inf if selection_mode == "min" else -math.inf
    seed = int(config.get("seed", 42))
    record = {"schema": "faceforgery-experimental-v1", "config": config,
              "config_sha256": cfg_hash, "seed": seed, "software": source_identity(),
              "selection_key": selection_key, "selection_mode": selection_mode,
              "resume_semantics": "completed epochs; partial epoch restarts",
              "trainable_parameters": sum(p.numel() for p in model.parameters() if p.requires_grad),
              "device": str(device), "amp": use_amp, "amp_dtype": str(dtype) if use_amp else None}
    if resume:
        bundle = torch.load(root / "last.pt", map_location="cpu", weights_only=True)
        if bundle.get("config_sha256") != cfg_hash:
            raise ValueError("Resume config differs from frozen training identity")
        load_state(model, bundle["state_dict"])
        optimizer.load_state_dict(bundle["optimizer"])
        scheduler.load_state_dict(bundle["scheduler"])
        scaler.load_state_dict(bundle["scaler"])
        restore_rng(bundle["rng_state"])
        first_epoch = bundle["epoch"] + 1
        global_step, best, stale = bundle["global_step"], bundle["best"], bundle["stale"]
        history = bundle["history"]
    else:
        seed_all(seed)
    write_json(root / "run.json", record)
    write_json(root / "status.json", {"state": "running", "config_sha256": cfg_hash})
    started = time.perf_counter()
    if device.type == "cuda":
        torch.cuda.reset_peak_memory_stats(device)
    parameters = [p for group in optimizer.param_groups for p in group["params"] if p.requires_grad]
    if len({id(p) for p in parameters}) != len(parameters):
        raise ValueError("Optimizer parameter groups overlap")
    if {id(p) for p in parameters} != {id(p) for p in model.parameters() if p.requires_grad}:
        raise ValueError("Optimizer must cover exactly all trainable parameters")
    try:
        for epoch in range(first_epoch, epochs):
            for owner in (getattr(train_loader, "dataset", None),
                          getattr(train_loader, "sampler", None),
                          getattr(train_loader, "batch_sampler", None)):
                if hasattr(owner, "set_epoch"):
                    owner.set_epoch(epoch)
            generator = getattr(train_loader, "generator", None)
            if generator is not None:
                generator.manual_seed(seed + epoch)
            model.train()
            optimizer.zero_grad(set_to_none=True)
            total_loss = total_weight = window_weight = 0.0
            components = {}
            for index, batch in enumerate(train_loader):
                context = torch.amp.autocast("cuda", dtype=dtype) if use_amp else nullcontext()
                with context:
                    loss, values, weight = loss_step(model, batch, {"epoch": epoch, "global_step": global_step})
                weight = float(weight)
                if loss.ndim or not torch.isfinite(loss) or not math.isfinite(weight) or weight <= 0:
                    raise ValueError("Loss must be a finite scalar with a positive weight count")
                scaler.scale(loss * weight).backward()
                total_loss += float(loss.detach()) * weight
                total_weight += weight
                window_weight += weight
                for key, value in values.items():
                    scalar = float(value.detach()) if torch.is_tensor(value) else float(value)
                    if not math.isfinite(scalar):
                        raise ValueError(f"Nonfinite loss component: {key}")
                    components[key] = components.get(key, 0.0) + scalar * weight
                if (index + 1) % grad_accum_steps == 0 or index + 1 == len(train_loader):
                    scaler.unscale_(optimizer)
                    for p in parameters:
                        if p.grad is not None:
                            p.grad.div_(window_weight)
                            if not torch.isfinite(p.grad).all():
                                raise ValueError("Nonfinite gradient; no silent skipped update")
                    if max_grad_norm is not None:
                        torch.nn.utils.clip_grad_norm_(parameters, max_grad_norm, error_if_nonfinite=True)
                    scaler.step(optimizer)
                    scaler.update()
                    optimizer.zero_grad(set_to_none=True)
                    window_weight = 0.0
                    global_step += 1
            model.eval()
            with torch.no_grad():
                metrics = _plain(validate(model))
            score = float(metrics[selection_key])
            if not math.isfinite(score):
                raise ValueError("Selection metric is undefined or nonfinite")
            scheduler.step(score)
            row = {"epoch": epoch, "global_step": global_step,
                   "train_loss": total_loss / total_weight,
                   "components": {k: v / total_weight for k, v in components.items()},
                   "validation": metrics}
            history.append(row)
            improved = score < best if selection_mode == "min" else score > best
            best, stale = (score, 0) if improved else (best, stale + 1)
            bundle = {"schema": record["schema"], "config_sha256": cfg_hash,
                      "state_dict": save_state(model), "optimizer": optimizer.state_dict(),
                      "scheduler": scheduler.state_dict(), "scaler": scaler.state_dict(),
                      "rng_state": rng_state(), "epoch": epoch, "global_step": global_step,
                      "best": best, "stale": stale, "history": history}
            if improved:
                atomic_torch(root / "best.pt", bundle)
            atomic_torch(root / "last.pt", bundle)
            write_json(root / "history.json", history)
            if stale >= int(training.get("early_stop_patience", epochs + 1)):
                break
        best_bundle = torch.load(root / "best.pt", map_location="cpu", weights_only=True)
        load_state(model, best_bundle["state_dict"])
        telemetry = {"runtime_seconds": time.perf_counter() - started,
                     "peak_vram_bytes": torch.cuda.max_memory_allocated(device) if device.type == "cuda" else 0,
                     "peak_rss_kib": resource.getrusage(resource.RUSAGE_SELF).ru_maxrss,
                     "trainable_parameters": record["trainable_parameters"],
                     "epochs_completed": len(history), "global_step": global_step,
                     "best_epoch": best_bundle["epoch"], "best_metric": best}
        write_json(root / "telemetry.json", telemetry)
        write_json(root / "status.json", {"state": "trained", "config_sha256": cfg_hash,
                   "artifacts": {name: digest_file(root / name) for name in
                                 ["run.json", "best.pt", "last.pt", "history.json", "telemetry.json"]}})
        return {"run_dir": str(root), "history": history, **telemetry}
    except BaseException as exc:
        write_json(root / "status.json", {"state": "failed", "error": str(exc), "config_sha256": cfg_hash})
        raise
