"""Explicit, single-device pilot training and checkpoint-safe epoch resume.

No test or external target split is loaded by training. A completed research run
requires artifacts and hashes, not just a best.pth left by an interrupted run.
"""

from __future__ import annotations
import json
import os
import random
import time
from pathlib import Path
import numpy as np
import torch
from torch.utils.data import DataLoader
from .experiments import plan
from .imaging import CanonicalDataset
from .inference import predict, calibrate
from .artifacts import save_predictions
from .manifests import load_manifest
from .models import (
    build_model,
    state_digest,
    supervised_loss,
    consistency_loss,
    distillation_loss,
)
from .provenance import SCHEMA, digest_file, write_csv, write_json
from .statistics import summary


def seed_all(seed):
    random.seed(seed)
    np.random.seed(seed % 2**32)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True


def atomic_torch(path, value):
    path = Path(path)
    temporary = path.with_name("." + path.name + ".partial")
    try:
        torch.save(value, temporary)
        os.replace(temporary, path)
    finally:
        temporary.unlink(missing_ok=True)


def rng_state():
    state = np.random.get_state()
    result = {
        "python": random.getstate(),
        "numpy_name": state[0],
        "numpy_keys": torch.tensor(state[1].astype(np.int64)),
        "numpy_pos": state[2],
        "numpy_gauss": state[3],
        "numpy_cached": state[4],
        "torch": torch.get_rng_state(),
    }
    if torch.cuda.is_available():
        result["cuda"] = torch.cuda.get_rng_state_all()
    return result


def restore_rng(state):
    random.setstate(state["python"])
    np.random.set_state(
        (
            state["numpy_name"],
            state["numpy_keys"].numpy().astype(np.uint32),
            state["numpy_pos"],
            state["numpy_gauss"],
            state["numpy_cached"],
        )
    )
    torch.set_rng_state(state["torch"])
    if "cuda" in state and torch.cuda.is_available():
        torch.cuda.set_rng_state_all(state["cuda"])


def verify_complete(run_dir):
    root = Path(run_dir)
    status = json.loads((root / "status.json").read_text())
    if status.get("state") != "complete":
        raise ValueError("Run has not completed")
    for name, checksum in status.get("artifacts", {}).items():
        if Path(name).name != name or digest_file(root / name) != checksum:
            raise ValueError("Completed run artifact is missing or modified")
    if not {
        "best.pt",
        "calibration.json",
        "validation_predictions.csv",
        "run.json",
    } <= set(status.get("artifacts", {})):
        raise ValueError("Incomplete completion certificate")
    return status


def load_pilot(run_dir, device="cpu", *, require_complete=True):
    root = Path(run_dir)
    if require_complete:
        verify_complete(root)
    record = json.loads((root / "run.json").read_text())
    bundle = torch.load(root / "best.pt", map_location="cpu", weights_only=True)
    if bundle.get("schema") != SCHEMA or bundle.get("run_id") != record["run_id"]:
        raise ValueError("Checkpoint/run identity mismatch")
    model = build_model(record["config"]["model"], initialize_pretrained=False)
    model.load_state_dict(bundle["state_dict"], strict=True)
    return model.to(device).eval(), record


def train(config: dict, *, device="cpu", resume=False):
    frozen = plan(config)
    root = Path(frozen["run_dir"])
    t = config["training"]
    device = torch.device(device)
    if device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA requested but unavailable")
    if root.exists() and not resume:
        raise FileExistsError(
            "Run exists. Choose --resume or a new frozen experiment identity."
        )
    if resume and not (root / "last.pt").is_file():
        raise FileNotFoundError("No resumable epoch checkpoint")
    if (root / "status.json").exists():
        old = json.loads((root / "status.json").read_text())
        if old.get("state") == "complete":
            verify_complete(root)
            return old
    root.mkdir(parents=True, exist_ok=True)
    lock = root / ".running.lock"
    descriptor = os.open(lock, os.O_CREAT | os.O_EXCL | os.O_WRONLY)
    os.write(descriptor, str(os.getpid()).encode())
    os.close(descriptor)
    started = time.perf_counter()
    try:
        write_json(
            root / "status.json", {"state": "running", "run_id": frozen["run_id"]}
        )
        seed_all(t["seed"])
        model = build_model(config["model"], initialize_pretrained=not resume).to(
            device
        )
        optimizer = torch.optim.AdamW(
            model.parameters(), lr=t["lr"], weight_decay=t["weight_decay"]
        )
        record = {
            **frozen,
            "initial_state_sha256": state_digest(model),
            "initial_backbone_sha256": state_digest(model.backbone)
            if model.backbone is not None
            else None,
            "parameter_count": sum(p.numel() for p in model.parameters()),
            "device": str(device),
            "cuda_device": torch.cuda.get_device_name(device)
            if device.type == "cuda"
            else None,
        }
        teacher = None
        if t["teacher_run"]:
            teacher, teacher_record = load_pilot(t["teacher_run"], device)
            if (
                teacher_record["train_manifest_sha256"]
                != frozen["train_manifest_sha256"]
                or teacher_record["val_manifest_sha256"]
                != frozen["val_manifest_sha256"]
            ):
                raise ValueError(
                    "Distillation requires the same certified source train/validation populations"
                )
            if teacher_record["config"]["training"]["image_size"] != t["image_size"]:
                raise ValueError(
                    "Teacher/student input sizes differ; define an explicit adaptation experiment"
                )
            teacher.requires_grad_(False)
            record["teacher_checkpoint_sha256"] = digest_file(
                Path(t["teacher_run"]) / "best.pt"
            )
        first, best, bad, history = 0, -1.0, 0, []
        if resume:
            previous = json.loads((root / "run.json").read_text())
            last = torch.load(root / "last.pt", map_location="cpu", weights_only=True)
            if (
                previous["run_id"] != frozen["run_id"]
                or last["run_id"] != frozen["run_id"]
            ):
                raise ValueError(
                    "Resume configuration, data, environment or source code changed"
                )
            if previous.get("teacher_checkpoint_sha256") != record.get(
                "teacher_checkpoint_sha256"
            ):
                raise ValueError("Teacher changed since training began")
            record = previous
            model.load_state_dict(last["state_dict"], strict=True)
            optimizer.load_state_dict(last["optimizer"])
            for state in optimizer.state.values():
                for k, v in state.items():
                    if isinstance(v, torch.Tensor):
                        state[k] = v.to(device)
            first, best, bad, history = (
                last["epoch"] + 1,
                last["best_auc"],
                last["bad_epochs"],
                last["history"],
            )
            restore_rng(last["rng"])
        else:
            write_json(root / "run.json", record)
        train_frame, _ = load_manifest(config["data"]["train_manifest"])
        val_frame, val_certificate = load_manifest(config["data"]["val_manifest"])
        ds = CanonicalDataset(
            train_frame,
            config["data"]["train_root"],
            t["image_size"],
            training=True,
            recipe=t["recipe"],
            seed=t["seed"],
        )
        for epoch in range(first, t["epochs"]):
            if bad >= t["patience"]:
                break
            ds.epoch = epoch
            generator = torch.Generator().manual_seed(t["seed"] + epoch)
            loader = DataLoader(
                ds,
                batch_size=t["batch_size"],
                shuffle=True,
                generator=generator,
                num_workers=t["workers"],
                persistent_workers=False,
            )
            model.train()
            running = 0.0
            examples = 0
            for batch in loader:
                x, y = batch["image"].to(device), batch["label"].to(device)
                optimizer.zero_grad(set_to_none=True)
                out = model.forward_details(x)
                loss = supervised_loss(out, y, t["aux_weight"])
                if t["paired_training"]:
                    clean = model.forward_details(batch["clean"].to(device))
                    loss = 0.5 * (
                        loss + supervised_loss(clean, y, t["aux_weight"])
                    ) + t["consistency_weight"] * consistency_loss(
                        out["logits"], clean["logits"]
                    )
                if teacher is not None:
                    with torch.no_grad():
                        teacher_logits = teacher(x)
                    loss = loss + t["distillation_weight"] * distillation_loss(
                        out["logits"], teacher_logits, t["temperature"]
                    )
                if not torch.isfinite(loss):
                    raise FloatingPointError("Nonfinite training loss")
                loss.backward()
                torch.nn.utils.clip_grad_norm_(
                    model.parameters(), 1.0, error_if_nonfinite=True
                )
                optimizer.step()
                running += float(loss.detach()) * len(y)
                examples += len(y)
            validation = predict(
                model,
                val_frame,
                config["data"]["val_root"],
                image_size=t["image_size"],
                device=device,
                batch_size=t["batch_size"],
                workers=t["workers"],
                hash_images=False,
            )
            val_metrics = summary(validation.label, validation.p_fake)
            score = val_metrics["auc"]
            row = {
                "epoch": epoch,
                "training_loss": running / examples,
                "val_auc": score,
                "n_train": examples,
            }
            history.append(row)
            if score > best:
                best, bad = score, 0
                atomic_torch(
                    root / "best.pt",
                    {
                        "schema": SCHEMA,
                        "run_id": frozen["run_id"],
                        "state_dict": model.state_dict(),
                        "epoch": epoch,
                        "val_auc": score,
                    },
                )
            else:
                bad += 1
            atomic_torch(
                root / "last.pt",
                {
                    "schema": SCHEMA,
                    "run_id": frozen["run_id"],
                    "state_dict": model.state_dict(),
                    "optimizer": optimizer.state_dict(),
                    "epoch": epoch,
                    "best_auc": best,
                    "bad_epochs": bad,
                    "history": history,
                    "rng": rng_state(),
                },
            )
            write_json(root / "history.json", history)
            print(json.dumps(row), flush=True)
        best_model, _ = load_pilot(root, device, require_complete=False)
        validation = predict(
            best_model,
            val_frame,
            config["data"]["val_root"],
            image_size=t["image_size"],
            device=device,
            batch_size=t["batch_size"],
            workers=t["workers"],
            hash_images=True,
        )
        save_predictions(
            root / "validation_predictions.csv",
            validation,
            manifest_record=val_certificate,
            model_sha256=digest_file(root / "best.pt"),
        )
        # An interrupted finalization can be repeated only with an identical certificate.
        cal_path = root / "calibration.json"
        if cal_path.exists():
            saved = json.loads(cal_path.read_text())
            if (
                saved.get("model_sha256") != digest_file(root / "best.pt")
                or saved.get("source_manifest_sha256")
                != val_certificate["manifest_sha256"]
            ):
                raise ValueError("Stale calibration during resume")
        else:
            calibrate(
                validation,
                manifest_record=val_certificate,
                output=cal_path,
                model_sha256=digest_file(root / "best.pt"),
            )
        files = [
            "best.pt",
            "last.pt",
            "calibration.json",
            "validation_predictions.csv",
            "validation_predictions.csv.json",
            "run.json",
            "history.json",
        ]
        status = {
            "state": "complete",
            "run_id": frozen["run_id"],
            "epochs_completed": len(history),
            "best_val_auc": best,
            "this_invocation_seconds": time.perf_counter() - started,
            "artifacts": {name: digest_file(root / name) for name in files},
            "evidence": "source training and validation only; external evaluation is separate",
        }
        write_json(root / "status.json", status)
        return status
    except Exception as error:
        write_json(
            root / "status.json",
            {
                "state": "failed",
                "run_id": frozen["run_id"],
                "error": f"{type(error).__name__}: {error}",
            },
        )
        raise
    finally:
        lock.unlink(missing_ok=True)
