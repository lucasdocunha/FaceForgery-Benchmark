"""Primary pilot XAI on a frozen shared cohort; no generated empirical examples.

End-to-end input maps are in RGB pixel coordinates, including derivatives through
the pilot's FFT branch. They are NOT native-frequency localizations. The retained
legacy XAI package separately supports native RGB/pure-frequency comparisons.
"""

from __future__ import annotations
import argparse
import io
import json
import os
from importlib.metadata import version
from pathlib import Path
import numpy as np
import pandas as pd
import torch
from torch import nn
from PIL import Image, ImageFilter
from .artifacts import load_predictions
from .engine import load_pilot, verify_complete
from .imaging import CanonicalDataset, to_tensor
from .manifests import load_manifest, binary
from .provenance import (
    SCHEMA,
    atomic_bytes,
    digest,
    digest_file,
    source_identity,
    write_csv,
    write_json,
)


class Margin(nn.Module):
    def __init__(self, model):
        super().__init__()
        self.model = model

    def forward(self, x):
        z = self.model(x)
        return z[:, 1] - z[:, 0]


def sample_cohort(predictions, calibration, output, seed=42):
    cal = json.loads(Path(calibration).read_text())
    frame, record = load_predictions(predictions, calibration=cal)
    if record["split"] not in {"test", "test_d"}:
        raise ValueError(
            "Task-1 diagnostic sampling requires a declared test/test_d population"
        )
    y = frame.label.to_numpy()
    pred = (frame.p_fake.to_numpy() >= cal["frame_threshold"]).astype(int)
    frame["reference_stratum"] = np.where(
        y == 1, np.where(pred == 1, "TP", "FN"), np.where(pred == 0, "TN", "FP")
    )
    parts = []
    for name in ["TP", "FN", "TN", "FP"]:
        candidates = frame.loc[frame.reference_stratum.eq(name)].sort_values(
            "sample_id"
        )
        if len(candidates) < 16:
            raise ValueError(f"Fewer than 16 examples in reference stratum {name}")
        rng = np.random.default_rng(int(digest([int(seed), name])[:16], 16))
        parts.append(candidates.iloc[rng.choice(len(candidates), 16, replace=False)])
    cohort = pd.concat(parts, ignore_index=True)[
        ["sample_id", "group_id", "label", "reference_stratum"]
    ]
    output = Path(output)
    if output.exists() or Path(str(output) + ".json").exists():
        raise FileExistsError("Use a new cohort path")
    write_csv(output, cohort)
    result = {
        "schema": SCHEMA,
        "kind": "reference-stratified-64",
        "seed": int(seed),
        "rows": 64,
        "dataset": record["dataset"],
        "reference_predictions_sha256": record["predictions_sha256"],
        "reference_calibration_sha256": digest_file(calibration),
        "cohort_sha256": digest_file(output),
        "interpretation": "16 TP/FN/TN/FP only for the fixed reference; diagnostic sample, not population prevalence",
    }
    write_json(str(output) + ".json", result)
    return result


def load_cohort(path):
    certificate = json.loads(Path(str(path) + ".json").read_text())
    if certificate.get("schema") != SCHEMA or certificate.get(
        "cohort_sha256"
    ) != digest_file(path):
        raise ValueError("Uncertified or modified cohort")
    frame = pd.read_csv(path, dtype={"sample_id": str, "group_id": str})
    if (
        not {"sample_id", "group_id", "label"} <= set(frame)
        or len(frame) != certificate["rows"]
        or frame.sample_id.duplicated().any()
    ):
        raise ValueError("Invalid cohort identities")
    frame["label"] = binary(frame.label)
    if certificate.get("kind") == "reference-stratified-64":
        if len(frame) != 64 or frame.reference_stratum.value_counts().to_dict() != {
            "TP": 16,
            "FN": 16,
            "TN": 16,
            "FP": 16,
        }:
            raise ValueError("Reference cohort is not the required 16x4 design")
    return frame, certificate


def attribute(model, x, method, *, baseline, steps=64, layer=None):
    from captum.attr import IntegratedGradients, Occlusion, LayerGradCam

    if not 2 <= steps <= 1024:
        raise ValueError("Integration steps must be in [2,1024]")
    model.eval()
    target = Margin(model)
    x = x.detach().requires_grad_(True)
    meta = {
        "method": method,
        "target": "fake_minus_real_logit_margin",
        "coordinate_system": "RGB input pixels",
        "captum_version": version("captum"),
    }
    if method == "integrated_gradients":
        values, residual = IntegratedGradients(target).attribute(
            x,
            baselines=baseline,
            n_steps=steps,
            internal_batch_size=8,
            return_convergence_delta=True,
        )
        meta.update(
            integration_steps=steps, completeness_residual=float(residual.item())
        )
    elif method == "occlusion":
        h, w = x.shape[-2:]
        window = (3, min(16, h), min(16, w))
        stride = (3, min(8, h), min(8, w))
        values = Occlusion(target).attribute(
            x,
            baselines=baseline,
            sliding_window_shapes=window,
            strides=stride,
            perturbations_per_eval=8,
        )
        meta.update(
            window=list(window),
            stride=list(stride),
            interpretation="baseline-replacement sensitivity, not causal validation",
        )
    elif method == "gradcam":
        if not layer:
            raise ValueError("Grad-CAM requires an explicit spatial CNN layer path")
        values = LayerGradCam(target, model.get_submodule(layer)).attribute(
            x, relu_attributions=True
        )
        if values.ndim != 4:
            raise ValueError(
                "Selected layer is not a spatial CNN map; no token/attention substitution"
            )
        meta.update(
            layer=layer,
            native_shape=list(values.shape),
            interpretation="rectified class-specific layer localization",
        )
        values = torch.nn.functional.interpolate(
            values, size=x.shape[-2:], mode="bilinear", align_corners=False
        )
    else:
        raise ValueError("Unsupported primary explanation method")
    if not torch.isfinite(values).all():
        raise FloatingPointError("Nonfinite attribution")
    meta["all_zero"] = bool((values == 0).all())
    return values.detach(), meta


def overlay_bytes(raw, attribution):
    image = raw.detach().cpu().numpy().transpose(1, 2, 0)
    values = attribution.detach().cpu().numpy().sum(0)
    scale = float(np.abs(values).max())
    normalized = values / max(scale, 1e-12)
    tint = np.zeros_like(image)
    tint[..., 0] = np.maximum(normalized, 0)
    tint[..., 2] = np.maximum(-normalized, 0)
    alpha = np.abs(normalized)[..., None] * 0.55
    display = np.clip(image * (1 - alpha) + tint * alpha, 0, 1)
    stream = io.BytesIO()
    Image.fromarray((display * 255).astype(np.uint8)).save(stream, format="PNG")
    return stream.getvalue()


def run(args):
    root = Path(args.run)
    verify_complete(root)
    run_record = json.loads((root / "run.json").read_text())
    calibration = json.loads((root / "calibration.json").read_text())
    cohort, cc = load_cohort(args.cohort)
    frame, mc = load_manifest(args.manifest)
    expected, pc = load_predictions(args.predictions, calibration=calibration)
    if (
        pc["model_sha256"] != digest_file(root / "best.pt")
        or pc["manifest_sha256"] != mc["manifest_sha256"]
    ):
        raise ValueError(
            "Pilot predictions do not belong to this checkpoint and manifest"
        )
    if cc["dataset"] != mc["dataset"] or not set(cohort.sample_id) <= set(
        frame.sample_id
    ):
        raise ValueError("Cohort does not belong to the target population")
    aligned = frame.set_index("sample_id").loc[cohort.sample_id]
    if not np.array_equal(
        aligned.label.to_numpy(), cohort.label.to_numpy()
    ) or not np.array_equal(aligned.group_id.to_numpy(), cohort.group_id.to_numpy()):
        raise ValueError("Cohort labels/groups differ from target manifest")
    if args.shard_count < 1 or not 0 <= args.shard_index < args.shard_count:
        raise ValueError("Invalid shard")
    positions = {ident: i for i, ident in enumerate(frame.sample_id)}
    indices = [
        positions[ident]
        for ident in cohort.sample_id
        if int(digest(ident)[:16], 16) % args.shard_count == args.shard_index
    ]
    contract = {
        "model_sha256": pc["model_sha256"],
        "manifest_sha256": mc["manifest_sha256"],
        "predictions_sha256": pc["predictions_sha256"],
        "cohort_sha256": cc["cohort_sha256"],
        "calibration_sha256": digest_file(root / "calibration.json"),
        "method": args.method,
        "baseline": args.baseline,
        "steps": args.steps,
        "layer": args.layer,
        "software": source_identity(),
        "probability_tolerance": 1e-5,
        "coordinate_system": "RGB input pixels, including the differentiable FFT path; NOT a native frequency map",
    }
    contract_id = digest(contract)
    out = Path(args.output) / contract_id[:16]
    result = {
        "contract_id": contract_id,
        "total_cohort": len(cohort),
        "shard_images": len(indices),
        "shard_index": args.shard_index,
        "shard_count": args.shard_count,
        "output": str(out),
        "state": "planned",
    }
    if not args.execute:
        return result
    out.mkdir(parents=True, exist_ok=True)
    plan_path = out / "plan.json"
    if plan_path.exists() and json.loads(plan_path.read_text()) != contract:
        raise ValueError("Output contract changed")
    write_json(plan_path, contract)
    model, _ = load_pilot(root, args.device)
    model.requires_grad_(False)
    ds = CanonicalDataset(
        frame,
        args.root,
        run_record["config"]["training"]["image_size"],
        hash_images=True,
    )
    by_id = expected.set_index("sample_id")
    completed = 0
    for index in indices:
        item = ds[index]
        ident = item["sample_id"]
        stem = digest(ident)
        npz_path = out / (stem + ".npz")
        image_path = out / (stem + ".png")
        meta_path = out / (stem + ".json")
        if meta_path.exists():
            saved = json.loads(meta_path.read_text())
            if not args.resume:
                raise FileExistsError("Explanation exists; use verified --resume")
            if (
                saved.get("contract_id") != contract_id
                or saved.get("image_sha256") != item["image_sha256"]
                or saved.get("npz_sha256") != digest_file(npz_path)
                or saved.get("overlay_sha256") != digest_file(image_path)
            ):
                raise ValueError("Existing attribution artifact changed")
            completed += 1
            continue
        lock = out / (stem + ".lock")
        fd = os.open(lock, os.O_CREAT | os.O_EXCL | os.O_WRONLY)
        os.close(fd)
        try:
            x = item["image"].unsqueeze(0).to(args.device)
            with torch.no_grad():
                actual = float(model(x).softmax(-1)[0, 1])
            if (
                not np.isfinite(actual)
                or abs(actual - float(by_id.loc[ident, "p_fake"])) > 1e-5
            ):
                raise ValueError(
                    f"Fresh prediction differs from frozen export for {ident}"
                )
            recorded_image_hash = str(by_id.loc[ident].get("image_sha256", ""))
            if recorded_image_hash and recorded_image_hash != item["image_sha256"]:
                raise ValueError("Source image changed since inference")
            if args.baseline == "zero":
                baseline = torch.zeros_like(x)
            else:
                pil = Image.fromarray(
                    (item["image"].numpy().transpose(1, 2, 0) * 255).astype(np.uint8)
                )
                baseline = (
                    to_tensor(pil.filter(ImageFilter.GaussianBlur(2.0)))
                    .unsqueeze(0)
                    .to(args.device)
                )
            attr, meta = attribute(
                model,
                x,
                args.method,
                baseline=baseline,
                steps=args.steps,
                layer=args.layer,
            )
            stream = io.BytesIO()
            np.savez_compressed(
                stream,
                input=item["image"].numpy(),
                baseline=baseline[0].cpu().numpy(),
                attribution=attr[0].cpu().numpy(),
            )
            atomic_bytes(npz_path, stream.getvalue())
            atomic_bytes(image_path, overlay_bytes(x[0], attr[0]))
            reference_row = cohort.set_index("sample_id").loc[ident]
            meta.update(
                contract_id=contract_id,
                sample_id=ident,
                label=item["label"],
                p_fake=actual,
                model_prediction=int(actual >= calibration["frame_threshold"]),
                model_threshold=calibration["frame_threshold"],
                reference_stratum=str(
                    reference_row.get("reference_stratum", "not-stratified")
                ),
                image_sha256=item["image_sha256"],
                npz_sha256=digest_file(npz_path),
                overlay_sha256=digest_file(image_path),
                display="per-image max-absolute normalization; red positive / blue negative; not shared magnitude scaling",
                baseline_semantics="black raw RGB"
                if args.baseline == "zero"
                else "raw RGB Gaussian blur radius 2 pixels",
            )
            write_json(meta_path, meta)
            completed += 1
        finally:
            lock.unlink(missing_ok=True)
    result.update(state="complete", completed=completed)
    write_json(
        out / f"shard-{args.shard_index:04d}-of-{args.shard_count:04d}.json", result
    )
    return result


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="command", required=True)
    sample = sub.add_parser("sample")
    for key in ["predictions", "calibration", "output"]:
        sample.add_argument("--" + key, required=True)
    sample.add_argument("--seed", type=int, default=42)
    execute = sub.add_parser("run")
    for key in ["run", "manifest", "root", "predictions", "cohort", "output"]:
        execute.add_argument("--" + key, required=True)
    execute.add_argument(
        "--method",
        choices=["integrated_gradients", "occlusion", "gradcam"],
        default="integrated_gradients",
    )
    execute.add_argument("--baseline", choices=["zero", "blur"], default="blur")
    execute.add_argument("--steps", type=int, default=64)
    execute.add_argument("--layer")
    execute.add_argument("--device", default="cpu")
    execute.add_argument("--shard-index", type=int, default=0)
    execute.add_argument("--shard-count", type=int, default=1)
    execute.add_argument("--execute", action="store_true")
    execute.add_argument("--resume", action="store_true")
    args = parser.parse_args(argv)
    result = (
        sample_cohort(args.predictions, args.calibration, args.output, args.seed)
        if args.command == "sample"
        else run(args)
    )
    print(json.dumps(result, indent=2, ensure_ascii=False, allow_nan=False))


if __name__ == "__main__":
    main()
