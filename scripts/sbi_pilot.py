"""Source-only equal-budget SBI comparison, with optional HF adaptation smoke."""
from __future__ import annotations

import argparse
import gc
import json
from pathlib import Path
import time

import numpy as np
import pandas as pd
import torch

from src.experimental.sbi import fit, load_model
from src.experimental.sbi.training import SBIClassifier
from src.robustness.inference import predict
from src.robustness.manifests import load_manifest
from src.robustness.provenance import digest_file, source_identity, write_json
from src.robustness.statistics import grouped_auc_interval


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    for key in ("train-manifest", "val-manifest", "train-root", "val-root", "landmarks", "generic-weights", "output"):
        parser.add_argument("--"+key, required=True)
    parser.add_argument("--hf-checkpoint")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--epochs", type=int, default=5)
    parser.add_argument("--image-size", type=int, default=224)
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--resume", action="store_true")
    args = parser.parse_args(argv)
    output = Path(args.output)
    if not 1 <= args.epochs <= 20 or args.batch_size < 2:
        parser.error("Use a bounded 1-20 epoch pilot and batch size >=2")
    if output.exists() and not args.resume:
        parser.error("Use a fresh output path or explicit --resume")
    train, _ = load_manifest(args.train_manifest)
    val, _ = load_manifest(args.val_manifest)
    if max(len(train),len(val)) > 10000:
        parser.error("This min-dataset pilot is capped at10000rows per split")
    output.mkdir(parents=True, exist_ok=True)
    record = {"purpose": "pilot/min-dataset/source-validation development", "software":source_identity(),
              "seed": args.seed, "train_n":len(train), "val_n":len(val), "epochs":args.epochs,
              "selection": "MFFI validation fakes are used for checkpoint and threshold selection",
              "test_accesses":0, "runs":[], "limitations":["202 independent source real faces in local min-train",
                  "MediaPipe hull differs from original SBI dlib81", "No local DF40 or Celeb-DF data"]}
    # Prove the faster device-batched encoder retains the verified legacy input.
    from src.robustness.legacy_encoding import encode_legacy_tensor
    raw = torch.rand(4,3,32,32, generator=torch.Generator().manual_seed(args.seed))
    expected = encode_legacy_tensor(raw,"srm",6)
    encoder = SBIClassifier(torch.nn.Identity(),"srm").to(args.device)
    encoded = encoder(raw.to(args.device)).cpu()
    torch.testing.assert_close(encoded,expected,atol=2e-6,rtol=2e-6)
    record["device_srm_max_abs_difference_from_legacy_cpu"] = float((encoded-expected).abs().max())
    del encoder
    schedule = [(arm,"imagenet") for arm in ("sbi","mffi","mixed")]
    if args.hf_checkpoint:
        schedule.append(("sbi","hf_mffi"))
    for arm, init in schedule:
        run = output / (arm if init == "imagenet" else "hf-sbi-smoke")
        cfg = {"name":"sbi-generic-"+arm if init == "imagenet" else "sbi-hf-smoke", "seed":args.seed,
               "output_dir":str(run), "data":{key:getattr(args,key) for key in
                ("train_manifest","val_manifest","train_root","val_root","landmarks")},
               "model":{"architecture":"mobilenet_v3_large", "initialization":init,
                        "weights":args.generic_weights if init == "imagenet" else args.hf_checkpoint,
                        "image_size":args.image_size, "mode":"srm", "train_backbone":True},
               "training":{"arm":arm,"epochs":args.epochs if init == "imagenet" else 1,
                  "batch_size":args.batch_size,"workers":0,"grad_accum_steps":4,
                  "lr_backbone":1e-4 if init == "imagenet" else 1e-5,
                  "lr_head":1e-3 if init == "imagenet" else 1e-4,
                  "early_stop_patience":args.epochs+1,"cache_images":True,"bootstrap_draws":300}}
        started = time.perf_counter()
        row = {"arm":arm, "initialization":init, "run_dir":str(run), "config":cfg}
        try:
            fit(cfg, device=args.device, resume=args.resume and (run/"last.pt").exists())
            raw = json.loads((run/"run.json").read_text())
            row.update(status="complete", metrics=json.loads((run/"validation_metrics.json").read_text()),
                       telemetry=json.loads((run/"telemetry.json").read_text()),
                       initial_model_state_sha256=raw["config"]["provenance"]["initial_model_state_sha256"],
                       best_checkpoint_sha256=digest_file(run/"best.pt"))
            model = load_model(run, args.device)
            restored = predict(model,val,args.val_root,image_size=args.image_size,device=args.device,batch_size=args.batch_size)
            saved = pd.read_csv(run/"validation_predictions.csv").sort_values("sample_id")
            error = float(np.max(np.abs(saved.p_fake.to_numpy()-restored.p_fake.to_numpy())))
            if error > 1e-6:
                raise ValueError(f"Reload prediction mismatch: {error}")
            row["reload_max_abs_error"] = error
            del model
        except Exception as exc:
            row.update(status="failed", error=f"{type(exc).__name__}: {exc}")
        row["wall_seconds_including_reload"] = time.perf_counter()-started
        record["runs"].append(row)
        write_json(output/"pilot.json",record)
        print(json.dumps({k:row[k] for k in ("arm","initialization","status","wall_seconds_including_reload")}),flush=True)
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
    clean = [r for r in record["runs"] if r["initialization"] == "imagenet" and r["status"] == "complete"]
    if len(clean) == 3:
        if len({r["initial_model_state_sha256"] for r in clean}) != 1 or len({r["telemetry"]["global_step"] for r in clean}) != 1:
            raise ValueError("Three-arm comparison did not preserve exact initial state and update budget")
        record["matched_initial_state_and_updates"] = True
        baseline = pd.read_csv(output/"mffi/validation_predictions.csv")
        record["paired_auc_delta_vs_mffi"] = {arm:grouped_auc_interval(baseline,pd.read_csv(output/arm/"validation_predictions.csv"),draws=300,seed=args.seed) for arm in ("sbi","mixed")}
        write_json(output/"pilot.json",record)
    return int(any(r["status"] != "complete" for r in record["runs"]))


if __name__ == "__main__":
    raise SystemExit(main())
