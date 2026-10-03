# SBI training and controlled comparisons

This is an independently implemented SBI-style training recipe. It uses one source real face, a perturbed view of that face, a landmark hull and a soft blending mask. It is not an exact reproduction of the original dlib 81 implementation. Scientific motivation and primary references are in [the planning review](planning/reconstruction_sbi.md).

The local clean proxy compares generic ImageNet initialization across `sbi`, `mffi` and `mixed`. The source real cohort, initial model state, seed, optimizer and number of updates are identical. Every epoch contains two slots per accepted real face: one real and one fake. SBI uses self-blends; MFFI rotates through genuine training fakes; mixed uses both. The pure SBI arm never reads training fake image pixels. Source validation includes MFFI fakes for checkpoint selection and frozen Youden calibration. Consequently this is unseen-forgery *training*, not a completely fake-free model-selection procedure.

HF initialization is a separate adaptation experiment. Those models already saw MFFI training fakes and were selected on full MFFI val, which contains min-val. Never describe this arm as a clean unseen-forgery experiment. Original HF preprocessing is preserved: ImageNet normalization followed by three fixed SRM residual filters, six channels total. Batched GPU preprocessing stays FP32 even inside AMP and is checked against the original implementation.

## Prepare landmarks once

Stage the MediaPipe Face Landmarker task asset explicitly under `TCC_PRETRAINED_ROOT`. Keep the detection environment separate from the CUDA training environment to avoid overlapping OpenCV distributions:

```bash
python3.12 -m venv .venv-landmarks
.venv-landmarks/bin/python -m pip install -r requirements-experimental-landmarks.txt
.venv-landmarks/bin/python -m src.experimental.sbi.landmarks \
  --manifest "$TCC_TRAIN_MANIFEST" --root "$TCC_TRAIN_ROOT" \
  --asset "$TCC_PRETRAINED_ROOT/face_landmarker.task" \
  --output "$TCC_DATA_ROOT/landmarks/train-real.sqlite"
```

The certified source manifest is required. Only real source images are detected. The SQLite file binds each record to image bytes, manifest and landmark asset hashes; it stores explicit failures for no face, multiple faces, invalid geometry and decoding errors. The default training policy fails on a missing/failed landmark. An explicit exclusion policy records the removed IDs and uses the same accepted cohort for every arm. SQLite geometry and masks are loaded on demand, with process-local connections. `cache_images: false` in server configs prevents full decoded-image RAM caches.

Inspect masks before full training. `preflight/sbi-mask-qa.png` shows four seed-selected local examples; all 202 local train reals were detected. MediaPipe landmarks cover less forehead than dlib 81. Hull indices and `forehead_extension` are configurable, but any change is a new recorded training condition, not an inference-time adjustment.

## Server commands

Set certified `TCC_TRAIN_MANIFEST`, `TCC_VAL_MANIFEST`, image roots `TCC_TRAIN_ROOT`, `TCC_VAL_ROOT`, plus `TCC_DATA_ROOT`, `TCC_PRETRAINED_ROOT`, `TCC_EXPERIMENT_ROOT`. The generic MobileNet weight file is `mobilenet_v3_large-5c1a4163.pth`; it is explicit and never downloaded during training. Each fresh run needs a distinct output directory.

```bash
for TCC_SEED in 42 123 2024 7 2025; do
  export TCC_SEED
  for TCC_SBI_ARM in sbi mffi mixed; do
    export TCC_SBI_ARM
    export TCC_RUN_DIR="$TCC_EXPERIMENT_ROOT/sbi-generic-$TCC_SBI_ARM/seed_$TCC_SEED"
    python research_cli.py experimental train --family sbi \
      --config configs/experimental/sbi_generic.yaml \
      --seed "$TCC_SEED" --device cuda --execute
  done
done
```

For adaptation, choose the exact family and matching seed checkpoint, together with its original `results/run_config.json` layout. DINO-SRM and CLIP-SRM are the full-server priorities; MobileNet-SRM is the inexpensive local check. No robust RGB checkpoint is assumed available.

```bash
export TCC_SEED=42
export TCC_SBI_INIT_CHECKPOINT="$TCC_MODELS_ROOT/dino/srm/finetune_robust/seed_$TCC_SEED/weights/best.pth"
export TCC_RUN_DIR="$TCC_EXPERIMENT_ROOT/sbi-hf-dino-srm/seed_$TCC_SEED"
python research_cli.py experimental train --family sbi \
  --config configs/experimental/sbi_hf_adaptation.yaml \
  --seed "$TCC_SEED" --device cuda --execute
```

Repeat with `clip` and all canonical seeds. Checkpoint loading validates the source architecture, original resolution, input channels and encoding; the trained artifact contains the complete classifier state and resolved network specification. Calibrate and evaluate with the common [server runbook](SERVER_RUNBOOK.md) and `configs/experimental/suites/sbi.yaml`.

## Local pilot reproduction

`scripts/sbi_pilot.py` accepts explicit source manifests, image roots, SQLite landmarks, generic weights and an output directory. It runs the three generic arms for five epochs by default, with batch 8 and accumulation 4, yielding 65 matched optimizer updates for 202 source reals. `--hf-checkpoint` adds a separately labeled one-epoch adaptation smoke; `--seed` changes the realization. Every run verifies full validation prediction reload, stores per-arm bootstrap intervals and compares paired AUC differences against MFFI. Failed runs remain in `pilot.json`. This bounded driver rejects source populations above 10000 rows; use the common training CLI for full scale.

Final pilot numbers and decisions are in [REPORT.md](REPORT.md). Calibration and model selection use source validation only. Degraded source-validation predictions reuse the frozen clean threshold. Test access requires a frozen candidate record and is never used to change these recipes.
