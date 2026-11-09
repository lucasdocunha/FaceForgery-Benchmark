# XAI for the new pilot models

The retained `explicability.py` pipeline still supports the original model families, twelve methods and native spatial/frequency comparisons. New pilot checkpoints are not silently masqueraded as those legacy model families. The additional entry point `python -m src.robustness.explain` connects a completed pilot, certified predictions and a fixed reference-stratified cohort to primary IG, occlusion and explicitly selected CNN-layer Grad-CAM.

## Freeze one 64-image cohort

```bash
python -m src.robustness.explain sample \
  --predictions outputs/evaluation/reference-clean/predictions.csv \
  --calibration outputs/research/<reference-run>/calibration.json \
  --output outputs/cohorts/reference64.csv --seed 42
```

This requires 16 TP, 16 FN, 16 TN and 16 FP from the reference checkpoint on the declared test population; a short stratum fails rather than changing the design. The same identities are reused for every pilot. The cohort CSV and JSON certificate are frozen together. Other models retain their own decisions and need not have the same stratum. This is a diagnostic sample, not an unbiased estimate of error prevalence.

## Plan and run a primary explanation

```bash
python -m src.robustness.explain run \
  --run outputs/research/<pilot-run> \
  --manifest data/canonical/mffi/test.csv --root /path/to/phase1/testset \
  --predictions outputs/evaluation/<pilot-run>-clean/predictions.csv \
  --cohort outputs/cohorts/reference64.csv \
  --method integrated_gradients --baseline blur --steps 64 \
  --output outputs/pilot-xai
# Review the plan, then append --execute --device cuda:0.
```

For Grad-CAM, specify an actual spatial CNN layer such as `--method gradcam --layer backbone.layer4` for the ResNet pilot. Nonspatial/token outputs are rejected; there is no silent substitution of attention maps. `--method occlusion` replaces declared RGB windows by the baseline. `--shard-count N --shard-index I` partitions IDs deterministically, and `--resume` verifies the contract, image and output hashes before skipping an existing cell. A `.lock` prevents two processes writing the same cell; remove a stale lock only after verifying no process is active.

Each executed cell includes raw input/baseline/attribution arrays, image and checkpoint identity, method metadata, the recomputed fake probability, an overlay and file hashes. Fresh scores must match the frozen export within 1e-5. IG saves a completeness residual; it is a numerical diagnostic, not a faithfulness guarantee. All-zero maps remain marked. Source images and outputs remain ignored by Git.

## Coordinate and interpretation boundaries

These pilot maps explain the **end-to-end fake-minus-real margin in RGB input coordinates**, including the differentiable FFT path inside the model. They are not native frequency maps and must not be placed on an FFT representation. Use the retained pure-frequency-model protocol for native spectral coordinates. The pilot's raw-zero baseline is a black RGB input, unlike zero in the retained legacy pipeline's already encoded space. The blurred baseline uses a fixed radius of two pixels after resizing. Each overlay is independently normalized; the raw signed values are authoritative.

The three primary adapters supplement, not replace, the retained XAI controls and coverage audits. A complete scientific analysis still requires baseline/budget sensitivity, parameter-dependence checks, appropriate perturbation controls and review of the full gallery. The code does not invent explanatory findings or infer demographics. The default 64-image cohort must not be used to estimate population explanation quality.
