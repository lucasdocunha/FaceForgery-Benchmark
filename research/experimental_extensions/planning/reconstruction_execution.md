# Reconstruction execution contract

Experiments A-D use `src.experimental.reconstruction`, the shared resumable experimental runtime and the existing audited inference/statistics/artifact functions. They do not use the legacy trainer, which evaluates test during fit. Genuine pretraining filters the certified train manifest to label 0. Source validation chooses AE checkpoints by genuine reconstruction quality; detector checkpoints use source-validation AUC. The first anomaly baseline is mean raw-RGB L1, with fixed larger-is-faker orientation. Its `[0,1]` score is an anomaly score, not a calibrated fake probability.

## Architectures and gradient policies

CAE, VAE and gated CAE have four strided convolution stages. The gated default permits only an H/8 skip; H/4 is the shallowest configurable boundary. VAE training samples its posterior, while inference uses its mean. The latent MLP consumes `mu`, `logvar` and analytical KL, with normalization fitted on train only. A fixed arithmetic mean combines spatial and latent predictions; `combine` saves both complete models into a standalone audited run.

`ae_mode` states exactly which updates are allowed:

| Mode | Classification gradients into AE | Reconstruction gradients | AE learning rate |
|---|---|---|---|
| `frozen` | None | None | No AE optimizer group |
| `recon_finetune` | None | Genuine examples only | `lr_ae` |
| `end_to_end` | All supervised examples | Genuine examples only | `lr_ae` |

The final AE in `end_to_end` has supervised fake exposure and is not a genuine-only model. Pure-fake windows in `recon_finetune` leave AE gradients absent, so AdamW does not apply decay or momentum updates to the AE in those windows. Backbone and head use `lr_backbone` and `lr_head`; parameter groups are disjoint and exhaustive.

`input_mode: x_only | residual_only | full` controls the required spatial ablation. All modes have the same physical 9-channel stem, or 10 with `gradient: true`, and identical initial backbone/head weights at the same seed. The single active representation is placed in channels 0-2, with channels 3-8 zeroed. Thus `residual_only` receives the same generic pretrained first-layer weights as `x_only`; it does not begin behind a zero-weight residual stem. Full fusion uses `[x, reconstruction, abs(x-reconstruction)]`. RGB/reconstruction receive ImageNet normalization and absolute residual is divided by ImageNet standard deviations. Optional Sobel magnitude is the last channel and is zero in `x_only`.

The server backbone is generic ImageNet ResNet-18 with an explicit offline state file. Its first RGB weights are copied and added channels are initialized to zero. `small` is a compact scratch control for CPU tests. Any full-fusion AUC must be compared to the matched RGB and residual-only arms before attributing usefulness to reconstruction.

For full ResNet fusion, zero initialization means CE reaches the AE only after the added stem channels begin learning. A focused test and the GPU smoke check zero AE classification gradient on the initial pass and a nonzero gradient after the first stem update. Genuine reconstruction can update a trainable AE from the first step independently of that classification path.

## Loss and exact resume

The composite objective is mean L1 + `lambda_ssim * (1-SSIM)` + `lambda_lpips * LPIPS` + `beta * KL`. SSIM and KL arithmetic run in FP32. LPIPS is frozen, preserves gradients to reconstructed pixels and requires a complete pretrained state, including its feature network. Construction never downloads weights and strict loading rejects partial states. An absent optional dependency or requested asset fails explicitly.

`kl_reduction: sum` means the batch mean of KL in nats per sample. `mean_per_dim` divides that quantity by latent dimension before beta weighting. Both nats per sample and nats per dimension are always logged. Configs use `mean_per_dim` with VAE beta maximum 0.001 as a starting setting, not a validated optimum. The CLI exposes the scale in each config; changing latent size or image resolution remains an ablation.

Linear and cyclic beta schedules depend only on successfully completed optimizer updates. Schedule configuration, global step, optimizer, scheduler, scaler, RNG and epoch are in immutable run/checkpoint identities. An interrupted partial epoch restarts from the last completed epoch. Tests interrupt a stochastic augmented VAE after one checkpoint and require exact uninterrupted/resumed weights and history for both schedules, with gradient accumulation.

Completed runs contain model/optimizer checkpoints, history, telemetry, source-validation predictions with SHA certificates, frozen Youden calibration and metrics. Source image bytes, manifests, initialization assets and implementation files are hashed. Inference rebuilds from a full checkpoint without the original AE run, generic backbone file or LPIPS asset. Calibration binds checkpoint bytes and `run.json`; changing architecture or score configuration invalidates the input contract. Evaluation identities distinguish a seed realization from its controlled condition for seed aggregation.

## Portable server commands

Set absolute paths in the environment: `TCC_TRAIN_MANIFEST`, `TCC_VAL_MANIFEST`, `TCC_TRAIN_ROOT`, `TCC_VAL_ROOT`, `TCC_RUN_DIR`, and, where required, `TCC_LPIPS_STATE`, `TCC_RESNET18_WEIGHTS`, `TCC_AE_RUN`. Manifests must be certified canonical manifests, with fake=1. Train and val are the only input populations accepted by fit. No workstation-specific path is in the configs. Every `server_*.yaml` uses 224 pixels.

```bash
python research_cli.py experimental train --family reconstruction \
  --config configs/experimental/reconstruction/server_vae.yaml \
  --seed 42 --device cuda:0 --execute

python research_cli.py experimental train --family reconstruction \
  --config configs/experimental/reconstruction/server_residual_frozen.yaml \
  --seed 42 --device cuda:0 --execute

python research_cli.py experimental train --family reconstruction \
  --config configs/experimental/reconstruction/server_latent_frozen.yaml \
  --seed 42 --device cuda:0 --execute

python research_cli.py evaluate-suite \
  --config configs/experimental/suites/reconstruction.yaml --execute
```

Use the canonical seed set 42, 123, 2024, 7, 2025, setting a fresh `TCC_RUN_DIR` for each condition and seed. Point `TCC_AE_RUN` at the matching completed pretraining run. The source configs provide CAE, VAE and gated AE, frozen spatial/latent heads and both spatial AE adaptation modes. Copy the spatial config for `x_only`, `residual_only` and `full`, keeping all remaining settings and generic initialization fixed. Evaluation uses the suite's Test, Test-D, DF-40 and Celeb-DF manifests, with the source-val threshold frozen. The suite dry run validates paths/contracts without model inference when `--execute` is omitted.

The fixed combined model needs no fitting. Set `TCC_SPATIAL_RUN` and `TCC_LATENT_RUN` and use the same public command:

```bash
python research_cli.py experimental train --family reconstruction \
  --config configs/experimental/reconstruction/mean_ensemble.yaml \
  --seed 42 --device cuda:0 --execute
```

It is also available through Python:

```python
from src.experimental.reconstruction import combine
combine(spatial_run, latent_run, output_dir, device="cuda:0")
```

The resulting run uses the same suite as an individual reconstruction model. Both components must use the same input resolution and certified train/val populations. Complete child weights are saved, so the combined inference run remains portable after its source run directories are removed.

## Local development campaign

`python -m src.experimental.reconstruction.pilot` accepts explicit source manifest/root paths, output directory and offline weights. `--phase ae` trains CAE/VAE/gated for three epochs on all 202 min-train reals and, if requested, a one-epoch LPIPS/cyclic-beta VAE smoke. `--phase detectors` trains all three frozen ResNet input arms for two epochs each on the full 1,000-image train population, a frozen latent head, one-epoch checks for both AE adaptation modes in spatial and latent heads, and the fixed spatial/latent mean. Default local resolution is 128, AE width 8, latent dimension 32, batch 8 and seed 42. Equal-budget arm settings are mechanically checked by a test.

Run serially under the shared GPU lock and the resource watchdog. Phase limits are at most 1,200 seconds and 3,072 MiB RSS; stop for memory pressure. A short structural GPU smoke is separate from this method comparison. The campaign never accepts a test manifest. It records and checks equal initial model-state hashes and realized optimizer-update counts across the three spatial input arms. `--phase report --reference-root PATH` reads only completed source-validation outputs and HF `predictions_val.csv` files. Historical HF filename IDs are joined explicitly to canonical `img_name`, with exact population and label checks. Summaries report AUC intervals, EER/F1/accuracy, raw-score variation, prediction correlation, repaired/regressed errors and fixed-mean AUC changes. The paired full-fusion versus RGB AUC interval uses the same bootstrap draws.

These are pilot/min-dataset/development results. A single seed and 202 genuine training faces do not establish full-benchmark improvements. HF checkpoints saw min-train and used a validation population containing min-val for selection. Manifest groups lack identities/videos; the supplied grouping cannot establish identity-independent confidence intervals. Code tests and structural smokes are implementation evidence, not quality results. Pilot results will be recorded separately after the authorized GPU campaign, before any frozen min-test access.
