# Experimental training and four-target evaluation

All selected families use `research_cli.py experimental` for training, frozen-feature caching, source-validation calibration and evaluation. Existing `evaluate-suite`, `calibrate-legacy`, `calibrate-run` and `aggregate-seeds` commands remain available. Commands are dry runs unless `--execute` is present. JSON and YAML configs expand declared environment variables recursively, reject unresolved variables, and resolve relative path fields from the config directory.

The six suite templates in `configs/experimental/suites/` cover reconstruction, SBI, metric learning, graph learning, VLM and MoE. A reconstruction template also evaluates genuine-only reconstruction scores, residual detectors, latent detectors and their exported spatial/latent mean ensemble. A metric or graph template evaluates the trained scorer and its saved normalizer/reference arrays. Each template takes a completed run directory as `TCC_EVAL_MODEL` and an input-bound calibration file as `TCC_EVAL_CALIBRATION`.

Training uses a separate run directory for each canonical seed. For example:

```bash
for seed in 42 123 2024 7 2025; do
  python research_cli.py experimental train \
    --family reconstruction --config configs/experimental/reconstruction/server_cae.yaml \
    --seed "$seed" --output "${TCC_OUTPUT_ROOT}/cae/seed_${seed}" \
    --device cuda:0 --execute
done
```

Replace the family/config with the chosen experiment. `--resume` restores a completed-epoch checkpoint; it does not authorize changing frozen scientific settings. `--seed` and `--output` overrides prevent all five realizations from writing to one configured directory. Pilots and benchmark outputs need separate locations and declared scopes.

Frozen feature extraction uses the original checkpoint builder and exact RGB or SRM preprocessing. SRM means RGB plus three residual channels computed after ImageNet normalization. Caches bind checkpoint/config hashes, certified manifest, image hashes, preprocessing, extraction point, code and packages. A changed key cannot reuse an existing cache.

```bash
python research_cli.py experimental cache \
  --checkpoint "$TCC_FEATURE_CHECKPOINT" --manifest "$TCC_VAL_MANIFEST" \
  --root "$TCC_VAL_ROOT" --output "$TCC_FEATURE_ROOT" \
  --device cuda:0 --batch-size 8 --workers 0 --execute
```

The command prints the immutable cache path and identity. Repeat for the target manifests with the same extractor. Metric/graph suite options name an explicit query cache for each target; MoE options name every ordered expert source, including certified reconstruction predictions. Cache preflight verifies file inventories, extractor contracts, sample annotations and current image hashes without constructing a network.

Every new family can produce an input-bound calibration with one command. Native image and VLM scorers need no cache options:

```bash
python research_cli.py experimental calibrate \
  --family reconstruction --run "$TCC_EVAL_MODEL" \
  --manifest "$TCC_VAL_MANIFEST" --root "$TCC_VAL_ROOT" \
  --output "$TCC_CALIBRATION_DIR" --threshold-policy youden \
  --device cuda:0 --batch-size 16 --workers 0 --execute
```

For metric/graph scorers, add `--options path/to/options.yaml` containing `cache: ${TCC_VAL_FEATURE_CACHE}`. For MoE, options contain `sources`, with the same expert names, roles and kinds as training. The calibration manifest must be the run's emitted `val_select.csv`; full validation includes router-fitting rows and is rejected. To calibrate a fusion baseline, add `method: mean`, `geometric`, `geometric_binary`, `logistic` or `expert_<name>` in its MoE options. Each method binds its own immutable checkpoint inventory and score contract. Graph protocol changes require their own source-validation contract and are rejected when they differ from the saved inference protocol.

Calibration is threshold selection, not a learned probability calibration. `youden` aliases the existing balanced-accuracy optimizer because both choose the same threshold, including the same smallest-threshold tie rule. EER and its interpolated ROC threshold are target-label diagnostics; the EER threshold never determines predictions. No target manifest is accepted by the calibration command. New experiments require the exact run bundle and input/score contract; unbound historical calibration is not accepted.

Set these paths before using a suite template:

| Variable | Meaning |
|---|---|
| `TCC_EVAL_MODEL` | Completed experimental run directory |
| `TCC_EVAL_CALIBRATION` | Frozen source-validation `calibration.json` |
| `TCC_OUTPUT_ROOT` | Fresh parent output location |
| `TCC_EVAL_TEST_MANIFEST`, `TCC_EVAL_TEST_ROOT` | Certified clean Test manifest and image root |
| `TCC_EVAL_TEST_D_MANIFEST`, `TCC_EVAL_TEST_D_ROOT` | Same logical Test IDs and labels, degraded image root |
| `TCC_EVAL_DF40_MANIFEST`, `TCC_EVAL_DF40_ROOT` | Certified DF-40 subset and image root |
| `TCC_EVAL_CELEB_MANIFEST`, `TCC_EVAL_CELEB_ROOT` | Certified official-list frame manifest and crops root |
| `TCC_EVAL_<TARGET>_FEATURE_CACHE` | Metric/graph query cache, with TARGET in TEST, TEST_D, DF40, CELEB |
| `TCC_EVAL_<TARGET>_SRM_CACHE`, `TCC_EVAL_<TARGET>_RGB_CACHE`, `TCC_EVAL_<TARGET>_RECON_PREDICTIONS` | Ordered MoE expert sources for each target |

DF-40 source CSVs with old absolute image paths are converted once using a declared prefix:

```bash
python -m src.robustness.df40 \
  --source "$TCC_DF40_SOURCE_CSV" --images-root "$TCC_EVAL_DF40_ROOT" \
  --path-prefix "$TCC_DF40_OLD_PREFIX" --output "$TCC_EVAL_DF40_MANIFEST" \
  --convention fake-is-1 --acknowledge-reviewed-subset
```

The server templates default to `pooled_all_real`, comparable to retained legacy pooled-real subgroup tables and confounded by source domain. To use `source_matched`, certify `source_domain` annotations with `--source-domain-column source_domain` and change the template policy. DF-40 fakes under `cdf/frames` overlap the Celeb-DF source domain, so the two target names do not establish source independence.

Celeb-DF official-list labels use real-is-1 and must be converted explicitly by the existing preparation pipeline. The suite never chooses polarity from AUC. Its primary Celeb-DF observation is one video, scored by the arithmetic mean of frame `p_fake`; source-frame thresholds transfer to videos when no source-video calibration exists. Frame metrics are secondary. The paired clean/degraded comparison requires identical sample IDs, labels and group IDs.

```bash
python research_cli.py evaluate-suite \
  --config configs/experimental/suites/reconstruction.yaml
python research_cli.py experimental evaluate \
  --config configs/experimental/suites/reconstruction.yaml --execute
```

Use the same commands with the selected family's template. Dry runs validate target/calibration/bundle metadata, cache dependencies and target pairing before model loading. Executed suites publish fresh per-target predictions, metrics JSON/CSV, ROC/score/confusion plots, subgroup reports, video predictions, completion certificates and `suite_metrics.json`. Reports contain AUC, AP, EER, F1, accuracy and balanced accuracy at the frozen source threshold. Bootstrap AUC intervals resample the declared groups; image-only MFFI IDs do not certify identity-level independence.

HPC evaluation uses the existing `scripts/slurm_evaluation.sh` wrapper with a suite YAML path and explicit environment variables. No workstation image path is required by a template. For independent seeds, pass each completed target directory to `aggregate-seeds`; exactly the declared seeds, target population, source-calibration population, condition and primary observation unit must match. Celeb summaries use video AUC. Across-seed sample standard deviation differs from bootstrap uncertainty within a target population.

Local fixtures validate orchestration and artifacts only. This machine has no real Test-D, DF-40 or Celeb-DF benchmark population, and no fixture metric is a full-benchmark result.
