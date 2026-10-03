# Experimental server runbook

Run commands from the repository root in the prepared environment. GPU commands belong inside a server allocation; CISIA evaluation can use `sbatch scripts/slurm_evaluation.sh <suite.yaml>`. Stage pretrained weights and optional family dependencies before training. Keep the recorded implementation and artifacts together: inference checks immutable bundle, preprocessing and cache identities.

## Paths and certified source populations

Set absolute paths for `TCC_TRAIN_MANIFEST`, `TCC_VAL_MANIFEST`, `TCC_TRAIN_ROOT`, `TCC_VAL_ROOT`, `TCC_RUN_DIR`, `TCC_OUTPUT_ROOT` and `TCC_PRETRAINED_ROOT`. `TCC_RUN_DIR` is one fresh model run; `TCC_OUTPUT_ROOT` holds fresh report directories. `TCC_EXPERIMENT_ROOT` below is an operator-chosen parent for seed runs. JSON/YAML expands environment variables, rejects unset variables and resolves relative paths from the config directory. CLI `--output` overrides the run destination, but variables referenced by the config must still be set.

Source CSVs must declare relative `img_name` paths and reviewed binary labels. Convert source train/validation separately, using the same dataset identity and the actual label convention:

```bash
python research_cli.py convert-manifest \
  --source "$TCC_TRAIN_SOURCE_CSV" --output "$TCC_TRAIN_MANIFEST" \
  --dataset mffi --split train --label-column label --convention fake-is-1
python research_cli.py convert-manifest \
  --source "$TCC_VAL_SOURCE_CSV" --output "$TCC_VAL_MANIFEST" \
  --dataset mffi --split val --label-column label --convention fake-is-1
python research_cli.py audit-images \
  --manifest "$TCC_TRAIN_MANIFEST" --root "$TCC_TRAIN_ROOT" \
  --output "$TCC_SOURCE_AUDIT" --hash-images
```

Use `--convention real-is-1` when that is the source convention. Add `--group-column <column>` when reviewed source/video/identity groups are available. Repeat the image audit for validation. Preserve each manifest's `.csv.json` certificate. Training rejects supplied train/validation overlap; image-only groups do not establish identity independence. Target manifests never enter fitting or source calibration.

## Train and cache

Choose a family and reviewed training YAML. These supplied configs use explicit TCC source paths and `TCC_RUN_DIR`:

| Family | Training config under `configs/experimental/` | Additional inputs |
|---|---|---|
| `reconstruction` | `reconstruction/server_cae.yaml`, `server_vae.yaml`, `server_gated.yaml` | `TCC_LPIPS_STATE` for enabled LPIPS |
| `sbi` | `sbi_generic.yaml` or `sbi_hf_adaptation.yaml` | Generic `TCC_SBI_ARM=sbi\|mffi\|mixed`; HF `TCC_SBI_INIT_CHECKPOINT`; staged MobileNet weights and landmark cache as declared in YAML |
| `metric` | `metric_srm.yaml` | `TCC_TRAIN_CACHE`, `TCC_VAL_CACHE`, `TCC_METRIC_KIND`, `TCC_LABEL_FRACTION`, `TCC_SEED`, `TCC_DEVICE` |
| `graph` | `graph_srm.yaml` or `graph_fullscale.yaml` | Same cache/seed/device/fraction variables, `TCC_GRAPH_KIND`; fullscale also `TCC_PROJECTION_RUN`, `TCC_LABEL_MASK`, `TCC_GRAPH_PROTOCOL` |
| `vlm` | `vlm_qwen3_2b_server.yaml` | Pinned local base under `TCC_PRETRAINED_ROOT`; Smol templates retain pilot sample limits |
| `moe` | `moe_frozen_soft.yaml` or `moe_frozen_top2.yaml` | `TCC_FEATURE_SRM_VAL`, `TCC_FEATURE_RGB_VAL`, `TCC_RECON_VAL_PREDICTIONS`, `TCC_RECON_CALIBRATION` |
| `moe`, strong SRM pair | `moe_srm_pair_new_expert.yaml` | `TCC_FEATURE_DINO_SRM_VAL`, `TCC_FEATURE_CLIP_SRM_VAL`, `TCC_NEW_EXPERT_ROLE`, `TCC_NEW_EXPERT_VAL_PREDICTIONS`, `TCC_NEW_EXPERT_CALIBRATION` |

```bash
family=reconstruction
train_config=configs/experimental/reconstruction/server_cae.yaml
device=cuda:0
for seed in 42 123 2024 7 2025; do
  export TCC_SEED="$seed"
  export TCC_RUN_DIR="${TCC_EXPERIMENT_ROOT}/${family}/seed_${seed}"
  python research_cli.py experimental train \
    --family "$family" --config "$train_config" --seed "$seed" \
    --output "$TCC_RUN_DIR" --device "$device"
  python research_cli.py experimental train \
    --family "$family" --config "$train_config" --seed "$seed" \
    --output "$TCC_RUN_DIR" --device "$device" --execute
done
```

Use a distinct condition directory for every architecture, SBI arm or ablation. `--resume` restores the same frozen settings. Reconstruction detectors require a matching completed VAE via `TCC_AE_RUN`; use `reconstruction/server_residual_*.yaml` and `server_latent_frozen.yaml`, with `TCC_RESNET18_WEIGHTS` where declared. `reconstruction/mean_ensemble.yaml` takes `TCC_SPATIAL_RUN` and `TCC_LATENT_RUN`; both components must share the requested seed, source populations and resolution.

SBI generic arms share initialization and update budget. Set `TCC_LANDMARK_CACHE` to the YAML's declared cache destination (`${TCC_DATA_ROOT}/landmarks/train-real.sqlite`) and prepare source landmarks once with the staged MediaPipe asset:

```bash
python -m src.experimental.sbi.landmarks \
  --manifest "$TCC_TRAIN_MANIFEST" --root "$TCC_TRAIN_ROOT" \
  --asset "$TCC_LANDMARK_ASSET" --output "$TCC_LANDMARK_CACHE"
```

For HF adaptation, explicitly choose the matching seed and DINO/CLIP/MobileNet checkpoint in `TCC_SBI_INIT_CHECKPOINT`. This condition has prior MFFI fake exposure and is recorded separately from generic SBI.

The cache interface extracts frozen features/logits from an existing legacy RGB or SRM checkpoint with its exact preprocessing. Repeat for source train, source validation and each target, using the same selected extractor. The printed `cache` path, rather than its parent directory, is the family input:

```bash
python research_cli.py experimental cache \
  --checkpoint "$TCC_FEATURE_CHECKPOINT" --manifest "$TCC_VAL_MANIFEST" \
  --root "$TCC_VAL_ROOT" --output "$TCC_FEATURE_ROOT" \
  --device cuda:0 --batch-size 8 --workers 0 --execute
```

Metric kinds include `linear`, `mlp`, `centroid`, `knn`, `supcon`; graph kinds include `linear`, `mlp`, `knn`, `lp`, `correct_smooth`, `gcn`, `gat`, `sage`. Use the same nested label mask for comparisons. A fullscale metric-to-graph projection must be SupCon trained with the exact graph mask; the saved `inductive`/`transductive` protocol cannot be overridden at inference.

## Freeze calibration

Set `TCC_EVAL_MODEL` to the completed run. Existing run calibration is usable when its recorded bundle and input contract still match. Native image/VLM example for fresh source-validation calibration and certified source scores:

```bash
python research_cli.py experimental calibrate \
  --family "$family" --run "$TCC_EVAL_MODEL" \
  --manifest "$TCC_VAL_MANIFEST" --root "$TCC_VAL_ROOT" \
  --output "$TCC_CALIBRATION_DIR" --threshold-policy youden \
  --device "$device" --batch-size 16 --workers 0 --execute
export TCC_EVAL_CALIBRATION="${TCC_CALIBRATION_DIR}/calibration.json"
```

For metric/graph, add `--options "$TCC_CALIBRATION_OPTIONS"` with this YAML; use CPU for cached scorer inference:

```yaml
cache: ${TCC_VAL_CACHE}
```

For limited VLM smoke runs, use `--manifest "${TCC_EVAL_MODEL}/selected_val.csv"` with the full validation image root. Full server Qwen uses the full source-validation manifest.

For MoE, use the run's emitted `val_select.csv`; full validation includes fitting rows and is rejected. The exact command retains the full validation root and expert cache sources:

```bash
python research_cli.py experimental calibrate \
  --family moe --run "$TCC_EVAL_MODEL" \
  --manifest "${TCC_EVAL_MODEL}/val_select.csv" --root "$TCC_VAL_ROOT" \
  --options "$TCC_CALIBRATION_OPTIONS" --output "$TCC_CALIBRATION_DIR" \
  --device cpu --threshold-policy youden --execute
```

Its options declare the same ordered frozen experts as training:

```yaml
sources:
  - {name: srm, role: srm, kind: feature_cache, path: "${TCC_FEATURE_SRM_VAL}"}
  - {name: rgb, role: rgb, kind: feature_cache, path: "${TCC_FEATURE_RGB_VAL}"}
  - {name: reconstruction, role: reconstruction, kind: predictions, path: "${TCC_RECON_VAL_PREDICTIONS}"}
```

Use reconstruction calibration's `validation_predictions.csv` and `.csv.json` as the frozen MoE source scores. For a MoE baseline, put `method: mean`, `geometric`, `geometric_binary`, `logistic` or `expert_<name>` in the calibration options and suite `model.options`; each method has its own bound inventory/calibration. Threshold selection is distinct from learned probability calibration. Final artifacts bind the exact run bundle, fake-is-1 score semantics and input contract; experimental evaluation rejects unbound calibration and requires no `allow_unbound_calibration` switch.

The strong-pair condition keeps DINO-SRM and CLIP-SRM and adds a source-validated new expert. Set `TCC_NEW_EXPERT_ROLE` to its actual saved family, `reconstruction` or `sbi`. Its calibration options use the ordered names `dino_srm`, `clip_srm`, `new_expert`, with roles `srm`, `srm`, and the declared new role:

```yaml
sources:
  - {name: dino_srm, role: srm, kind: feature_cache, path: "${TCC_FEATURE_DINO_SRM_VAL}"}
  - {name: clip_srm, role: srm, kind: feature_cache, path: "${TCC_FEATURE_CLIP_SRM_VAL}"}
  - {name: new_expert, role: "${TCC_NEW_EXPERT_ROLE}", kind: predictions, path: "${TCC_NEW_EXPERT_VAL_PREDICTIONS}"}
```

Both server MoE conditions use `expert_seed_policy: matched`: every expert's recorded seed must equal the router seed. Stage the corresponding HF checkpoints and new-expert run for each canonical seed. The explicit `fixed` policy is reserved for downstream router variation on unchanged experts and records that narrower uncertainty scope. Compare the new expert and router against the original two-expert mean/geometric controls on the identical held-out rows; adding a weak expert can make a three-way mean artificially easy to beat.

## Prepare the four targets

Set `TCC_EVAL_<TARGET>_MANIFEST` and `TCC_EVAL_<TARGET>_ROOT` for `TARGET=TEST,TEST_D,DF40,CELEB`.

| Target | Required semantics |
|---|---|
| Test | Clean MFFI target, certified split `test` |
| Test-D | Same logical sample IDs, labels, groups and relative names as Test, altered image bytes; certified split `test_d` |
| DF-40 | Reviewed subset with generator/paradigm annotations; templates use `pooled_all_real`, retaining the source-domain confounder |
| Celeb-DF v2 | Full official-list extraction, real-is-1 converted explicitly; video is primary, arithmetic mean frame `p_fake`; frame metrics are secondary |

Convert the reviewed clean Test CSV twice with `convert-manifest`, using the same dataset, label/group settings and `img_name` values but `--split test` and `--split test_d`. Bind their roots to clean and degraded crops, respectively. Conversion excludes split from logical IDs, preserving pairing.

DF-40 source CSVs use `img_path,target,method`. Strip the declared old absolute prefix when moving crops to the server:

```bash
python -m src.robustness.df40 \
  --source "$TCC_DF40_SOURCE_CSV" --images-root "$TCC_EVAL_DF40_ROOT" \
  --path-prefix "$TCC_DF40_OLD_PREFIX" --output "$TCC_EVAL_DF40_MANIFEST" \
  --convention fake-is-1 --acknowledge-reviewed-subset
python scripts/prepare_celeb_df.py \
  --dataset-root "$TCC_CELEB_VIDEO_ROOT" --output-dir "$TCC_EVAL_CELEB_ROOT" \
  --manifest-out "$TCC_EVAL_CELEB_MANIFEST" --frames-per-video 15 \
  --target-size 512 --no-face error
```

DF-40 `source_matched` is opt-in: declare `--source-domain-column source_domain` during conversion and update the suite policy. DF-40 `cdf/frames` and Celeb-DF share a source domain. Celeb preparation reads the official list, validates label polarity and certifies extraction coverage; limited smoke manifests cannot be benchmark scope. Source-frame thresholds transfer to video means unless source-video calibration is available.

A deterministic local degradation proxy can be built from a reviewed validation population for inference only:

```bash
python -m src.robustness.degraded_proxy \
  --manifest "$TCC_PROXY_SOURCE_MANIFEST" --root "$TCC_PROXY_SOURCE_ROOT" \
  --output "$TCC_PROXY_IMAGE_ROOT" --clean-manifest "$TCC_PROXY_CLEAN_MANIFEST" \
  --degraded-manifest "$TCC_PROXY_DEGRADED_MANIFEST" --seed 42 --image-size 224
```

The helper uses the repository robust augmentation operator, one sample-ID-derived draw per image, and lossless PNG bytes with original relative names. Its artifact inventory pins the recipe, code/packages, source certificate and image hashes. Both aliases preserve source dataset/IDs/groups/labels and use split `pilot`; clean images are referenced without copying. Evaluate these aliases in suite `scope: pilot` with frozen source thresholds. They are a degradation proxy, not the original Test-D distribution.

## Evaluate and retain artifacts

The six templates are `configs/experimental/suites/{reconstruction,sbi,metric,graph,vlm,moe}.yaml`. All require the four target path pairs, `TCC_EVAL_MODEL`, `TCC_EVAL_CALIBRATION` and `TCC_OUTPUT_ROOT`. Metric/graph additionally require `TCC_EVAL_<TARGET>_FEATURE_CACHE`; MoE requires `TCC_EVAL_<TARGET>_SRM_CACHE`, `TCC_EVAL_<TARGET>_RGB_CACHE` and `TCC_EVAL_<TARGET>_RECON_PREDICTIONS`, with expert order/names/contracts matching training. VLM requires its original staged base model and processor.

The additional `moe_srm_pair.yaml` suite matches the strong-pair condition. For each `TARGET=TEST,TEST_D,DF40,CELEB`, provide `TCC_EVAL_<TARGET>_DINO_SRM_CACHE`, `TCC_EVAL_<TARGET>_CLIP_SRM_CACHE` and `TCC_EVAL_<TARGET>_NEW_EXPERT_PREDICTIONS`; keep `TCC_NEW_EXPERT_ROLE` fixed. Produce each target's new-expert predictions through its own frozen suite first. All source/target caches must come from the same selected checkpoint realization and preprocessing. Neither source calibration nor router fitting reads target scores.

```bash
python research_cli.py experimental evaluate \
  --config "configs/experimental/suites/${family}.yaml" \
  --output "$TCC_SUITE_OUTPUT"
python research_cli.py experimental evaluate \
  --config "configs/experimental/suites/${family}.yaml" \
  --output "$TCC_SUITE_OUTPUT" --execute
```

Dry runs validate manifests, pairing, calibration, immutable bundles and cached image identities without building networks. Execute uses a fresh output directory. Retain `suite.json`, `suite_metrics.json`, `status.json`, and each target's predictions plus certificates, metrics, plots and subgroup/video outputs. A complete `status.json` certifies its artifact hashes. Metrics use the frozen source threshold; target EER thresholds are diagnostics. Group bootstrap intervals differ from across-seed sample standard deviation.

```bash
python research_cli.py aggregate-seeds \
  --evaluations "$TCC_SEED42_TARGET_OUTPUT" "$TCC_SEED123_TARGET_OUTPUT" \
                "$TCC_SEED2024_TARGET_OUTPUT" "$TCC_SEED7_TARGET_OUTPUT" "$TCC_SEED2025_TARGET_OUTPUT" \
  --seeds 42 123 2024 7 2025 --output "$TCC_SEED_REPORT"
```

Aggregate each target separately. Population, source-calibration population, condition and primary unit must match; Celeb aggregation uses video AUC. The original RGB/SRM checkpoint is a fixed dependency for downstream metric/graph seed comparisons. Local synthetic tests certify code/artifact behavior and supply no full-target benchmark result.

For separate scheduling, optionally split a reviewed full suite into four single-target YAMLs, preserving the expanded paths and matching cache subset:

```python
import copy, os
from pathlib import Path
import yaml
from src.experimental.configuration import read_document

suite = read_document(os.environ["TCC_SUITE_CONFIG"])
destination = Path(os.environ["TCC_SPLIT_CONFIG_DIR"])
destination.mkdir(parents=True, exist_ok=True)
for target in suite["targets"]:
    single = copy.deepcopy(suite)
    single["targets"] = [target]
    single["output"] = str(Path(os.environ["TCC_SUITE_OUTPUT"]) / target["name"])
    options = single["model"].get("options", {})
    if "target_caches" in options:
        options["target_caches"] = {target["name"]: options["target_caches"][target["name"]]}
    (destination / (target["name"] + ".yaml")).write_text(yaml.safe_dump(single))
```

```bash
for target in test test_d df40 celeb_df_v2; do
  python research_cli.py experimental evaluate \
    --config "${TCC_SPLIT_CONFIG_DIR}/${target}.yaml" --execute
done
```

The single full-suite command is preferred for the paired Test/Test-D bootstrap result. Separate jobs omit that paired comparison unless their certified predictions are recombined with a paired analysis.
