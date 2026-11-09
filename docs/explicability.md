# Explicability extension: implementation and experiment handoff

This extension is proposed only through **`explicability` → `ICLR`**, PR #1. The established ICLR training, model-building, evaluation, dependency lock, and legacy visualization paths remain unchanged. This is a **code and manuscript deliverable before full experiment execution**. Dataset access or a GPU is not a prerequisite for reviewing or merging the implementation. No benchmark training, full-cohort image inference, or GPU experiment was launched for this revision.

## Entry points and manuscript

- `explicability.py`: prepare, describe, run, compare, report, and export-tables.
- `configs/explicability.yaml`: frozen roster, methods, sampling seed, baselines, and estimator budgets.
- `src/explicability/`: identity contracts, cohort preparation, method adapters, execution, native-coordinate rendering, coverage, and table export.
- `paper/explicability/main.tex` and `references.bib`: revised article with attributed earlier results, new methodology, evidence-grounded hypotheses, and a clearly marked pre-execution boundary.
- `scripts/prepare_blinded_xai_review.py`: blinded human review of hard samples and matched controls, without automatically inferring attributes.
- `tests/explicability/`: fast synthetic checks; not trained-detector experiments.

The earlier SIBGRAPI paper and current fine-tuned release are not interchangeable experimental populations. The revised article does not transfer original scratch/frozen-model scores into a fine-tuned table, invent new heatmaps, or claim causal/demographic findings before the relevant analysis exists.

## Preserve the working environment

Use the existing ICLR environment and its normal checkpoint/pretrained-cache locations. Add only the opt-in attribution dependency:

```bash
uv run --with captum==0.9.0 python explicability.py --help
# Alternatively, inside the existing environment:
python -m pip install -r requirements-explicability.txt
```

The original strict checkpoint loader is reused. Its original pretrained-cache requirements still apply; this extension does not silently replace a fine-tuned architecture with scratch defaults just to avoid loading its initialization. Keep `results/run_config.json`, `weights/best.pth`, and the original prediction exports under their normal family/mode/regime/seed directories. Missing caches or incompatible state dictionaries are explicit failures, not grounds for using another model.

The new encoder follows `ImageDataset` rather than the legacy standalone visualization helper, whose phase rescaling differs. It preserves recorded six/seven-channel hybrid checkpoint inputs. Corrupt source images fail rather than being replaced by a neighboring CSV row. These safeguards affect only the new opt-in pipeline.

## 1. Freeze the analysis without executing models

Existing prediction CSVs must contain `id,y_true,prob_pos`; original manifests must contain `img_name,label`. The model root must contain the three RGB seeds and the validation metrics used for representation selection. Task 2 additionally needs the selected frequency checkpoint's validation/test predictions and configuration. Weights need not be present during preparation; execution subsequently binds their hashes.

```bash
MODELS=/path/to/existing/models
OUT=explicabilidade

uv run python explicability.py prepare \
  --models-root "$MODELS" \
  --val-manifest data/raw/val.csv \
  --test-manifest data/raw/test.csv \
  --acknowledge-legacy-row-ids \
  --output "$OUT"
```

The acknowledgement means the supplied manifests are the **original row order** used by the legacy numeric IDs, not newly sorted/rebuilt CSVs. Complete coverage and matching labels are checked, but they cannot prove byte-level correspondence. Execution verifies each selected image's fresh prediction before accepting its attribution.

Validation export defects are rejected by default. When using the previously audited export with identical duplicated rows, repeat preparation into a **new output directory** with `--curate-identical-validation`. This creates an explicitly recorded derived validation population: identical duplicates may be collapsed, missing observations are logged and never imputed, all models must cover the same observed population, and non-identical duplicates fail. Do not silently modify the original CSVs.

Add `--include-degraded` to compute a separately reported Test-D shared-error and metric summary from existing RGB predictions. Test-D numeric IDs are assumed to follow the original clean manifest; the assumption is recorded. The primary image XAI tasks operate on the clean test population. The degraded prediction summary is not presented as completed degraded-image XAI.

Preparation produces `plan.json`, immutable cohort CSVs, per-model Task 1 outcomes, threshold-calibrated prediction metrics, and the three-seed shared-error sensitivity CSV. It imports no detector and opens no image. It can be reviewed before scheduling any long experiment:

```bash
uv run python explicability.py describe --plan "$OUT/plan.json"
uv run python explicability.py run --plan "$OUT/plan.json" --task task1 --dry-run
uv run python explicability.py export-tables --output "$OUT"
```

The dry run reports workload dimensions, not invented wall-clock estimates. Table export uses supplied predictions and does not label the output a fresh inference reproduction. Reported seed standard deviations are not confidence intervals or proof of superiority.

## 2. What each protocol does

**Task 1:** exactly 64 unique images, 16 TP/FN/TN/FP according to the fixed ResNet RGB seed-42 reference. Every primary architecture receives the same identities and order. Other models retain their own confusion strata. Insufficient reference strata raise an error. This diagnostic cohort is not a representative sample for population accuracy or average explanation quality.

**Task 2:** select a matched RGB/pure-frequency architecture using validation AUC only. All three seeds must exist and exceed the 0.60 floor in both domains. Rank by mean worst-domain AUC, then smaller positive RGB-to-frequency loss. The chosen primary-seed pair explains the same 64 source images. This does not feed frequency tensors into an RGB-only checkpoint, train a new representation, or conflate representation robustness with degradation robustness. DCT requires matched DCT-trained checkpoints and is not silently substituted for FFT.

**Task 3:** include the complete shared-error intersection across the six primary RGB checkpoints, with no hidden cap. Empty intersections are valid. Separately retain the 18-checkpoint intersection as a seed-roster sensitivity result. Draw equal-size label-matched controls outside the six-checkpoint intersection. A control is not necessarily correct for every model. Nuisance matching beyond labels requires annotations.

## 3. Execute later, only when deliberately requested

The primary methods are Grad-CAM, Integrated Gradients, Gradient SHAP, Kernel SHAP, LIME, saliency, input-times-gradient, SmoothGrad, occlusion, feature ablation, Shapley sampling, and attention rollout. Unsupported architecture/method combinations are recorded; no different method is substituted under the requested label.

```bash
IMAGES=/path/to/original/test/images

for TASK in task1 task2 task3 task3_controls; do
  uv run --with captum==0.9.0 python explicability.py run \
    --plan "$OUT/plan.json" --task "$TASK" \
    --models-root "$MODELS" --images-root "$IMAGES" \
    --device cuda --resume
 done
```

This loop is an operator command, **not a workflow automatically executed by the repository or this revision**. The CLI defaults to CPU unless CUDA is explicitly requested. A prediction difference beyond the frozen tolerance, or a changed decision, stops the affected task before heatmaps are accepted. Check source order, preprocessing, checkpoint provenance, and precision instead of loosening tolerances to hide a mismatch.

For parallel work, use `--model`, `--method`, and `--shards N --shard-index K`. Image assignment is `id % N`; run every shard `K=0...N-1`. Sharding does not change the cohort. The same output root can be used for disjoint cells; avoid overlapping workers on the same cell. Example:

```bash
uv run --with captum==0.9.0 python explicability.py run \
  --plan "$OUT/plan.json" --task task3 \
  --models-root "$MODELS" --images-root "$IMAGES" \
  --model resnet.none.finetune.seed_42 \
  --method integrated_gradients --shards 4 --shard-index 0 \
  --device cuda --resume
```

Resume verifies model, image, plan, code/environment, and artifact identities. It never trusts a filename alone. Failures are append-only records; they are not complete cells. A crashed worker can leave a `.lock`; remove that lock only after verifying that no worker owns the cell. Changing the roster, preprocessing, baseline, seed, estimator budget, software, or source code requires a new output contract, not a silent overwrite.

## 4. Figures, controls, and complete coverage

```bash
uv run --with captum==0.9.0 python explicability.py compare --plan "$OUT/plan.json"
uv run python explicability.py report --plan "$OUT/plan.json" --require-complete
```

Raw arrays and metadata live in `task1/<model>/<method>/`, `task2/<model>/<method>/`, and `task3/<model>/<method>/`; controls have a separate `task3_controls/` root. Task 2 boards are under `task2/comparisons/<method>/` as PNG and PDF. RGB and FFT maps remain in their native coordinates. Complex real/imaginary components are displayed separately. A frequency bin is not a facial location, and independently normalized maps must not be interpreted as comparable absolute strengths.

All class-specific methods explain the fixed fake-minus-real logit margin. IG residuals, stochastic settings, baselines, masks, raw surrogate coefficients, constant-map flags, and runtime are saved. Kernel SHAP and LIME use actual Captum estimators; group-density displays are not individual-pixel Shapley values. RGB channels share a group; hybrid modalities remain separate. Default grid perturbations in frequency space need not produce a realizable image. Optional radial grouping supports band-oriented sensitivity, not a causal claim about compression.

Default controls include binary-head randomization and group insertion/deletion with a matched random-order baseline. Head-only randomization is **not** full cascading randomization; class-agnostic rollout can legitimately remain unchanged. Perturbation curves are off-manifold sensitivity diagnostics, **not** ROAR or causal validation. The default single replicate does not establish stochastic stability. For a separate sensitivity experiment, freeze another configuration with more replicates and/or a blurred-RGB baseline.

`report --require-complete` fails when cells are missing or corrupt. Unsupported cells are explicitly accounted for, not fabricated. Coverage does not certify explanation faithfulness: inspect control results, numerical residuals, constant maps, and visual validity before using a figure in the article.

## 5. Blinded shared-error review

```bash
uv run python scripts/prepare_blinded_xai_review.py \
  --plan "$OUT/plan.json" --images-root "$IMAGES" \
  --output "$OUT/reviewer_pack" \
  --private-linkage "$OUT/operator_only_linkage.json"
```

Share only the reviewer pack, under the dataset's permissions; never share the operator linkage with blinded reviewers. Reviewers see original image copies without predictions, heatmaps, or hard/control membership. Use independent annotation copies for at least two raters and retain disagreement and unassessable cases. The supplied fields concern visible illumination, occlusion, compression, and blur; the script does not infer demographic attributes. Valid subgroup claims require appropriate authorized annotations, sufficient denominators, uncertainty, source controls, and a prespecified analysis. A heatmap alone cannot establish the cause of a failure.

## Verification and publication boundary

The lightweight CI compiles Python and runs tiny synthetic CPU tests, with no dataset downloads, pretrained-weight downloads, or detector training. The paper workflow compiles LaTeX and checks references. These checks are deliberately separate from weeks-long research execution.

The revised manuscript is complete as a **pre-execution research revision**. Its previous benchmark table is explicitly attributed to the original work; its new XAI conclusions are research questions, not invented findings. After the team executes the frozen protocols, generated tables and figures can replace that boundary and support observed conclusions. Author approval, prior-publication overlap, and any venue-specific submission preparation remain human review responsibilities, not reasons to block this implementation.
