# Audited robustness workflow — operator guide

All examples run from the repository root on **`robust-generalization` in the Lucas-PG fork**. Upstream is read-only. Training, evaluation and artifact analysis are separate operations. The examples are commands to run later, not a claim that the research experiments have been executed.

## 0. Preserve history and select an environment

The imported source is `lucasdocunha/FaceForgery-Benchmark@f00475116eb350aca8d6dbd08df999ab6f21991a`. The previous XAI, six-page article, ten-page reference and pt-BR presentation were merged without deleting them. `research/robustness/import.json` records both histories. The PR base is the fork's `ICLR`; merging remains a human action.

Use a separate environment for this workflow. The original `pyproject.toml` and `uv.lock` remain available for historical reproduction. Do not silently upgrade a historical experiment and label it an exact rerun.

```bash
python3.11 -m venv .venv-research
source .venv-research/bin/activate
python -m pip install --upgrade pip
# CPU checks / artifact-only preparation:
python -m pip install torch==2.6.0 torchvision==0.21.0 \
  --index-url https://download.pytorch.org/whl/cpu
python -m pip install -r requirements-research.txt
python -m pip check
python -m pytest tests/robustness tests/explicability -q
```

For an authorized NVIDIA research machine, install the same torch/torchvision pair from a compatible official CUDA index instead of the CPU index. For example, PyTorch lists `cu124` wheels for that pair. Driver compatibility and performance must be checked on the actual host. The versions above are a tested compatibility environment, not a claim to be the latest versions or a full security audit. Only load trusted checkpoints; `weights_only=True` is used, and it is not a substitute for trust, hash verification and a patched runtime. See the official installation matrix: https://docs.pytorch.org/get-started/previous-versions/ .

`requirements-research.txt` pins direct dependencies only. Each research run records the installed versions and code hashes; CI exports `pip freeze`. Keep that environment with the exact commit for resume. No font files, models, source face images, or private transcripts are committed by these commands.

## 1. Audit existing runs before adding compute

```bash
python research_cli.py inventory --models-root /path/to/models \
  --output outputs/audit/existing-runs.json
```

This checks checkpoint/configuration presence, prediction schemas, duplicate IDs and required split exports. `artifacts_present_not_reproduced` is deliberately **not** `complete`: the initializer, preprocessing and image-level reproduction are not proven by filenames. Review `research/robustness/AUDIT.md`, especially the `finetune_robust` constructor issue and inconsistent seed-completion tables.

Freeze the target model roster, checkpoint hashes, canonical seed set, source partitions, class meanings and metric definitions before claiming a comparison. Do not count a checkpoint with missing metrics as a completed seed.

## 2. Canonicalize manifests explicitly

Canonical labels are **0 real / 1 fake**. The required CSV columns are `img_name,label,sample_id,group_id,dataset,split`. The sidecar `<file.csv>.json` binds those exact bytes and records the declaration. A hash is not a signature or a proof that the labels are correct.

The MFFI examples assume its source CSV column is named `label` and that the team has independently verified fake-positive semantics. Do not infer semantics solely from numbers.

```bash
for SPLIT in train val test; do
  python research_cli.py convert-manifest \
    --source data/raw/$SPLIT.csv \
    --output data/canonical/mffi/$SPLIT.csv \
    --dataset mffi --split $SPLIT \
    --label-column label --convention fake-is-1
done
```

Use `--group-column source_identity` when suitable source/identity annotations exist in the original CSV. Otherwise groups default to images and the provenance explicitly marks cross-image dependence unresolved. For videos, use video IDs; for repeated identities spanning videos, prefer identity groups where valid metadata permits. Generator and `source_domain` columns are copied if supplied and must be complete for per-generator reporting. They are never guessed from faces.

The default ID hashes dataset plus relative path, excluding split so clean/degraded counterparts can share a logical identity. The **ordering of the converted CSV is retained** for explicitly acknowledged legacy positional exports. Do not sort or replace that file before legacy import. A new CSV version is required for edits.

Audit images before research execution:

```bash
python research_cli.py audit-images --manifest data/canonical/mffi/train.csv \
  --root /path/to/phase1/trainset --hash-images \
  --output outputs/audit/mffi-train-images.json
```

The audit reports invalid files and exact duplicate bytes; it does not certify that different images are different people. Training validates disjoint supplied sample/group/source/content identities between source train and source validation. A missing `source_id` or content hash remains an explicit limitation rather than an invented clean bill of health.

For Test-D, create a separate canonical manifest with `split=test_d` but retain the same dataset name and logical image filenames. Use the actual degraded root. Different names require a verified identity mapping; never align them by sorted row number.

## 3. Repair the Celeb-DF route without erasing the old evidence

The old list convention is **1 real / 0 fake**. New preparation verifies official folder semantics and converts once. It uses full relative video paths as IDs and lossless PNG crops, and writes extraction coverage and hashes. A missing/undecodable video or undetected face fails certification by default.

```bash
python scripts/prepare_celeb_df.py \
  --dataset-root /path/to/Celeb-DF-v2 \
  --output-dir /path/to/celeb-df-crops-v2 \
  --manifest-out data/canonical/celeb-df-v2/test.csv \
  --frames-per-video 15 --target-size 512
```

`--no-face center` explicitly enables and records center fallback. Review the fallback rate; it is not equivalent to a detected face. `--limit-videos N` creates a manifest labeled `smoke_test`, not the full test set. Use fresh versioned output paths. The extractor is intentionally deterministic and conservative rather than an automatically scheduled GPU job. Face detector/crop differences from another paper must be disclosed.

Existing crops can instead be retained to isolate the label correction:

```bash
python research_cli.py convert-manifest \
  --source data/celeb_df/test.csv \
  --output data/canonical/celeb-df-legacy/test.csv \
  --dataset celeb-df-legacy --split test \
  --label-column target --convention real-is-1 --group-column video_id
```

This **does not repair old filename-stem collisions, inconsistent crops, or omitted frames**. Audit those before using this route. If the old crop paths collide, regenerate with full identities. Do not publish `1-AUC` alone as a corrected experiment.

## 4. Reuse old prediction exports without rerunning models

This is the lowest-compute route when complete, correctly identifiable exports exist. The checkpoint file is hashed but not deserialized. Declare what the original label and class-1 probability mean separately.

```bash
python research_cli.py import-predictions \
  --source /path/to/run/results/predictions_val.csv \
  --manifest data/canonical/mffi/val.csv \
  --checkpoint /path/to/run/weights/best.pth \
  --source-labels fake-is-1 --class-one fake --acknowledge-row-order \
  --output outputs/imported/run-a/validation.csv
python research_cli.py calibrate-export \
  --predictions outputs/imported/run-a/validation.csv \
  --output outputs/imported/run-a/calibration.json
```

Missing/duplicate IDs are rejected. The flag acknowledges the **original manifest row order**, not permission to assume any arbitrary alignment. An import is labeled an existing-export reanalysis, not fresh image inference. If a previous export used real-positive labels but fake-positive scores, set `--source-labels real-is-1 --class-one fake`; do not reverse both indiscriminately.

## 5. Fresh evaluation of historical checkpoints

First freeze thresholds from source validation. Historical builders may require their original pretrained cache to construct a model before the full trained state is loaded; supply the authorized cache and exact saved config. The loader retains unversioned `legacy-v1` constructor semantics. It does not guess a different architecture to satisfy an incompatible state dict.

```bash
python research_cli.py calibrate-legacy \
  --checkpoint /path/to/run/weights/best.pth \
  --manifest data/canonical/mffi/val.csv --root /path/to/phase1/valset \
  --class-one fake --device cuda:0 --output outputs/calibration/run-a
python evaluate_celeb_df.py \
  --checkpoint /path/to/run/weights/best.pth \
  --manifest-csv data/canonical/celeb-df-v2/test.csv \
  --crops-dir /path/to/celeb-df-crops-v2 \
  --calibration outputs/calibration/run-a/calibration.json \
  --class-one fake --device cuda:0 --output-dir outputs/evaluation/run-a-celeb
```

The new evaluator is deliberately **not compatible** with the old `--models-root --skip-existing` batch invocation. Old queue scripts are historical and should not be launched unchanged. Use an explicit frozen list of checkpoint commands. Output includes raw keyed frame scores, video-mean scores, frame/video metrics, calibration identity, image hashes and a completion record. AUC is not accuracy; one-class subgroup AUC is undefined, never zero. Target labels are never used to choose orientation or thresholds.

Legacy weights use the original dataset's NumPy FFT helpers. New pilot models use their own differentiable internal torch FFT. Both paths are versioned; do not silently switch a historical checkpoint's preprocessing to a new one.

## 6. Compare errors and grouped uncertainty before fusion

```bash
python research_cli.py compare \
  --reference outputs/evaluation/rgb/predictions.csv \
  --other outputs/evaluation/frequency/predictions.csv \
  --reference-calibration outputs/calibration/rgb/calibration.json \
  --other-calibration outputs/calibration/frequency/calibration.json \
  --draws 1000 --seed 42 --output outputs/comparisons/rgb-frequency
```

Inputs must have valid prediction sidecars and belong to the supplied checkpoint calibrations. The sample sets, labels and group identities must match exactly. Output distinguishes repaired, regressed, both-correct and both-wrong cases and includes a paired cluster-bootstrap interval for `other AUC − reference AUC`. The oracle accuracy bound is explicitly non-deployable. These intervals condition on the fixed trained models and grouping; they do not capture training-seed uncertainty. With image-only groups, identity/source correlation remains unresolved.

`report-generators` uses each fake-generator group together with real examples from the same supplied source domain. A missing real reference yields undefined AUC, and the macro average reports how many groups were defined. Report these coverage counts; do not present a partial macro as full coverage.

`shared-failures --predictions ... --calibrations ... --output ...` retains the entire intersection for the declared roster, including an empty result. It does not label missing predictions as errors or call the result a vulnerability universal to all models.

## 7. Pilot a small ablation ladder

Edit the paths in `configs/research/pilot-rgb.yaml` to the real **source train/validation** locations. Do not add test paths to training configs. `robust_v1` is a new explicitly recorded training recipe, not a claim to exactly reproduce the heterogeneous older robust scripts.

```bash
python research_cli.py expand-plan \
  --template configs/research/pilot-rgb.yaml \
  --variants rgb_basic rgb_robust --seeds 42 \
  --output outputs/plans/augmentation-pilot
python research_cli.py train \
  --config outputs/plans/augmentation-pilot/rgb_robust_seed42.yaml
# Review the printed plan. Only the following command actually trains:
python research_cli.py train \
  --config outputs/plans/augmentation-pilot/rgb_robust_seed42.yaml \
  --execute --device cuda:0
```

`expand-plan` generates YAML and a plan, **not a shell queue or HPC job submission**. Respect the machine scheduler, allocations and other users. Start with the augmentation pair. Only after useful validation complementarity is established, add the fusion ablations from `ablation-plan.json`.

The ladder separates single-view augmentation, paired clean/strong CE, paired JS consistency, early channel concatenation, late mean-logit fusion, adaptive residual fusion and an equal-parameter auxiliary spatial control. The control uses luminance with the same auxiliary encoder; it is not asserted to match runtime because FFT cost differs. The spectral-only branch is intentionally small and is not a capacity-matched replacement for a large pretrained detector.

The training objective is CE on the final logits; paired runs average clean/corrupted CE; optional JS penalizes prediction disagreement; optional auxiliary CE trains branch predictions. Adaptive fusion uses `RGB logits + gate × auxiliary logits` with optional branch dropout. The gate is **not** calibrated uncertainty. The defaults are starting budgets, not measured optimal hyperparameters. The default checkpoint criterion is clean source-validation AUC. RGB-only JS and paired-CE controls are essential to distinguish representation gains from consistency or extra-forward-pass gains.

## 8. Resume and evaluate a new pilot

The training command prints a content-identified output directory. It records config, initial weights, source code/environment and manifest hashes. Best and last checkpoints are separate. A partial checkpoint does not count as completion.

```bash
python research_cli.py train --config /path/to/frozen-config.yaml --execute --resume --device cuda:0
python research_cli.py evaluate --run outputs/research/<run-id> \
  --manifest data/canonical/mffi/test.csv --root /path/to/phase1/testset \
  --device cuda:0 --output outputs/evaluation/<run-id>-clean
```

Resume is at completed epoch boundaries. Use the **same code commit, environment, config and certified data**. A new recipe/teacher/data identity produces a new run. A crashed process may leave `.running.lock`; verify that no process is active before removing that lock manually. Never delete an active lock to start a competing writer. A missing or modified artifact invalidates completion. Interrupted partial-epoch work is repeated from the last complete epoch.

Optional standard distillation takes `training.teacher_run`, a positive `distillation_weight` and `temperature`. The teacher must be a completed pilot with the same source train/validation manifests and input size. Its checkpoint is bound to the plan. Compare the same student without a teacher and with ordinary KL distillation. This is **not an implementation of all ensemble-teacher or literature-specific distillation methods**.

## 9. Confirm with paired seeds and measure cost

Only after freezing finalists, expand the explicit canonical seed set, for example `[42,123,2024,987,7]`. Do not pool an extra seed for only one system. `aggregate-seeds` requires exactly the requested seed set for every variant/dataset/split and the same evaluation manifest. It reports sample SD across seeds separately from the image/video-bootstrap interval.

```bash
python research_cli.py profile --run outputs/research/<run-id> \
  --device cuda:0 --batch-size 1 --warmup 10 --iterations 50 \
  --output outputs/cost/<run-id>.json
```

This measures **resident-tensor model plus internal RGB/FFT encoding**, not end-to-end deployment. It excludes decoding, face extraction, transfer and I/O. It records synchronized median/p95 latency, raw timings, parameter count, CUDA memory where available and environment. Measure on the same hardware/dtype/batch size before comparing cost; do not infer GPU latency from CPU tests or parameter count.

## 10. Connect to the retained XAI pipeline and publication artifacts

`export-xai` writes an explicit legacy-format manifest, prediction table and identity map from certified inputs. Configure the retained `explicability.py` pipeline using those exact exported orders and its full validation/test roster. The bridge does not automatically run XAI or claim the pilot architecture is an existing legacy model family.

Use the preserved three tasks for matched clean/degraded examples, repaired/regressed cases and fixed-roster intersections. Keep the 64-image reference-stratified sample as a diagnostic sample, not a population estimate. A separate random/weighted sample is needed for population explanation summaries. Frequency bins are not facial coordinates; signed input maps and rectified CAMs are not interchangeable.

See `research/robustness/PUBLICATION.md` for the results-to-paper contract. Existing six-page `paper.pdf`, ten-page reference and `/presentation` remain intact. Future observed results must be regenerated from verified artifacts, with exact dataset/protocol, exclusions, source overlap and seed counts. No script changes the primary PDF, submits a paper, publishes face images, or merges the PR automatically.

## New-pilot XAI and cross-seed equivalence

For new pilot checkpoint explanations, use the separate `python -m src.robustness.explain` entry point and [pilot XAI guide](pilot-xai.md). It supports fixed-cohort IG, occlusion and declared CNN-layer Grad-CAM with fresh-score checks, hashes and verified resume. Its maps are end-to-end RGB-input attributions through the FFT path, **not native-frequency maps**. The original twelve-method native-coordinate workflow remains available separately.

New evaluation records include the model/training condition, source manifest hashes, preprocessing, augmentation definition and software identity. `aggregate-seeds` rejects runs with different conditions even if they share a display name. The seed is the excluded experimental factor; local output/data paths are not treated as scientific conditions. Teacher bytes may differ across seeds, so a consistent teacher-selection policy must still be reviewed. The initial backbone-state hash is recorded separately to verify paired initialization.
