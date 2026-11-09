# DINO-SRM representation pilots, 2026-10-03

These are min-data source-validation development results. They are not final
Test, Test-D, DF-40 or Celeb-DF measurements. No test split was opened by these
runs. The fixed DINO-SRM backbone already used all MFFI training labels and full
validation for checkpoint selection, so the 5%/10% results concern downstream
adaptation only. These pilots do not establish end-to-end label efficiency.

All 21 scheduled fits completed. Seed 42; 1,000 train and 1,000 validation images;
at most ten epochs, patience five; two CPU threads; no GPU allocation. Fitting
took 20.50 seconds in total, excluding feature extraction and later bootstrap
analysis. Process peak RSS was 974,596 KiB (a shared-process high-water mark,
not an isolated peak per model). That clean replay used integrated implementation
`0f7079f`; complete code/package/cache identities and all candidate metrics are
in [the JSON record](dino_srm_min_val.json). The [CSV](dino_srm_min_val.csv)
also lists losses, parameters, epochs, optimizer steps, thresholds and runtimes.
The original 21 successful fits at `35695e0` remain in
[the initial summary](dino_srm_min_val_initial.json). A representative original
bundle correctly [failed the later implementation-hash check](original_bundle_reload_rejection.json).
All 21 fits were then rerun with identical settings in a fresh source-only output
directory. AUCs agree within 1.2e-16; there was no new configuration search.
Final SupCon and GAT bundles [passed reload checks](final_bundle_reload_verification.json)
against their saved validation CSVs, with maximum absolute score differences
below 6e-17. The old runs and the rejected reload attempt were retained, not
overwritten. SGD coefficient counts and iteration budgets are recorded correctly
in the final telemetry.

The checkpoint's own logits had validation AUC 0.99401, accuracy 0.970 and EER
0.03186 at its validation-selected Youden threshold. Accuracy and F1 below also
use each method's own validation-selected threshold. Validation was reused for
selection and reporting; these are optimistic development estimates.

| Metric method, all 1,000 training labels | AUC | F1 | Accuracy | EER |
| --- | ---: | ---: | ---: | ---: |
| Raw-feature cosine centroids | 0.99399 | 0.9718 | 0.967 | 0.0338 |
| Raw-feature cosine k-NN | 0.95790 | 0.9625 | 0.955 | 0.0722 |
| Linear probe | 0.95334 | 0.9607 | 0.953 | 0.0662 |
| MLP probe | 0.99097 | 0.9610 | 0.955 | 0.0466 |
| SupCon, cosine centroid score | 0.98980 | 0.9719 | 0.967 | 0.0355 |

SupCon used 128 dimensions, a 64-unit hidden layer, balanced batches and zero
hard-negative weighting. Its Euclidean centroid AUC was 0.99237 as a diagnostic;
that score did not replace the predeclared cosine selection rule. Cosine
silhouette was 0.85144. The [t-SNE plot](dino_srm_supcon_min_val_tsne.png) uses all
1,000 validation rows and seed 42; it was inspected for rendering and never used
for selection.

| Graph comparison, identical masks and input geometry | 5% AUC (51 labels) | 10% AUC (101 labels) |
| --- | ---: | ---: |
| L2-regularized logistic SGD | 0.98183 | 0.95557 |
| MLP | 0.99378 | 0.99368 |
| Cosine k-NN | 0.98505 | 0.97449 |
| Label propagation | 0.98883 | 0.98967 |
| Fixed-scale Correct-and-Smooth | 0.98990 | 0.98525 |
| GCN | 0.98316 | 0.98574 |
| GAT | 0.99266 | 0.98996 |
| GraphSAGE | 0.99160 | 0.99066 |

Masks contain 11 real/40 fake labels at 5% and 21 real/80 fake labels at 10%,
using per-class ceiling counts. Every comparator within a graph fraction used
the same mask and unit-normalized standardized features. Graphs were static,
directed cosine k=10; GNN fanouts were 10/5 with a 128-dimensional trainable
projector and 32 hidden units. G/H probe preprocessing and widths differ, so
comparisons across the two tables are not a label-budget ablation.

The clean comparison is near the HF checkpoint ceiling and cannot establish
that a method is ineffective. SupCon did not exceed the MLP or raw centroids;
no GNN exceeded the same-mask MLP. The 1,000-draw paired image-group
bootstrap gave SupCon minus MLP AUC difference -0.00118, 95% interval
[-0.00488, 0.00264], and 5%-GAT minus 5%-MLP -0.00112, interval
[-0.00248, 0.00005]. All eight predeclared paired contrasts are retained in the
JSON. These intervals are conditional on validation-selected fits and image-only
groups. Identity/video dependence is unresolved, and they are not independent
confirmation or across-seed uncertainty.

SupCon scores correlated 0.9348 with checkpoint scores and repaired three of its
validation errors while introducing six others. At 5% labels, GAT correlated
0.9949, repaired seven and introduced fourteen. Raw centroids correlated 0.9803,
repaired four and introduced seven. Error overlap is only a diagnostic; no
improved ensemble is claimed or selected from these observations.

## Fixed degraded-validation follow-up

The same 1,000 min-val images were scored after one fixed degradation draw per
image, using certified recipe `9c99d6ba69c6`. The HF logits fall from 0.99401 to
0.96143 AUC, giving the controls more room to differ. This is a source-development
proxy, not Test-D or untouched validation: min-val was part of the full
validation split used to select the HF checkpoint. All methods retain their
original clean-validation Youden thresholds. Proxy labels were removed before
the prediction callbacks and used only for reporting.

The [JSON record](dino_srm_degraded_proxy.json) and
[CSV](dino_srm_degraded_proxy.csv) include clean and degraded AUC, F1, accuracy,
EER, frozen thresholds and confusion counts for all 21 fits plus the original
checkpoint logits. The degraded results below use those original thresholds.

| Metric method | Clean AUC | Degraded AUC | Degraded F1 | Degraded accuracy | Degraded EER |
| --- | ---: | ---: | ---: | ---: | ---: |
| Original DINO-SRM logits | 0.99401 | 0.96143 | 0.9077 | 0.894 | 0.1054 |
| Raw-feature cosine centroids | 0.99399 | 0.96156 | 0.9003 | 0.887 | 0.1078 |
| Raw-feature cosine k-NN | 0.95790 | 0.85648 | 0.8900 | 0.862 | 0.2129 |
| Linear probe | 0.95334 | 0.89066 | 0.8840 | 0.852 | 0.1716 |
| MLP probe | 0.99097 | 0.95373 | 0.8928 | 0.879 | 0.1115 |
| SupCon, cosine centroid score | 0.98980 | 0.93921 | 0.9108 | 0.897 | 0.1078 |

| Graph comparison | 5% clean AUC | 5% degraded AUC | 10% clean AUC | 10% degraded AUC |
| --- | ---: | ---: | ---: | ---: |
| L2-regularized logistic SGD | 0.98183 | 0.94638 | 0.95557 | 0.90569 |
| MLP | 0.99378 | 0.95839 | 0.99368 | 0.95882 |
| Cosine k-NN | 0.98505 | 0.94991 | 0.97449 | 0.92378 |
| Label propagation | 0.98883 | 0.94520 | 0.98967 | 0.94670 |
| Fixed-scale Correct-and-Smooth | 0.98990 | 0.95318 | 0.98525 | 0.94528 |
| GCN | 0.98316 | 0.94930 | 0.98574 | 0.95178 |
| GAT | 0.99266 | 0.95449 | 0.98996 | 0.95399 |
| GraphSAGE | 0.99160 | 0.95538 | 0.99066 | 0.95570 |

On the degraded proxy, SupCon minus MLP AUC is -0.01452, with a 95% paired
interval of [-0.02495, -0.00405]. Against raw centroids it is -0.02235
[-0.03249, -0.01209]; against the linear probe it is +0.04855
[0.03335, 0.06414]. All six GCN/GAT/GraphSAGE minus same-mask MLP comparisons
at 5% and 10% have negative intervals. All nine paired controls, on both clean
and degraded images, are in the JSON. They use 1,000 image-group bootstrap
draws and unadjusted 95% intervals, conditional on the selected fits. Image-only
groups leave identity/video dependence unresolved.

These proxy AUC results support retaining the simpler controls for this local
setting; they do not establish cross-dataset superiority. SupCon has higher
degraded accuracy than the MLP at the original operating thresholds despite
lower AUC, so the metrics should not be treated as interchangeable. No learned
metric or GNN is promoted. Seed 42 controls only downstream fitting and label
selection; the HF extractor is fixed and already supervised with all MFFI
training labels. There is no claim of upstream seed independence.

The compatibility repair at integrated commit `724d4f3` keeps the extraction
commit as provenance while comparing checkpoint, code, package and preprocessing
hashes. The [replay audit](dino_srm_degraded_replay.json) records all 21 old-bundle
implementation rejections and the 21 new fits. Only output paths and the t-SNE
diagnostic flag changed; no hyperparameters, masks or selection rules changed.
The old fits, failed audits and first reporting pass remain in
[the attempt inventory](dino_srm_degraded_retained_attempts.json).

The first strict comparison rejected four GCN/GAT score arrays at a 1e-12
tolerance. Their maximum drift was 1.1921e-7; the documented float32 tolerance
is 1e-6 with zero relative tolerance. All original thresholds, label masks,
confusion counts and selected epochs remain exact, and clean AUCs agree within
1e-12. Every new bundle reload matches its saved scores exactly when the
certified CSV is parsed with round-trip float precision. The initial reporting
pass used default CSV parsing, which moved one clean score below its exact
threshold in six methods; that pass is retained, and the final clean metrics
match the original records. Degraded prediction hashes are identical across
both reporting passes.

The replay took 18.40 seconds with peak process RSS of 934,312 KiB and no GPU
allocation. One DINO-SRM extraction took 29.01 seconds, with peak process RSS
of 2,103,276 KiB and peak allocated VRAM of 700,614,656 bytes. Jobs used the
shared GPU or CPU lock, two CPU threads and a 3 GiB / 1,200-second watchdog.
The cache contract repair passed 64 focused tests in 6.33 seconds. No t-SNE
plot was regenerated and no test split was read.

The [follow-up driver](degraded_proxy_followup.py) exposes `replay`,
`verify-replay`, `extract` and `score` commands. Point `PYTHONPATH` to the
integrated implementation. Replay the original pilot into a fresh directory,
verify it into a separate record, extract the certified proxy under the GPU
lock, then score the verified replay with that extraction record. All paths
are explicit arguments; `--help` documents each command. Source caches, full
prediction files, driver snapshots and model bundles remain outside git.

## Metric-to-graph dependency smoke

The [additional smoke](dino_srm_dependency_smoke.json), rerun under final model
code at `8fd3449`, trained a two-epoch SupCon head on the selected 51 labels, then
a two-epoch PyG GraphSAGE on exactly the same mask and projected representation.
The graph's learned projector rebuilt neighbors at both epochs with different
adjacency hashes. Reloaded predictions were bit-exact. Source-validation AUC was
0.98707 for the metric head and 0.98493 for the dynamic graph; these are dependency
smoke metrics, not an additional selection sweep. Runtime was 0.760 and 1.741
seconds, peak shared-process RSS 943,764 KiB, zero GPU allocation. Rebuild times
and hashes are retained in the JSON.

## Reproduction and validation

```bash
python -m scripts.representation_pilot \
  --cache-index "$TCC_DINO_SRM_CACHE_INDEX" \
  --output "$TCC_PILOT_OUTPUT" --epochs 10 --seed 42
python -m scripts.representation_dependency_smoke \
  --cache-index "$TCC_DINO_SRM_CACHE_INDEX" \
  --output "$TCC_DEPENDENCY_OUTPUT"
```

The cache index must contain exactly `train` and `val` cache paths. Both commands
require fresh output directories and at most 10,000 rows per split. The original
arrays, model files, histories and predictions remain outside git. The report
retains all successful and failed scheduled candidates; this pilot had no failed
training candidate. The separate original-bundle reload rejection is recorded
above. Optional FAISS 1.15.1 was tested separately, with no dense fallback.

Final focused validation: 58 tests passed in 6.06 seconds. Coverage includes
cache integrity, all scoring/training variants, native and PyG GCN/GAT/SAGE,
masked-label invariance, query batch invariance, shared projection masks, dynamic
rebuild lineage, ANN recall/memory guards, a bounded minibatch over a 524,429-node
reference, checkpoint reload and frozen calibration. The evaluation workstream
also completed actual metric-centroid and graph-GCN train/calibrate/dry-run/execute
flows across all four synthetic target kinds. Real full-target execution and
canonical five-seed scientific comparisons remain server work.
