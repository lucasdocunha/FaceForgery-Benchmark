# DINO-SRM representation pilots, 2026-10-03

These are min-data source-validation development results. They are not final
Test, Test-D, DF-40 or Celeb-DF measurements. No test split was opened by these
runs. The fixed DINO-SRM backbone already used all MFFI training labels and full
validation for checkpoint selection, so the 5%/10% results concern downstream
adaptation only. These pilots do not establish end-to-end label efficiency.

All 21 scheduled fits completed. Seed 42; 1,000 train and 1,000 validation images;
at most ten epochs, patience five; two CPU threads; no GPU allocation. Fitting
took 20.47 seconds in total, excluding feature extraction and later bootstrap
analysis. Process peak RSS was 976,220 KiB (a shared-process high-water mark,
not an isolated peak per model). The original fitting implementation was
`35695e0`; complete code/package/cache identities and all candidate metrics are
in [the JSON record](dino_srm_min_val.json). The [CSV](dino_srm_min_val.csv)
also lists losses, parameters, epochs, optimizer steps, thresholds and runtimes.
SGD baseline parameter and step counts were derived from saved coefficients and
the recorded iteration budget, correcting their original nonparametric telemetry
placeholder. No scores or predictions were changed.

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

No learned metric or GNN is promoted. SupCon did not exceed the MLP or raw
centroids; no GNN exceeded the same-mask MLP. The 1,000-draw paired image-group
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
  --cache-index "$FFB_DINO_SRM_CACHE_INDEX" \
  --output "$FFB_PILOT_OUTPUT" --epochs 10 --seed 42
python -m scripts.representation_dependency_smoke \
  --cache-index "$FFB_DINO_SRM_CACHE_INDEX" \
  --output "$FFB_DEPENDENCY_OUTPUT"
```

The cache index must contain exactly `train` and `val` cache paths. Both commands
require fresh output directories and at most 10,000 rows per split. The original
arrays, model files, histories and predictions remain outside git. The report
retains all successful and failed scheduled candidates; this pilot had no failed
candidate. Optional FAISS 1.15.1 was tested separately, with no dense fallback.

Final focused validation: 58 tests passed in 6.06 seconds. Coverage includes
cache integrity, all scoring/training variants, native and PyG GCN/GAT/SAGE,
masked-label invariance, query batch invariance, shared projection masks, dynamic
rebuild lineage, ANN recall/memory guards, a bounded minibatch over a 524,429-node
reference, checkpoint reload and frozen calibration. The evaluation workstream
also completed actual metric-centroid and graph-GCN train/calibrate/dry-run/execute
flows across all four synthetic target kinds. Real full-target execution and
canonical five-seed scientific comparisons remain server work.
