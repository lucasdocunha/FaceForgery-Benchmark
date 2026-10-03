# Cached representations and low-label graphs

Experiments G and H consume immutable caches from `src.experimental.features`.
The cache contains the input to the checkpoint's final classification layer,
float32 logits and certified row identities. Extraction reproduces the original
six-channel SRM input, including SRM on ImageNet-normalized RGB. A query cache
must match the fitted checkpoint, extraction settings, code and package identity.
Prediction callbacks also verify current image hashes and join by `sample_id`.

These are development methods, not new benchmark results. The released HF
backbones already saw all MFFI training labels and were selected on full
validation. A 5% label run therefore measures downstream label efficiency only.
Validation labels select checkpoints and freeze Youden thresholds; they are
additional disclosed supervision. No query labels enter model prediction,
neighbor construction, loss, propagation or residual correction.

## Training and required comparisons

The common entry point is `research_cli.py experimental train --config FILE
--execute`. A call without `--execute` validates the configuration. Set the
following environment variables to portable paths and a declared seed:

```bash
export TCC_TRAIN_CACHE=/data/features/source-train-cache
export TCC_VAL_CACHE=/data/features/source-val-cache
export TCC_RUN_DIR=/runs/metric-supcon-seed42
export TCC_SEED=42 TCC_DEVICE=cpu TCC_LABEL_FRACTION=1.0 TCC_METRIC_KIND=supcon
python research_cli.py experimental train --config configs/experimental/metric_srm.yaml --execute
```

All three templates require `TCC_RUN_DIR`, `TCC_SEED`, `TCC_DEVICE`,
`TCC_TRAIN_CACHE`, `TCC_VAL_CACHE` and `TCC_LABEL_FRACTION`. Metric training also
requires `TCC_METRIC_KIND`; graph training requires `TCC_GRAPH_KIND`. Fractions
must be in `(0, 1]`; use 0.05 and 0.10 for the declared low-label comparison.
`graph_fullscale.yaml` additionally requires `TCC_GRAPH_PROTOCOL` (`inductive`
or `transductive`), `TCC_PROJECTION_RUN` (a fitted SupCon directory) and
`TCC_LABEL_MASK` (its `label_mask.json`). Paths and variable expansion are checked
by the shared CLI. The full-scale graph fit checks that the projection and graph
use the exact same selected training labels and matching requested fraction.

Metric kinds are `linear`, `mlp`, `centroid`, `knn` and `supcon`. The SupCon head
has a 128- or 256-dimensional L2-normalized output. Its score can be `centroid`,
`knn` or a separately fitted `linear` probe. The backbone remains frozen.
`hardness: 0` is ordinary supervised contrastive loss; positive hardness applies
detached, per-anchor mean-one weights only to opposite-class negatives. Balanced
batches have two or more distinct rows of each class. There are no duplicate
views fabricated from a single cached vector and no additional class weights.

Compare the original cached checkpoint logits, raw-feature centroids/k-NN,
regularized linear and MLP probes, then SupCon. `standardize: true` fits a
training-only, label-independent scaler. Set it false for raw cosine geometry.
Centroid scores have fixed fake polarity: cosine-to-fake minus cosine-to-real,
Euclidean distance-to-real minus distance-to-fake, or squared-distance difference.
Only squared Euclidean and cosine scores have identical rankings after unit
normalization. Distance scores are bounded scores, not calibrated probabilities.
Silhouette and a seeded, at-most-1000-row t-SNE plot are diagnostic only.

For every graph fraction, run all eight kinds under the same seed and label mask:

```bash
export TCC_LABEL_FRACTION=0.05
for TCC_GRAPH_KIND in linear mlp knn lp correct_smooth gcn gat sage; do
  export TCC_GRAPH_KIND
  export TCC_RUN_DIR=/runs/graph-${TCC_GRAPH_KIND}-fraction05-seed42
  python research_cli.py experimental train --config configs/experimental/graph_srm.yaml --execute
done
```

Repeat the comparison at 10% and later the canonical seeds 42, 123, 2024, 7 and
2025. The persisted `label_mask.json` lists selected IDs and permitted labels.
Masks are nested, deterministic and grouped by `source_id` when available,
otherwise `group_id`. A supplied `label_mask` file reuses the exact mask and
rejects unknown IDs, changed selected labels or split groups. All remaining
labels become -1 before training. The LR baseline uses L2-regularized SGD
logistic loss with averaged iterates. C&S is explicitly fixed-scale residual
correction followed by clamped smoothing; it is not autoscaled C&S.

Each graph run saves LR, k-NN, LP and C&S development diagnostics in addition to
its selected method. The separate MLP run is required for the comparison. Shared
baselines do not imply that validation-selected thresholds were fitted on test:
each method's `metrics.json` uses its own source-validation Youden threshold.

## Graph protocols and scale

`model.protocol: inductive` is the primary protocol. Each query has a disjoint
two-hop ego graph whose neighbors come only from source training nodes. Queries
cannot update source states or connect to other queries. The implementation
tests query batch invariance for native and PyG GCN, GAT and GraphSAGE, and checks
directed GCN normalization against a hand calculation. GNN graph policy is
explicitly directed; LP/C&S additionally support union and mutual edges.
GNNs use unweighted adjacency, with learned attention for GAT. LP/C&S use
nonnegative `(1 + cosine) / 2` edge weights and explicit self loops.

`model.protocol: transductive` fits the classifier on source training data but
uses a source-plus-one-query-population graph at validation and inference.
Query-query edges and collective propagation are allowed. The calibration
contract binds this protocol, and switching protocol on a fitted artifact fails.
Never combine Test, Test-D, DF-40 or Celeb-DF query populations. Transductive
scores depend on the target population; they are not per-image zero-shot scores.

The static graph is required. `graph.dynamic: true` trains a projector on the
selected labels and refreshes detached neighbors every `rebuild_interval`
epochs; selection and inference rebuild from the chosen projector. Normalization
is never cached across graph changes. Completed-epoch dynamic resume requires
`rebuild_interval: 1`; wider intervals fail explicitly because replay of stale
training adjacency is not implemented. Label-free `noise_fraction` rewires
source edges. Oracle opposite-class rewiring exists only as an explicit small
stress-test primitive and is never used for selection.

Exact search is tiled and refuses references above 10,000 nodes. It does not
silently substitute a dense search for an unavailable ANN library. Full-scale
configuration uses CPU FAISS HNSW, sparse SciPy propagation and bounded two-hop
PyG/native edge-scatter training. There is no dependency on PyG sampling kernels.
FAISS recall@k is audited against exact search on at most 128 fixed queries and
must reach 0.95. Audit clean ANN edges before optional random edge perturbation.

At 524,429 nodes with 128 dimensions, k=20 and M=16, stored normalized features
cost about 256 MiB; neighbor indices/scores cost 120 MiB; the conservative HNSW
estimate is about 499 MiB, excluding process and dataframe memory. Directed CSR
and propagation have additional costs. `memory_estimate.json` records component
estimates, and the ANN index guard enforces `memory_budget_mb`. These are not
measured peaks: dataframe metadata, page cache, framework temporaries, allocator
overhead and concurrent copies require extra RAM. Dense pairwise FP32 similarity
alone would exceed 1 TiB. The bounded-ego test uses a 524,429-node reference and
checks that minibatch storage depends only on batch size and fanouts.

Use a low-label SupCon projection to reduce full-scale graph dimension:

```bash
# First fit metric_srm.yaml at the SAME fraction, seed and source cache.
export TCC_PROJECTION_RUN=/runs/metric-supcon-fraction05-seed42
export TCC_LABEL_MASK=$TCC_PROJECTION_RUN/label_mask.json
export TCC_LABEL_FRACTION=0.05 TCC_GRAPH_KIND=sage TCC_GRAPH_PROTOCOL=inductive
export TCC_RUN_DIR=/runs/graph-sage-fraction05-seed42
python research_cli.py experimental train --config configs/experimental/graph_fullscale.yaml --execute
```

The graph run checks that every selected projection label exactly matches its
own mask. It copies the small projector and normalizer into its portable bundle.
A fully supervised projector cannot be reused in a 5% downstream graph run.
This restriction does not erase prior full-label supervision of the HF backbone.

## Portable inference and artifacts

`fit_metric_run(config)` and `fit_graph_run(config)` return fitted directories.
`predict_metric_run(run_dir, features)` and `predict_graph_run(run_dir, features,
protocol=None)` take unlabeled arrays. `describe_run` returns the artifact path,
research identity and input contract. `feature_cache_predictor(run_dir,
{target_name: cache_path}, protocol=None)` returns the common suite callbacks;
each has a read-only `validate_population(frame, root)` hook.

Supply one certified target cache per target in the common evaluation suite's
`model.options.target_caches`, with `model.type: experimental` and `family:
metric` or `graph`. `options.protocol` must match graph validation. All four
target kinds use the same frozen calibration and common forensic metrics.
The operator must extract each cache from the original frozen backbone first.
Inference needs the fitted bundle plus target caches and the matching images;
it does not need the original training caches or a second training command.

The bundle binds model weights, scaler, permitted labels, reference features,
neighbor arrays, projection and propagation state by SHA-256. `run.json`,
`history.json`, `pipeline_telemetry.json`, `metrics.json`, mask files and graph
diagnostics record provenance and runtime. Feature arrays and checkpoints stay
outside git. Completed-run `resume` is idempotent; changed configs or cache
identities fail instead of replacing a frozen calibration.
Inference also checks the recorded implementation hashes. A changed predictor
requires its recorded code version or a new fit/calibration. The evaluation
condition separates downstream seed and run paths from fixed extractor and
method identity; aggregation across seeds therefore uses the same frozen
backbone. Imported masks keep their original selection seed even when the
downstream initialization seed changes.

Optional dependencies are `torch-geometric` for `backend: pyg` and `faiss-cpu`
for `graph.backend: faiss`. Explicit `backend: native` uses ordinary torch sparse
edge scatter. Missing requested backends fail with actionable errors.
