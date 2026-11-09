# Software verification ledger

Verification is tied to the exact source version. Passing these checks establishes software behavior on the tested fixtures, not new empirical face-forgery results.

## Complete suite: passed

Implementation commit: **`2b17a3e74b21e9040425120293930b6bba27ee53`**.

The pull-request integration check completed **209 tests and 20 subtests, all passing, in 41.05 seconds**. This is the complete `tests/` directory: imported benchmark tests, preserved XAI tests and new robustness tests. Dependency consistency (`pip check`), Python compilation and the research, pilot-XAI, Celeb evaluation and Celeb preparation CLI help checks also passed.

- [Complete CPU suite, run 35165090852](https://github.com/Lucas-PG/FaceForgery-Benchmark/actions/runs/35165090852)
- [Separate preserved-XAI check, run 35165090850](https://github.com/Lucas-PG/FaceForgery-Benchmark/actions/runs/35165090850)
- The PR check used GitHub's synthetic merge ref `30600be0b9c73f3532e67737774128ca18df30cb`, combining the implementation commit above with the unchanged fork `ICLR` base. This did **not** merge or update that branch.

Environment: Python 3.11.16, PyTorch 2.6.0+cpu, torchvision 0.21.0+cpu and the direct dependency set in `requirements-research.txt`. The run artifact includes `pip freeze`, test output, JUnit results, checked commit and a code snapshot. Model-hub downloads are disabled in the test job. The dependency file is not represented as a full transitive lockfile or security certification.

## What was exercised

- Canonical 0-real/1-fake conversion; explicit score orientation; exact prediction/manifest population alignment; hash and path safety; rejection of missing, duplicate or corrupted artifacts.
- Source-validation-only threshold fitting; one-class AUC represented as undefined; consistent video aggregation; group-bootstrap paired comparisons; per-generator real-reference requirements; error complementarity and empty shared-error intersections.
- Complete synthetic pilot training, saving, loading and evaluation. An interrupted run resumed at an epoch boundary produced the same model tensors as uninterrupted training under the tested CPU configuration.
- Matched augmentation determinism; pilot forward/backward paths; equal auxiliary parameter counts; teacher gradient isolation; experimental-condition fingerprints; bounded config expansion and measured toy model-compute profiling.
- Historical versus new initialization semantics across all six model families; strict old-constructor reconstruction; all original RGB/FFT/hybrid input encodings compared against the actual dataset helpers; missing-image substitution rejected.
- Primary IG, occlusion and CNN-layer Grad-CAM on generated tensors; exact immutable 64-image reference strata; output rendering and integrity checks. These are not real-image explanation-quality results.
- Generated video extraction with collision-safe full video identities, explicit center fallback and failure-without-certification. No real facial videos were used.
- Reviewed DF40-local-subset conversion, portable absolute-path mapping, explicit subset scope and missing-image rejection. This does not establish official DF40 coverage.
- Preserved six-page primary PDF, ten-page reference, original SIBGRAPI PDF, pt-BR presentation and notebook hashes against their existing build records.
- Imported benchmark model smoke tests and its small train/evaluate/ensemble/table/heatmap workflow. The obsolete default three-seed matrix assertion was replaced by an explicit three-seed test plus a test of the actual declared default configuration; no production seed population was changed to satisfy the test.

## Earlier checkpoints and resolved failures

The first implementation checkpoint passed 46 synthetic tests in 6.02 seconds at `1665f8b84c0c5e8e4143c60a8cae6e9c788ff279` ([run 35162138135](https://github.com/Lucas-PG/FaceForgery-Benchmark/actions/runs/35162138135)). Later passes exposed and fixed a test-module naming collision, a Pandas 3 read-only mask in shared-error accumulation and the stale imported seed-count assertion. The complete passing run above supersedes that earlier partial-suite evidence.

## What has not been established

No full MFFI/Celeb-DF/DF40 inference, GPU training matrix, modern detector reproduction, new trained-checkpoint accuracy, formal robustness, real-image attribution faithfulness, demographic effect, causal mechanism, runtime on the institution's GPU or publication acceptance is claimed. Pretrained RGB initialization paths are reviewed and versioned, but CPU smoke training does not download and train the released pretrained checkpoints.

The primary paper and presentation were compiled in earlier work. This revision verifies their preserved bytes; it does not claim to have regenerated them with new experimental findings. Existing historical result tables remain unchanged and must pass the provenance audit before serving as corrected evidence.

Later edits require their own checks. The PR's current check status should be reviewed together with this commit-specific record; documentation-only updates do not turn the recorded runtime into a measurement of a different run.
