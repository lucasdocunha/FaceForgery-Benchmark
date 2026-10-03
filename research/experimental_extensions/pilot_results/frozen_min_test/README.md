# Frozen min-test comparison: pilot evidence

These are local pilot results on the same 1,000 MFFI min-test images: 423 real and 577 fake. Canonical suite target `test` means clean **min-test (pilot)**; `test_d` means **min-test-degraded-proxy (pilot)**. The degraded images use the already frozen seed-42, 224-pixel recipe. They are not benchmark Test-D. No DF-40 or Celeb-DF result is produced here.

The four candidates and source-validation Youden thresholds were frozen in `8d01770`. Retry authorization was committed in `39e6961` and explicitly messaged before inference. Actual execution HEADs are recorded per run in `comparison.json`. All candidate, bundle, checkpoint, calibration, recipe and frozen evaluation-code hashes verified. Only the interrupted SBI suite output directory changed; the other three configs remained byte-identical.

| Candidate, seed 42 | Clean AUC [95% CI] | Degraded proxy AUC [95% CI] | Frozen threshold |
|---|---:|---:|---:|
| Generic SBI-only | 0.5341 [0.4997, 0.5670] | 0.5492 [0.5205, 0.5845] | 0.28816831 |
| Generic MFFI control | 0.6300 [0.5968, 0.6618] | 0.5596 [0.5295, 0.5992] | 0.76313090 |
| Generic mixed | 0.6360 [0.6007, 0.6727] | 0.5338 [0.4969, 0.5704] | 0.31827322 |
| Unadapted HF DINO-SRM | 0.9192 [0.9037, 0.9362] | 0.8274 [0.8040, 0.8525] | 0.19193278 |

The paired AUC contrasts below use the generic MFFI arm as the matched backbone, initialization, cohort and training-budget control. The HF DINO checkpoint is a stronger reference with a different training regime, not a matched intervention.

| Candidate minus matched MFFI | Clean delta [95% CI] | Degraded proxy delta [95% CI] |
|---|---:|---:|
| Generic SBI-only | -0.0959 [-0.1422, -0.0508] | -0.0104 [-0.0567, 0.0431] |
| Generic mixed | 0.0060 [-0.0266, 0.0404] | -0.0258 [-0.0632, 0.0140] |

Generic SBI-only is worse on clean min-test. The intervals for its degraded contrast and both mixed-arm contrasts include zero. These fixed one-seed pilots do not support a benefit from generic SBI or mixed training.

| Candidate | Paired degraded-minus-clean AUC [95% CI] | Suite seconds | Sampled peak RSS, MiB |
|---|---:|---:|---:|
| Generic SBI-only | 0.0151 [-0.0147, 0.0471] | 13.51 | 1867.8 |
| Generic MFFI control | -0.0704 [-0.1027, -0.0388] | 12.51 | 1867.5 |
| Generic mixed | -0.1022 [-0.1400, -0.0617] | 12.51 | 1869.6 |
| Unadapted HF DINO-SRM | -0.0918 [-0.1126, -0.0731] | 59.04 | 1896.8 |

All intervals use 300 image-group bootstrap draws, seed 42, 95% percentile intervals and 1,000 supplied image-only groups. They are conditional on the fitted checkpoints; identity/video/source dependence and across-seed uncertainty remain unresolved. Thresholded metrics use the exact frozen source-validation thresholds. EER is a ROC diagnostic and does not change those thresholds.

`comparison.json` retains all thresholded and score metrics, normalized confusion matrices, group intervals, paired contrasts, fresh prediction and artifact hashes, exact commands, runtime/RSS records, and target-access history. `metrics.csv` is the eight-row compact metric export. Raw certified prediction CSVs remain in the local output paths recorded in the JSON.

`aborted_attempt/` preserves the original 5.117-second resource failure verbatim. It emitted no prediction CSV or quality metrics. The interrupted suite had begun clean-target inference, so partial image access is acknowledged without claiming a known image count. `retry_preflight.json` records the output-only config diff and immutable hashes before authorization. Four successful clean/degraded suite accesses followed; offline summary and plot export reuse those certified scores without new inference.

The six DINO figures are exported from those existing certified scores with explicit pilot titles. `plots/dino_min_test_clean/` shows **min-test (pilot)**, and `plots/dino_min_test_degraded_proxy/` shows **min-test-degraded-proxy (pilot)**. Each directory contains `frame_roc.png`, `frame_confusion.png` (row-normalized, true class on rows), `frame_scores.png`, and the small ROC coordinates CSV. Class 1 is fake; the score plot marks the frozen source-validation threshold. The original suite plots and all suite artifacts remain unchanged. `plot_export_record.json` binds every exported plot to its source prediction, metrics and calibration hashes.
