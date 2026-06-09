# Comprehensive Model Evaluation Guide

## Overview

This guide explains how to evaluate all trained models across all Fourier modes on multiple test datasets and generate unified metrics.

## Scripts

### 1. `test_evaluation_setup.py` (Pre-flight checks)
Validates environment setup before running full evaluation.

**Run:**
```bash
python test_evaluation_setup.py
```

**What it checks:**
- ✓ All imports work (ImageDataset, model factories, evaluation functions)
- ✓ Datasets exist and are readable (raw/test, raw_min/test, raw/val)
- ✓ Model discovery structure is correct

**Expected output:**
```
[Phase 1] Testing imports...
  ✓ ImageDataset, FourierMode
  ✓ phase1_split_root
  ✓ evaluate_classifier
  ✓ create_resnet_model

[Phase 2] Testing dataset loading...
  ✓ test_raw        →  1234 images
  ✓ test_raw_min    →   567 images
  ✓ val_raw         →   890 images

[Phase 3] Testing model discovery...
  Checking: models/resnet
  ✓ resnet       none                 → best_resnet.pth
  ✓ resnet       magnitude            → best_resnet.pth
  ...
```

---

### 2. `evaluate_all_models.py` (Full evaluation)
Loads all trained models and evaluates them on all test datasets.

**Run:**
```bash
python evaluate_all_models.py
```

**What it does:**

1. **Phase 1: Model Discovery** — Scans directory structure:
   ```
   models/
   ├── resnet/{none,magnitude,phase,complex,concat,frequency_3,concat_frequency}/weights/
   ├── mobilenet/mobilenetv3_large/{...}/weights/
   ├── xception/{...}/weights/
   ├── vit/vit_scratch/{...}/weights/
   └── clip/{...}/weights/
   ```

2. **Phase 2-3: Dataset Loading & Model Evaluation** — For each model:
   - Load datasets (3 datasets × appropriate channels for Fourier mode)
   - Run inference with `evaluate_classifier`
   - Collect metrics: accuracy, precision, recall, F1, AUC, specificity, confusion matrix
   - Free GPU memory after each model

3. **Phase 4: Aggregation** — Combine results into single CSV

**Output:**
- `models/all_models_metrics_unified.csv` — Main results file
- `evaluation.log` — Detailed execution log

**CSV Columns:**
```
architecture       | Arch name (resnet, mobilenet, xception, vit, clip)
fourier_mode       | Fourier mode (none, magnitude, phase, concat, ...)
model_variant      | Full name (e.g., resnet_magnitude)
dataset            | Test dataset (test_raw, test_raw_min, val_raw)
accuracy           | Classification accuracy
precision          | Precision score
recall             | Recall / Sensitivity
f1                 | F1-score
auc                | ROC-AUC
specificity        | TN / (TN + FP)
sensitivity        | Same as recall
loss               | BCE loss on dataset
tp, fp, fn, tn     | Confusion matrix values
optimal_threshold  | Operating threshold (currently 0.5)
```

---

## Expected Results

### Model Inventory
When all models are trained, you should discover:

- **ResNet**: 7 Fourier modes × 1 default architecture = 7 variants
- **MobileNet**: 7 Fourier modes × 1 size (large) = 7 variants
- **Xception**: 7 Fourier modes = 7 variants
- **ViT**: 1 mode (none only) = 1 variant (conditional in original)
- **CLIP**: 1 mode (none only) = 1 variant (conditional in original)

**Total**: ~30 model variants (exact number depends on what was trained)

### Evaluation Matrix
```
Models × Datasets = Total Evaluations
~30 × 3 = ~90 evaluations
```

---

## File Structure (on Server)

```
/home/lucas/tcc/
├── models/
│   ├── resnet/
│   │   ├── none/
│   │   │   ├── weights/
│   │   │   │   ├── best_resnet.pth
│   │   │   │   └── resnet.pth
│   │   │   ├── results/
│   │   │   │   ├── metrics_summary.csv
│   │   │   │   └── predictions.csv
│   │   │   └── plots/
│   │   ├── magnitude/
│   │   │   ├── weights/...
│   │   │   ├── results/...
│   │   │   └── plots/...
│   │   └── ... (other Fourier modes)
│   ├── mobilenet/mobilenetv3_large/{same structure}
│   ├── xception/{same structure}
│   ├── vit/vit_scratch/
│   │   └── none/{same structure}
│   ├── clip/{same structure}
│   └── all_models_metrics_unified.csv ← OUTPUT FILE
├── data/
│   ├── raw/
│   │   ├── test.csv
│   │   ├── val.csv
│   │   └── train.csv
│   └── raw_min/
│       ├── test.csv
│       ├── val.csv
│       └── train.csv
├── evaluate_all_models.py ← RUN THIS
├── test_evaluation_setup.py ← RUN THIS FIRST
└── evaluation.log ← Created after running
```

---

## Usage Workflow

### On Local Machine (Preparation)
```bash
# Both scripts are already created and compiled
# They are ready to be deployed to the server
```

### On Server (Execution)

```bash
# Step 1: Verify environment
python test_evaluation_setup.py

# Expected output shows model count and dataset availability
# If all ✓, proceed to step 2

# Step 2: Run full evaluation
python evaluate_all_models.py

# Stdout shows progress with tqdm bar
# evaluation.log records all details
# Final CSV saved to: models/all_models_metrics_unified.csv
```

---

## Monitoring Execution

### Real-time Progress
The script shows:
```
Total evaluations: |████████░░░░| 67/90 [2:45<1:12, 2.15it/s]
```

### Example Log Output
```
2026-06-02 14:30:15 - INFO - [Phase 1] Discovering models...
2026-06-02 14:30:16 - INFO - Discovered 30 trained models across 5 architectures
2026-06-02 14:30:17 - INFO - [Phase 2-3] Loading datasets and evaluating models...
2026-06-02 14:31:02 - INFO - Loading model: resnet_none
2026-06-02 14:31:15 - INFO - ✓ resnet_none               on test_raw        : Acc=0.9234, F1=0.9156, AUC=0.9812
2026-06-02 14:31:28 - INFO - ✓ resnet_none               on test_raw_min    : Acc=0.9167, F1=0.9089, AUC=0.9756
2026-06-02 14:31:41 - INFO - ✓ resnet_none               on val_raw         : Acc=0.9401, F1=0.9312, AUC=0.9891
...
```

### GPU Monitoring
The script automatically manages GPU memory:
- Loads one model at a time
- Evaluates on all 3 datasets
- Clears GPU cache between models

---

## Output Analysis

### Quick Stats
The script prints summary at end:
```
[Phase 4] Aggregating results...

================================================================================
EVALUATION SUMMARY
================================================================================

Total results: 90
Architectures: 5
Fourier modes: 7
Test datasets: 3

Best models (highest F1) per dataset:
  test_raw: xception_concat (F1=0.9512, AUC=0.9889)
  test_raw_min: resnet_magnitude (F1=0.9401, AUC=0.9834)
  val_raw: mobilenet_frequency_3 (F1=0.9687, AUC=0.9923)

Average metrics by architecture:
               accuracy        f1       auc
architecture
clip              0.9123    0.9045   0.9634
mobilenet         0.9234    0.9167   0.9712
resnet            0.9189    0.9101   0.9689
vit               0.9267    0.9201   0.9756
xception          0.9301    0.9234   0.9798

================================================================================
✓ Results saved to: models/all_models_metrics_unified.csv
```

### Analyzing CSV Results
```python
import pandas as pd

df = pd.read_csv("models/all_models_metrics_unified.csv")

# Best models overall
print(df.nlargest(5, 'f1')[['model_variant', 'dataset', 'f1', 'auc']])

# Average by architecture
print(df.groupby('architecture')[['accuracy', 'f1', 'auc']].mean())

# Best model per dataset
for dataset in df['dataset'].unique():
    subset = df[df['dataset'] == dataset]
    best = subset.loc[subset['f1'].idxmax()]
    print(f"{dataset}: {best['model_variant']} (F1={best['f1']:.4f})")
```

---

## Troubleshooting

### Issue: "No models discovered"
**Cause:** Models directory structure doesn't match expected format

**Fix:** Check that models exist at:
```
models/
├── resnet/{fourier_mode}/weights/{best_}resnet.pth
├── mobilenet/mobilenetv3_large/{fourier_mode}/weights/best_mobilenetv3_large.pth
├── xception/{fourier_mode}/weights/best_xception.pth
├── vit/vit_scratch/none/weights/best_vit.pth
└── clip/none/weights/best_clip.pth
```

### Issue: "Dataset CSV not found"
**Cause:** Missing data files

**Fix:** Verify:
```bash
ls data/raw/test.csv data/raw/val.csv
ls data/raw_min/test.csv
```

### Issue: GPU out of memory
**Cause:** Large models or batch size too high

**Fix:** Script uses batch_size=32; if OOM occurs:
1. Reduce `batch_size` in `load_test_datasets_for_fourier()` (line ~150)
2. Or use CPU: add `--device cpu` (if implemented)

### Issue: Slow evaluation
**Cause:** num_workers or large images

**Fix:** Adjust in `load_test_datasets_for_fourier()`:
```python
loader = DataLoader(
    dataset,
    batch_size=32,
    num_workers=2,  # Reduce from 4 if slower
    ...
)
```

---

## Performance Expectations

### Timing (approximate)
- Model discovery: < 1 second
- Per-model inference (3 datasets): 15-30 seconds
- Total for 30 models: 10-15 minutes
- (Depends on GPU, image size, batch_size)

### GPU Memory (approximate)
- Single model: 2-4 GB
- Script manages cleanup between models
- No accumulation expected

### Output File Size
- CSV with 90 rows: ~20-30 KB
- Typical metrics per row: 15 columns

---

## Next Steps

1. **Deploy** both `.py` scripts to server
2. **Verify** directory structure matches expected format
3. **Run** `python test_evaluation_setup.py` to validate
4. **Execute** `python evaluate_all_models.py`
5. **Analyze** results in `models/all_models_metrics_unified.csv`
6. **Optionally:** Create plots, comparison tables, ensemble strategies based on results

---

## References

- [evaluation.py](src/pipelines/evaluation.py) — Core metrics calculation
- [ImageDataset](src/data/data.py) — Data loading with Fourier modes
- Model creation functions in `src/models/`
- Training pipelines in `src/pipelines/`
