# Face-forgery detection benchmark

Unified training and evaluation for ResNet, Xception, MobileNet, ViT, CLIP, and DINOv3 across
seven spatial/Fourier inputs and both scratch and fine-tuning regimes. Every matrix run is repeated
with controlled seeds so tables report mean ± standard deviation.

## Setup and environments

Python 3.11–3.12 is required.

```bash
uv venv
uv pip install -e .
uv run pytest -q
```

The pipeline uses environment variables instead of machine-specific profiles:

- `TCC_DATASET_ROOT`: image root containing `trainset`, `valset`, and `testset`.
- `TCC_DATA_ROOT`: CSV root containing `raw/` and `raw_min/`.
- `TCC_MODELS_ROOT`: new checkpoint root; defaults to `./models`.
- `TCC_OUTPUT_ROOT`: table, ensemble, and heatmap root; defaults to `./`.

Files such as `.env.slurm`, `.env.server1`, and `.env.server2` are intentionally ignored.

## Configuration and training

Shared defaults are in `configs/base.yaml`; each `configs/<family>.yaml` supplies family overrides.
CLI values override both. Set `raw_min: true` for local smoke runs.

```bash
python train.py --config configs/resnet.yaml --fourier concat --regime scratch --seed 42
python train.py --config configs/resnet.yaml --fourier none --regime scratch --seed 42 \
  --epochs 1 --data-limit 32 --raw-min
python run_matrix.py --regime finetune --only resnet,vit --gpus 0,1 --workers-per-gpu 2
python run_matrix.py --regime scratch --dry-run
```

The matrix is six families × seven modes × three configured seeds. Fine-tuning supports all channel
counts (1, 2, 3, 4, and 6) through shared first-layer adaptation. Spectral modes disable spatial
augmentation to keep Fourier/spatial alignment. Training uses class-balanced sampling, AMP,
mixup/cutmix, gradient clipping, ReduceLROnPlateau, validation-selected thresholds, and early stop.

Outputs follow one layout:

```text
models/<family>/<mode>/<regime>/seed_<N>/
  weights/{best,final}.pth
  results/{metrics,outputs,predictions}_{val,test}.(csv|npz)
  plots/{confusion_matrix,roc_auc}.png
```

Model sources are torchvision for ResNet/MobileNet, torch for scratch Xception, timm for pretrained
Xception and DINOv3, and Hugging Face `ViTModel`/`CLIPVisionModel` for both transformer regimes.

## Evaluation, ensembles, and tables

```bash
python evaluate.py --splits val,test --only-model-family resnet
python evaluate.py --splits val,test,test_d --test-d-csv /path/test_d.csv \
  --test-d-images-dir /path/test_d/images
python ensemble.py --strategy search --pool best-mode
python ensemble.py --strategy search --pool all
python make_tables.py
```

Evaluation reconstructs every family from its seeded checkpoint and writes per-split metrics and
predictions. Ensemble candidates average their available seeds. `best-mode` keeps the best
validation mode per family×regime (at most 12) and searches subsets exhaustively in parallel;
`all` keeps family×mode×regime candidates (at most 84) and uses greedy incremental search. Direct
strategies are mean, validation-AUC weighted, majority, max, geometric, and logistic stacking.
Selection/fitting occurs only on validation; the unchanged combination is reported on Test and
Test-Hard.

`make_tables.py` writes `tables/results_full.csv`, `results_full.md`, and a booktabs
`results_paper.tex` containing one mean±standard-deviation table per split.

## Heatmaps

```bash
python generate_heatmaps.py --checkpoint models/.../weights/best.pth --image image.jpg
python generate_heatmaps.py --checkpoint models/.../weights/best.pth \
  --image one.jpg two.jpg three.jpg --grid
```

`auto` uses layer-hook Grad-CAM for CNNs and residual Attention Rollout for ViT/CLIP. The selected
DINOv3 implementation is ConvNeXt-based and therefore has no attention matrix; it transparently
uses Grad-CAM. Fourier checkpoints receive the same input encoding used during training.

## Migration and tests

Pre-refactor outputs were moved to ignored `models_legacy_backup/`; new discovery never reads them.
The duplicated training, evaluation, ensemble, and heatmap scripts were removed. Tests use
`data/raw_min` CSVs plus generated fixture images and cover all families/channel counts, training,
discovery/evaluation, both ensemble pools, tables, plots, heatmaps, and an end-to-end CPU workflow.
The full production matrix remains GPU work outside the local test suite.
