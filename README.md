# Face-forgery detection benchmark

Unified training and evaluation for ResNet, Xception, MobileNet, ViT, CLIP, and DINOv3 across
seven spatial/Fourier inputs and both scratch and fine-tuning regimes. Every matrix run is repeated
with controlled seeds so tables report mean ± standard deviation.

## Setup and environments

Python 3.11–3.12 is required.

```bash
uv sync            # reads uv.lock + the pytorch index configured in pyproject.toml
uv run pytest -q   # expect 50 passed
```

Use `uv sync`, not `pip install -r requirements.txt`. `requirements.txt` is a convenience export
that pins `torch==2.5.1+cu121`, but `uv export` does not emit the PyTorch index URL, so plain pip
cannot resolve it. If you must use pip, pass the index explicitly:

```bash
pip install -r requirements.txt --extra-index-url https://download.pytorch.org/whl/cu121
```

### Check the GPU before running the matrix

The pinned `cu121` build supports up to compute capability 9.0. Blackwell cards (RTX 50xx,
`compute_cap` 12.0) need CUDA 12.8+, so `cu121` fails there with
`no kernel image is available for execution on the device`.

```bash
nvidia-smi --query-gpu=name,compute_cap --format=csv
# compute_cap <= 9.0  -> current cu121 pin is fine
# compute_cap >= 12.0 -> repoint [[tool.uv.index]] in pyproject.toml to cu128 (torch >= 2.9), then `uv lock`
```

### On machines with more than one GPU, prefer `run_matrix.py`

`run_tasks_on_gpus` pins each worker to a single GPU via `CUDA_VISIBLE_DEVICES` and forces
`multi_gpu=False`, so the matrix runs one GPU per process and never builds a `DataParallel`. That
is the path that has actually been exercised.

Running `train.py` directly on a multi-GPU box takes a different route: it inherits
`multi_gpu: true` from `configs/base.yaml` and wraps the model in `DataParallel`, which has not
been run in this project. For a single ad-hoc run there, either set `multi_gpu: false` in the
config or pin the device:

```bash
CUDA_VISIBLE_DEVICES=0 python train.py --config configs/resnet.yaml --fourier none --regime scratch --seed 42
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
python ensemble.py --strategy weighted --subset search --pool best-mode
python make_tables.py
```

Evaluation reconstructs every family from its seeded checkpoint and writes per-split metrics and
predictions. Each run also carries a `results/run_config.json`, written by the trainer, which is
the single source of truth for rebuilding its architecture: re-running evaluation is idempotent
and never rewrites it.

Ensemble candidates average their available seeds. `--pool` sizes the candidate pool: `best-mode`
keeps the best validation mode per family×regime (at most 12), `all` keeps every
family×mode×regime candidate (at most 84). `--subset search` picks the best subset on validation
(exhaustive in parallel up to 12 candidates, greedy incremental beyond), `--subset all` keeps the
whole pool. `--strategy` chooses how members are combined: mean, validation-AUC weighted,
majority, max, geometric, or logistic stacking. `--strategy search` is kept as a historical alias
for `--strategy mean --subset search`. Selection/fitting occurs only on validation; the unchanged
combination is reported on whichever held-out splits were evaluated (`--splits`, default
`val,test,test_d`; held-out splits without outputs are skipped rather than emptying the pool).

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
