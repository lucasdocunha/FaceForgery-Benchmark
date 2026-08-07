# Face-forgery detection benchmark

Unified reproducible training and evaluation for ResNet, Xception, MobileNet, ViT, CLIP and
DINOv3 across seven spatial/Fourier representations and scratch or fine-tuning regimes. Runs use
configured seeds so results can be reported as mean ± standard deviation.

## Setup

Python 3.11–3.12 is required. Install and validate with `uv venv`, `uv pip install -e .`, and
`uv run pytest -q`.

Configure environments through `TCC_DATASET_ROOT` (images), `TCC_DATA_ROOT` (CSVs),
`TCC_MODELS_ROOT` (checkpoints), and `TCC_OUTPUT_ROOT` (tables/heatmaps). The output roots default
to `./models` and `./`.

## Training

Defaults live in `configs/base.yaml`; family overrides are in `configs/<family>.yaml`.

```bash
uv run python train.py --config configs/resnet.yaml --fourier concat --regime scratch --seed 42
uv run python run_matrix.py --regime finetune --only resnet,vit --gpus 0,1
uv run python run_matrix.py --regime scratch --dry-run
```

All seven modes (`none`, `magnitude`, `phase`, `complex`, `concat`, `frequency_3`, and
`concat_frequency`) work in both regimes. Outputs use:

```text
models/<family>/<mode>/<regime>/seed_<N>/{weights,results,plots}/
```

ResNet/MobileNet use torchvision, scratch Xception uses torch and pretrained Xception uses timm,
ViT/CLIP use Transformers in both regimes, and DINOv3 uses timm.

## Evaluation and reporting

```bash
uv run python evaluate.py --splits val,test --only-model-family resnet
uv run python ensemble.py --strategy search --pool best-mode
uv run python make_tables.py
uv run python generate_heatmaps.py --checkpoint models/.../weights/best.pth --image image.jpg
```

Evaluation writes per-run metrics and predictions. Ensemble subsets are selected on validation and
applied unchanged to Test/Test-Hard. Tables are exported as CSV, Markdown, and booktabs LaTeX and
aggregate seeds. `--method auto` selects CNN Grad-CAM-style or transformer attention attribution.

Pre-refactor checkpoints are outside new discovery under `models_legacy_backup/`; the new matrix
must be retrained. Tests use the small `data/raw_min` fixtures. Full production training requires
the target GPU environment and is intentionally outside the local suite.
