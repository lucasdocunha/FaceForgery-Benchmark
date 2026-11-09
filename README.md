# FaceForgery Benchmark — Robustness, Generalization & Explainability

[![Python 3.10+](https://img.shields.io/badge/python-3.10%20%7C%203.11%20%7C%203.12-blue.svg)](https://www.python.org/)
[![PyTorch 2.6+](https://img.shields.io/badge/PyTorch-2.6+-ee4c2c.svg)](https://pytorch.org/)
[![Tests Passing](https://img.shields.io/badge/tests-209%20passed-brightgreen.svg)]()
[![Hardware](https://img.shields.io/badge/GPU-Dual%20RTX%203090%2024GB-76b900.svg)]()
[![License](https://img.shields.io/badge/license-MIT-green.svg)](LICENSE)

Framework unificado de pesquisa para detecção forense de manipulações faciais (*Deepfakes*), avaliando robustez sob degradações realistas, generalização cruzada entre geradores (*out-of-distribution*) e explicabilidade axiomática de decisões neurais.

Desenvolvido no âmbito do Programa de Pós-Graduação em Informática (**PPGIA**) da Pontifícia Universidade Católica do Paraná (**PUCPR**).

---

## 📌 Escopo e Contribuições Científicas

1. **Benchmark de Arquiteturas:** Avaliação de 6 famílias neurais representativas:
   - **CNNs:** ResNet-18, MobileNetV3-Large, Xception.
   - **Vision Transformers:** ViT-B/16 (ImageNet-21k), CLIP ViT-B/16 (OpenAI multimodal).
   - **Auto-Supervisionado:** DINOv3 (ConvNeXt-Tiny).
2. **Representações Frequenciais (2D-FFT):** Análise comparativa entre domínio espacial (RGB puro) e 6 variantes de transformada de Fourier (`magnitude`, `phase`, `complex`, `frequency_3`, `concat`, `concat_frequency`).
3. **Treinamento Robusto Estocástico (`RandomizedRobustAugment`):** Pipeline de aumentos severos (compressão H.264/JPEG, desfoques gaussianos/movimento, ruídos isotrópicos) avaliado em 5 sementes canônicas (42, 123, 2024, 7, 2025).
4. **Generalização Cross-Dataset:**
   - **DeepFake-40 (DF-40):** 40 métodos geradores contemporâneos (Midjourney, SDXL, Flux, SimSwap, etc.).
   - **Celeb-DF v2:** Avaliação temporal agregada por vídeo com **72.60% de Vídeo AUC** no comitê robusto.
5. **Explicabilidade e Interpretabilidade (XAI):** Módulo com 12 métodos (Grad-CAM, Integrated Gradients, Occlusion, etc.) e análise decomposta em anéis de frequência Fourier.
6. **Rigor Estatístico e Proveniência:** Intervalos de confiança de 95% via Cluster-Bootstrap pareado e hashes SHA-256 de auditoria.

---

## 📑 Navegação Rápida do Repositório

### 1. Resultados Oficiais e Tabelas Canônicas ([`results/mostrar_rayson/`](results/mostrar_rayson/))
- [**Tabela 1: Modelos Individuais Finetune**](results/mostrar_rayson/tabela1-resultados-modelos-finetune.md) — 42 combinações (6 modelos $\times$ 7 modos Fourier) com $\mu \pm \sigma$.
- [**Tabela 2: Ensembles Baseline (Clean)**](results/mostrar_rayson/tabela2-resultados-ensemble.md) — Fusões de 2 a 6 modelos sem treino robusto.
- [**Tabela 3: Mixture of Experts (MoE)**](results/mostrar_rayson/tabela3-resultados-moe.md) — MoE Standard vs MoE Frequencial 7C.
- [**Tabela 4: Modelos Robustos RGB (5 Sementes)**](results/mostrar_rayson/tabela4-seedrobusta-rgb.md) — Todas as 5 sementes canônicas concluídas.
- [**Tabela 5: Cross-Dataset no DF-40**](results/mostrar_rayson/tabela5-crossdata-df40.md) — Ranking de 40 geradores modernos.
- [**Tabela 6: Cross-Dataset no Celeb-DF v2**](results/mostrar_rayson/tabela6-crossdata-celebdf.md) — Frame e Vídeo AUC com polaridade corrigida.
- [**Tabela 7: Ensembles de Modelos Robustos**](results/mostrar_rayson/tabela7-ensemble-robusto.md) — Comitês campeões atingindo **72.60% Vídeo AUC** no Celeb-DF.

### 2. Pacote de Publicação e Defesa da Banca
- [**Artigo Principal (IEEE / SIBGRAPI 6 páginas)**](paper/six-page/): Código LaTeX em [`paper/six-page/main.tex`](paper/six-page/main.tex) e PDF compilado em [`paper/six-page/main.pdf`](paper/six-page/main.pdf).
- [**Artigo Estendido com XAI (10 páginas)**](paper/explicability/): Código LaTeX em [`paper/explicability/main.tex`](paper/explicability/main.tex) e PDF compilado em [`paper/explicability/main.pdf`](paper/explicability/main.pdf).
- [**Apresentação da Banca**](presentation/): Slides em HTML interativo ([`presentation/apresentacao.html`](presentation/apresentacao.html)), PPTX e PDF.
- [**Roteiro Narrativo da Apresentação**](presentation/roteiro.md): Roteiro de fala minuto a minuto detalhando o que explicar em cada slide para a banca.

### 3. Documentação Técnica Aprofundada ([`docs/`](docs/))
- [**Detalhamento de Datasets e Benchmarks**](docs/detalhamento_datasets_benchmarks.md): Guia exaustivo de balanceamento, características e taxonomia de geradores do MFFI, Celeb-DF v2 e DF-40.
- [**Análise Técnica dos Modelos**](docs/analise_tecnica_modelos_deepfake.md): Relatório técnico dissecando as 6 arquiteturas.
- [**Auditoria do Dataset DF-40**](docs/df40-audit.md): Protocolos e escopo dos métodos generativos do DF-40.
- [**Guia de Explicabilidade (XAI)**](docs/explicability.md): Manual dos métodos de atribuição e renderização.
- [**Guia do Piloto XAI**](docs/pilot-xai.md): Instruções para geração de coortes determinísticas.
- [**Manual de Robustez**](docs/robustness.md): Guia de execução de pipelines de robustez e calibração.
- [**Auditoria de Proveniência**](research/robustness/AUDIT.md): Diagnóstico e resolução dos bugs históricos de avaliação.

---

## 🚀 Como Executar

### Ambiente e Dependências
```bash
conda activate cae
pip install -r requirements.txt
```

### Execução da Suíte de Testes
```bash
pytest tests/
```
*Garante conformidade de todos os 209 testes unitários e de integração.*

### Treinamento de Modelos
```bash
# Treino padrão
python train.py --config configs/resnet.yaml --fourier none --regime finetune --seed 42

# Treino com robustez estocástica
python train.py --config configs/resnet.yaml --fourier none --regime finetune_robust --robust --seed 42
```

### CLI de Pesquisa Unificada
```bash
# Visualizar comandos de auditoria, calibração e avaliação
python research_cli.py --help

# Avaliação com calibração estrita
python research_cli.py evaluate-legacy --help
```

---

## 🏛️ Estrutura do Repositório

```text
├── configs/                     # Arquivos YAML de configuração de modelos e treinos
├── data/                        # Manifestos de datasets (raw_min, df40, celeb_df)
├── docs/                        # Manuais e documentação técnica aprofundada
├── paper/                       # Manuscritos LaTeX (versões 6 e 10 páginas)
├── presentation/                # Apresentação para a banca (HTML, PPTX, PDF, roteiro)
├── research/                    # Racionais científicos e relatórios de auditoria
├── research_cli.py              # CLI de pesquisa, auditoria e proveniência
├── results/                     # Resultados consolidados e tabelas oficiais (mostrar_rayson/)
├── scripts/                     # Scripts auxiliares de treino, ensemble e avaliação
├── src/                         # Código-fonte principal
│   ├── data/                   # DataLoaders, augmentations e transformada 2D-FFT
│   ├── explicability/          # Pipeline de explicabilidade e atribuição (XAI)
│   ├── models/                 # Construtores de modelos, MoE e regimes
│   ├── pipelines/              # Loops de treino, avaliação e checkpoints
│   └── robustness/             # Engine de auditoria, proveniência e estatística pareada
├── tables/                      # Dados brutos das métricas em formato CSV
├── tests/                       # Suíte completa de 209 testes automatizados
└── train.py                     # Script principal de treinamento
```

---

## Environment Setup

### 1. Requirements

- Python 3.11 or 3.12
- Linux (Ubuntu 20.04+ recommended)
- CUDA 12.1+ (or compatible GPU driver)

### 2. Installation via `uv` (Recommended)

Using [`uv`](https://github.com/astral-sh/uv) ensures exact dependency synchronization and PyTorch CUDA wheel resolution:

```bash
# Clone the repository
git clone https://github.com/lucasdocunha/tcc.git
cd tcc

# Install dependencies and sync virtualenv
uv sync

# Run the complete test suite (50 tests)
uv run pytest -q
```

### 3. Installation via `pip` (Alternative)

If using standard `pip`, pass the PyTorch index URL explicitly:

```bash
pip install -r requirements.txt --extra-index-url https://download.pytorch.org/whl/cu121
```

> [!TIP]
> **GPU Compatibility**: The pinned `cu121` build supports GPUs up to Compute Capability 9.0 (Ada/Hopper/Ampere/Turing). For newer Blackwell architecture GPUs (`compute_cap` $\ge$ 12.0), update the PyTorch index in `pyproject.toml` to `cu128` (PyTorch $\ge$ 2.9) and execute `uv lock`.

---

## Data Preparation & Environment Variables

The pipeline supports both local directories and cluster environments via environment variables:

| Variable | Description | Default |
| :--- | :--- | :--- |
| `TCC_DATASET_ROOT` | Root directory containing image folders (`trainset/`, `valset/`, `testset/`) | `/media/ssd2/lucas.ocunha/datasets/phase1` |
| `TCC_DATA_ROOT` | Root directory containing CSV split manifests (`raw/` and `raw_min/`) | `<repo>/data` |
| `TCC_MODELS_ROOT` | Directory where trained checkpoints and metrics are saved | `<repo>/models` |
| `TCC_OUTPUT_ROOT` | Root directory for outputs (`figures/` and `results/`) | `<repo>` |

You can export these variables in your shell or `.bashrc`:
```bash
export TCC_DATASET_ROOT="/path/to/datasets/phase1"
export TCC_DATA_ROOT="/path/to/tcc/data"
export TCC_MODELS_ROOT="/path/to/tcc/models"
export TCC_OUTPUT_ROOT="/path/to/tcc"
```

---

## Step-by-Step Reproduction Guide

### Step 1: Smoke Testing (Local Verification)

Verify the full pipeline in seconds on CPU or local GPU using a minimal subset:

```bash
# Train a ResNet-18 on raw_min for 1 epoch on 32 images
uv run python train.py --config configs/resnet.yaml --fourier none --regime scratch --seed 42 --epochs 1 --data-limit 32 --raw-min
```

### Step 2: Training Individual Models

Train any architecture with any Fourier mode and regime:

```bash
# Example 1: ResNet-18 with concatenated RGB+FFT Magnitude from scratch
uv run python train.py --config configs/resnet.yaml --fourier concat --regime scratch --seed 42

# Example 2: Pre-trained ViT-B/16 with FFT Magnitude fine-tuning
uv run python train.py --config configs/vit.yaml --fourier magnitude --regime finetune --seed 42

# Example 3: Pre-trained DINOv3 (ConvNeXt) on Spatial RGB
uv run python train.py --config configs/dino.yaml --fourier none --regime finetune --seed 42
```

Trained checkpoints and evaluation artifacts are organized systematically:
```text
models/<family>/<fourier_mode>/<regime>/seed_<seed>/
├── weights/
│   ├── best.pth
│   └── final.pth
├── results/
│   ├── run_config.json
│   ├── metrics_val.csv
│   ├── outputs_val.npz
│   └── predictions_val.csv
└── plots/
    ├── confusion_matrix.png
    └── roc_auc.png
```

### Step 3: Executing the Full Benchmark Matrix

The complete paper benchmark encompasses **126 models** (6 families × 7 Fourier modes × 3 random seeds: `42`, `123`, `2024`). Use `run_matrix.py` to distribute runs across available GPUs:

```bash
# Dry run to inspect the scheduled 126 runs
uv run python run_matrix.py --regime scratch --dry-run

# Run full fine-tuning matrix distributed over GPUs 0, 1, 2, 3
uv run python run_matrix.py --regime finetune --gpus 0,1,2,3 --workers-per-gpu 1

# Run only ResNet and DINOv3 on GPU 0
uv run python run_matrix.py --regime finetune --only resnet,dino --gpus 0
```

### Step 4: Model Evaluation on Clean and Degraded Splits

Evaluate all trained checkpoints across clean validation/test sets and degraded test sets (`test_d`):

```bash
# Standard evaluation on val and test splits
uv run python evaluate.py --splits val,test

# Extended evaluation including degraded in-the-wild partition (test_d)
uv run python evaluate.py \
  --splits val,test,test_d \
  --test-d-csv /path/to/test_d.csv \
  --test-d-images-dir /path/to/test_d/images
```

This aggregates all metrics into `models/all_metrics_by_split.csv`.

### Step 5: Multi-Model Ensembles

Evaluate multi-model fusion and ensemble selection:

```bash
# Parallel exhaustive subset search across the best validation mode per family (at most 12 candidates)
uv run python ensemble.py --strategy search --pool best-mode

# Greedy forward selection across all 84 family x mode combinations
uv run python ensemble.py --strategy search --pool all

# Weighted average by validation AUC
uv run python ensemble.py --strategy weighted --pool best-mode
```

Ensemble results and selected candidate weights are saved to `results/ensemble/`.

### Step 6: Generating Paper LaTeX Tables

Generate publication-ready booktabs LaTeX tables (`results_paper.tex`), formatted Markdown (`results_full.md`), and raw CSVs:

```bash
uv run python make_tables.py
```

Outputs are saved directly to `results/tables/`:
- `results_paper.tex`: Contains formatted `mean ± std` LaTeX tables for each evaluated split (`val`, `test`, `test_d`).
- `results_full.csv`: Full tabular metric summary.
- `results_full.md`: Markdown summary.

### Step 7: Explainability & Heatmap Generation (XAI)

Generate attribution heatmaps to inspect whether models focus on facial artifacts, frequency anomalies, or background cues:

```bash
# Generate single image heatmap (Auto: Grad-CAM for CNNs/DINOv3, Attention Rollout for ViT/CLIP)
uv run python generate_heatmaps.py \
  --checkpoint models/resnet/none/scratch/seed_42/weights/best.pth \
  --image /path/to/sample.jpg

# Generate a grid comparison across multiple images
uv run python generate_heatmaps.py \
  --checkpoint models/vit/none/finetune/seed_42/weights/best.pth \
  --image img1.jpg img2.jpg img3.jpg \
  --grid \
  --output figures/heatmaps/vit_grid.png
```

To reproduce paper Fourier decomposition figures:
```bash
uv run python plot_frequencia.py
uv run python plot_transformacoes.py
uv run python save_fourier_samples.py
```

---

## Summary of Main Findings

1. **Clean vs. Degraded Generalization Gap**:
   - Classical detectors like Xception achieve high ROC-AUC on clean data ($\approx 0.884$) but suffer severe performance collapse under realistic degradation ($\approx 0.582$).
2. **Self-Supervised Representation Superiority**:
   - Pre-trained self-supervised backbones (**DINOv3**) preserve high discriminability under domain shift and compression artifacts compared to purely supervised baselines.
3. **Spectral vs. Spatial Complementarity**:
   - Purely spectral representations (Phase, Magnitude) alone underperform full RGB inputs, but hybrid representations (`concat`: Spatial + FFT Magnitude) provide enhanced robustness when spatial cues are degraded.
4. **Attribution Analysis**:
   - Grad-CAM and Attention Rollout reveal that transformer and self-supervised models maintain attention on facial landmarks, whereas standard CNNs overfit to high-frequency background noise.

---

## Test Suite

The project includes an extensive suite of **50 automated tests** covering data loaders, 2D FFT transformations, first-layer channel adaptation, architecture smoke tests, checkpoint safety, ensemble strategies, table generation, and end-to-end workflows:

```bash
uv run pytest -v
```

---

## Authors & Contributors

*Pontifical Catholic University of Paraná (PUCPR), Curitiba, PR, Brazil*

| Author | Email | GitHub | LinkedIn |
| :--- | :--- | :---: | :---: |
| **Lucas Cunha** | [`lucas.ocunha@ppgia.pucpr.br`](mailto:lucas.ocunha@ppgia.pucpr.br) | [![GitHub](https://img.shields.io/badge/GitHub-lucasdocunha-181717?logo=github)](https://github.com/lucasdocunha) | [![LinkedIn](https://img.shields.io/badge/LinkedIn-lucasdoc-0A66C2?logo=linkedin)](https://www.linkedin.com/in/lucasdoc) |
| **Lucas Sotomaior** | [`lucas.apereira@ppgia.pucpr.br`](mailto:lucas.apereira@ppgia.pucpr.br) | [![GitHub](https://img.shields.io/badge/GitHub-LucasSotomaiorAPereira-181717?logo=github)](https://github.com/LucasSotomaiorAPereira) | [![LinkedIn](https://img.shields.io/badge/LinkedIn-lucassotomaior-0A66C2?logo=linkedin)](https://www.linkedin.com/in/lucassotomaior/) |
| **Lucas Gasperin** | [`lucas.gasperin@pucpr.edu.br`](mailto:lucas.gasperin@pucpr.edu.br) | [![GitHub](https://img.shields.io/badge/GitHub-Lucas--PG-181717?logo=github)](https://github.com/Lucas-PG) | [![LinkedIn](https://img.shields.io/badge/LinkedIn-lucas--gasperin-0A66C2?logo=linkedin)](https://www.linkedin.com/in/lucas-gasperin/) |
| **Beatriz Caldas** | [`beatriz.caldas@pucpr.edu.br`](mailto:beatriz.caldas@pucpr.edu.br) | [![GitHub](https://img.shields.io/badge/GitHub-beatriz--caldas-181717?logo=github)](https://github.com/beatriz-caldas) | [![LinkedIn](https://img.shields.io/badge/LinkedIn-beatrizcaldas01-0A66C2?logo=linkedin)](https://www.linkedin.com/in/beatrizcaldas01/) |
| **Eduardo Pianovski** | [`eduardo.pianovski@pucpr.edu.br`](mailto:eduardo.pianovski@pucpr.edu.br) | [![GitHub](https://img.shields.io/badge/GitHub-dudupnetto-181717?logo=github)](https://github.com/dudupnetto) | [![LinkedIn](https://img.shields.io/badge/LinkedIn-eduardo--pianovski-0A66C2?logo=linkedin)](https://www.linkedin.com/in/eduardo-pianovski-netto-357320269/) |
| **Rayson Laroca** | [`rayson@ppgia.pucpr.br`](mailto:rayson@ppgia.pucpr.br) | [![GitHub](https://img.shields.io/badge/GitHub-raysonlaroca-181717?logo=github)](https://github.com/raysonlaroca) | [![LinkedIn](https://img.shields.io/badge/LinkedIn-raysonlaroca-0A66C2?logo=linkedin)](https://www.linkedin.com/in/raysonlaroca/) |



---

## Citation

If you use this codebase or benchmark in your research, please cite our paper:

```bibtex
@article{cunha2026benchmarking,
  title = {Benchmarking Spatial, Spectral, and Self-Supervised Cues for Face Forgery Detection under Realistic Degradation},
  author = {Lucas {Cunha} and Lucas {Sotomaior} and Lucas {Gasperin} and Beatriz {Caldas} and Eduardo {Pianovski} and Rayson {Laroca}},
  year = {2026},
  journal = {Conference on Graphics, Patterns and Images (SIBGRAPI)},
  volume = {},
  number = {},
  pages = {1-6},
  doi = {},
  issn = {},
}
```

---

## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## Execução no cluster CISIA

Para SLURM/H100, use [o guia CISIA](docs/cisia-hpc.md) e os entrypoints
`scripts/slurm_*.sh`. O guia cobre ambiente CUDA 12.8, caches/checkpoints locais,
publicação versionada, retomada, recuperação de falhas e validações pendentes no
cluster. Não execute treinos ou preparadores de modelos diretamente no Shell Access.
