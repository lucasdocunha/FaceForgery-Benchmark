# FaceForgery Benchmark — TCC

**Além do Conjunto Limpo: Resíduos de Ruído, Treinamento Robusto e Generalização entre Bases na Detecção de Faces Manipuladas**

Código do TCC (PUCPR) que estende o benchmark do artigo SIBGRAPI 2026
(*Benchmarking Spatial, Spectral, and Self-Supervised Cues for Face Forgery Detection under Realistic Degradation*).
O artigo do TCC está em [`paper/TCC_versao_final.tex`](paper/TCC_versao_final.tex).

O trabalho responde a três perguntas:

1. **Treino robusto** — degradar as imagens de treino (rotação, contraste, brilho, nitidez, desfoque, JPEG, ruído) reduz a queda entre o teste limpo e o teste difícil do MFFI?
2. **Representações de entrada** — RGB, 6 variantes de Fourier e resíduos de ruído **SRM** (RGB+SRM e SRM-only): qual generaliza melhor?
3. **Generalização entre bases e ensembles** — modelos treinados só no MFFI funcionam no **DF-40** e no **Celeb-DF v2**? Ensembles ajudam?

---

## Como o código funciona

```text
manifesto CSV ─► ImageDataset ─► transformação ─► modelo ─► Trainer ─► models/<família>/<repr>/<regime>/seed_<s>/
 (data/*.csv)    src/data/data.py   RGB / FFT / SRM   src/models/   src/pipelines/      weights/best.pth
                                    + aumento robusto                                   results/outputs_*.npz, metrics_*.csv
                                         │
          avaliação (MFFI test / test_d, DF-40, Celeb-DF) ─► ensembles ─► tabelas (tables/) e figuras (figures/)
```

| Peça | Onde | O que faz |
| :--- | :--- | :--- |
| Dados | `src/data/data.py`, `src/data/paths.py` | Lê os manifestos CSV, aplica a representação de entrada e resolve onde estão imagens e modelos. |
| Representações | `src/data/data.py`, `src/forensics/srm.py` | `--fourier`: `none` (RGB), `magnitude`, `phase`, `complex`, `frequency_3`, `concat` (RGB+Mag), `concat_frequency` (RGB+Freq, 7 canais), `srm` (RGB+SRM, 6 canais), `srm_only` (3 canais). |
| Treino robusto | `src/data/augmentations.py` (`RandomizedRobustAugment`) | Ativado por `--regime finetune_robust` ou `--robust`. Degradações e probabilidades da Tabela de aumentos do TCC. |
| Modelos | `src/models/` + `configs/*.yaml` | Xception, ResNet-18, MobileNetV3-L, ViT-B/16, CLIP ViT-B/16, DINOv3 ConvNeXt-B. A primeira camada é adaptada ao número de canais. |
| Treino/avaliação | `src/pipelines/training.py`, `evaluation.py`, `checkpoints.py` | AdamW, amostragem balanceada, parada antecipada; limiar escolhido na validação; salva `best.pth`, predições e métricas. |
| Ensembles | `src/pipelines/ensemble.py`, `scripts/calculate_4groups_top2_top4_metrics.py` | Fusão (média, pesos, stacking) com membros escolhidos na validação do MFFI. |
| CLI de auditoria | `research_cli.py`, `src/robustness/` | Conversão/auditoria de manifestos, calibração, avaliação com proveniência (SHA-256), bootstrap pareado. **Não é necessário para reproduzir as tabelas do TCC.** |

Sementes canônicas: **42, 123, 2024, 7, 2025** (o protocolo 2 do DF-40 usa 42, 123 e 2024).

---

## 1. Instalação

Requisitos: Linux, Python 3.11/3.12, GPU NVIDIA com CUDA 12.1+.

```bash
git clone https://github.com/lucasdocunha/FaceForgery-Benchmark.git
cd FaceForgery-Benchmark
git checkout tcc
uv sync                      # cria .venv com torch 2.5.1+cu121
uv run pytest -q             # ~430 testes; os de XAI são pulados sem captum

# opcional (mapas de explicabilidade): uv pip install -r requirements-explicability.txt
```

Sem `uv`: `pip install -r requirements.txt --extra-index-url https://download.pytorch.org/whl/cu121`.

## 2. Dados e caminhos

As imagens **não** estão no repositório. Os caminhos têm como padrão o servidor do laboratório e podem ser trocados por variáveis de ambiente (`src/data/paths.py`):

| Variável | Conteúdo | Padrão |
| :--- | :--- | :--- |
| `TCC_DATASET_ROOT` | MFFI: `trainset/`, `valset/`, `testset/`, `test_d/` | `/media/ssd2/lucas.ocunha/datasets/phase1` |
| `TCC_TEST_D_ROOT` | Imagens do teste difícil (`test_d`) | `<phase1>/test_d` |
| `TCC_DF40_ROOT` | DF-40 extraído (com `paths_cache.json`) | `/media/ssd2/lucas.ocunha/datasets/df40_extracted` |
| `TCC_CELEB_DF_ROOT` | Recortes de face do Celeb-DF v2 | `/media/ssd2/lucas.ocunha/datasets/celeb_df_crops` |
| `TCC_DATA_ROOT` | Manifestos CSV | `<repo>/data` |
| `TCC_MODELS_ROOT` | Checkpoints e predições | `/media/ssd2/lucas.ocunha/models-tcc`, senão `<repo>/models` |

Manifestos: `data/raw/` (MFFI) e `data/raw_min/` (subconjunto para testes) estão versionados. Os do DF-40 e do Celeb-DF são gerados uma vez:

```bash
uv run python scripts/unpack_df40.py                       # extrai o DF-40 em $TCC_DF40_ROOT
uv run python scripts/prepare_df40_benchmark.py            # -> data/df40/test.csv (protocolo 1, balanceado)
uv run python scripts/prepare_celeb_df.py \
    --dataset-root /caminho/Celeb-DF-v2 --output-dir "$TCC_CELEB_DF_ROOT" \
    --manifest-out data/celeb_df/test.csv --frames-per-video 15
```

## 3. Treino

Um modelo:

```bash
# teste rápido (CPU serve): 1 época, 32 imagens
uv run python train.py --config configs/resnet.yaml --fourier none --regime finetune --seed 42 --epochs 1 --data-limit 32 --raw-min

# finetuning convencional
uv run python train.py --config configs/dino.yaml --fourier none --regime finetune --seed 42
# finetuning robusto (c/DA) com RGB+SRM
uv run python train.py --config configs/dino.yaml --fourier srm --regime finetune_robust --seed 42
```

Campanhas completas usadas no TCC (2 GPUs):

```bash
# finetuning convencional: 6 famílias × representações × sementes
uv run python run_matrix.py --regime finetune --gpus 0,1 --workers-per-gpu 1

# robusto RGB, 5 sementes (GPU 0: CLIP, DINOv3, ResNet | GPU 1: ViT, MobileNet, Xception)
uv run python scripts/launch_robust_5seeds.py --gpu-id 0 &
uv run python scripts/launch_robust_5seeds.py --gpu-id 1 &

# robusto com SRM, 5 sementes; já avalia em test, test_d, DF-40 e Celeb-DF
uv run python scripts/run_forensics_campaign.py --gpu 0 --mode srm &
uv run python scripts/run_forensics_campaign.py --gpu 1 --mode srm_only &
```

Cada execução grava em `models/<família>/<repr>/<regime>/seed_<s>/` os pesos (`weights/best.pth`), a configuração (`results/run_config.json`), as métricas e as predições por split (`results/metrics_*.csv`, `outputs_*.npz`).

## 4. Avaliação

```bash
# MFFI: teste limpo e teste difícil, para todos os checkpoints -> models/all_metrics_by_split.csv
uv run python evaluate.py --splits val,test,test_d --test-d-csv /caminho/test_d.csv --test-d-images-dir "$TCC_TEST_D_ROOT"

# DF-40 protocolo 1 (AUC, balanceado): predições em cache para os ensembles
uv run python scripts/extract_df40_predictions.py --modes none,srm,srm_only,concat_frequency --device cuda:0

# DF-40 protocolo 2 (revocação nas 650.549 falsas, por técnica): uma passada por representação
uv run python scripts/run_df40_grouped_seeds.py --worker-id 0 --device cuda:0
uv run python scripts/run_df40_grouped_seeds.py --worker-id 1 --device cuda:1
uv run python scripts/build_df40_seeds_summary.py

# Celeb-DF v2 (AUC por vídeo)
uv run python run_all_celeb_df.py
```

## 5. Tabelas e figuras do TCC

| No artigo | Comando | Saída |
| :--- | :--- | :--- |
| Tab. MFFI (finetuning × robusto) e Tab. famílias × representações | `uv run python make_tables.py` e `uv run python scripts/update_tables_3seeds.py` | `results/tables/results_paper.tex`, `tables/` |
| Tab. ensembles Top-2 a Top-4 | `uv run python scripts/calculate_4groups_top2_top4_metrics.py` → `scripts/generate_4groups_tex.py` | `tables/benchmark_4groups_top2_top4.{csv,md,tex}` |
| Tab. revocação por técnica (DF-40 P2) | `uv run python scripts/export_tabela6_650k.py` | `tables/tabela6_top3_e_ensemble_mffi_val.*` |
| Ensembles SRM e recorte DF-40 por paradigma/técnica | `bash scripts/reproduce_all_results.sh --ensembles` / `--df40-breakdown` | `tables/ensembles_srm_5seeds_summary.csv`, `tables/df40_srm_breakdown_*.csv` |
| Fig. amostras das bases | `uv run python scripts/export_dataset_sample_images.py` e `scripts/export_mffi_degraded_samples.py` | `figures/amostras/` |
| Fig. resíduos SRM / transformações | `uv run python plot_transformacoes.py` | `figures/srm/`, `figures/fourier/` |
| Heatmaps (Grad-CAM / Attention Rollout) | `bash scripts/reproduce_all_results.sh --heatmaps` | `figures/heatmap_clip_dino_fusion_*.png` |

> A versão final da tabela por técnica com o DINOv3 RGB+SRM (`tables/tabela6_top3_e_dino_srm.*`) foi montada à mão a partir de `tables/df40_all_fakes_models/`; não há script que a gere.

Atalho para tudo que vem **depois** do treino (exige os modelos em `$TCC_MODELS_ROOT`):

```bash
bash scripts/reproduce_all_results.sh --all            # ensembles + DF-40 por categoria + heatmaps
bash scripts/reproduce_all_results.sh --full-pipeline  # inclui o retreino das 5 sementes (dias de GPU)
bash scripts/reproduce_all_results.sh --help
```

---

## Estrutura

```text
train.py, evaluate.py, run_matrix.py, ensemble.py   # entradas principais (treino, avaliação, matriz, ensembles)
make_tables.py, make_df40_tables.py, make_celeb_df_tables.py, plot_*.py
research_cli.py         # CLI de auditoria/calibração (src/robustness)
configs/                # YAML por família (+ experimental/ e research/)
data/                   # manifestos CSV
src/data/               # dataset, Fourier, aumento robusto, caminhos
src/forensics/          # filtros SRM (e DTCWT)
src/models/             # arquiteturas e adaptação de canais
src/pipelines/          # treino, avaliação, checkpoints, ensembles
src/robustness/         # proveniência, calibração, estatística pareada
src/explicability/      # métodos de XAI
scripts/                # campanhas, DF-40, Celeb-DF, ensembles, tabelas, jobs SLURM (slurm_*.sh)
tables/, results/, figures/   # resultados versionados
paper/                  # LaTeX do TCC e dos artigos
docs/                   # guias (datasets, CISIA, robustez, XAI)
research/, src/experimental/, configs/experimental/  # extensões fora do TCC (MoE, SBI, VLM, grafos)
```

## Cluster CISIA (SLURM)

Use os `scripts/slurm_*.sh` e siga [docs/cisia-hpc.md](docs/cisia-hpc.md). Não rode treinos diretamente no Shell Access.

## Documentação adicional

- [Bases e protocolos](docs/detalhamento_datasets_benchmarks.md) · [Auditoria do DF-40](docs/df40-audit.md)
- [Robustez e calibração (`research_cli.py`)](docs/robustness.md) · [Explicabilidade](docs/explicability.md)
- Registro de experimentos no Linear: [AGENTS.md](AGENTS.md)

## Autores

Lucas Cunha, Lucas Sotomaior, Lucas Gasperin, Beatriz Caldas, Eduardo Pianovski e Rayson Laroca (orientador) — PUCPR.

## Citação (artigo base)

```bibtex
@article{cunha2026benchmarking,
  title   = {Benchmarking Spatial, Spectral, and Self-Supervised Cues for Face Forgery Detection under Realistic Degradation},
  author  = {Lucas {Cunha} and Lucas {Sotomaior} and Lucas {Gasperin} and Beatriz {Caldas} and Eduardo {Pianovski} and Rayson {Laroca}},
  year    = {2026},
  journal = {Conference on Graphics, Patterns and Images (SIBGRAPI)},
}
```
