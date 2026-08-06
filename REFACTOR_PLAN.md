# Refatoração geral do pipeline de detecção de manipulação (TCC)

## Contexto

O código hoje cobre 6 famílias de modelo (ResNet18, Xception, MobileNet, ViT, CLIP, DINOv3) × 7 modos de imagem (RGB, log-magnitude, phase, complex spectrum, rgb+mag, high-pass, rgb+freq) × 2 regimes (from-scratch, fine-tuning), avaliados em 2 splits de teste (normal/Test e difícil/Test-Hard). Isso cresceu organicamente: cada família de modelo tem seu próprio arquivo em `src/pipelines/` com loop de época, otimizador, scheduler, sampler e early-stopping reimplementados à mão (~90% duplicado entre eles); `evaluate_trained.py` só reconstrói ViT (`SUPPORTED_FAMILIES = {"vit"}`) apesar de ter branches mortos para as outras 5; `evaluate_all_models.py` está quebrado (chama funções que não existem mais); há 4 scripts de ensemble parcialmente redundantes; não existe controle de seed nem agregação estatística entre execuções; os plots não têm paleta/fonte consistente entre si; e tudo roda em 3 ambientes diferentes (1 cluster Slurm + 2 servidores com interface gráfica), com acesso a GPU variável (alguns multi-GPU, outros não).

Objetivo: um pipeline único, fácil de manter, onde treinar-do-zero, fine-tuning, avaliação, ensemble, tabelas de resultado e heatmaps/Grad-CAM seguem exatamente o mesmo padrão de configuração e descoberta de checkpoints — com repetição controlada por seed (para reportar média±desvio), caminhos de dados/modelos/saída resolvidos dinamicamente por ambiente, e todo código vindo de `torch`/`transformers` (com `timm` só onde não há alternativa nativa: DINOv3 e Xception pretrained).

## Decisões já fechadas com o usuário (não renegociar sem necessidade)

1. **Fontes de modelo**: ResNet/MobileNet via `torchvision.models`. ViT/CLIP via classes do `transformers` (`ViTModel`/`CLIPVisionModel`) tanto para scratch (peso aleatório, `ViTConfig(...)`/`CLIPVisionConfig(...)`) quanto para fine-tuning (`from_pretrained`) — descarta o Transformer artesanal atual. Xception: implementação manual em torch para scratch; `timm` só na variante pretrained. DINOv3: `timm` (repo HF é gated).
2. **Fine-tuning em todos os 7 modos**: adapta a 1ª camada/patch-embedding para N canais (replicando o padrão já usado em `src/models/resnet.py` na adaptação de `conv1` e `src/models/mobilenet.py::_adapt_first_conv`) — sem restringir fine-tuning a RGB.
3. **Arquitetura de treino**: um `Trainer` genérico único; cada família expõe só `build(config)`, `freeze_backbone(model)`, `unfreeze_for_finetune(model, n)`.
4. **Config**: YAML (`configs/base.yaml` + overrides por experimento), carregado via CLI.
5. **Entrypoints**: `train.py --config <yaml>` roda 1 run; `run_matrix.py --regime scratch|finetune` expande a matriz e distribui via `run_tasks_on_gpus` (já existe em `src/utils/multiprocess.py`, reutilizar sem alterar a assinatura).
6. **Ensemble**: `ensemble.py` com estratégias mean/weighted/majority/max/geometric/stacking (regressão logística) **e** busca do melhor subconjunto (greedy quando exaustivo for inviável). Pool de candidatos configurável: `--pool best-mode` (12: melhor modo por modelo×regime) ou `--pool all` (até 84). Busca no **val**, relatório final em **test** e **test_d**.
7. **Seeds**: `seeds: [42, 123, 2024]` no config; todo run_matrix multiplica a matriz por elas; layout de pasta ganha nível `seed_<N>/`; agregação em média±desvio.
8. **Layout de checkpoint unificado**: `models/<família>/<modo>/<regime>/seed_<N>/{weights,results,plots}/`.
9. **Checkpoints existentes**: `models/` atual → `models_legacy_backup/` (fora do pipeline novo); tudo é retreinado do zero.
10. **Scripts obsoletos removidos** (sem pasta legacy): `main.py`, `fine_tuning.py`, `evaluate_all_models.py`, `evaluate_trained_models.py`, `fusion.py`, `strong_fusion.py`, `_run_fusion.py`, e os 4 scripts de heatmap da raiz (substituídos por `generate_heatmaps.py`).
11. **Tabelas**: `make_tables.py` gera CSV/Markdown + LaTeX (booktabs), agregando por seed.
12. **Plots**: `src/plots/style.py` central (paleta fixa por modelo, fonte, `rcParams`), usado por todo gerador de gráfico.
13. **Heatmap/Grad-CAM**: `generate_heatmaps.py` único via registry, Grad-CAM (CNNs) + Attention Rollout (Transformers) + modo grid.
14. **Testes**: `pytest`, sempre com `data/raw_min` (reaproveitar fixtures de `tests/conftest.py`: `tiny_phase1_dataset`/`tiny_short_split_dataset`).
15. **Multi-ambiente (1 Slurm + 2 servidores, com/sem multi-GPU)**: estende o padrão de env vars já usado em `src/data/paths.py` (`TCC_DATASET_ROOT`, `TCC_DATA_ROOT`) em vez de criar um sistema de "perfil de ambiente" novo. GPUs seguem via `--gpus`/auto-detect de `run_tasks_on_gpus` (já cai para CPU se não achar CUDA).

## O que se reaproveita sem reescrever

- `src/pipelines/evaluation.py` (`evaluate_classifier`, `binary_metrics`, `best_threshold`, `safe_auc`, sanitização de logits) — já é puro/genérico, fica como está.
- `src/data/data.py` (`ImageDataset`, `FourierMode`) — só muda `ALL_FOURIER_MODES` para conter os 7 modos ativos (hoje só tem `"concat_frequency"`).
- `src/data/paths.py` (`phase1_split_root`, `data_root`) — só ganha as 2 funções novas descritas abaixo (`models_root()`, `output_root()`), no mesmo padrão de env var.
- `src/utils/multiprocess.py::run_tasks_on_gpus` — inalterado, é o motor de `run_matrix.py`.
- `src/pipelines/training.py` (`maybe_data_parallel`, `unwrap_model`, `model_state_dict`, `mixup_batch`/`cutmix_batch`/`apply_mixup_or_cutmix`/`mixup_loss`) — mantidos, só ganham companhia do novo `Trainer` no mesmo arquivo (ou um `Trainer` em módulo novo que importa esses utilitários).
- `src/pipelines/evaluate_trained.py` — não é descartado, é a **base** do novo módulo de discovery/avaliação: `discover_trained_runs`, `build_model_from_run`, `evaluate_trained_runs`, `SplitSpec`/`build_split_specs` já implementam exatamente o padrão que precisa ser generalizado (hoje restrito a `SUPPORTED_FAMILIES = {"vit"}` e ao layout sem regime/seed).
- `src/plots/plots.py` (`plot_confusion_matrix`, `plot_roc_auc`, `save_metrics_csv`) — mantidos, só passam a chamar `src/plots/style.py::apply_style()` e a receber o path já resolvido com seed/regime.

## Estrutura nova

```
configs/
  base.yaml                    # defaults compartilhados (epochs, lr, batch_size, seeds, early_stop_patience...)
  resnet.yaml, xception.yaml, mobilenet.yaml, vit.yaml, clip.yaml, dino.yaml   # overrides por família
  # regime e fourier_mode são passados via CLI (--regime, --fourier) ou geridos pelo run_matrix.py

src/models/
  registry.py            # MODEL_REGISTRY: nome -> {build, freeze_backbone, unfreeze_for_finetune}
  _channel_adapt.py       # NOVO: extrai a cirurgia de canal (hoje só em mobilenet.py) para uso comum
  resnet.py, mobilenet.py  # torchvision, quase inalterados (só plugam no registry)
  xception.py             # scratch em torch puro (mantém Xception atual) + pretrained via timm com adaptação de 1ª camada
  vit.py, clip.py         # reescritos: wrapper único sobre transformers.ViTModel / CLIPVisionModel
                          #   (scratch=ViTConfig/CLIPVisionConfig com num_channels=N e peso aleatório;
                          #    finetune=from_pretrained + cirurgia de canal na patch-embedding, replicando
                          #    o padrão de _adapt_first_conv de mobilenet.py)
  dino.py                 # timm mantido; remove a checagem `if x.shape[1] != 3: raise ValueError(...)`
                          #   e aplica a mesma cirurgia de canal na stem/patch-embed do backbone

src/pipelines/
  training.py       # Trainer genérico (loop de época/AMP/scheduler/early-stop/checkpoint) + utilitários já existentes
  evaluation.py      # inalterado
  checkpoints.py     # discovery unificado (extensão de evaluate_trained.py: todas as 6 famílias, regime+seed no path)
  config.py          # TrainingConfig (dataclass) + loader de YAML com merge base->override->CLI

src/plots/
  style.py            # NOVO: MODEL_COLORS, apply_style(), fonte/paleta
  plots.py            # inalterado, exceto import de style
  heatmap.py          # consolida heatmap.py + resnet_heatmap_generator.py + transformer_heatmap_generator.py + attention_rollout.py

train.py            # roda 1 config (--config path.yaml [--fourier ...] [--seed ...] [--regime scratch|finetune])
run_matrix.py       # expande família x modo x regime x seed, distribui via run_tasks_on_gpus, expõe --gpus/--workers-per-gpu
evaluate.py         # substitui evaluate_trained_models.py/evaluate_all_models.py: generaliza evaluate_trained_runs
                     #   para as 6 famílias + regime/seed, splits val/test/test_d
ensemble.py         # estratégias + busca de subconjunto (substitui ensemble_fusion.py/fusion.py/strong_fusion.py/_run_fusion.py)
make_tables.py       # agrega metrics_*.csv por seed (média±desvio), exporta CSV/Markdown/LaTeX
generate_heatmaps.py # Grad-CAM (CNN) / Attention Rollout (Transformer) / grid, via checkpoints.py
```

## Detalhe por módulo

### 1. `src/pipelines/config.py`
`TrainingConfig` dataclass com todos os campos hoje espalhados como kwargs em `run_resnet`/`run_vit`/etc (epochs, batch_size, lr_head, lr_backbone, weight_decay, early_stop_patience, dropout, seeds, augment, threshold_strategy, mixup_alpha, image_size...) + `model_family`, `fourier_mode`, `regime` (`scratch`/`finetune`), `unfreeze_last_n`. Loader: lê `base.yaml`, aplica override do YAML da família, aplica overrides de CLI. Sem Pydantic extra — dataclass + `yaml.safe_load` é suficiente (adicionar `pyyaml` ao `pyproject.toml`).

### 2. `src/models/registry.py`
Dict simples `{"resnet": ModelSpec(build=..., freeze_backbone=..., unfreeze_for_finetune=...), "xception": ..., ...}`. Cada família implementa essas 3 funções no seu próprio arquivo (padrão já existe parcialmente: `resnet.py` já tem `freeze_backbone`/`unfreeze_last_blocks`, só precisa de um `build(config)` que decida `in_channels` a partir do modo Fourier e chame `resnet(...)`).

### 3. ViT/CLIP (`src/models/vit.py`, `src/models/clip.py`)
Reescrever como um único wrapper por família:
```python
def build_vit(config: TrainingConfig) -> nn.Module:
    if config.regime == "scratch":
        hf_config = ViTConfig(image_size=config.image_size, patch_size=16,
                               hidden_size=256, num_hidden_layers=6,
                               num_attention_heads=8, num_channels=config.in_channels)
        backbone = ViTModel(hf_config)  # peso aleatório
    else:
        backbone = ViTModel.from_pretrained("google/vit-base-patch16-224")
        adapt_patch_embedding_channels(backbone, config.in_channels)  # cirurgia como em mobilenet.py
    return ClassifierWrapper(backbone, num_classes=2, dropout=config.dropout)
```
`adapt_patch_embedding_channels` (em `src/models/_channel_adapt.py`) replica a lógica já validada em `mobilenet.py::_adapt_first_conv` (média para 1 canal, repetição+escala para >3), aplicada à `Conv2d` de `backbone.embeddings.patch_embeddings.projection` (ViT) / equivalente em `CLIPVisionModel`. Mesmo wrapper serve para os dois regimes — elimina `ResidualAttentionBlock`/`PatchEmbedding`/`ConvStemPatchEmbedding` customizados.

### 4. DINOv3 (`src/models/dino.py`)
Remove `if x.shape[1] != 3: raise ValueError(...)` em `forward`. Adiciona a mesma cirurgia de canal na stem do backbone `timm` (ConvNeXt tem `stem[0]` Conv2d, análogo ao `features[0][0]` de mobilenet — reaproveitar `adapt_patch_embedding_channels`/`_channel_adapt.py` em vez de duplicar a lógica 6x entre resnet/mobilenet/xception/vit/clip/dino).

### 5. `src/pipelines/training.py` — `Trainer`
Extrai o loop de `run_resnet` (o `for epoch in range(epochs)` de `src/pipelines/resnet.py`) para uma função/classe genérica que recebe `model`, `train_loader`, `val_loader`, `test_loader`, `config`, `output_dir` e faz: AMP, mixup/cutmix (reaproveita `apply_mixup_or_cutmix`/`mixup_loss` já existentes), `ReduceLROnPlateau`, early stopping por `epochs_without_improvement`, salvamento de `best.pth`/`final.pth` via `model_state_dict` (já existe), chamada a `evaluate_classifier`/`best_threshold`/`binary_metrics` (já existem em `evaluation.py`), e ao final chama `plot_confusion_matrix`/`plot_roc_auc`/`save_metrics_csv` (já existem em `plots.py`) apontando para `output_dir` = `models_root()/<família>/<modo>/<regime>/seed_<N>/`. Otimizador/param-groups (head vs backbone com LRs diferentes) fica um método pequeno que cada `ModelSpec` pode customizar (ex: ResNet separa `layer3`/`layer4`; ViT/CLIP separam por `unfreeze_last_n`).

### 6. `train.py`
```
python train.py --config configs/resnet.yaml --fourier concat --regime scratch --seed 42
```
Resolve config → `registry.build()` → `Trainer.fit()`. Substitui a chamada direta a `run_resnet(...)` que hoje só existe dentro de `main.py`/`fine_tuning.py`.

### 7. `run_matrix.py`
```
python run_matrix.py --regime scratch                       # todas as familias x 7 modos x seeds do config
python run_matrix.py --regime finetune --only resnet,vit
python run_matrix.py --regime scratch --gpus 0,1 --workers-per-gpu 2
```
Gera a lista de tarefas (família, modo, seed) e monta `tasks = [{"fn": train_from_config, "name": ..., "kwargs": {...}} ...]`, chama `run_tasks_on_gpus(tasks, gpus=..., workers_per_gpu=...)` sem alterar essa função.

### 8. `src/pipelines/checkpoints.py`
Generaliza `evaluate_trained.py`:
- `SUPPORTED_FAMILIES` passa a ser as 6 famílias.
- `discover_trained_runs` passa a extrair também `regime` e `seed` do path (`models/<família>/<modo>/<regime>/seed_<N>/weights/best.pth`).
- `build_model_from_run` ganha os branches de ViT/CLIP/DINO usando os novos `build_vit`/`build_clip`/DINO do registry em vez dos construtores antigos.
- `evaluate_trained_runs` continua igual em espírito (usa `evaluate_classifier` de `evaluation.py`), só precisa iterar por seed também e escrever em `results/metrics_{split}.csv` dentro de cada pasta `seed_<N>/`.

### 9. `evaluate.py`
CLI fina que só chama `evaluate_trained_runs` de `checkpoints.py` para todas as famílias, nos splits `val,test,test_d`. Substitui `evaluate_trained_models.py` (shim de 6 linhas) e `evaluate_all_models.py` (quebrado).

### 10. `ensemble.py`
- Usa `checkpoints.py::discover_trained_runs` para achar todos os candidatos.
- Módulo `src/pipelines/ensemble_strategies.py` com funções puras: `mean`, `weighted_by_val_auc`, `majority_vote`, `max_prob`, `geometric_mean`, `stacking_logreg` (usa `sklearn.linear_model.LogisticRegression`, já é dependência via `scikit-learn`).
- Busca de subconjunto: reaproveita a lógica de `ensemble_fusion.py` (`ProcessPoolExecutor` para busca combinatória) quando `--pool best-mode` (≤12, viável exaustivo); troca para greedy incremental quando `--pool all` (até 84).
- Fluxo: escolhe subconjunto pelo AUC no **val** → aplica a mesma combinação nas predições de **test** e **test_d** → salva relatório (`ensemble_report.csv` + `ensemble_predictions_{split}.csv`).

### 11. `make_tables.py`
Lê todos os `results/metrics_{split}.csv` descobertos por `checkpoints.py`, agrupa por (família, modo, regime, split) e agrega `mean`/`std` sobre as seeds. Exporta:
- `tables/results_full.csv` e `.md` (uma linha por combinação, todas as métricas).
- `tables/results_paper.tex` (booktabs, só as colunas relevantes para o `final.tex`, uma tabela por split, seguindo o exemplo de tabela que já aparece no `README.md`).

### 12. `src/plots/style.py`
```python
MODEL_COLORS = {"resnet": "#4477AA", "xception": "#EE6677", "mobilenet": "#228833",
                 "vit": "#CCBB44", "clip": "#66CCEE", "dino": "#AA3377"}
def apply_style(): plt.rcParams.update({"font.family": "serif", "font.size": 11, ...})
```
`plots.py::plot_confusion_matrix/plot_roc_auc` chamam `apply_style()` no início; `plot_roc_auc` passa a usar `MODEL_COLORS[family]` em vez do `color` fixo `#4C72B0` atual.

### 13. `generate_heatmaps.py`
Consolida `src/plots/heatmap.py` + `resnet_heatmap_generator.py` + `transformer_heatmap_generator.py` + `attention_rollout.py` num único módulo `src/plots/heatmap.py` com `generate(model, family, image, method="auto")`, onde `method="auto"` escolhe Grad-CAM para `{resnet, xception, mobilenet}` e Attention Rollout para `{vit, clip, dino}`. O script da raiz só faz parsing de CLI e chama `checkpoints.py::load_model_from_run` + `heatmap.generate(...)`. `--grid` gera o grid lado-a-lado (substitui `generate_heatmap_grid.py`).

## Ambientes (Slurm + 2 servidores, com/sem multi-GPU)

Hoje `src/data/paths.py` já resolve `phase1_split_root`/`data_root` via `TCC_DATASET_ROOT`/`TCC_DATA_ROOT` (env var com fallback para um path fixo) — esse é o mecanismo a estender, não substituir:

- **`TCC_MODELS_ROOT`** (novo, mesmo padrão, função `models_root()` em `paths.py`): raiz onde `train.py`/`run_matrix.py` gravam e `checkpoints.py`/`evaluate.py`/`ensemble.py` leem `models/<família>/<modo>/<regime>/seed_<N>/...`. Fallback para `./models` (cwd) se não definida — preserva o comportamento atual de rodar a partir da raiz do repo.
- **`TCC_OUTPUT_ROOT`** (novo, mesmo padrão, função `output_root()`): raiz para `tables/` (saída de `make_tables.py`) e heatmaps gerados por `generate_heatmaps.py`. Fallback para `./` (cwd).
- Cada um dos 3 ambientes só precisa exportar essas 4 vars uma vez (`.env.slurm`, `.env.server1`, `.env.server2`, não versionados — mesma lógica de segredo/local-config que já existe para `TCC_DATASET_ROOT`). Nenhum código de detecção de ambiente é necessário.
- GPUs: `run_matrix.py` expõe `--gpus 0,1,2,3` (repassado direto para `run_tasks_on_gpus(tasks, gpus=...)`, que já existe e já cai para CPU quando `torch.cuda.is_available()` é falso) e `--workers-per-gpu`. Em ambientes com 1 GPU só ou nenhuma, basta omitir `--gpus` (auto-detect) ou passar `--gpus 0`; no Slurm, `CUDA_VISIBLE_DEVICES` já é setado pelo alocador do job e o auto-detect existente respeita isso sem mudança de código.

## Migração de dados existentes

1. `mv models models_legacy_backup` (fora do controle do novo pipeline; git-ignorado se ainda não estiver).
2. `models_finetuned/` já está vazio — remover ou deixar como está (`.gitkeep`), o novo layout unifica scratch+finetune sob `models/`.
3. Atualizar `.gitignore` se necessário para `models_legacy_backup/`.

## Remoção de código obsoleto

`git rm main.py fine_tuning.py evaluate_all_models.py evaluate_trained_models.py fusion.py strong_fusion.py _run_fusion.py generate_resnet_heatmaps.py generate_transformer_heatmaps.py generate_heatmap_grid.py generate_extra_same_image_heatmaps.py README_IMPLEMENTATION.py`, e os módulos que eles usavam exclusivamente (`src/plots/resnet_heatmap_generator.py`, `src/plots/transformer_heatmap_generator.py` após seu conteúdo ser incorporado em `src/plots/heatmap.py`).

## Testes (`pytest`, sempre via `raw_min`/fixtures existentes)

- `tests/test_<familia>_smoke.py` para as 6 famílias (adicionar `test_dino_smoke.py`, hoje inexistente), usando `tiny_phase1_dataset`: build via registry, forward shape, canais extras (1/2/4/6), rejeição de pretrained externo indevido quando `allow_pretrained=False`.
- `tests/test_trainer.py`: `Trainer.fit` por 1-2 épocas no dataset minúsculo, checa checkpoint salvo e early stopping.
- `tests/test_config.py`: merge de `base.yaml` + override + CLI.
- `tests/test_checkpoints.py`: `discover_trained_runs` reconhece `família/modo/regime/seed_N` para as 6 famílias (substitui/estende `test_evaluate_trained_models.py`).
- `tests/test_ensemble.py`: estratégias (mean/weighted/stacking) e busca de subconjunto sobre métricas mockadas; `--pool best-mode` vs `--pool all`.
- `tests/test_make_tables.py`: agregação média±desvio sobre 3 seeds sintéticas.
- Manter `tests/test_evaluation_safety.py`, `tests/test_plots_safety.py`, `tests/test_heatmap.py` (adaptar imports ao novo `src/plots/heatmap.py` consolidado), `tests/test_data_paths.py` (estender para `models_root()`/`output_root()`).
- Adicionar `pyyaml` a `dependencies` no `pyproject.toml`.

## Ordem de execução recomendada (fases)

1. `src/models/_channel_adapt.py` (extrai a cirurgia de canal hoje só em `mobilenet.py`) + `src/pipelines/config.py` + `src/models/registry.py` + `models_root()`/`output_root()` em `paths.py`.
2. Portar ResNet e MobileNet para o registry (menor risco, arquitetura já é torchvision puro) + `Trainer` genérico validado contra eles (comparar métricas com o comportamento atual via smoke test).
3. Reescrever Xception (mantém scratch atual, adapta pretrained) e DINOv3 (remove restrição de canal) no registry.
4. Reescrever ViT/CLIP com `transformers.ViTModel`/`CLIPVisionModel`.
5. `train.py` + `run_matrix.py` (com `--gpus`/`--workers-per-gpu`).
6. `checkpoints.py` (generaliza `evaluate_trained.py`) + `evaluate.py`.
7. `ensemble.py` + `ensemble_strategies.py`.
8. `make_tables.py`.
9. `src/plots/style.py` + consolidação de heatmap em `generate_heatmaps.py`.
10. Remoção dos scripts obsoletos + arquivamento de `models/` → `models_legacy_backup/` + atualização do `README.md`.
11. Testes passando (`pytest -q`) antes de disparar a matriz completa de treino real (isso é trabalho de GPU, fora do escopo desta sessão de código).

## Verificação

- `pytest -q` deve passar (todos os smoke tests + novos testes de Trainer/registry/checkpoints/ensemble/tables), rodando em segundos via `raw_min`.
- Smoke manual: `python train.py --config configs/resnet.yaml --fourier none --regime scratch --seed 42 --epochs 1 --data-limit 32` conclui e grava em `models/resnet/none/scratch/seed_42/`.
- `python run_matrix.py --regime scratch --dry-run` lista as combinações planejadas sem rodar (flag nova, só para inspeção).
- `python evaluate.py --only-model-family resnet` gera `all_metrics_by_split.csv` sem erro.
- `python ensemble.py --strategy search --pool best-mode` roda sobre os metrics mockados dos smoke tests e produz `ensemble_report.csv`.
- `python make_tables.py` gera `tables/results_full.csv/.md/.tex` sem exceção.
- `python generate_heatmaps.py --checkpoint <um dos smoke> --image <qualquer jpg de raw_min> --method auto` produz um PNG.
