# Fatos do cluster CISIA

Levantados em 2026-09-25 no próprio cluster (sinfo, df, module show, página de boas-vindas do OOD).

## Nós e partições

- 2 nós: `babbagevm` e `boolevm`, cada um com 4× NVIDIA H100 80GB HBM3 (SXM5), driver 580, x86_64. Ubuntu 24.04.
- **O Shell Access (OOD → Clusters → HPC CISIA Shell Access) cai no `boolevm`, que também roda jobs.** Não há nó de login separado: no shell, só preparação (conda/pip, `apptainer pull`, git, `sbatch`/`squeue`/`sacct`, ler logs). Qualquer cálculo — até teste rápido — vai por `sbatch`/`srun`.

| Partição | Tempo máx. | Recurso de GPU | Limites |
|---|---|---|---|
| `gpu` (padrão) | sem limite | `--gres=gpu:1`, `gpu:2` ou `gpu:4` (também aceita `gpu:h100:N`) | `--mem` ≤ 884G, `--cpus-per-task` ≤ 124 |
| `shared` | 4h | `--gres=mps:<%>` (14, 25, 50…; % dos SMs, total 100 no nó) | 2 jobs rodando por usuário (`QOSMaxJobsPerUserLimit`); **VRAM não é particionada** (OOM derruba o job); `gpu:` e `mps:` no mesmo job = erro de submissão |

- Nome de partição é só `gpu` ou `shared` — `-p gpu:1` é inválido.
- Dependências: `sbatch --dependency=afterok:$JOB1`; filho de pai que falhou fica `DependencyNeverSatisfied`. Template "Pipeline com Dependências" no Job Composer.

## Software

- Conda em `/opt/conda` (`source /opt/conda/etc/profile.d/conda.sh`); conda 26. Envs do sistema: `base`, `jupyter-base`, `rapids`.
- Lmod: `module` **só existe em shell de login** (`#!/bin/bash -l` ou `bash -lc`). Único módulo de aplicação: `ollama/0.20.7`.
- Apptainer 1.5 (`apptainer pull x.sif docker://...`, `apptainer exec x.sif ...`). NGC: `sudo ngc-pull nvcr.io/nvaie/<img>` grava em `/datasets/containers_ngc/`.
- **Usuário não tem sudo** (exceto `ngc-pull`). Docker não serve.
- PyTorch: wheels `cu128` (`--index-url https://download.pytorch.org/whl/cu128`).
- Internet: o Shell Access (que é nó de cômputo) instala via pip/conda; assume-se que jobs também têm saída (não verificado dentro de job).

## Armazenamento

| Caminho | Tipo | Uso |
|---|---|---|
| `$HOME` = `/users/home/$USER` | NFS (Isilon) | código, envs conda (`~/.conda/envs`) |
| `/projects/models/$USER` | NFS | modelos treinados/finais |
| `/datasets` | NFS, gravável | datasets finais, compartilhados; `ai_models/` (pesos do cluster, só leitura), `containers_ngc/` |
| `/scratch`, `/tmp` | disco local do nó (ext4, ~1.9T livres, divide com o SO) | cache efêmero de job |
| `/home/$USER` | disco local de **um** nó | **não usar** — não existe igual no outro nó |

- **Escrita pesada/binária no NFS falha**: gravar blob de modelo (Ollama) em `/users/home` ou `/projects` deu `EIO` no `close()`. Cache de runtime sempre em disco local.
- Máximo 500–1.000 arquivos por pasta (regra do cluster; pastas maiores travam a aba Files e o storage). Pasta superlotada: manipular só pelo terminal.
- `sbatch` não cria o diretório de `--output`: se `logs/` não existir, o job falha sem log.

## Contato do usuário (fixo no template)

- `--mail-user=lucas.ocunha@ppgia.pucpr.br`
