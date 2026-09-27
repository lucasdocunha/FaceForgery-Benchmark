# FaceForgery no CISIA

Revisão baseada em `skill-hpc/` (referência de 25/09/2026). O escopo é o fluxo
`run_matrix.py`/`train.py`, o treino robusto e os preparadores de backbones. Scripts
históricos de outras máquinas não são entrypoints homologados por este guia.
A skill em si não foi alterada. Resultados científicos, splits, arquiteturas,
seeds padrão e regras de seleção de threshold não foram recalculados ou trocados.

## Prontidão

**Código preparado, com validações de software; homologação no cluster pendente.**
CI em CPU não comprova acesso ao CISIA, disponibilidade da fila, internet nos nós,
montagens, quota, performance NFS ou compatibilidade real H100/driver/conda.
Nenhum job real foi submetido como parte desta alteração.

| Checagem da skill | Implementação | Estado que ainda depende do cluster |
|---|---|---|
| Ambiente | `scripts/cisia_common.sh`: shell de login, source conda, ativação por nome em `$HOME/.conda/envs`, sem instalações no job | Ambiente e dependências disponíveis nos dois nós |
| Recursos | 1 GPU, 8 CPUs, 64 GiB, 48 h por treino; preparador só CPU, 32 GiB/4 h | Fila, quotas e capacidade para a configuração escolhida |
| Armazenamento | `src/hpc/runtime.py`: caches e checkpoints locais, versões finais via staging/rename, limpeza por job | Espaço livre, quota e saúde das montagens |
| Dataset | `/datasets`, override explícito sem fallback e bloqueio de caminhos de workstation nos manifests | Layout real de MFFI/test_d/DF40 e permissões de leitura |
| Modelos | Pesos compartilhados primeiro; origem do usuário depois; cache local como fallback; `MODELO.md`/`JOB.json` | Diretórios compartilhados efetivamente existentes e pesos compatíveis |
| Logs | Cabeçalho, config efetiva, progresso de treino em linhas com batches/s e ETA, métricas, destinos e status final | `.out`/`.err` são abertos pelo SLURM; criar `logs/` antes de submeter |
| Serviços | Nenhum servidor é iniciado neste fluxo | Não se aplica |

A apresentação do progresso da avaliação em `src/pipelines/evaluation.py` permanece
inalterada (`tqdm`). Portanto, a padronização completa dos logs de avaliação em
linhas simples ainda é uma pendência; o progresso de treino já usa esse formato.

## Preparação no Shell Access — sem treino/testes fora de alocação

O Shell Access chega a um nó compartilhado, **não** a um login node dedicado.
Preparação de ambiente, Git, listagem de arquivos e `sbatch` podem ser feitos ali;
treino, avaliação, construção de modelos e testes devem executar sob SLURM.

Primeiro confira o estado, sem presumir que os caminhos de outra conta existem:

```bash
source /opt/conda/etc/profile.d/conda.sh
printf 'HOME=%s\n' "$HOME"
conda env list
conda list -n tcc | grep -iE 'torch|torchvision|timm|transformers|numpy|pandas|scikit-learn'
ls -ld /scratch/"$USER" /projects/models/"$USER" /datasets/Images/MFFI
ls /datasets/Images/MFFI
sed -n '1,160p' /datasets/ai_models/README.md
```

Para criar **um ambiente novo**, sem substituir o `tcc` existente, a receita abaixo
usa Python 3.12 e o par oficial PyTorch 2.10.0 / torchvision 0.25.0 com CUDA 12.8.
Execute na raiz do checkout. `requirements-cisia.txt` reutiliza as dependências
científicas já declaradas em `requirements-research.txt`; é um conjunto de pins
diretos, não um lockfile transitivo completo. Não misture com o `uv.lock`/`requirements.txt` de workstation, que continuam preservados com CUDA 12.1.

```bash
source /opt/conda/etc/profile.d/conda.sh
export CONDA_ENVS_PATH="$HOME/.conda/envs"
mkdir -p /scratch/"$USER"
(
  setup_cache=$(mktemp -d /scratch/"$USER"/faceforgery-env.XXXXXX)
  trap 'rm -rf -- "$setup_cache"' EXIT
  export CONDA_PKGS_DIRS="$setup_cache/conda" PIP_CACHE_DIR="$setup_cache/pip"
  export TMPDIR="$setup_cache/tmp"
  mkdir -p "$TMPDIR"
  conda create -n tcc-hpc python=3.12 pip -y
  conda activate tcc-hpc
  python -m pip install torch==2.10.0 torchvision==0.25.0 --index-url https://download.pytorch.org/whl/cu128
  python -m pip install -r requirements-cisia.txt
  python -m pip check
)
export CISIA_CONDA_ENV=tcc-hpc
```

Não use `sudo`, Docker ou `apt`; nada disso é necessário. Problemas de CA devem ser
corrigidos com um bundle confiável (`SSL_CERT_FILE`/`REQUESTS_CA_BUNDLE` conforme o
cliente), nunca desativando a verificação TLS. O job não instala pacotes.

## Submeter

Mantenha o checkout dentro de `$HOME` (`/users/home/$USER/...`). Os submitters
resolvem a raiz real do checkout e criam `logs/` antes de chamar `sbatch`. Nos
entrypoints, a raiz vem de `TCC_PROJECT_ROOT` ou `SLURM_SUBMIT_DIR`: não se usa a
localização da cópia do script criada pelo SLURM em seu spool.

```bash
cd "$HOME/research/FaceForgery-Benchmark"  # ajuste para seu checkout real
export CISIA_CONDA_ENV=tcc-hpc             # ou tcc, se já compatível
export TCC_DATASET_ROOT=/datasets/Images/MFFI
export TCC_DATA_ROOT="$PWD/data"           # manifests raw/{train,val,test}.csv
mkdir -p logs saidas

# Um modelo, matriz normal (modo, seeds e número de épocas vêm dos YAMLs):
sbatch scripts/slurm_resnet.sh scratch 1

# Todos os seis modelos:
bash scripts/submit_all_cisia.sh scratch 1

# Um modelo robusto: famílias, seeds, regime, workers do DataLoader:
bash scripts/submit_all_robust_cisia.sh resnet 42,123,2024,7,2025 finetune_robust 4
```

O e-mail padrão foi mantido conforme `skill-hpc/template.sh`; outra conta deve
substituí-lo na submissão, por exemplo `sbatch --mail-user=seu-email ...`.
Não interprete a mensagem de submissão como treino bem-sucedido. Se um `sbatch`
falhar, o submitter para imediatamente e informa erro; jobs anteriores já aceitos
continuam na fila, com seus IDs impressos.

### Pesos pré-treinados

Confira o catálogo `/datasets/ai_models/README.md` antes de baixar. O diretório
padrão procurado é `/datasets/ai_models/faceforgery/pretrained`, **não uma afirmação
de que ele exista no cluster**. Para outro layout compatível, defina
`CISIA_SHARED_PRETRAINED_ROOT` para a raiz com subpastas `resnet/`, `clip/`, etc.
Por família, a precedência é compartilhado → `TCC_PRETRAINED_ROOT` (ou
`/projects/models/$USER/pretrained`) → download no cache local do job.

```bash
# Preparação/verificação sob alocação CPU; sem reservar GPU desnecessária:
sbatch scripts/download_pretrained_cisia.sh
# slurm_download_pretrained.sh é um entrypoint equivalente.
```

O preparador copia pesos existentes para staging local antes de escrever e
recarrega as seis arquiteturas para verificar compatibilidade. A publicação
imprime a versão final. Aponte os próximos treinos para **essa raiz publicada**:

```bash
export TCC_PRETRAINED_ROOT=/projects/models/"$USER"/faceforgery/preload-models_DATA_JOB_VERSAO
sbatch scripts/slurm_resnet.sh finetune 1
```

Não copie literalmente o nome ilustrativo acima. Use o caminho real do `.out`.
Downloads torchvision têm verificação SHA-256; arquivos temporários são exclusivos
e arquivos válidos anteriores não são truncados após falha. Falta de acesso à
internet, CA inválida, peso parcial/incompatível ou modelo gated devem ser
resolvidos antes da matriz longa. Variáveis de modo offline continuam respeitadas.

## Onde os arquivos ficam

| Conteúdo | Local |
|---|---|
| Código, manifests e ambiente conda | `$HOME` compartilhado |
| Dataset, somente leitura | `/datasets/...` |
| Cache HF/Torch/XDG/pip/CUDA/Triton/Matplotlib/Numba e temporários | `${TMPDIR:-/scratch/$USER}/job_<id>_<sufixo>/cache/` |
| Checkpoints durante o treino | Mesmo workspace local, em `models/` |
| Modelo final versionado | `/projects/models/$USER/faceforgery/<job>_<data>_<id>_<versao>/` |
| Relatórios, predições, tabelas e gráficos | `<checkout>/saidas/<mesma-versao>/` |
| Progresso e erros | `<checkout>/logs/%x_%j.out` e `.err` |

`TMPDIR` deve resolver para `/scratch` ou `/tmp`, não para um symlink no NFS.
`CISIA_MODELS_ROOT` pode escolher outra subpasta de `/projects/models/$USER`;
`CISIA_OUTPUT_ROOT` pode escolher uma subpasta de `<checkout>/saidas`.
O runner substitui `TCC_MODELS_ROOT`/`TCC_OUTPUT_ROOT` **no subprocesso** pelo
workspace local: use as variáveis `CISIA_*_ROOT` para configurar publicação.

A estrutura `family/mode/regime/seed_N/{weights,results,plots}` fica preservada na
versão publicada para os leitores existentes. As pastas `results` e `plots` são
também copiadas uma vez para `saidas/.../runs/`, sem pesos. Isso duplica relatórios
para compatibilidade, não checkpoints por época. Cada seed mantém somente o
checkpoint validation-best (`best.pth`), sem uma cópia final redundante.

`JOB.json`/`MODELO.md` registram commit, versões de bibliotecas, comando, dataset,
job/nó/partição, duração, configs efetivas e métricas por seed. O campo
`workload_exit_code` descreve o processo de treino/preparação. O status final do
SLURM pode também indicar falha de publicação/limpeza: confira os logs.

### Retomada e falhas

Por padrão, cada submissão produz uma versão nova, sem sobrescrever resultados
anteriores. Para reutilizar explicitamente uma versão do **mesmo experimento**:

```bash
export CISIA_RESUME_FROM=/projects/models/"$USER"/faceforgery/VERSAO_ANTERIOR
sbatch scripts/slurm_resnet.sh scratch 1
```

Essa opção copia a árvore anterior para disco local e reutiliza a lógica existente
que pula seeds/modos concluídos. **Não é retomada de optimizer/epoch**: tarefas
incompletas recomeçam. Não use uma versão com hiperparâmetros/splits diferentes;
`--force` exige reexecução no runner da matriz. Não rode uma avaliação científica
sobre uma versão cujo treino falhou sem verificar a completude de cada seed.

Falhas de treino retornam erro ao SLURM, inclusive falhas de workers. Artefatos
parciais são publicados com status não zero. Falhas de cópia não descartam a única
cópia dos checkpoints: caches são limpos, o workspace é retido e o `.err` imprime
`RECOVERY`, nó e caminho. Recupere apenas esse workspace por meio de uma alocação
no nó indicado; depois remova-o. Staging não substitui versões anteriores.

Não há promessa de atomicidade conjunta entre dois filesystems. Cada diretório
final usa sua própria publicação por rename. Um relatório pode falhar depois de
os modelos já terem sido publicados; o job ainda retorna erro e retém a origem.
SIGTERM/SIGINT são encaminhados somente ao grupo de processos do workload, mas
SIGKILL, perda do nó ou expiração do período de graça podem impedir a cópia final.
Planeje tempo de exportação antes do limite e espaço para a matriz inteira.
O mínimo inicial é 20 GiB livres (`CISIA_MIN_FREE_GB`), **não uma estimativa de
capacidade suficiente para todas as arquiteturas/configurações**.

## Validação antes de gastar uma matriz completa

Um smoke de treino deve ser submetido, nunca executado diretamente no Shell Access:

```bash
mkdir -p logs saidas
# Use manifests raw e imagens reais, uma seed e um modo, uma época.
sbatch --time=00:30:00 scripts/slurm_resnet.sh scratch 1 --seeds 42 --fourier none --epochs 1
squeue -u "$USER"
# Substitua ID pelo job efetivamente recebido:
sacct -j ID --format=JobID,State,ExitCode,Elapsed,MaxRSS
```

Confira GPU/CUDA no cabeçalho, erros no `.err`, métricas finais, caminhos publicados
e `workload_exit_code`. O pequeno teste não mede toda a matriz, todos os backbones
ou a estabilidade prolongada do NFS. O fluxo robusto também precisa que `test_d`
exista; DF40 só é avaliado quando seu manifest opcional existe em
`$TCC_DATA_ROOT/df40/test.csv`, mantendo a política anterior de avaliação opcional.

A CI `CISIA HPC checks` roda em CPU/Python 3.12: sintaxe Python/Bash, suíte de
software, subprocessos com exceção e morte abrupta, máscaras de GPU numéricas e
UUID/MIG simuladas, persistência atômica, recuperação de falha de publicação,
submitters com `sbatch` simulado e um treino real mínimo em CPU. Não submete jobs,
não baixa os datasets científicos e não usa resultados fabricados como benchmark.

## Referências técnicas

- Regras locais: [`../skill-hpc/SKILL.md`](../skill-hpc/SKILL.md) e documentos vinculados.
- SLURM, diretório de submissão, abertura de logs e argumentos: <https://slurm.schedmd.com/sbatch.html>.
- SLURM, propagação do exit code: <https://slurm.schedmd.com/job_exit_code.html>.
- Python, processos, filas, encerramento e deadlocks: <https://docs.python.org/3/library/multiprocessing.html>.
- Par oficial torch/torchvision e CUDA 12.8: <https://pytorch.org/get-started/previous-versions/>.
