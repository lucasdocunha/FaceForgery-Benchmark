#!/bin/bash -l
#SBATCH --job-name=<nome-do-job>
#SBATCH --gres=gpu:1
#SBATCH --mem=64G
#SBATCH --cpus-per-task=8
#SBATCH --output=logs/%x_%j.out
#SBATCH --error=logs/%x_%j.err
#SBATCH --time=48:00:00
#SBATCH --mail-user=lucas.ocunha@ppgia.pucpr.br
#SBATCH --mail-type=ALL
# Submeter da raiz do projeto, com logs/ já criada: mkdir -p logs && sbatch <este>.sh
# Recursos acima são o ponto de partida — ajuste ao que o código usa (ver recursos.md).

source /opt/conda/etc/profile.d/conda.sh
conda activate <env>            # env em $HOME/.conda/envs (visível nos dois nós)
set -euo pipefail

# module load <modulo>          # só funciona por causa do "bash -l" no shebang

PROJETO_DIR="$(pwd)"
mkdir -p "$PROJETO_DIR/saidas"

# Cache efêmero no disco local do nó, nunca no NFS; apagado ao sair.
CACHE_DIR="${TMPDIR:-/scratch/$USER}/job_${SLURM_JOB_ID}"
mkdir -p "$CACHE_DIR"
export HF_HOME="$CACHE_DIR/hf" TORCH_HOME="$CACHE_DIR/torch" PIP_CACHE_DIR="$CACHE_DIR/pip"
export PYTHONUNBUFFERED=1       # .out atualiza durante o job
limpar() { rm -rf "$CACHE_DIR"; }
trap limpar EXIT

# Cabeçalho do log
INICIO=$(date +%s)
echo "=== job ${SLURM_JOB_ID} (${SLURM_JOB_NAME}) | nó $(hostname) | partição ${SLURM_JOB_PARTITION}"
echo "=== GPU: $(nvidia-smi --query-gpu=name,driver_version --format=csv,noheader 2>/dev/null || echo nenhuma)"
echo "=== env: ${CONDA_DEFAULT_ENV} | commit: $(git -C "$PROJETO_DIR" rev-parse --short HEAD 2>/dev/null || echo sem-git)"
echo "=== args: $* | início: $(date -Is)"

set +e
python <entrypoint.py> "$@"
STATUS=$?
set -e

# Rodapé do log
echo "=== fim: $(date -Is) | duração: $(( $(date +%s) - INICIO ))s | código de saída: ${STATUS}"
exit "$STATUS"
