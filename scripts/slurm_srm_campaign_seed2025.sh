#!/bin/bash -l
#SBATCH --job-name=srm-2025
#SBATCH --partition=gpu
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=48:00:00
#SBATCH --output=logs/%x_%j.out
#SBATCH --error=logs/%x_%j.err
#SBATCH --mail-user=lucas.ocunha@ppgia.pucpr.br
#SBATCH --mail-type=ALL
#
# Script SLURM padronizado para o cluster CISIA (H100) / Workstation
# Executa a última semente canônica (Seed 2025) da campanha forense SRM
# Submissão:
#   mkdir -p logs saidas && sbatch scripts/slurm_srm_campaign_seed2025.sh
#
# Para rodar apenas um modelo específico (ex: resnet):
#   sbatch scripts/slurm_srm_campaign_seed2025.sh resnet

set -euo pipefail

# 1. Diretório do projeto
PROJECT_DIR="${TCC_PROJECT_ROOT:-${SLURM_SUBMIT_DIR:-$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)}}"
cd "$PROJECT_DIR"
mkdir -p "$PROJECT_DIR/logs" "$PROJECT_DIR/saidas"

# 2. Ativação do ambiente Conda (regra do CISIA: shell de login + perfil do conda)
if [[ -f /opt/conda/etc/profile.d/conda.sh ]]; then
    source /opt/conda/etc/profile.d/conda.sh
elif [[ -f "$HOME/miniconda3/etc/profile.d/conda.sh" ]]; then
    source "$HOME/miniconda3/etc/profile.d/conda.sh"
elif [[ -f "$HOME/.conda/etc/profile.d/conda.sh" ]]; then
    source "$HOME/.conda/etc/profile.d/conda.sh"
fi

CONDA_ENV_NAME="${CISIA_CONDA_ENV:-tcc}"
if conda info --envs | grep -q "^${CONDA_ENV_NAME}[[:space:]]"; then
    conda activate "$CONDA_ENV_NAME"
elif conda info --envs | grep -q "^cae[[:space:]]"; then
    conda activate cae
else
    echo "⚠️  Ambiente $CONDA_ENV_NAME não encontrado; usando o ambiente conda ativo."
fi

# 3. Cache efêmero no disco local (EXT4 /scratch), NUNCA no NFS (evita erro EIO)
CACHE_DIR="${TMPDIR:-/scratch/$USER}/job_${SLURM_JOB_ID:-manual_$$}"
mkdir -p "$CACHE_DIR/hf" "$CACHE_DIR/torch" "$CACHE_DIR/pip" "$CACHE_DIR/tmp"
export HF_HOME="$CACHE_DIR/hf"
export TORCH_HOME="$CACHE_DIR/torch"
export PIP_CACHE_DIR="$CACHE_DIR/pip"
export TMPDIR="$CACHE_DIR/tmp"
export PYTHONUNBUFFERED=1
export PYTHONDONTWRITEBYTECODE=1

# Garante limpeza absoluta do cache local ao finalizar ou em caso de erro
limpar_scratch() {
    local status=$?
    echo "Limpando diretório temporário local em $CACHE_DIR..."
    rm -rf "$CACHE_DIR"
    exit "$status"
}
trap limpar_scratch EXIT

# 4. Configuração de caminhos portáveis (Cluster CISIA vs Workstation)
if [[ -d "/datasets/Images/MFFI" ]]; then
    export TCC_DATASET_ROOT="/datasets/Images/MFFI"
    export TCC_MODELS_ROOT="${TCC_MODELS_ROOT:-/projects/models/$USER/models-tcc}"
    export TCC_OUTPUT_ROOT="${TCC_OUTPUT_ROOT:-$PROJECT_DIR/saidas}"
    if [[ -d "/datasets/Images/celeb_df_crops" ]]; then
        export TCC_CELEB_CROPS_DIR="/datasets/Images/celeb_df_crops"
    elif [[ -d "/datasets/celeb_df_crops" ]]; then
        export TCC_CELEB_CROPS_DIR="/datasets/celeb_df_crops"
    fi
fi

# 5. Cabeçalho obrigatório de auditoria e telemetria
INICIO=$(date +%s)
echo "=========================================================================="
echo "=== JOB ID: ${SLURM_JOB_ID:-LOCAL} (${SLURM_JOB_NAME:-srm-2025})"
echo "=== Nó: $(hostname) | Partição: ${SLURM_JOB_PARTITION:-gpu}"
echo "=== GPU: $(nvidia-smi --query-gpu=name,memory.total,driver_version --format=csv,noheader 2>/dev/null || echo 'Nenhuma GPU detectada')"
echo "=== Python: $(which python) ($(python --version 2>&1))"
echo "=== Conda Env: ${CONDA_DEFAULT_ENV:-none} (Prefix: ${CONDA_PREFIX:-none})"
echo "=== Commit: $(git rev-parse --short HEAD 2>/dev/null || echo 'sem-git')"
echo "=== Início: $(date -Is)"
echo "=========================================================================="

# 6. Parâmetros da Campanha
FAMILIES="${1:-mobilenet,resnet,xception,vit,clip,dino}"
SEED=2025
MODE="srm"
NUM_WORKERS=7   # Deixa 1 CPU livre das 8 alocadas para o loop principal do PyTorch

echo "Iniciando campanha forense SRM..."
echo "  • Modo:       $MODE"
echo "  • Semente:    $SEED (Última Semente Canônica)"
echo "  • Famílias:   $FAMILIES"
echo "  • Workers:    $NUM_WORKERS"
echo "=========================================================================="

# 7. Execução da Campanha SRM
python -u scripts/run_forensics_campaign.py \
    --gpu 0 \
    --mode "$MODE" \
    --families "$FAMILIES" \
    --seeds "$SEED" \
    --epochs 15 \
    --batch-size 64 \
    --num-workers "$NUM_WORKERS"

STATUS=$?

# 8. Rodapé com tempo de execução e código de saída
FIM=$(date +%s)
DURACAO=$(( FIM - INICIO ))
HORAS=$(( DURACAO / 3600 ))
MINUTOS=$(( (DURACAO % 3600) / 60 ))
SEGUNDOS=$(( DURACAO % 60 ))

echo "=========================================================================="
echo "=== Finalizado: $(date -Is)"
echo "=== Duração Total: ${HORAS}h ${MINUTOS}m ${SEGUNDOS}s (${DURACAO}s)"
echo "=== Código de saída: $STATUS"
echo "=========================================================================="

exit "$STATUS"
