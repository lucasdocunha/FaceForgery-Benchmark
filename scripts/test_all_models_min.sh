#!/bin/bash
#SBATCH --job-name=tcc-test-min
#SBATCH --partition=gpu
#SBATCH --output=logs/%x_%j.out
#SBATCH --error=logs/%x_%j.err
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --mail-user=lucas.ocunha@ppgia.pucpr.br
#SBATCH --mail-type=ALL
#SBATCH --time=01:00:00

set -euo pipefail

# ==========================================
# 1. Configurações de Ambiente (Paths CISIA)
# ==========================================
PROJECT_DIR="${TCC_PROJECT_ROOT:-${SLURM_SUBMIT_DIR:-$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)}}"
cd "$PROJECT_DIR"
mkdir -p "$PROJECT_DIR/logs"

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
elif conda info --envs | grep -q "^tcc-hpc[[:space:]]"; then
    conda activate tcc-hpc
elif conda info --envs | grep -q "^cae[[:space:]]"; then
    conda activate cae
fi

export TCC_DATASET_ROOT="${TCC_DATASET_ROOT:-/datasets/Images/MFFI}"
export TCC_DATA_ROOT="${TCC_DATA_ROOT:-$PROJECT_DIR/data}"
# Salva os resultados do teste no scratch/tmp local do nó para evitar estouro de cota
export TCC_MODELS_ROOT="${TCC_MODELS_ROOT:-/tmp/models_test_min}"
export TCC_OUTPUT_ROOT="${TCC_OUTPUT_ROOT:-$PROJECT_DIR}"

mkdir -p "$TCC_MODELS_ROOT"

# ==========================================
# 2. Parâmetros do Teste de Sanidade
# ==========================================
export PYTHONUNBUFFERED=1

# Parâmetros opcionais:
#   $1 = modo fourier (padrão: none)
#   $2 = épocas (padrão: 1)
#   $3 = regime (padrão: scratch)
FOURIER_MODE="${1:-none}"
EPOCHS="${2:-1}"
REGIME="${3:-scratch}"

echo "=========================================================="
echo "Job ID: $SLURM_JOB_ID | Nó: $(hostname)"
echo "TESTE DE SANIDADE (SMOKE TEST) NO DATASET MIN"
echo "Modelos : resnet, xception, mobilenet, vit, clip, dino"
echo "Dataset : data/raw_min (1.000 imagens)"
echo "Regime  : $REGIME"
echo "Épocas  : $EPOCHS"
echo "Modo    : $FOURIER_MODE"
echo "Data    : $(date)"
echo "=========================================================="

EXTRA_ARGS=""
if [ "$FOURIER_MODE" != "all" ]; then
    EXTRA_ARGS="--fourier $FOURIER_MODE"
fi

python -u run_matrix.py \
    --regime "$REGIME" \
    --raw-min \
    --epochs "$EPOCHS" \
    --seeds 42 \
    --force \
    --workers-per-gpu 1 \
    $EXTRA_ARGS

echo "=========================================================="
echo "TESTE CONCLUÍDO COM SUCESSO EM: $(date)"
echo "Verifique os resultados salvos em: $TCC_MODELS_ROOT"
echo "=========================================================="
