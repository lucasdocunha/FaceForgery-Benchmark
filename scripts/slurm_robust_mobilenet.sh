#!/bin/bash
#SBATCH --job-name=tcc-rob-mobilenet
#SBATCH --partition=gpu
#SBATCH --output=%x_%j.out
#SBATCH --error=%x_%j.err
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=16
#SBATCH --mem=64G
#SBATCH --mail-user=lucas.ocunha@ppgia.pucpr.br
#SBATCH --mail-type=ALL
#SBATCH --time=120:00:00

# Ativação do ambiente no CISIA
source /opt/conda/etc/profile.d/conda.sh
conda activate tcc

# ==========================================
# 1. Configurações de Ambiente (Paths CISIA)
# ==========================================
export TCC_DATASET_ROOT=/datasets/Images/MFFI
export TCC_DATA_ROOT=/users/home/lucas.ocunha/research/TCC/data
export TCC_MODELS_ROOT=/projects/lucas.ocunha/models
export TCC_OUTPUT_ROOT=/users/home/lucas.ocunha/research/TCC

# Redirecionar cache do HuggingFace e Torch para /projects (evita estouro de cota e I/O error na /users/home)
export HF_HOME=/projects/lucas.ocunha/.cache/huggingface
export TORCH_HOME=/projects/lucas.ocunha/.cache/torch
mkdir -p "$HF_HOME" "$TORCH_HOME"

export PYTHONUNBUFFERED=1

# Ir para a pasta do repositório no CISIA
cd /users/home/lucas.ocunha/research/TCC

# ==========================================
# 2. Execução do Treinamento Robusto (5 Seeds)
# ==========================================
SEEDS="${1:-42,123,2024,7,2025}"
REGIME="${2:-finetune_robust}"
WORKERS="${3:-8}"

echo "=========================================================="
echo "Job ID: $SLURM_JOB_ID | Nó: $(hostname)"
echo "Iniciando benchmark robusto: mobilenet"
echo "Seeds: $SEEDS | Regime: $REGIME | Workers: $WORKERS"
echo "Cache HF: $HF_HOME | Cache Torch: $TORCH_HOME"
echo "Data de início: $(date)"
echo "=========================================================="

python -u scripts/train_robust_seeds.py \
    --family "mobilenet" \
    --seeds "$SEEDS" \
    --regime "$REGIME" \
    --num-workers "$WORKERS"

EXIT_CODE=$?

echo "=========================================================="
echo "Treinamento robusto mobilenet finalizado em: $(date) (exit: $EXIT_CODE)"
echo "=========================================================="
exit $EXIT_CODE
