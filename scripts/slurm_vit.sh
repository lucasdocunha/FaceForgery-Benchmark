#!/bin/bash
#SBATCH --job-name=tcc-vit
#SBATCH --output=%x_%j.out
#SBATCH --error=%x_%j.err
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=16
#SBATCH --mem=64G
#SBATCH --mail-user=lucas.ocunha@ppgia.pucpr.br
#SBATCH --mail-type=ALL
#SBATCH --time=120:00:00

source /opt/conda/etc/profile.d/conda.sh
conda activate tcc

# ==========================================
# 1. Configurações de Ambiente (Paths CISIA)
# ==========================================
export TCC_DATASET_ROOT=/datasets/Images/MFFI
export TCC_DATA_ROOT=/users/home/lucas.ocunha/research/TCC/data
export TCC_MODELS_ROOT=/projects/lucas.ocunha/models
export TCC_OUTPUT_ROOT=/users/home/lucas.ocunha/research/TCC

# Ir para a pasta do repositório
cd /users/home/lucas.ocunha/research/TCC

# ==========================================
# 2. Execução do Modelo ViT
# ==========================================
export PYTHONUNBUFFERED=1

REGIME="${1:-scratch}"
WORKERS="${2:-1}"

echo "=========================================================="
echo "Job ID: $SLURM_JOB_ID | Nó: $(hostname)"
echo "Iniciando treinamento: vit (regime: $REGIME, workers: $WORKERS)"
echo "Data de início: $(date)"
echo "=========================================================="

python -u run_matrix.py \
    --regime "$REGIME" \
    --only vit \
    --workers-per-gpu "$WORKERS"

echo "=========================================================="
echo "Treinamento vit finalizado em: $(date)"
echo "=========================================================="
