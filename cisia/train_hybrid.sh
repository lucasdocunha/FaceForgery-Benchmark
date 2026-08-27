#!/bin/bash
#SBATCH --job-name=hybrid-training
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
export TCC_DATA_ROOT=/home/lucas.ocunha/research/TCC/data
export TCC_MODELS_ROOT=/projects/lucas.ocunha/models
export TCC_OUTPUT_ROOT=/home/lucas.ocunha/research/TCC

# Ir para a pasta do repositório
cd /home/lucas.ocunha/research/TCC
git pull origin main

echo "=========================================="
echo "Job ID:        $SLURM_JOB_ID"
echo "Nó executando: $(hostname)"
echo "Início:        $(date)"
echo "GPU alocada:   $CUDA_VISIBLE_DEVICES"
echo "=========================================="
nvidia-smi

# ==========================================
# 2. Configurações de Sistema
# ==========================================
ulimit -n 65535

# ==========================================
# 3. Execução do Treinamento do Modelo Híbrido
# ==========================================
SEEDS=(42 123 2024 7 999)

for SEED in "${SEEDS[@]}"; do
    echo "------------------------------------------"
    echo "Iniciando Treino Hybrid | Seed: $SEED | $(date)"
    echo "------------------------------------------"
    PYTHONUNBUFFERED=1 python -u train.py \
        --config configs/hybrid.yaml \
        --regime finetune \
        --seed "$SEED"
done

echo "=========================================="
echo "Fim: $(date)"
echo "=========================================="
