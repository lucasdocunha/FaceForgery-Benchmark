#!/bin/bash
#SBATCH --job-name=MFFI-from-scratch
#SBATCH --output=%x_%j.out
#SBATCH --error=%x_%j.err
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=32
#SBATCH --mem=128G
#SBATCH --mail-user=lucas.ocunha@ppgia.pucpr.br
#SBATCH --mail-type=ALL
#SBATCH --time=360:00:00

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

echo "=========================================="
echo "Job ID:        $SLURM_JOB_ID"
echo "Nó executando: $(hostname)"
echo "Início:        $(date)"
echo "GPU alocada:   $CUDA_VISIBLE_DEVICES"
echo "=========================================="
nvidia-smi

# ==========================================
# 3. Execução da Matriz Scratch
# ==========================================
# --gpus 0            : Usa a H100 alocada pelo Slurm
# --workers-per-gpu 6 : Roda 6 modelos em paralelo na H100
# 32 CPUs alocadas garantem que os 24 subprocessos de DataLoader rodem com alta vazão
PYTHONUNBUFFERED=1 python -u run_matrix.py \
    --regime scratch \
    --gpus 0 \
    --workers-per-gpu 6

echo "=========================================="
echo "Fim: $(date)"
echo "=========================================="