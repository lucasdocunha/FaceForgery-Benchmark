#!/bin/bash -l
#SBATCH --job-name=TCC-df40-res
#SBATCH --partition=dev
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=28G
#SBATCH --output=logs/TCC-df40-cross_resnet_%j.out
#SBATCH --error=logs/TCC-df40-cross_resnet_%j.err

set -euo pipefail
cd /home/lucas.ocunha/tcc
export PYTHONPATH="/home/lucas.ocunha/tcc"

echo "=============================================================================="
echo "🚀 INICIANDO TREINAMENTO DF-40 + BENCHMARK REVERSO: RESNET-18 (SRM)"
echo "Node: $(hostname) | Data: $(date)"
echo "CUDA_VISIBLE_DEVICES: ${CUDA_VISIBLE_DEVICES:-none}"
echo "=============================================================================="

PYTHON="/home/lucas.ocunha/tcc/.venv/bin/python"

$PYTHON scripts/train_and_eval_df40_cross.py \
    --family resnet \
    --mode srm \
    --epochs 10 \
    --batch-size 64 \
    --eval-batch-size 256 \
    --lr-head 1e-3 \
    --lr-backbone 1e-4 \
    --seed 42 \
    --num-workers 6 \
    --eval-only

echo "=============================================================================="
echo "🎉 FINALIZADO TREINAMENTO DF-40 + BENCHMARK REVERSO RESNET!"
echo "=============================================================================="
