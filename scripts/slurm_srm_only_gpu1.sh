#!/bin/bash -l
#SBATCH --job-name=TCC-17-srm1
#SBATCH --partition=dev
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=28G
#SBATCH --output=logs/TCC-17-srm_only_gpu1_%j.out
#SBATCH --error=logs/TCC-17-srm_only_gpu1_%j.err

set -euo pipefail
cd /home/lucas.ocunha/tcc

echo "=== Iniciando Worker Slurm GPU 1 (Transformers: vit, clip, dino) ==="
echo "Node: $(hostname) | Data: $(date)"
echo "CUDA_VISIBLE_DEVICES: ${CUDA_VISIBLE_DEVICES:-none}"

/home/lucas.ocunha/tcc/.venv/bin/python -u scripts/run_forensics_campaign.py \
    --gpu 0 \
    --mode srm_only \
    --families "vit,clip,dino" \
    --seeds "42,123,2024,7,2025" \
    --epochs 15 \
    --batch-size 32 \
    --num-workers 4

echo "=== Finalizado Worker Slurm GPU 1 ==="
