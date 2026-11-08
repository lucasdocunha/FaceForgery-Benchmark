#!/bin/bash -l
#SBATCH --job-name=HF-Sync
#SBATCH --partition=dev
#SBATCH --cpus-per-task=4
#SBATCH --mem=8G
#SBATCH --output=logs/sync_hf_%j.out
#SBATCH --error=logs/sync_hf_%j.err

set -euo pipefail
cd /home/lucas.ocunha/tcc

PYTHON="/home/lucas.ocunha/.conda/envs/cae/bin/python"

echo "=============================================================================="
echo "🌐 INICIANDO SINCRONIZAÇÃO COMPLETA COM HUGGING FACE (lucasoc/MFFI-Models)"
echo "Node: $(hostname) | Data: $(date)"
echo "=============================================================================="

"$PYTHON" -u scripts/sync_missing_to_huggingface.py

echo "=============================================================================="
echo "🎉 SINCRONIZAÇÃO HUGGING FACE CONCLUÍDA COM SUCESSO!"
echo "=============================================================================="
