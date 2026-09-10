#!/bin/bash
#SBATCH --job-name=preload-models
#SBATCH --partition=gpu
#SBATCH --output=%x_%j.out
#SBATCH --error=%x_%j.err
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --mail-user=lucas.ocunha@ppgia.pucpr.br
#SBATCH --mail-type=ALL
#SBATCH --time=04:00:00

# ==============================================================================
# Script Slurm para baixar e verificar todos os backbones pré-treinados no CISIA
# Submeter com:
#   sbatch scripts/download_pretrained_cisia.sh
#   ou:
#   sbatch scripts/slurm_download_pretrained.sh
# ==============================================================================

set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_DIR="$(cd "$SCRIPT_DIR/.." && pwd)"

echo "=========================================================="
echo "Job ID: $SLURM_JOB_ID | Nó: $(hostname)"
echo "📥 CONFIGURAÇÃO E PRÉ-CARREGAMENTO DE MODELOS NO CISIA"
echo "Data de início: $(date)"
echo "=========================================================="

# 1. Ativar ambiente Conda no CISIA
if [ -f "/opt/conda/etc/profile.d/conda.sh" ]; then
    source /opt/conda/etc/profile.d/conda.sh
    conda activate tcc
elif command -v conda &> /dev/null; then
    eval "$(conda shell.bash hook)"
    conda activate tcc
fi

# 2. Configurações de Caminhos no /projects
export TCC_MODELS_ROOT=/projects/models/lucas.ocunha
export TCC_PRETRAINED_ROOT=/projects/models/lucas.ocunha/pretrained
export HF_HOME=/projects/models/lucas.ocunha/.cache/huggingface
export TORCH_HOME=/projects/models/lucas.ocunha/.cache/torch
export PYTHONUNBUFFERED=1

mkdir -p "$TCC_PRETRAINED_ROOT" "$HF_HOME" "$TORCH_HOME"

cd /users/home/lucas.ocunha/research/TCC

# 3. Executar o script Python de download e verificação
python -u scripts/setup_pretrained_cisia.py --target-dir "$TCC_PRETRAINED_ROOT"

EXIT_CODE=$?

echo "=========================================================="
echo "🎉 TODOS OS MODELOS FORAM BAIXADOS / REAPROVEITADOS!"
echo "Eles estão salvos de forma independente em:"
echo "   $TCC_PRETRAINED_ROOT"
echo ""
echo "Finalizado em: $(date) (exit: $EXIT_CODE)"
echo "🚀 Próximo passo: submeter os jobs para o Slurm:"
echo "   ./scripts/submit_all_robust_cisia.sh"
echo "=========================================================="
exit $EXIT_CODE
