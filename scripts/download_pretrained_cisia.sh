#!/bin/bash
# ==============================================================================
# Script para baixar e verificar todos os backbones pré-treinados no CISIA
# Executar no nó de login (boolevm) onde há acesso à internet:
#   ./scripts/download_pretrained_cisia.sh
# ==============================================================================

set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_DIR="$(cd "$SCRIPT_DIR/.." && pwd)"

echo "=========================================================="
echo "📥 CONFIGURAÇÃO E PRÉ-CARREGAMENTO DE MODELOS NO CISIA"
echo "Host: $(hostname) | Data: $(date)"
echo "=========================================================="

# 1. Ativar ambiente Conda
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

cd "$REPO_DIR"

# 3. Executar o script Python de download e verificação
python -u scripts/setup_pretrained_cisia.py --target-dir "$TCC_PRETRAINED_ROOT"

echo "=========================================================="
echo "🎉 TODOS OS MODELOS FORAM BAIXADOS / REAPROVEITADOS!"
echo "Eles estão salvos de forma independente em:"
echo "   $TCC_PRETRAINED_ROOT"
echo ""
echo "🚀 Próximo passo: submeter os jobs para o Slurm:"
echo "   ./scripts/submit_all_robust_cisia.sh"
echo "=========================================================="
