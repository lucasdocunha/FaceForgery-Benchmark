#!/bin/bash
#SBATCH --job-name=tcc-test-min
#SBATCH --partition=gpu
#SBATCH --output=%x_%j.out
#SBATCH --error=%x_%j.err
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=16
#SBATCH --mem=32G
#SBATCH --mail-user=lucas.ocunha@ppgia.pucpr.br
#SBATCH --mail-type=ALL
#SBATCH --time=01:00:00

source /opt/conda/etc/profile.d/conda.sh
conda activate tcc

# ==========================================
# 1. Configurações de Ambiente (Paths CISIA)
# ==========================================
export TCC_DATASET_ROOT=/datasets/Images/MFFI
export TCC_DATA_ROOT=/users/home/lucas.ocunha/research/TCC/data
# Salva os resultados do teste no /tmp local do nó para evitar estouro de cota em /projects
export TCC_MODELS_ROOT=/tmp/models_test_min
export TCC_OUTPUT_ROOT=/users/home/lucas.ocunha/research/TCC

mkdir -p "$TCC_MODELS_ROOT"
cd /users/home/lucas.ocunha/research/TCC

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
