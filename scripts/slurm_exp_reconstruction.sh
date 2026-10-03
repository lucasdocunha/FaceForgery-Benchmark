#!/bin/bash -l
#SBATCH --job-name=exp-cae
#SBATCH --partition=gpu
#SBATCH --output=logs/%x_%j.out
#SBATCH --error=logs/%x_%j.err
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=48:00:00
#SBATCH --mail-user=lucas.ocunha@ppgia.pucpr.br
#SBATCH --mail-type=ALL

# ==============================================================================
# Abordagem 1: Autoencoders de Reconstrução Espacial-Frequencial (CAE)
# Treina exclusivamente em rostos reais nas 5 sementes canônicas.
# ==============================================================================
set -euo pipefail

ROOT="${TCC_PROJECT_ROOT:-${SLURM_SUBMIT_DIR:-$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)}}"
cd "$ROOT"
mkdir -p "$ROOT/logs" "$ROOT/saidas"

# Ativação do ambiente Conda
if [[ -f /opt/conda/etc/profile.d/conda.sh ]]; then
    source /opt/conda/etc/profile.d/conda.sh
elif [[ -f "$HOME/miniconda3/etc/profile.d/conda.sh" ]]; then
    source "$HOME/miniconda3/etc/profile.d/conda.sh"
fi

CONDA_ENV_NAME="${CISIA_CONDA_ENV:-tcc-hpc}"
if conda info --envs | grep -q "^${CONDA_ENV_NAME}[[:space:]]"; then
    conda activate "$CONDA_ENV_NAME"
elif conda info --envs | grep -q "^tcc[[:space:]]"; then
    conda activate tcc
fi

# Caches locais efêmeros no scratch do nó
CACHE_DIR="${TMPDIR:-/scratch/$USER}/job_${SLURM_JOB_ID:-manual_$$}"
mkdir -p "$CACHE_DIR/hf" "$CACHE_DIR/torch" "$CACHE_DIR/pip" "$CACHE_DIR/tmp"
export HF_HOME="$CACHE_DIR/hf"
export TORCH_HOME="$CACHE_DIR/torch"
export PIP_CACHE_DIR="$CACHE_DIR/pip"
export TMPDIR="$CACHE_DIR/tmp"
export PYTHONUNBUFFERED=1
export PYTHONDONTWRITEBYTECODE=1
export TCC_SKIP_UNREADABLE=1

cleanup() {
    echo "Limpando diretório scratch temporário $CACHE_DIR..."
    rm -rf -- "$CACHE_DIR"
}
trap cleanup EXIT

# Variáveis do dataset e modelos
export TCC_DATASET_ROOT="${TCC_DATASET_ROOT:-/datasets/Images/MFFI}"
export TCC_DATA_ROOT="${TCC_DATA_ROOT:-$ROOT/data}"
export TCC_TRAIN_MANIFEST="${TCC_DATA_ROOT}/manifests/train.csv"
export TCC_VAL_MANIFEST="${TCC_DATA_ROOT}/manifests/val.csv"
export TCC_TRAIN_ROOT="${TCC_DATASET_ROOT}/trainset"
export TCC_VAL_ROOT="${TCC_DATASET_ROOT}/valset"
export TCC_OUTPUT_ROOT="${ROOT}/saidas/reconstruction"
mkdir -p "$TCC_OUTPUT_ROOT" "${TCC_DATA_ROOT}/manifests"

USER_MODELS_ROOT="/projects/models/$USER/faceforgery/experimental/reconstruction"
mkdir -p "$USER_MODELS_ROOT"

# Garante que os manifestos canônicos certificados com checksum sha256 existam
if [[ ! -f "$TCC_TRAIN_MANIFEST" || ! -f "${TCC_TRAIN_MANIFEST}.json" ]]; then
    echo "▶️  Gerando manifesto canônico certificado para train..."
    python research_cli.py convert-manifest \
        --source "${TCC_DATA_ROOT}/raw/train.csv" \
        --output "$TCC_TRAIN_MANIFEST" \
        --dataset mffi --split train --label-column target --convention fake-is-1
fi
if [[ ! -f "$TCC_VAL_MANIFEST" || ! -f "${TCC_VAL_MANIFEST}.json" ]]; then
    echo "▶️  Gerando manifesto canônico certificado para val..."
    python research_cli.py convert-manifest \
        --source "${TCC_DATA_ROOT}/raw/val.csv" \
        --output "$TCC_VAL_MANIFEST" \
        --dataset mffi --split val --label-column target --convention fake-is-1
fi

SEEDS=(42 123 2024 7 2025)
CONFIG_FILE="${ROOT}/configs/experimental/reconstruction/server_cae_standalone.yaml"

echo "=============================================================================="
echo "🚀 INICIANDO EXPERIMENTO 1: AUTOENCODER DE RECONSTRUÇÃO (CAE) - 5 SEEDS"
echo "Job ID   : ${SLURM_JOB_ID:-LOCAL}"
echo "Nó       : $(hostname)"
echo "Config   : $CONFIG_FILE"
echo "Seeds    : ${SEEDS[*]}"
echo "Data     : $(date)"
echo "=============================================================================="

for SEED in "${SEEDS[@]}"; do
    echo ""
    echo "------------------------------------------------------------------------------"
    echo "▶️  Executando CAE | Seed $SEED..."
    echo "------------------------------------------------------------------------------"
    export TCC_SEED="$SEED"
    export TCC_RUN_DIR="${USER_MODELS_ROOT}/cae_seed_${SEED}"
    mkdir -p "$TCC_RUN_DIR"

    # Treinamento do Autoencoder
    python research_cli.py experimental train \
        --family reconstruction \
        --config "$CONFIG_FILE" \
        --seed "$SEED" \
        --output "$TCC_RUN_DIR" \
        --device cuda:0 \
        --execute

    # Calibração do limiar ótimo Youden na validação
    CALIB_DIR="${TCC_RUN_DIR}/calibration"
    rm -rf "$CALIB_DIR"
    python research_cli.py experimental calibrate \
        --family reconstruction \
        --run "$TCC_RUN_DIR" \
        --manifest "$TCC_VAL_MANIFEST" \
        --root "$TCC_VAL_ROOT" \
        --output "$CALIB_DIR" \
        --threshold-policy youden \
        --device cuda:0 \
        --execute

    echo "✅ Seed $SEED concluída com sucesso! Modelo salvo em: $TCC_RUN_DIR"
done

echo ""
echo "=============================================================================="
echo "🎉 EXPERIMENTO CAE CONCLUÍDO EM TODAS AS 5 SEEDS: $(date)"
echo "=============================================================================="
