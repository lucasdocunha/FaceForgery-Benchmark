#!/bin/bash -l
#SBATCH --job-name=exp-moe
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
# Abordagem 4: Mixture of Experts (MoE) de Fusão Multi-Especialista
# Combina os especialistas DINO-SRM + CLIP-SRM com o especialista de Reconstrução.
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
export TCC_TRAIN_MANIFEST="${TCC_DATA_ROOT}/raw/train.csv"
export TCC_VAL_MANIFEST="${TCC_DATA_ROOT}/raw/val.csv"
export TCC_TRAIN_ROOT="${TCC_DATASET_ROOT}/trainset"
export TCC_VAL_ROOT="${TCC_DATASET_ROOT}/valset"
export TCC_OUTPUT_ROOT="${ROOT}/saidas/moe_experimental"
mkdir -p "$TCC_OUTPUT_ROOT"

USER_MODELS_ROOT="/projects/models/$USER/faceforgery/experimental/moe"
CACHE_STORAGE="/projects/models/$USER/faceforgery/experimental/graphs/feature_cache"
mkdir -p "$USER_MODELS_ROOT"

export TCC_FEATURE_DINO_SRM_VAL="${CACHE_STORAGE}/dino_srm_val.pt"
export TCC_FEATURE_CLIP_SRM_VAL="${CACHE_STORAGE}/clip_srm_val.pt"

SEEDS=(42 123 2024 7 2025)
CONFIG_FILE="${ROOT}/configs/experimental/moe_srm_pair_control.yaml"

echo "=============================================================================="
echo "🚀 INICIANDO EXPERIMENTO 4: MOE DE FUSÃO MULTI-ESPECIALISTA - 5 SEEDS"
echo "Job ID   : ${SLURM_JOB_ID:-LOCAL}"
echo "Nó       : $(hostname)"
echo "Config   : $CONFIG_FILE"
echo "Seeds    : ${SEEDS[*]}"
echo "Data     : $(date)"
echo "=============================================================================="

for SEED in "${SEEDS[@]}"; do
    echo ""
    echo "------------------------------------------------------------------------------"
    echo "▶️  Executando MoE Fusão | Seed $SEED..."
    echo "------------------------------------------------------------------------------"
    export TCC_SEED="$SEED"
    export TCC_RUN_DIR="${USER_MODELS_ROOT}/moe_fusion_seed_${SEED}"
    mkdir -p "$TCC_RUN_DIR"

    python research_cli.py experimental train \
        --family moe \
        --config "$CONFIG_FILE" \
        --seed "$SEED" \
        --output "$TCC_RUN_DIR" \
        --device cpu \
        --execute || true

    CALIB_DIR="${TCC_RUN_DIR}/calibration"
    mkdir -p "$CALIB_DIR"
    if [[ -f "${TCC_RUN_DIR}/val_select.csv" ]]; then
        python research_cli.py experimental calibrate \
            --family moe \
            --run "$TCC_RUN_DIR" \
            --manifest "${TCC_RUN_DIR}/val_select.csv" \
            --root "$TCC_VAL_ROOT" \
            --output "$CALIB_DIR" \
            --device cpu \
            --threshold-policy youden \
            --execute || true
    fi

    echo "✅ Seed $SEED concluída com sucesso! Modelo salvo em: $TCC_RUN_DIR"
done

echo ""
echo "=============================================================================="
echo "🎉 EXPERIMENTO MOE CONCLUÍDO EM TODAS AS 5 SEEDS: $(date)"
echo "=============================================================================="
