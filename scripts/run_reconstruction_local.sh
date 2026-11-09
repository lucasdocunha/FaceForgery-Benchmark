#!/usr/bin/env bash
set -euo pipefail

# ==============================================================================
# Execução Local: Autoencoder de Reconstrução Forense (CAE) - 5 Seeds
# Executa de forma isolada na GPU RTX 3090 (cuda:0).
# ==============================================================================

REPO_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "${REPO_DIR}"

PYTHON="${REPO_DIR}/.venv/bin/python"
if [[ ! -f "$PYTHON" ]]; then
    PYTHON="$(command -v python3)"
fi

export PYTHONUNBUFFERED=1
export PYTHONDONTWRITEBYTECODE=1
export TCC_SKIP_UNREADABLE=1

# Variáveis do dataset e modelos
export TCC_DATA_ROOT="${REPO_DIR}/data"
export TCC_TRAIN_MANIFEST="${TCC_DATA_ROOT}/manifests/train.csv"
export TCC_VAL_MANIFEST="${TCC_DATA_ROOT}/manifests/val.csv"
export TCC_TRAIN_ROOT="${TCC_DATA_ROOT}/datasets/phase1/trainset"
export TCC_VAL_ROOT="${TCC_DATA_ROOT}/datasets/phase1/valset"
export TCC_OUTPUT_ROOT="${REPO_DIR}/saidas/reconstruction"

MODELS_BASE="/media/ssd2/lucas.ocunha/models-tcc/experimental/reconstruction"
mkdir -p "$MODELS_BASE" "$TCC_OUTPUT_ROOT" "${REPO_DIR}/logs"

CONFIG_FILE="${REPO_DIR}/configs/experimental/reconstruction/server_cae_standalone.yaml"
DEVICE="cuda:0"
SEEDS=(42 123 2024 7 2025)

echo "=============================================================================="
echo "🚀 INICIANDO EXPERIMENTO RECONSTRUCTION CAE - 5 SEEDS (LOCAL)"
echo "Host     : $(hostname)"
echo "Device   : $DEVICE"
echo "Config   : $CONFIG_FILE"
echo "Destino  : $MODELS_BASE"
echo "Seeds    : ${SEEDS[*]}"
echo "Início   : $(date)"
echo "=============================================================================="

for SEED in "${SEEDS[@]}"; do
    echo ""
    echo "------------------------------------------------------------------------------"
    echo "▶️  [$(date +%T)] Treinando Autoencoder CAE | Seed $SEED..."
    echo "------------------------------------------------------------------------------"
    
    RUN_DIR="${MODELS_BASE}/cae_seed_${SEED}"
    export TCC_RUN_DIR="$RUN_DIR"
    export TCC_SEED="$SEED"
    rm -rf "$RUN_DIR"
    mkdir -p "$RUN_DIR"

    "${PYTHON}" "${REPO_DIR}/research_cli.py" experimental train \
        --family reconstruction \
        --config "$CONFIG_FILE" \
        --seed "$SEED" \
        --output "$RUN_DIR" \
        --device "$DEVICE" \
        --execute

    CALIB_DIR="${RUN_DIR}/calibration"
    rm -rf "$CALIB_DIR"

    echo "▶️  [$(date +%T)] Calibrando threshold Youden para Seed $SEED..."
    "${PYTHON}" "${REPO_DIR}/research_cli.py" experimental calibrate \
        --family reconstruction \
        --run "$RUN_DIR" \
        --manifest "$TCC_VAL_MANIFEST" \
        --root "$TCC_VAL_ROOT" \
        --output "$CALIB_DIR" \
        --threshold-policy youden \
        --batch-size 32 \
        --device "$DEVICE" \
        --execute

    echo "✅ [$(date +%T)] Seed $SEED concluída com sucesso em: $RUN_DIR"
done

echo ""
echo "=============================================================================="
echo "🎉 EXPERIMENTO RECONSTRUCTION CAE CONCLUÍDO EM TODAS AS 5 SEEDS EM: $(date)"
echo "=============================================================================="
