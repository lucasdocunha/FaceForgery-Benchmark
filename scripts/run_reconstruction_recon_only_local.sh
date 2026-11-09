#!/usr/bin/env bash
set -euo pipefail

# ==============================================================================
# Execução Local: ResNet-18 treinada na Face Reconstruída pelo CAE (recon_only)
# Avalia as 5 seeds utilizando os Autoencoders pré-treinados correspondentes.
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
export TCC_OUTPUT_ROOT="${REPO_DIR}/saidas/reconstruction_recon_only"
export TCC_RESNET18_WEIGHTS="/home/lucas.ocunha/.cache/torch/hub/checkpoints/resnet18-f37072fd.pth"

AE_BASE="/media/ssd2/lucas.ocunha/models-tcc/experimental/reconstruction"
MODELS_BASE="/media/ssd2/lucas.ocunha/models-tcc/experimental/reconstruction_recon_only"
mkdir -p "$MODELS_BASE" "$TCC_OUTPUT_ROOT" "${REPO_DIR}/logs"

CONFIG_FILE="${REPO_DIR}/configs/experimental/reconstruction/server_cae_recon_only.yaml"
DEVICE="${1:-cuda:0}"
SEEDS=(42 123 2024 7 2025)

echo "=============================================================================="
echo "🚀 INICIANDO EXPERIMENTO RESNET-18 (RECON_ONLY) - 5 SEEDS"
echo "Host     : $(hostname)"
echo "Device   : $DEVICE"
echo "Config   : $CONFIG_FILE"
echo "Destino  : $MODELS_BASE"
echo "Início   : $(date)"
echo "=============================================================================="

for SEED in "${SEEDS[@]}"; do
    AE_DIR="${AE_BASE}/cae_seed_${SEED}"
    
    # Aguarda caso o Autoencoder da seed ainda esteja treinando
    while [[ ! -f "${AE_DIR}/status.json" ]] || ! grep -q '"state": "complete"' "${AE_DIR}/status.json" 2>/dev/null; do
        echo "⏳ [$(date +%T)] Aguardando conclusão do Autoencoder da Seed $SEED em $AE_DIR..."
        sleep 60
    done

    echo ""
    echo "------------------------------------------------------------------------------"
    echo "▶️  [$(date +%T)] Treinando ResNet-18 (recon_only) | Seed $SEED..."
    echo "------------------------------------------------------------------------------"
    
    RUN_DIR="${MODELS_BASE}/recon_only_seed_${SEED}"
    export TCC_RUN_DIR="$RUN_DIR"
    export TCC_AE_RUN="$AE_DIR"
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

    echo "✅ [$(date +%T)] ResNet-18 recon_only Seed $SEED concluída em: $RUN_DIR"
done

echo ""
echo "=============================================================================="
echo "🎉 PIPELINE RESNET-18 RECON_ONLY CONCLUÍDO EM TODAS AS 5 SEEDS EM: $(date)"
echo "=============================================================================="
