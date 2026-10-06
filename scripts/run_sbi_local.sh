#!/usr/bin/env bash
set -euo pipefail

# ==============================================================================
# Execução Local: Self-Blended Images (SBI)
# Executa de forma isolada na GPU RTX 3090 (cuda:1).
# Treina MobileNet-V3-Large com representação SRM em 5 seeds canônicas.
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

# Configurações de Dados e Modelos
export TCC_DATA_ROOT="${REPO_DIR}/data"
export TCC_TRAIN_MANIFEST="${TCC_DATA_ROOT}/manifests/train.csv"
export TCC_VAL_MANIFEST="${TCC_DATA_ROOT}/manifests/val.csv"
export TCC_TRAIN_ROOT="${TCC_DATA_ROOT}/datasets/phase1/trainset"
export TCC_VAL_ROOT="${TCC_DATA_ROOT}/datasets/phase1/valset"
export TCC_PRETRAINED_ROOT="/home/lucas.ocunha/.cache/torch/hub/checkpoints"
export TCC_OUTPUT_ROOT="${REPO_DIR}/saidas/sbi"

DEVICE="${1:-cuda:1}"
ARM="${2:-sbi}" # sbi, mixed, or mffi
SEEDS=(42 123 2024 7 2025)

MODELS_BASE="/media/ssd2/lucas.ocunha/models-tcc/experimental/sbi"
mkdir -p "$MODELS_BASE" "$TCC_OUTPUT_ROOT" "${REPO_DIR}/logs"

CONFIG_FILE="${REPO_DIR}/configs/experimental/sbi_generic.yaml"

echo "=============================================================================="
echo "🚀 INICIANDO EXPERIMENTO 2: SELF-BLENDED IMAGES (SBI) - 5 SEEDS"
echo "Host     : $(hostname)"
echo "Device   : $DEVICE"
echo "Arm      : $ARM"
echo "Config   : $CONFIG_FILE"
echo "Destino  : $MODELS_BASE"
echo "Seeds    : ${SEEDS[*]}"
echo "Início   : $(date)"
echo "=============================================================================="

export TCC_SBI_ARM="$ARM"

for SEED in "${SEEDS[@]}"; do
    echo ""
    echo "=============================================================================="
    echo "▶️  [$(date +%T)] Treinando SBI ($ARM) | Seed $SEED..."
    echo "=============================================================================="
    export TCC_SEED="$SEED"
    RUN_DIR="${MODELS_BASE}/sbi_${ARM}_seed_${SEED}"
    CALIB_DIR="${RUN_DIR}/calibration"
    export TCC_RUN_DIR="$RUN_DIR"

    if [[ -f "${RUN_DIR}/best.pt" && -f "${CALIB_DIR}/calibration.json" ]]; then
        echo "⏭️  [$(date +%T)] SBI ($ARM) Seed $SEED já treinado e calibrado. Pulando..."
        continue
    fi

    if [[ ! -f "${RUN_DIR}/best.pt" ]]; then
        rm -rf "$RUN_DIR"
        "${PYTHON}" "${REPO_DIR}/research_cli.py" experimental train \
            --family sbi \
            --config "$CONFIG_FILE" \
            --seed "$SEED" \
            --output "$RUN_DIR" \
            --device "$DEVICE" \
            --execute
    fi

    rm -rf "$CALIB_DIR"
    "${PYTHON}" "${REPO_DIR}/research_cli.py" experimental calibrate \
        --family sbi \
        --run "$RUN_DIR" \
        --manifest "$TCC_VAL_MANIFEST" \
        --root "$TCC_VAL_ROOT" \
        --output "$CALIB_DIR" \
        --threshold-policy youden \
        --device "$DEVICE" \
        --execute

    echo "✅ [$(date +%T)] SBI ($ARM) Seed $SEED concluída com sucesso em: $RUN_DIR"
done

echo ""
echo "=============================================================================="
echo "🎉 PIPELINE SBI CONCLUÍDO EM TODAS AS 5 SEEDS: $(date)"
echo "=============================================================================="
