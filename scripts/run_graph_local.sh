#!/usr/bin/env bash
set -euo pipefail

# ==============================================================================
# Execução Local: Probing Estrutural e Redes Neurais em Grafos (GNNs & SupCon)
# Executa de forma isolada na GPU RTX 3090 (cuda:1).
# Treina GAT, GCN, GraphSAGE e SupCon nas 5 seeds canônicas.
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
export TCC_OUTPUT_ROOT="${REPO_DIR}/saidas/graphs"

DEVICE="${1:-cuda:1}"
SEEDS=(42 123 2024 7 2025)

MODELS_BASE="/media/ssd2/lucas.ocunha/models-tcc/experimental/graphs"
CACHE_STORAGE="${MODELS_BASE}/feature_cache"
mkdir -p "$MODELS_BASE" "$CACHE_STORAGE" "$TCC_OUTPUT_ROOT" "${REPO_DIR}/logs"

DINO_CHECKPOINT="/media/ssd2/lucas.ocunha/models-tcc/dino/srm/finetune_robust/seed_42/weights/best.pth"

echo "=============================================================================="
echo "🚀 INICIANDO EXPERIMENTO 3: GRAFOS E PROBING DE REPRESENTAÇÕES (GNNs)"
echo "Host     : $(hostname)"
echo "Device   : $DEVICE"
echo "Checkpoint: $DINO_CHECKPOINT"
echo "Destino  : $MODELS_BASE"
echo "Seeds    : ${SEEDS[*]}"
echo "Início   : $(date)"
echo "=============================================================================="

# 1. Extração do cache de features DINO-SRM (se ainda não existir)
TRAIN_CACHE_DIR=$(find "${CACHE_STORAGE}/train" -name "cache.json" -exec dirname {} \; 2>/dev/null | head -n 1 || true)
VAL_CACHE_DIR=$(find "${CACHE_STORAGE}/val" -name "cache.json" -exec dirname {} \; 2>/dev/null | head -n 1 || true)

if [[ -z "$TRAIN_CACHE_DIR" || ! -f "${TRAIN_CACHE_DIR}/cache.json" ]]; then
    echo "▶️  [$(date +%T)] Extraindo cache de embeddings DINO-SRM (Train)..."
    "${PYTHON}" "${REPO_DIR}/research_cli.py" experimental cache \
        --checkpoint "$DINO_CHECKPOINT" \
        --manifest "$TCC_TRAIN_MANIFEST" \
        --root "$TCC_TRAIN_ROOT" \
        --output "${CACHE_STORAGE}/train" \
        --device "$DEVICE" \
        --batch-size 64 \
        --workers 4 \
        --execute
    TRAIN_CACHE_DIR=$(find "${CACHE_STORAGE}/train" -name "cache.json" -exec dirname {} \; | head -n 1)
fi

if [[ -z "$VAL_CACHE_DIR" || ! -f "${VAL_CACHE_DIR}/cache.json" ]]; then
    echo "▶️  [$(date +%T)] Extraindo cache de embeddings DINO-SRM (Val)..."
    "${PYTHON}" "${REPO_DIR}/research_cli.py" experimental cache \
        --checkpoint "$DINO_CHECKPOINT" \
        --manifest "$TCC_VAL_MANIFEST" \
        --root "$TCC_VAL_ROOT" \
        --output "${CACHE_STORAGE}/val" \
        --device "$DEVICE" \
        --batch-size 64 \
        --workers 4 \
        --execute
    VAL_CACHE_DIR=$(find "${CACHE_STORAGE}/val" -name "cache.json" -exec dirname {} \; | head -n 1)
fi

echo "✅ Cache Train: $TRAIN_CACHE_DIR"
echo "✅ Cache Val  : $VAL_CACHE_DIR"

export TCC_TRAIN_CACHE="$TRAIN_CACHE_DIR"
export TCC_VAL_CACHE="$VAL_CACHE_DIR"
export TCC_DEVICE="$DEVICE"
export TCC_LABEL_FRACTION="1.0"
export TCC_GRAPH_BACKEND="pyg"

# 2. Treinamento das GNNs e Probes Métricos em 5 Seeds
for SEED in "${SEEDS[@]}"; do
    echo ""
    echo "=============================================================================="
    echo "▶️  [$(date +%T)] Executando Probing em Grafos | Seed $SEED..."
    echo "=============================================================================="
    export TCC_SEED="$SEED"

    # SupCon (Supervised Contrastive Learning)
    echo "▶️  [$(date +%T)] Treinando SupCon | Seed $SEED..."
    RUN_DIR="${MODELS_BASE}/metric_supcon_seed_${SEED}"
    export TCC_RUN_DIR="$RUN_DIR"
    export TCC_METRIC_KIND="supcon"
    rm -rf "$RUN_DIR"
    "${PYTHON}" "${REPO_DIR}/research_cli.py" experimental train \
        --family metric \
        --config "${REPO_DIR}/configs/experimental/metric_srm.yaml" \
        --seed "$SEED" \
        --output "$RUN_DIR" \
        --device "$DEVICE" \
        --execute

    CALIB_DIR="${RUN_DIR}/calibration"
    rm -rf "$CALIB_DIR"
    "${PYTHON}" "${REPO_DIR}/research_cli.py" experimental calibrate \
        --family metric \
        --run "$RUN_DIR" \
        --manifest "$TCC_VAL_MANIFEST" \
        --root "$TCC_VAL_ROOT" \
        --output "$CALIB_DIR" \
        --options "{\"cache\": \"${VAL_CACHE_DIR}\"}" \
        --device "$DEVICE" \
        --execute

    # GAT (Graph Attention Network)
    echo "▶️  [$(date +%T)] Treinando GAT | Seed $SEED..."
    RUN_DIR="${MODELS_BASE}/graph_gat_seed_${SEED}"
    export TCC_RUN_DIR="$RUN_DIR"
    export TCC_GRAPH_KIND="gat"
    rm -rf "$RUN_DIR"
    "${PYTHON}" "${REPO_DIR}/research_cli.py" experimental train \
        --family graph \
        --config "${REPO_DIR}/configs/experimental/graph_srm.yaml" \
        --seed "$SEED" \
        --output "$RUN_DIR" \
        --device "$DEVICE" \
        --execute

    CALIB_DIR="${RUN_DIR}/calibration"
    rm -rf "$CALIB_DIR"
    "${PYTHON}" "${REPO_DIR}/research_cli.py" experimental calibrate \
        --family graph \
        --run "$RUN_DIR" \
        --manifest "$TCC_VAL_MANIFEST" \
        --root "$TCC_VAL_ROOT" \
        --output "$CALIB_DIR" \
        --options "{\"cache\": \"${VAL_CACHE_DIR}\"}" \
        --device "$DEVICE" \
        --execute

    # GCN (Graph Convolutional Network)
    echo "▶️  [$(date +%T)] Treinando GCN | Seed $SEED..."
    RUN_DIR="${MODELS_BASE}/graph_gcn_seed_${SEED}"
    export TCC_RUN_DIR="$RUN_DIR"
    export TCC_GRAPH_KIND="gcn"
    rm -rf "$RUN_DIR"
    "${PYTHON}" "${REPO_DIR}/research_cli.py" experimental train \
        --family graph \
        --config "${REPO_DIR}/configs/experimental/graph_srm.yaml" \
        --seed "$SEED" \
        --output "$RUN_DIR" \
        --device "$DEVICE" \
        --execute

    CALIB_DIR="${RUN_DIR}/calibration"
    rm -rf "$CALIB_DIR"
    "${PYTHON}" "${REPO_DIR}/research_cli.py" experimental calibrate \
        --family graph \
        --run "$RUN_DIR" \
        --manifest "$TCC_VAL_MANIFEST" \
        --root "$TCC_VAL_ROOT" \
        --output "$CALIB_DIR" \
        --options "{\"cache\": \"${VAL_CACHE_DIR}\"}" \
        --device "$DEVICE" \
        --execute

    # GraphSAGE
    echo "▶️  [$(date +%T)] Treinando GraphSAGE | Seed $SEED..."
    RUN_DIR="${MODELS_BASE}/graph_sage_seed_${SEED}"
    export TCC_RUN_DIR="$RUN_DIR"
    export TCC_GRAPH_KIND="sage"
    rm -rf "$RUN_DIR"
    "${PYTHON}" "${REPO_DIR}/research_cli.py" experimental train \
        --family graph \
        --config "${REPO_DIR}/configs/experimental/graph_srm.yaml" \
        --seed "$SEED" \
        --output "$RUN_DIR" \
        --device "$DEVICE" \
        --execute

    CALIB_DIR="${RUN_DIR}/calibration"
    rm -rf "$CALIB_DIR"
    "${PYTHON}" "${REPO_DIR}/research_cli.py" experimental calibrate \
        --family graph \
        --run "$RUN_DIR" \
        --manifest "$TCC_VAL_MANIFEST" \
        --root "$TCC_VAL_ROOT" \
        --output "$CALIB_DIR" \
        --options "{\"cache\": \"${VAL_CACHE_DIR}\"}" \
        --device "$DEVICE" \
        --execute

    echo "✅ [$(date +%T)] Seed $SEED concluída com sucesso para todas as GNNs e Probes Métricos!"
done

echo ""
echo "=============================================================================="
echo "🎉 PIPELINE DE GRAFOS CONCLUÍDO EM TODAS AS 5 SEEDS EM: $(date)"
echo "=============================================================================="
