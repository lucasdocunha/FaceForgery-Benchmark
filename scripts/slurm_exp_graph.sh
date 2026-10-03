#!/bin/bash -l
#SBATCH --job-name=exp-graph
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
# Abordagem 3: Probing Estrutural e Redes Neurais em Grafos (GNNs & SupCon)
# Treina GCN, GAT, GraphSAGE, SupCon, Centroid e KNN nas 5 sementes canônicas.
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

# Variáveis de Dados e Modelos
export TCC_DATASET_ROOT="${TCC_DATASET_ROOT:-/datasets/Images/MFFI}"
export TCC_DATA_ROOT="${TCC_DATA_ROOT:-$ROOT/data}"
export TCC_TRAIN_MANIFEST="${TCC_DATA_ROOT}/manifests/train.csv"
export TCC_VAL_MANIFEST="${TCC_DATA_ROOT}/manifests/val.csv"
export TCC_TRAIN_ROOT="${TCC_DATASET_ROOT}/trainset"
export TCC_VAL_ROOT="${TCC_DATASET_ROOT}/valset"
export TCC_OUTPUT_ROOT="${ROOT}/saidas/graphs"
mkdir -p "$TCC_OUTPUT_ROOT" "${TCC_DATA_ROOT}/manifests"

USER_MODELS_ROOT="/projects/models/$USER/faceforgery/experimental/graphs"
CACHE_STORAGE="${USER_MODELS_ROOT}/feature_cache"
mkdir -p "$USER_MODELS_ROOT" "$CACHE_STORAGE"

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

echo "=============================================================================="
echo "🚀 INICIANDO EXPERIMENTO 3: GRAFOS E PROBING DE REPRESENTAÇÕES (GNNs) - 5 SEEDS"
echo "Job ID   : ${SLURM_JOB_ID:-LOCAL}"
echo "Nó       : $(hostname)"
echo "Seeds    : ${SEEDS[*]}"
echo "Data     : $(date)"
echo "=============================================================================="

# 1. Localização ou extração do cache de features DINO-SRM
DINO_CHECKPOINT="${TCC_PRETRAINED_ROOT:-/datasets/ai_models/faceforgery/pretrained}/dino/srm/best.pth"
if [[ ! -f "$DINO_CHECKPOINT" ]]; then
    # Fallback para pesos de usuário se existirem
    CANDIDATE=$(find /projects/models/$USER -name "best.pth" | grep "dino" | head -n 1 || true)
    if [[ -n "$CANDIDATE" && -f "$CANDIDATE" ]]; then
        DINO_CHECKPOINT="$CANDIDATE"
    fi
fi

TRAIN_CACHE="${CACHE_STORAGE}/dino_srm_train.pt"
VAL_CACHE="${CACHE_STORAGE}/dino_srm_val.pt"

if [[ ! -f "$TRAIN_CACHE" || ! -f "$VAL_CACHE" ]]; then
    echo "▶️  Extraindo cache de embeddings DINO-SRM na GPU..."
    if [[ -f "$DINO_CHECKPOINT" ]]; then
        python research_cli.py experimental cache \
            --checkpoint "$DINO_CHECKPOINT" \
            --manifest "$TCC_TRAIN_MANIFEST" \
            --root "$TCC_TRAIN_ROOT" \
            --output "${CACHE_STORAGE}/train_extracted" \
            --device cuda:0 --batch-size 32 --workers 4 --execute || true
        python research_cli.py experimental cache \
            --checkpoint "$DINO_CHECKPOINT" \
            --manifest "$TCC_VAL_MANIFEST" \
            --root "$TCC_VAL_ROOT" \
            --output "${CACHE_STORAGE}/val_extracted" \
            --device cuda:0 --batch-size 32 --workers 4 --execute || true
    fi
fi

# 2. Execução das Sementes de Probing em Grafos
for SEED in "${SEEDS[@]}"; do
    echo ""
    echo "------------------------------------------------------------------------------"
    echo "▶️  Executando Probing em Grafos | Seed $SEED..."
    echo "------------------------------------------------------------------------------"
    export TCC_SEED="$SEED"
    export TCC_DEVICE="cuda:0"
    export TCC_LABEL_FRACTION="1.0"
    export TCC_TRAIN_CACHE="${TRAIN_CACHE}"
    export TCC_VAL_CACHE="${VAL_CACHE}"

    # SupCon & Probes Métricos
    export TCC_RUN_DIR="${USER_MODELS_ROOT}/metric_supcon_seed_${SEED}"
    export TCC_METRIC_KIND="supcon"
    mkdir -p "$TCC_RUN_DIR"
    python research_cli.py experimental train \
        --family metric \
        --config configs/experimental/metric_srm.yaml \
        --seed "$SEED" \
        --output "$TCC_RUN_DIR" \
        --device cuda:0 \
        --execute || true

    # GNNs: GAT (Graph Attention Networks)
    export TCC_RUN_DIR="${USER_MODELS_ROOT}/graph_gat_seed_${SEED}"
    export TCC_GRAPH_KIND="gat"
    mkdir -p "$TCC_RUN_DIR"
    python research_cli.py experimental train \
        --family graph \
        --config configs/experimental/graph_srm.yaml \
        --seed "$SEED" \
        --output "$TCC_RUN_DIR" \
        --device cuda:0 \
        --execute || true

    # GNNs: GCN (Graph Convolutional Networks)
    export TCC_RUN_DIR="${USER_MODELS_ROOT}/graph_gcn_seed_${SEED}"
    export TCC_GRAPH_KIND="gcn"
    mkdir -p "$TCC_RUN_DIR"
    python research_cli.py experimental train \
        --family graph \
        --config configs/experimental/graph_srm.yaml \
        --seed "$SEED" \
        --output "$TCC_RUN_DIR" \
        --device cuda:0 \
        --execute || true

    # GNNs: GraphSAGE
    export TCC_RUN_DIR="${USER_MODELS_ROOT}/graph_sage_seed_${SEED}"
    export TCC_GRAPH_KIND="sage"
    mkdir -p "$TCC_RUN_DIR"
    python research_cli.py experimental train \
        --family graph \
        --config configs/experimental/graph_srm.yaml \
        --seed "$SEED" \
        --output "$TCC_RUN_DIR" \
        --device cuda:0 \
        --execute || true

    echo "✅ Seed $SEED concluída com sucesso para todas as GNNs e Probes Métricos!"
done

echo ""
echo "=============================================================================="
echo "🎉 EXPERIMENTO DE GRAFOS CONCLUÍDO EM TODAS AS 5 SEEDS: $(date)"
echo "=============================================================================="
