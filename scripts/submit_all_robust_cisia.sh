#!/bin/bash
# Submete jobs Slurm no CISIA para treinar os modelos robustos (5 seeds)
# Uso:
#   ./scripts/submit_all_robust_cisia.sh [families] [seeds] [regime] [workers]
# Exemplos:
#   ./scripts/submit_all_robust_cisia.sh
#   ./scripts/submit_all_robust_cisia.sh clip,dino
#   ./scripts/submit_all_robust_cisia.sh all "42,123,2024,7,2025" finetune_robust 8

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_DIR="$(cd "$SCRIPT_DIR/.." && pwd)"

FAMILIES_ARG="${1:-all}"
SEEDS="${2:-42,123,2024,7,2025}"
REGIME="${3:-finetune_robust}"
WORKERS="${4:-8}"

if [ "$FAMILIES_ARG" = "all" ] || [ -z "$FAMILIES_ARG" ]; then
    TARGET_FAMILIES=("resnet" "xception" "mobilenet" "vit" "clip" "dino")
else
    IFS=',' read -r -a TARGET_FAMILIES <<< "$FAMILIES_ARG"
fi

echo "=========================================================="
echo "🚀 SUBMETENDO BENCHMARK ROBUSTO AO SLURM CISIA"
echo "Famílias (${#TARGET_FAMILIES[@]}): ${TARGET_FAMILIES[*]}"
echo "Seeds: $SEEDS | Regime: $REGIME | Workers: $WORKERS"
echo "Data: $(date)"
echo "=========================================================="

mkdir -p "$REPO_DIR/logs"
cd "$REPO_DIR"

for fam in "${TARGET_FAMILIES[@]}"; do
    fam=$(echo "$fam" | tr -d '[:space:]')
    SCRIPT="$SCRIPT_DIR/slurm_robust_${fam}.sh"
    if [ -f "$SCRIPT" ]; then
        echo "Submetendo $fam ($SCRIPT)..."
        sbatch "$SCRIPT" "$SEEDS" "$REGIME" "$WORKERS"
    else
        echo "❌ Erro: Script $SCRIPT não encontrado!"
    fi
done

echo "=========================================================="
echo "✅ Todos os jobs solicitados foram despachados para a fila!"
echo "Para monitorar o status dos jobs, use:"
echo "   squeue -u \$USER"
echo "Para cancelar todos os seus jobs:"
echo "   scancel -u \$USER"
echo "=========================================================="
