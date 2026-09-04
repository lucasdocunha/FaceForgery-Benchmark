#!/bin/bash
# Submete todos os 6 modelos ao Slurm no CISIA
# Uso:
#   ./scripts/submit_all_cisia.sh [scratch|finetune] [workers_per_gpu]
#   ou diretamente de dentro de scripts/:
#   ./submit_all_cisia.sh [scratch|finetune] [workers_per_gpu]

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_DIR="$(cd "$SCRIPT_DIR/.." && pwd)"

REGIME="${1:-scratch}"
WORKERS="${2:-2}"

echo "=========================================================="
echo "Submetendo 6 jobs ao Slurm CISIA (Regime: $REGIME, Workers: $WORKERS)"
echo "Data: $(date)"
echo "=========================================================="

mkdir -p "$REPO_DIR/logs"
cd "$REPO_DIR"

FAMILIES=("resnet" "xception" "mobilenet" "vit" "clip" "dino")

for fam in "${FAMILIES[@]}"; do
    SCRIPT="$SCRIPT_DIR/slurm_${fam}.sh"
    if [ -f "$SCRIPT" ]; then
        echo "Submetendo $fam ($SCRIPT)..."
        sbatch "$SCRIPT" "$REGIME" "$WORKERS"
    else
        echo "Erro: script $SCRIPT não encontrado!"
    fi
done

echo "=========================================================="
echo "Todos os jobs foram despachados para a fila do Slurm!"
echo "Use 'squeue -u \$USER' para monitorar o status dos nós."
echo "=========================================================="
