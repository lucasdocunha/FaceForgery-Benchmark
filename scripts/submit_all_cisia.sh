#!/bin/bash
# Usage: bash scripts/submit_all_cisia.sh [scratch|finetune] [workers-per-gpu]
set -euo pipefail
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
REGIME="${1:-scratch}"
WORKERS="${2:-1}"
[[ $# -le 2 && "$REGIME" =~ ^(scratch|finetune)$ && "$WORKERS" =~ ^[1-8]$ ]] || {
    echo "Expected scratch|finetune and workers-per-gpu from 1 to 8" >&2; exit 2;
}
export TCC_PROJECT_ROOT="$ROOT"
mkdir -p "$ROOT/logs"
cd "$ROOT"
for family in resnet xception mobilenet vit clip dino; do
    job=$(sbatch --parsable --chdir="$ROOT" "scripts/slurm_${family}.sh" "$REGIME" "$WORKERS")
    echo "$family submitted: $job"
done
