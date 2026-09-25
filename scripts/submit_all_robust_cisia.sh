#!/bin/bash
# Usage: bash scripts/submit_all_robust_cisia.sh [all|families] [seeds] [regime] [loader-workers]
set -euo pipefail
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
FAMILIES="${1:-all}"
SEEDS="${2:-42,123,2024,7,2025}"
REGIME="${3:-finetune_robust}"
WORKERS="${4:-4}"
[[ $# -le 4 && "$SEEDS" =~ ^[0-9]+(,[0-9]+)*$ && "$REGIME" =~ ^(scratch_robust|finetune_robust)$ && "$WORKERS" =~ ^[0-7]$ ]] || {
    echo "Invalid arguments: use integer seeds, scratch_robust|finetune_robust and 0..7 loader workers" >&2; exit 2;
}
[[ "$FAMILIES" != all ]] || FAMILIES="resnet,xception,mobilenet,vit,clip,dino"
[[ "$FAMILIES" =~ ^(resnet|xception|mobilenet|vit|clip|dino)(,(resnet|xception|mobilenet|vit|clip|dino))*$ ]] || {
    echo "Unknown/empty family in: $FAMILIES" >&2; exit 2;
}
IFS=',' read -r -a families <<< "$FAMILIES"
# Validate every family before submitting anything. Never print success for sbatch failure.
export TCC_PROJECT_ROOT="$ROOT"
mkdir -p "$ROOT/logs"
cd "$ROOT"
for family in "${families[@]}"; do
    job=$(sbatch --parsable --chdir="$ROOT" "scripts/slurm_robust_${family}.sh" "$SEEDS" "$REGIME" "$WORKERS")
    echo "$family submitted: $job"
done
