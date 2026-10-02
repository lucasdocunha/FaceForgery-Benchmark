#!/bin/bash -l
#SBATCH --job-name=tcc-evaluation
#SBATCH --partition=gpu
#SBATCH --output=logs/%x_%j.out
#SBATCH --error=logs/%x_%j.err
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --time=06:00:00
# Submit from the repository root with one reviewed suite YAML path.
set -euo pipefail
ROOT="${TCC_PROJECT_ROOT:-${SLURM_SUBMIT_DIR:?Use sbatch from the repository root}}"
[[ $# == 1 ]] || { echo "Usage: sbatch scripts/slurm_evaluation.sh config.yaml" >&2; exit 2; }
source "$ROOT/scripts/cisia_common.sh" evaluate "$1"
