#!/bin/bash -l
#SBATCH --job-name=preload-models
#SBATCH --partition=gpu
#SBATCH --output=logs/%x_%j.out
#SBATCH --error=logs/%x_%j.err
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --time=04:00:00
#SBATCH --mail-user=lucas.ocunha@ppgia.pucpr.br
#SBATCH --mail-type=ALL
# Submit from the repository root: mkdir -p logs && sbatch scripts/slurm_download_pretrained.sh
set -euo pipefail
ROOT="${TCC_PROJECT_ROOT:-${SLURM_SUBMIT_DIR:?Use sbatch from the repository root}}"
source "$ROOT/scripts/cisia_common.sh" pretrained "$@"
