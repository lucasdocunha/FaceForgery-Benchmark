#!/bin/bash -l
#SBATCH --job-name=tcc-resnet
#SBATCH --partition=gpu
#SBATCH --output=logs/%x_%j.out
#SBATCH --error=logs/%x_%j.err
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=48:00:00
#SBATCH --mail-user=lucas.ocunha@ppgia.pucpr.br
#SBATCH --mail-type=ALL
# Submit from the repository root: mkdir -p logs && sbatch scripts/slurm_resnet.sh
set -euo pipefail
ROOT="${TCC_PROJECT_ROOT:-${SLURM_SUBMIT_DIR:?Use sbatch from the repository root}}"
source "$ROOT/scripts/cisia_common.sh" matrix resnet "$@"
