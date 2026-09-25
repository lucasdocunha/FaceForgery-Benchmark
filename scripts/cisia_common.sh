#!/bin/bash -l
# Sourced by sbatch entrypoints. SLURM runs a spool copy of the entrypoint;
# BASH_SOURCE therefore cannot be used to locate the repository in those files.
set -euo pipefail
CISIA_STARTED=$SECONDS
cisia_footer() {
    local status=$?
    echo "=== shell exit=$status | duration=$((SECONDS - CISIA_STARTED))s | $(date -Is)"
}
trap cisia_footer EXIT
: "${SLURM_JOB_ID:?Submit this script with sbatch, not bash on Shell Access}"
export TCC_PROJECT_ROOT="${TCC_PROJECT_ROOT:-${SLURM_SUBMIT_DIR:?Submit from the repository root}}"
cd "$TCC_PROJECT_ROOT"
[[ -f train.py && -d src/hpc ]] || { echo "Invalid checkout: $TCC_PROJECT_ROOT" >&2; exit 2; }
# A pre-created named environment is visible from both compute nodes.
CISIA_CONDA_ENV="${CISIA_CONDA_ENV:-tcc}"
[[ "$CISIA_CONDA_ENV" =~ ^[A-Za-z0-9_.-]+$ ]] || { echo "Use a conda environment name, not a path" >&2; exit 2; }
source /opt/conda/etc/profile.d/conda.sh
conda activate "$CISIA_CONDA_ENV"
[[ "$CONDA_PREFIX" == "$HOME/.conda/envs/"* ]] || {
    echo "Conda environment must be under $HOME/.conda/envs; got $CONDA_PREFIX" >&2; exit 2;
}
export PYTHONUNBUFFERED=1 PYTHONDONTWRITEBYTECODE=1
# exec gives the job runner direct ownership of SLURM's batch signals/status.
exec python -u -m src.hpc.runtime "$@"
