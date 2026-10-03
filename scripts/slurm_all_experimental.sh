#!/bin/bash -l
#SBATCH --job-name=exp-all
#SBATCH --partition=gpu
#SBATCH --output=logs/%x_%j.out
#SBATCH --error=logs/%x_%j.err
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=72:00:00
#SBATCH --mail-user=lucas.ocunha@ppgia.pucpr.br
#SBATCH --mail-type=ALL

# ==============================================================================
# Execução Sequencial Monolítica de Todas as 5 Abordagens Experimentais
# Ideal para alocação única contínua no SLURM.
# ==============================================================================
set -euo pipefail

ROOT="${TCC_PROJECT_ROOT:-${SLURM_SUBMIT_DIR:-$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)}}"
cd "$ROOT"
mkdir -p "$ROOT/logs" "$ROOT/saidas"

echo "=============================================================================="
echo "🚀 INICIANDO SUÍTE COMPLETA DAS 5 ABORDAGENS EXPERIMENTAIS (ALOCAÇÃO ÚNICA)"
echo "Job ID   : ${SLURM_JOB_ID:-LOCAL}"
echo "Nó       : $(hostname)"
echo "Data     : $(date)"
echo "=============================================================================="

echo ">>> [1/5] Executando Autoencoders de Reconstrução (CAE)..."
bash scripts/slurm_exp_reconstruction.sh || true

echo ">>> [2/5] Executando Probing em Grafos e GNNs..."
bash scripts/slurm_exp_graph.sh || true

echo ">>> [3/5] Executando Self-Blended Images (SBI)..."
bash scripts/slurm_exp_sbi.sh || true

echo ">>> [4/5] Executando MoE de Fusão Multi-Especialista..."
bash scripts/slurm_exp_moe.sh || true

echo ">>> [5/5] Executando Vision-Language Models (VLM Forensics)..."
bash scripts/slurm_exp_vlm.sh || true

echo "=============================================================================="
echo "🎉 TODAS AS 5 ABORDAGENS EXPERIMENTAIS CONCLUÍDAS COM SUCESSO EM: $(date)"
echo "=============================================================================="
