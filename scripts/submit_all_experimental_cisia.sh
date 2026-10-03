#!/bin/bash
# ==============================================================================
# Disparador em Lote: Todas as 5 Novas Abordagens Experimentais no CISIA
# Uso: bash scripts/submit_all_experimental_cisia.sh
# ==============================================================================
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
export TCC_PROJECT_ROOT="$ROOT"
mkdir -p "$ROOT/logs" "$ROOT/saidas"
cd "$ROOT"

echo "=============================================================================="
echo "🚀 SUBMETENDO TODAS AS 5 ABORDAGENS EXPERIMENTAIS AO SLURM (5 SEEDS CADA)"
echo "Diretório Raiz: $ROOT"
echo "Data/Hora     : $(date)"
echo "=============================================================================="

# 1. Autoencoder de Reconstrução Espacial-Frequencial (CAE)
JOB_CAE=$(sbatch --parsable --chdir="$ROOT" "scripts/slurm_exp_reconstruction.sh")
echo "▶️  [1/5] Reconstrução (CAE) submetido com Job ID: $JOB_CAE"

# 2. Probing Estrutural e Grafos (GNNs: GCN, GAT, GraphSAGE + SupCon)
JOB_GRAPH=$(sbatch --parsable --chdir="$ROOT" "scripts/slurm_exp_graph.sh")
echo "▶️  [2/5] Grafos & GNNs submetido com Job ID: $JOB_GRAPH"

# 3. Self-Blended Images (SBI)
JOB_SBI=$(sbatch --parsable --chdir="$ROOT" "scripts/slurm_exp_sbi.sh")
echo "▶️  [3/5] Self-Blended Images (SBI) submetido com Job ID: $JOB_SBI"

# 4. Mixture of Experts (MoE Fusão Multi-Especialista)
JOB_MOE=$(sbatch --parsable --chdir="$ROOT" "scripts/slurm_exp_moe.sh")
echo "▶️  [4/5] MoE Multi-Especialista submetido com Job ID: $JOB_MOE"

# 5. Vision-Language Models (VLM Forensics)
JOB_VLM=$(sbatch --parsable --chdir="$ROOT" "scripts/slurm_exp_vlm.sh")
echo "▶️  [5/5] Vision-Language Models (VLM) submetido com Job ID: $JOB_VLM"

echo "=============================================================================="
echo "🎉 Todos os 5 experimentos foram enfileirados com sucesso no SLURM!"
echo "Acompanhe com: squeue -u \$USER"
echo "=============================================================================="
