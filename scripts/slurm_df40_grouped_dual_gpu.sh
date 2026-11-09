#!/bin/bash -l
#SBATCH --job-name=DF40-Fast
#SBATCH --partition=dev
#SBATCH --gres=gpu:2
#SBATCH --cpus-per-task=16
#SBATCH --mem=48G
#SBATCH --output=logs/df40_grouped_%j.out
#SBATCH --error=logs/df40_grouped_%j.err

set -euo pipefail
cd /home/lucas.ocunha/tcc

PYTHON="/home/lucas.ocunha/.conda/envs/cae/bin/python"

echo "=============================================================================="
echo "🚀 INICIANDO INFERÊNCIA AGRUPADA DF-40 (ALL SEEDS - 2x RTX 3090)"
echo "Node: $(hostname) | Data: $(date)"
echo "CUDA_VISIBLE_DEVICES: ${CUDA_VISIBLE_DEVICES:-none}"
echo "=============================================================================="

mkdir -p tables/df40_all_fakes_models logs

# Inicia Worker 0 na GPU 0
"$PYTHON" -u scripts/run_df40_grouped_seeds.py --worker-id 0 --device cuda:0 > logs/df40_grouped_worker0.log 2>&1 &
PID0=$!

# Inicia Worker 1 na GPU 1
"$PYTHON" -u scripts/run_df40_grouped_seeds.py --worker-id 1 --device cuda:1 > logs/df40_grouped_worker1.log 2>&1 &
PID1=$!

echo "Worker 0 iniciado (PID $PID0, cuda:0) -> logs/df40_grouped_worker0.log"
echo "Worker 1 iniciado (PID $PID1, cuda:1) -> logs/df40_grouped_worker1.log"

wait $PID0
wait $PID1

echo "=============================================================================="
echo "🎉 AMBOS OS WORKERS FINALIZADOS COM SUCESSO!"
echo "Consolidando matrizes de estatísticas (Mean ± Std)..."
"$PYTHON" scripts/build_df40_seeds_summary.py
echo "=============================================================================="
