#!/usr/bin/env bash
set -euo pipefail

export PYTHONUNBUFFERED=1

REPO_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
PYTHON="${REPO_DIR}/.venv/bin/python"
LOGS_DIR="${REPO_DIR}/logs"
mkdir -p "${LOGS_DIR}"

echo "================================================================"
echo "Iniciando pipeline dos dois modelos MoE em: $(date)"
echo "Diretório do repositório: ${REPO_DIR}"
echo "Argumentos extras: $*"
echo "================================================================"

echo ""
echo ">>> [1/2] Iniciando treinamento MoE Standard (Spatial RGB)..."
"${PYTHON}" "${REPO_DIR}/train.py" --config "${REPO_DIR}/configs/moe_standard.yaml" "$@"

echo ""
echo ">>> [2/2] Iniciando treinamento MoE Frequency (Spatial + Fourier FFT)..."
"${PYTHON}" "${REPO_DIR}/train.py" --config "${REPO_DIR}/configs/moe_frequency.yaml" "$@"

echo ""
echo "================================================================"
echo "Treinamento dos dois modelos MoE concluído com sucesso em: $(date)"
echo "================================================================"
