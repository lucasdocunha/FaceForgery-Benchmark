#!/usr/bin/env bash
set -euo pipefail

export PYTHONUNBUFFERED=1

REPO_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "${REPO_DIR}"

PYTHON=""
if [[ -n "${CONDA_PREFIX:-}" && -f "$CONDA_PREFIX/bin/python" ]]; then
    PYTHON="$CONDA_PREFIX/bin/python"
elif [[ -f "${REPO_DIR}/.venv/bin/python" ]]; then
    PYTHON="${REPO_DIR}/.venv/bin/python"
elif [[ -f "$HOME/.conda/envs/cae/bin/python" ]]; then
    PYTHON="$HOME/.conda/envs/cae/bin/python"
else
    PYTHON="$(command -v python3)"
fi

LOGS_DIR="${REPO_DIR}/logs"
mkdir -p "${LOGS_DIR}"

echo "================================================================"
echo "Iniciando pipeline v2 dos modelos MoE Robustos em: $(date)"
echo "Diretório do repositório: ${REPO_DIR}"
echo "GPU:"
nvidia-smi --query-gpu=name,memory.total,memory.free --format=csv,noheader || true
echo "================================================================"

echo ""
echo ">>> [1/4] Iniciando treinamento MoE Frequency v2 (Robusto: Spatial + Fourier FFT)..."
"${PYTHON}" "${REPO_DIR}/train.py" --config "${REPO_DIR}/configs/moe_frequency_robust.yaml" "$@"

echo ""
echo ">>> [2/4] Avaliando MoE Frequency v2 no teste difícil (test_d)..."
"${PYTHON}" "${REPO_DIR}/evaluate.py" \
    --data-dir "${REPO_DIR}/data/raw" \
    --splits test_d \
    --test-d-csv "${REPO_DIR}/data/raw/test.csv" \
    --test-d-images-dir "${REPO_DIR}/data/datasets/phase1/test_d" \
    --only-model-family moe_frequency \
    --batch-size 64 \
    --num-workers 4 \
    --skip-existing

echo ""
echo ">>> [3/4] Iniciando treinamento MoE Standard v2 (Robusto: Spatial RGB)..."
"${PYTHON}" "${REPO_DIR}/train.py" --config "${REPO_DIR}/configs/moe_standard_robust.yaml" "$@"

echo ""
echo ">>> [4/4] Avaliando MoE Standard v2 no teste difícil (test_d)..."
"${PYTHON}" "${REPO_DIR}/evaluate.py" \
    --data-dir "${REPO_DIR}/data/raw" \
    --splits test_d \
    --test-d-csv "${REPO_DIR}/data/raw/test.csv" \
    --test-d-images-dir "${REPO_DIR}/data/datasets/phase1/test_d" \
    --only-model-family moe_standard \
    --batch-size 64 \
    --num-workers 4 \
    --skip-existing

echo ""
echo ">>> Atualizando tabelas com make_tables.py..."
"${PYTHON}" "${REPO_DIR}/make_tables.py"

echo ""
echo ">>> Fazendo upload dos novos modelos para o Hugging Face..."
"${PYTHON}" -c "
import os
from pathlib import Path
from huggingface_hub import HfApi
from src.data.paths import models_root

token = os.environ.get('HF_TOKEN') or (open('.hf_token').read().strip() if os.path.exists('.hf_token') else None)
if token:
    api = HfApi(token=token)
    m_root = models_root()
    for rel_path in [
        'moe_frequency/concat_frequency/scratch_robust/seed_42',
        'moe_standard/none/scratch_robust/seed_42',
    ]:
        model_path = m_root / rel_path
        if model_path.exists():
            print(f'Uploading {model_path} -> {rel_path}...')
            info = api.upload_folder(
                folder_path=str(model_path),
                path_in_repo=rel_path,
                repo_id='lucasoc/MFFI-Models',
                repo_type='model',
                ignore_patterns=['**/final.pth', '**/*.npz'],
                commit_message=f'feat(models): upload {rel_path} robust weights and evaluation results',
            )
            print('Upload concluído:', info)
else:
    print('HF_TOKEN não encontrado, pulando upload no Hugging Face.')
"

echo ""
echo ">>> Sincronizando tabelas com o Git..."
git fetch origin ICLR || true
git pull --rebase origin ICLR || true
git add results/tables/
git commit -m "feat(tables): add MoE v2 robust benchmark results for frequency and standard" || true
git push origin ICLR || true

echo ""
echo "================================================================"
echo "Pipeline completo dos dois modelos MoE Robustos (v2) concluído em: $(date)"
echo "================================================================"
