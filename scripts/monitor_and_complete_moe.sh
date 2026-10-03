#!/usr/bin/env bash
# ==============================================================================
# Monitor e Encadeador Automático do Pipeline MoE Robusto v2
# ==============================================================================
set -euo pipefail

REPO_DIR="/home/lucas.ocunha/tcc"
cd "$REPO_DIR"

PYTHON="$REPO_DIR/.venv/bin/python"
LOGS_DIR="$REPO_DIR/logs"
mkdir -p "$LOGS_DIR"

MONITOR_LOG="$LOGS_DIR/moe_pipeline_orchestrator.log"

exec >> "$MONITOR_LOG" 2>&1

echo "================================================================"
echo "Iniciando monitoramento do treinamento MoE em: $(date)"
echo "================================================================"

# 1. Aguarda o processo atual do moe_frequency finalizar
while pgrep -f "train.py --config configs/moe_frequency_robust.yaml" > /dev/null; do
    sleep 30
done

echo ">>> [1/4 Concluído] MoE Frequency finalizou o treinamento em $(date)"

# 2. Avaliação do MoE Frequency no test_d
echo ">>> [2/4] Avaliando MoE Frequency no split difícil (test_d)..."
"$PYTHON" "$REPO_DIR/evaluate.py" \
    --data-dir "$REPO_DIR/data/raw" \
    --splits test_d \
    --test-d-csv "$REPO_DIR/data/raw/test.csv" \
    --test-d-images-dir "$REPO_DIR/data/datasets/phase1/test_d" \
    --only-model-family moe_frequency \
    --batch-size 64 \
    --num-workers 4 \
    --skip-existing

# 3. Treinamento do MoE Standard (RGB Puro)
echo ">>> [3/4] Iniciando treinamento do MoE Standard (Spatial RGB)..."
"$PYTHON" "$REPO_DIR/train.py" --config "$REPO_DIR/configs/moe_standard_robust.yaml"

# 4. Avaliação do MoE Standard no test_d
echo ">>> [4/4] Avaliando MoE Standard no split difícil (test_d)..."
"$PYTHON" "$REPO_DIR/evaluate.py" \
    --data-dir "$REPO_DIR/data/raw" \
    --splits test_d \
    --test-d-csv "$REPO_DIR/data/raw/test.csv" \
    --test-d-images-dir "$REPO_DIR/data/datasets/phase1/test_d" \
    --only-model-family moe_standard \
    --batch-size 64 \
    --num-workers 4 \
    --skip-existing

# 5. Atualização das Tabelas Oficiais
echo ">>> Consolidando tabelas com make_tables.py..."
"$PYTHON" "$REPO_DIR/make_tables.py"

# 6. Upload dos modelos para o Hugging Face
echo ">>> Publicando novos pesos no Hugging Face (lucasoc/MFFI-Models)..."
"$PYTHON" -c "
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

# 7. Sincronização Git
echo ">>> Sincronizando tabelas com o Git origin/ICLR..."
git fetch origin ICLR || true
git pull --rebase origin ICLR || true
git add results/tables/
git commit -m "feat(tables): add MoE v2 robust benchmark results for frequency and standard" || true
git push origin ICLR || true

echo "================================================================"
echo "Pipeline completo dos dois modelos MoE concluído com sucesso em: $(date)"
echo "================================================================"
