#!/usr/bin/env bash
set -euo pipefail

REPO_DIR="/home/lucas/projetos/TCC"
cd "${REPO_DIR}"
PYTHON="${REPO_DIR}/.venv/bin/python"
PID=2462782

echo "[$(date)] Monitorando treinamento do MoE Frequency (PID ${PID})..."

# Aguardar enquanto o PID de treino estiver ativo
while kill -0 "${PID}" 2>/dev/null; do
    sleep 60
done

echo "[$(date)] Processo de treinamento (PID ${PID}) finalizado!"
sleep 15

# Verificar se o treino gerou metrics_test.csv
RESULTS_DIR="${REPO_DIR}/models/moe_frequency/concat_frequency/scratch/seed_42/results"
if [ ! -f "${RESULTS_DIR}/metrics_test.csv" ]; then
    echo "[$(date)] ERRO: metrics_test.csv não encontrado em ${RESULTS_DIR}. O treino pode ter falhado."
    exit 1
fi

echo "[$(date)] Treino e teste normal concluídos com sucesso!"
echo "[$(date)] Iniciando avaliação no conjunto difícil (test_d)..."

"${PYTHON}" evaluate.py \
    --models-root "${REPO_DIR}/models" \
    --data-dir "${REPO_DIR}/data/raw" \
    --splits test_d \
    --test-d-csv "${REPO_DIR}/data/raw/test.csv" \
    --test-d-images-dir "${REPO_DIR}/data/datasets/phase1/test_d" \
    --only-model-family moe_frequency \
    --batch-size 64 \
    --num-workers 4

echo "[$(date)] Avaliação de test_d concluída!"

echo "[$(date)] Atualizando tabelas com make_tables.py..."
"${PYTHON}" make_tables.py

echo "[$(date)] Fazendo upload dos pesos e resultados para o Hugging Face..."
"${PYTHON}" -c "
from huggingface_hub import HfApi
token = os.environ.get('HF_TOKEN') or (open('.hf_token').read().strip() if os.path.exists('.hf_token') else None)
api = HfApi(token=token)
info = api.upload_folder(
    folder_path='${REPO_DIR}/models/moe_frequency/concat_frequency/scratch/seed_42',
    path_in_repo='moe_frequency/concat_frequency/scratch/seed_42',
    repo_id='lucasoc/MFFI-Models',
    repo_type='model',
    ignore_patterns=['**/final.pth', '**/*.npz'],
    commit_message='feat(models): upload 7-expert MoE Frequency weights and evaluation results',
)
print('Upload HF concluído com sucesso:', info)
"

echo "[$(date)] Atualizando repositório Git..."
git fetch origin ICLR || true
git pull --rebase origin ICLR || true
git add results/tables/
git commit -m "feat(tables): add 7-expert MoE Frequency test and test_d evaluation metrics" || true
git push origin ICLR || true

echo "================================================================"
echo "[$(date)] PIPELINE COMPLETO DO MOE FREQUENCY FINALIZADO COM SUCESSO!"
echo "================================================================"
