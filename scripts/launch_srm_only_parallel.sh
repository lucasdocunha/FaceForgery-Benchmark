#!/usr/bin/env bash
# ==============================================================================
# Campanha Paralela de Treinamento e Avaliação: SRM Puro (srm_only, 3 canais)
#
# Executa os 6 modelos nas 5 sementes canônicas (42, 123, 2024, 7, 2025)
# distribuídos perfeitamente entre as duas GPUs NVIDIA RTX 3090:
#
# GPU 0: CNNs (mobilenet, resnet, xception)  -> 15 modelos
# GPU 1: Transformers/Foundation (vit, clip, dino) -> 15 modelos
#
# Cada modelo treinado avalia automaticamente:
#   1. Validação (metrics_val.csv)
#   2. Teste Limpo (metrics_test.csv, outputs_test.npz)
#   3. Teste Difícil (metrics_test_d.csv, outputs_test_d.npz)
#   4. DF-40 (metrics_df40.csv, outputs_df40.npz)
#   5. Celeb-DF v2 (metrics_celeb_df.csv, metrics_celeb_df_video.csv)
# ==============================================================================

set -euo pipefail

REPO_DIR="/home/lucas.ocunha/tcc"
cd "$REPO_DIR"

mkdir -p "$REPO_DIR/logs" "$REPO_DIR/saidas/tables"

VENV_PY="$REPO_DIR/.venv/bin/python"

echo "=============================================================================="
echo "🚀 INICIANDO CAMPANHA FORENSE: SRM PURO (SRM_ONLY - 3 CANAIS DE RESÍDUO)"
echo "Data: $(date)"
echo "GPU 0: MobileNet, ResNet, Xception (Seeds: 42, 123, 2024, 7, 2025)"
echo "GPU 1: ViT, CLIP, DINO            (Seeds: 42, 123, 2024, 7, 2025)"
echo "=============================================================================="

# Launch GPU 0 in background
nohup "$VENV_PY" -u "$REPO_DIR/scripts/run_forensics_campaign.py" \
    --gpu 0 \
    --mode srm_only \
    --families "mobilenet,resnet,xception" \
    --seeds "42,123,2024,7,2025" \
    --epochs 15 \
    --batch-size 64 \
    --num-workers 4 \
    > "$REPO_DIR/logs/srm_only_gpu0.log" 2>&1 &

PID_GPU0=$!
echo "✅ GPU 0 iniciada com PID: $PID_GPU0 (Log: logs/srm_only_gpu0.log)"

# Launch GPU 1 in background
nohup "$VENV_PY" -u "$REPO_DIR/scripts/run_forensics_campaign.py" \
    --gpu 1 \
    --mode srm_only \
    --families "vit,clip,dino" \
    --seeds "42,123,2024,7,2025" \
    --epochs 15 \
    --batch-size 32 \
    --num-workers 4 \
    > "$REPO_DIR/logs/srm_only_gpu1.log" 2>&1 &

PID_GPU1=$!
echo "✅ GPU 1 iniciada com PID: $PID_GPU1 (Log: logs/srm_only_gpu1.log)"

echo "------------------------------------------------------------------------------"
echo "Processos em execução paralela. Monitore com:"
echo "  tail -f logs/srm_only_gpu0.log"
echo "  tail -f logs/srm_only_gpu1.log"
echo "=============================================================================="
