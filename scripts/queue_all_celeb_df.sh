#!/bin/bash
# Executa a avaliação de todos os modelos no Celeb-DF v2 distribuída entre GPU 0 e GPU 1
set -euo pipefail

REPO_DIR="/home/lucas.ocunha/tcc"
PYTHON="$REPO_DIR/.venv/bin/python"

mkdir -p "$REPO_DIR/logs"

echo "=== Iniciando fila de avaliação Celeb-DF v2 para TODAS as famílias ==="
echo "GPU 0: ResNet, MobileNet, Xception"
echo "GPU 1: CLIP, ViT, DINO"

# Worker GPU 0 (CNNs)
(
    for fam in resnet mobilenet xception; do
        echo "[GPU 0] [$(date)] Iniciando avaliação de $fam..."
        "$PYTHON" "$REPO_DIR/evaluate_celeb_df.py" --only-family "$fam" --device cuda:0 --batch-size 64 --skip-existing
        "$PYTHON" "$REPO_DIR/make_celeb_df_tables.py"
    done
    echo "[GPU 0] [$(date)] Todas as CNNs foram avaliadas!"
) >> "$REPO_DIR/logs/celeb_df_gpu0.log" 2>&1 &
PID_GPU0=$!

# Worker GPU 1 (Transformers / Foundation)
(
    for fam in clip vit dino; do
        echo "[GPU 1] [$(date)] Iniciando avaliação de $fam..."
        "$PYTHON" "$REPO_DIR/evaluate_celeb_df.py" --only-family "$fam" --device cuda:1 --batch-size 64 --skip-existing
        "$PYTHON" "$REPO_DIR/make_celeb_df_tables.py"
    done
    echo "[GPU 1] [$(date)] Todos os Transformers e DINO foram avaliados!"
) >> "$REPO_DIR/logs/celeb_df_gpu1.log" 2>&1 &
PID_GPU1=$!

echo "Fila iniciada com sucesso!"
echo "PID GPU 0: $PID_GPU0 (log: logs/celeb_df_gpu0.log)"
echo "PID GPU 1: $PID_GPU1 (log: logs/celeb_df_gpu1.log)"
