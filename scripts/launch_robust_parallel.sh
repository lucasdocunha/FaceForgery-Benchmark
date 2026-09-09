#!/bin/bash
# Lança os dois grupos de treino em paralelo nas GPUs 0 e 1
cd /home/lucas.ocunha/tcc
mkdir -p logs

echo "[$(date)] Iniciando GPU0 (CLIP + DINO + ResNet)" >> logs/robust_launch.log
CUDA_VISIBLE_DEVICES=0 PYTHONUNBUFFERED=1 \
  .venv/bin/python -u -c "
import os, sys
os.environ['CUDA_VISIBLE_DEVICES'] = '0'
sys.path.insert(0, '/home/lucas.ocunha/tcc')
from scripts._robust_common import run_gpu_group
run_gpu_group(0, [
    ('clip',   'configs/clip.yaml',   16, 20),
    ('dino',   'configs/dino.yaml',   16, 20),
    ('resnet', 'configs/resnet.yaml', 32, 25),
])
" >> logs/robust_gpu0.log 2>&1 &
GPU0_PID=$!
echo "[$(date)] GPU0 PID=$GPU0_PID" >> logs/robust_launch.log

echo "[$(date)] Iniciando GPU1 (ViT + MobileNet + Xception)" >> logs/robust_launch.log
CUDA_VISIBLE_DEVICES=1 PYTHONUNBUFFERED=1 \
  .venv/bin/python -u -c "
import os, sys
os.environ['CUDA_VISIBLE_DEVICES'] = '1'
sys.path.insert(0, '/home/lucas.ocunha/tcc')
from scripts._robust_common import run_gpu_group
run_gpu_group(1, [
    ('vit',       'configs/vit.yaml',       32, 25),
    ('mobilenet', 'configs/mobilenet.yaml', 32, 20),
    ('xception',  'configs/xception.yaml',  32, 20),
])
" >> logs/robust_gpu1.log 2>&1 &
GPU1_PID=$!
echo "[$(date)] GPU1 PID=$GPU1_PID" >> logs/robust_launch.log

echo "GPU0 PID=$GPU0_PID | GPU1 PID=$GPU1_PID"
echo "$GPU0_PID $GPU1_PID" > /tmp/robust_pids.txt

# Aguarda os dois terminarem
wait $GPU0_PID
GPU0_CODE=$?
echo "[$(date)] GPU0 terminou com código $GPU0_CODE" >> logs/robust_launch.log

wait $GPU1_PID
GPU1_CODE=$?
echo "[$(date)] GPU1 terminou com código $GPU1_CODE" >> logs/robust_launch.log

# Roda o ensemble final
echo "[$(date)] Rodando ensemble de todos os modelos robustos..." >> logs/robust_launch.log
PYTHONUNBUFFERED=1 .venv/bin/python -u scripts/ensemble_robust_all.py \
  >> logs/robust_ensemble.log 2>&1
echo "[$(date)] Ensemble concluído." >> logs/robust_launch.log
echo "=== TUDO CONCLUÍDO ==="
