#!/usr/bin/env bash
set -e

REPO_DIR="/home/lucas.ocunha/tcc"
cd "$REPO_DIR"
mkdir -p logs

export LD_PRELOAD=/home/lucas.ocunha/.conda/envs/cae/lib/libstdc++.so.6
export PYTHONPATH=.
PYTHON_BIN=/home/lucas.ocunha/.conda/envs/cae/bin/python

# Families ordered from lightest to heaviest:
FAMILIES="mobilenet,resnet,xception,vit,clip,dino"
SEEDS="42,123,2024,7,2025"

echo "=== Launching Forensics Campaigns ==="
echo "Order: $FAMILIES"
echo "Seeds: $SEEDS"

# Launch GPU 0: SRM
nohup $PYTHON_BIN -u scripts/run_forensics_campaign.py \
    --gpu 0 \
    --mode srm \
    --families "$FAMILIES" \
    --seeds "$SEEDS" \
    --epochs 15 \
    --batch-size 64 \
    --num-workers 8 > logs/campaign_srm_gpu0.log 2>&1 </dev/null &

PID_SRM=$!
echo "GPU 0 (SRM) PID: $PID_SRM -> logs/campaign_srm_gpu0.log"

# Launch GPU 1: DTCWT
nohup $PYTHON_BIN -u scripts/run_forensics_campaign.py \
    --gpu 1 \
    --mode dtcwt \
    --families "$FAMILIES" \
    --seeds "$SEEDS" \
    --epochs 15 \
    --batch-size 64 \
    --num-workers 8 > logs/campaign_dtcwt_gpu1.log 2>&1 </dev/null &

PID_DTCWT=$!
echo "GPU 1 (DTCWT) PID: $PID_DTCWT -> logs/campaign_dtcwt_gpu1.log"

echo "Both campaigns successfully spawned in background!"
