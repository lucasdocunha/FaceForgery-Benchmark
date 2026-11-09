#!/bin/bash
cd /home/lucas.ocunha/tcc
while pgrep -f "run_matrix.py --regime finetune" > /dev/null; do sleep 30; done
echo "$(date): finetune concluido, disparando scratch" >> queue.log
source .env.local
PYTHONUNBUFFERED=1 nohup .venv/bin/python -u run_matrix.py --regime scratch --gpus 0,1 --workers-per-gpu 2 > run_scratch.log 2>&1 &
disown
SCRATCH_PID=$!
while pgrep -f "run_matrix.py --regime scratch" > /dev/null; do sleep 30; done
echo "$(date): scratch concluido - matriz completa" >> queue.log
