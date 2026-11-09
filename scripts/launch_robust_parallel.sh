#!/bin/bash
# Lança o orquestrador em 2 fases (Fase 1: MobileNet, ResNet, Xception -> Fase 2: DINO, ViT -> Fase 3: Ensemble)
cd /home/lucas.ocunha/tcc
mkdir -p logs

echo "[$(date)] Iniciando Orquestrador Robusto em 2 Fases..." >> logs/robust_launch.log
nohup .venv/bin/python -u scripts/run_parallel_robust.py > logs/robust_main.log 2>&1 &
MAIN_PID=$!
echo $MAIN_PID > /tmp/robust_main_pid.txt
echo "[$(date)] Orquestrador iniciado com PID $MAIN_PID" >> logs/robust_launch.log
echo "🚀 Orquestrador Robusto iniciado em background. PID: $MAIN_PID"
echo "Acompanhe o progresso com: tail -f logs/robust_main.log"
