#!/usr/bin/env python3
"""
Orquestrador que lança os 2 grupos de treino em subprocessos paralelos,
com redirecionamento de stdout/stderr para arquivos de log separados.
Ao final, roda o ensemble automático.
"""
import subprocess, sys, time
from pathlib import Path

TCC = Path("/home/lucas.ocunha/tcc")
PYTHON = str(TCC / ".venv/bin/python")
LOG_DIR = TCC / "logs"
LOG_DIR.mkdir(exist_ok=True)

GPU0_CODE = """
import os, sys
os.environ['CUDA_VISIBLE_DEVICES'] = '0'
sys.path.insert(0, '/home/lucas.ocunha/tcc')
from scripts._robust_common import run_gpu_group
run_gpu_group(0, [
    ('clip',   'configs/clip.yaml',   16, 20),
    ('dino',   'configs/dino.yaml',   16, 20),
    ('resnet', 'configs/resnet.yaml', 32, 25),
])
"""

GPU1_CODE = """
import os, sys
os.environ['CUDA_VISIBLE_DEVICES'] = '1'
sys.path.insert(0, '/home/lucas.ocunha/tcc')
from scripts._robust_common import run_gpu_group
run_gpu_group(1, [
    ('vit',       'configs/vit.yaml',       32, 25),
    ('mobilenet', 'configs/mobilenet.yaml', 32, 20),
    ('xception',  'configs/xception.yaml',  32, 20),
])
"""

def launch(code, log_path, env_gpu):
    import os
    env = os.environ.copy()
    env["CUDA_VISIBLE_DEVICES"] = env_gpu
    env["PYTHONUNBUFFERED"] = "1"
    f = open(log_path, "w", buffering=1)
    proc = subprocess.Popen(
        [PYTHON, "-u", "-c", code],
        stdout=f, stderr=subprocess.STDOUT,
        env=env, cwd=str(TCC),
    )
    return proc, f

print("🚀 Lançando GPU0 (CLIP + DINO + ResNet)...", flush=True)
p0, f0 = launch(GPU0_CODE, LOG_DIR / "robust_gpu0.log", "0")

print("🚀 Lançando GPU1 (ViT + MobileNet + Xception)...", flush=True)
p1, f1 = launch(GPU1_CODE, LOG_DIR / "robust_gpu1.log", "1")

print(f"PIDs: GPU0={p0.pid}  GPU1={p1.pid}", flush=True)

# Monitor simples
t0 = time.time()
while True:
    r0 = p0.poll()
    r1 = p1.poll()
    elapsed = (time.time() - t0) / 60
    print(f"[{elapsed:.0f}min] GPU0={'DONE' if r0 is not None else 'RUNNING'}  GPU1={'DONE' if r1 is not None else 'RUNNING'}", flush=True)
    if r0 is not None and r1 is not None:
        break
    time.sleep(120)  # checa a cada 2 min

f0.close(); f1.close()
print(f"\n✅ Ambas as GPUs concluídas. GPU0 exit={r0}  GPU1 exit={r1}", flush=True)

# Roda ensemble automaticamente
print("\n🔗 Rodando ensemble de todos os modelos robustos...", flush=True)
with open(LOG_DIR / "robust_ensemble.log", "w") as fe:
    result = subprocess.run(
        [PYTHON, "-u", "scripts/ensemble_robust_all.py"],
        stdout=fe, stderr=subprocess.STDOUT, cwd=str(TCC),
        env={**__import__("os").environ, "PYTHONUNBUFFERED": "1"},
    )
print(f"Ensemble concluído (exit={result.returncode})", flush=True)

# Exibe resumo final dos logs
print("\n=== RESUMO GPU0 ===")
lines = open(LOG_DIR / "robust_gpu0.log").readlines()
for l in lines[-20:]: print(l, end="")
print("\n=== RESUMO GPU1 ===")
lines = open(LOG_DIR / "robust_gpu1.log").readlines()
for l in lines[-20:]: print(l, end="")
print("\n=== ENSEMBLE ===")
lines = open(LOG_DIR / "robust_ensemble.log").readlines()
for l in lines[-30:]: print(l, end="")
