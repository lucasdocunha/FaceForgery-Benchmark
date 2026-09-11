#!/usr/bin/env python3
"""
Orquestrador em 2 Fases para Treinamento de Modelos Robustos (seed=987, fourier=none):
- FASE 1 (Novos modelos solicitados primeiro):
    - GPU 0: ResNet (25 épocas)
    - GPU 1: MobileNet (20 épocas) -> Xception (20 épocas)
- FASE 2 (Modelos que sofreram interrupção/erro):
    - GPU 0: DINO (20 épocas, com bfloat16 e lr estável)
    - GPU 1: ViT (25 épocas)
- FASE 3: Ensemble automático consolidando os 6 modelos robustos e gerando tabelas/gráficos.
"""
from __future__ import annotations

import os
import subprocess
import sys
import time
from pathlib import Path

TCC = Path("/home/lucas.ocunha/tcc")
PYTHON = str(TCC / ".venv/bin/python")
LOG_DIR = TCC / "logs"
LOG_DIR.mkdir(exist_ok=True)

# Códigos das subprocessos da FASE 1
PHASE1_GPU0_CODE = """
import os, sys
os.environ['CUDA_VISIBLE_DEVICES'] = '0'
sys.path.insert(0, '/home/lucas.ocunha/tcc')
from scripts._robust_common import run_gpu_group
run_gpu_group(0, [
    ('resnet', 'configs/resnet.yaml', 32, 25),
], tag='phase1')
"""

PHASE1_GPU1_CODE = """
import os, sys
os.environ['CUDA_VISIBLE_DEVICES'] = '1'
sys.path.insert(0, '/home/lucas.ocunha/tcc')
from scripts._robust_common import run_gpu_group
run_gpu_group(1, [
    ('mobilenet', 'configs/mobilenet.yaml', 32, 20),
    ('xception',  'configs/xception.yaml',  32, 20),
], tag='phase1')
"""

# Códigos das subprocessos da FASE 2
PHASE2_GPU0_CODE = """
import os, sys
os.environ['CUDA_VISIBLE_DEVICES'] = '0'
sys.path.insert(0, '/home/lucas.ocunha/tcc')
from scripts._robust_common import run_gpu_group
run_gpu_group(0, [
    ('dino', 'configs/dino.yaml', 16, 20),
], tag='phase2')
"""

PHASE2_GPU1_CODE = """
import os, sys
os.environ['CUDA_VISIBLE_DEVICES'] = '1'
sys.path.insert(0, '/home/lucas.ocunha/tcc')
from scripts._robust_common import run_gpu_group
run_gpu_group(1, [
    ('vit', 'configs/vit.yaml', 32, 25),
], tag='phase2')
"""


def launch(code: str, log_path: Path, env_gpu: str):
    env = os.environ.copy()
    env["CUDA_VISIBLE_DEVICES"] = env_gpu
    env["PYTHONUNBUFFERED"] = "1"
    f = open(log_path, "a", buffering=1)
    proc = subprocess.Popen(
        [PYTHON, "-u", "-c", code],
        stdout=f,
        stderr=subprocess.STDOUT,
        env=env,
        cwd=str(TCC),
    )
    return proc, f


def wait_group(p0, p1, f0, f1, desc: str):
    t0 = time.time()
    print(f"\n--- [INÍCIO {desc}] PIDs: GPU0={p0.pid} | GPU1={p1.pid} ---", flush=True)
    with open(LOG_DIR / "robust_main.log", "a") as mlog:
        mlog.write(f"[{time.strftime('%Y-%m-%d %H:%M:%S')}] INÍCIO {desc} (GPU0 PID={p0.pid}, GPU1 PID={p1.pid})\n")

    while True:
        r0 = p0.poll()
        r1 = p1.poll()
        elapsed = (time.time() - t0) / 60
        status_str = f"[{elapsed:.0f}min] {desc} -> GPU0={'DONE' if r0 is not None else 'RUNNING'} | GPU1={'DONE' if r1 is not None else 'RUNNING'}"
        print(status_str, flush=True)
        with open(LOG_DIR / "robust_main.log", "a") as mlog:
            mlog.write(status_str + "\n")
        if r0 is not None and r1 is not None:
            break
        time.sleep(120)

    f0.close()
    f1.close()
    print(f"✅ {desc} Concluída. GPU0 exit={r0} | GPU1 exit={r1}", flush=True)
    with open(LOG_DIR / "robust_main.log", "a") as mlog:
        mlog.write(f"[{time.strftime('%Y-%m-%d %H:%M:%S')}] FIM {desc}. GPU0 exit={r0} | GPU1 exit={r1}\n")
    return r0, r1


def main():
    start_total = time.time()
    print("=" * 70, flush=True)
    print("🚀 INICIANDO PIPELINE ROBUSTO EM 2 FASES", flush=True)
    print("=" * 70, flush=True)

    # -------------------------------------------------------------
    # FASE 1: MobileNet, ResNet e Xception
    # -------------------------------------------------------------
    print("\n📦 [FASE 1] Lançando MobileNet, ResNet e Xception...", flush=True)
    p0, f0 = launch(PHASE1_GPU0_CODE, LOG_DIR / "robust_gpu0.log", "0")
    p1, f1 = launch(PHASE1_GPU1_CODE, LOG_DIR / "robust_gpu1.log", "1")
    r0, r1 = wait_group(p0, p1, f0, f1, "FASE 1 (ResNet / MobileNet+Xception)")

    if r0 != 0 or r1 != 0:
        print(f"⚠️ Atenção: Fase 1 terminou com código diferente de zero: GPU0={r0}, GPU1={r1}", flush=True)

    # -------------------------------------------------------------
    # FASE 2: DINO e ViT (modelos que sofreram erro/interrupção)
    # -------------------------------------------------------------
    print("\n📦 [FASE 2] Lançando DINO e ViT...", flush=True)
    p0_2, f0_2 = launch(PHASE2_GPU0_CODE, LOG_DIR / "robust_gpu0.log", "0")
    p1_2, f1_2 = launch(PHASE2_GPU1_CODE, LOG_DIR / "robust_gpu1.log", "1")
    r0_2, r1_2 = wait_group(p0_2, p1_2, f0_2, f1_2, "FASE 2 (DINO / ViT)")

    if r0_2 != 0 or r1_2 != 0:
        print(f"⚠️ Atenção: Fase 2 terminou com código diferente de zero: GPU0={r0_2}, GPU1={r1_2}", flush=True)

    # -------------------------------------------------------------
    # FASE 3: Ensemble de todos os 6 modelos
    # -------------------------------------------------------------
    print("\n🔗 [FASE 3] Rodando ensemble de todos os 6 modelos robustos...", flush=True)
    with open(LOG_DIR / "robust_ensemble.log", "w") as fe:
        ens_proc = subprocess.run(
            [PYTHON, "-u", "scripts/ensemble_robust_all.py"],
            stdout=fe,
            stderr=subprocess.STDOUT,
            cwd=str(TCC),
            env={**os.environ, "PYTHONUNBUFFERED": "1"},
        )
    print(f"Ensemble finalizado com exit={ens_proc.returncode}", flush=True)

    total_min = (time.time() - start_total) / 60
    print(f"\n🎉 PIPELINE ROBUSTO COMPLETO EM {total_min:.1f} MINUTOS!", flush=True)


if __name__ == "__main__":
    main()
