"""Fila assíncrona multi-GPU em Python para processar todas as famílias no Celeb-DF v2."""

import subprocess
import sys
import time
from pathlib import Path
from concurrent.futures import ThreadPoolExecutor

REPO_DIR = Path(__file__).resolve().parent.parent
PYTHON = sys.executable

def is_family_running(fam: str) -> bool:
    try:
        out = subprocess.check_output(["pgrep", "-f", f"only-family {fam}"]).decode()
        # Ignora o próprio processo
        lines = [line.strip() for line in out.strip().split("\n") if line.strip()]
        return len(lines) > 0
    except subprocess.CalledProcessError:
        return False

def run_family_on_gpu(family: str, gpu_id: int):
    cmd = [
        PYTHON,
        str(REPO_DIR / "evaluate_celeb_df.py"),
        "--only-family", family,
        "--device", f"cuda:{gpu_id}",
        "--batch-size", "64",
        "--skip-existing",
    ]
    print(f"[Worker GPU {gpu_id}] [{time.strftime('%X')}] Iniciando {family}...", flush=True)
    res = subprocess.run(cmd)
    print(f"[Worker GPU {gpu_id}] [{time.strftime('%X')}] {family} finalizado com código {res.returncode}", flush=True)
    
    # Atualiza as tabelas
    subprocess.run([PYTHON, str(REPO_DIR / "make_celeb_df_tables.py")])
    return family, res.returncode

def worker_gpu0():
    print("[GPU 0] Aguardando MobileNet (já em execução)...", flush=True)
    while is_family_running("mobilenet"):
        time.sleep(10)
    print("[GPU 0] MobileNet concluído! Disparando Xception...", flush=True)
    run_family_on_gpu("xception", 0)
    print("[GPU 0] Xception concluído! Verificando ResNet (scratch restante)...", flush=True)
    run_family_on_gpu("resnet", 0)
    print("=== [GPU 0] Todas as tarefas de CNN concluídas! ===", flush=True)

def worker_gpu1():
    print("[GPU 1] Aguardando ViT (já em execução)...", flush=True)
    while is_family_running("vit"):
        time.sleep(10)
    print("[GPU 1] ViT concluído! Disparando DINOv3...", flush=True)
    run_family_on_gpu("dino", 1)
    print("=== [GPU 1] Todas as tarefas de Transformers/DINO concluídas! ===", flush=True)

def main():
    print("Iniciando orquestrador contínuo para avaliar TODAS as famílias nas 2 GPUs...", flush=True)
    with ThreadPoolExecutor(max_workers=2) as executor:
        f0 = executor.submit(worker_gpu0)
        f1 = executor.submit(worker_gpu1)
        f0.result()
        f1.result()
    print("=== TODOS OS 225 MODELOS FORAM AVALIADOS NO CELEB-DF V2 ===", flush=True)
    subprocess.run([PYTHON, str(REPO_DIR / "make_celeb_df_tables.py")])

if __name__ == "__main__":
    main()
