#!/usr/bin/env python3
"""
Fila dinâmica de inferência exaustiva do DF-40 para 3 sementes (seed_42, seed_123, seed_2024).
Estratégia:
- Round 1: Todas as 60 configurações para seed_42
- Round 2: Todas as 60 configurações para seed_123
- Round 3: Todas as 60 configurações para seed_2024
"""

from __future__ import annotations
import argparse
import os
import subprocess
import sys
import time
from pathlib import Path

ROOT_DIR = Path(__file__).resolve().parent.parent
PYTHON_EXEC = sys.executable

FAMILIES = [
    ("resnet", 512),
    ("mobilenet", 512),
    ("xception", 256),
    ("vit", 256),
    ("clip", 256),
    ("dino", 256),
]

METHODS = [
    ("rgb", "none", "finetune"),
    ("rgb_robust", "none", "finetune_robust"),
    ("srm_only", "srm_only", "finetune_robust"),
    ("srm", "srm", "finetune_robust"),
    ("magnitude", "magnitude", "finetune"),
    ("phase", "phase", "finetune"),
    ("complex", "complex", "finetune"),
    ("frequency_3", "frequency_3", "finetune"),
    ("concat", "concat", "finetune"),
    ("concat_freq", "concat_frequency", "finetune"),
]

SEEDS = ["seed_42", "seed_123", "seed_2024"]
MODELS_DIR = Path("/media/ssd2/lucas.ocunha/models-tcc")
TABLES_DIR = ROOT_DIR / "tables" / "df40_all_fakes_models"


def get_all_tasks():
    tasks = []
    # Loop externo em SEEDS para garantir que a seed_42 rode 100% primeiro!
    for seed in SEEDS:
        seed_tasks = []
        for fam, bs in FAMILIES:
            for label, mode, regime in METHODS:
                ckpt = MODELS_DIR / fam / mode / regime / seed / "weights" / "best.pth"
                if not ckpt.exists():
                    continue
                run_name = f"{fam}_{label}_{seed}"
                seed_tasks.append({
                    "run_name": run_name,
                    "family": fam,
                    "label": label,
                    "mode": mode,
                    "regime": regime,
                    "seed": seed,
                    "checkpoint": str(ckpt),
                    "batch_size": bs,
                })
        tasks.extend(seed_tasks)
    return tasks


def main():
    parser = argparse.ArgumentParser(description="Worker de Fila DF-40")
    parser.add_argument("--worker-id", type=int, default=0, help="ID do worker (0 ou 1)")
    parser.add_argument("--device", type=str, default="cuda:0", help="Dispositivo PyTorch (cuda:0, cuda:1)")
    args = parser.parse_args()

    TABLES_DIR.mkdir(parents=True, exist_ok=True)
    all_tasks = get_all_tasks()

    # Organizar por seed
    tasks_by_seed = {s: [t for t in all_tasks if t["seed"] == s] for s in SEEDS}

    print(f"[{time.strftime('%X')}] [Worker {args.worker_id}] Iniciado no dispositivo {args.device}.")
    print(f"Total de tarefas no pool: {len(all_tasks)} (Round 1: {len(tasks_by_seed['seed_42'])}, Round 2: {len(tasks_by_seed['seed_123'])}, Round 3: {len(tasks_by_seed['seed_2024'])})")

    processed_count = 0

    for current_seed in SEEDS:
        pool = list(tasks_by_seed[current_seed])
        # Worker 0 pega do início (CNNs), Worker 1 pega do final (Transformers) para evitar colisão
        if args.worker_id == 1:
            pool = list(reversed(pool))

        print(f"\n{'#'*80}")
        print(f"[{time.strftime('%X')}] 🏁 [Worker {args.worker_id}] Iniciando ROUND: {current_seed} ({len(pool)} tarefas)")
        print(f"{'#'*80}")

        for task in pool:
            run_name = task["run_name"]
            out_csv = TABLES_DIR / f"df40_all_fakes_{run_name}.csv"
            lock_file = TABLES_DIR / f"df40_all_fakes_{run_name}.lock"

            # Checar se já concluído
            if out_csv.exists() and out_csv.stat().st_size > 100:
                continue

            # Tentativa de Lock Atômico
            try:
                fd = os.open(str(lock_file), os.O_CREAT | os.O_EXCL | os.O_WRONLY)
                with os.fdopen(fd, "w") as f:
                    f.write(f"worker_{args.worker_id}_{args.device}_{os.getpid()}\n")
            except FileExistsError:
                # Outro worker já assumiu esta tarefa
                continue

            print(f"\n{'='*80}")
            print(f"[{time.strftime('%X')}] 🚀 [Worker {args.worker_id} - {args.device}] Reivindicou tarefa: {run_name}")
            print(f"Checkpoint: {task['checkpoint']}")
            print(f"{'='*80}")

            t0 = time.time()
            cmd = [
                PYTHON_EXEC, "-u", str(ROOT_DIR / "scripts" / "evaluate_df40_all_fakes_generic.py"),
                "--model-name", run_name,
                "--family", task["family"],
                "--mode", task["mode"],
                "--checkpoint", task["checkpoint"],
                "--batch-size", str(task["batch_size"]),
                "--device", args.device,
                "--num-workers", "10",
                "--output-dir", str(TABLES_DIR.relative_to(ROOT_DIR)),
            ]

            try:
                res = subprocess.run(cmd, check=True)
                dt = time.time() - t0
                print(f"[{time.strftime('%X')}] ✅ [Worker {args.worker_id}] Concluído com sucesso: {run_name} em {dt/60:.1f} min")
                processed_count += 1
            except subprocess.CalledProcessError as e:
                print(f"[{time.strftime('%X')}] ❌ [Worker {args.worker_id}] Erro ao processar {run_name}: {e}")
            finally:
                if lock_file.exists():
                    lock_file.unlink()

    print(f"\n🎉 [Worker {args.worker_id}] Todas as 3 sementes concluídas! Total processado por este worker: {processed_count}")


if __name__ == "__main__":
    main()
