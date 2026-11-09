#!/usr/bin/env python3
"""
Fila dinâmica de inferência exaustiva do DF-40 AGRUPADA POR REPRESENTAÇÃO.
Em vez de passar 60 vezes pelo dataset por semente, passa apenas 10 vezes
(1 vez por representação), avaliando todas as famílias (CNNs e Transformers)
simultaneamente no mesmo lote.

Ordem de execução:
- Round 1: seed_42 (apenas os grupos com modelos ainda pendentes)
- Round 2: seed_123 (todos os 10 grupos)
- Round 3: seed_2024 (todos os 10 grupos)
"""

from __future__ import annotations
import argparse
import json
import os
import subprocess
import sys
import time
from collections import defaultdict
from pathlib import Path

ROOT_DIR = Path(__file__).resolve().parent.parent
PYTHON_EXEC = sys.executable

FAMILIES = ["resnet", "mobilenet", "xception", "vit", "clip", "dino"]

# (label, fourier_mode, regime)
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

SEEDS = ["seed_42", "seed_123", "seed_2024", "seed_7", "seed_2025"]
MODELS_DIR = Path("/media/ssd2/lucas.ocunha/models-tcc")
TABLES_DIR = ROOT_DIR / "tables" / "df40_all_fakes_models"


def get_all_grouped_tasks():
    """
    Retorna grupos de modelos organizados por (seed, mode).
    Cada grupo contém todos os modelos disponíveis para aquela semente e representação.
    """
    groups = []

    for seed in SEEDS:
        # Agrupar por fourier_mode (ex: none, phase, magnitude...)
        mode_groups = defaultdict(list)

        for fam in FAMILIES:
            for label, mode, regime in METHODS:
                ckpt = MODELS_DIR / fam / mode / regime / seed / "weights" / "best.pth"
                if not ckpt.exists():
                    continue

                run_name = f"{fam}_{label}_{seed}"
                mode_groups[mode].append({
                    "run_name": run_name,
                    "family": fam,
                    "label": label,
                    "mode": mode,
                    "regime": regime,
                    "seed": seed,
                    "checkpoint": str(ckpt),
                })

        # Para cada modo nesta semente, criar a definição do grupo
        for mode, model_list in mode_groups.items():
            group_id = f"{seed}_{mode}"
            groups.append({
                "group_id": group_id,
                "seed": seed,
                "mode": mode,
                "models": model_list,
            })

    return groups


def main():
    parser = argparse.ArgumentParser(description="Worker de Fila Agrupada DF-40")
    parser.add_argument("--worker-id", type=int, default=0, help="ID do worker (0 ou 1)")
    parser.add_argument("--device", type=str, default="cuda:0", help="Dispositivo PyTorch (cuda:0, cuda:1)")
    parser.add_argument("--batch-size", type=int, default=256, help="Batch size do DataLoader")
    parser.add_argument("--num-workers", type=int, default=10, help="Workers DataLoader")
    args = parser.parse_args()

    TABLES_DIR.mkdir(parents=True, exist_ok=True)
    all_groups = get_all_grouped_tasks()

    print(f"[{time.strftime('%X')}] 🚀 [Worker {args.worker_id}] Iniciado no dispositivo {args.device}.")
    print(f"Total de grupos no pool: {len(all_groups)} (SEEDS: {SEEDS})")

    processed_groups = 0

    for current_seed in SEEDS:
        seed_groups = [g for g in all_groups if g["seed"] == current_seed]

        # Worker 0 pega do início, Worker 1 pega do final para balanceamento
        if args.worker_id == 1:
            seed_groups = list(reversed(seed_groups))

        print(f"\n{'#'*80}")
        print(f"[{time.strftime('%X')}] 🏁 [Worker {args.worker_id}] Iniciando ROUND: {current_seed} ({len(seed_groups)} grupos)")
        print(f"{'#'*80}")

        for grp in seed_groups:
            group_id = grp["group_id"]
            mode = grp["mode"]
            models = grp["models"]

            # Checar se todos os modelos do grupo já estão concluídos
            pending_in_group = []
            for m in models:
                out_csv = TABLES_DIR / f"df40_all_fakes_{m['run_name']}.csv"
                if not (out_csv.exists() and out_csv.stat().st_size > 100):
                    pending_in_group.append(m)

            if not pending_in_group:
                continue

            lock_file = TABLES_DIR / f"group_{group_id}.lock"

            # Tentativa de Lock Atômico no Grupo
            try:
                fd = os.open(str(lock_file), os.O_CREAT | os.O_EXCL | os.O_WRONLY)
                with os.fdopen(fd, "w") as f:
                    f.write(f"worker_{args.worker_id}_{args.device}_{os.getpid()}\n")
            except FileExistsError:
                # Outro worker já assumiu este grupo
                continue

            print(f"\n{'='*80}")
            print(f"[{time.strftime('%X')}] 🎯 [Worker {args.worker_id} - {args.device}] Reivindicou GRUPO: {group_id}")
            print(f"Modo: {mode.upper()} | Modelos Pendentes: {len(pending_in_group)} de {len(models)}")
            for m in pending_in_group:
                print(f"  • {m['run_name']} ({m['family']})")
            print(f"{'='*80}")

            # Salvar JSON temporário dos modelos deste grupo
            tmp_json = ROOT_DIR / "scripts" / f"_tmp_group_{group_id}_{args.worker_id}.json"
            with open(tmp_json, "w") as f:
                json.dump(pending_in_group, f, indent=2)

            t0 = time.time()
            cmd = [
                PYTHON_EXEC, "-u", str(ROOT_DIR / "scripts" / "evaluate_df40_grouped.py"),
                "--mode", mode,
                "--models-json", str(tmp_json),
                "--device", args.device,
                "--batch-size", str(args.batch_size),
                "--num-workers", str(args.num_workers),
                "--output-dir", str(TABLES_DIR.relative_to(ROOT_DIR)),
            ]

            try:
                res = subprocess.run(cmd, check=True)
                dt = time.time() - t0
                print(f"[{time.strftime('%X')}] ✅ [Worker {args.worker_id}] Grupo {group_id} concluído com sucesso em {dt/60:.1f} min!")
                processed_groups += 1
            except subprocess.CalledProcessError as e:
                print(f"[{time.strftime('%X')}] ❌ [Worker {args.worker_id}] Erro ao processar grupo {group_id}: {e}")
            finally:
                if lock_file.exists():
                    lock_file.unlink()
                if tmp_json.exists():
                    tmp_json.unlink()

    print(f"\n🎉 [Worker {args.worker_id}] Todas as tarefas agrupadas concluídas! Total de grupos processados: {processed_groups}")


if __name__ == "__main__":
    main()
