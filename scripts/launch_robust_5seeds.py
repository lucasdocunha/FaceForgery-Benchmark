#!/usr/bin/env python3
"""Disparador e orquestrador de treinamento dos modelos robustos nas 5 sementes (42, 123, 2024, 7, 2025).

Distribui os modelos entre as GPUs disponíveis:
- GPU 0: clip, dino, resnet
- GPU 1: vit, mobilenet, xception

Aplica RandomizedRobustAugment no treino e avalia automaticamente:
- Validação (val.csv -> limiar ótimo)
- Teste Limpo (test.csv)
- Teste Corrompido / Difícil (test_d)
- Benchmark DF40 (data/df40/test.csv)
"""

from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

import pandas as pd
import torch

ROOT_DIR = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT_DIR))

from scripts.train_robust_seeds import train_single_seed

GPU_ASSIGNMENTS = {
    0: ["clip", "dino", "resnet"],
    1: ["vit", "mobilenet", "xception"],
}

DEFAULT_SEEDS = [42, 123, 2024, 7, 2025]


def run_worker(gpu_id: int, families: list[str], seeds: list[int], regime: str = "finetune_robust", num_workers: int = 4):
    device = torch.device(f"cuda:{gpu_id}" if torch.cuda.is_available() else "cpu")
    print(f"\n{'='*75}")
    print(f"🚀 INICIANDO WORKER ROBUSTO NA GPU {gpu_id} ({device})")
    print(f"Famílias ({len(families)}): {families}")
    print(f"Seeds ({len(seeds)}): {seeds}")
    print(f"Regime: {regime}")
    print(f"{'='*75}\n", flush=True)

    log_dir = ROOT_DIR / "logs"
    log_dir.mkdir(parents=True, exist_ok=True)
    summary_path = ROOT_DIR / "tables" / f"robust_5seeds_gpu{gpu_id}.csv"
    summary_path.parent.mkdir(parents=True, exist_ok=True)

    all_results = []
    t_start = time.time()

    for f_idx, family in enumerate(families, 1):
        print(f"\n>>> [{f_idx}/{len(families)}] Processando família: {family.upper()} na GPU {gpu_id} <<<", flush=True)
        for s_idx, seed in enumerate(seeds, 1):
            print(f"\n--- [GPU {gpu_id}] {family.upper()} (Seed {seed} | {s_idx}/{len(seeds)}) ---", flush=True)
            try:
                res = train_single_seed(
                    family=family,
                    seed=seed,
                    regime=regime,
                    fourier_mode="none",
                    num_workers=num_workers,
                    device=device,
                )
                all_results.append(res)
                # Salva progresso incremental
                pd.DataFrame(all_results).to_csv(summary_path, index=False)
            except Exception as e:
                print(f"❌ [ERRO] Falha em {family} seed {seed} na GPU {gpu_id}: {e}", flush=True)
                import traceback
                traceback.print_exc()

    total_time = (time.time() - t_start) / 60
    print(f"\n{'='*75}")
    print(f"🎉 WORKER GPU {gpu_id} FINALIZADO! Tempo total: {total_time:.1f} min")
    print(f"Resultados consolidados em: {summary_path}")
    print(f"{'='*75}\n", flush=True)


def main():
    parser = argparse.ArgumentParser(description="Lançador de treino robusto multi-seed (5 seeds)")
    parser.add_argument("--gpu-id", type=int, default=0, choices=[0, 1], help="ID da GPU (0 ou 1)")
    parser.add_argument("--families", type=str, default=None, help="Famílias customizadas separadas por vírgula")
    parser.add_argument("--seeds", type=str, default="42,123,2024,7,2025", help="Sementes separadas por vírgula")
    parser.add_argument("--regime", type=str, default="finetune_robust", help="Regime de treinamento")
    parser.add_argument("--num-workers", type=int, default=4, help="Workers do DataLoader")
    args = parser.parse_args()

    seeds = [int(s.strip()) for s in args.seeds.split(",") if s.strip()]
    if args.families:
        families = [f.strip().lower() for f in args.families.split(",") if f.strip()]
    else:
        families = GPU_ASSIGNMENTS.get(args.gpu_id, ["resnet"])

    run_worker(
        gpu_id=args.gpu_id,
        families=families,
        seeds=seeds,
        regime=args.regime,
        num_workers=args.num_workers,
    )


if __name__ == "__main__":
    main()
