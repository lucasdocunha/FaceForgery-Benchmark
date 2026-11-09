"""Script mestre para agendar e orquestrar a avaliação no Celeb-DF v2.

Permite rodar famílias em GPUs específicas, monitorar progresso e consolidar
as tabelas finais de resultados (Frame e Vídeo) em Markdown, CSV e LaTeX.
"""

from __future__ import annotations

import argparse
import subprocess
import sys
from pathlib import Path

from make_celeb_df_tables import make_tables

FAMILIES = ("resnet", "mobilenet", "xception", "vit", "clip", "dino")


def main():
    p = argparse.ArgumentParser(description="Orquestrador de testes no Celeb-DF v2")
    p.add_argument("--gpus", default="0,1", help="IDs das GPUs separadas por vírgula (ex: '0,1')")
    p.add_argument("--families", default=",".join(FAMILIES), help="Famílias a avaliar")
    p.add_argument("--regime", default="finetune", choices=("finetune", "scratch", "all"))
    p.add_argument("--batch-size", type=int, default=64)
    p.add_argument("--num-workers", type=int, default=4)
    p.add_argument("--skip-existing", action="store_true", default=True)
    p.add_argument("--aggregate-only", action="store_true", help="Apenas gera as tabelas finais consolidadas")
    args = p.parse_args()

    if args.aggregate_only:
        print("[AGGREGATE] Consolidando tabelas a partir dos resultados em disco...")
        agg_frame, agg_video = make_tables()
        print("\n" + "=" * 80)
        print("RESUMO CELEB-DF (NÍVEL DE VÍDEO):")
        print("=" * 80)
        if not agg_video.empty:
            print(agg_video[["model_family", "fourier_mode", "regime", "auc_mean", "auc_std", "acc_mean", "acc_std", "f1_mean"]].to_markdown(index=False))
        return

    gpu_list = [g.strip() for g in args.gpus.split(",")]
    fam_list = [f.strip() for f in args.families.split(",")]

    print(f"Famílias selecionadas: {fam_list}")
    print(f"GPUs configuradas: {gpu_list}")

    for idx, fam in enumerate(fam_list):
        assigned_gpu = gpu_list[idx % len(gpu_list)]
        cmd = [
            sys.executable,
            "evaluate_celeb_df.py",
            "--only-family", fam,
            "--regime", args.regime,
            "--device", f"cuda:{assigned_gpu}",
            "--batch-size", str(args.batch_size),
            "--num-workers", str(args.num_workers),
        ]
        if args.skip_existing:
            cmd.append("--skip-existing")

        print(f"\n[LAUNCH] Iniciando avaliação de '{fam}' na GPU {assigned_gpu}...")
        res = subprocess.run(cmd)
        if res.returncode != 0:
            print(f"[WARN] Erro ao avaliar família {fam} (código {res.returncode})")

    print("\n[DONE] Todas as famílias concluídas! Gerando tabelas consolidadas...")
    make_tables()


if __name__ == "__main__":
    main()
