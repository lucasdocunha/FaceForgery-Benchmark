#!/usr/bin/env python3
"""
Sincroniza automaticamente todos os checkpoints e métricas ausentes
do SSD local para o repositório Hugging Face: lucasoc/MFFI-Models.
"""

from __future__ import annotations
import os
import sys
import time
from pathlib import Path
from huggingface_hub import HfApi

REPO_ID = "lucasoc/MFFI-Models"
LOCAL_ROOT = Path("/media/ssd2/lucas.ocunha/models-tcc")
PROJECT_ROOT = Path("/home/lucas.ocunha/tcc")
TOKEN_FILE = PROJECT_ROOT / ".hf_token"


def main():
    if not TOKEN_FILE.exists():
        print(f"❌ Token não encontrado em {TOKEN_FILE}")
        sys.exit(1)

    token = TOKEN_FILE.read_text().strip()
    api = HfApi(token=token)

    print(f"=== Sincronização Inteligente Hugging Face: {REPO_ID} ===")
    print("Obtendo lista de arquivos existentes no Hugging Face...")
    hf_files = set(api.list_repo_files(repo_id=REPO_ID, repo_type="model"))
    print(f"Total de arquivos já presentes no HF: {len(hf_files)}")

    # Identificar diretórios de runs locais cujos best.pth faltam no HF
    missing_runs = []
    for pth in LOCAL_ROOT.glob("**/best.pth"):
        rel_pth = str(pth.relative_to(LOCAL_ROOT))
        if "backups_interrupted" in rel_pth:
            continue
        if rel_pth not in hf_files:
            run_dir = pth.parent.parent
            missing_runs.append((rel_pth, run_dir))

    print(f"\nEncontrados {len(missing_runs)} checkpoints ausentes no Hugging Face.")

    if not missing_runs:
        print("🎉 Todos os checkpoints locais já estão sincronizados no Hugging Face!")
        return

    # Padrões ignorados para economizar banda e não subir lixo/previsões gigantes
    ignore_patterns = [
        "**/final.pth",
        "**/*.npz",
        "**/predictions*.csv",
        "**/*outputs*.npz",
        "**/backups_interrupted/**",
        "**/checkpoints_epoch_*.pth",
    ]

    total = len(missing_runs)
    success = 0
    errors = 0

    t_start = time.time()
    for idx, (rel_pth, run_dir) in enumerate(missing_runs, 1):
        rel_run_dir = run_dir.relative_to(LOCAL_ROOT)
        print(f"\n[{idx}/{total}] 📤 Enviando run: {rel_run_dir}...", flush=True)
        t0 = time.time()
        try:
            api.upload_folder(
                folder_path=str(run_dir),
                path_in_repo=str(rel_run_dir),
                repo_id=REPO_ID,
                repo_type="model",
                ignore_patterns=ignore_patterns,
                commit_message=f"feat(weights): sync {rel_run_dir} (best.pth and metrics)",
            )
            dt = time.time() - t0
            print(f"     ✅ Sucesso em {dt:.1f}s: {rel_pth}", flush=True)
            success += 1
        except Exception as e:
            print(f"     ❌ Erro ao enviar {rel_run_dir}: {e}", flush=True)
            errors += 1

    total_time = (time.time() - t_start) / 60.0
    print("\n" + "=" * 80)
    print(f"🎉 Sincronização concluída em {total_time:.1f} minutos!")
    print(f"Enviados com sucesso: {success} | Erros: {errors}")
    print(f"🔗 Repositório: https://huggingface.co/{REPO_ID}")
    print("=" * 80)


if __name__ == "__main__":
    main()
