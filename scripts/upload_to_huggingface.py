#!/usr/bin/env python3
"""
Script utilitário para fazer upload estruturado dos pesos e metadados dos modelos para o Hugging Face Hub.
Permite criar o repositório (público ou privado) e subir os arquivos com barra de progresso e retries.
"""
from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from src.data.paths import models_root


def main():
    parser = argparse.ArgumentParser(description="Upload de modelos treinados para o Hugging Face Hub")
    parser.add_argument("--repo-id", type=str, required=True, help="Nome do repositório no HF (ex: 'seu-usuario/tcc-models')")
    parser.add_argument("--models-dir", type=str, default=str(models_root()), help="Caminho dos modelos locais (padrão: models_root)")
    parser.add_argument("--private", action="store_true", help="Criar o repositório como privado")
    parser.add_argument("--only-best", action="store_true", help="Subir apenas checkpoints best.pth (ignorar final.pth)")
    parser.add_argument("--include-predictions", action="store_true", help="Incluir arquivos de predição outputs_*.npz")
    parser.add_argument("--token", type=str, default=None, help="Hugging Face Token (ou use a variável de ambiente HF_TOKEN)")
    parser.add_argument("--dry-run", action="store_true", help="Apenas simula e lista os arquivos que seriam enviados sem fazer upload")
    args = parser.parse_args()

    models_dir = Path(args.models_dir)
    if not models_dir.exists():
        print(f"❌ Diretório de modelos não encontrado: {models_dir}")
        sys.exit(1)

    try:
        from huggingface_hub import HfApi, get_token
    except ImportError:
        print("❌ Biblioteca 'huggingface_hub' não encontrada. Instale com: pip install huggingface_hub")
        sys.exit(1)

    token = args.token or os.environ.get("HF_TOKEN") or get_token()
    if not token and not args.dry_run:
        print("\n❌ Nenhum token do Hugging Face encontrado!")
        print("   Por favor, faça login com:")
        print("     hf auth login")
        print("   Ou defina a variável:")
        print("     export HF_TOKEN='seu_token_hf'")
        sys.exit(1)

    api = HfApi(token=token)

    # Definir padrões de exclusão
    ignore_patterns = ["backups_interrupted/**", "**/backups_interrupted/**", "tmp/**", "temp/**"]
    if args.only_best:
        ignore_patterns.append("**/final.pth")
    if not args.include_predictions:
        ignore_patterns.append("**/*.npz")

    print(f"📁 Diretório fonte: {models_dir}")
    print(f"🌐 Repositório destino: {args.repo_id} (Tipo: model, Privado: {args.private})")
    print(f"🚫 Padrões ignorados: {ignore_patterns}")

    if args.dry_run:
        print("\n🔍 Executando modo Dry-Run (analisando arquivos)...")
        total_files = 0
        total_bytes = 0
        for p in models_dir.rglob("*"):
            if not p.is_file():
                continue
            rel = p.relative_to(models_dir)
            if "backups_interrupted" in rel.parts:
                continue
            if args.only_best and p.name == "final.pth":
                continue
            if not args.include_predictions and p.suffix == ".npz":
                continue
            total_files += 1
            total_bytes += p.stat().st_size
        gb = total_bytes / (1024**3)
        print(f"✅ Total de arquivos selecionados: {total_files}")
        print(f"📦 Tamanho total aproximado: {gb:.2f} GB")
        print("\nPronto para upload real! Remova a flag --dry-run para iniciar.")
        return

    print("\n🔨 Verificando/criando repositório no Hugging Face...")
    api.create_repo(repo_id=args.repo_id, repo_type="model", private=args.private, exist_ok=True)
    print("✅ Repositório pronto.")

    print("\n🚀 Iniciando upload otimizado para grandes volumes (upload_large_folder)...")
    print("   (Suporta paralelismo com múltiplos workers, chunking e relatórios periódicos)")
    if hasattr(api, "upload_large_folder"):
        api.upload_large_folder(
            folder_path=str(models_dir),
            repo_id=args.repo_id,
            repo_type="model",
            ignore_patterns=ignore_patterns,
            print_report=True,
            print_report_every=30,
            num_workers=8,
        )
    else:
        api.upload_folder(
            folder_path=str(models_dir),
            repo_id=args.repo_id,
            repo_type="model",
            ignore_patterns=ignore_patterns,
        )
    print(f"\n🎉 UPLOAD CONCLUÍDO COM SUCESSO!")
    print(f"🔗 Acesse em: https://huggingface.co/{args.repo_id}")


if __name__ == "__main__":
    main()
