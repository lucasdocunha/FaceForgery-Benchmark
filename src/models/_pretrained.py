from __future__ import annotations

import os
from pathlib import Path
from typing import Any

from src.data.paths import models_root, _REPO_DATA_ROOT


def get_candidate_pretrained_dirs(family: str) -> list[Path]:
    """Retorna os caminhos candidatos onde os pesos pré-treinados podem estar armazenados.
    
    A ordem prioriza variáveis explícitas, depois o diretório do usuário no cluster HPC,
    depois o diretório compartilhado do dataset e, finalmente, os modelos locais do repo.
    """
    candidates = []
    explicit = os.environ.get("TCC_PRETRAINED_ROOT")
    if explicit:
        candidates.append(Path(explicit).expanduser() / family)

    user = os.environ.get("USER", "")
    if user:
        candidates.append(Path("/projects/models") / user / "pretrained" / family)
        candidates.append(Path("/users/home") / user / "models" / "pretrained" / family)

    candidates.append(Path("/datasets/ai_models/faceforgery/pretrained") / family)
    candidates.append(models_root() / "pretrained" / family)
    candidates.append(_REPO_DATA_ROOT.parent / "models" / "pretrained" / family)

    seen = set()
    result = []
    for c in candidates:
        if c not in seen:
            seen.add(c)
            result.append(c)
    return result


def is_valid_safetensors(path: Path, min_size_mb: float = 10.0) -> bool:
    """Verifica se um arquivo .safetensors existe, tem tamanho mínimo e header íntegro."""
    if not path.is_file():
        return False
    try:
        if path.stat().st_size < min_size_mb * 1024 * 1024:
            return False
        from safetensors.torch import safe_open
        with safe_open(str(path), framework="pt") as f:
            return len(f.keys()) > 0
    except Exception:
        return False


def is_valid_bin(path: Path, min_size_mb: float = 10.0) -> bool:
    """Verifica se um arquivo .bin/.pth existe e tem tamanho plausível."""
    if not path.is_file():
        return False
    try:
        return path.stat().st_size >= min_size_mb * 1024 * 1024
    except Exception:
        return False


def load_hf_backbone(model_class: Any, family: str, repo_id: str) -> Any:
    """Carrega o backbone Hugging Face com tolerância a arquivos truncados, NFS e offline compute nodes."""
    candidate_dirs = get_candidate_pretrained_dirs(family)

    # 1. Procura em cada diretório candidato local
    for local_dir in candidate_dirs:
        if not local_dir.is_dir():
            continue

        safetensors_file = local_dir / "model.safetensors"
        bin_file = local_dir / "pytorch_model.bin"

        # 1a. Tenta safetensors local se o arquivo for íntegro
        if is_valid_safetensors(safetensors_file):
            try:
                print(f"[{family}] Carregando pesos pré-treinados locais (safetensors) de: {local_dir}", flush=True)
                return model_class.from_pretrained(str(local_dir), use_safetensors=True)
            except Exception as e:
                print(f"[{family}] ⚠️ Falha ao abrir safetensors local de {local_dir}: {e}. Tentando alternativas...", flush=True)
        elif safetensors_file.exists():
            print(
                f"[{family}] ⚠️ Arquivo safetensors em {safetensors_file} corrompido ou truncado "
                f"({safetensors_file.stat().st_size} bytes). Pulando...",
                flush=True,
            )

        # 1b. Tenta pytorch_model.bin se safetensors não estiver disponível ou falhar
        if is_valid_bin(bin_file):
            try:
                print(f"[{family}] Carregando pesos pré-treinados locais (.bin) de: {local_dir}", flush=True)
                return model_class.from_pretrained(str(local_dir), use_safetensors=False)
            except Exception as e:
                print(f"[{family}] ⚠️ Falha ao abrir .bin de {local_dir}: {e}. Tentando alternativas...", flush=True)

    # 2. Se nenhum local for válido, tenta carregar diretamente do Hugging Face Hub ou cache local offline
    for use_safe in (True, False):
        try:
            print(f"[{family}] Carregando do Hugging Face Hub: {repo_id} (use_safetensors={use_safe})...", flush=True)
            return model_class.from_pretrained(repo_id, use_safetensors=use_safe)
        except Exception:
            try:
                print(f"[{family}] Tentando cache offline do Hugging Face para {repo_id} (use_safetensors={use_safe})...", flush=True)
                return model_class.from_pretrained(repo_id, use_safetensors=use_safe, local_files_only=True)
            except Exception:
                pass

    checked = [str(d) for d in candidate_dirs if d.exists()]
    raise RuntimeError(
        f"[{family}] Falha crítica ao carregar pesos pré-treinados ({repo_id}).\n"
        f"Diretórios locais verificados: {checked if checked else 'Nenhum diretório existente'}.\n"
        f"Possível causa no cluster: arquivo model.safetensors vazio (0 bytes), incompleto ou sem permissão de leitura.\n"
        f"Para resolver no CISIA, re-baixe os pesos executando: sbatch scripts/download_pretrained_cisia.sh"
    )
