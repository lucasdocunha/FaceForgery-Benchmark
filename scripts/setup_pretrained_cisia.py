#!/usr/bin/env python3
"""Prepare/verify pretrained backbones inside an allocated CISIA job.

Submit from the repository root after creating logs/:
    sbatch scripts/download_pretrained_cisia.sh

Downloads and serialization use job-local storage. The job runner publishes a
versioned read-only source under /projects/models/$USER only when the process
finishes. No package installation or TLS verification bypass is performed here.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import re
import tempfile
from urllib.parse import urlsplit
import os
import sys
from pathlib import Path

ROOT_DIR = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT_DIR))

from src.data.paths import pretrained_root
from src.utils.atomic import atomic_copy, atomic_torch_save


def get_dir_size_mb(path: Path) -> float:
    if not path.exists():
        return 0.0
    if path.is_file():
        return path.stat().st_size / (1024 * 1024)
    total = sum(f.stat().st_size for f in path.rglob("*") if f.is_file())
    return total / (1024 * 1024)


def safe_save_state_dict(state_dict, target_file: Path) -> None:
    atomic_torch_save(state_dict, target_file)


def download_file_safely(url: str, target_file: Path) -> None:
    """Stream a verified HTTPS download to a unique local file, then publish it."""
    import urllib.request

    parsed = urlsplit(url)
    prefix = re.search(r"-([a-f0-9]{8,64})\.", parsed.path)
    if parsed.scheme != "https" or prefix is None:
        raise ValueError("Expected an HTTPS torchvision URL with a SHA-256 filename prefix")
    target_file.parent.mkdir(parents=True, exist_ok=True)
    digest = hashlib.sha256()
    # TMPDIR is assigned by src.hpc.runtime before this process starts.
    with tempfile.TemporaryDirectory(prefix="pretrained-") as temporary:
        downloaded = Path(temporary) / "weights.pth"
        request = urllib.request.Request(url, headers={"User-Agent": "FaceForgery-Benchmark"})
        with urllib.request.urlopen(request, timeout=120) as response, downloaded.open("wb") as output:
            while chunk := response.read(1024 * 1024):
                output.write(chunk)
                digest.update(chunk)
        if not digest.hexdigest().startswith(prefix.group(1)):
            raise ValueError(f"SHA-256 mismatch for {url}")
        atomic_copy(downloaded, target_file)


def setup_clip(target_dir: Path) -> None:
    clip_dir = target_dir / "clip"
    clip_dir.mkdir(parents=True, exist_ok=True)
    weight_files = list(clip_dir.glob("*.safetensors")) + list(clip_dir.glob("*.bin"))
    has_config = (clip_dir / "config.json").exists()

    print("\n" + "─" * 60)
    print("1/6 🔍 CLIP (openai/clip-vit-base-patch16)")
    if weight_files and has_config:
        sz = get_dir_size_mb(clip_dir)
        print(f"   ✅ [JÁ EXISTE] Reaproveitando pesos locais em: {clip_dir} ({sz:.1f} MB)")
        return

    print(f"   ⬇️  Baixando pesos do CLIP e salvando em: {clip_dir}...")
    from transformers import CLIPVisionModel
    repo_id = "openai/clip-vit-base-patch16"
    model = CLIPVisionModel.from_pretrained(repo_id, use_safetensors=True)
    model.save_pretrained(str(clip_dir), safe_serialization=True)
    sz = get_dir_size_mb(clip_dir)
    print(f"   ✅ [CONCLUÍDO] CLIP salvo com sucesso em {clip_dir} ({sz:.1f} MB)")


def setup_vit(target_dir: Path) -> None:
    vit_dir = target_dir / "vit"
    vit_dir.mkdir(parents=True, exist_ok=True)
    weight_files = list(vit_dir.glob("*.safetensors")) + list(vit_dir.glob("*.bin"))
    has_config = (vit_dir / "config.json").exists()

    print("\n" + "─" * 60)
    print("2/6 🔍 ViT (google/vit-base-patch16-224)")
    if weight_files and has_config:
        sz = get_dir_size_mb(vit_dir)
        print(f"   ✅ [JÁ EXISTE] Reaproveitando pesos locais em: {vit_dir} ({sz:.1f} MB)")
        return

    print(f"   ⬇️  Baixando pesos do ViT e salvando em: {vit_dir}...")
    from transformers import ViTModel
    repo_id = "google/vit-base-patch16-224"
    model = ViTModel.from_pretrained(repo_id, use_safetensors=True)
    model.save_pretrained(str(vit_dir), safe_serialization=True)
    sz = get_dir_size_mb(vit_dir)
    print(f"   ✅ [CONCLUÍDO] ViT salvo com sucesso em {vit_dir} ({sz:.1f} MB)")


def setup_dino(target_dir: Path) -> None:
    dino_dir = target_dir / "dino"
    dino_dir.mkdir(parents=True, exist_ok=True)
    target_file = dino_dir / "base.pth"
    alt_file = dino_dir / "convnext_base.pth"

    print("\n" + "─" * 60)
    print("3/6 🔍 DINOv3 (convnext_base.dinov3_lvd1689m via timm)")
    if target_file.exists() or alt_file.exists():
        found = target_file if target_file.exists() else alt_file
        sz = get_dir_size_mb(found)
        print(f"   ✅ [JÁ EXISTE] Reaproveitando pesos locais em: {found} ({sz:.1f} MB)")
        return

    print(f"   ⬇️  Baixando pesos do DINOv3 ConvNeXt-Base e salvando em: {target_file}...")
    import timm
    import torch
    model = timm.create_model("convnext_base.dinov3_lvd1689m", pretrained=True, num_classes=0)
    safe_save_state_dict(model.state_dict(), target_file)
    sz = get_dir_size_mb(target_file)
    print(f"   ✅ [CONCLUÍDO] DINO salvo com sucesso em {target_file} ({sz:.1f} MB)")


def setup_resnet(target_dir: Path) -> None:
    resnet_dir = target_dir / "resnet"
    resnet_dir.mkdir(parents=True, exist_ok=True)
    target_file = resnet_dir / "resnet18.pth"

    print("\n" + "─" * 60)
    print("4/6 🔍 ResNet (resnet18 torchvision)")
    if target_file.exists():
        sz = get_dir_size_mb(target_file)
        print(f"   ✅ [JÁ EXISTE] Reaproveitando pesos locais em: {target_file} ({sz:.1f} MB)")
        return

    print(f"   ⬇️  Baixando pesos do ResNet-18...")
    import torchvision.models as tvm
    try:
        url = tvm.ResNet18_Weights.DEFAULT.url
    except Exception:
        url = "https://download.pytorch.org/models/resnet18-f37072fd.pth"
    download_file_safely(url, target_file)
    sz = get_dir_size_mb(target_file)
    print(f"   ✅ [CONCLUÍDO] ResNet-18 salvo com sucesso em {target_file} ({sz:.1f} MB)")


def setup_mobilenet(target_dir: Path) -> None:
    mobilenet_dir = target_dir / "mobilenet"
    mobilenet_dir.mkdir(parents=True, exist_ok=True)
    target_file = mobilenet_dir / "mobilenet_v3_large.pth"

    print("\n" + "─" * 60)
    print("5/6 🔍 MobileNet (mobilenet_v3_large torchvision)")
    if target_file.exists():
        sz = get_dir_size_mb(target_file)
        print(f"   ✅ [JÁ EXISTE] Reaproveitando pesos locais em: {target_file} ({sz:.1f} MB)")
        return

    print(f"   ⬇️  Baixando pesos do MobileNetV3-Large...")
    import torchvision.models as tvm
    try:
        url = tvm.MobileNet_V3_Large_Weights.DEFAULT.url
    except Exception:
        url = "https://download.pytorch.org/models/mobilenet_v3_large-8738ca79.pth"
    download_file_safely(url, target_file)
    sz = get_dir_size_mb(target_file)
    print(f"   ✅ [CONCLUÍDO] MobileNetV3 salvo com sucesso em {target_file} ({sz:.1f} MB)")


def setup_xception(target_dir: Path) -> None:
    xc_dir = target_dir / "xception"
    xc_dir.mkdir(parents=True, exist_ok=True)
    target_file = xc_dir / "legacy_xception.pth"
    alt_file = xc_dir / "xception.pth"

    print("\n" + "─" * 60)
    print("6/6 🔍 Xception (legacy_xception via timm)")
    if target_file.exists() or alt_file.exists():
        found = target_file if target_file.exists() else alt_file
        sz = get_dir_size_mb(found)
        print(f"   ✅ [JÁ EXISTE] Reaproveitando pesos locais em: {found} ({sz:.1f} MB)")
        return

    print(f"   ⬇️  Baixando pesos do Xception e salvando em: {target_file}...")
    import timm
    import torch
    import torch.hub
    model = timm.create_model("legacy_xception", pretrained=True, num_classes=0)
    safe_save_state_dict(model.state_dict(), target_file)
    sz = get_dir_size_mb(target_file)
    print(f"   ✅ [CONCLUÍDO] Xception salvo com sucesso em {target_file} ({sz:.1f} MB)")


def main() -> None:
    parser = argparse.ArgumentParser(description="Configura pastas dedicadas para modelos pré-treinados no CISIA")
    parser.add_argument(
        "--target-dir",
        type=Path,
        default=None,
        help="Local job staging directory (normally supplied by TCC_PRETRAINED_ROOT)",
    )
    args = parser.parse_args()

    if not os.environ.get("SLURM_JOB_ID") or not os.environ.get("TCC_JOB_DIR"):
        parser.error("Use sbatch scripts/download_pretrained_cisia.sh; direct Shell Access execution is not allowed")
    target_dir = (args.target_dir or pretrained_root()).resolve()
    if not target_dir.is_relative_to(Path(os.environ["TCC_JOB_DIR"]).resolve()):
        parser.error("Target must be within the job-local workspace; final publication is handled by the runner")
    target_dir.mkdir(parents=True, exist_ok=True)

    print("\n" + "█" * 70)
    print("  VERIFICAÇÃO E CONFIGURAÇÃO DE MODELOS PRÉ-TREINADOS (CISIA)")
    print(f"  Diretório Base: {target_dir}")
    print("█" * 70)


    # Executa a verificação/download para cada modelo
    setup_clip(target_dir)
    setup_vit(target_dir)
    setup_dino(target_dir)
    setup_resnet(target_dir)
    setup_mobilenet(target_dir)
    setup_xception(target_dir)

    # Validate serialization/architecture compatibility using the same builders
    # as training, not just file existence. This runs on allocated CPUs.
    os.environ["TCC_PRETRAINED_ROOT"] = str(target_dir)
    from src.models.registry import get_model_spec
    from src.pipelines.config import load_config
    verified = {}
    for family in ("resnet", "xception", "mobilenet", "vit", "clip", "dino"):
        config = load_config(ROOT_DIR / "configs" / f"{family}.yaml", {"regime": "finetune", "multi_gpu": False})
        model = get_model_spec(family).build(config)
        verified[family] = {"config": config.to_dict(), "parameters": sum(p.numel() for p in model.parameters())}
        del model
        print(f"Verified local backbone: {family}", flush=True)
    (target_dir / "BACKBONES.json").write_text(json.dumps(verified, indent=2), encoding="utf-8")

    print("\n" + "█" * 70)
    print("🎉 STATUS FINAL: TODAS AS 6 PASTAS ESTÃO PRONTAS E REAPROVEITÁVEIS!")
    print(f"Estrutura organizada em: {target_dir}")
    for sub in sorted(target_dir.glob("*")):
        if sub.is_dir():
            files = list(sub.glob("*"))
            sz = get_dir_size_mb(sub)
            print(f"   📁 {sub.name:<12s} -> {len(files)} arquivo(s), total: {sz:.1f} MB")
    print("█" * 70)
    print("\nValidated staging files; the runner will print the final published path.\n")


if __name__ == "__main__":
    main()
