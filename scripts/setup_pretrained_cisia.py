#!/usr/bin/env python3
"""Configura e verifica os modelos pré-treinados no cluster CISIA.

Cria uma pasta dedicada para cada arquitetura em:
    /projects/models/lucas.ocunha/pretrained/
        ├── clip/
        ├── vit/
        ├── dino/
        ├── resnet/
        ├── mobilenet/
        └── xception/

Para cada modelo:
- Verifica se os pesos já existem na pasta.
- Se JÁ EXISTIREM: reaproveita instantaneamente sem fazer download.
- Se NÃO EXISTIREM: baixa no nó de login (boolevm) e salva de forma definitiva na pasta correspondente.

Execute este script no nó de login:
    python scripts/setup_pretrained_cisia.py
"""

from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path

ROOT_DIR = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT_DIR))

from src.data.paths import pretrained_root

DEFAULT_TARGET = Path("/projects/models/lucas.ocunha/pretrained")


def get_dir_size_mb(path: Path) -> float:
    if not path.exists():
        return 0.0
    if path.is_file():
        return path.stat().st_size / (1024 * 1024)
    total = sum(f.stat().st_size for f in path.rglob("*") if f.is_file())
    return total / (1024 * 1024)


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
    from transformers import CLIPVisionConfig, CLIPVisionModel
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
    from transformers import ViTConfig, ViTModel
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
    torch.save(model.state_dict(), target_file)
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

    print(f"   ⬇️  Baixando pesos do ResNet-18 e salvando em: {target_file}...")
    import torch
    import torchvision.models as tvm
    model = tvm.resnet18(weights=tvm.ResNet18_Weights.DEFAULT)
    torch.save(model.state_dict(), target_file)
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

    print(f"   ⬇️  Baixando pesos do MobileNetV3-Large e salvando em: {target_file}...")
    import torch
    import torchvision.models as tvm
    model = tvm.mobilenet_v3_large(weights=tvm.MobileNet_V3_Large_Weights.DEFAULT)
    torch.save(model.state_dict(), target_file)
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
    model = timm.create_model("legacy_xception", pretrained=True, num_classes=0)
    torch.save(model.state_dict(), target_file)
    sz = get_dir_size_mb(target_file)
    print(f"   ✅ [CONCLUÍDO] Xception salvo com sucesso em {target_file} ({sz:.1f} MB)")


def main() -> None:
    parser = argparse.ArgumentParser(description="Configura pastas dedicadas para modelos pré-treinados no CISIA")
    parser.add_argument(
        "--target-dir",
        type=Path,
        default=None,
        help="Diretório raiz onde as subpastas serão criadas (padrão: /projects/models/lucas.ocunha/pretrained)",
    )
    args = parser.parse_args()

    target_dir = args.target_dir or pretrained_root()
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

    print("\n" + "█" * 70)
    print("🎉 STATUS FINAL: TODAS AS 6 PASTAS ESTÃO PRONTAS E REAPROVEITÁVEIS!")
    print(f"Estrutura organizada em: {target_dir}")
    for sub in sorted(target_dir.glob("*")):
        if sub.is_dir():
            files = list(sub.glob("*"))
            sz = get_dir_size_mb(sub)
            print(f"   📁 {sub.name:<12s} -> {len(files)} arquivo(s), total: {sz:.1f} MB")
    print("█" * 70)
    print("\nOs jobs do Slurm carregarão diretamente dessas pastas sem usar a internet!\n")


if __name__ == "__main__":
    main()
