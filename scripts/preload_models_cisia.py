#!/usr/bin/env python3
"""Pré-carrega todos os backbones pré-treinados no cache do CISIA em /projects/models/lucas.ocunha.

Execute este script no nó de login (boolevm) onde há acesso à internet garantido:
    python scripts/preload_models_cisia.py

Assim, os nós de computação do Slurm carregarão os modelos offline direto do disco (/projects/models/lucas.ocunha/.cache/),
evitando erros de download e bloqueios de rede no cluster.
"""

import os
from pathlib import Path

# Configura o diretório de cache exatamente em /projects/models/lucas.ocunha/.cache
CISIA_PROJECTS_MODELS = Path("/projects/models/lucas.ocunha")
if CISIA_PROJECTS_MODELS.exists():
    cache_base = CISIA_PROJECTS_MODELS / ".cache"
    cache_base.mkdir(parents=True, exist_ok=True)
    os.environ.setdefault("HF_HOME", str(cache_base / "huggingface"))
    os.environ.setdefault("TORCH_HOME", str(cache_base / "torch"))
else:
    os.environ.setdefault("HF_HOME", os.path.expanduser("~/.cache/huggingface"))
    os.environ.setdefault("TORCH_HOME", os.path.expanduser("~/.cache/torch"))

hf_home = Path(os.environ["HF_HOME"])
torch_home = Path(os.environ["TORCH_HOME"])
hf_home.mkdir(parents=True, exist_ok=True)
torch_home.mkdir(parents=True, exist_ok=True)

print("==========================================================")
print("📥 PRÉ-CARREGAMENTO DE MODELOS NO CLUSTER CISIA")
print(f"Diretório de Modelos: {CISIA_PROJECTS_MODELS if CISIA_PROJECTS_MODELS.exists() else 'Local'}")
print(f"HF_HOME:              {hf_home}")
print(f"TORCH_HOME:           {torch_home}")
print("==========================================================\n")

# 1. CLIP ViT-B/16
print("1/4 Baixando CLIP (openai/clip-vit-base-patch16)...")
from transformers import CLIPVisionModel
CLIPVisionModel.from_pretrained("openai/clip-vit-base-patch16", use_safetensors=True)
print("   ✅ CLIP salvo em cache com sucesso!")

# 2. ViT-Base/16
print("2/4 Baixando ViT (google/vit-base-patch16-224)...")
from transformers import ViTModel
ViTModel.from_pretrained("google/vit-base-patch16-224", use_safetensors=True)
print("   ✅ ViT salvo em cache com sucesso!")

# 3. DINOv3 ConvNeXt-Base
print("3/4 Baixando DINO (convnext_base.dinov3_lvd1689m via timm)...")
import timm
timm.create_model("convnext_base.dinov3_lvd1689m", pretrained=True, num_classes=0)
print("   ✅ DINO salvo em cache com sucesso!")

# 4. ResNet & MobileNet (torchvision)
print("4/4 Baixando ResNet-18 e MobileNetV3 (torchvision)...")
import torchvision.models as tvm
tvm.resnet18(weights=tvm.ResNet18_Weights.DEFAULT)
tvm.mobilenet_v3_large(weights=tvm.MobileNet_V3_Large_Weights.DEFAULT)
print("   ✅ ResNet e MobileNet salvos em cache com sucesso!")

print("\n==========================================================")
print("🎉 Todos os backbones foram baixados e persistidos em:")
print(f"   {hf_home}")
print(f"   {torch_home}")
print("Agora qualquer job do Slurm rodará 100% offline direto do disco.")
print("==========================================================")
