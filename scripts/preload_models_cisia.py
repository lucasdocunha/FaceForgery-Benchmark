#!/usr/bin/env python3
"""Pré-carrega todos os backbones pré-treinados no cache do CISIA.

Execute este script no nó de login (boolevm) onde há acesso à internet garantido.
Assim, os nós de computação do Slurm carregarão os modelos offline direto do disco (/projects/),
evitando erros de download e bloqueios de rede no cluster.
"""

import os
from pathlib import Path

# Garante que o cache aponte para /projects no CISIA se a variável não estiver setada
if "HF_HOME" not in os.environ and Path("/projects/lucas.ocunha").exists():
    os.environ["HF_HOME"] = "/projects/lucas.ocunha/.cache/huggingface"
if "TORCH_HOME" not in os.environ and Path("/projects/lucas.ocunha").exists():
    os.environ["TORCH_HOME"] = "/projects/lucas.ocunha/.cache/torch"

print("==========================================================")
print("📥 PRÉ-CARREGAMENTO DE MODELOS NO CLUSTER")
print(f"HF_HOME:    {os.environ.get('HF_HOME', '~/.cache/huggingface')}")
print(f"TORCH_HOME: {os.environ.get('TORCH_HOME', '~/.cache/torch')}")
print("==========================================================\n")

# 1. CLIP ViT-B/16
print("1/4 Baixando CLIP (openai/clip-vit-base-patch16)...")
from transformers import CLIPVisionModel
CLIPVisionModel.from_pretrained("openai/clip-vit-base-patch16", use_safetensors=True)
print("   ✅ CLIP em cache!")

# 2. ViT-Base/16
print("2/4 Baixando ViT (google/vit-base-patch16-224)...")
from transformers import ViTModel
ViTModel.from_pretrained("google/vit-base-patch16-224")
print("   ✅ ViT em cache!")

# 3. DINOv3 ConvNeXt-Base
print("3/4 Baixando DINO (convnext_base.dinov3_lvd1689m via timm)...")
import timm
timm.create_model("convnext_base.dinov3_lvd1689m", pretrained=True, num_classes=0)
print("   ✅ DINO em cache!")

# 4. ResNet & MobileNet (torchvision)
print("4/4 Baixando ResNet-18 e MobileNetV3 (torchvision)...")
import torchvision.models as tvm
tvm.resnet18(weights=tvm.ResNet18_Weights.DEFAULT)
tvm.mobilenet_v3_large(weights=tvm.MobileNet_V3_Large_Weights.DEFAULT)
print("   ✅ ResNet e MobileNet em cache!")

print("\n==========================================================")
print("🎉 Todos os 6 modelos foram baixados e cacheados com sucesso!")
print("Agora os jobs do Slurm rodarão 100% offline a partir do disco.")
print("==========================================================")
