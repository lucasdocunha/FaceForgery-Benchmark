#!/usr/bin/env python3
"""GPU 0 — CLIP (reaproveita seed 987) + DINO + ResNet com RandomizedRobustAugment."""
import os
os.environ["CUDA_VISIBLE_DEVICES"] = "0"
import sys
sys.path.insert(0, "/home/lucas.ocunha/tcc")
from scripts._robust_common import run_gpu_group

# CLIP já treinado (seed 987 existe) — apenas avalia test_d
# DINO e ResNet serão treinados do zero com robust aug
run_gpu_group(
    gpu_id=0,
    configs=[
        ("clip",   "configs/clip.yaml",   16, 20),
        ("dino",   "configs/dino.yaml",   16, 20),
        ("resnet", "configs/resnet.yaml", 32, 25),
    ],
)
