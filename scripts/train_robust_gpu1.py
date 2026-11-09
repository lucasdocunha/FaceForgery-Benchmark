#!/usr/bin/env python3
"""GPU 1 — ViT + MobileNet + Xception com RandomizedRobustAugment."""
import os
os.environ["CUDA_VISIBLE_DEVICES"] = "1"
import sys
sys.path.insert(0, "/home/lucas.ocunha/tcc")
from scripts._robust_common import run_gpu_group

run_gpu_group(
    gpu_id=1,
    configs=[
        ("vit",        "configs/vit.yaml",       32, 25),
        ("mobilenet",  "configs/mobilenet.yaml",  32, 20),
        ("xception",   "configs/xception.yaml",   32, 20),
    ],
)
