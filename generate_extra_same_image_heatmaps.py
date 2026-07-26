"""
Gera heatmaps para 2 novos IDs (1 Fake + 1 Real) em todos os 6 modelos,
salvando em heatmaps_same_image/<model>/ para usar no grid.
"""
import sys
from pathlib import Path
import torch

from src.plots.resnet_heatmap_generator import generate_resnet_heatmaps
from src.plots.transformer_heatmap_generator import generate_transformer_heatmaps
from src.data.paths import phase1_split_root

# IDs escolhidos: 1 Fake (class0) e 1 Real (class1), diferentes dos já existentes
NEW_IDS = [159795, 181946]

TEST_CSV   = Path("data/raw/test.csv")
IMAGES_DIR = phase1_split_root("test")
DEVICE     = torch.device("cuda" if torch.cuda.is_available() else "cpu")

MODELS = [
    # (family, model_dir)
    ("resnet",     Path("models/resnet")),
    ("xception",   Path("models/xception/none")),
    ("mobilenet",  Path("models/mobilenet/mobilenetv3_large/none")),
    ("clip",       Path("models/clip/clip_vit_scratch/none")),
    ("vit",        Path("models/vit/vit_scratch/none")),
    ("dino",       Path("models/dino/dinov3_base/rgb")),
]

for family, model_dir in MODELS:
    out_dir = Path("heatmaps_same_image") / family
    print(f"\n→ {family}  ({model_dir})")
    try:
        if family == "resnet":
            manifest = generate_resnet_heatmaps(
                model_dir=model_dir,
                test_csv=TEST_CSV,
                images_dir=IMAGES_DIR,
                output_dir=out_dir,
                ids=NEW_IDS,
                device=DEVICE,
            )
        else:
            manifest = generate_transformer_heatmaps(
                model_dir=model_dir,
                family=family,
                test_csv=TEST_CSV,
                images_dir=IMAGES_DIR,
                output_dir=out_dir,
                ids=NEW_IDS,
                device=DEVICE,
            )
        print(f"   {len(manifest)} heatmaps salvos em {out_dir}")
    except Exception as exc:
        print(f"   ERRO: {exc}", file=sys.stderr)
