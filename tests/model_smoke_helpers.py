from pathlib import Path

import pytest
import torch
from PIL import Image

from src.data.data import encode_pil_image
from src.models.registry import MODEL_REGISTRY
from src.pipelines.config import TrainingConfig

MODES = ("phase", "complex", "concat", "concat_frequency")


def assert_family_smoke(family: str, tiny_phase1_dataset: Path):
    image_path = next((tiny_phase1_dataset / "trainset").glob("*.jpg"))
    image = Image.open(image_path).convert("RGB")
    for mode in MODES:
        config = TrainingConfig(
            model_family=family, fourier_mode=mode, image_size=32, patch_size=8,
            hidden_size=32, num_hidden_layers=1, num_attention_heads=4,
            variant="small", model_size="tiny", multi_gpu=False,
        )
        model = MODEL_REGISTRY[family].build(config).eval()
        value = encode_pil_image(image, mode, 32).unsqueeze(0)
        with torch.no_grad():
            assert model(value).shape == (1, 2)
        del model
    rejected = TrainingConfig(model_family=family, regime="finetune", allow_pretrained=False)
    with pytest.raises(ValueError):
        MODEL_REGISTRY[family].build(rejected)
