import pytest
from src.models.registry import MODEL_REGISTRY
def test_dino_is_registered(): assert "dino" in MODEL_REGISTRY
