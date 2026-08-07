import torch

from src.models.registry import MODEL_REGISTRY
from src.pipelines.config import TrainingConfig
from src.plots.heatmap import attention_rollout, grad_cam, grid, overlay


class TinyCNN(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.features = torch.nn.Sequential(torch.nn.Conv2d(3, 4, 3, padding=1), torch.nn.ReLU())
        self.classifier = torch.nn.Linear(4, 2)
    def forward(self, x):
        return self.classifier(self.features(x).mean((2, 3)))


def test_gradcam_overlay_and_grid_shapes():
    images = torch.rand(2, 3, 32, 32)
    heatmaps = grad_cam(TinyCNN(), images)
    assert heatmaps.shape == (2, 1, 32, 32)
    assert overlay(images, heatmaps).shape == images.shape
    assert grid(images, heatmaps).ndim == 3


def test_attention_rollout_uses_transformer_attention():
    config = TrainingConfig(model_family="vit", image_size=32, patch_size=8,
                            hidden_size=32, num_hidden_layers=2, num_attention_heads=4)
    model = MODEL_REGISTRY["vit"].build(config).eval()
    heatmap = attention_rollout(model, torch.rand(1, 3, 32, 32))
    assert heatmap.shape == (1, 1, 32, 32)
    assert torch.isfinite(heatmap).all()
