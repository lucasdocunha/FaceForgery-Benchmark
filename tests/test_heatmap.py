import torch

from src.models.registry import MODEL_REGISTRY
from src.pipelines.config import TrainingConfig
from src.plots.heatmap import (
    attention_rollout,
    channel_shapley,
    generate,
    grad_cam,
    gradient_shap,
    grid,
    kernel_shap,
    overlay,
)


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


def test_gradient_shap_shapes_and_values():
    images = torch.rand(2, 3, 32, 32)
    heatmaps = gradient_shap(TinyCNN(), images, n_steps=8)
    assert heatmaps.shape == (2, 1, 32, 32)
    assert torch.isfinite(heatmaps).all()
    assert (heatmaps >= 0.0).all() and (heatmaps <= 1.0).all()


def test_gradient_shap_with_custom_baseline_and_target():
    images = torch.rand(1, 3, 32, 32)
    baseline = torch.ones(1, 3, 32, 32) * 0.5
    heatmaps = gradient_shap(TinyCNN(), images, target_class=1, baselines=baseline, n_steps=4)
    assert heatmaps.shape == (1, 1, 32, 32)
    assert torch.isfinite(heatmaps).all()


def test_kernel_shap_shapes_and_values():
    images = torch.rand(2, 3, 32, 32)
    heatmaps = kernel_shap(TinyCNN(), images, patch_size=8, n_samples=32)
    assert heatmaps.shape == (2, 1, 32, 32)
    assert torch.isfinite(heatmaps).all()
    assert (heatmaps >= 0.0).all() and (heatmaps <= 1.0).all()


def test_channel_shapley_exact_sum():
    class LinearChannelModel(torch.nn.Module):
        def forward(self, x):
            means = x.mean(dim=(-2, -1))
            weights = torch.tensor([2.0, 3.0, 5.0], device=x.device)
            score = (means * weights).sum(dim=-1, keepdim=True)
            return torch.cat([torch.zeros_like(score), score], dim=-1)

    x = torch.tensor([[[[1.0]], [[2.0]], [[3.0]]]])
    phi = channel_shapley(LinearChannelModel(), x, target_class=1, baseline_value=0.0)
    total_shap = sum(phi.values())
    assert abs(total_shap - 23.0) < 1e-4
    assert abs(phi["channel_0"] - 2.0) < 1e-4
    assert abs(phi["channel_1"] - 6.0) < 1e-4
    assert abs(phi["channel_2"] - 15.0) < 1e-4


def test_generate_supports_shapley_on_moe():
    config = TrainingConfig(
        model_family="moe_standard",
        fourier_mode="none",
        num_experts=2,
        variant="small",
        image_size=32,
    )
    model = MODEL_REGISTRY["moe_standard"].build(config).eval()
    image = torch.rand(1, 3, 32, 32)
    heatmap = generate(model, "moe_standard", image, method="auto", n_steps=4)
    assert heatmap.shape == (1, 1, 32, 32)
    assert torch.isfinite(heatmap).all()
