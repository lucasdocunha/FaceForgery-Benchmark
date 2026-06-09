import pytest
import torch


def _assert_valid_heatmap(heatmap: torch.Tensor, image_size: int) -> None:
    assert heatmap.shape == (image_size, image_size)
    assert torch.isfinite(heatmap).all()
    assert float(heatmap.min()) >= 0.0
    assert float(heatmap.max()) <= 1.0


def test_gradcam_heatmap_supports_current_vision_models():
    from src.models.clip import CLIPVisionClassifier
    from src.models.mobilenet import mobilenetv3_small
    from src.models.resnet import resnet
    from src.models.vit import VisionTransformerClassifier
    from src.models.xception import xception
    from src.plots.heatmap import gradcam_heatmap

    image_size = 64
    cases = [
        (
            resnet(num_classes=2, pretrained=False, architecture="resnet18"),
            torch.randn(1, 3, image_size, image_size),
        ),
        (
            mobilenetv3_small(num_classes=2, in_channels=6, pretrained=False),
            torch.randn(1, 6, image_size, image_size),
        ),
        (
            xception(pretrained=False, in_channels=3, num_classes=2),
            torch.randn(1, 3, image_size, image_size),
        ),
        (
            VisionTransformerClassifier(
                image_size=image_size,
                patch_size=16,
                hidden_size=32,
                num_hidden_layers=1,
                num_attention_heads=4,
                in_channels=4,
            ),
            torch.randn(1, 4, image_size, image_size),
        ),
        (
            CLIPVisionClassifier(
                num_classes=2,
                image_size=image_size,
                patch_size=16,
                hidden_size=32,
                projection_dim=16,
                num_hidden_layers=1,
                num_attention_heads=4,
            ),
            torch.randn(1, 3, image_size, image_size),
        ),
    ]

    for model, image in cases:
        model.eval()
        heatmap = gradcam_heatmap(model, image, target_class=1)
        _assert_valid_heatmap(heatmap, image_size)


def test_overlay_heatmap_accepts_project_channel_modes_and_saves_png(tmp_path):
    from src.plots.heatmap import gradcam_heatmap, overlay_heatmap, save_heatmap_overlay
    from src.models.vit import VisionTransformerClassifier

    image_size = 64
    image = torch.randn(1, 6, image_size, image_size)
    model = VisionTransformerClassifier(
        image_size=image_size,
        patch_size=16,
        hidden_size=32,
        num_hidden_layers=1,
        num_attention_heads=4,
        in_channels=6,
    )
    model.eval()

    heatmap = gradcam_heatmap(model, image, target_class=1)
    overlay = overlay_heatmap(image, heatmap, alpha=0.5)
    output_path = save_heatmap_overlay(
        image,
        heatmap,
        tmp_path / "plots" / "heatmap.png",
        alpha=0.5,
    )

    assert overlay.shape == (image_size, image_size, 3)
    assert overlay.dtype == torch.uint8
    assert output_path.exists()


def test_find_target_layer_rejects_models_without_supported_layers():
    from src.plots.heatmap import find_target_layer

    model = torch.nn.Sequential(torch.nn.Flatten(), torch.nn.Linear(4, 2))

    with pytest.raises(ValueError, match="No compatible layer"):
        find_target_layer(model)
