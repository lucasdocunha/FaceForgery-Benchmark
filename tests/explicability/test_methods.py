"""Synthetic tensors only. These checks do not validate benchmark conclusions."""
from pathlib import Path
import tempfile
import unittest
import numpy as np
import pandas as pd
from PIL import Image
import torch
from torch import nn
from torchvision import transforms
from src.explicability.methods import METHODS, UnsupportedMethod, explain, feature_mask, perturbation_curves
from src.explicability.runtime import EvaluationEncoder
from src.data.data import ImageDataset
from src.explicability.render import panels, render_cell


class TinyCNN(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv = nn.Conv2d(3, 4, 3, padding=1)
        self.head = nn.Linear(4, 2)
    def forward(self, x):
        return self.head(torch.tanh(self.conv(x)).mean((-2, -1)))


class SumClassifier(nn.Module):
    def forward(self, x):
        score = x.flatten(1).sum(1)
        return torch.stack((-score, score), 1)


class MethodTests(unittest.TestCase):
    def setUp(self):
        torch.set_num_threads(1); torch.manual_seed(42)
        self.model = TinyCNN().eval()
        self.x = torch.rand(1, 3, 8, 8)
        self.b = torch.zeros_like(self.x)
        self.settings = {"samples": 64, "ig_steps": 8, "gradient_samples": 4,
            "shapley_permutations": 4, "patch_size": 4, "perturbations_per_eval": 4}

    def test_method_adapters(self):
        for method in METHODS:
            with self.subTest(method=method):
                if method == "attention_rollout":
                    with self.assertRaises(UnsupportedMethod):
                        explain(self.model, "resnet", "none", self.x, self.b, method, self.settings, 42)
                    continue
                attribution, mask, meta = explain(self.model, "resnet", "none", self.x, self.b, method, self.settings, 42)
                self.assertEqual(attribution.shape[-2:], (8, 8))
                self.assertTrue(torch.isfinite(attribution).all())
                self.assertEqual(meta["target"], "logit_fake_minus_real")
                self.assertEqual(mask.shape, self.x.shape)

    def test_ig_has_fixed_margin_and_completeness(self):
        attr, _, meta = explain(SumClassifier(), "resnet", "none", self.x, self.b,
                               "integrated_gradients", self.settings, 42)
        self.assertTrue(torch.allclose(attr, self.x * 2, atol=1e-5))
        self.assertLess(meta["completeness_residual"], 1e-4)

    def test_frozen_parameters_still_allow_input_gradients(self):
        for parameter in self.model.parameters(): parameter.requires_grad_(False)
        attr, _, _ = explain(self.model, "resnet", "none", self.x, self.b, "gradcam", self.settings, 42)
        self.assertTrue(torch.isfinite(attr).all())

    def test_stochastic_seed_is_reproducible(self):
        a, _, _ = explain(self.model, "resnet", "none", self.x, self.b, "gradient_shap", self.settings, 99)
        b, _, _ = explain(self.model, "resnet", "none", self.x, self.b, "gradient_shap", self.settings, 99)
        self.assertTrue(torch.equal(a, b))

    def test_partial_edge_patches_and_hybrid_groups(self):
        x = torch.zeros(1, 7, 9, 11)
        mask = feature_mask(x, "concat_frequency", patch=4)
        self.assertEqual(mask.shape, x.shape)
        self.assertTrue(torch.equal(mask[0, 0], mask[0, 2]))
        self.assertGreater(int(mask[0, 3].min()), int(mask[0, 2].max()))
        self.assertEqual(len(torch.unique(mask)), 45)

    def test_radial_mask_couples_complex_components(self):
        x = torch.zeros(1, 2, 16, 16)
        mask = feature_mask(x, "complex", partition="radial")
        self.assertTrue(torch.equal(mask[0, 0], mask[0, 1]))

    def test_group_density_preserves_group_total(self):
        attr, _, meta = explain(self.model, "resnet", "none", self.x, self.b,
                               "kernel_shap", self.settings, 42)
        self.assertAlmostEqual(float(attr.sum()), float(meta["group_coefficients"].sum()), places=5)

    def test_encoding_matches_iclr_dataset_for_all_channel_modes(self):
        with tempfile.TemporaryDirectory() as d:
            root = Path(d)
            array = np.random.default_rng(0).integers(0, 256, (14, 12, 3), dtype=np.uint8)
            image = Image.fromarray(array); image.save(root / "test.png")
            pd.DataFrame({"img_name": ["test.png"], "label": [0]}).to_csv(root / "data.csv", index=False)
            transform = transforms.Compose([transforms.Resize((8, 8)), transforms.ToTensor(),
                transforms.Normalize([.485, .456, .406], [.229, .224, .225])])
            for mode, channels in (("none", 3), ("magnitude", 1), ("phase", 1), ("complex", 2),
                ("frequency_3", 1), ("concat", 4), ("concat_frequency", 6), ("concat_frequency", 7)):
                with self.subTest(mode=mode, channels=channels):
                    dataset = ImageDataset(root / "data.csv", root, transform=transform, fourier=mode,
                                           spatial_size=(8, 8), in_channels=channels)
                    encoded, rgb = EvaluationEncoder(mode, 8, channels).encode(image)
                    expected, _, _ = dataset[0]
                    self.assertTrue(torch.allclose(encoded, expected, atol=1e-6))
                    views = panels(encoded.numpy(), rgb.numpy(), np.ones((1, channels, 8, 8)), mode)
                    self.assertGreater(len(views), 0)

    def test_curves_have_correct_endpoints(self):
        attr, mask, _ = explain(self.model, "resnet", "none", self.x, self.b,
                               "integrated_gradients", self.settings, 42)
        curves = perturbation_curves(self.model, self.x, self.b, attr, mask, 42, points=3)
        self.assertAlmostEqual(curves["ranked"]["deletion"][0], curves["ranked"]["insertion"][-1])
        self.assertAlmostEqual(curves["ranked"]["deletion"][-1], curves["ranked"]["insertion"][0])

    def test_render_is_an_actual_file_not_a_result_claim(self):
        with tempfile.TemporaryDirectory() as d:
            path = Path(d) / "synthetic-render.png"
            render_cell(self.x[0].numpy(), self.x[0].numpy(), self.x.numpy(), "none", "saliency", path,
                        label="SYNTHETIC TEST TENSOR; NOT EXPERIMENTAL EVIDENCE")
            self.assertGreater(path.stat().st_size, 100)


if __name__ == "__main__": unittest.main()
