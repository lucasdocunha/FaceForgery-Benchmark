import torch
from torch import nn
import pytest
from src.robustness.models import (
    PilotDetector,
    KINDS,
    consistency_loss,
    distillation_loss,
)
from src.robustness.imaging import encode_tensor


class TinyBackbone(nn.Module):
    num_features = 8

    def __init__(self, channels=3):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv2d(channels, 8, 3, padding=1), nn.AdaptiveAvgPool2d(1), nn.Flatten()
        )

    def forward(self, x):
        return self.net(x)


@pytest.mark.parametrize("kind", sorted(KINDS))
def test_model_forward_backward(kind):
    model = PilotDetector(
        TinyBackbone(4 if kind == "early_concat" else 3), kind, width=4
    )
    x = torch.rand(2, 3, 32, 32, requires_grad=True)
    out = model(x)
    assert out.shape == (2, 2) and torch.isfinite(out).all()
    out.sum().backward()
    assert torch.isfinite(x.grad).all()


def test_spatial_control_parameter_match():
    a = PilotDetector(TinyBackbone(), "adaptive", width=4)
    b = PilotDetector(TinyBackbone(), "spatial_control", width=4)
    assert sum(p.numel() for p in a.parameters()) == sum(
        p.numel() for p in b.parameters()
    )


def test_gate_is_rgb_fallback_when_suppressed():
    m = PilotDetector(TinyBackbone(), "adaptive", width=4).eval()
    with torch.no_grad():
        m.gate[-1].bias.fill_(-100)
    d = m.forward_details(torch.rand(2, 3, 32, 32))
    assert torch.allclose(d["logits"], d["rgb_logits"])


def test_consistency_zero_and_distillation_teacher_detached():
    s = torch.randn(4, 2, requires_grad=True)
    t = torch.randn(4, 2, requires_grad=True)
    assert consistency_loss(s, s).item() == pytest.approx(0, abs=1e-7)
    distillation_loss(s, t).backward()
    assert s.grad is not None and t.grad is None


@pytest.mark.parametrize(
    "mode,channels",
    [
        ("none", 3),
        ("magnitude", 1),
        ("phase", 1),
        ("complex", 2),
        ("concat", 4),
        ("frequency_3", 1),
        ("concat_frequency", 7),
    ],
)
def test_encoder_contract(mode, channels):
    x = torch.rand(2, 3, 32, 32)
    y = encode_tensor(x, mode, channels)
    assert y.shape == (2, channels, 32, 32) and torch.isfinite(y).all()
    if mode == "concat_frequency":
        assert encode_tensor(x, mode, 6).shape[1] == 6


def test_constant_spectrum_remains_finite():
    for mode in ["magnitude", "phase", "complex", "frequency_3", "concat_frequency"]:
        assert torch.isfinite(encode_tensor(torch.zeros(2, 3, 16, 16), mode)).all()
