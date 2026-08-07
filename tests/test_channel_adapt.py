import torch
import torch.nn as nn

from src.models._channel_adapt import adapt_conv2d_channels


def test_channel_adaptation_uses_mean_and_scaled_repetition():
    conv = nn.Conv2d(3, 2, 1, bias=False)
    with torch.no_grad():
        conv.weight.copy_(torch.arange(6, dtype=torch.float32).reshape(2, 3, 1, 1))
    one = adapt_conv2d_channels(conv, 1)
    six = adapt_conv2d_channels(conv, 6)
    assert torch.allclose(one.weight, conv.weight.mean(dim=1, keepdim=True))
    assert torch.allclose(six.weight, conv.weight.repeat(1, 2, 1, 1) * .5)
