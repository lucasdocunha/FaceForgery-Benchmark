"""Evaluation-only adapter using the original NumPy FFT dataset helpers.

New trainable pilots use differentiable torch FFT internally. Existing weights
must instead retain their original evaluation representation; this adapter does
not use the older standalone visualization helper's extra phase rescaling.
"""

import torch
from .imaging import normalize_rgb, MODES


def encode_legacy_tensor(raw: torch.Tensor, mode: str, in_channels=None):
    from src.data.data import ImageDataset

    if raw.ndim != 4 or raw.shape[1] != 3 or mode not in MODES:
        raise ValueError("Expected BCHW RGB and a known legacy representation")
    helper = ImageDataset.__new__(ImageDataset)
    items = []
    for image in raw.detach().cpu():
        rgb = normalize_rgb(image)
        if mode == "none":
            out = rgb
        elif mode == "srm":
            from src.forensics.srm import extract_srm_residuals

            out = torch.cat([rgb, extract_srm_residuals(rgb[None])[0]])
        elif mode == "srm_only":
            from src.forensics.srm import extract_srm_residuals

            out = extract_srm_residuals(rgb[None])[0]
        elif mode == "dtcwt":
            from src.forensics.dtcwt_module import extract_dtcwt_features

            bands = extract_dtcwt_features(rgb[None], mode="directional_only")[0]
            out = torch.cat([rgb, bands])
        elif mode == "magnitude":
            out = helper._fft_magnitude(image) * 2 - 1
        elif mode == "phase":
            out = helper._fft_phase(image) * 2 - 1
        elif mode == "complex":
            out = helper._fft_complex(image)
        elif mode == "frequency_3":
            out = helper._fft_highpass(image) * 2 - 1
        elif mode == "concat":
            out = torch.cat([rgb, helper._fft_magnitude(image) * 2 - 1])
        else:
            out = torch.cat(
                [
                    rgb,
                    helper._fft_magnitude(image) * 2 - 1,
                    helper._fft_phase(image) * 2 - 1,
                    helper._fft_highpass(image) * 2 - 1,
                    helper._fft_lowpass(image) * 2 - 1,
                ]
            )
            if in_channels == 6:
                out = out[:6]
        if in_channels is not None and out.shape[0] != in_channels:
            raise ValueError("Legacy representation/checkpoint channel mismatch")
        if not torch.isfinite(out).all():
            raise ValueError("Nonfinite legacy encoding")
        items.append(out)
    return torch.stack(items).to(raw.device)
