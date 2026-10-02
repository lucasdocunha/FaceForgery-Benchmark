import numpy as np
import pandas as pd
import pytest
import torch
from PIL import Image
from torchvision import transforms
from src.data.data import ImageDataset
from src.robustness.legacy_encoding import encode_legacy_tensor
from src.robustness.imaging import to_tensor


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
        ("concat_frequency", 6),
        ("srm", 6),
    ],
)
def test_legacy_input_matches_actual_dataset(tmp_path, mode, channels):
    rng = np.random.default_rng(42)
    image = Image.fromarray(rng.integers(0, 256, (32, 32, 3), dtype=np.uint8))
    image.save(tmp_path / "image.png")
    csv = tmp_path / "m.csv"
    pd.DataFrame({"img_name": ["image.png"], "label": [1]}).to_csv(csv, index=False)
    transform = transforms.Compose(
        [
            transforms.Resize((32, 32)),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
        ]
    )
    dataset = ImageDataset(
        csv,
        tmp_path,
        transform=transform,
        fourier=mode,
        spatial_size=(32, 32),
        in_channels=channels,
    )
    expected, _, _ = dataset[0]
    actual = encode_legacy_tensor(to_tensor(image).unsqueeze(0), mode, channels)[0]
    assert torch.allclose(actual, expected, atol=2e-6, rtol=2e-6)


def test_srm_resized_jpeg_and_batch_order_match_dataset(tmp_path):
    from src.robustness.imaging import CanonicalDataset, encode_tensor

    rng = np.random.default_rng(8)
    names = []
    for index in range(3):
        name = f"{index}.jpg"
        names.append(name)
        Image.fromarray(rng.integers(0, 256, (37, 49, 3), dtype=np.uint8)).save(tmp_path / name)
    csv = tmp_path / "images.csv"
    pd.DataFrame({"img_name": names, "label": [0, 1, 0]}).to_csv(csv, index=False)
    dataset = ImageDataset(csv, tmp_path, spatial_size=(32, 32), fourier="srm", in_channels=6,
                           allow_skip_unreadable=False)
    frame = pd.DataFrame({"img_name": names, "label": [0, 1, 0], "sample_id": names,
                          "group_id": names, "dataset": "fixture", "split": "test"})
    canonical = CanonicalDataset(frame, tmp_path, 32)
    raw = torch.stack([canonical[index]["image"] for index in range(3)])
    expected = torch.stack([dataset[index][0] for index in range(3)])
    actual = encode_legacy_tensor(raw, "srm", 6)
    assert torch.allclose(actual, expected, atol=2e-6, rtol=2e-6)
    assert torch.allclose(encode_tensor(raw, "srm", 6), expected, atol=2e-6, rtol=2e-6)
    assert torch.allclose(encode_legacy_tensor(raw.flip(0), "srm", 6).flip(0), actual)
    with pytest.raises(ValueError, match="channel mismatch"):
        encode_legacy_tensor(raw, "srm", 3)


def test_srm_keeps_normalized_input_and_tlu_semantics():
    from src.robustness.imaging import normalize_rgb
    from src.forensics.srm import extract_srm_residuals

    raw = torch.zeros(1, 3, 32, 32)
    raw[:, :, ::2, ::2] = 1
    encoded = encode_legacy_tensor(raw, "srm", 6)
    expected_rgb = normalize_rgb(raw)
    assert torch.equal(encoded[:, :3], expected_rgb)
    assert torch.equal(encoded[:, 3:], extract_srm_residuals(expected_rgb))
    assert encoded[:, 3:].abs().max() <= 1
    assert not torch.allclose(encoded[:, 3:], extract_srm_residuals(raw))


def test_original_dataset_never_replaces_missing_sample(tmp_path):
    Image.fromarray(np.zeros((32, 32, 3), dtype=np.uint8)).save(tmp_path / "exists.png")
    csv = tmp_path / "m.csv"
    pd.DataFrame({"img_name": ["missing.png", "exists.png"], "label": [0, 1]}).to_csv(
        csv, index=False
    )
    dataset = ImageDataset(csv, tmp_path, spatial_size=(32, 32))
    with pytest.raises(RuntimeError, match="substitution is forbidden"):
        dataset[0]


def test_dataset_allows_skip_unreadable_when_configured(tmp_path):
    Image.fromarray(np.ones((32, 32, 3), dtype=np.uint8) * 128).save(tmp_path / "exists.png")
    csv = tmp_path / "m.csv"
    pd.DataFrame({"img_name": ["missing.png", "exists.png"], "label": [0, 1]}).to_csv(
        csv, index=False
    )
    dataset = ImageDataset(csv, tmp_path, spatial_size=(32, 32), allow_skip_unreadable=True)
    item, label, _ = dataset[0]
    assert label == 1
