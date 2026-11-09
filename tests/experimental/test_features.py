import json

import numpy as np
import pytest

from src.experimental.features import open_cache
from src.robustness.provenance import digest, digest_file


def write_fixture(root):
    np.save(root / "features.npy", np.ones((2, 4), dtype=np.float16))
    np.save(root / "logits.npy", np.array([[1., 0.], [0., 1.]], dtype=np.float32))
    (root / "rows.csv").write_text("sample_id,group_id,label,image_sha256\na,a,0,hasha\nb,b,1,hashb\n")
    key = {"checkpoint_sha256": "weights", "manifest_sha256": "manifest",
           "extraction": {"srm_on_normalized": True}, "commit": "commit"}
    meta = {"schema": "faceforgery-features-v1", "status": "complete", "identity": digest(key),
            "key": key, "sample_ids_sha256": digest(["a", "b"]),
            "files": {n: digest_file(root / n) for n in ["features.npy", "logits.npy", "rows.csv"]}}
    (root / "cache.json").write_text(json.dumps(meta))
    return meta


def test_cache_preserves_types_and_rejects_key_changes(tmp_path):
    meta = write_fixture(tmp_path)
    store = open_cache(tmp_path, meta["identity"])
    assert store.features.dtype == np.float16
    assert store.logits.dtype == np.float32
    assert store.frame.sample_id.tolist() == ["a", "b"]
    with pytest.raises(ValueError, match="key mismatch"):
        open_cache(tmp_path, "different preprocessing or checkpoint")
    meta["key"]["extraction"]["srm_on_normalized"] = False
    (tmp_path / "cache.json").write_text(json.dumps(meta))
    with pytest.raises(ValueError, match="key mismatch"):
        open_cache(tmp_path)


def test_cache_rejects_partial_and_changed_arrays(tmp_path):
    meta = write_fixture(tmp_path)
    meta["status"] = "writing"
    (tmp_path / "cache.json").write_text(json.dumps(meta))
    with pytest.raises(ValueError, match="Incomplete"):
        open_cache(tmp_path)
    write_fixture(tmp_path)
    np.save(tmp_path / "features.npy", np.zeros((2, 4), dtype=np.float16))
    with pytest.raises(ValueError, match="artifact changed"):
        open_cache(tmp_path)


def test_extracted_features_reproduce_checkpoint_logits(tmp_path):
    import pandas as pd
    import torch
    from PIL import Image
    from src.experimental.features import extract_features, final_linear
    from src.models.registry import MODEL_REGISTRY
    from src.pipelines.config import TrainingConfig
    from src.robustness.imaging import CanonicalDataset
    from src.robustness.legacy_encoding import encode_legacy_tensor
    from src.robustness.manifests import save_manifest

    torch.manual_seed(42)
    config = TrainingConfig(model_family="resnet", fourier_mode="srm", image_size=32,
                            regime="scratch", allow_pretrained=False)
    model = MODEL_REGISTRY["resnet"].build(config).eval()
    run = tmp_path / "models/resnet/srm/scratch/seed_42"
    (run / "weights").mkdir(parents=True)
    (run / "results").mkdir()
    (run / "results/run_config.json").write_text(json.dumps(config.to_dict()))
    checkpoint = run / "weights/best.pth"
    torch.save(model.state_dict(), checkpoint)
    images = tmp_path / "images"
    images.mkdir()
    rng = np.random.default_rng(5)
    for i in range(2):
        Image.fromarray(rng.integers(0, 256, (39, 47, 3), dtype=np.uint8)).save(images / f"{i}.png")
    frame = pd.DataFrame({"img_name": ["0.png", "1.png"], "label": [0, 1],
                          "sample_id": ["a", "b"], "group_id": ["a", "b"],
                          "dataset": ["fixture"] * 2, "split": ["val"] * 2})
    manifest = tmp_path / "val.csv"
    save_manifest(frame, manifest, {})
    cache = extract_features(checkpoint, manifest, images, tmp_path / "cache", batch_size=2, use_amp=False)
    store = open_cache(cache)
    ds = CanonicalDataset(frame, images, 32)
    raw = torch.stack([ds[i]["image"] for i in range(2)])
    inputs = encode_legacy_tensor(raw, "srm", 6)
    captured = []
    hook = final_linear(model)[1].register_forward_pre_hook(lambda module, args: captured.append(args[0].detach()))
    with torch.no_grad():
        expected = model(inputs).numpy()
    hook.remove()
    np.testing.assert_allclose(store.logits, expected, atol=1e-6, rtol=1e-6)
    np.testing.assert_array_equal(store.features, captured[0].numpy().astype(np.float16))
    assert extract_features(checkpoint, manifest, images, tmp_path / "cache", use_amp=False) == cache
