import copy
import json
from pathlib import Path
import random
import sqlite3

import numpy as np
import pandas as pd
from PIL import Image
import pytest
import torch

from src.experimental.sbi.data import SBIDataset, blend, face_mask
from src.experimental.sbi.cache import LandmarkStore
from src.experimental.sbi.training import describe_run, fit, load_model
from src.robustness.inference import predict
from src.robustness.manifests import load_manifest, save_manifest
from src.robustness.provenance import digest_file, write_json


@pytest.fixture
def sbi_source(tmp_path):
    data, records = {}, []
    for split in ("train", "val"):
        root = tmp_path / split
        root.mkdir()
        rows = []
        for index in range(8):
            name = f"{split}-{index}"
            rng = np.random.default_rng(index)
            pixels = rng.integers(40, 210, size=(32,32,3), dtype=np.uint8)
            path = root / f"{name}.png"
            Image.fromarray(pixels).save(path)
            rows.append(dict(sample_id=name, img_name=path.name, label=index%2,
                             group_id=name, dataset="min-sbi-fixture", split=split))
            if split == "train" and index%2 == 0:
                records.append(dict(sample_id=name, img_name=path.name, label=0, status="ok",
                    image_sha256=digest_file(path), landmarks=[[.2,.3],[.4,.15],[.7,.25],[.8,.6],[.5,.85],[.2,.7]]))
        manifest = tmp_path / f"{split}.csv"
        certificate = save_manifest(pd.DataFrame(rows), manifest, {})
        data[split+"_manifest"], data[split+"_root"] = str(manifest), str(root)
        if split == "train":
            tc = certificate
    path = tmp_path / "landmarks.json"
    write_json(path, dict(schema="faceforgery-landmarks-v1", source="external_normalized_landmarks",
                         coordinates="normalized_xy", manifest_sha256=tc["manifest_sha256"], records=records))
    data["landmarks"] = str(path)
    return dict(name="sbi-smoke", seed=42, output_dir=str(tmp_path/"run"), data=data,
                model=dict(architecture="resnet18", initialization="scratch", image_size=32, train_backbone=False),
                training=dict(epochs=1, batch_size=4, grad_accum_steps=2, bootstrap_draws=20, post_augment=False))


def dataset(cfg, **kw):
    frame, cert = load_manifest(cfg["data"]["train_manifest"])
    return SBIDataset(frame, cfg["data"]["train_root"], cfg["data"]["landmarks"],
                      manifest_sha256=cert["manifest_sha256"], image_size=32, **kw)


def test_mask_blending_pairing_and_deterministic_epochs(sbi_source):
    ds = dataset(sbi_source, arm="sbi", cache_images=True)
    assert len(ds) == 8
    a, b = ds[0], ds[1]
    assert a["label"] == 0 and b["label"] == 1
    assert a["base_sample_id"] == b["base_sample_id"] and a["group_id"] == b["group_id"]
    assert b["kind"] == "sbi" and not torch.equal(a["image"], b["image"])
    assert torch.equal(b["image"], ds[1]["image"])
    ds.set_epoch(1)
    assert not torch.equal(b["image"], ds[1]["image"])
    ds.set_epoch(0)
    assert torch.equal(b["image"], ds[1]["image"])
    real = ds._image(ds.real.iloc[0])
    output, alpha = blend(real, ds.masks[ds.real.iloc[0].sample_id], random.Random(42))
    assert 0 <= alpha.min() < alpha.max() <= 1
    assert .005 < (alpha > 0).mean() < 1
    assert np.abs(np.asarray(real,dtype=float)-np.asarray(output)).sum() > 0
    with pytest.raises(ValueError, match="Degenerate"):
        face_mask([[0,0],[.3,.3],[.6,.6]], 32)


def test_three_arms_have_equal_budget_and_no_mffi_reads_for_clean_arm(sbi_source):
    arms = [dataset(sbi_source, arm=name) for name in ("sbi", "mffi", "mixed")]
    assert all(len(ds) == 8 and sum(ds[i]["label"] for i in range(8)) == 4 for ds in arms)
    assert {arms[0][i]["kind"] for i in range(8)} == {"real", "sbi"}
    assert {arms[1][i]["kind"] for i in range(8)} == {"real", "mffi"}
    assert {arms[2][i]["kind"] for i in range(8)} == {"real", "sbi", "mffi"}
    for row in arms[0].fake.itertuples(index=False):
        (Path(sbi_source["data"]["train_root"])/row.img_name).unlink()
    clean = dataset(sbi_source, arm="sbi", cache_images=True)
    assert all(clean[i]["image"].shape == (3,32,32) for i in range(8))


def test_landmark_failure_is_explicit_and_exclusion_recorded(sbi_source):
    path = Path(sbi_source["data"]["landmarks"])
    cache = json.loads(path.read_text())
    cache["records"][0]["status"] = "no_face"
    write_json(path, cache)
    with pytest.raises(ValueError, match="Landmark failures"):
        dataset(sbi_source)
    ds = dataset(sbi_source, failure_policy="exclude")
    assert len(ds) == 6 and ds.cohort["failures"] == {"train-0":"no_face"}
    cache["manifest_sha256"] = "0"*64
    write_json(path, cache)
    with pytest.raises(ValueError, match="bind"):
        dataset(sbi_source)


def test_sqlite_geometry_is_lazy_and_identical(sbi_source, tmp_path):
    cache = json.loads(Path(sbi_source["data"]["landmarks"]).read_text())
    records = cache.pop("records")
    path = tmp_path / "faces.sqlite"
    with sqlite3.connect(path) as db:
        db.execute("CREATE TABLE faces (sample_id TEXT PRIMARY KEY,status TEXT,image_sha256 TEXT,label INTEGER,record TEXT)")
        db.execute("CREATE TABLE metadata (key TEXT PRIMARY KEY,value TEXT)")
        for row in records:
            db.execute("INSERT INTO faces VALUES (?,?,?,?,?)", (row["sample_id"],row["status"],row["image_sha256"],0,json.dumps(row)))
        db.execute("INSERT INTO metadata VALUES (?,?)", ("manifest",json.dumps(cache)))
    store = LandmarkStore(path)
    assert store.records is None and "landmarks" not in store.catalog["train-0"]
    assert store["train-0"] == records[0]
    original = dataset(sbi_source, post_augment=False)
    sbi_source["data"]["landmarks"] = str(path)
    bounded = dataset(sbi_source, post_augment=False)
    assert not bounded.images and not bounded.masks
    assert torch.equal(original[1]["image"], bounded[1]["image"])


@pytest.mark.parametrize("arm", ["sbi", "mffi", "mixed"])
def test_fit_calibrate_offline_reload_all_arms(sbi_source, arm, tmp_path):
    sbi_source["training"]["arm"] = arm
    result = fit(sbi_source)
    root = Path(result["run_dir"])
    description = describe_run(root)
    assert len(description["input_contract"]["bundle_sha256"]) == 64
    assert result["validation"]["auc"] is not None and result["global_step"] == 1
    saved = pd.read_csv(root/"validation_predictions.csv")
    model = load_model(root)
    frame, _ = load_manifest(sbi_source["data"]["val_manifest"])
    reloaded = predict(model, frame, sbi_source["data"]["val_root"], image_size=32, batch_size=4)
    np.testing.assert_allclose(saved.p_fake, reloaded.p_fake, atol=1e-7)
    assert fit(sbi_source, resume=True)["already_complete"]
    (root/"best.pt").write_bytes(b"changed")
    with pytest.raises(ValueError, match="artifact"):
        load_model(root)


def test_source_overlap_and_test_training_rejected(sbi_source):
    cfg = copy.deepcopy(sbi_source)
    cfg["data"]["train_manifest"] = cfg["data"]["val_manifest"]
    with pytest.raises(ValueError, match="source train"):
        fit(cfg)


def test_hf_adaptation_preserves_scores_and_reload_needs_no_initial_weights(sbi_source, tmp_path):
    from dataclasses import asdict
    from src.pipelines.config import TrainingConfig
    from src.models.registry import MODEL_REGISTRY
    from src.experimental.sbi.training import _initialize, normalize_config
    from src.robustness.legacy_encoding import encode_legacy_tensor
    parent = tmp_path / "models/mobilenet/srm/scratch/seed_42"
    (parent/"weights").mkdir(parents=True)
    (parent/"results").mkdir()
    source = TrainingConfig(model_family="mobilenet", fourier_mode="srm", regime="scratch",
                            allow_pretrained=False, variant="small", image_size=32)
    original = MODEL_REGISTRY["mobilenet"].build(source).eval()
    torch.save(original.state_dict(), parent/"weights/best.pth")
    write_json(parent/"results/run_config.json", asdict(source))
    sbi_source["model"].update(initialization="hf_mffi", weights=str(parent/"weights/best.pth"))
    adapted, _ = _initialize(normalize_config(sbi_source))
    raw = torch.rand(2,3,32,32)
    adapted.eval()
    torch.testing.assert_close(adapted(raw), original(encode_legacy_tensor(raw,"srm",6)), atol=0, rtol=0)
    fit(sbi_source)
    (parent/"weights/best.pth").unlink()
    descriptor = describe_run(sbi_source["output_dir"])
    assert descriptor["run_record"]["config"]["provenance"]["prior_mffi_fake_exposure"]
    assert load_model(sbi_source["output_dir"])(raw).shape == (2,2)
