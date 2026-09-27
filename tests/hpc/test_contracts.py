from __future__ import annotations

import hashlib
import importlib
import io
import os
from pathlib import Path
import ssl
import subprocess
import urllib.request

import pytest

ROOT = Path(__file__).resolve().parents[2]


def test_bad_explicit_dataset_root_does_not_fall_back(tmp_path, monkeypatch):
    from src.data.paths import phase1_split_root
    monkeypatch.setenv("TCC_DATASET_ROOT", str(tmp_path / "missing"))
    with pytest.raises(FileNotFoundError, match="TCC_DATASET_ROOT"):
        phase1_split_root("train")


@pytest.mark.parametrize("name", ["/home/old-machine/image.jpg", "../../outside/image.jpg"])
def test_hpc_csv_cannot_override_dataset_storage_with_workstation_paths(tmp_path, monkeypatch, name):
    from src.data.data import ImageDataset
    manifest = tmp_path / "images.csv"
    manifest.write_text(f"img_name,label\n{name},1\n")
    monkeypatch.setenv("TCC_JOB_DIR", str(tmp_path))
    with pytest.raises(ValueError, match="outside /datasets"):
        ImageDataset(manifest, Path("/datasets/mffi"))


def test_setup_import_keeps_tls_verification_and_has_no_downloads(monkeypatch):
    original = ssl._create_default_https_context

    def forbidden(*_args, **_kwargs):
        pytest.fail("Import triggered a download")

    monkeypatch.setattr(urllib.request, "urlopen", forbidden)
    module = importlib.import_module("scripts.setup_pretrained_cisia")
    importlib.reload(module)
    importlib.reload(importlib.import_module("scripts.preload_models_cisia"))
    assert ssl._create_default_https_context is original


@pytest.mark.parametrize("correct", [True, False])
def test_download_verifies_hash_and_does_not_overwrite_on_mismatch(tmp_path, monkeypatch, correct):
    from scripts.setup_pretrained_cisia import download_file_safely
    payload = b"pretend model weights"
    digest = hashlib.sha256(payload).hexdigest()[:8] if correct else "00000000"
    url = f"https://download.pytorch.org/models/model-{digest}.pth"
    target = tmp_path / "model.pth"
    target.write_bytes(b"previous valid checkpoint")
    monkeypatch.setattr(urllib.request, "urlopen", lambda _request, timeout: io.BytesIO(payload))
    if correct:
        download_file_safely(url, target)
        assert target.read_bytes() == payload
    else:
        with pytest.raises(ValueError, match="SHA-256"):
            download_file_safely(url, target)
        assert target.read_bytes() == b"previous valid checkpoint"


def fake_sbatch(tmp_path, *, fail_on=0):
    binary = tmp_path / "sbatch"
    calls = tmp_path / "calls"
    binary.write_text(f'''#!/bin/bash
set -eu
printf '%s|%s\\n' "$PWD" "$*" >> "{calls}"
count=$(wc -l < "{calls}")
[[ "$count" != "{fail_on}" ]] || exit 19
echo "$count"
''')
    binary.chmod(0o755)
    return dict(os.environ, PATH=f"{tmp_path}:{os.environ['PATH']}"), calls


def test_submit_stops_at_first_scheduler_failure(tmp_path):
    environment, calls = fake_sbatch(tmp_path, fail_on=2)
    result = subprocess.run(["bash", str(ROOT / "scripts/submit_all_cisia.sh")],
                            cwd=tmp_path, env=environment, capture_output=True, text=True)
    assert result.returncode == 19
    assert len(calls.read_text().splitlines()) == 2
    assert "xception submitted" not in result.stdout


def test_invalid_family_submits_nothing(tmp_path):
    environment, calls = fake_sbatch(tmp_path)
    result = subprocess.run(["bash", str(ROOT / "scripts/submit_all_robust_cisia.sh"), "resnet,typo"],
                            cwd=tmp_path, env=environment, capture_output=True, text=True)
    assert result.returncode == 2 and not calls.exists()


def test_submit_from_other_working_directory_is_spool_safe(tmp_path):
    environment, calls = fake_sbatch(tmp_path)
    result = subprocess.run(["bash", str(ROOT / "scripts/submit_all_robust_cisia.sh"), "resnet"],
                            cwd=tmp_path, env=environment, capture_output=True, text=True)
    assert result.returncode == 0, result.stderr
    assert calls.read_text().startswith(str(ROOT) + "|")
    assert f"--chdir={ROOT}" in calls.read_text()
    assert (ROOT / "logs").is_dir()


@pytest.mark.parametrize("robust", [True, False])
def test_all_model_wrappers_follow_the_same_hpc_contract(robust):
    from src.hpc.runtime import FAMILIES
    for family in FAMILIES:
        script = ROOT / "scripts" / f"slurm_{'robust_' if robust else ''}{family}.sh"
        text = script.read_text()
        assert text.startswith("#!/bin/bash -l\n")
        for directive in ("--gres=gpu:1", "--cpus-per-task=8", "--mem=64G", "--time=48:00:00",
                          "--output=logs/%x_%j.out", "--error=logs/%x_%j.err"):
            assert f"#SBATCH {directive}" in text
        assert "SLURM_SUBMIT_DIR" in text and "cisia_common.sh" in text
        assert "BASH_SOURCE" not in text
        assert "CUDA_VISIBLE_DEVICES=" not in text
        assert "PYTHONHTTPSVERIFY" not in text
        subprocess.run(["bash", "-n", str(script)], check=True)


def test_pretrained_job_does_not_reserve_an_unused_gpu():
    for name in ("download_pretrained_cisia.sh", "slurm_download_pretrained.sh"):
        text = (ROOT / "scripts" / name).read_text()
        assert "#SBATCH --gres" not in text
        assert "--cpus-per-task=8" in text and "--mem=32G" in text
