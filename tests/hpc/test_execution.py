from __future__ import annotations

import os
from pathlib import Path
import subprocess
import sys
from types import SimpleNamespace

import pytest
import torch
from torch.utils.data import DataLoader, TensorDataset

from src.utils import multiprocess
from src.utils.atomic import atomic_copy, atomic_torch_save
from src.utils.progress import progress


@pytest.mark.parametrize("mask,expected", [("2,7", ["7", "2"]),
    ("GPU-aa,GPU-bb", ["GPU-bb", "GPU-aa"]), ("MIG-aa,MIG-bb", ["MIG-bb", "MIG-aa"]),
    (None, ["1", "0"])])
def test_map_logical_indices_without_escaping_allocation(mask, expected):
    assert multiprocess._gpu_tokens([1, 0], 2, mask) == expected


@pytest.mark.parametrize("indices,count,mask", [([2], 2, "4,7"), ([0, 0], 1, "7"),
    ([-1], 1, "7"), ([0], 2, "7"), ([0], 1, "")])
def test_bad_gpu_selection_fails(indices, count, mask):
    with pytest.raises(ValueError):
        multiprocess._gpu_tokens(indices, count, mask)


def test_single_gpu_keeps_parent_scheduler_mask(monkeypatch):
    monkeypatch.setenv("CUDA_VISIBLE_DEVICES", "7")
    monkeypatch.setattr(torch.cuda, "is_available", lambda: True)
    monkeypatch.setattr(torch.cuda, "device_count", lambda: 1)
    devices, seen = [], []
    monkeypatch.setattr(torch.cuda, "set_device", devices.append)

    def task(multi_gpu=True):
        seen.append((os.environ["CUDA_VISIBLE_DEVICES"], multi_gpu))

    multiprocess.run_tasks_on_gpus([{"fn": task}], gpus=[0])
    assert seen == [("7", False)] and devices == [0]
    assert os.environ["CUDA_VISIBLE_DEVICES"] == "7"


def test_single_worker_failure_is_not_swallowed(monkeypatch):
    monkeypatch.setattr(torch.cuda, "is_available", lambda: False)

    def task(multi_gpu=True):
        raise ValueError("failed training")

    with pytest.raises(ValueError, match="failed training"):
        multiprocess.run_tasks_on_gpus([{"fn": task}], gpus=[])


def test_cpu_budget_respects_slurm_and_affinity(monkeypatch):
    monkeypatch.setattr(os, "sched_getaffinity", lambda _pid: {0, 1, 2, 3})
    monkeypatch.setenv("SLURM_CPUS_PER_TASK", "2")
    assert multiprocess._cpu_budget() == 2
    monkeypatch.setenv("SLURM_CPUS_PER_TASK", "12")
    assert multiprocess._cpu_budget() == 4


@pytest.mark.parametrize("mode", ["success", "failure", "death"])
def test_spawned_workers_report_failures_without_hanging(mode, tmp_path):
    environment = dict(os.environ, SLURM_CPUS_PER_TASK="2", CUDA_VISIBLE_DEVICES="",
                       OMP_NUM_THREADS="1", MKL_NUM_THREADS="1", OPENBLAS_NUM_THREADS="1")
    result = subprocess.run([sys.executable, "-m", "tests.hpc.worker_probe", mode, str(tmp_path)],
                            env=environment, capture_output=True, text=True, timeout=45)
    if mode == "success":
        assert result.returncode == 0, result.stdout + result.stderr
        assert len(list(tmp_path.glob("task-*"))) == 2
        assert all(path.read_text() == "" for path in tmp_path.glob("task-*"))
    else:
        assert result.returncode != 0
        assert "RuntimeError" in result.stderr


def test_atomic_copy_preserves_previous_file_on_copy_failure(tmp_path, monkeypatch):
    from src.utils import atomic
    source, target = tmp_path / "source", tmp_path / "target"
    source.write_bytes(b"new")
    target.write_bytes(b"valid")

    def fail(input_file, output, **_kwargs):
        output.write(b"partial")
        raise OSError("NFS unavailable")

    monkeypatch.setattr(atomic.shutil, "copyfileobj", fail)
    with pytest.raises(OSError):
        atomic_copy(source, target)
    assert target.read_bytes() == b"valid"
    assert source.read_bytes() == b"new"
    assert not list(tmp_path.glob(".target.*"))


def test_atomic_checkpoint_preserves_previous_file_on_serialization_failure(tmp_path, monkeypatch):
    target = tmp_path / "best.pth"
    atomic_torch_save({"x": torch.tensor([1.0])}, target)
    previous = target.read_bytes()

    def fail(_value, output):
        output.write(b"bad")
        raise OSError("disk full")

    monkeypatch.setattr(torch, "save", fail)
    with pytest.raises(OSError):
        atomic_torch_save({}, target)
    assert target.read_bytes() == previous
    assert not list(tmp_path.glob(".best.pth.*"))


def test_atomic_copy_and_checkpoint_roundtrip(tmp_path):
    target = tmp_path / "weights" / "best.pth"
    atomic_torch_save({"x": torch.tensor([2])}, target)
    copy = tmp_path / "published" / "best.pth"
    atomic_copy(target, copy)
    assert torch.load(copy, weights_only=True)["x"].item() == 2


def test_progress_does_not_change_iteration_or_write_terminal_controls(capsys):
    assert list(progress(range(3), "Train", interval=0)) == [0, 1, 2]
    text = capsys.readouterr().out
    assert "3/3 batches" in text and "batches/s" in text and "ETA" in text
    assert "\r" not in text and "\x1b" not in text


def test_actual_cpu_training_keeps_one_best_checkpoint_and_reports_metrics(tmp_path, capsys):
    from src.pipelines.config import TrainingConfig
    from src.pipelines.training import Trainer
    torch.set_num_threads(1)
    loader = DataLoader(TensorDataset(torch.randn(8, 3), torch.tensor([0, 1] * 4), torch.arange(8)), batch_size=4)
    model = torch.nn.Linear(3, 2)
    config = TrainingConfig(epochs=2, num_workers=0, multi_gpu=False, lr_head=0, lr_backbone=0)
    spec = SimpleNamespace(parameter_groups=lambda model, _config: [{"params": model.parameters(), "lr": 0}])
    result = Trainer(model, loader, loader, loader, config, tmp_path, spec, device="cpu").fit()
    assert result["y_true"].shape == (8,)
    assert [path.name for path in (tmp_path / "weights").iterdir()] == ["best.pth"]
    assert (tmp_path / "results" / "run_config.json").exists()
    assert (tmp_path / "results" / "outputs_test.npz").exists()
    text = capsys.readouterr().out
    assert "Effective training configuration:" in text and "Final test metrics:" in text
    assert "batches/s" in text
