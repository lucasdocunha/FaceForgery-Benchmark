from __future__ import annotations

import json
import os
from pathlib import Path
import subprocess
import sys
from types import SimpleNamespace

import pytest

from src.hpc import runtime


def test_cache_paths_are_job_local_and_mask_is_preserved(tmp_path):
    environment = runtime.cache_environment(tmp_path, {"CUDA_VISIBLE_DEVICES": "3", "PYTHONHTTPSVERIFY": "0",
                                                     "HF_HUB_CACHE": "/home/shared-cache"})
    for key in runtime.CACHE_PATHS:
        assert Path(environment[key]).is_relative_to(tmp_path)
        assert Path(environment[key]).is_dir()
    assert environment["CUDA_VISIBLE_DEVICES"] == "3"
    assert "PYTHONHTTPSVERIFY" not in environment
    assert environment["OMP_NUM_THREADS"] == "1"


def test_storage_validation_rejects_symlink_escape(tmp_path):
    allowed, outside = tmp_path / "local", tmp_path / "nfs"
    allowed.mkdir()
    outside.mkdir()
    (allowed / "escape").symlink_to(outside, target_is_directory=True)
    with pytest.raises(ValueError, match="must be under"):
        runtime.require_under(allowed / "escape" / "cache", (allowed,), "Cache")
    assert runtime.require_under(allowed / "job", (allowed,), "Cache") == allowed / "job"


def test_publish_is_new_version_only_and_cleans_staging(tmp_path):
    source, final = tmp_path / "local", tmp_path / "versions" / "job1"
    source.mkdir()
    (source / "weights.pth").write_bytes(b"valid")
    runtime.publish_tree(source, final)
    assert (final / "weights.pth").read_bytes() == b"valid"
    with pytest.raises(FileExistsError):
        runtime.publish_tree(source, final)
    assert not list(final.parent.glob("*.partial-*"))


def test_publication_failure_retains_source(tmp_path, monkeypatch):
    source = tmp_path / "local"
    source.mkdir()
    (source / "checkpoint").write_bytes(b"valid")

    def fail(*_args, **_kwargs):
        raise OSError("NFS failed")

    monkeypatch.setattr(runtime.shutil, "copytree", fail)
    with pytest.raises(OSError, match="NFS"):
        runtime.publish_tree(source, tmp_path / "out" / "v1")
    assert (source / "checkpoint").read_bytes() == b"valid"
    assert not (tmp_path / "out" / "v1").exists()
    assert not list((tmp_path / "out").iterdir())


def populate(workspace: Path):
    run = workspace / "models" / "resnet" / "none" / "scratch" / "seed_42"
    (run / "weights").mkdir(parents=True, exist_ok=True)
    (run / "weights" / "best.pth").write_bytes(b"checkpoint")
    (run / "results").mkdir()
    (run / "results" / "run_config.json").write_text('{"seed": 42}')
    (run / "results" / "metrics_test.csv").write_text("acc,f1,auc\n0.8,0.7,0.9\n")
    return run


def test_publish_preserves_evaluation_layout_and_excludes_weights_from_reports(tmp_path):
    workspace = tmp_path / "job"
    workspace.mkdir()
    run = populate(workspace)
    models, reports = tmp_path / "published", tmp_path / "reports"
    runtime.publish_outputs(workspace, models, reports, {"exit_code": 0, "commit": "abc"})
    relative = run.relative_to(workspace / "models")
    assert (models / relative / "weights" / "best.pth").exists()
    assert (models / relative / "results" / "run_config.json").exists()
    assert (reports / "runs" / relative / "results" / "metrics_test.csv").exists()
    assert not list(reports.rglob("*.pth"))
    card = (models / "MODELO.md").read_text()
    assert '"seed": 42' in card and '"acc": "0.8"' in card
    assert json.loads((reports / "JOB.json").read_text())["exit_code"] == 0


def test_pretrained_precedence_is_read_only(tmp_path):
    shared, user, workspace = [tmp_path / name for name in ("shared", "user", "workspace")]
    for root in (shared, user):
        (root / "resnet").mkdir(parents=True)
        (root / "resnet" / "resnet18.pth").write_text(root.name)
    (user / "dino").mkdir()
    workspace.mkdir()
    target = runtime.stage_pretrained(workspace, [shared, user], copy=False)
    assert (target / "resnet").resolve() == shared / "resnet"
    assert (target / "dino").resolve() == user / "dino"
    assert not (target / "clip").exists()
    runtime.shutil.rmtree(workspace)
    assert (shared / "resnet" / "resnet18.pth").read_text() == "shared"


def test_pretrained_preparation_copies_before_writing(tmp_path):
    shared, workspace = tmp_path / "shared", tmp_path / "workspace"
    (shared / "resnet").mkdir(parents=True)
    (shared / "resnet" / "weights").write_text("original")
    workspace.mkdir()
    target = runtime.stage_pretrained(workspace, [shared], copy=True)
    (target / "resnet" / "weights").write_text("updated")
    assert (shared / "resnet" / "weights").read_text() == "original"


@pytest.mark.parametrize("kind,family,args,cpus", [
    ("matrix", "resnet", ["bad"], 8), ("matrix", "resnet", ["scratch", "0"], 8),
    ("robust", "resnet", ["42,", "finetune_robust", "4"], 8),
    ("robust", "resnet", ["42", "finetune_robust", "8"], 8),
    ("pretrained", "unexpected", [], 8), ("matrix", "unknown", [], 8),
])
def test_invalid_job_arguments_fail(kind, family, args, cpus):
    with pytest.raises(ValueError):
        runtime.build_command(kind, family, args, cpus)


def test_worker_budget_and_smoke_flags():
    command = runtime.build_command("matrix", "resnet", ["scratch", "2", "--epochs", "1"], 8)
    assert command[command.index("--num-workers") + 1] == "3"
    assert command[-2:] == ["--epochs", "1"]
    assert runtime.build_command("pretrained", None, [], 8)[-1] == "scripts/setup_pretrained_cisia.py"


def test_unallocated_entrypoint_refuses_work(monkeypatch):
    monkeypatch.delenv("SLURM_JOB_ID", raising=False)
    with pytest.raises(SystemExit) as exc:
        runtime.main(["matrix", "resnet"])
    assert exc.value.code == 2


def test_workload_exit_code_is_preserved(tmp_path):
    status = runtime.run_command([sys.executable, "-c", "raise SystemExit(7)"], dict(os.environ), tmp_path)
    assert status == 7


@pytest.mark.parametrize("workload_status,publication_fails", [(0, False), (7, False), (0, True)])
def test_job_lifecycle_cleanup_and_recovery(tmp_path, monkeypatch, capsys, workload_status, publication_fails):
    project, scratch, final, reports, data = [tmp_path / name for name in ("project", "scratch", "final", "reports", "data")]
    for path in (project, scratch, data):
        path.mkdir()
    (project / "train.py").touch()
    variables = {"SLURM_JOB_ID": "123", "SLURM_CPUS_PER_TASK": "8", "USER": "tester",
                 "TCC_PROJECT_ROOT": str(project), "TMPDIR": str(scratch), "CISIA_MIN_FREE_GB": "0",
                 "CISIA_MODELS_ROOT": str(final), "CISIA_OUTPUT_ROOT": str(reports), "TCC_DATASET_ROOT": str(data)}
    for key, value in variables.items():
        monkeypatch.setenv(key, value)
    for key in ("CISIA_RESUME_FROM", "TCC_PRETRAINED_ROOT"):
        monkeypatch.delenv(key, raising=False)
    # Storage policy itself is tested above; emulate cluster mounts in a temp tree.
    monkeypatch.setattr(runtime, "require_under", lambda path, _roots, _label: path)
    monkeypatch.setattr(runtime.subprocess, "run", lambda *_args, **_kwargs: SimpleNamespace(stdout="commit123\n"))

    def workload(_command, environment, _project):
        populate(Path(environment["TCC_JOB_DIR"]))
        return workload_status

    monkeypatch.setattr(runtime, "run_command", workload)
    if publication_fails:
        def fail(*_args, **_kwargs):
            raise OSError("NFS error")
        monkeypatch.setattr(runtime, "publish_outputs", fail)
    status = runtime.main(["matrix", "resnet"])
    if publication_fails:
        assert status != 0
        assert list(scratch.glob("job_123_*/models/*/*/*/seed_*/weights/best.pth"))
        assert not list(scratch.glob("job_123_*/cache"))
        assert "RECOVERY" in capsys.readouterr().err
    else:
        assert status == workload_status
        assert not list(scratch.iterdir())
        metadata = json.loads(next(final.glob("*/JOB.json")).read_text())
        assert metadata["workload_exit_code"] == workload_status and metadata["commit"] == "commit123"
