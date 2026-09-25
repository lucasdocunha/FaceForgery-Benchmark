"""Run CISIA workloads with local scratch and versioned final publication.

Only the standard library is imported here: cache locations must be established
before torch/transformers/timm are imported by the workload subprocess.
"""
from __future__ import annotations

import argparse
import csv
from datetime import datetime, timezone
import json
from importlib.metadata import PackageNotFoundError, version
import os
from pathlib import Path
import re
import shlex
import shutil
import signal
import socket
import subprocess
import sys
import tempfile
import time
import uuid

FAMILIES = ("resnet", "xception", "mobilenet", "vit", "clip", "dino")
CACHE_PATHS = {
    "HF_HOME": "hf", "HF_HUB_CACHE": "hf/hub", "HUGGINGFACE_HUB_CACHE": "hf/hub",
    "HF_ASSETS_CACHE": "hf/assets", "HF_XET_CACHE": "hf/xet", "TRANSFORMERS_CACHE": "hf/hub",
    "TORCH_HOME": "torch", "PIP_CACHE_DIR": "pip", "XDG_CACHE_HOME": "xdg",
    "MPLCONFIGDIR": "matplotlib", "TRITON_CACHE_DIR": "triton", "CUDA_CACHE_PATH": "cuda",
    "NUMBA_CACHE_DIR": "numba", "TMPDIR": "tmp",
}


def require_under(path: Path, roots: tuple[Path, ...], label: str) -> Path:
    """Resolve symlinks before validating storage policy."""
    resolved = path.expanduser().resolve()
    if not any(resolved.is_relative_to(root.resolve()) for root in roots):
        raise ValueError(f"{label} must be under {', '.join(map(str, roots))}: {resolved}")
    return resolved


def cache_environment(workspace: Path, inherited: dict[str, str]) -> dict[str, str]:
    environment = dict(inherited)
    for variable, relative in CACHE_PATHS.items():
        path = workspace / "cache" / relative
        path.mkdir(parents=True, exist_ok=True)
        environment[variable] = str(path)
    environment.update(PYTHONUNBUFFERED="1", PYTHONDONTWRITEBYTECODE="1", MPLBACKEND="Agg",
                       OMP_NUM_THREADS="1", MKL_NUM_THREADS="1", OPENBLAS_NUM_THREADS="1",
                       NUMEXPR_NUM_THREADS="1", TCC_JOB_DIR=str(workspace))
    # Do not inherit the former scripts' blanket TLS bypass.
    environment.pop("PYTHONHTTPSVERIFY", None)
    return environment


def publish_tree(source: Path, destination: Path) -> None:
    """Publish a new version using staging on the destination filesystem."""
    destination.parent.mkdir(parents=True, exist_ok=True)
    if destination.exists():
        raise FileExistsError(f"Refusing to overwrite published version: {destination}")
    staging = Path(tempfile.mkdtemp(prefix=f".{destination.name}.partial-", dir=destination.parent))
    try:
        shutil.copytree(source, staging, dirs_exist_ok=True)
        # Surface delayed write errors before treating the copy as durable.
        for path in staging.rglob("*"):
            if path.is_file():
                with path.open("rb") as handle:
                    os.fsync(handle.fileno())
        if destination.exists():
            raise FileExistsError(destination)
        staging.rename(destination)
    finally:
        if staging.exists():
            shutil.rmtree(staging)


def publish_outputs(workspace: Path, models_destination: Path, reports_destination: Path,
                    metadata: dict, *, pretrained: bool = False) -> None:
    """Copy reports without weights; preserve the existing trained-run layout."""
    models = workspace / ("pretrained" if pretrained else "models")
    reports = workspace / "outputs"
    models.mkdir(exist_ok=True)
    reports.mkdir(exist_ok=True)
    if not pretrained:
        for folder in models.glob("*/*/*/seed_*/*"):
            if folder.is_dir() and folder.name in {"results", "plots"}:
                shutil.copytree(folder, reports / "runs" / folder.relative_to(models), dirs_exist_ok=True)
    config_paths = list(models.glob("*/*/*/seed_*/results/run_config.json"))
    metadata["run_configs"] = {str(path.relative_to(models)): json.loads(path.read_text()) for path in config_paths}
    metrics = {}
    for path in models.glob("*/*/*/seed_*/results/metrics_*.csv"):
        with path.open(newline="", encoding="utf-8") as handle:
            metrics[str(path.relative_to(models))] = list(csv.DictReader(handle))
    metadata["metrics"] = metrics
    document = json.dumps(metadata, indent=2, sort_keys=True)
    for root in (models, reports):
        (root / "JOB.json").write_text(document + "\n", encoding="utf-8")
    card = ("# FaceForgery — CISIA\n\n"
            "Status, dataset, commit, job, arguments, effective hyperparameters and metrics:\n\n"
            f"```json\n{document}\n```\n\n"
            "The original per-seed results/config layout is retained beside weights for compatibility.\n"
            "Predictions, tables and plots are also published under the project's saidas directory.\n"
            "A nonzero workload exit code denotes partial/failed training, not a completed benchmark.\n"
            "SLURM may additionally report failure if final publication/cleanup fails; consult its logs.\n")
    (models / "MODELO.md").write_text(card, encoding="utf-8")
    publish_tree(models, models_destination)
    publish_tree(reports, reports_destination)
    print(f"Published models: {models_destination}", flush=True)
    print(f"Published reports: {reports_destination}", flush=True)
    print(f"Files: models={sum(p.is_file() for p in models.rglob('*'))}, "
          f"reports={sum(p.is_file() for p in reports.rglob('*'))}", flush=True)


def run_command(command: list[str], environment: dict[str, str], project: Path) -> int:
    """Forward termination only to this workload's process group."""
    process = subprocess.Popen(command, cwd=project, env=environment, start_new_session=True)
    interrupted = []

    def forward(signum, _frame):
        interrupted.append(signum)
        try:
            os.killpg(process.pid, signum)
        except ProcessLookupError:
            pass

    previous = {sig: signal.signal(sig, forward) for sig in (signal.SIGTERM, signal.SIGINT)}
    try:
        while process.poll() is None:
            if interrupted:
                try:
                    process.wait(timeout=20)
                except subprocess.TimeoutExpired:
                    os.killpg(process.pid, signal.SIGKILL)
                    process.wait()
                break
            time.sleep(0.1)
        return 128 + interrupted[0] if interrupted else (process.returncode if process.returncode >= 0 else 128 - process.returncode)
    finally:
        if process.poll() is None:
            os.killpg(process.pid, signal.SIGTERM)
            try:
                process.wait(timeout=5)
            except subprocess.TimeoutExpired:
                os.killpg(process.pid, signal.SIGKILL)
                process.wait()
        for sig, handler in previous.items():
            signal.signal(sig, handler)


def build_command(kind: str, family: str | None, arguments: list[str], cpus: int) -> list[str]:
    if kind == "pretrained":
        if family is not None or arguments:
            raise ValueError("Pretrained job accepts no positional arguments; use TCC_PRETRAINED_ROOT as source")
        return [sys.executable, "-u", "scripts/setup_pretrained_cisia.py"]
    if family not in FAMILIES:
        raise ValueError(f"Unknown family: {family}")
    if kind == "matrix":
        regime = arguments[0] if arguments else "scratch"
        workers = int(arguments[1]) if len(arguments) > 1 else 1
        if regime not in {"scratch", "finetune"} or workers < 1 or workers > cpus:
            raise ValueError("Expected scratch|finetune and 1 <= workers-per-gpu <= allocated CPUs")
        loaders = min(4, max(0, cpus // workers - 1))
        return [sys.executable, "-u", "run_matrix.py", "--regime", regime, "--only", family,
                "--workers-per-gpu", str(workers), "--num-workers", str(loaders), *arguments[2:]]
    seeds = arguments[0] if arguments else "42,123,2024,7,2025"
    regime = arguments[1] if len(arguments) > 1 else "finetune_robust"
    workers = int(arguments[2]) if len(arguments) > 2 else min(4, max(0, cpus - 1))
    if not re.fullmatch(r"\d+(,\d+)*", seeds) or regime not in {"scratch_robust", "finetune_robust"}:
        raise ValueError("Expected comma-separated integer seeds and scratch_robust|finetune_robust")
    if not 0 <= workers < cpus:
        raise ValueError("num-workers must be non-negative and leave one allocated CPU for the trainer")
    return [sys.executable, "-u", "scripts/train_robust_seeds.py", "--family", family,
            "--seeds", seeds, "--regime", regime, "--num-workers", str(workers), *arguments[3:]]


def stage_pretrained(workspace: Path, sources: list[Path], *, copy: bool) -> Path:
    """Prefer shared cluster weights, then an explicitly supplied/user source."""
    target = workspace / "pretrained"
    target.mkdir()
    for family in FAMILIES:
        for source in sources:
            directory = source / family
            if directory.is_dir():
                if copy:
                    shutil.copytree(directory, target / family)
                else:
                    (target / family).symlink_to(directory.resolve(), target_is_directory=True)
                print(f"Pretrained {family}: {directory} (read-only source)", flush=True)
                break
    return target


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("kind", choices=("matrix", "robust", "pretrained"))
    parser.add_argument("family", nargs="?")
    parser.add_argument("arguments", nargs=argparse.REMAINDER)
    args = parser.parse_args(argv)
    environment = dict(os.environ)
    job = environment.get("SLURM_JOB_ID", "")
    if not re.fullmatch(r"\d+", job):
        parser.error("CISIA workloads must run in a sbatch/srun allocation, never directly in Shell Access")
    project = Path(environment.get("TCC_PROJECT_ROOT", environment.get("SLURM_SUBMIT_DIR", "."))).resolve()
    if not (project / "train.py").is_file():
        parser.error(f"Not a FaceForgery checkout: {project}; set TCC_PROJECT_ROOT or submit from the repo root")
    require_under(project, (Path.home(),), "Project")
    user_models = Path("/projects/models") / environment["USER"]
    final_root = require_under(Path(environment.get("CISIA_MODELS_ROOT", str(user_models / "faceforgery"))),
                               (user_models,), "Final models")
    output_root = require_under(Path(environment.get("CISIA_OUTPUT_ROOT", str(project / "saidas"))),
                                (project / "saidas",), "Reports")
    data = require_under(Path(environment.get("TCC_DATASET_ROOT", "/datasets/Images/MFFI")),
                         (Path("/datasets"),), "Input dataset")
    if args.kind != "pretrained" and not data.is_dir():
        parser.error(f"Dataset does not exist: {data}")
    cpus = int(environment.get("SLURM_CPUS_PER_TASK", "1"))
    if cpus < 1:
        parser.error("SLURM_CPUS_PER_TASK must be positive")
    command = build_command(args.kind, args.family, args.arguments, cpus)
    base = require_under(Path(environment.get("TMPDIR", f"/scratch/{environment['USER']}")),
                         (Path("/scratch"), Path("/tmp")), "Job-local scratch")
    base.mkdir(parents=True, exist_ok=True)
    free = shutil.disk_usage(base).free
    minimum = float(environment.get("CISIA_MIN_FREE_GB", "20")) * 1024 ** 3
    if minimum < 0:
        parser.error("CISIA_MIN_FREE_GB must be non-negative")
    if free < minimum:
        parser.error(f"Insufficient scratch space: {free / 1024**3:.1f} GiB available; need {minimum / 1024**3:.1f}")
    started = time.monotonic()
    workspace = Path(tempfile.mkdtemp(prefix=f"job_{job}_", dir=base))
    status, published = 1, False
    tag = re.sub(r"[^A-Za-z0-9_.-]", "_", environment.get("SLURM_JOB_NAME", "faceforgery"))
    tag += f"_{datetime.now(timezone.utc):%Y-%m-%d}_{job}_{uuid.uuid4().hex[:8]}"
    metadata = {"job_id": job, "name": environment.get("SLURM_JOB_NAME"), "node": socket.gethostname(),
                "partition": environment.get("SLURM_JOB_PARTITION"), "conda_env": environment.get("CONDA_DEFAULT_ENV"),
                "started_utc": datetime.now(timezone.utc).isoformat(), "command": command,
                "dataset": str(data), "cpus": cpus, "workspace": str(workspace)}
    try:
        env = cache_environment(workspace, environment)
        env.update(TCC_DATASET_ROOT=str(data), TCC_DATA_ROOT=environment.get("TCC_DATA_ROOT", str(project / "data")),
                   TCC_MODELS_ROOT=str(workspace / "models"), TCC_OUTPUT_ROOT=str(workspace / "outputs"))
        for directory in (workspace / "models", workspace / "outputs"):
            directory.mkdir()
        shared = Path(environment.get("CISIA_SHARED_PRETRAINED_ROOT", "/datasets/ai_models/faceforgery/pretrained"))
        sources = [shared, Path(environment.get("TCC_PRETRAINED_ROOT", str(user_models / "pretrained")))]
        for source in sources:
            require_under(source, (Path("/datasets/ai_models"), user_models), "Pretrained source")
        env["TCC_PRETRAINED_ROOT"] = str(stage_pretrained(workspace, sources, copy=args.kind == "pretrained"))
        resume = environment.get("CISIA_RESUME_FROM")
        if resume and args.kind != "pretrained":
            previous = require_under(Path(resume), (user_models,), "Resume source")
            shutil.copytree(previous, workspace / "models", dirs_exist_ok=True)
            metadata["resume_from"] = str(previous)
        commit = subprocess.run(["git", "rev-parse", "HEAD"], cwd=project, text=True, capture_output=True, check=True)
        metadata["commit"] = commit.stdout.strip()
        metadata["python"] = sys.version
        metadata["packages"] = {}
        for package in ("torch", "torchvision", "timm", "transformers", "numpy", "pandas", "scikit-learn"):
            try:
                metadata["packages"][package] = version(package)
            except PackageNotFoundError:
                metadata["packages"][package] = "NOT INSTALLED"
        metadata["pretrained_sources"] = [str(path) for path in sources]
        print("=== CISIA job ===\n" + json.dumps(metadata, indent=2), flush=True)
        print(f"Scratch free: {free / 1024**3:.1f} GiB | command: {shlex.join(command)}", flush=True)
        # CUDA/env validation is itself inside the allocation, never the login shell.
        if args.kind != "pretrained":
            check = [sys.executable, "-u", "-c", "import torch; "
                     "assert torch.cuda.is_available(), 'Allocated GPU is not available to PyTorch'; "
                     "print('PyTorch', torch.__version__, 'CUDA', torch.version.cuda, "
                     "'GPU', torch.cuda.get_device_name(0), flush=True)"]
            subprocess.run(check, cwd=project, env=env, check=True)
        status = run_command(command, env, project)
    except Exception as error:
        print(f"CISIA job failed: {error}", file=sys.stderr, flush=True)
    finally:
        metadata.update(workload_exit_code=status, duration_seconds=round(time.monotonic() - started, 2),
                        finished_utc=datetime.now(timezone.utc).isoformat())
        try:
            publish_outputs(workspace, final_root / tag, output_root / tag, metadata,
                            pretrained=args.kind == "pretrained")
            published = True
        except Exception as error:
            status = status or 1
            print(f"Publication failed: {error}\nRECOVERY: retained artifacts on node {socket.gethostname()} "
                  f"at {workspace}; copy them before deleting this directory.", file=sys.stderr, flush=True)
        # Caches are always removed. Unpublished checkpoints are never discarded.
        shutil.rmtree(workspace / "cache", ignore_errors=True)
        if published:
            shutil.rmtree(workspace)
    return status


def cli() -> None:
    started, status = time.monotonic(), 1
    print(f"CISIA start: job={os.environ.get('SLURM_JOB_ID', 'unallocated')} "
          f"node={socket.gethostname()} time={datetime.now(timezone.utc).isoformat()}", flush=True)
    try:
        status = main()
    except SystemExit as error:
        status = error.code if isinstance(error.code, int) else 1
    except Exception as error:
        print(f"CISIA preflight failed: {error}", file=sys.stderr, flush=True)
    finally:
        print(f"=== finished | duration={time.monotonic() - started:.1f}s | exit={status}", flush=True)
    raise SystemExit(status)


if __name__ == "__main__":
    cli()
