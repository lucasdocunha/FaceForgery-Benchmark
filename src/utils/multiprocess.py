"""Bounded task execution that respects the scheduler's CUDA visibility mask."""
from __future__ import annotations

import logging
import multiprocessing as mp
import os
import queue
import sys
import time
import traceback

import torch

logger = logging.getLogger(__name__)


def _cpu_budget() -> int:
    affinity = len(os.sched_getaffinity(0)) if hasattr(os, "sched_getaffinity") else (os.cpu_count() or 1)
    requested = os.environ.get("SLURM_CPUS_PER_TASK")
    return max(1, min(affinity, int(requested))) if requested else max(1, affinity)


def _gpu_tokens(gpus: list[int], count: int, mask: str | None) -> list[str]:
    """Map logical CUDA indices to inherited numeric, GPU UUID or MIG tokens."""
    if len(gpus) != len(set(gpus)) or any(index < 0 or index >= count for index in gpus):
        raise ValueError(f"gpus must be unique logical indices in [0, {count}); got {gpus}")
    tokens = [value.strip() for value in mask.split(",")] if mask is not None else [str(i) for i in range(count)]
    if len(tokens) < count or any(not token for token in tokens[:count]):
        raise ValueError("CUDA_VISIBLE_DEVICES is inconsistent with visible CUDA devices")
    return [tokens[index] for index in gpus]


def _invoke(task: dict) -> None:
    function = task["fn"]
    name = task.get("name", function.__name__)
    started = time.monotonic()
    logger.info("Starting task: %s", name)
    kwargs = dict(task.get("kwargs", {}))
    kwargs["multi_gpu"] = False
    function(*task.get("args", ()), **kwargs)
    logger.info("Completed task: %s in %.1fs", name, time.monotonic() - started)


def _worker_fn(tasks, results, gpu_token: str | None, log_level: int) -> None:
    # Spawned children have not initialized CUDA. Never replace a scheduler mask
    # with an unmapped logical index, and never modify the parent's mask.
    os.environ["CUDA_VISIBLE_DEVICES"] = gpu_token if gpu_token is not None else ""
    logging.basicConfig(level=log_level, stream=sys.stdout,
                        format="%(asctime)s %(levelname)s %(message)s", force=True)
    if gpu_token is not None:
        torch.cuda.set_device(0)
    while True:
        item = tasks.get()
        if item is None:
            return
        index, task = item
        try:
            _invoke(task)
        except Exception:
            error = traceback.format_exc()
            logger.error("Task failed: %s\n%s", task.get("name", index), error)
            results.put((index, error))
        else:
            results.put((index, None))


def run_tasks_on_gpus(tasks: list[dict], gpus: list[int] | None = None,
                      workers_per_gpu: int = 1) -> None:
    """Run tasks on *logical visible* GPUs; any task or worker failure raises.

    CPU fallback is bounded by CPU affinity/SLURM and task count. No queue.join()
    waits forever for task_done() from a worker killed by an OOM or a signal.
    """
    if workers_per_gpu < 1:
        raise ValueError("workers_per_gpu must be positive")
    if not tasks:
        logger.info("No pending tasks.")
        return
    count = torch.cuda.device_count() if torch.cuda.is_available() else 0
    selected = list(range(count)) if gpus is None else list(gpus)
    tokens = _gpu_tokens(selected, count, os.environ.get("CUDA_VISIBLE_DEVICES")) if selected else []
    worker_tokens = ([token for token in tokens for _ in range(workers_per_gpu)]
                     if tokens else [None] * _cpu_budget())[:len(tasks)]

    # Single-GPU sequential mode avoids IPC and keeps the already initialized
    # CUDA namespace intact. CPU mode may only run in-process on a CPU host.
    if len(worker_tokens) == 1 and (tokens or count == 0):
        if selected:
            torch.cuda.set_device(selected[0])
        for task in tasks:
            _invoke(task)  # Failure deliberately propagates to SLURM.
        return

    context = mp.get_context("spawn")
    task_queue, result_queue = context.Queue(), context.Queue()
    processes = []
    completed, failures = set(), []
    try:
        for index, task in enumerate(tasks):
            task_queue.put((index, task))
        for _ in worker_tokens:
            task_queue.put(None)
        for token in worker_tokens:
            process = context.Process(target=_worker_fn,
                                      args=(task_queue, result_queue, token, logging.INFO))
            process.start()
            processes.append(process)
        while len(completed) < len(tasks):
            try:
                index, error = result_queue.get(timeout=0.2)
                completed.add(index)
                if error:
                    failures.append(f"{tasks[index].get('name', index)}: {error}")
            except queue.Empty:
                crashed = [p for p in processes if p.exitcode not in (None, 0)]
                if crashed:
                    raise RuntimeError("Training worker exited unexpectedly: " +
                                       ", ".join(f"pid={p.pid}, exit={p.exitcode}" for p in crashed))
                if all(not p.is_alive() for p in processes):
                    raise RuntimeError(f"Workers exited with only {len(completed)}/{len(tasks)} results")
        for process in processes:
            process.join(timeout=30)
            if process.is_alive() or process.exitcode != 0:
                raise RuntimeError(f"Worker pid={process.pid} did not exit cleanly ({process.exitcode})")
        if failures:
            raise RuntimeError(f"{len(failures)} training task(s) failed:\n" + "\n".join(failures))
    finally:
        for process in processes:
            if process.is_alive():
                process.terminate()
        for process in processes:
            process.join(timeout=5)
            if process.is_alive():
                process.kill()
                process.join(timeout=5)
        # A dead consumer must not make interpreter shutdown wait on its feeder.
        for pending_queue in (task_queue, result_queue):
            pending_queue.cancel_join_thread()
            pending_queue.close()
