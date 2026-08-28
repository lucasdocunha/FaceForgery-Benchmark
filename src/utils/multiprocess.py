from __future__ import annotations

import logging
import multiprocessing as mp
import os
import time
import traceback
import torch

logger = logging.getLogger(__name__)


def _worker_fn(
    task_queue: mp.JoinableQueue,
    gpu_id: int | None,
    log_level: int,
) -> None:
    """
    Worker function executed in a separate spawned process.
    Configures the environment for the specified GPU and runs tasks from the queue.
    """
    # 1. Configure environment variable to restrict process to specific GPU
    if gpu_id is not None:
        os.environ["CUDA_VISIBLE_DEVICES"] = str(gpu_id)
        # Ensure PyTorch initializes CUDA correctly on this specific GPU
        if torch.cuda.is_available():
            # Force eager initialization on the assigned GPU
            torch.cuda.set_device(0)
    else:
        # If no GPU is assigned, force CPU mode
        os.environ["CUDA_VISIBLE_DEVICES"] = ""

    # Ensure file_system sharing strategy to avoid open file descriptor exhaustion
    try:
        torch.multiprocessing.set_sharing_strategy("file_system")
    except Exception:
        pass

    # 2. Configure logging inside the child process
    effective_level = log_level if log_level <= logging.INFO else logging.INFO
    logging.basicConfig(
        level=effective_level,
        format=f"%(asctime)s [Worker GPU {gpu_id if gpu_id is not None else 'CPU'}] [%(levelname)s] %(name)s: %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
        force=True,  # Resets the root logger configuration
    )
    
    # Get logger for child process
    child_logger = logging.getLogger(__name__)

    # 3. Pull tasks from the queue and run them
    while True:
        task = task_queue.get()
        if task is None:
            task_queue.task_done()
            break

        fn, name, args, kwargs = task
        gpu_prefix = f"GPU {gpu_id}" if gpu_id is not None else "CPU"
        child_logger.info(f"===> [{gpu_prefix}] Starting task: {name}")

        start_time = time.time()
        try:
            # Enforce single GPU within the task by overriding multi_gpu
            kwargs = dict(kwargs)
            kwargs["multi_gpu"] = False

            # Run the actual pipeline function
            fn(*args, **kwargs)

            elapsed = time.time() - start_time
            child_logger.info(
                f"===> [{gpu_prefix}] Completed task: {name} in {elapsed:.2f}s"
            )
        except Exception as e:
            elapsed = time.time() - start_time
            child_logger.error(
                f"===> [{gpu_prefix}] Failed task: {name} after {elapsed:.2f}s with error: {e}"
            )
            traceback.print_exc()
        finally:
            task_queue.task_done()


def run_tasks_on_gpus(
    tasks: list[dict],
    gpus: list[int] | None = None,
    workers_per_gpu: int = 1,
) -> None:
    """
    Runs a list of training tasks concurrently using multiprocessing,
    distributing them across the specified GPUs.

    Args:
        tasks: A list of dicts. Each dict must contain:
               - 'fn': The callable function to run (e.g. run_resnet).
               - 'name': A unique string name for the task.
               - 'args': Positional arguments for 'fn' (optional, defaults to ()).
               - 'kwargs': Keyword arguments for 'fn' (optional, defaults to {}).
        gpus: A list of GPU physical IDs to use (e.g. [0, 1]).
              If None, all available GPUs are auto-detected.
              If no GPUs are found, falls back to CPU multiprocessing.
        workers_per_gpu: Number of worker processes to spawn per physical GPU,
              so multiple tasks can share the same GPU concurrently (useful when
              a single GPU has enough VRAM to hold several models at once).
    """
    if not tasks:
        logger.warning("No tasks provided to run_tasks_on_gpus.")
        return

    # 1. Determine available GPUs
    if gpus is None:
        if torch.cuda.is_available():
            num_gpus = torch.cuda.device_count()
            gpus = list(range(num_gpus))
            logger.info(f"Auto-detected {num_gpus} GPUs: {gpus}")
        else:
            gpus = []
            logger.warning("No CUDA GPUs detected. Running on CPU.")

    # 2. Setup multiprocessing context (always use 'spawn' for CUDA safety)
    try:
        torch.multiprocessing.set_sharing_strategy("file_system")
    except Exception:
        pass
    ctx = mp.get_context("spawn")

    # 3. Create a task queue and load all tasks
    task_queue = ctx.JoinableQueue()
    for task_info in tasks:
        fn = task_info["fn"]
        name = task_info.get("name", fn.__name__)
        args = task_info.get("args", ())
        kwargs = task_info.get("kwargs", {})
        task_queue.put((fn, name, args, kwargs))

    # Determine workers pool based on GPUs or CPU count
    if len(gpus) > 0:
        worker_gpus = [gpu_id for gpu_id in gpus for _ in range(workers_per_gpu)]
        num_workers = len(worker_gpus)
        logger.info(
            f"Spawning {num_workers} worker processes ({workers_per_gpu} per GPU), "
            f"mapping to physical GPUs: {gpus}"
        )
    else:
        # Fallback to CPU with a pool of processes
        num_workers = max(1, os.cpu_count() // 2)
        worker_gpus = [None] * num_workers
        logger.info(f"Spawning {num_workers} CPU worker processes.")

    # Add poison pills/sentinels to stop the workers when queue is empty
    for _ in range(num_workers):
        task_queue.put(None)

    # 4. Spawn the worker processes
    processes = []
    log_level = logging.getLogger().getEffectiveLevel()

    for i in range(num_workers):
        gpu_id = worker_gpus[i]
        p = ctx.Process(
            target=_worker_fn,
            args=(task_queue, gpu_id, log_level),
            name=f"Worker-{gpu_id if gpu_id is not None else 'CPU'}-{i}",
        )
        p.start()
        processes.append(p)

    logger.info(
        f"All {num_workers} worker processes spawned. Processing tasks..."
    )

    # 5. Wait for all tasks to be completed in the queue
    task_queue.join()
    logger.info("All tasks in the queue have finished execution.")

    # 6. Wait for all processes to finish cleanly
    for p in processes:
        p.join()
    logger.info("All worker processes have exited successfully.")
