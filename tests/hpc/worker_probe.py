"""Subprocess probe so a worker regression cannot hang the test runner."""
import os
from pathlib import Path
import sys

from src.utils.multiprocess import run_tasks_on_gpus


def succeed(path, multi_gpu=True):
    assert not multi_gpu
    Path(path).write_text(os.environ.get("CUDA_VISIBLE_DEVICES", "unset"))


def fail(multi_gpu=True):
    raise ValueError("intentional task failure")


def die(multi_gpu=True):
    os._exit(9)


if __name__ == "__main__":
    mode, directory = sys.argv[1:]
    function = {"success": succeed, "failure": fail, "death": die}[mode]
    tasks = [{"fn": function, "name": f"probe-{index}",
              "args": (str(Path(directory) / f"task-{index}"),) if mode == "success" else ()}
             for index in range(2)]
    run_tasks_on_gpus(tasks, gpus=[])
