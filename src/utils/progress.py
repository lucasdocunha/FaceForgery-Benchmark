"""Line-oriented progress for batch logs, without terminal control sequences."""
from __future__ import annotations

import time
from collections.abc import Iterable, Iterator
from typing import TypeVar

T = TypeVar("T")


def progress(items: Iterable[T], description: str, interval: float = 60.0) -> Iterator[T]:
    """Report completed batches, throughput and ETA, at most once per interval."""
    total = len(items) if hasattr(items, "__len__") else None
    start = last = time.monotonic()
    done = 0
    print(f"{description}: start | total={total if total is not None else '?'} batches", flush=True)
    for item in items:
        yield item
        done += 1
        now = time.monotonic()
        if now - last >= interval or (total is not None and done == total):
            elapsed = max(now - start, 1e-9)
            rate = done / elapsed
            eta = f"{(total - done) / rate:.0f}s" if total is not None else "?"
            print(f"{description}: {done}/{total if total is not None else '?'} batches "
                  f"| {rate:.2f} batches/s | ETA {eta}", flush=True)
            last = now
    print(f"{description}: finished {done} batches in {time.monotonic() - start:.1f}s", flush=True)
