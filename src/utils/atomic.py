"""Publish files without truncating an existing artifact on failure."""
from __future__ import annotations

import os
from pathlib import Path
import shutil
import tempfile


def atomic_copy(source: str | Path, destination: str | Path) -> None:
    """Copy to a unique sibling, flush it, then rename on the destination FS.

    Unlike copying onto the final name, an interrupted/failed NFS write leaves
    the previous artifact intact. This does not make an unavailable NFS healthy:
    errors propagate and callers must retain their local source for recovery.
    """
    destination = Path(destination)
    destination.parent.mkdir(parents=True, exist_ok=True)
    fd, name = tempfile.mkstemp(prefix=f".{destination.name}.", dir=destination.parent)
    temporary = Path(name)
    try:
        with os.fdopen(fd, "wb") as output, Path(source).open("rb") as input_file:
            shutil.copyfileobj(input_file, output, length=1024 * 1024)
            output.flush()
            os.fsync(output.fileno())
        os.replace(temporary, destination)
    finally:
        temporary.unlink(missing_ok=True)


def atomic_torch_save(value: object, destination: str | Path) -> None:
    """Serialize once beside the checkpoint, then atomically replace it.

    The CISIA runner places checkpoint directories on job-local disk. Outside
    that runner the destination controls storage, just like torch.save itself.
    No broad retry can overwrite a valid checkpoint after an I/O exception.
    """
    import torch

    destination = Path(destination)
    destination.parent.mkdir(parents=True, exist_ok=True)
    fd, name = tempfile.mkstemp(prefix=f".{destination.name}.", dir=destination.parent)
    temporary = Path(name)
    try:
        with os.fdopen(fd, "wb") as output:
            torch.save(value, output)
            output.flush()
            os.fsync(output.fileno())
        os.replace(temporary, destination)
    finally:
        temporary.unlink(missing_ok=True)
