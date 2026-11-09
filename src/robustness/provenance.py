"""Content identities and atomic artifacts; no network or implicit overwrite."""

from __future__ import annotations
import hashlib
import json
import os
import platform
import subprocess
import tempfile
from importlib.metadata import PackageNotFoundError, version
from pathlib import Path, PurePosixPath
from typing import Any

SCHEMA = "faceforgery-research-v1"


def digest_file(path: str | Path) -> str:
    h = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(1 << 20), b""):
            h.update(block)
    return h.hexdigest()


def canonical(value: Any) -> bytes:
    return json.dumps(
        value,
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=False,
        allow_nan=False,
    ).encode("utf-8")


def digest(value: Any) -> str:
    return hashlib.sha256(canonical(value)).hexdigest()


def relative_path(value: str) -> str:
    if not isinstance(value, str) or not value.strip() or "\\" in value:
        raise ValueError("Expected a nonempty POSIX relative path")
    p = PurePosixPath(value)
    if p.is_absolute() or ".." in p.parts or ":" in value or str(p) in {".", ""}:
        raise ValueError(f"Unsafe relative path: {value!r}")
    return str(p)


def contained(root: str | Path, name: str) -> Path:
    root = Path(root).resolve()
    target = (root / relative_path(name)).resolve()
    if not target.is_relative_to(root):
        raise ValueError("Path or symbolic link escapes its dataset root")
    return target


def atomic_bytes(path: str | Path, data: bytes) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, temp = tempfile.mkstemp(prefix=f".{path.name}.", dir=path.parent)
    try:
        with os.fdopen(fd, "wb") as stream:
            stream.write(data)
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(temp, path)
    finally:
        if os.path.exists(temp):
            os.unlink(temp)


def write_json(path: str | Path, value: Any) -> None:
    atomic_bytes(
        path,
        json.dumps(value, indent=2, ensure_ascii=False, allow_nan=False).encode("utf-8")
        + b"\n",
    )


def write_csv(path: str | Path, frame) -> None:
    atomic_bytes(path, frame.to_csv(index=False).encode("utf-8"))


def source_identity() -> dict:
    root = Path(__file__).resolve().parents[2]
    files = sorted((root / "src" / "robustness").glob("*.py"))
    code = {p.name: digest_file(p) for p in files}
    try:
        commit = subprocess.check_output(
            ["git", "rev-parse", "HEAD"], cwd=root, stderr=subprocess.DEVNULL, text=True
        ).strip()
    except (OSError, subprocess.CalledProcessError):
        commit = "unversioned-copy"
    packages = {}
    for name in (
        "torch",
        "torchvision",
        "timm",
        "numpy",
        "pandas",
        "scikit-learn",
        "Pillow",
    ):
        try:
            packages[name] = version(name)
        except PackageNotFoundError:
            packages[name] = "not-installed"
    return {
        "commit": commit,
        "code_sha256": code,
        "python": platform.python_version(),
        "packages": packages,
    }
