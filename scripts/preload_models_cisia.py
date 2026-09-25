#!/usr/bin/env python3
"""Compatibility entrypoint; submit scripts/download_pretrained_cisia.sh instead."""
from __future__ import annotations


def main() -> None:
    from pathlib import Path
    import sys
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
    from scripts.setup_pretrained_cisia import main as prepare
    prepare()


if __name__ == "__main__":
    main()
