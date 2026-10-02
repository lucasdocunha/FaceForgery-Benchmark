"""Standalone source-training entry point; common orchestration may call fit directly."""

import argparse
import json
import os
from pathlib import Path

import yaml

from .training import fit


def _resolve(value):
    if isinstance(value, dict):
        return {key: _resolve(item) for key, item in value.items()}
    if isinstance(value, list):
        return [_resolve(item) for item in value]
    if isinstance(value, str):
        resolved = os.path.expandvars(value)
        if "${" in resolved:
            raise ValueError(f"Unset environment variable in config: {resolved}")
        return resolved
    return value


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", required=True)
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--resume", action="store_true")
    args = parser.parse_args()
    config = _resolve(yaml.safe_load(Path(args.config).read_text()))
    print(json.dumps(fit(config, device=args.device, resume=args.resume), indent=2))


if __name__ == "__main__":
    main()
