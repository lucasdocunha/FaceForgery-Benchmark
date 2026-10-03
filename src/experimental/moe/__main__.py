"""Standalone frozen MoE fit, using the shared environment-expanding config loader."""

import argparse
import json
import os
import re
from pathlib import Path

import yaml
from .training import fit


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("config")
    args = parser.parse_args()
    raw = os.path.expandvars(Path(args.config).read_text())
    if re.search(r"\$\{[^}]+\}", raw):
        raise ValueError("Set every environment variable referenced by the MoE configuration")
    print(json.dumps({"run_dir": str(fit(yaml.safe_load(raw)))}))


if __name__ == "__main__":
    main()
