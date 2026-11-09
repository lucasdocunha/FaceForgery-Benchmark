"""Download immutable public evidence; never execute checkpoint pickle or train models.

This deliberately does not claim that released fine-tuned runs reproduce the
scratch-model table in the earlier paper. See docs/explicability/PROVENANCE.md.
"""
from __future__ import annotations

import argparse
import concurrent.futures
import hashlib
import json
import time
import urllib.parse
import urllib.request
from pathlib import Path

HF_REPO = "lucasoc/MFFI-Models"
HF_REVISION = "3bef179cdc08850e1d720182e55a040d00c9d582"
FAMILIES = {"resnet", "xception", "mobilenet", "vit", "clip", "dino"}


def download(url: str, path: Path, max_bytes: int = 100_000_000) -> dict:
    """Bounded, retried download with content hash and atomic replacement."""
    for attempt in range(3):
        try:
            request = urllib.request.Request(url, headers={"User-Agent": "FaceForgery-Benchmark-research/1.0"})
            with urllib.request.urlopen(request, timeout=60) as response:
                data = response.read(max_bytes + 1)
            if len(data) > max_bytes:
                raise ValueError(f"Download exceeds limit: {max_bytes} bytes")
            path.parent.mkdir(parents=True, exist_ok=True)
            temp = path.with_suffix(path.suffix + ".partial")
            temp.write_bytes(data)
            temp.replace(path)
            return {"url": url, "path": str(path), "ok": True, "bytes": len(data),
                    "sha256": hashlib.sha256(data).hexdigest()}
        except Exception as error:
            if attempt == 2:
                return {"url": url, "path": str(path), "ok": False,
                        "error": f"{type(error).__name__}: {error}"}
            time.sleep(attempt + 1)
    raise AssertionError("unreachable")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--rgb-predictions", action="store_true",
                        help="Include val/test/test_d predictions for six RGB families, seeds 42/123/2024")
    parser.add_argument("--workers", type=int, default=8)
    args = parser.parse_args()
    if not 1 <= args.workers <= 16:
        parser.error("workers must be in [1,16]")
    args.output.mkdir(parents=True, exist_ok=True)
    api = f"https://huggingface.co/api/models/{HF_REPO}/revision/{HF_REVISION}"
    status = [download(api, args.output / "hub.json")]
    if not status[0]["ok"]:
        raise RuntimeError(status[0])
    hub = json.loads((args.output / "hub.json").read_text())
    if hub["sha"] != HF_REVISION:
        raise ValueError("Hub returned a different revision")
    names = [item["rfilename"] for item in hub["siblings"]]
    chosen = []
    for name in names:
        p = name.split("/")
        metadata = name in {"README.md", "all_metrics_by_split.csv"} or name.endswith("run_config.json")
        metadata |= ("/results/metrics_" in name and name.endswith(".csv"))
        rgb = (args.rgb_predictions and len(p) == 6 and p[0] in FAMILIES
               and p[1:3] == ["none", "finetune"] and p[3] in {"seed_42", "seed_123", "seed_2024"}
               and p[-1] in {"predictions_val.csv", "predictions_test.csv", "predictions_test_d.csv"})
        if metadata or rgb:
            chosen.append(name)
    base = f"https://huggingface.co/{HF_REPO}/resolve/{HF_REVISION}/"
    with concurrent.futures.ThreadPoolExecutor(max_workers=args.workers) as pool:
        futures = {pool.submit(download, base + urllib.parse.quote(name, safe="/"),
                               args.output / "released" / name): name for name in sorted(chosen)}
        for index, future in enumerate(concurrent.futures.as_completed(futures), 1):
            result = future.result()
            result["repo_path"] = futures[future]
            status.append(result)
            if not result["ok"] or index % 100 == 0:
                print(index, len(chosen), result["repo_path"], result["ok"], flush=True)
    report = {"repository": HF_REPO, "revision": HF_REVISION,
              "selected_files": len(chosen), "files": sorted(status, key=lambda x: x["path"])}
    (args.output / "downloads.json").write_text(json.dumps(report, indent=2) + "\n")
    failures = [item for item in status if not item["ok"]]
    print(f"Downloaded {len(status) - len(failures)}/{len(status)} files", flush=True)
    if failures:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
