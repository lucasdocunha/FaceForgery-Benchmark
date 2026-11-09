"""Exercise a same-mask metric projector followed by a dynamic sampled GNN."""

import argparse
import json
from pathlib import Path

import numpy as np

from src.experimental.feature_training import fit_metric_run
from src.experimental.features import open_cache
from src.experimental.graph_training import fit_graph_run, predict_graph_run
from src.robustness.provenance import source_identity, write_json


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--cache-index", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args(argv)
    if args.output.exists():
        parser.error("Use a fresh output directory")
    caches = json.loads(args.cache_index.read_text())
    if set(caches) != {"train", "val"}:
        parser.error("Exactly source train and validation caches required")
    train, val = open_cache(caches["train"]), open_cache(caches["val"])
    if max(len(train.frame), len(val.frame)) > 10000:
        parser.error("This smoke requires at most 10000 rows per split")
    args.output.mkdir(parents=True)
    common = {"seed": 42, "caches": caches,
              "training": {"device": "cpu", "cpu_threads": 2, "epochs": 2, "batch_size": 32,
                           "lr": .001, "baseline_epochs": 10}}
    metric = fit_metric_run({**common, "family": "metric", "run_dir": str(args.output / "metric"),
                             "model": {"kind": "supcon", "embedding_dim": 128, "hidden_dim": 64,
                                       "label_fraction": .05, "hardness": 0, "tsne": False}})
    graph = fit_graph_run({**common, "family": "graph", "run_dir": str(args.output / "graph"),
                           "projection_run": str(metric), "label_mask": str(metric / "label_mask.json"),
                           "model": {"kind": "sage", "backend": "pyg", "protocol": "inductive",
                                     "label_fraction": .05, "standardize": False, "hidden_dim": 32,
                                     "projection_dim": 128, "fanouts": [10, 5],
                                     "graph": {"backend": "exact", "k": 10, "policy": "directed",
                                               "dynamic": True, "rebuild_interval": 1}}})
    a, b = np.load(metric / "label_mask.npy"), np.load(graph / "label_mask.npy")
    if not np.array_equal(a, b):
        raise AssertionError("Metric and graph masks diverged")
    probabilities = predict_graph_run(graph, val.features)
    repeated = predict_graph_run(graph, val.features)
    np.testing.assert_array_equal(probabilities, repeated)
    rebuilds = json.loads((graph / "graph_rebuilds.json").read_text())
    if [row["epoch"] for row in rebuilds] != [0, 1]:
        raise AssertionError("Expected a dynamic rebuild in each smoke epoch")
    result = {"scope": "min-data dependency smoke, not method selection", "software": source_identity(),
              "train_cache": train.identity, "val_cache": val.identity, "seed": 42,
              "train_n": len(train.frame), "val_n": len(val.frame), "selected_labels": int(a.sum()),
              "metric_mask_equals_graph_mask": True, "reload_exact": True,
              "graph_rebuilds": rebuilds,
              "metric": json.loads((metric / "metrics.json").read_text())["frame"],
              "graph": json.loads((graph / "metrics.json").read_text())["frame"],
              "telemetry": {name: json.loads((root / "pipeline_telemetry.json").read_text())
                            for name, root in (("metric", metric), ("graph", graph))}}
    write_json(args.output / "smoke.json", result)
    print(json.dumps(result), flush=True)


if __name__ == "__main__":
    main()
