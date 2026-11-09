"""Coverage and table generation. Absence of an artifact is never a zero result."""
from __future__ import annotations
import json
from pathlib import Path
from .contracts import contained, sha256
from .prepare import read_plan


def describe(plan_path):
    plan = read_plan(plan_path)
    methods = plan["config"]["methods"]
    replicates = int(plan["config"]["settings"].get("replicates", 1))
    tasks = {name: {"images": len(cohort["samples"]), "models": len(cohort["roster"]),
                   "methods": len(methods), "replicates": replicates,
                   "requested_cells": len(cohort["samples"]) * len(cohort["roster"]) * len(methods) * replicates}
             for name, cohort in plan["cohorts"].items()}
    return {"plan_id": plan["plan_id"], "tasks": tasks,
            "note": "Counts are an execution plan, not measured runtime or completed experiments"}


def coverage(plan_path, output):
    plan, root = read_plan(plan_path), Path(output)
    result = {"plan_id": plan["plan_id"], "tasks": {}, "scientific_validation": "not established by coverage"}
    for task, cohort in plan["cohorts"].items():
        counts = {"complete": 0, "unsupported": 0, "missing": 0, "corrupt": 0,
                  "constant_maps": 0, "head_controls_complete": 0, "curve_controls_complete": 0}
        issues = []
        for model in cohort["roster"]:
            for method in plan["config"]["methods"]:
                for row in cohort["samples"]:
                    for rep in range(int(plan["config"]["settings"].get("replicates", 1))):
                        name = f"{row['id']}_r{rep}"
                        directory = root / task / model / method
                        path = directory / f"{name}.json"
                        cell = f"{model}/{method}/{name}"
                        if not path.exists():
                            counts["missing"] += 1; issues.append({"cell": cell, "status": "missing"}); continue
                        try:
                            record = json.loads(path.read_text())
                            identity = record["identity"]
                            expected = {"plan_id": plan["plan_id"], "task": task, "model": model,
                                        "method": method, "id": row["id"], "replicate": rep}
                            if any(identity[k] != v for k, v in expected.items()):
                                raise ValueError("Cell identity mismatch")
                            if record["status"] == "unsupported":
                                counts["unsupported"] += 1
                                issues.append({"cell": cell, "status": "unsupported", "reason": record["reason"]}); continue
                            if record["status"] != "complete":
                                raise ValueError("Unexpected final status")
                            for filename, checksum in record["artifacts"].items():
                                if Path(filename).name != filename or sha256(directory / filename) != checksum:
                                    raise ValueError("Artifact checksum mismatch")
                            if sha256(contained(root, record["asset"])) != identity["asset_sha256"]:
                                raise ValueError("Input artifact checksum mismatch")
                            counts["complete"] += 1
                            metadata = record["metadata"]
                            counts["constant_maps"] += int(metadata.get("constant", False))
                            counts["head_controls_complete"] += int(metadata.get("randomization_status") == "completed")
                            counts["curve_controls_complete"] += int("perturbation_curves" in metadata)
                        except (ValueError, KeyError, OSError, TypeError) as error:
                            counts["corrupt"] += 1; issues.append({"cell": cell, "status": "corrupt", "error": str(error)})
        counts["requested"] = sum(counts[k] for k in ("complete", "unsupported", "missing", "corrupt"))
        counts["all_cells_accounted_for"] = counts["missing"] == 0 and counts["corrupt"] == 0
        result["tasks"][task] = {**counts, "issues": issues}
    root.mkdir(parents=True, exist_ok=True)
    (root / "coverage.json").write_text(json.dumps(result, indent=2) + "\n")
    return result


def export_tables(output):
    """Numerical tables from prepare outputs only; image findings are never filled in."""
    import pandas as pd
    root = Path(output)
    frame = pd.read_csv(root / "prediction_metrics.csv")
    if frame.empty:
        raise ValueError("No prediction metrics have been generated")
    parts = frame["model"].str.rsplit(".seed_", n=1, expand=True)
    frame["configuration"] = parts[0]
    rows = []
    for (split, configuration), group in frame.groupby(["split", "configuration"]):
        rows.append({"split": split, "configuration": configuration, "n_seeds": len(group),
            "auc_mean": group.auc.mean(), "auc_sample_std": group.auc.std(ddof=1),
            "balanced_accuracy_mean": group.balanced_accuracy.mean(),
            "balanced_accuracy_sample_std": group.balanced_accuracy.std(ddof=1)})
    summary = pd.DataFrame(rows)
    summary.to_csv(root / "prediction_summary_by_seed.csv", index=False)
    text = [r"% Generated from exported predictions; not new image inference.",
            r"\begin{tabular}{llrrr}", r"\toprule",
            r"Split & Configuration & Seeds & ROC-AUC & Balanced accuracy \\", r"\midrule"]
    for row in rows:
        def value(mean, std):
            return f"{mean:.3f}" if pd.isna(std) else f"${mean:.3f} \\pm {std:.3f}$"
        split = row["split"].replace("_", r"\_")
        config = row["configuration"].replace("_", r"\_")
        text.append(f"{split} & {config} & {row['n_seeds']} & "
                    f"{value(row['auc_mean'], row['auc_sample_std'])} & "
                    f"{value(row['balanced_accuracy_mean'], row['balanced_accuracy_sample_std'])} \\\\")
    text.extend([r"\bottomrule", r"\end{tabular}"])
    (root / "prediction_tables.tex").write_text("\n".join(text) + "\n")
    return len(rows)
