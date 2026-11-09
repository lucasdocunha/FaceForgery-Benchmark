"""Opt-in XAI entry point. Nothing executes until an explicit subcommand is used."""
from __future__ import annotations
import argparse
import json
from pathlib import Path


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="command", required=True)
    prepare = sub.add_parser("prepare", help="Freeze protocols using existing predictions; no image inference")
    prepare.add_argument("--config", type=Path, default=Path("configs/explicability.yaml"))
    prepare.add_argument("--models-root", type=Path, required=True)
    prepare.add_argument("--val-manifest", type=Path, default=Path("data/raw/val.csv"))
    prepare.add_argument("--test-manifest", type=Path, default=Path("data/raw/test.csv"))
    prepare.add_argument("--output", type=Path, default=Path("explicabilidade"))
    prepare.add_argument("--acknowledge-legacy-row-ids", action="store_true")
    prepare.add_argument("--curate-identical-validation", action="store_true")
    prepare.add_argument("--include-degraded", action="store_true")
    describe = sub.add_parser("describe", help="Print frozen task sizes without loading images or checkpoints")
    describe.add_argument("--plan", type=Path, required=True)
    run = sub.add_parser("run", help="Execute one task, optionally sharded, only when explicitly requested")
    run.add_argument("--plan", type=Path, required=True)
    run.add_argument("--task", choices=("task1", "task2", "task3", "task3_controls"), required=True)
    run.add_argument("--models-root", type=Path)
    run.add_argument("--images-root", type=Path)
    run.add_argument("--output", type=Path)
    run.add_argument("--model")
    run.add_argument("--method")
    run.add_argument("--shard-index", type=int, default=0)
    run.add_argument("--shards", type=int, default=1)
    run.add_argument("--device", default="cpu")
    run.add_argument("--resume", action="store_true")
    run.add_argument("--dry-run", action="store_true", help="Describe only; no torch import or image access")
    compare = sub.add_parser("compare", help="Generate Task 2 boards from completed saved attributions")
    compare.add_argument("--plan", type=Path, required=True)
    compare.add_argument("--output", type=Path)
    report = sub.add_parser("report", help="Audit requested-cell coverage; missing is not a zero measurement")
    report.add_argument("--plan", type=Path, required=True)
    report.add_argument("--output", type=Path)
    report.add_argument("--require-complete", action="store_true")
    export = sub.add_parser("export-tables", help="Generate seed summaries and LaTeX from existing prediction metrics")
    export.add_argument("--output", type=Path, default=Path("explicabilidade"))
    args = parser.parse_args(argv)
    if args.command == "prepare":
        from src.explicability.prepare import prepare as prepare_plan
        plan = prepare_plan(args.config, args.models_root, args.val_manifest, args.test_manifest, args.output,
            acknowledge_legacy_row_ids=args.acknowledge_legacy_row_ids,
            curate_identical_validation=args.curate_identical_validation, include_degraded=args.include_degraded)
        print(json.dumps({"plan_id": plan["plan_id"], "execution_status": plan["execution_status"]}, indent=2))
    elif args.command == "describe" or (args.command == "run" and args.dry_run):
        from src.explicability.report import describe
        print(json.dumps(describe(args.plan), indent=2))
    elif args.command == "run":
        if args.models_root is None or args.images_root is None:
            parser.error("run requires --models-root and --images-root unless --dry-run is supplied")
        from src.explicability.runtime import run as execute
        execute(args.plan, args.models_root, args.images_root, args.output or args.plan.parent, args.task,
            model_filter=args.model, method_filter=args.method, shard_index=args.shard_index,
            shards=args.shards, device=args.device, resume=args.resume)
    elif args.command == "compare":
        from src.explicability.render import compare as render
        print(json.dumps(render(args.plan, args.output or args.plan.parent), indent=2))
    elif args.command == "report":
        from src.explicability.report import coverage
        result = coverage(args.plan, args.output or args.plan.parent)
        print(json.dumps({k: {n: v for n, v in row.items() if n != "issues"} for k, row in result["tasks"].items()}, indent=2))
        if args.require_complete and not all(row["all_cells_accounted_for"] for row in result["tasks"].values()):
            raise SystemExit(2)
    else:
        from src.explicability.report import export_tables
        print(f"Exported {export_tables(args.output)} table rows")


if __name__ == "__main__":
    main()
