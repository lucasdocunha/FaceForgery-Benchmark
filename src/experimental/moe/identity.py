"""Controlled expert policies versus exact frozen checkpoint realizations."""

from __future__ import annotations

import json
from pathlib import Path

from src.robustness.provenance import digest, digest_file


def expert_conditions(specs, experts, *, policy="fixed", seed=42):
    if policy not in {"fixed", "matched"}:
        raise ValueError("Expert seed policy must be fixed or matched")
    conditions, observed_seeds = [], []
    for spec, expert in zip(specs, experts):
        contract = expert.contract
        if policy == "fixed":
            conditions.append({"seed_policy": "fixed", "expert_contract": contract,
                               "expert_source_identity": expert.identity})
            observed_seeds.append(None)
            continue
        if contract["kind"] == "feature_cache":
            metadata = expert.source_metadata
            origin = metadata.get("checkpoint", {}).get("path")
            if not origin:
                raise ValueError("Matched expert seeds require the verified source checkpoint and run config")
            checkpoint = Path(origin)
            config_path = checkpoint.parent.parent / "results" / "run_config.json"
            if (not checkpoint.is_file() or not config_path.is_file()
                    or digest_file(checkpoint) != contract["checkpoint_sha256"]
                    or digest_file(config_path) != contract["run_config_sha256"]):
                raise ValueError("Matched expert source checkpoint or run config differs from cache provenance")
            from src.pipelines.checkpoints import config_from_run, run_from_checkpoint
            source_run = run_from_checkpoint(checkpoint)
            raw = json.loads(config_path.read_text())
            if "seed" in raw and int(raw["seed"]) != source_run.seed:
                raise ValueError("Source expert run config seed differs from checkpoint layout")
            normalized = config_from_run(source_run).to_dict()
            normalized = {key: value for key, value in normalized.items()
                          if key not in {"seed", "seeds", "num_workers", "multi_gpu"}}
            observed_seed = source_run.seed
            source_condition = {"family": "legacy-hf", "model_training_policy": normalized,
                                "extraction": contract["extraction"], "packages": contract["packages"],
                                "code_sha256": contract["code_sha256"], "feature_dim": contract["feature_dim"],
                                "score": contract["score"]}
        elif contract["kind"] == "predictions" and contract["role"] == "reconstruction":
            try:
                from src.experimental.reconstruction import describe_run
            except ImportError as error:
                raise ValueError("Matched reconstruction expert requires the verified reconstruction metadata reader") from error
            description = describe_run(Path(spec["path"]).parent)
            if (digest_file(description["checkpoint_path"]) != contract["checkpoint_sha256"]
                    or description["input_contract"] != contract["input_contract"]):
                raise ValueError("Reconstruction source bundle differs from the certified expert scores")
            research = description["research_run"]
            if digest(research["condition"]) != research["condition_sha256"]:
                raise ValueError("Reconstruction controlled condition hash mismatch")
            observed_seed = int(research["seed"])
            source_condition = {"family": "reconstruction", "condition": research["condition"],
                                "feature_columns": contract["feature_columns"], "score": contract["score"]}
        else:
            raise ValueError("Matched expert seeds require a verified comparable training policy; use fixed for other experts")
        if observed_seed != int(seed):
            raise ValueError("Matched expert seed differs from fusion training seed")
        population_columns = [name for name in ("sample_id", "group_id", "img_name", "label", "image_sha256", "source_id", "video_id")
                              if name in expert.frame]
        population = expert.frame[population_columns].sort_values("sample_id").to_dict("records")
        conditions.append({"name": contract["name"], "role": contract["role"], "kind": contract["kind"],
                           "seed_policy": "matched", "source_condition": source_condition,
                           "source_population_sha256": digest(population)})
        observed_seeds.append(observed_seed)
    return conditions, observed_seeds
