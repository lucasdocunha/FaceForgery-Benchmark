"""Versioned initialization semantics, separate from augmentation and run names.

New runs treat finetune_robust exactly like finetune for initialization.
Unversioned historical configs retain the old constructors during checkpoint
reconstruction. This preserves architecture compatibility, not evidence that
an old run actually used the initialization implied by its directory name.
"""

VALID_REGIMES = {"scratch", "finetune", "scratch_robust", "finetune_robust"}


def uses_pretraining(config, family: str) -> bool:
    regime = str(config.regime)
    if regime not in VALID_REGIMES:
        raise ValueError(f"Unknown training regime: {regime}")
    contract = getattr(config, "initialization_contract", "regime-v2")
    if contract == "legacy-v1":
        return (
            regime != "scratch" if family in {"clip", "vit"} else regime == "finetune"
        )
    if contract != "regime-v2":
        raise ValueError(f"Unknown initialization contract: {contract}")
    return regime.removesuffix("_robust") == "finetune"
