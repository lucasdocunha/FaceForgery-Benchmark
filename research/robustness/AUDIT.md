# Evaluation and provenance audit

Audit baseline: upstream `ICLR@f00475116eb350aca8d6dbd08df999ab6f21991a`. The source was read, not modified. Original files remain retrievable at that commit; imported reports are not rewritten into purported corrected experiments.

## 1. Celeb-DF label convention — confirmed code mismatch

The old preparation script documented official list values `1=real, 0=fake` and copied them directly to `target`. The evaluator consumed class-1 probabilities while the research convention is fake-positive. If those exact scripts/checkpoint conventions generated a table, its label semantics are inconsistent. This does not prove which locally edited scripts ran on the GPU machines.

For fixed binary scores, reversing only labels gives AUC' = 1 − AUC. This identity is a diagnostic, **not permission to invert below-chance results or automatically publish corrected tables**. Verify known real/fake examples, original manifests, output orientation, and producing code. If raw scores and identities exist, use `import-predictions` with explicit semantics and original row-order acknowledgement; otherwise re-infer the saved model rather than guessing per-sample scores from an aggregate AUC.

The replacement preparation script verifies labels against official folder names, converts once to fake-positive labels, namespaces full relative video paths, records PNG crop hashes and extraction coverage, and refuses to publish a manifest if any selected video fails. A bounded subset is labeled `smoke_test`, not a complete test set. The old silent center crop is now an explicit `--no-face center` choice. PNG and the square crop implementation change preprocessing from the former JPEG95 extractor; compare these pipelines deliberately, not as a label-only ablation.

The replacement evaluator writes keyed frame predictions, video predictions, metrics and provenance. Frame thresholds are frozen on source validation; source-video thresholds are used when available. A frame threshold transferred to video aggregation is explicitly labeled. There is no target threshold optimization or filename-only stale-cache reuse.

## 2. Robust regime naming — confirmed constructor inconsistency

The imported config and CLI accept `finetune_robust` and `scratch_robust`, but ResNet, MobileNet, DINO and Xception builders tested exact equality with `finetune`. CLIP and ViT tested exact equality with `scratch` before choosing the pretrained path. This creates **family-dependent initialization semantics** for suffixed regimes. It does not establish that every reported robust run was affected: some separate scripts build with unsuffixed `finetune` and rename outputs or use different local versions.

New configs record `initialization_contract=regime-v2`. All six builders now use the same suffix-independent policy. `config_from_run()` assigns `legacy-v1` when an old saved config lacks the field, preserving the original constructor choice, especially where pretrained and scratch Xception/transformer architectures differ. Loading a full state dict remains strict. Never retroactively add `regime-v2` to a historical config merely to make its label look correct. A genuine architecture mismatch needs its exact original builder/version, not a guessed fallback.

Before attributing a measured robustness gain to augmentation, require: same initialization source, paired seeds, architecture, freeze policy, optimizer, budget, source population, and evaluation preprocessing. Directory names alone prove none of these.

## 3. Missing images and identity — confirmed unsafe legacy behavior

The original dataset returned the next image on a read error, changing image, label, and index under an apparently successful batch. The shared loader now raises an error identifying the original index/path. The new canonical dataset is also fail-fast. This intentional behavior change may cause legacy jobs with previously hidden bad files to stop; repair or explicitly exclude those files in a new versioned manifest, never silently substitute them.

Canonical IDs are strings. A numeric legacy ID is accepted only with exact population coverage and an explicit original-row-order acknowledgement. Different dataset names do not prove different people or image sources: perform content/identity/source overlap audits across train, validation and external tests. Hashes catch identical bytes, not every near-duplicate or common identity. Per-video bootstrap groups do not remove correlation across videos of the same person.

## 4. Historical summaries — reports, not certificates

`results/mostrar_rayson/tabela4-seedrobusta-rgb.md` contains inconsistent completion counts, including rows labeled complete whose metrics are dashes. Some comparisons mix one, four, five or six completed seeds. `inventory` enumerates concrete missing/invalid artifacts; `aggregate-seeds` requires exactly the declared seed population. No missing seed is imputed and standard deviations are not reconstructed from a single run.

Validation exports previously inspected in the project included an identical duplicate numeric ID and a missing ID. The new importer rejects these by default. The retained XAI preparation has an explicitly named derived-view curation option; that does not repair the original export. Prefer regenerating a complete prediction export or a separately certified common observed population with fully disclosed exclusions.

## 5. Selection and comparability

Best Test-D numbers seen during development are exploratory, not untouched confirmation. Keep selection on source validation and predeclare finalists before confirmation. The pilot trainer never loads test data. Its default checkpoint criterion is **clean source-validation AUC**; it does not claim to optimize an unimplemented corruption-validation objective.

A larger ensemble, a two-branch fusion, or Jensen–Shannon consistency is not inherently a new scientific contribution. Compare spatial-only extra-capacity controls, ordinary late fusion, paired-CE without consistency, and RGB-only consistency. Distillation is an optional standard-KL baseline, not a claimed innovation or an automatically executed new study.

## What remains an external research responsibility

The team must verify the actual historical run artifacts and local source versions, arrange licensed dataset access, review identity/source overlap and annotations, select a held-out confirmation protocol, and execute the declared experiments. This is not a request for new access to complete the code PR; it is the distinction between delivered software and empirical evidence.
