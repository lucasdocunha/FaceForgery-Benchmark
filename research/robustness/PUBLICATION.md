# From the implementation to a publishable research contribution

## Proposed scientific story

Working title: **Spatial–Spectral Cue Reliability under Degradation and Forgery Distribution Shift**.

Research claim to test, not a result to assume: a spectral branch can provide useful complementary evidence in some shifted conditions, but that usefulness must be established against a strong augmented RGB baseline, simple fusion and additional spatial capacity. Explainability then examines repaired and persistent failures with controlled instances and targets.

The current six-page `paper.pdf` remains the preserved XAI author-review manuscript. The ten-page reference, original paper and presentation are unchanged. They are not silently relabeled as results of this new pilot. The following is the writing/measurement contract for the next revision; no empirical results have been manufactured to fill it.

## Three contributions, each conditional on its evidence

| Candidate contribution | Evidence required | What would not establish it |
|---|---|---|
| A reproducible account of degradation × forgery/source shift | Verified populations, per-generator/source evaluation, matched preprocessing, paired seeds, transparent coverage | A larger list of AUC values from incompatible pipelines |
| A useful response to unreliable/complementary cues | Augmented RGB, simple fusion, matched-capacity spatial control, paired-CE and RGB-only JS ablations; independent confirmation | A gate name, two branches, or a best test-selected row |
| A mechanism-oriented explanation of improvements and failures | Fixed-image clean/degraded and before/after-training comparisons, raw maps, numerical controls, matched hard/control groups | Twelve attractive heatmaps, unannotated demographic guesses, or post-hoc examples alone |

If the second contribution is not supported, write an honest benchmark/measurement study or a journal extension rather than fabricating a successful new architecture. If efficiency is the strongest result, pivot to measured compression with no-teacher and standard-distillation student controls. These are alternative coherent papers, not automatic separate publications from each added module.

## Six-page main-text plan

Keep the internal working edition at **six total pages including references, original IEEE/SIBGRAPI two-column layout**. A target journal may require a different submission length/style; this internal constraint is not represented as its policy.

1. **Introduction and positioning (~0.75 page).** Define the gap: clean-set ranking is not reliable under degradation/source/generator shift. Identify the earlier SIBGRAPI benchmark transparently and state the genuinely new questions and evidence.
2. **Protocol and data validity (~1 page).** Define fake-positive labels, preprocessing, train/validation/confirmation separation, source overlap, frame/video units, seed populations and primary metrics. The Celeb-DF mismatch is a resolved measurement issue only after artifacts establish the correction; it is not the main claimed research novelty.
3. **Method (~1 page).** Describe shared augmentation, RGB/spectral encoders, residual gating, paired CE/JS and the meaning of each ablation. Clearly distinguish generic established components from the contribution being claimed.
4. **Results and ablations (~1.5 pages).** Include only numbers from verified run/evaluation certificates. Report matched clean/degraded and external settings, uncertainty, complete seed counts and cost. Separate externally reported unmatched systems from same-protocol reproductions.
5. **Failure analysis, limitations and conclusion (~0.75 page).** Use fixed matched examples, repaired/regressed categories and generator/source conditions. No causal or demographic conclusion without the needed annotations/interventions.
6. **References (~1 page).** Prioritize the directly relevant dataset, modern detection, frequency-debiasing and explanation-evaluation sources. Put the complete experimental/operational inventory in the preserved extended material rather than shrinking type or margins.

These are editorial allocations, not fixed page breaks. Do not shrink body text, margins or references to accommodate an unfocused experiment inventory.

## Figures and tables to generate after execution

**Figure 1: Experimental design.** Source training, source validation, untouched confirmation, canonical IDs and paired preprocessing. Distinguish the legacy-checkpoint route from new-pilot training.

**Table 1: Primary generalization results.** Dataset/domain, forgery group, clean/degraded condition, frame/video unit, exact n and seeds, AUC plus separate seed SD and group-bootstrap interval. Undefined subgroup AUC remains undefined; report the missing real reference and coverage.

**Table 2: Factorized ablation.** RGB basic → RGB robust → paired CE → RGB JS; then early concat, late mean, adaptive CE, adaptive JS and spatial-control JS. Compare only identical budgets and source populations; paired views consume additional forward passes and should be disclosed.

**Figure 2: Accuracy/cost trade-off.** One point per fully specified system, same hardware, precision, input and batch. Keep model-only timing distinct from end-to-end face extraction/decoding/transfer. Never estimate deployment speed from CPU unit tests.

**Figure 3: Matched failures.** Fixed IDs before/after degradation or augmentation, reference stratum plus each model's own outcome, signed raw attribution and declared display normalization. Frequency bins are not facial landmarks. Supply complete galleries and controls separately to avoid selective illustration.

## Required experimental gates

**Gate A — measurement.** Confirm class meaning with known real/fake cases; complete source/target identity audits; verify initialization contract and original builder; reject duplicate/missing predictions. No interpretation of suspected Celeb-DF rankings before this gate.

**Gate B — augmentation.** A matched comparison with paired seeds, same initialization and budget. Separate the exact new `robust_v1` recipe from the heterogeneous historical robust scripts. Changes in initialization, sampler or epoch budget cannot be described as an augmentation-only effect.

**Gate C — complementarity.** On validation, determine how often spectral predictions repair RGB errors and how often they regress correct decisions. The oracle selector is only a non-deployable bound. Lack of complementary signal is a reason not to expand fusion training.

**Gate D — method.** Freeze a small pilot and run the controls above. The implementation's gate and JS term are standard research ingredients; perform a focused novelty review before claiming a new algorithm. Select finalists without looking for the best target-test row.

**Gate E — confirmation.** Exactly the declared seeds and evaluation populations, an untouched confirmation setting, modern relevant same-protocol baselines, complete subgroup coverage, uncertainty and measured costs. DF40 and Celeb-DF source relationships must be disclosed rather than counting names as independent domains.

**Gate F — publication.** Human authors verify all tables, figures, citations, licenses, AI-use disclosure, and the relationship to the accepted SIBGRAPI work. Check the specific venue's extension, dual-submission, anonymity and formatting policies. An implementation PR does not establish acceptance readiness of unexecuted empirical claims.

## Suggested venue decision

A rigorous forensic robustness study should first be assessed for topical fit with a specialist venue such as IEEE TIFS. A genuinely distinct, broadly validated learning method may justify a major vision conference; a broader learning principle may fit ICLR/NeurIPS. An expanded benchmark can instead be a transparent journal extension with clearly itemized differences from SIBGRAPI. These are strategic directions, not rankings, guarantees or acceptance probabilities. Check current author guidance when choosing the route; do not promise that 25–30% new text is a universal sufficient condition.

## Handoff to the next writing pass

The operator should provide the completed run/evaluation certificates and generated comparison/group/seed reports, not screenshots of selected table rows. The next writing pass can replace hypotheses only where these artifacts support observations. Preserve null results, exclusions, pipeline corrections and the provenance of imported historical numbers. Never mix scratch/frozen results from the old paper with fine-tuned/robust results as if they were the same experiment.
