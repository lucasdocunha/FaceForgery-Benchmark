# Experimental extensions progress

## 2026-10-02 Gate 1 preparation

Base: remote ICLR `1b2b0e27693b695114c888c8e12e874714d950a6`. Branch: `research/experimental-extensions`. Planning audits integrated through `f5a493c`. No runtime source, dependency pins, paper or prior results changed.

Director messages 001 and 002 read. Hard deadline 22:30 -03; code freeze 21:30. Message 002 supersedes parallel implementation: all four subagents interrupted, no new agents authorized, implementation paused pending Gate 1 reply. The account quota stop at 97% takes precedence over the timeline; usage is reported by the director, not measured locally. Earlier workstream budgets in planning notes are superseded by the solo plan.

- Baseline on unmodified source: 265 tests and 20 subtests passed in 124.01 seconds. CPU suite, Python 3.12.13, ROCm torch 2.10.0, torchvision 0.25.0, pinned research requirements.
- GPU: gfx1200, approximately 16 GiB; BF16 matmul/backward and torchvision NMS passed. Batch-4 synthetic training: ResNet18 153.56 images/s; MobileNetV3-L 73.25 images/s after warm-up. Training probe peak RSS 2.03 GiB. These are synthetic throughput probes, not model quality results.
- Data: 3,000 images, 1,000 per split, no missing/unreadable images or exact cross-split duplicates. Real/fake counts train 202/798, val 408/592, test 423/577. Predecoded 224 RGB uint8 cache is outside git.
- Eight seed-42 RGB/SRM checkpoints downloaded using physical local directories at HF revision f2dee3c52c665053f06182307841f8cd68be3916. Seven completed repository-loader min-val/min-test checks; CLIP RGB remains running. Optional NF4/PyG/LPIPS runtime probes are queued under the same GPU lock.
- Every checkpoint min-test access is compatibility verification only, never candidate selection; each completed check has a JSON in preflight/checkpoints. Predictions remain outside git. Test is untouched by method training, which has not begun.
- CPU MediaPipe 1.0.1 detected one valid landmark set for all 202 train reals in 2.81 seconds, peak RSS 258.35 MiB. Geometry was checked; blend-mask visual QA remains pending. SmolVLM256M and landmark assets are staged outside git.

## Hypotheses and decisions

1. Reuse Stack B calibration, strict artifacts, metrics and completed-epoch resume; legacy fit automatically evaluates test.
2. Exact SRM means six channels with residuals after ImageNet normalization. Preserve it for pretrained reuse.
3. SBI from generic pretraining is the clean unseen-training-forgery proxy; HF-initialized SBI is adaptation after fake exposure. MFFI-val selection still exposes the method-selection process to MFFI fakes.
4. HF-supervised feature GNNs measure downstream low-label adaptation, not end-to-end label efficiency. Dense full-scale cosine graphs are infeasible.
5. Fit learned fusion on deterministic val_fit, select/calibrate on val_select; inherited HF checkpoint selection still limits independence.
6. Primary computed baseline corrections and literature are in planning/. None is a new benchmark measurement.

## Open tasks and ownership

Lead owns all subsequent work. A-D reconstruction, E SBI/evaluation, F VLM, G metric learning, H graph learning and I MoE are planned, not implemented. Next, after director review: audited evaluator and feature cache, then a coherent reconstruction/SBI path with actual min-data smoke; G/I only if quota permits. F/H and remaining variants require an explicit scope decision if quota precludes implementation. Never mark an unimplemented path server-ready. Final-server TODOs include real four-target execution, canonical five seeds, independent fusion fitting and label-efficiency controls.

Known inherited defects are documented in planning/evaluation_audit.md. No rejected method or claimed improvement yet. No expensive pilot or new training has run. No push or PR until Gate 4 reply and authenticated channel.

## 2026-10-02 16:49 -03: Gate 1 approved, parallel implementation restored

Messages 003/004 supersede the solo/no-agent state above. Full A-I scope remains. Owners and worktrees:

| Owner | Worktree / branch | Scope | Next action |
|---|---|---|---|
| Lead | FaceForgery-Benchmark / research/experimental-extensions | Runtime, feature cache, SBI, integration, records | Land shared runner/cache contracts; retry stopped probes |
| Evaluation, GPT 6.1 max | wt-evaluation / research/plan-evaluation | Stack B metrics/SRM/suite/CLI/HPC/tests | First tested evaluator slice |
| Reconstruction, Astra max | wt-reconstruction / research/plan-reconstruction | A-D modules/train/load/tests/configs | Build models/losses then integrate runtime |
| Representations, Astra max | wt-representations / research/plan-representations | G/H, mandatory baselines and scalable graphs | Structural tests against declared feature-store API |
| VLM, GPT 6.1 max | wt-vlm / research/plan-vlm | F, then I if available | Scoring/masking/PEFT tests, real Smol smoke via lead |

Common runtime: src.experimental.runtime.fit_model with loss_step(model,batch,step_state) returning mean loss, scalar metrics and weight count; validate(model) returns scalar metrics. Artifacts run.json, best.pt/last.pt with state_dict, optimizer, RNG, epoch/global_step; optional PEFT state callbacks. FeatureStore exposes float16 features, float32 logits, canonical frame, identity and metadata including manifest certificate. Evaluation accepts canonical prediction callback for VLM and cached methods.

Current jobs: the first verification/probe queue has finished. Seven checkpoints passed. CLIP RGB and optional probes were stopped by the watchdog when system memory pressure exceeded its limit; this is a resource stop, not a model/kernel failure. Pressure cleared before any retry. CPU tests now share locks/cpu-tests.lock; GPU jobs retain locks/gpu.lock. No method pilot yet.

Pause protocol: send all owners stop-and-commit, record commands/PIDs/logs for any live task jobs, commit this record, write at most ten outbox/paused-state.md lines, end turn. No new agents beyond these four. Code freeze 21:30; Gate 2 19:15, Gate 3 20:45, Gate 4 22:00; push only after director reply.

## 2026-10-02 17:00:33 -0300: PAUSED by director message 005

Quota pause requested. All four owners instructed to stop immediately, commit WIP and wait for RESUME. Do not start work until RESUME. Background jobs may finish. Gate 1 remains approved with changes and full A-I scope. No push or PR.

Main branch ends at 0724ea1 before this pause record. Integrated: shared resumable runtime 6bf94d0, provenance-bound feature cache 7cdc5b5, VLM implementation 0724ea1, reconstruction models/losses 5a8132d, first evaluation slice a042256. Main worktree was clean. Feature tests: 3 passed in 3.17 s; runtime/cache earlier batch: 4 passed in 1.92 s. VLM owner: 5 CPU tests passed including actual tiny PEFT backward/reload; no real Smol pilot launched at snapshot.

All eight HF RGB/SRM checkpoint compatibility checks passed. CLIP RGB retry min-val AUC 0.98190746, min-test 0.91680085; retry completed in 17.51 s, peak sampled RSS 1855828 KiB. Only mandated compatibility test accesses; no method test selection. New verification JSON remains local/verification/clip-none-retry/verification.json for later small-report copy. NF4 double-quant LoRA backward, PyG GCN/GAT/SAGE backward and pretrained Alex LPIPS backward probes all passed on ROCm after resource-pressure retry. Full LPIPS state available at local/pretrained/lpips-alex-full.pth (9.5 MiB).

Workstreams at pause (all paths relative to /home/lucas/ffb-research):
- wt-evaluation, research/plan-evaluation: first eb6dcbb integrated; suite/CLI/HPC/four-target synthetic test WIP, owner told to commit. Read newest branch commit at resume.
- wt-reconstruction, research/plan-reconstruction: cb43dfa adds fit/load/configs, not integrated yet. No jobs.
- wt-representations, research/plan-representations: 0aae693 core metric/graph code tested (16 tests, 2.22 s), then 4c00ca1 WIP training/inference; neither integrated. Do not cherry-pick copies of shared commits 76605ea/203ea5e.
- wt-vlm, research/plan-vlm: 6e30ed4 integrated as 0724ea1. Actual Smol pilot was approved but pause cancels any unstarted launch. Next task after real BF16/NF4 pilot is MoE I with val_fit/val_select and frozen experts.

Running background job snapshot:
- PID 95917; cwd `/home/lucas/ffb-research/wt-evaluation`; command `flock /home/lucas/ffb-research/locks/cpu-tests.lock env OMP_NUM_THREADS=2 MKL_NUM_THREADS=2 OPENBLAS_NUM_THREADS=2 HIP_VISIBLE_DEVICES= CUDA_VISIBLE_DEVICES= /home/lucas/ffb-research/FaceForgery-Benchmark/.venv/bin/python -m pytest tests/robustness tests/hpc/test_runtime.py tests/hpc/test_contracts.py -q`; log: tool-session stdout/stderr, no on-disk log; owner must report result on resume.
- PID 95924; cwd `/home/lucas/ffb-research/wt-evaluation`; command `/home/lucas/ffb-research/FaceForgery-Benchmark/.venv/bin/python -m pytest tests/robustness tests/hpc/test_runtime.py tests/hpc/test_contracts.py -q`; log: tool-session stdout/stderr, no on-disk log; owner must report result on resume.

No GPU job running in the pause snapshot. Completed prior jobs: CLIP retry session 18418, LPIPS export 41003, cache tests 26350, runtime/cache tests 17603, all exit 0.

Exact next actions after RESUME:
1. Read new director messages and agent pause replies; collect evaluation test result and WIP hashes, then inspect/cherry-pick only workstream-owned commits. Preserve protections and locks.
2. Send feature API/commit 7cdc5b5 to representations owner: open_cache returns float16 features, float32 logits, canonical frame, identity, metadata; extract_features binds checkpoint, manifest, preprocessing and code. First extract MobileNet SRM train/val caches, then remaining frozen experts, outside git. Queue behind any resumed Smol pilot with GPU flock and bounded watchdog. No caches have been extracted yet.
3. Run actual pinned Smol256M BF16 16-train/16-val four-update pilot and exact reload with VLM owner, <=3072 MiB RSS, <=600 s watchdog; NF4 after successful BF16 if resources permit. Keep all pilots off min-test until comparison frozen.
4. Integrate reconstruction fit/load; run genuine-only CAE/VAE/gated and reconstruction-error baseline, then residual/latent detectors. Record residual correlation against HF RGB/SRM on min-val. Full offline LPIPS state now staged.
5. Lead still must implement portable SBI landmark/mask/training path and three equal-budget generic-init arms (SBI, MFFI, mixed); MediaPipe landmark cache is local/data/train_real_landmarks.json, 202/202 valid. Include mask QA, counted failures and source-configurable hull.
6. Complete metric/graph cache round trips and low-label baselines, full-scale sparse graph server path; complete MoE baselines and leakage-safe calibration; integrate new CLI/server configs, full tests, REPORT and experiment matrix. Gate 2 19:15, Gate 3 20:45, freeze 21:30, Gate 4 22:00, final deadline 22:30 -03 unless director updates.

Final pause replies: evaluation committed 2b62bae9e80eb809b136fb860d503cd38ac7b30a and reports no running jobs; reconstruction cb43dfa and representations 4c00ca1 report no jobs; VLM 6e30ed4 reports no jobs and confirms GPU pilot was not launched. All owners stopped. The evaluation PIDs above were only the earlier snapshot.

## 2026-10-03 09:36 -03: resumed and integrated branch green

Message006 restores work after reboot; new gates: implementation11:15, pilots12:45, code freeze13:15, final13:40, hard stop14:20 today. Push remains held for Gate4 reply and forwarded SSH socket. New sysfs GPU path card0. Available memory13GiB, zero swap use, no memory pressure. No unrelated paths/processes touched.

Respawned original model choices in existing worktrees: evaluation_resume GPT6.1 max (suite/CLI), reconstruction_resume Astra max (A-D), representations_resume Astra max (G/H), vlm_moe_resume GPT6.1 max (F/I); no additional agents. Lead owns SBI/runtime/features/shared dependencies and integration. Saved commits integrated as817224b/fe052b3/8f4c4c3/8fcde4e. Fixture stdout capture fixed05d7b88. Runtime resume operational flags removed from run identity62d55ee,2tests pass. Full main suite338passed+20subtests in50.43s; no outstanding failure at that checkpoint.

SBI first implementation0ad82aa: matched404-example epoch for202 source reals, equal real/fake counts, sbi/mffi/mixed arms, seeded views, no silent landmark substitution, generic offline initialization and HF adaptation, differentialLR, full-state reload and bound calibration. Seven structural/round-trip tests passed; disk-backed landmark cache and condition identity added afterward, targeted verification ongoing. Production CPU landmark pass202/202 in2.36s; mask QA four seed-selected examples is in preflight/sbi-mask-qa.png/json and visually inspected. MediaPipe covers less forehead than dlib81, mask source/indices/extension configurable. SQLite geometry and on-demand masks keep full-scale RAM bounded; local image caching is opt-in.

All six MobileNet/DINO/CLIP RGB/SRM train+val cache pairs now complete under GPU lock,1000 images each split. float16 features and float32 logits; immutable identities, source hashes and RAM/runtime in preflight/frozen_features.json. Index files outside git: local/features/{mobilenet,dino,clip}-{srm,none}-index.json. All jobs exited0. No new min-test access. Optional NF4/PyG/LPIPS probe successes and eighthCLIP RGB compatibility result copied to preflight.

Director007 forwarded to reconstruction: matched x-only/residual-only/full ablations, explicit AE gradient modes, KL scale and224px server configs. Director008 forwarded to evaluation: all-family suite and calibrationCLI, pooled-real DF40 default, portable templates/dryruns. Workstreams implementing these before Gate2. Main full pytest session17916 is complete; feature queue73638 complete; landmark job89484 and QA35288 complete; no GPU job running. Latest targeted SBI tests session49676, log local/logs/sbi-final-unit.log.

Next: commit SQLite/SBI server configs and optional requirements, integrate pending G/H and MoE/VLM revisions, run all-family synthetic suite and cache-backed smokes. Pilot priorities remain SBI matched clean proxy; reconstruction controls/zero-shot/correlation; metric/graph baselines; held-out fusion; small VLM feasibility. Every pilot capped20minutes, serial GPU flock, watchdog3072MiB and memory pressure. No results selected or rejected yet.

## 2026-10-03 10:00 -03: Gate 2 implementation complete, pilots active

This entry supersedes earlier implementation-status and deadline entries. A-I implementations, common CLI and four-target synthetic evaluation are integrated. Main `d831dd4` passed 432 tests and 20 subtests in 66.84 s; evidence in preflight/gate2_pytest.json. Subsequent `b42a59f` adds explicit server reconstruction controls; `e7bb244` fixes complete NF4 vision-subtree exclusion, with six focused tests passed by owner. Director007/008 changes are addressed, including identical reconstruction control initialization, AE gradient semantics, explicit KL scale, 224px server configurations, all-family artifact rebuild/calibration, pooled-real DF40 policy and portable suite templates.

SBI production code and portable pilot driver committed `21cc723`; latest focused SBI/orchestration check: 26 passed in8.54s. Five-epoch generic MobileNet-SRM SBI/MFFI/mixed comparison preserves equal 65 updates and exact initial state, plus a separately labeled one-epoch HF adaptation smoke. Running/queued command: `scripts/sbi_pilot.py` with certified min-train/min-val, SQLite landmarks and explicit offline initial weights; output local/pilots/sbi-matched-seed42, log local/logs/sbi-pilot.log, watchdog local/logs/sbi-pilot-watchdog.json, tool session91170. No test access is authorized for development; freeze all three before their one-time comparative min-test evaluation.

Reconstruction owner authorized AE then detector phases in src.experimental.reconstruction.pilot, under GPU lock,3072MiB/1200s per phase. Local 128px CAE spatial controls are explicitly distinct from 224px VAE server starters. Source-only error scores and HF correlation report follow. Actual reconstruction predictions feed the required third MoE expert.

Representations owner finished21 DINO-SRM pilots and a same-mask metric-to-dynamic-graph smoke. SupCon valAUC0.989799 versus frozen-logit0.994013;5% GAT0.992664 versus same-mask MLP0.993781. Paired CIs do not support promotion. All results, including negative comparisons, are retained in pilot_results/representations. These are downstream low-label experiments on an already MFFI-supervised extractor. The owner is idle.

Actual SmolVLM-256M BF16 four-update pilot:16train/16val,460800 trainable LoRA parameters, exact reload, valAUC0.6640625; feasibility only. NF4 attempt stopped on an explicit software guard for unexpected vision-module conversion, not hardware exhaustion. Final-code BF16 and fixed NF4 reruns authorized with unchanged settings and180s/3072MiB bounds. Two-expert MobileNet MoE val_select AUC0.974508 versus mean0.972077; paired95%CI for gain[-0.001534,0.006956], so no advantage claim. Three-expert comparison pending reconstruction.

Gate2 report written to director/outbox/gate2-impl.md early. Evaluation owner now documents portable server commands; lead owns matrix, REPORT and final integration. Message009 quota pacing: same models, short reports, targeted reads/tests, full tests only at integration points. RAM9.4GiB available,swap0,no memory pressure at09:59. Protected paths untouched. Gate3 by12:45, freeze13:15, Gate4 by13:40, hard stop14:20; no push before director reply/socket.

## 2026-10-03 10:13 -03: approved Gate 2, two-seed SBI and degraded source evidence

Director010 approves Gate2. Added a second matched SBI seed and fixed degraded min-val inference. Director011 service-tier policy applied: no global config edits; do not reuse completed Astra threads, continue current reconstruction task, keep the existing Sol threads. Representations thread completed after final-code replay, no jobs. Reconstruction is completing its existing detector/correlation/proxy task; evaluation handles frozen-suite preparation; VLM/MoE handles three-expert fusion and fresh-process VLM reload.

SBI generic AUCs (sbi/mffi/mixed): seed42 0.44093/0.69054/0.64661; seed123 0.52246/0.74004/0.72308. Each realization has identical initial weights and65 updates per arm, exact reload, no test access. Paired SBI-minus-MFFI intervals are wholly negative for both seeds; mixed has no reproducible advantage. Separate HF adaptation42 is0.94307 versus unadapted0.94569. Evidence and full epoch losses are in pilot_results/sbi/seed42.json and seed123.json. GPU jobs91170/16451 completed,178.61/153.59seconds; RSS2553732/2519324KiB.

The deterministic224px min-val-degraded-proxy uses the existing augmentation operator with per-sample seed42 hashing, recipe9c99d6ba69c622db9a2dd842d819d038f49b12cd965bc607cdc44102448ec660. Both aliases preserve source IDs and labels, certify split=pilot and bind image inventories. All seven SBI clean/proxy suites passed frozen-calibration evaluation. Degraded AUCs: seed42 0.50000/0.62695/0.58651; seed123 0.49116/0.64550/0.60533; HF adaptation0.83453. No useful robustness signal from near-chance SBI. Session94844 completed65.04seconds, RSS1987628KiB. MobileNet SRM/RGB proxy caches also complete (session63273,12.01seconds,RSS1939900KiB), index local/features/mobilenet-val-degraded-index.json.

Reconstruction AE phase passed48.53seconds; error-only CAE/VAE/gated AUC0.47954/0.48253/0.50922. Low correlation did not help fixed means: all32 HF comparisons lost AUC. Detector phase passed160.10seconds with RSS2897188KiB; matched x-only/residual-only/full AUC0.61942/0.61168/0.63734. Owner is computing paired uncertainty and proxy evidence before any contribution claim. Full residual validation CSV/calibration now feed the mandatory three-expert MoE. No local training exceeds the watchdog caps.

Both final-code VLM fits completed four updates, but same-process reload crossed the3GiB RSS guard; these stops are retained. BF16/NF416-image AUC0.66406/0.42188. NF4 vision remained full precision after the exclusion fix and had no local memory advantage. Fresh-process inference-only reload checks are authorized after higher-priority jobs; no tuning or retraining. MoE matched-upstream-seed condition identity is being repaired without weakening realization hash checks. G/H seed scope remains downstream seeds on a fixed verified extractor.

REPORT.md and the21-condition experiment_matrix.yaml were drafted in ffadd90, together with explicit portable source paths, full-run output variables and server control budgets. Every matrix training config passed shared CLI dry resolution. SERVER_RUNBOOK.md includes certified population preparation, family commands, held-out calibration, all-four suite and separate scheduling. Initial diff audit:156 changed files, no protected-path/CUDA-pin modifications, no checkpoint/cache artifacts or file over1MB; inherited em dash in src/hpc/runtime.py was not added by this patch. Remote ICLR fetched and remains1b2b0e27693b695114c888c8e12e874714d950a6.

Next: finish reconstruction/fusion proxies and exact VLM reloads; commit frozen candidate hashes for all three generic SBI seed42 arms plus HF DINO-SRM; then execute the single authorized min-test/min-test-degraded-proxy pass. Evaluation owner is preparing source-only DINO calibration and configs; it has no permission to touch test until the freeze record is committed. Final Gate3 report, integrated tests and artifact review follow. No push before Gate4 reply/socket.


## 2026-10-03 10:32 -03: director 012 checks and stronger adaptation smokes

Director 012 requests held-out self-blend sanity inference, an unadapted MobileNet-SRM degradation control, degraded G/H/fusion comparisons, qualified SBI interpretation and prose spacing repairs. Evaluation owns the first two, the existing Sol VLM/MoE owner owns fusion and fresh-process VLM reloads, and a fresh Astra representations_proxy thread owns G/H degraded inference. Completed Astra threads remain unused under the service-tier policy. No global configuration was changed.

All 13 reconstruction runs and four source-proxy suites completed; final evidence is integrated. The full residual detector's advantage over the matched RGB control is uncertain on both clean and degraded validation. Every one of 104 naive means with HF scores reduced clean HF AUC. Three-expert MoE degraded AUC is 0.79324 versus SRM 0.83469 and LR 0.80925; its final paired controls are being recorded. No test access has occurred.

Fixed one-epoch DINO-SRM and CLIP-SRM SBI adaptations completed 13 optimizer updates each. Both initial processes exceeded the RSS guard while restoring the best bundle after training, and their failure records remain. Shared runtime commit aebd160 memory-maps saved optimizer bundles during best/last restoration; three exact-resume tests passed, including AdamW. Resumed finalization performed zero further updates and completed under the same guard. Clean min-val AUC is 0.967510 for DINO and 0.785502 for CLIP, both below their verified unadapted checkpoints. Fresh-process clean/degraded inference is queued, with no training or tuning. Resume telemetry measures finalization only; the original attempt and retry wall times will be reported separately.

The G/H cache representation contract incorrectly treated an unrelated git commit as a representation change despite identical implementation/checkpoint/preprocessing/package hashes. Commit 724d4f3 removes only that nonsemantic field from the cross-population key; cache provenance still retains the commit. Sixty-four focused tests passed. All 21 existing source settings are replayed unchanged because the stricter model implementation hash changed, with old rejection evidence retained. A single DINO-SRM degraded cache supplies every comparison; no target labels enter fitting.

REPORT.md, SBI.md and the draft PR description have a careful prose spacing pass and the requested negative-to-inconclusive SBI interpretation. Strong HF adaptation remains priority 1 on the literature and weak Celeb-DF column, with separate DF-40 swap/reenactment reports. The strong SRM-pair MoE server config now enforces matched upstream seeds and has explicit source-calibration/target-cache instructions. Gate 3 follows completed source evidence and one precommitted frozen test comparison. Push remains blocked on the required Gate 4 director reply and authenticated channel.

## 2026-10-03 12:29 -03: restart recovery and final evidence integration

Director 015 reports an 85-minute credential-store hang and restores the same session with file-based credential storage. No project configuration was changed for that recovery. The new deadlines supersede earlier entries: Gate 3 by 12:50, code freeze 13:20, Gate 4 by 13:45, push/PR from 13:45 to 14:05, hard stop 14:20. Earlier agent threads were lost; only evaluation_final and moe_final were recreated, both GPT 6.1 Sol max in separate worktrees. Reconstruction and representation owners have no remaining implementation assignment.

All requested source-side additions from director 012 are now recorded. Held-out SBI uses 408 validation real faces and 408 self-blends with zero generation failures; pure SBI AUC is 0.5878/0.6141 for seeds 42/123. Both selected checkpoints are only 13 updates old despite the completed 65-update allocation. The result is inconclusive and consistent with undertraining or selection-objective mismatch. Unadapted MobileNet-SRM degraded AUC is 0.8405 versus adapted 0.8345; the paired difference interval includes zero. All 22 G/H clean/degraded rows and nine comparisons are retained; SupCon and all six GNN conditions fail to improve the relevant shallow controls. Final VLM fresh-process reloads passed for BF16 and NF4 with maximum score errors 0.0 and 1.11e-16. No further model search is planned.

Director 013's protocol-comparability section is added to REPORT.md. The cited SBI paper reports 93.18% Celeb-DF AUC; the authors separately report 92.87% for their released c23 checkpoint, so the paper number is not labeled c23-specific. FF++-only DINO/CLIP initialization is a disclosed server preparation TODO: MFFI-supervised weights cannot become FF++-only by changing a manifest. Director 014's requested PR testing/decisions bullets and full coverage table are drafted in director/outbox/pr-description.md.

The named-expert source configuration bug is fixed in `ad9c096`: only actual path fields are resolved, preserving expert names, roles and kinds. A failing regression preceded the fix; 25 focused tests then passed. Matched native SBI prediction experts are certified in `c863185`; 23 MoE/SBI tests passed, including five canonical upstream seeds and changed-artifact rejection. Existing reconstruction/cache expert conditions are unchanged. Native run prediction CSVs and calibrations, not separate calibration exports, are documented in the server runbook. Strict implementation guards invalidate earlier MoE bundles; their rejection evidence is retained before an unchanged source-only two/three-expert replay.

The final three-arm generic SBI seed 42 comparison plus original DINO-SRM was frozen in `8d01770`. Proxy materialization succeeded, but the first suite stopped for system memory pressure after 5.12 seconds with no prediction or quality output. Failure files and an output-directory-only retry amendment are committed in `39e6961`. All frozen model/calibration/software/proxy hashes were reverified before explicit retry authorization. Evaluation owns the serial four-suite GPU queue and subsequent offline paired summary; MoE replay waits for that queue to finish. No source threshold, candidate, seed, polarity or training setting changed after freeze.

At 12:26, available RAM was about 12 GiB and memory PSI averages were zero; both earlier jobs had ended. Current evidence is under local/evaluation/final-pilot and local/logs/final-*. The last complete suite passed 445 tests plus 20 subtests at `109c494`; a final full suite is reserved after the two narrow source fixes above. Next: integrate frozen comparison and MoE replay, deliver Gate 3, run final tests/config audit, inspect the complete protected-path/artifact diff, finalize documentation and request Gate 4 review. Push remains held for the director's reply and authenticated channel.

## 2026-10-03 12:44 -03: pilot campaign and final regression suite complete

One completed frozen comparison is integrated in `1b7a2fb`, following the preserved aborted attempt and output-only retry amendment. Clean/degraded-proxy min-test AUC is 0.5341/0.5492 for generic SBI, 0.6300/0.5596 for matched MFFI, 0.6360/0.5338 for mixed, and 0.9192/0.8274 for original DINO-SRM. The clean SBI-minus-MFFI difference is -0.0959 with 95% CI [-0.1422, -0.0508]; both mixed comparisons and the degraded SBI comparison include zero. DINO clean AUC exactly matches the original compatibility check. All source thresholds stayed frozen. Four successful suites took 97.56 seconds combined, maximum sampled RSS 1942360 KiB. The evidence includes all access history, hashes, thresholded metrics, paired uncertainty and six explicitly labeled forensic plots. Representative plots were visually checked. No recipe or candidate changed after scoring.

The final source edit is `72dbdbf`, which removes one trailing blank line identified by the branch-wide whitespace audit. Both MoE configurations were replayed with identical native settings and fresh output directories under those final bytes. Every clean/degraded method score, source threshold and checkpoint tensor equals the historical result; maximum reload difference is 3.33e-16. Final two/three-expert runs took 23.01/27.51 seconds and used 902524/972632 KiB peak RSS. Evidence `4ae02df` retains initial code-hash rejection, pre-format replays, the formatting-hash rejection and one immediate watchdog-stopped attempt. No min-test access occurred in these source-only replays.

Director 016 identified the system-pressure stop during final pytest as overly sensitive. The old 1% full-PSI guard stopped that incomplete run after 46.59 seconds at 1466420 KiB own RSS; it is not counted as a pass. The task-local watchdog now stops below 2 GiB MemAvailable or above 10% full avg10, keeping the 3072 MiB process cap and time cap. The old script and failure record remain. The full identical pytest retry passed 449 tests plus 20 subtests in 70.56 seconds: 184 net added cases over the unmodified 265-test baseline. Peak RSS was 1653676 KiB, minimum available RAM 9169532 KiB and maximum full avg10 0.18%. Evidence is in preflight/final_pytest.json and final_pytest_stopped.json. No process outside this task was modified.

The refreshed matrix audit passed all 22 configurations and 154 CLI commands in 2.50 seconds without assets, fitting or target inference. An independent read-only review checked 46 PR coverage paths, all 16 runbook Python command examples and the evaluation Slurm wiring without a broken path or command. Both final owners have completed their work; no model job or GPU task remains. Source is frozen. Gate 3 reports these negative-to-inconclusive results and unchanged server priorities; remaining work is the final branch artifact audit, reviewable Gate 4 handoff and the director-authorized push/PR. The hard stop remains 14:20.
