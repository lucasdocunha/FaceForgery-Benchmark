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
