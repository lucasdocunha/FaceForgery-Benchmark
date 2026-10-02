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
