# Experimental extensions: implementation and pilot report

Status: draft during the 2026-10-03 source-validation campaign. The final selection, degraded-proxy evidence, frozen min-test pass and final validation will replace the pending items below. No full-benchmark result is claimed.

The branch adds a common, provenance-bound execution path for reconstruction, SBI, compact VLM, metric learning, graph learning and frozen-expert fusion. All requested method families have tested training, artifact reload, calibration and four-target synthetic evaluation paths. The initial local evidence is mostly negative: the learned feature and graph methods did not beat strong shallow controls, reconstruction error alone is near chance, and the first generic SBI realization is weaker than matched MFFI training. These results constrain the server campaign rather than justify accuracy claims.

## Existing evidence and comparison targets

The historical values below come from computed repository CSVs, audited in [planning/evaluation_audit.md](planning/evaluation_audit.md), not from new inference. Historical threshold policies and prediction provenance differ from the new evaluator. Re-evaluate comparison checkpoints through the audited suite before making a new benchmark claim.

| Target | Historical comparison | AUC | Qualification |
| --- | --- | ---: | --- |
| Test | Robust global stacking | 0.942629 | One ensemble realization; stored zero SD is not uncertainty |
| Test | DINO RGB | 0.935800 +/- 0.006169 | Canonical five-seed sample SD |
| Test-D | Robust CLIP+DINO geometric | 0.878055 +/- 0.010988 | Member-seed identities not fully recorded in the summary row |
| Test-D | SRM CLIP+DINO geometric | 0.872449 +/- 0.004939 | Canonical five-seed sample SD |
| DF-40 | Robust CLIP self-ensemble | 0.843199 | One five-checkpoint ensemble; includes seed987 instead of2025 |
| DF-40 | SRM CLIP+DINO geometric | 0.835343 +/- 0.017354 | Canonical five-seed sample SD |
| Celeb-DF v2 video | SRM DINO+MobileNet mean | 0.773913 +/- 0.035340 | Weakest target and primary cross-dataset motivation |

The audited weak-technique examples are specific to the SRM CLIP+DINO mean ensemble: HeyGen0.566, StarGAN v2 0.574, DDIM0.588 and CollabDiff0.641. They are not universal rankings across models. For that ensemble, commercial avatars are weaker than the aggregated diffusion paradigm. New subgroup reports retain the exact model, technique counts and real-reference policy.

## What was implemented

| Direction | Implementation and scientific controls |
| --- | --- |
| A | Real-only CAE, convolutional VAE and gated deep-skip AE; structural restriction prevents shallow skip bypass |
| B | Residual fusion `[x, x_hat, abs(x-x_hat)]`, optional Sobel, identical initial backbones for x-only/residual-only/full controls, frozen/reconstruction-finetune/end-to-end modes |
| C | VAE mu/logvar/analytic-KL classifier and a self-contained spatial/latent mean artifact |
| D | FP32 L1/SSIM/KL, frozen offline pretrained LPIPS with pixel gradients, linear/cyclic beta, exact stochastic resume; explicit sum or mean-per-dimension KL and nats/sample logging |
| E | Landmark-bound SBI/MFFI/mixed training with matched cohorts and budgets; generic and HF adaptation separated; exact legacy SRM; audited shared forensic evaluation |
| F | <=3B compact VLM, BF16 LoRA and NF4 decoder QLoRA, frozen vision, teacher-forced multi-token Real/Fake scoring, strict base/adapter/processor inventory |
| G | Cached SRM features, normalized128/256 embeddings, supervised contrastive/hard-negative weighting, centroid/kNN/linear/MLP controls, silhouette and t-SNE diagnostics |
| H | GCN/GAT/GraphSAGE, shared label masks, linear/MLP/kNN/label-propagation/Correct-and-Smooth controls, bounded ANN neighborhoods, dynamic graphs, inductive/transductive contracts and edge-noise studies |
| I | Frozen pretrained experts, soft/top-k router, temperature, load balancing, dropout, individual/mean/geometric/LR comparisons on deterministic val_fit/val_select |

Non-classifier objectives use a separate experimental runtime. This preserves the legacy trainer and its existing behavior while providing AMP, correctly weighted accumulation, completed-epoch resume, optimizer/scheduler/scaler/RNG state, full or PEFT checkpoint callbacks, telemetry and explicit failures. The original MoE remains intact; the new late-fusion MoE avoids training scratch experts or inheriting historical test-evaluation side effects.

The shared Stack B evaluator now includes exact SRM and DTCWT reuse, AUC/AP/EER/F1, source-selected Youden accuracy and balanced accuracy, normalized confusion matrices, ROC/score plots, certified predictions, grouped bootstrap intervals and canonical seed aggregation. Youden and balanced-accuracy maximization choose the same ROC operating point; the recorded policy remains explicit. Interpolated EER thresholds are diagnostics and never deployment thresholds. Anomaly-score transforms are bounded ranking scores, not probability calibration.

## Local environment and scope

The workstation has a Ryzen7 5700X (16 CPU threads),15GiB RAM and an AMD Radeon RX9060XT gfx1200 with16GiB VRAM. The local environment is Python3.12.13, torch2.10.0+rocm7.1, torchvision0.25.0+rocm7.1. Exact setup and optional dependencies are in [ENVIRONMENT.md](ENVIRONMENT.md). Existing server CUDA dependency pins were preserved. GPU work is serialized, uses two CPU threads and a3072MiB sampled-RSS watchdog; jobs are capped at20minutes, with tighter VLM bounds. After reboot there was no memory pressure or swap use. No unrelated team processes or directories were touched.

The local dataset contains1000 images in each original split. Train has202 real/798 fake, val408/592 and test423/577. AE training uses only those202 source train reals. Images were predecoded for local use; all eight selected HF RGB/SRM checkpoints were rebuilt with the original architecture and exact224px ImageNet/SRM preprocessing and checked on min-val/min-test before reuse. Six MobileNet/DINO/CLIP RGB/SRM train+val feature-cache pairs were extracted once. Caches use float16 features, float32 logits, canonical row identities and hashes of images, manifests, checkpoints, preprocessing and implementation.

HF checkpoints already trained on min-train and selected on full MFFI val, which contains min-val. Thus their source feature pilots are optimistic and their low-label experiments measure downstream adaptation, not end-to-end label efficiency. There are no local DF-40 or Celeb-DF benchmark images. Synthetic four-target tests establish execution contracts, not model quality. Min-val and degraded min-val are development populations. The final min-test access policy is recorded separately below.

## Development pilot results

Every row below is a pilot/min-dataset result. The source-val sample is reused for model selection and threshold fitting unless val_select is explicitly stated. Bootstrap intervals are conditional on the trained checkpoint and supplied image groups. MFFI identity/video groups are unavailable, so intervals do not correct unknown identity dependence. They are not five-seed uncertainty estimates. Differences of a few hundredths should not be promoted without stronger evidence.

### SBI, generic initialization and equal budget

Each arm uses MobileNetV3-Large with exact SRM, identical seeded initial weights,404 balanced examples per epoch, five epochs and65 optimizer updates. The matched supervised control rotates real MFFI fakes; mixed uses both fake sources. Model selection still uses MFFI validation fakes. All202 source real landmark detections passed; four seed-selected masks were inspected. See [SBI.md](SBI.md) for the recipe and reproduction commands.

| Seed | Arm | Min-val AUC [95% bootstrap CI] | EER | Interpretation |
| ---: | --- | --- | ---: | --- |
|42| SBI |0.4409 [0.4057,0.4712]|0.5490|Poor transfer to MFFI fakes; polarity was not flipped |
|42| MFFI |0.6905 [0.6590,0.7208]|0.3530|Matched supervised control |
|42| Mixed |0.6466 [0.6167,0.6767]|0.3964|Below the supervised control |
|123| SBI |0.5225 [0.4864,0.5621]|0.4899|Same settings; near-chance transfer |
|123| MFFI |0.7400 [0.7124,0.7663]|0.3260|Matched supervised control |
|123| Mixed |0.7231 [0.6909,0.7511]|0.3431|No clear gain over MFFI |

Seed42 paired AUC differences versus MFFI: SBI -0.2496 [-0.2976,-0.1994], mixed -0.0439 [-0.0804,-0.0057]. Full reload predictions matched within1e-6. The complete campaign, including an independently labeled HF adaptation smoke, took178.61seconds and peaked at2553732KiB sampled RSS. The generic model has4205026 trainable parameters. Full metrics, budgets, loss/history references and provenance are in [pilot_results/sbi/seed42.json](pilot_results/sbi/seed42.json).

Seed123 repeats the qualitative result: SBI-minus-MFFI -0.2176 [-0.2653,-0.1750]; mixed-minus-MFFI -0.0170 [-0.0447,0.0087]. Its three arms completed in153.59seconds with exact reload and the same65 updates. Two seeds expose substantial realization noise but do not establish across-seed uncertainty. The independently implemented recipe on202 source reals has no demonstrated benefit here. Evidence: [pilot_results/sbi/seed123.json](pilot_results/sbi/seed123.json).

The one-epoch MobileNet-SRM HF adaptation smoke reached0.9431 [0.9289,0.9531] on min-val. Its unadapted verified checkpoint had0.9457. This is prior-supervised adaptation evidence and does not establish improvement or unseen-forgery performance.

### Reconstruction

Three-epoch128px width8 latent32 real-only models completed78 updates each. Error-only AUCs were CAE0.47954, VAE0.48253 and gated0.50922; all intervals include0.5. A separate one-epoch LPIPS/cyclic-beta VAE smoke had0.49561. The four-run AE phase took48.53seconds, with sampled RSS2410.5MiB. Across32 comparisons to the eight HF checkpoints, Pearson correlations ranged -0.02446 to0.06030. Low correlation did not make the scores useful: every naive fixed mean reduced the corresponding HF AUC, by0.00485 to0.03379. Evidence: [pilots/reconstruction_ae.json](pilots/reconstruction_ae.json).

Matched x-only/residual-only/full controls, latent detector, adaptation-mode smokes, spatial/latent mean and degraded-proxy comparisons are pending completion. These local spatial pilots use128px CAE; server starters deliberately use224px VAE. They are different recorded architectural conditions, so a local winner is not an exact validation of the server architecture.

### Metric learning and graph baselines

All21 DINO-SRM runs completed and were replayed without setting changes under the final implementation; total fit wall time was20.50seconds and peak RSS974596KiB, with no GPU use. SupCon AUC0.989799 did not beat the backbone logits0.994013 or raw centroid0.993993. At5% labels (51 selected images), GAT0.992664 did not beat same-mask MLP0.993781; at10% (101 images), GraphSAGE0.990656 did not beat MLP0.993678. SupCon-minus-MLP paired AUC CI was [-0.00488,0.00264];5% GAT-minus-MLP was [-0.00248,0.00005].

A separate same-mask SupCon-to-dynamic-GraphSAGE smoke completed with two distinct neighbor rebuilds and exact reload. Learned embeddings may be used as a graph input, but these results do not justify promoting them. The original run's rejection after implementation bytes changed was retained, followed by identical final-code replay and passing reload checks. All runs, including weak controls, are in [pilot_results/representations/README.md](pilot_results/representations/README.md).

### Frozen fusion and VLM feasibility

The two-expert MobileNet SRM/RGB ablation used494 val_fit and506 val_select images. Router AUC0.974508 versus mean0.972077 has paired gain CI[-0.001534,0.006956]; no routing advantage is established. It trained82082 parameters for96 updates. The mandatory third reconstruction expert and degraded inference are pending. See [moe.md](moe.md) and the retained [two-expert record](pilots/moe_mobilenet_twoexpert_seed42.json).

SmolVLM-256M is a feasibility study with16 train and16 val images, four updates and460800 LoRA parameters. BF16 achieved AUC0.6640625 versus zero-shot0.65625; NF4 achieved0.421875 versus0.3828125. These tiny observations cannot rank VLMs. The score is `softmax([sum log p(Real tokens), sum log p(Fake tokens)])[Fake]`, with each label teacher-forced after the identical image/prompt prefix. Multi-token labels and answer masking have explicit tests.

The first NF4 attempt correctly rejected unexpectedly quantized vision descendants; a pinned-Transformers exclusion mismatch was fixed and regression-tested. Final BF16/NF4 fits completed, but same-process base reload crossed the3GiB RSS guard. Fresh-process inference checks are pending. Detailed attempts and immutable base inventories are retained in [pilots/vlm_smol256_seed42.json](pilots/vlm_smol256_seed42.json). On this backend, NF4 retained489568512 FP32 bytes plus67276800 packed bytes, exceeding the BF16 base's514813056 bytes; there is no demonstrated local memory benefit. Do not extrapolate this small model/backend observation to CUDA servers. Qwen3-VL-2B is the server template; a nominal Qwen2.5-VL-3B was excluded because actual parameter count exceeds the strict cap.

### Degraded proxy and frozen test access

Pending: fixed repo-operator min-val-degraded-proxy, inference only with clean frozen thresholds; then at most three frozen candidates plus HF DINO-SRM through the actual audited pilot suite on min-test and min-test-degraded-proxy. The manifest recipe, candidate/checkpoint/calibration hashes and every access will be committed before the final pass. No new min-test access has occurred since the eight required checkpoint compatibility checks at this draft checkpoint.

## Server campaign and experiment matrix

The executable command inventory is [experiment_matrix.yaml](experiment_matrix.yaml); setup, certified manifests, cache preparation, calibration and four-target commands are in [SERVER_RUNBOOK.md](SERVER_RUNBOOK.md). Every condition supports canonical seeds42,123,2024,7,2025 and distinct outputs. Priorities are conditional on the completed source-val pilots, not claims of superiority.

| Priority | Matrix rows | Architecture / regime | Why run it |
| ---: | --- | --- | --- |
|1| sbi-hf-dino-srm; sbi-hf-clip-srm | Verified strong SRM backbones, SBI adaptation | Controlled cross-dataset hypothesis, preserving unadapted and simple-fusion baselines |
|1| sbi-generic-sbi; sbi-generic-mffi; sbi-generic-mixed | Generic MobileNet, matched budgets | Establish whether the negative tiny-source result persists with enough genuine faces |
|2| srm-pair-plus-new-expert | DINO-SRM + CLIP-SRM + certified new expert | Advance only if source-held-out complementarity warrants it |
|2| moe-srm-rgb-reconstruction-soft; top2 | Frozen three-role experts | Mandatory routing attribution against all simple controls |
|3| reconstruction-cae; vae; gated |224px real-only AE | Test bottleneck/discrepancy hypothesis at full genuine-data scale |
|3| reconstruction-residual-x_only; residual_only; frozen | Identical ResNet initialization | Separate RGB classification from reconstruction contribution |
|3| reconstruction-residual-recon_finetune; end_to_end | Explicit AE gradient semantics | Distinguish reconstruction adaptation from CE gradient training |
|3| reconstruction-latent; spatial-latent-mean | Latent MLP and fixed fusion | Complementarity test with a cheap control |
|4| vlm-qwen3-2b-nf4 | Frozen vision, text QLoRA | Exploratory independent expert; validate hardware fit first |
|4| metric-srm | SupCon and linear/MLP/centroid/kNN | Exploratory only after stronger controls |
|4| graph-srm-low-label | GNN and five same-mask controls | Downstream label-efficiency study; no local GNN promotion |

For each selected trained run, calibration is frozen from certified MFFI val, or val_select for learned fusion. One suite config then executes Test, paired Test-D, DF-40 and Celeb-DF v2; the matrix also lists four separate commands after the runbook's split-config preparation. The combined suite is preferred because it produces the paired Test/Test-D interval. DF-40 defaults to `pooled_all_real`, matching historical tables but confounding source domain. Switch to `source_matched` only after certifying annotations. DF-40 `cdf/frames` fakes overlap the Celeb-DF source domain. Celeb-DF's official1=real labels are explicitly converted to internal1=fake, and its primary score is mean frame `p_fake` per video across the official518-video list.

The full-scale graph path uses bounded feature caches, FAISS neighborhood search with sampled recall checks, sparse neighborhoods and ego minibatches. It does not form a dense all-pairs graph. Inductive target queries cannot alter source neighborhoods or one another; transductive runs are separately named, calibrated and reported. Full-scale runtime and memory remain server measurements, not inferred from1000-image pilots.

## Validation, failures and remaining limitations

The unmodified baseline passed265 tests and20 subtests. Gate2 integration passed432 tests and20 subtests in66.84seconds; focused NF4 regression checks passed afterward. All families have actual fit/calibrate/reload/four-target synthetic fixtures, including absolute DF-40 paths, paired degradation IDs and Celeb video aggregation. Final integrated suite, config audit, protected-path diff, artifact review, frozen min-test evidence and clean branch status are pending before Gate4.

Retained failures include memory-pressure-stopped preflight jobs, the old-code bundle reload rejection, the NF4 subtree guard failure, same-process VLM reload RSS stops and an early two-expert router saturation caused by near-constant feature scaling. Fixes and retries preserve the original evidence. No labels or score polarity were selected by AUC, no target threshold was fitted and no unreadable image was silently replaced.

Legacy cleanup candidates were documented rather than modified: automatic test evaluation from the old trainer; unreadable-image substitution; double class balancing; misleading finetune/finetune_robust paths; hardcoded thresholds and paths; incomplete historical provenance; old MoE scratch-pretraining assumptions; dense expert compute despite sparse routing. No paper, manuscript, prior results, tables or figures were changed. Full-scale CUDA execution, the four real benchmark populations and canonical multi-seed comparisons remain the server campaign.
