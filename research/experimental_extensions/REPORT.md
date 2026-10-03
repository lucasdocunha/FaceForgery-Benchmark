# Experimental extensions: implementation and pilot report

Status: local implementation, pilot campaign and final test suite complete on 2026-10-03. Full-scale CUDA and benchmark validation remain server work. No full-benchmark result is claimed.

The branch adds a common, provenance-bound execution path for reconstruction, SBI, compact VLM, metric learning, graph learning and frozen-expert fusion. All requested method families have tested training, artifact reload, calibration and four-target synthetic evaluation paths. The local comparisons are weak or inconclusive: reconstruction error is near chance, generic SBI is weaker than matched MFFI training on the available MFFI fakes, and feature methods face a selection-related ceiling on clean min-val. These measurements validate mechanisms and identify limitations; they do not settle the cross-dataset hypotheses.

## Existing evidence and comparison targets

The historical values below come from computed repository CSVs, audited in [planning/evaluation_audit.md](planning/evaluation_audit.md), not from new inference. Historical threshold policies and prediction provenance differ from the new evaluator. Re-evaluate comparison checkpoints through the audited suite before making a new benchmark claim.

| Target | Historical comparison | AUC | Qualification |
| --- | --- | ---: | --- |
| Test | Robust global stacking | 0.942629 | One ensemble realization; stored zero SD is not uncertainty |
| Test | DINO RGB | 0.935800 +/- 0.006169 | Canonical five-seed sample SD |
| Test-D | Robust CLIP+DINO geometric | 0.878055 +/- 0.010988 | Member-seed identities not fully recorded in the summary row |
| Test-D | SRM CLIP+DINO geometric | 0.872449 +/- 0.004939 | Canonical five-seed sample SD |
| DF-40 | Robust CLIP self-ensemble | 0.843199 | One five-checkpoint ensemble; includes seed 987 instead of 2025 |
| DF-40 | SRM CLIP+DINO geometric | 0.835343 +/- 0.017354 | Canonical five-seed sample SD |
| Celeb-DF v2 video | SRM DINO+MobileNet mean | 0.773913 +/- 0.035340 | Weakest target and primary cross-dataset motivation |

The audited weak-technique examples are specific to the SRM CLIP+DINO mean ensemble: HeyGen 0.566, StarGAN v2 0.574, DDIM 0.588 and CollabDiff 0.641. They are not universal rankings across models. For that ensemble, commercial avatars are weaker than the aggregated diffusion paradigm. New subgroup reports retain the exact model, technique counts and real-reference policy.

## What was implemented

| Direction | Implementation and scientific controls |
| --- | --- |
| A | Real-only CAE, convolutional VAE and gated deep-skip AE; structural restriction prevents shallow skip bypass |
| B | Residual fusion `[x, x_hat, abs(x-x_hat)]`, optional Sobel, identical initial backbones for x-only/residual-only/full controls, frozen/reconstruction-finetune/end-to-end modes |
| C | VAE mu/logvar/analytic-KL classifier and a self-contained spatial/latent mean artifact |
| D | FP32 L1/SSIM/KL, frozen offline pretrained LPIPS with pixel gradients, linear/cyclic beta, exact stochastic resume; explicit sum or mean-per-dimension KL and nats/sample logging |
| E | Landmark-bound SBI/MFFI/mixed training with matched cohorts and budgets; generic and HF adaptation separated; exact legacy SRM; audited shared forensic evaluation |
| F | <=3B compact VLM, BF16 LoRA and NF4 decoder QLoRA, frozen vision, teacher-forced multi-token Real/Fake scoring, strict base/adapter/processor inventory |
| G | Cached SRM features, normalized 128/256 embeddings, supervised contrastive/hard-negative weighting, centroid/kNN/linear/MLP controls, silhouette and t-SNE diagnostics |
| H | GCN/GAT/GraphSAGE, shared label masks, linear/MLP/kNN/label-propagation/Correct-and-Smooth controls, bounded ANN neighborhoods, dynamic graphs, inductive/transductive contracts and edge-noise studies |
| I | Frozen pretrained experts, soft/top-k router, temperature, load balancing, dropout, individual/mean/geometric/LR comparisons on deterministic val_fit/val_select |

Non-classifier objectives use a separate experimental runtime. This preserves the legacy trainer and its existing behavior while providing AMP, correctly weighted accumulation, completed-epoch resume, optimizer/scheduler/scaler/RNG state, full or PEFT checkpoint callbacks, telemetry and explicit failures. The original MoE remains intact; the new late-fusion MoE avoids training scratch experts or inheriting historical test-evaluation side effects.

The shared Stack B evaluator now includes exact SRM and DTCWT reuse, AUC/AP/EER/F1, source-selected Youden accuracy and balanced accuracy, normalized confusion matrices, ROC/score plots, certified predictions, grouped bootstrap intervals and canonical seed aggregation. Youden and balanced-accuracy maximization choose the same ROC operating point; the recorded policy remains explicit. Interpolated EER thresholds are diagnostics and never deployment thresholds. Anomaly-score transforms are bounded ranking scores, not probability calibration.

## Local environment and scope

The workstation has a Ryzen 7 5700X (16 CPU threads), 15 GiB RAM and an AMD Radeon RX 9060 XT gfx1200 with 16 GiB VRAM. The local environment is Python 3.12.13, torch 2.10.0+rocm7.1, torchvision 0.25.0+rocm7.1. Exact setup and optional dependencies are in [ENVIRONMENT.md](ENVIRONMENT.md). Existing server CUDA dependency pins were preserved. GPU work is serialized, uses two CPU threads and a 3072 MiB sampled-RSS watchdog; jobs are capped at 20 minutes, with tighter VLM bounds. Occasional system memory pressure stopped individual jobs; resource snapshots and retries are retained. No unrelated team processes or directories were touched.

The local dataset contains 1000 images in each original split. Train has 202 real/798 fake, val 408/592 and test 423/577. AE training uses only those 202 source train reals. Images were predecoded for local use; all eight selected HF RGB/SRM checkpoints were rebuilt with the original architecture and exact 224 px ImageNet/SRM preprocessing and checked on min-val/min-test before reuse. Six MobileNet/DINO/CLIP RGB/SRM train+val feature-cache pairs were extracted once. Caches use float16 features, float32 logits, canonical row identities and hashes of images, manifests, checkpoints, preprocessing and implementation.

HF checkpoints already trained on min-train and selected on full MFFI val, which contains min-val. Thus their source feature pilots are optimistic and their low-label experiments measure downstream adaptation, not end-to-end label efficiency. There are no local DF-40 or Celeb-DF benchmark images. Synthetic four-target tests establish execution contracts, not model quality. Min-val and degraded min-val are development populations. The final min-test access policy is recorded separately below.

## Development pilot results

Every row below is a pilot/min-dataset result. The source-val sample is reused for model selection and threshold fitting unless val_select is explicitly stated. Bootstrap intervals are conditional on the trained checkpoint and supplied image groups. MFFI identity/video groups are unavailable, so intervals do not correct unknown identity dependence. They are not five-seed uncertainty estimates. Differences of a few hundredths should not be promoted without stronger evidence.

### SBI, generic initialization and equal budget

Each arm uses MobileNetV3-Large with exact SRM, identical seeded initial weights, 404 balanced examples per epoch, five epochs and 65 allocated optimizer updates. The matched supervised control rotates real MFFI fakes; mixed uses both fake sources. Model selection still uses MFFI validation fakes. All runs completed their budget, but both pure SBI seeds selected their epoch-0 checkpoint after 13 updates. Selected MFFI/mixed checkpoints were steps 52/65 for seed 42 and 65/52 for seed 123. Thus equal training allocation does not imply equal age of the selected checkpoint. All 202 source real landmark detections passed; four seed-selected masks were inspected. See [SBI.md](SBI.md) for the recipe and reproduction commands.

| Seed | Arm | Min-val AUC [95% bootstrap CI] | EER | Interpretation |
| ---: | --- | --- | ---: | --- |
| 42 | SBI | 0.4409 [0.4057, 0.4712] | 0.5490 | Poor transfer to MFFI fakes; polarity was not flipped |
| 42 | MFFI | 0.6905 [0.6590, 0.7208] | 0.3530 | Matched supervised control |
| 42 | Mixed | 0.6466 [0.6167, 0.6767] | 0.3964 | Below the supervised control |
| 123 | SBI | 0.5225 [0.4864, 0.5621] | 0.4899 | Same settings; near-chance transfer |
| 123 | MFFI | 0.7400 [0.7124, 0.7663] | 0.3260 | Matched supervised control |
| 123 | Mixed | 0.7231 [0.6909, 0.7511] | 0.3431 | No clear gain over MFFI |

Seed 42 paired AUC differences versus MFFI: SBI -0.2496 [-0.2976, -0.1994], mixed -0.0439 [-0.0804, -0.0057]. Full reload predictions matched within 1e-6. The complete campaign, including an independently labeled HF adaptation smoke, took 178.61 seconds and peaked at 2553732 KiB sampled RSS. The generic model has 4205026 trainable parameters. Full metrics, budgets, loss/history references and provenance are in [pilot_results/sbi/seed42.json](pilot_results/sbi/seed42.json).

Seed 123 repeats the qualitative result: SBI-minus-MFFI -0.2176 [-0.2653, -0.1750]; mixed-minus-MFFI -0.0170 [-0.0447, 0.0087]. Its three arms completed in 153.59 seconds with exact reload and the same 65 updates. Two seeds expose substantial realization noise but do not establish across-seed uncertainty. The independently implemented recipe on 202 source reals has no demonstrated benefit on these MFFI validation fakes. Evidence: [pilot_results/sbi/seed123.json](pilot_results/sbi/seed123.json).

The one-epoch MobileNet-SRM HF adaptation smoke reached 0.9431 [0.9289, 0.9531] on min-val. Its unadapted verified checkpoint had 0.9457. This is prior-supervised adaptation evidence and does not establish improvement or unseen-forgery performance.

Fixed one-epoch DINO/CLIP adaptation smokes used the same 202 reals, 13 updates, batch 2 with accumulation 16, and learning rates 1e-5/1e-4 for backbone/head. Neither was tuned after scoring. Their clean/degraded AUCs were:

| HF SRM model | Unadapted clean / degraded | SBI-adapted clean / degraded | Adapted-minus-unadapted degraded AUC [95% CI] |
| --- | --- | --- | --- |
| DINO | 0.9940 / 0.9614 | 0.9675 / 0.8958 | -0.0656 [-0.0819, -0.0517] |
| CLIP | 0.9840 / 0.9345 | 0.7855 / 0.7302 | -0.2043 [-0.2360, -0.1735] |

Both completed their only epoch, then hit the RSS guard during best-bundle restoration. Memory-mapped checkpoint loading fixed that finalization cost; resumption performed zero additional updates. Fresh-process clean/degraded suites reproduced the saved clean scores exactly, in 61.53 seconds for DINO and 27.52 seconds for CLIP, below the RSS cap. The records retain initial stops, finalization-only telemetry, complete histories, unadapted controls and all calibrated metrics. These short adaptations show source-discrimination loss, especially for CLIP; they provide no local efficacy support for this recipe. Evidence: [pilot_results/sbi/hf_strong_adaptation_seed42.json](pilot_results/sbi/hf_strong_adaptation_seed42.json).

The held-out sanity check uses only the 408 validation real faces and 408 generated self-blends. All landmark detections and generation steps passed; four fixed-seed example pairs were visually inspected. The recipe is unchanged, no fitting occurs, MFFI-val thresholds remain frozen, and the bootstrap resamples each real/SBI pair together.

| Seed | SBI-only held-out AUC [95% CI] | MFFI-control held-out AUC | Mixed held-out AUC |
| ---: | --- | ---: | ---: |
| 42 | 0.5878 [0.5653, 0.6113] | 0.5159 | 0.7020 |
| 123 | 0.6141 [0.5896, 0.6403] | 0.5292 | 0.6836 |

Pure SBI learns some held-out signal, but separation is modest and selected-checkpoint age is only 13 updates. This does not establish strong artifact anti-transfer or a generator defect; the pilots are inconclusive and consistent with undertraining or a mismatch between checkpoint selection and the SBI objective. All six sanity suites finished in 27.02 seconds, peak sampled RSS 1953808 KiB. Full metrics and provenance are in [the held-out SBI and MobileNet control record](pilot_results/heldout_sbi_and_mobilenet_controls.json).

SBI targets blending-boundary artifacts associated with face swaps and reenactment. MFFI includes generation families that need not have such a boundary, and the local manifests lack manipulation-family labels. MFFI-val is therefore a weak proxy for the intended Celeb-DF v2 face-swap and DF-40 swap/reenactment hypotheses. Even the full 65-update allocation on 202 real faces is far below the published SBI regime. Local evidence is negative-to-inconclusive and does not test those cross-dataset hypotheses. The priority 1 DINO/CLIP adaptation campaign rests on the literature and Celeb-DF being the weakest benchmark column, with separate DF-40 swap/reenactment reporting.

### Reconstruction

Three-epoch 128 px width 8 latent 32 real-only models completed 78 updates each. Error-only AUCs were CAE 0.47954, VAE 0.48253 and gated 0.50922; all intervals include 0.5. A separate one-epoch LPIPS/cyclic-beta VAE smoke had 0.49561. The four-run AE phase took 48.53 seconds, with sampled RSS 2410.5 MiB. Across 32 comparisons to the eight HF checkpoints, Pearson correlations ranged -0.02446 to 0.06030. Low correlation did not make the scores useful: every naive fixed mean reduced the corresponding HF AUC, by 0.00485 to 0.03379. Evidence: [pilots/reconstruction_ae.json](pilots/reconstruction_ae.json).

All 13 reconstruction runs completed. Matched x-only/residual-only/full detectors share the exact initial state and 250 updates over two epochs. Clean AUCs are 0.61942/0.61168/0.63734; full-minus-x is 0.01792 with paired 95% CI [-0.00129, 0.03924]. Latent AUC is 0.50725 and spatial/latent mean is 0.60216, below spatial alone. Adaptation modes are one-epoch gradient/fit smokes and cannot be compared as equal-budget efficacy trials.

On the fixed degraded proxy, x-only/residual/full become 0.55725/0.54068/0.56024. Full-minus-x is 0.00299 with CI [-0.01590, 0.02291]. All three discriminative controls lose AUC under degradation; full drops 0.07710 with CI [-0.10568, -0.04647]. Across 104 clean HF comparisons, every naive fixed mean lowers HF AUC, by 0.00447 to 0.04245. Thus there is no supported reconstruction contribution at this scale. The detector phase took 160.10 seconds and the four proxy suites 35.02 seconds, both under the watchdog. Full source and proxy evidence is in [pilots/reconstruction_pilot.json](pilots/reconstruction_pilot.json) and [pilots/reconstruction_proxy.json](pilots/reconstruction_proxy.json). These local spatial pilots use 128 px CAE; server starters deliberately use 224 px VAE. They are different recorded architectural conditions, so a local winner is not an exact validation of the server architecture.

### Metric learning and graph baselines

All 21 DINO-SRM source configurations completed. The initial integration replay took 20.50 seconds; the final source-code replay took 18.40 seconds, peak RSS 934312 KiB, with no GPU use and unchanged settings. SupCon AUC 0.989799 did not beat the backbone logits 0.994013 or raw centroid 0.993993. At 5% labels (51 selected images), GAT 0.992664 did not beat same-mask MLP 0.993781; at 10% (101 images), GraphSAGE 0.990656 did not beat MLP 0.993678. SupCon-minus-MLP paired AUC CI was [-0.00488, 0.00264]; 5% GAT-minus-MLP was [-0.00248, 0.00005].

A separate same-mask SupCon-to-dynamic-GraphSAGE smoke completed with two distinct neighbor rebuilds and exact reload. The HF 0.994 clean-val ceiling is partly explained by min-val belonging to the full validation split that selected the backbone. Failure to exceed that ceiling is weak evidence about generalization. The fixed degraded proxy provides more room to differ, while keeping every source threshold and label mask unchanged:

| Representation/control | Clean AUC | Degraded AUC |
| --- | ---: | ---: |
| Original DINO-SRM logits | 0.99401 | 0.96143 |
| Raw centroids | 0.99399 | 0.96156 |
| All-label MLP | 0.99097 | 0.95373 |
| SupCon | 0.98980 | 0.93921 |
| 5% MLP | 0.99378 | 0.95839 |
| 5% GCN / GAT / GraphSAGE | 0.98316 / 0.99266 / 0.99160 | 0.94930 / 0.95449 / 0.95538 |
| 10% MLP | 0.99368 | 0.95882 |
| 10% GCN / GAT / GraphSAGE | 0.98574 / 0.98996 / 0.99066 | 0.95178 / 0.95399 / 0.95570 |

Degraded SupCon-minus-MLP AUC is -0.01452 [-0.02495, -0.00405]. All six GNN-minus-same-mask-MLP intervals are below zero; raw centroids remain a strong control. These are conditional development comparisons, not independent or across-seed confirmation. All 22 clean/degraded metric rows and nine paired controls are in [the proxy record](pilot_results/representations/dino_srm_degraded_proxy.json) and [CSV](pilot_results/representations/dino_srm_degraded_proxy.csv). The original bundle rejection, overstrict floating-point audit and CSV parsing audit were retained. Final reloads reproduce predictions exactly, and round-trip CSV parsing preserves threshold ties and confusion counts. See [the complete representation record](pilot_results/representations/README.md). No learned metric or GNN is promoted over the stronger controls.

### Frozen fusion and VLM feasibility

The two-expert MobileNet SRM/RGB ablation used 494 val_fit and 506 val_select images. Router AUC 0.974508 versus mean 0.972077 has paired gain CI [-0.001534, 0.006956]; no routing advantage is established. It trained 82082 parameters for 96 updates. The mandatory third reconstruction expert has now been fitted on the same 494/506 split. Clean val_select AUCs are router 0.97626, LR 0.97196, mean 0.96198 and geometric 0.96811; single SRM/RGB/reconstruction experts score 0.94893/0.95947/0.65438. Adding a weak expert depresses naive averaging, so router superiority to the three-way mean alone is not evidence that reconstruction helps.

On degraded val_select, AUCs are router 0.79324, LR 0.80925, mean 0.78861, geometric 0.72921; single SRM/RGB/reconstruction score 0.83469/0.65938/0.57213. The router loses 0.18302 from clean validation, paired 95% CI [-0.21947, -0.14703]. Its degraded difference from SRM is -0.04145 [-0.07425, -0.00906] and from LR is -0.01601 [-0.03649, 0.00464]. The two-expert router/mean/LR score 0.79394/0.81074/0.80940 on the same degraded rows. Three-expert versus two-expert router clean gain is only 0.00175 [-0.00321, 0.00660]; no reconstruction benefit is established. See [moe.md](moe.md), the [three-expert clean record](pilots/moe_mobilenet_reconstruction_seed42.json), its [degraded controls](pilots/moe_mobilenet_reconstruction_degraded_seed42.json), and the retained [two-expert record](pilots/moe_mobilenet_twoexpert_seed42.json).

Both two/three-expert configurations were replayed unchanged against the final implementation after the native SBI metadata fix and an EOF formatting correction. Every clean/degraded method score, source threshold and checkpoint tensor matched the earlier result exactly; fresh reload error was at most 3.33e-16. The two replays took 23.01/27.51 seconds and peaked at 902524/972632 KiB RSS. Strict source-hash rejections and the immediately watchdog-stopped attempt remain in [the final replay record](pilots/moe_current_code_replay_seed42.json). No scientific setting was changed.

SmolVLM-256M is a feasibility study with 16 train and 16 val images, four updates and 460800 LoRA parameters. BF16 achieved AUC 0.6640625 versus zero-shot 0.65625; NF4 achieved 0.421875 versus 0.3828125. These tiny observations cannot rank VLMs. The score is `softmax([sum log p(Real tokens), sum log p(Fake tokens)])[Fake]`, with each label teacher-forced after the identical image/prompt prefix. Multi-token labels and answer masking have explicit tests.

The first NF4 attempt correctly rejected unexpectedly quantized vision descendants; a pinned-Transformers exclusion mismatch was fixed and regression-tested. Final BF16/NF4 fits completed, but same-process base reload crossed the 3 GiB RSS guard. Fresh-process inference-only reloads then passed in 11.01/11.51 seconds, with maximum score errors 0.0/1.11e-16 and peak sampled RSS 2330120/2345880 KiB. Detailed attempts and immutable base inventories are retained in [pilots/vlm_smol256_seed42.json](pilots/vlm_smol256_seed42.json). On this backend, NF4 retained 489568512 FP32 bytes plus 67276800 packed bytes, exceeding the BF16 base's 514813056 bytes; there is no demonstrated local memory benefit. Do not extrapolate this small model/backend observation to CUDA servers. Qwen3-VL-2B is the server template; a nominal Qwen2.5-VL-3B was excluded because actual parameter count exceeds the strict cap.

### Degraded proxy and frozen test access

A fixed seed 42 draw from the repository RandomizedRobustAugment operator creates `min-val-degraded-proxy`: 224 px, rotation/color/brightness/sharpness/blur/JPEG/noise according to that existing recipe, lossless RGB8 output, no severity selection. Per-image seeds derive from SHA256(seed, sample_id), and all source/degraded bytes are inventoried. It is not the benchmark Test-D distribution. All scoring reuses clean-val thresholds without refitting.

| SBI condition | Clean min-val AUC | Degraded-proxy AUC | Paired delta 95% CI |
| --- | ---: | ---: | --- |
| Generic SBI 42 | 0.4409 | 0.5000 | [0.0231, 0.0941] |
| Generic MFFI 42 | 0.6905 | 0.6270 | [-0.0898, -0.0308] |
| Generic mixed 42 | 0.6466 | 0.5865 | [-0.0925, -0.0278] |
| Unadapted HF MobileNet-SRM 42 | 0.9457 | 0.8405 | [-0.1272, -0.0834] |
| HF SBI adaptation 42 | 0.9431 | 0.8345 | [-0.1282, -0.0881] |
| Generic SBI 123 | 0.5225 | 0.4912 | [-0.0646, 0.0012] |
| Generic MFFI 123 | 0.7400 | 0.6455 | [-0.1254, -0.0630] |
| Generic mixed 123 | 0.7231 | 0.6053 | [-0.1501, -0.0857] |

An increase from below chance to chance is not useful robustness. The supervised/mixed arms lose discrimination and threshold accuracy under this degradation. The seven clean/degraded audited suites completed in 65.04 seconds, peak sampled RSS 1987628 KiB. Evidence: [pilot_results/sbi/validation_degraded_proxy.json](pilot_results/sbi/validation_degraded_proxy.json). The unadapted MobileNet control reuses verified clean predictions and matching original-logit degraded caches. Adaptation-minus-unadapted degraded AUC is -0.0060 [-0.0137, 0.0030], so no robustness improvement is established; [paired evidence](pilot_results/sbi/mobilenet_adaptation_control.json) retains the original thresholds. Reconstruction and held-out fusion proxy results above also show no supported robustness gain.

The frozen min-test comparison used all three generic SBI seed 42 arms and HF DINO-SRM. Seed 42 was the first predeclared realization; seed 123 checked source-side variability and was not used to choose a favorable test realization. Checkpoint, calibration and software hashes were committed in `8d01770` before proxy materialization or scoring. The first SBI suite stopped after 5.12 seconds because of system memory pressure, with no prediction CSV or quality metric emitted. Its partial image access is unknown. The retry changed only its output directory, preserved the failure evidence and rechecked every frozen input. That amendment was committed in `39e6961` before retry authorization. One completed comparison then scored each frozen model on 1000 clean and 1000 paired degraded images. The eight earlier original-checkpoint compatibility checks remain part of the recorded access history.

| Frozen seed 42 model | Min-test AUC [95% CI] | Min-test-degraded-proxy AUC [95% CI] | Frozen source threshold |
| --- | --- | --- | ---: |
| Generic SBI | 0.5341 [0.4997, 0.5670] | 0.5492 [0.5205, 0.5845] | 0.2882 |
| Generic MFFI | 0.6300 [0.5968, 0.6618] | 0.5596 [0.5295, 0.5992] | 0.7631 |
| Generic mixed | 0.6360 [0.6007, 0.6727] | 0.5338 [0.4969, 0.5704] | 0.3183 |
| Original HF DINO-SRM | 0.9192 [0.9037, 0.9362] | 0.8274 [0.8040, 0.8525] | 0.1919 |

SBI-minus-MFFI clean AUC is -0.0959 [-0.1422, -0.0508]; mixed-minus-MFFI is +0.0060 [-0.0266, 0.0404]. Both degraded differences include zero. This frozen one-seed comparison supports no SBI or mixed-training benefit. The SBI source threshold misclassifies 92.7% of clean test reals as fake, illustrating poor threshold transfer; the threshold remains unchanged. DINO's clean AUC exactly reproduces the original compatibility check, but its larger prior training budget makes it a reference, not a matched attribution control. The four successful suites took 97.56 seconds combined, with maximum sampled RSS 1942360 KiB. [Comparison evidence](pilot_results/frozen_min_test/comparison.json), [all calibrated metrics](pilot_results/frozen_min_test/metrics.csv) and [labeled diagnostic plots](pilot_results/frozen_min_test/README.md) retain the uncertainty, artifacts, failure and retry history. No training, calibration, candidate or recipe changed after these target scores.

## Server campaign and experiment matrix

The executable command inventory is [experiment_matrix.yaml](experiment_matrix.yaml); setup, certified manifests, cache preparation, calibration and four-target commands are in [SERVER_RUNBOOK.md](SERVER_RUNBOOK.md). Every condition supports canonical seeds 42, 123, 2024, 7, 2025 and distinct outputs. Priorities are conditional on the completed source-val pilots, not claims of superiority.

| Priority | Matrix rows | Architecture / regime | Why run it |
| ---: | --- | --- | --- |
| 1 | sbi-hf-dino-srm; sbi-hf-clip-srm | Verified strong SRM backbones, SBI adaptation | Controlled cross-dataset hypothesis, preserving unadapted and simple-fusion baselines |
| 1 | sbi-generic-sbi; sbi-generic-mffi; sbi-generic-mixed | Generic MobileNet, matched budgets | Establish whether the negative tiny-source result persists with enough genuine faces |
| 2 | srm-pair-control; srm-pair-plus-new-expert | DINO-SRM + CLIP-SRM, with and without a certified new expert | Advance only if source-held-out complementarity warrants it |
| 2 | moe-srm-rgb-reconstruction-soft; top2 | Frozen three-role experts | Mandatory routing attribution against all simple controls |
| 3 | reconstruction-cae; vae; gated | 224 px real-only AE | Test bottleneck/discrepancy hypothesis at full genuine-data scale |
| 3 | reconstruction-residual-x_only; residual_only; frozen | Identical ResNet initialization | Separate RGB classification from reconstruction contribution |
| 3 | reconstruction-residual-recon_finetune; end_to_end | Explicit AE gradient semantics | Distinguish reconstruction adaptation from CE gradient training |
| 3 | reconstruction-latent; spatial-latent-mean | Latent MLP and fixed fusion | Complementarity test with a cheap control |
| 4 | vlm-qwen3-2b-nf4 | Frozen vision, text QLoRA | Exploratory independent expert; validate hardware fit first |
| 4 | metric-srm | SupCon and linear/MLP/centroid/kNN | Exploratory only after stronger controls |
| 4 | graph-srm-low-label | GNN and five same-mask controls | Downstream label-efficiency study; no local GNN promotion |

For each selected trained run, calibration is frozen from certified MFFI val, or val_select for learned fusion. One suite config then executes Test, paired Test-D, DF-40 and Celeb-DF v2; the matrix also lists four separate commands after the runbook's split-config preparation. The combined suite is preferred because it produces the paired Test/Test-D interval. DF-40 defaults to `pooled_all_real`, matching historical tables but confounding source domain. Switch to `source_matched` only after certifying annotations. DF-40 `cdf/frames` fakes overlap the Celeb-DF source domain. Celeb-DF's official 1=real labels are explicitly converted to internal 1=fake, and its primary score is mean frame `p_fake` per video across the official 518-video list.

The full-scale graph path uses bounded feature caches, FAISS neighborhood search with sampled recall checks, sparse neighborhoods and ego minibatches. It does not form a dense all-pairs graph. Inductive target queries cannot alter source neighborhoods or one another; transductive runs are separately named, calibrated and reported. Full-scale runtime and memory remain server measurements, not inferred from 1000-image pilots.

## Comparability with published results

Our supervised checkpoints and adaptation campaign use MFFI phase-1, while commonly cited cross-dataset tables train on FaceForensics++ and test on Celeb-DF v2.
Compression, frame sampling, face crops, source validation and initialization must match the comparator; a shared target name is insufficient.
[Shiohara and Yamasaki, CVPR 2022](https://openaccess.thecvf.com/content/CVPR2022/papers/Shiohara_Detecting_Deepfakes_With_Self-Blended_Images_CVPR_2022_paper.pdf), report 93.18% Celeb-DF AUC for SBI; the [authors' repository](https://github.com/mapooon/SelfBlendedImages#test) separately reports 92.87% for its released FF++ c23 checkpoint.
Those results cannot be directly ranked against this repository's MFFI-trained results, and our 24-technique DF-40 subset is not the official full DF-40 protocol.
A state-of-the-art claim requires a protocol-matched run, including the specified official splits and comparable video aggregation.
Server TODO: obtain FF++ c23, prepare train/validation crops and certify manifests with the existing converter, preserving source video groups and split independence.
The experimental CLI, SBI recipe, landmark cache and Celeb-DF video-mean suite already accept certified populations and can be reused.
FF++ acquisition, official crop/sampling reproduction, its source manifests and the official DF-40 split are not supplied or validated here.
Strict FF++-only comparisons also require suitable generic DINO/CLIP initialization support; the current HF-adaptation template carries prior MFFI fake exposure and cannot be relabeled FF++-only merely by replacing a manifest.
Treat that initializer extension and protocol verification as server preparation TODOs, and retain a separately labeled extra-data condition if MFFI-supervised initialization is used.
Run the priority-1 SBI DINO/CLIP campaign under MFFI for benchmark continuity and, if FF++ is available, under the matched FF++ to Celeb-DF protocol for literature comparability.
The independent MediaPipe SBI recipe is an experimental variant, so it must be compared against a faithfully reproduced published SBI control rather than described as that method's exact reproduction.

## Validation, failures and remaining limitations

The unmodified baseline passed 265 tests and 20 subtests. The final integrated suite passed 449 tests and 20 subtests in 70.56 seconds, a net addition of 184 test cases. Peak sampled RSS was 1653676 KiB, minimum available RAM 9169532 KiB, and maximum memory PSI full avg10 0.18%. Named-expert configuration and matched SBI metadata regressions are included. All 22 matrix configurations resolve, and all 154 documented train/calibrate/suite commands pass the real CLI parser. These preflights do not load full-server assets. All families have actual fit/calibrate/reload/four-target synthetic fixtures, including absolute DF-40 paths, paired degradation IDs and Celeb video aggregation. Evidence is in [preflight/final_pytest.json](preflight/final_pytest.json) and [preflight/matrix_validation.json](preflight/matrix_validation.json). The full suite includes every existing test; no pre-existing failure is waived. The final matrix audit covers 22 configurations and 154 commands, with no model fitting or target access.

Retained failures include watchdog-stopped preflight jobs, strict old-code bundle rejection, the NF4 subtree guard failure, same-process VLM reload RSS stops and an early two-expert router saturation caused by near-constant feature scaling. An incomplete final pytest attempt was stopped by the earlier 1% full-PSI guard; director message 016 judged that guard overly sensitive and authorized the successful retry with a 10% full-PSI limit, 2 GiB minimum available RAM and the unchanged 3072 MiB process cap. The incomplete attempt is retained in [preflight/final_pytest_stopped.json](preflight/final_pytest_stopped.json). Fixes and retries preserve the original evidence. No labels or score polarity were selected by AUC, no target threshold was fitted and no unreadable image was silently replaced.

Legacy cleanup candidates were documented rather than modified: automatic test evaluation from the old trainer; unreadable-image substitution; double class balancing; misleading finetune/finetune_robust paths; hardcoded thresholds and paths; incomplete historical provenance; old MoE scratch-pretraining assumptions; dense expert compute despite sparse routing. No paper, manuscript, prior results, tables or figures were changed. Full-scale CUDA execution, the four real benchmark populations and canonical multi-seed comparisons remain the server campaign.
