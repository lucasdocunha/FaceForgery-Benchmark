# Research basis and bounded scientific claims

Literature and repository review: 16 September 2026. The implementation is a research platform and controlled pilot, not evidence that a new method wins or is novel. The previous `research/explicability/RESEARCH.md` and bibliography are preserved. This document refocuses the next stage on robust generalization.

## Primary question

**When do spectral cues complement a strong pretrained RGB detector under degradation and generator/source shift, and can their contribution be used without worsening failures where they are unreliable?**

The experiment must allow a negative answer. A weak standalone spectral detector might repair useful RGB errors, but a larger fusion model might simply exploit extra capacity. Measure complementarity and appropriate controls before assigning a mechanism to a gain. A learned gate can collapse or exploit source shortcuts; neither its name nor its heatmap proves calibrated reliability.

## Sources that drive the design

### Evaluation and dataset structure

**DeepfakeBench — Yan et al., NeurIPS 2023 Datasets and Benchmarks.** Standardizes preprocessing, datasets and evaluation to address misleading comparisons. This motivates exact input/population contracts, a separate historical route, and matched protocols rather than transplanting leaderboard numbers across settings. We do not claim our implementation is a reproduction of all its detectors.
https://proceedings.neurips.cc/paper_files/paper/2023/hash/0e735e4b4f07de483cbe250130992726-Abstract-Datasets_and_Benchmarks.html

**DF40 — Yan et al., NeurIPS 2024 Datasets and Benchmarks.** Studies a broader forgery-method space and evaluation protocols. Its official repository identifies FF++ and Celeb-DF source domains for many generated samples. Therefore, a list of dataset names is not a proof of independent source distributions. Preserve generator labels and source domains and document possible identity/content overlap. Reversing train/test direction changes the question and is not directly comparable to a published opposite-direction result.
https://proceedings.nips.cc/paper_files/paper/2024/hash/34239f60eca7ce9bee5280aaf81362d8-Abstract-Datasets_and_Benchmarks_Track.html
https://github.com/YZY-stack/DF40

**Celeb-DF — Li et al., CVPR 2020; official repository.** This remains a useful dataset source even though its publication is older. The older preparation script's own stated official labels and folder names are cross-checked explicitly. Do not change class conventions in response to target AUC. The new crop procedure must be disclosed because frame extraction, detector, context and recompression influence the evaluated input.
https://github.com/yuezunli/celeb-deepfakeforensics

**MFFI — official project and paper.** Preserve its documented forgery/source structure and verify the exact local partition manifests. Source labels and generator metadata should be reviewed rather than inferred from face appearance. Existing MFFI values in older papers and current fine-tuned runs are different experiment collections.
https://github.com/inclusionConf/MFFI
https://arxiv.org/abs/2509.05592

### Frequency dependence and contemporary comparisons

**FreqDebias — Kashiani, Talemi and Afghah, CVPR 2025, pp. 8775–8785.** Explicitly identifies spectral bias and uses forgery mixup plus local/global consistency mechanisms. This is a central related work and a candidate matched-protocol comparator. Our fixed pixel-space corruption recipe and simple prediction-level Jensen–Shannon term are **not FreqDebias** and must not be described as a novel discovery of frequency debiasing.
https://openaccess.thecvf.com/content/CVPR2025/html/Kashiani_FreqDebias_Towards_Generalizable_Deepfake_Detection_via_Consistency-Driven_Frequency_Debiasing_CVPR_2025_paper.html

**Beyond Spatial Frequency — Kim et al., ICCV 2025, pp. 11198–11207.** Models pixel-wise temporal frequency in video. This illustrates that a frame-only model averaged across video frames and a temporal detector receive different information. Keep such comparisons in separately described settings rather than treating the published video result as a matched frame baseline. No temporal-frequency branch is implemented here.
https://openaccess.thecvf.com/content/ICCV2025/html/Kim_Beyond_Spatial_Frequency_Pixel-wise_Temporal_Frequency-based_Deepfake_Video_Detection_ICCV_2025_paper.html

**Wavelet-Driven Generalizable Framework — Baru et al., WACV Workshops 2025, pp. 1661–1669.** A further spatial/frequency-related candidate using wavelets with CLIP features. Record the workshop status accurately. It is not a reproduced result or an asserted best baseline here. Reproduction feasibility, released code/weights, data licenses and protocol matching must be verified before allocating compute.
https://openaccess.thecvf.com/content/WACV2025W/MAPA/html/Baru_Wavelet-Driven_Generalizable_Framework_for_Deepfake_Face_Forgery_Detection_WACVW_2025_paper.html

A two-branch RGB/frequency encoder, late fusion, confidence-like gate, or standard consistency penalty is not sufficient novelty by itself. The defensible contribution must be a specific advance or reproducible insight under controlled distribution shifts. Do not use citation counts, a recent preprint title or unverified leaderboard claims to establish novelty. Additional 2026 preprints discussed informally in the meeting must be independently verified and positioned before being cited as accepted research.

### Explainability and efficiency

**Integrated Gradients — Sundararajan, Taly and Yan, ICML 2017.** Motivates a fixed scalar output, declared baseline and completeness diagnostics. IG does not establish a causal mechanism or baseline invariance. The retained XAI pipeline's fixed fake-minus-real margin and explicit numerical records follow these distinctions.
https://proceedings.mlr.press/v70/sundararajan17a.html

**Sanity Checks for Saliency Maps — Adebayo et al., NeurIPS 2018.** Shows why visually plausible maps alone are insufficient. Keep parameter-dependence checks and distinguish full-network randomization from the retained head-only control. Class-agnostic attention is not expected to behave like a class-specific attribution.
https://papers.nips.cc/paper/2018/hash/294a8ed24b1ad22ec2e7efea049b8737-Abstract.html

**Distilling the Knowledge in a Neural Network — Hinton, Vinyals and Dean, 2015 preprint / NIPS 2014 Deep Learning Workshop.** Establishes the teacher–student efficiency motivation. The optional temperature-scaled KL objective here is a standard baseline, not new distillation research. A compression claim additionally needs a measured comparable student/teacher trade-off under the target shifts.
https://arxiv.org/abs/1503.02531

## Design decisions

The primary training implementation uses paired image-space clean/strong views, deriving frequency representations only after the same augmentation. A channel-concatenation baseline, late mean-logit fusion, adaptive residual fusion and equal-parameter auxiliary spatial control isolate different sources of apparent gain. Plain paired CE and RGB-only consistency separate the extra forward pass and consistency objective from spectral information. The small spectral-only baseline is not claimed to match the large pretrained RGB model's capacity.

The prototype residual is `z = z_RGB + sigmoid(g(features)) × z_aux`. Auxiliary branch dropout preserves an available RGB path during training. Gate outputs are descriptive internal variables; they are not probabilities that an expert is correct. The implementation does not enforce a proven invariant, certify robustness, or guarantee a novel accepted-paper contribution.

Use source validation for model/threshold selection and label all development on already observed Test-D as exploratory. Confirmatory settings need a frozen selection rule, independent source/identity review, matched seeds, predefined primary metrics, and transparent group coverage. Paired cluster-bootstrap intervals quantify sampling variability conditional on the provided grouping and fixed checkpoints; across-seed SD is reported separately. Multiple exploratory subgroups must not be selectively promoted as independent confirmatory findings.

## Resource priorities

Stop the broad scratch grid unless a small matched control is scientifically necessary. First audit labels and existing run artifacts, then evaluate complementarity using saved scores. Run a limited ResNet-based pilot before replicating a promising finding on another backbone and the full seed set. A failed or null pilot should stop expansion rather than trigger arbitrary tuning against the target test set. Estimate actual runtime from a measured pilot, not from parameter count or the size of the YAML matrix. Respect the institution's GPU allocations and scheduler.

The route to a stronger paper is a focused contribution with reliable evidence, not a guaranteed venue outcome. Venue scope and extension policies should be checked again at submission; there is no universal rule that a numerical percentage of changed text guarantees an acceptable journal extension.
