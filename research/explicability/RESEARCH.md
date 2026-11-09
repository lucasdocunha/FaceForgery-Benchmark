# Research basis and evidence boundaries

Research cutoff: 11 September 2026. This is an extension and reproducibility audit, not a claim that changing prose turns the earlier accepted paper into a new publication.

## What the released evidence establishes

The starting point is `ICLR` commit `1643db7a82f111f1a468dba986fe29593761b6be`. The immutable Hugging Face release is `lucasoc/MFFI-Models@3bef179cdc08850e1d720182e55a040d00c9d582`. The paper at that commit describes predominantly scratch-trained models and a frozen DINOv3 backbone; the released collection is predominantly fine-tuned and uses a ConvNeXt DINOv3 backbone. They are different experiments. Never transplant the earlier paper's numbers into the new release's tables or interpret their differences as a controlled pretraining ablation.

The completed reanalysis covers 54 RGB prediction files: six families, three seeds (42, 123, 2024), and validation/clean-test/degraded-test splits. Raw ROC-AUC values reproduce the corresponding released per-run metrics to floating-point precision (maximum absolute difference 1.11e-16). This checks exported predictions and arithmetic, not a rerun of image inference.

Every one of the 18 RGB validation exports has 147,363 rows but duplicates legacy ID 131900 and omits 131899. Both copies of 131900 have identical labels and predictions within each export. The explicitly named derived validation view collapses only identical duplicates, preserves the raw files, records the missing ID without imputing it, and contains 147,362 images. All 36 test exports have 181,947 unique IDs and agree with the supplied test manifest's labels. Legacy numeric-ID-to-filename linkage remains an explicit original-manifest-order assumption until image inference is verified.

The decision rule for this reanalysis maximizes balanced accuracy on the derived validation view, with a deterministic smallest-threshold tie break. It is then frozen for both test splits. The primary explanation roster is the six RGB seed-42 checkpoints, not an unspecified set of every possible architecture. The three-seed/18-checkpoint error intersection is a separate sensitivity analysis.

The official image download required authentication and returned HTTP 401 during this audit. Consequently, full-cohort attribution heatmaps, causal failure explanations, demographic analyses, and image-level reproduction of the exported predictions are **not completed evidence**. Passing technical tests or loading a checkpoint does not remove that boundary.

## Protocol decisions and supporting research

### 1. Shared samples, fixed target, and accountable explanations

A single set of 64 images cannot generally contain 16 examples in every confusion stratum for every model simultaneously. Define the strata using a declared reference checkpoint (`resnet.none.finetune.seed42`) and freeze one seed-42 cohort with 16 TP, FN, TN, and FP. Keep each other model's own confusion outcome in the manifest. This is a diagnostic case-control sample, not an unbiased estimator of accuracy, failure prevalence, or explanation quality over the population.

Explain the fixed fake-versus-real logit margin, not a different argmax class at each interpolation point or perturbation. Save signed raw attributions, preprocessing details, feature partitions, baselines, numerical residuals, RNG settings, target layers, hashes, and execution status. Pretty heatmaps are not an explanation-quality metric.

- Sundararajan, Taly, and Yan. **Axiomatic Attribution for Deep Networks.** ICML 2017, PMLR 70:3319–3328. Integrated Gradients and completeness motivate the fixed-target path and numerical residual checks. https://proceedings.mlr.press/v70/sundararajan17a.html
- Selvaraju et al. **Grad-CAM: Visual Explanations from Deep Networks via Gradient-Based Localization.** ICCV 2017. Class-specific gradients, declared layers, and actual backward paths are required; attention maps are not interchangeable with Grad-CAM. https://openaccess.thecvf.com/content_iccv_2017/html/Selvaraju_Grad-CAM_Visual_Explanations_ICCV_2017_paper.html
- Lundberg and Lee. **A Unified Approach to Interpreting Model Predictions.** NeurIPS 2017. Use an actual SHAP estimator rather than relabeling interpolation or an arbitrary linear regression as SHAP. https://papers.nips.cc/paper/7062-a-unified-approach-to-interpreting-model-predictions
- Ribeiro, Singh, and Guestrin. **“Why Should I Trust You?” Explaining the Predictions of Any Classifier.** KDD 2016, pp. 1135–1144. LIME is a local surrogate with a specified perturbation distribution and regularization, not an exact causal explanation. https://doi.org/10.1145/2939672.2939778
- Abnar and Zuidema. **Quantifying Attention Flow in Transformers.** ACL 2020, pp. 4190–4197. Rollout is retained as a separately labeled, class-agnostic attention diagnostic; it must not silently replace a class-specific method. https://aclanthology.org/2020.acl-main.385/
- Adebayo et al. **Sanity Checks for Saliency Maps.** NeurIPS 2018. Motivates parameter-randomization controls and rejection of visual appeal as sufficient validation. https://arxiv.org/abs/1810.03292
- Hooker, Erhan, Kindermans, and Kim. **A Benchmark for Interpretability Methods in Deep Neural Networks.** NeurIPS 2019. Perturbing features changes the data distribution; deletion/insertion curves without retraining are diagnostics, not a substitute for ROAR or a causal guarantee. https://arxiv.org/abs/1806.10758
- Hedström et al. **Quantus: An Explainable AI Toolkit for Responsible Evaluation of Neural Network Explanations and Beyond.** JMLR 24(34):1–11, 2023. Supports evaluating several distinct explanation properties rather than counting methods or selecting visually attractive maps. https://jmlr.org/papers/v24/22-0142.html

### 2. Native spatial and frequency coordinates

A Fourier coefficient is indexed by frequency, not by the location of an eye, nose, or mouth. Overlay an RGB model's input attribution on RGB, and a frequency model's attribution on the exact frequency representation it consumed. Do not plot a frequency-input Grad-CAM on an RGB face and infer facial localization. Hybrid inputs must preserve separate channel groups.

Select a matched RGB/pure-frequency architecture pair using validation evidence only. Require complete paired seeds and a prespecified minimum AUC in both domains; maximize the mean worst-domain validation AUC and use degradation only as a tie break. Otherwise a nearly random model can appear 'robust' simply because its performance has little room to fall. The reported selection uses released validation aggregates; it is exploratory within this release, not a preregistration made before the original experiments.

- Kashiani, Talemi, and Afghah. **FreqDebias: Towards Generalizable Deepfake Detection via Consistency-Driven Frequency Debiasing.** CVPR 2025, pp. 8775–8785. Frequency reliance can itself be a shortcut; frequency-domain activation is not inherently evidence of better generalization. https://openaccess.thecvf.com/content/CVPR2025/html/Kashiani_FreqDebias_Towards_Generalizable_Deepfake_Detection_via_Consistency-Driven_Frequency_Debiasing_CVPR_2025_paper.html
- Zheng, Deng, and Zhang. **Towards Attributions of Input Variables in a Coalition.** ICML 2025, PMLR 267:78115–78138. Coalition attribution depends on the feature partition and is not generally the sum of separately attributed variables. Save patch/band group coefficients and do not call their display-density maps pixel Shapley values. https://proceedings.mlr.press/v267/zheng25d.html
- Zaher, Trzaskowski, Nguyen, and Roosta. **Manifold Integrated Gradients: Riemannian Geometry for Feature Attribution.** ICML 2024, PMLR 235:58090–58104. Highlights the limitations of straight-line off-manifold attribution paths. The implementation uses conventional IG and explicitly does not claim manifold alignment. https://proceedings.mlr.press/v235/zaher24a.html

### 3. Recent advances that constrain, rather than inflate, our claims

- Ali, Raza, Gan, and Khan. **FocusViT: Faithful Explanations for Vision Transformers via Gradient-Guided Layer-Skipping.** AISTATS 2026, PMLR 300:1522–1530. Combines class-specific gradients and validation-based layer aggregation. It is a relevant extension beyond plain rollout, but is **not implemented or evaluated here**; selecting explanation layers using the held-out test cohort would be inappropriate. https://proceedings.mlr.press/v300/ali26a.html
- Anani, Lorenz, Fritz, and Schiele. **Pixel-level Certified Explanations via Randomized Smoothing.** ICML 2025, PMLR 267:1505–1533. Distinguishes empirical smoothing from formal robustness certification. Our SmoothGrad adapter is **not a certified explanation method**. https://proceedings.mlr.press/v267/anani25a.html

The choice to retain well-established methods alongside recent research is intentional: recent work informs limitations and evaluation design; it does not make an untested newly added method scientifically superior. Do not claim 'maximum possible explainability' or that twelve methods constitute twelve independent confirmations.

### 4. Shared-error investigation

Intersect errors only after validating complete coverage, unique identifiers, label agreement, and a frozen roster. Missing predictions must raise an error rather than become failures or silently disappear in an inner join. The empty intersection is a valid result. Use all shared-error samples for the declared roster and retain label-matched controls. A sample that fools all six fixed checkpoints is not a universal adversarial example across all architectures, seeds, datasets, or future systems.

Lighting, occlusion, compression, source identity, manipulation type, and skin-tone hypotheses require appropriate annotations and controlled comparisons. This project does not infer race, ethnicity, or skin-tone categories from unannotated face images. Label-only matching does not remove nuisance-variable confounding. Human review should be blinded to model attribution maps when annotating nuisance conditions, include inter-rater agreement, and distinguish association from cause.

## Implementation sources

Captum 0.9.0 documentation was used to check estimator APIs, baseline semantics, feature masks, and sample budgets:
- https://captum.ai/api/integrated_gradients.html
- https://captum.ai/api/gradient_shap.html
- https://captum.ai/api/kernel_shap.html
- https://captum.ai/api/lime.html
- https://captum.ai/api/shapley_value_sampling.html

## Publication and access checks

- Earlier manuscript: Cunha et al., **Benchmarking Spatial, Spectral, and Self-Supervised Cues for Face Forgery Detection under Realistic Degradation**, arXiv:2609.01511, 2026. Its PDF states acceptance at SIBGRAPI 2026. Cite it as prior work and retain its provenance. https://arxiv.org/abs/2609.01511
- MFFI: Miao et al., **MFFI: Multi-Dimensional Face Forgery Image Dataset for Real-World Scenarios**, ACM Multimedia 2025; arXiv:2509.05592. Official dataset access: https://modelscope.cn/datasets/DDLteam/MFFI
- ICLR 2027 author instructions, checked 11 September 2026: nine-page initial main-text limit, double-blind submission, required AI-use statement, and prohibition on substantially similar previously published/accepted work. https://iclr.cc/Conferences/2027/AuthorGuidelines
- AI-use disclosure must accurately describe research, programming, and writing assistance and retain human author responsibility. The authors have not yet signed off on this revision. https://iclr.cc/Conferences/2027/AIPolicyForAuthors

**Submission gate:** source-image access, image-level prediction verification, full Task 1/2/3 execution and controls, review of the earlier publication's overlap, and author approval remain required. No acceptance, submission eligibility, or completed visual finding is asserted by this branch.
