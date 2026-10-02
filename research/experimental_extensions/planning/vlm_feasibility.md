# Compact VLM feasibility and contract, Experiment F

Gate 1 planning only, 2026-10-02. Read the full director brief, original request and deadline message 001. No model weights were downloaded, no environment was changed and no GPU operation was run. Hardware fit below is an estimate pending the lead's serial probes. Implementation follows Gate 1 review and the shared runner interface supplied by the lead.

## Decision and evidence

Use `HuggingFaceTB/SmolVLM-256M-Instruct` for the first actual min-dataset smoke, with frozen vision tower/connector and BF16 decoder LoRA. Preserve an explicit NF4 QLoRA configuration for the same model and CUDA servers. Only stage 500M or larger models after the smallest path passes. This is a feasibility and complementary-expert study, not an expected replacement for the strongest SRM detector.

[SmolVLM research](https://arxiv.org/abs/2504.05299) motivates the small architecture and compressed visual tokens; its inference memory claim does not establish training RSS. [VLFFD, CVPR 2025](https://arxiv.org/abs/2502.20698) supports investigating visual-language forgery supervision but also documents unreliable generated explanations. The local labels supply no region explanations, so train only `Real`/`Fake`, without invented rationales. Neither source establishes forensic performance of a 256M model. [LoRA](https://arxiv.org/abs/2106.09685) and [QLoRA](https://arxiv.org/abs/2305.14314) justify frozen-base adaptation and NF4 respectively; their LLM results are not VLM hardware measurements.

Exact stored parameter counts were read from each publisher's `/api/models/<repo>` `safetensors.total`, not inferred from names. BF16 weights-only GiB is `2 * total / 2**30`, excluding adapters, activations, runtime and host staging.

| Publisher checkpoint | Stored parameters | BF16 weights, GiB | Role |
| --- | ---: | ---: | --- |
| [SmolVLM-256M-Instruct](https://huggingface.co/api/models/HuggingFaceTB/SmolVLM-256M-Instruct) | 256,484,928 | 0.478 | Local smoke and optional short pilot |
| [SmolVLM-500M-Instruct](https://huggingface.co/api/models/HuggingFaceTB/SmolVLM-500M-Instruct) | 507,482,304 | 0.945 | Escalation only after measured fit |
| [InternVL3-1B-hf](https://huggingface.co/api/models/OpenGVLab/InternVL3-1B-hf) | 938,193,024 | 1.748 | Deferred alternative, native HF implementation |
| [Qwen3-VL-2B-Instruct](https://huggingface.co/api/models/Qwen/Qwen3-VL-2B-Instruct) | 2,127,532,032 | 3.963 | Preferred larger CUDA server comparison |
| [Qwen2.5-VL-3B-Instruct](https://huggingface.co/api/models/Qwen/Qwen2.5-VL-3B-Instruct) | 3,754,622,976 | 6.994 | Exceeds a strict total-parameter 3B cap |

Pin SmolVLM 256M to revision `7e3e67edbbed1bf9888184d9df282b700a323964` for staging and metadata. Its model type is `idefics3`, architecture `Idefics3ForConditionalGeneration`, processor `Idefics3Processor`, image sequence length 64, image patch size 16 and vision image size 512. SmolVLM2-256M-Video-Instruct has the same stored parameter count but F32 checkpoint tensors at the inspected revision. Image-only SmolVLM avoids that larger staging artifact and unnecessary video processing. Celeb-DF still uses the shared mean-of-frame-probabilities protocol.

## APIs and memory controls

The repository already pins Transformers 5.14.1 in `pyproject.toml:24` and `requirements-research.txt:12`. Keep existing pins. [Exact 5.14.1 auto-model source](https://github.com/huggingface/transformers/blob/v5.14.1/src/transformers/models/auto/modeling_auto.py) provides `AutoModelForImageTextToText`; `AutoModelForVision2Seq` is absent. Use `AutoProcessor` and `AutoModelForImageTextToText`, `trust_remote_code=False`, local safetensors and `local_files_only=True` after staging.

[Exact loader source](https://github.com/huggingface/transformers/blob/v5.14.1/src/transformers/modeling_utils.py) uses `dtype=`, keeps `torch_dtype` as a deprecated alias, initializes on the meta device and discards `low_cpu_mem_usage` as unused. Thus that flag cannot enforce the host RAM limit. Load directly with an explicit single-device map, for example `device_map={"": 0}` on the locked accelerator, and measure RSS throughout loading. Avoid automatic CPU/disk offload and a CPU load followed by whole-model `.to()`. Do not let the legacy evaluator relocate a dispatched quantized model.

Use SDPA first and eager attention for the offline structural fixture. FlashAttention is optional and must not be assumed because ROCm reports the PyTorch device name `cuda`. No AMD branching belongs in experiment code.

The [publisher processor files](https://huggingface.co/HuggingFaceTB/SmolVLM-256M-Instruct/tree/7e3e67edbbed1bf9888184d9df282b700a323964) default to a 2048 longest edge with image splitting, which can multiply patches. Bound the local processor to longest edge 512, max image size 512 and `do_image_splitting=False`; assert one image patch and 64 image tokens per image. Persist these overrides. Keep checkpoint image sequence length and patch geometry consistent. Use RGB bytes/PIL images and the processor's mean/std 0.5; do not apply the legacy ImageNet-normalized 224 tensor as processor input. Record any upstream 224 crop cache resizing as preprocessing provenance.

BF16 LoRA initial proposal: rank 8, alpha 16, dropout 0.05, bias none, decoder attention q/v projections, AdamW over trainable adapters only, batch 1, accumulation 8, `use_cache=False`, non-reentrant gradient checkpointing. [Idefics3 5.14.1 source](https://github.com/huggingface/transformers/blob/v5.14.1/src/transformers/models/idefics3/modeling_idefics3.py) exposes `model.text_model`, `model.vision_model`, `model.connector` and `lm_head`. Select full decoder module names or an anchored regex, then assert every adapted module is in the text decoder. Bare q/v suffix matching also reaches the vision tower. Freeze the base before adaptation and verify finite nonzero adapter gradients with no frozen-base gradients.

[PEFT 0.21.2 preparation source](https://github.com/huggingface/peft/blob/v0.21.2/src/peft/utils/other.py) upcasts all non-`Params4bit` FP16/BF16 parameters to FP32, including a skipped vision tower. Do not call `prepare_model_for_kbit_training` for the BF16 fallback. For QLoRA, inspect and record the resulting dtype distribution and budget those FP32 tensors. Configure NF4, double quantization and BF16 compute through `BitsAndBytesConfig`; deliberately skip the frozen vision tower/connector and verify that the intended decoder linears actually became 4-bit. Transformers 5.14.1's [4-bit quantizer](https://github.com/huggingface/transformers/blob/v5.14.1/src/transformers/quantizers/quantizer_bnb_4bit.py) uses `llm_int8_skip_modules` for this exclusion despite its historical name.

## ROCm NF4 and CUDA path

[Stable bitsandbytes 0.50.2 documentation](https://huggingface.co/docs/bitsandbytes/v0.50.2/en/installation) explicitly lists Linux ROCm 7.1.1 and gfx1200, plus NF4 support on RDNA. The exact [PyPI metadata](https://pypi.org/pypi/bitsandbytes/0.50.2/json) gives a 43,139,553-byte Linux x86-64 wheel. A 128 KiB HTTP range of its ZIP directory confirmed `libbitsandbytes_rocm71.so`, without downloading the wheel. This is distribution evidence, not local execution evidence.

Recommended lead-owned optional installation for the already installed torch 2.10 ROCm 7.1 venv:

```bash
uv pip install --python "$FFB_VENV/bin/python" --no-deps --only-binary=:all: --index-url https://pypi.org/simple bitsandbytes==0.50.2
```

Wheel SHA-256: `55348a9a4a21bfd99cf8c7b32fe67b4030ae5c2a05738e03c1747f65fa6ec283`. Its runtime requirements are torch >=2.4,<3, numpy and packaging; `--no-deps` protects the existing ROCm torch. Candidate optional pins from [PEFT metadata](https://pypi.org/pypi/peft/0.21.2/json) and [Accelerate metadata](https://pypi.org/pypi/accelerate/1.15.0/json) are `peft==0.21.2`, `accelerate==1.15.0` and their missing `psutil` dependency, subject to import/API tests with the pinned Transformers. Add them only to a new optional dependency file owned by the lead.

The lead's locked probe must exercise CUDA-device NF4 quantization, a small `Linear4bit` forward/backward and an actual LoRA parameter update with finite nonzero gradients. Check BF16 compute and checkpoint reload, not just successful import or GPU detection. [0.50.2 loader source](https://github.com/bitsandbytes-foundation/bitsandbytes/blob/0.50.2/bitsandbytes/cextension.py) selects ROCm from `torch.version.hip`; do not force a CUDA version or conceal backend warnings. A missing library/kernel or failed backward produces an explicit unsupported-quantization error. BF16 fallback is a separately named/configured run, with the failure recorded, never a silent substitution.

H100 CUDA 12.8 and RTX 3090 satisfy the documented NF4 compute-capability requirement. Use the same code/config schema and CUDA-device map, with a locked server capability probe first. Qwen3-VL-2B QLoRA is a later comparison with its own bounded processor and verified total parameter count, not a local prerequisite. No ROCm-only package index belongs in server configs.

## Exact score and answer-mask contract

Prompt: `Classify the face image as authentic or manipulated. Answer with exactly Real or Fake.` The image and prompt are identical for both candidate branches, without ground-truth labels or image filenames. Label 0 maps to `Real`; label 1 maps to `Fake`. Never select polarity from AUC.

Let P be the processor-expanded prompt ending at the assistant generation boundary. Let c be the complete token sequence for the candidate's label text, including the template-required leading space, but excluding assistant headers, closing markers and EOS. The inspected Smol chat template renders the generation boundary as `Assistant:` while a completed answer starts `Assistant: Real` or `Assistant: Fake`; therefore isolated tokenization of `Real` is not authoritative. Obtain the label span from the full rendered/processed candidate and assert its prefix equals the processed generation prompt. Store both actual token ID sequences. If prefix tokenization differs, fail and repair the declared boundary rather than guessing indices.

For each image and candidate y:

```text
S_y = sum over j of log_softmax(logits at the physical position
      immediately before c_y[j], float32)[c_y[j]]
s = S_Fake - S_Real
p_fake = exp(S_Fake - logsumexp(S_Real, S_Fake)) = sigmoid(s)
binary_logits = [S_Real, S_Fake]
```

Teacher forcing feeds the same prompt/image plus preceding candidate tokens. Sum all label-token log probabilities; do not average by length, score only the first token or read the vocabulary's raw `Fake` entry without checking tokenization. EOS is excluded from this scoring policy. The normalized score is conditional on the two chosen verbalizers, not proof of calibrated real-world authenticity. Use `eval()`/inference mode and no sampling. A shared one-token fast path is allowed only after exact span verification.

Training renders the true assistant label plus the model's end-of-utterance marker. Initialize labels to -100, supervise only the answer continuation and its closing marker, and mask the entire prompt, image tokens, headers and padding. The Smol template lacks generation blocks, so do not assume `return_assistant_tokens_mask` supplies a usable mask. Derive offsets after multimodal expansion and before padding. Assert at least one supervised label token per row and reject truncation of either visual or answer tokens.

Use right padding for teacher-forced SFT/scoring initially. Compute answer positions from per-row physical masks, not `logits[:, -1]` or a padded batch width. Attention masks, not token equality, identify padding: some models use the same ID for EOS and pad. Keep input IDs/masks integer when moving batches, and cast only floating pixel tensors. [Transformers causal-loss source](https://github.com/huggingface/transformers/blob/v5.14.1/src/transformers/loss/loss_utils.py) already shifts labels once; passing pre-shifted labels to model loss shifts twice. For selected-logit scoring, explicitly gather logits from t-1 for target token t. A hand-computed two-token example and mixed-length padded batch must agree with unbatched teacher forcing.

## Integration, rebuild and validation

Use a separate supervised-language trainer under the experimental runner. Stack A's `src/pipelines/training.py` expects tensor batches and two-class CE, so it cannot directly supervise answer tokens. Reuse its seed conventions and the shared run metadata, accumulation/AMP policy and output layout without forcing token batches through its classifier loop.

Provide a VLM predictor returning `[S_Real, S_Fake]` or `p_fake` for the lead's shared inference interface. Preserve sample IDs and image hashes. Reuse `src/robustness/artifacts.py:18` certificates, `inference.py:81` calibration and the shared four-target metrics. VLM input decoding/preprocessing must be explicit in provenance. Existing Stack B calls `.to(device)` in its classifier predictor, which needs dispatch-aware handling for quantized models. Calibration binds to the adapter/base/processor/scoring bundle, not just an adapter weight file.

Save adapters with `PeftModel.save_pretrained(..., safe_serialization=True)`, processor files, and a versioned rebuild JSON containing base repository/revision, base file hashes, model class, quantization/dtype, exact adapter targets, processor overrides, prompt/template hash, label token IDs, EOS policy, seed and dependency versions. Hash a canonical bundle manifest and verify referenced hashes when loading; use that immutable identity for calibration. Base weights stay outside git. Resolve portable base paths from `TCC_PRETRAINED_ROOT`; avoid persisting a workstation absolute path in `adapter_config.json`.

Rebuild the base with recorded kwargs, then `PeftModel.from_pretrained(base, adapter_dir, is_trainable=False)`. Reload the saved processor and validate prompt/label IDs before scoring. Never merge a quantized adapter implicitly. Save optimizer/scheduler/RNG/accumulation state separately if resume is exposed. Round-trip scores must agree in the same backend/dtype within a declared tolerance. Missing base/processor/optional package or quantization support fails with a useful error.

Offline structural fixture: a tiny real `Idefics3ForConditionalGeneration` configuration, tiny Llama decoder, RGB vision encoder, matching image tokens, actual PEFT q/v adapter, masked answer loss, backward/update and adapter save/rebuild. It exercises the genuine multimodal architecture without network access. It is randomly initialized and cannot validate pretrained fit or count as the required min-dataset smoke.

Smallest genuine offline pretrained smoke: after lead-approved staging, load pinned SmolVLM 256M and its complete processor locally, use 16 deterministic balanced min-train images and 16 min-val images, take four optimizer updates, score both candidates, save/rebuild and export certified predictions. Use no min-test for tuning. Run with offline HF settings so missing files fail before training. The same test must run on NF4 when the backend probe passes; otherwise run the explicit BF16 variant and retain a clear unsupported NF4 result.

## Time and rejection budget

Under message 001, deliver F implementation and CPU contract tests during the 17:15 to 19:15 integration window. Request one GPU slot for the genuine smoke, capped at 10 minutes including load/save/reload. An optional seed-42 pilot is capped at 20 further minutes and 32 optimizer updates on a deterministic 128-image training subset, with validation only; skip it if shared infrastructure or higher-priority work needs the slot. Stop new pilots by 20:45 and preserve the 21:30 code freeze. No local 2B training or multi-seed sweep.

Use one GPU lock, one CPU thread and workers=0 initially. Target process RSS <=2 GiB; stop this process at 2.5 GiB RSS or worsening memory pressure, before the 3 GiB hard limit. Stop on nonfinite loss/gradients, failed quantized backward, missing base-gradient isolation, failed rebuild, repeated allocation errors or exhausted stage wall time. Record peak RSS/VRAM and step time; weight arithmetic is not a fit result. Do not spend the deadline compiling custom ROCm kernels.

For a surviving pilot, require nondegenerate probabilities, changing image-conditioned scores, finite adapter updates and validation improvement over the frozen zero-shot VLM. Compare with existing SRM/RGB outputs and simple fusion on the lead's permitted validation split. If it is weak or highly redundant, retain the tested server-ready path and report the weak pilot instead of escalating compute. Small validation differences and one-seed outcomes cannot establish full-benchmark superiority; final Test/Test-D/DF-40/Celeb-DF scoring requires a frozen candidate and the shared audited evaluator.
