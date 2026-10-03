# Local preflight environment

The local ROCm environment is separate from the unchanged CUDA server pins. Exact measurements and package identities are in preflight/. The machine is Arch Linux, Ryzen 7 5700X (16 threads), 15 GiB RAM and Radeon gfx1200 with approximately 16 GiB VRAM. Initial disk availability was about 302 GiB. Available RAM varied between 8 and 10 GiB during setup; retain the conservative 3 GiB process limit.

```bash
uv venv --python 3.12 .venv
uv pip install --python .venv/bin/python torch==2.10.0 torchvision==0.25.0 --index-url https://download.pytorch.org/whl/rocm7.1
uv pip install --python .venv/bin/python -r requirements-research.txt plotly==6.6.0
uv pip install --python .venv/bin/python bitsandbytes==0.50.2 torch-geometric==2.8.0.post1 lpips==0.1.4 peft==0.21.2 accelerate==1.15.0 psutil==7.2.2
HIP_VISIBLE_DEVICES='' CUDA_VISIBLE_DEVICES='' OMP_NUM_THREADS=2 OPENBLAS_NUM_THREADS=2 MPLBACKEND=Agg HF_HUB_OFFLINE=1 .venv/bin/python -m pytest -q
```

Python is 3.12.13. PyTorch reports 2.10.0+rocm7.1 and HIP 7.1.25424; torchvision reports 0.25.0+rocm7.1. Do not run uv sync in this local environment because project pins intentionally select CUDA. No global packages or unrelated sessions were changed.

MediaPipe uses a separate Python 3.12 venv with mediapipe==1.0.1, pandas==3.0.2 and pillow==12.2.0, avoiding overlap between its opencv-contrib dependency and the research environment's headless OpenCV. The local face-landmarker asset SHA256 is 64184e229b263107bc2b804c6625db1341ff2bb731874b0bcc2fe6544e0bc9ff. Asset and landmarks stay outside git. CPU delegate only.

Every GPU command uses flock on the task's own GPU lock. Threads are limited to two, workers initially zero. The local watchdog terminates only its own process group at 3,072 MiB RSS, low available RAM, sustained memory pressure, or the specified time cap. Core GPU initialization reached 2.54 GiB transient RSS; the measured training probe stayed below 2.1 GiB. Local diagnostic scripts and full logs remain in the task-local staging area; portable production commands will be documented after implementation.

After the 2026-10-03 reboot the GPU moved to `/sys/class/drm/card0/device/`; read `mem_info_vram_used` and `gpu_busy_percent` there. No monitoring script hardcoded card1. Available RAM was13GiB and swap unused at resume. Keep the same process watchdog in case contention returns. Production SQLite landmark preprocessing detected202/202 source train reals in2.36seconds. Optional NF4/PyG/LPIPS backward probes all passed; small JSON evidence is in preflight/.

## Server environment and explicit asset staging

Use a separate research environment on the server, with a matched CUDA torch/torchvision installation appropriate to its driver. Preserve the repository's original pinned environments. Install the optional research additions with `python -m pip install -r requirements-research.txt -r requirements-experimental.txt`; this file is a direct-dependency set, not a complete transitive lock. Record `python -m pip freeze` for the actual campaign. Local ROCm success is not a claim that the server CUDA combination has been executed; first run a bounded training/reload smoke in the allocated CUDA environment.

All training paths load explicit assets offline. Set `TCC_PRETRAINED_ROOT`, `TCC_MODELS_ROOT` and `TCC_LPIPS_STATE` to absolute paths outside git. These staging commands may access the network; run them before the offline campaign. Never use a symlinked Hugging Face cache as the MFFI checkpoint tree, because the legacy rebuild resolves paths.

```bash
python - <<'PY'
import os
from pathlib import Path
import torch
from torchvision.models import MobileNet_V3_Large_Weights, ResNet18_Weights
import lpips

root = Path(os.environ["TCC_PRETRAINED_ROOT"])
root.mkdir(parents=True, exist_ok=True)
assets = {
    "mobilenet_v3_large-5c1a4163.pth": MobileNet_V3_Large_Weights.IMAGENET1K_V1,
    "resnet18-f37072fd.pth": ResNet18_Weights.IMAGENET1K_V1,
}
for name, weights in assets.items():
    output = root / name
    if output.exists():
        raise FileExistsError(output)
    torch.save(weights.get_state_dict(progress=True, check_hash=True), output)
lpips_output = Path(os.environ["TCC_LPIPS_STATE"])
if lpips_output.exists():
    raise FileExistsError(lpips_output)
lpips_output.parent.mkdir(parents=True, exist_ok=True)
network = lpips.LPIPS(net="alex", pnet_rand=False, pretrained=True).eval()
torch.save(network.state_dict(), lpips_output)
PY
```

The LPIPS export must contain both AlexNet features and learned linear weights, not only the small LPIPS calibration file. Training strictly validates the full state. The script preserves official tensor values but serializes a new file; record that file's SHA and use the same copy for all compared seeds. Set `TCC_RESNET18_WEIGHTS` to the staged ResNet file. MediaPipe preprocessing uses a separately installed environment and a staged `face_landmarker.task`; the asset SHA used locally is recorded above and in the landmark cache. A changed asset defines a new condition and needs fresh landmark QA.

Select only necessary MFFI checkpoint families, regimes and seeds in this staging call:

```bash
python - <<'PY'
import os
from huggingface_hub import snapshot_download

patterns = []
for family, mode, regime in [
    ("dino", "srm", "finetune_robust"),
    ("clip", "srm", "finetune_robust"),
    ("clip", "none", "finetune"),
]:
    for seed in [42, 123, 2024, 7, 2025]:
        prefix = f"{family}/{mode}/{regime}/seed_{seed}"
        patterns.extend([f"{prefix}/weights/best.pth", f"{prefix}/results/run_config.json"])
snapshot_download("lucasoc/MFFI-Models", revision="f2dee3c52c665053f06182307841f8cd68be3916",
                  local_dir=os.environ["TCC_MODELS_ROOT"], allow_patterns=patterns)
PY
```

The compact VLM bases use pinned snapshots: SmolVLM-256M-Instruct at `7e3e67edbbed1bf9888184d9df282b700a323964`, or Qwen3-VL-2B-Instruct at `89644892e4d85e24eaac8bacfd4f463576704203`. Stage the chosen snapshot with `snapshot_download(repo_id, revision=revision, local_dir=Path(TCC_PRETRAINED_ROOT)/model_name)`, including processor/tokenizer files. Set `HF_HUB_OFFLINE=1` and `TRANSFORMERS_OFFLINE=1` for execution. Artifact loaders verify base files, processor, quantization, adapter targets and scoring policy. An unavailable quantization backend must fail explicitly or use a separately named BF16 config; it never silently changes the requested method.

Complete portable commands are in [SERVER_RUNBOOK.md](SERVER_RUNBOOK.md). Data, checkpoints and feature caches remain outside git. Record device, package versions, source hashes, memory and runtime for every server run.
