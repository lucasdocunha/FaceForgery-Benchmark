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
