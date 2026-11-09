import sys
import time
from pathlib import Path
from huggingface_hub import HfApi

REPO_ID = "lucasoc/MFFI-Models"
MODELS_ROOT = Path("/media/ssd2/lucas.ocunha/models-tcc")
PROJECT_ROOT = Path("/home/lucas.ocunha/tcc")

MODELS = ["resnet", "xception", "clip", "dino", "vit"]

IGNORE_PATTERNS = [
    "**/final.pth",
    "**/*.npz",
    "**/predictions.csv",
    "**/predictions_test.csv",
    "**/predictions_test_d.csv",
    "**/predictions_val.csv",
]

def main():
    api = HfApi()
    print(f"=== Sincronizando com Hugging Face: {REPO_ID} ===")
    
    # 1. Upload dos modelos SRM restantes
    for model in MODELS:
        local_path = MODELS_ROOT / model / "srm" / "finetune_robust"
        repo_path = f"{model}/srm/finetune_robust"
        
        if not local_path.exists():
            print(f"[-] Diretório não encontrado: {local_path}, pulando...")
            continue
            
        print(f"\n[+] Enviando {model} SRM ({local_path} -> {repo_path})...")
        t0 = time.time()
        try:
            info = api.upload_folder(
                folder_path=str(local_path),
                path_in_repo=repo_path,
                repo_id=REPO_ID,
                repo_type="model",
                ignore_patterns=IGNORE_PATTERNS,
                commit_message=f"feat(models): upload {model} SRM robust weights, metrics and plots (5 seeds)",
            )
            print(f"    Concluído em {time.time() - t0:.1f}s: {info}")
        except Exception as e:
            print(f"    [ERRO] Falha ao enviar {model}: {e}")

    # 2. Upload das Tabelas Consolidadas
    tables_to_upload = [
        "tables/tabela9_srm_5seeds.csv",
        "tables/srm_5seeds_summary.csv",
        "tables/df40_srm_breakdown_by_paradigm.csv",
        "tables/df40_srm_breakdown_by_technique.csv",
        "tables/mega_ensembles_srm_all_models.csv",
        "tables/ensembles_srm_5seeds_summary.csv",
    ]
    print("\n[+] Enviando tabelas consolidadas...")
    for rel_path in tables_to_upload:
        full_path = PROJECT_ROOT / rel_path
        if full_path.exists():
            try:
                api.upload_file(
                    path_or_fileobj=str(full_path),
                    path_in_repo=rel_path,
                    repo_id=REPO_ID,
                    repo_type="model",
                    commit_message=f"docs(tables): add {Path(rel_path).name} to repo",
                )
                print(f"    Tabela enviada: {rel_path}")
            except Exception as e:
                print(f"    [ERRO] Falha ao enviar tabela {rel_path}: {e}")

    # 3. Upload das Figuras de Heatmap
    figures_to_upload = [
        "figures/heatmap_clip_dino_fusion_3rows.png",
        "figures/heatmap_clip_dino_fusion_4rows.png",
    ]
    print("\n[+] Enviando figuras de atenção multimodal...")
    for rel_path in figures_to_upload:
        full_path = PROJECT_ROOT / rel_path
        if full_path.exists():
            try:
                api.upload_file(
                    path_or_fileobj=str(full_path),
                    path_in_repo=rel_path,
                    repo_id=REPO_ID,
                    repo_type="model",
                    commit_message=f"feat(figures): add {Path(rel_path).name} attention rollouts",
                )
                print(f"    Figura enviada: {rel_path}")
            except Exception as e:
                print(f"    [ERRO] Falha ao enviar figura {rel_path}: {e}")

    # 4. Atualizar README.md no Hugging Face
    print("\n[+] Atualizando README.md com SRM e resultados do benchmark...")
    updated_readme = """---
license: mit
tags:
- deepfake-detection
- face-forgery
- fourier-transform
- steganalysis
- srm
- computer-vision
- pytorch
---

# MFFI: Model Checkpoints & Benchmarks for Face Forgery Detection

Public weights, representations, and experiment artifacts for the project:
> **Benchmarking Spatial, Spectral, and Self-Supervised Cues for Face Forgery Detection under Realistic Degradation**  
> 🔗 Official Codebase: [github.com/lucasdocunha/tcc](https://github.com/lucasdocunha/tcc)

---

## 📌 Overview

This repository hosts official pre-trained and fine-tuned model checkpoints (`best.pth`), evaluation metrics, and experiment configurations evaluated across **FaceForensics++ (clean and test_d)**, **Celeb-DF v2**, and the **DF-40 (DeepFake-40)** benchmarks.

### Included Model Families:
- **CLIP-ViT-B/16** (Multimodal Vision Transformer, OpenAI)
- **DINO (ConvNeXt-Base)** (Self-supervised distillation via DINOv3 LVD-1689M)
- **ViT-B/16** (Vision Transformer, ImageNet-21k)
- **Xception** (Depthwise Separable CNN)
- **ResNet-18** (Residual CNN)
- **MobileNetV3-Large** (Efficient Mobile CNN with Squeeze-and-Excitation)
- **Mixture of Experts (MoE)** (Standard and Frequency-specialized MoE architectures)

### Feature Representations & Frequency Modes:
- `none`: Spatial domain RGB (3 channels)
- `srm`: **Steganalysis Rich Models (SRM)** high-pass residual filters (3 channels: horizontal, vertical, diagonal edge residuals)
- `magnitude`: FFT Log-Magnitude $\\log(|F| + 1)$ (1 channel)
- `phase`: FFT Phase angle $[0, 1]$ (1 channel)
- `complex`: Real and Imaginary components (2 channels)
- `concat`: Spatial RGB concatenated with FFT Magnitude (4 channels)
- `frequency_3`: High-pass filtered magnitude (1 channel)
- `concat_frequency`: Multi-domain RGB + Magnitude + Phase + HighPass + LowPass (7 channels)

---

## 🏆 Benchmark Highlights: SRM & Top Ensembles

### 1. Top Ensemble (CLIP-SRM + DINO-SRM)
Evaluating across 5 statistical seeds (`42`, `123`, `2024`, `7`, `2025`):
- **FF++ Test Limpo:** **94.17% AUC** (Mega 10x) / **93.07% ± 0.61%** (Ensemble 2x)
- **FF++ Test-D (Difícil):** **88.60% AUC** (Mega 10x) / **87.24% ± 0.49%** (Ensemble 2x)
- **DF-40 (40 Generative Methods):** **84.15% AUC** (Mega 10x) / **83.53% ± 1.74%** (Ensemble 2x)
- **Celeb-DF v2:** **75.06% AUC** (Mega 10x) / **74.11% ± 3.14%** (Ensemble 2x)

### 2. DF-40 Breakdown by Paradigm:
- **Editing (T2I & Inversion):** 97.47% Recall
- **Face Swap:** 93.30% Recall
- **Diffusion (SD 2.1, PixArt, DiT):** 93.32% Recall (SD 2.1: 100%, PixArt: 99.6%)
- **GANs (StyleGAN 2/3, VQGAN):** 90.96% Recall (StyleGAN 2/3: 100%)
- **Commercial Avatars & Talking Heads:** 68.53% AUC (HeyGen: 33.2% recall)

---

## 📂 Repository Structure

```text
├── <model_family>/
│   └── srm/
│       └── finetune_robust/
│           └── seed_<seed>/
│               ├── weights/best.pth       # Best checkpoint saved by validation ROC-AUC
│               ├── results/metrics_*.csv  # Full metrics across test, test_d, df40, celeb_df
│               ├── results/run_config.json# Hyperparameters and seed configuration
│               └── plots/*.png            # Confusion matrix and ROC-AUC plots
├── tables/                                # Consolidated CSV benchmark tables
│   ├── tabela9_srm_5seeds.csv
│   ├── df40_srm_breakdown_by_paradigm.csv
│   └── df40_srm_breakdown_by_technique.csv
└── figures/                               # Multimodal attention rollout maps
    └── heatmap_clip_dino_fusion_4rows.png
```

---

## 🚀 How to Download & Load Weights

```python
import torch
from huggingface_hub import hf_hub_download

# Download best.pth for CLIP (SRM mode, finetune_robust, seed 42)
checkpoint_path = hf_hub_download(
    repo_id="lucasoc/MFFI-Models",
    filename="clip/srm/finetune_robust/seed_42/weights/best.pth"
)

# Load state dict
state_dict = torch.load(checkpoint_path, map_location="cpu", weights_only=True)
print(f"Loaded {len(state_dict)} tensors successfully!")
```

---

## 📄 Codebase & Citation

For reproduction scripts, training configurations, and data preparation pipelines:
- **Repository:** [https://github.com/lucasdocunha/tcc](https://github.com/lucasdocunha/tcc)
- **License:** MIT License
"""
    try:
        api.upload_file(
            path_or_fileobj=updated_readme.encode("utf-8"),
            path_in_repo="README.md",
            repo_id=REPO_ID,
            repo_type="model",
            commit_message="docs(readme): update README with SRM representation, DF-40 breakdown and top ensembles",
        )
        print("    README.md atualizado com sucesso no Hugging Face!")
    except Exception as e:
        print(f"    [ERRO] Falha ao atualizar README.md: {e}")

    print("\n=== Sincronização com Hugging Face Finalizada com Sucesso! ===")

if __name__ == "__main__":
    main()
