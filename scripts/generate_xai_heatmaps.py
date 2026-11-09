"""Gera mapas de calor de explicabilidade (XAI) reais em GPU para os modelos campeões."""

import os
from pathlib import Path
import torch
import numpy as np
from PIL import Image
import matplotlib.pyplot as plt
from torchvision import transforms

from src.data.paths import models_root
from src.pipelines.checkpoints import discover_trained_runs, load_model_from_run, config_from_run
from src.robustness.explain import attribute, overlay_bytes


def main():
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    print(f"Executando pipeline XAI no dispositivo: {device}", flush=True)

    out_dir = Path("results/xai_heatmaps")
    out_dir.mkdir(parents=True, exist_ok=True)

    crops_dir = Path("/media/ssd2/lucas.ocunha/datasets/celeb_df_crops")
    samples = {
        "real": {"path": crops_dir / "00208_f00028.jpg", "label": "Real (YouTube-real)"},
        "fake": {"path": crops_dir / "id1_id0_0007_f00020.jpg", "label": "Fake (Celeb-synthesis)"},
    }

    # Descoberta dos runs robustos
    runs = discover_trained_runs(models_root())
    
    def get_run(family):
        cands = [r for r in runs if r.model_family == family and ("robust" in str(r.run_dir) or r.seed == 987)]
        return cands[0]

    model_names = ["resnet", "dino", "clip"]
    models = {}
    configs = {}
    print("\n--- 1. Carregando modelos campeões na GPU ---", flush=True)
    for m in model_names:
        r = get_run(m)
        configs[m] = config_from_run(r)
        models[m] = load_model_from_run(r, device).eval()
        print(f" [✓] {m.upper()} carregado com sucesso!", flush=True)

    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
    ])

    results_grid = {}

    print("\n--- 2. Executando Atribuição XAI (Grad-CAM e Integrated Gradients) ---", flush=True)
    for sample_key, sample_info in samples.items():
        print(f"\nProcessando amostra {sample_key.upper()} ({sample_info['label']})...", flush=True)
        img_raw = Image.open(sample_info["path"]).convert("RGB").resize((224, 224))
        raw_tensor = transforms.ToTensor()(img_raw)  # [3, 224, 224] in [0, 1]
        x_norm = transform(img_raw).unsqueeze(0).to(device)  # [1, 3, 224, 224]

        # Salva original
        img_raw.save(out_dir / f"{sample_key}_original.png")
        results_grid[f"{sample_key}_original"] = img_raw

        # Predições
        print("  Logits e Probabilidades:")
        for m in model_names:
            with torch.no_grad():
                logits = models[m](x_norm)
                prob_fake = torch.softmax(logits, dim=-1)[0, 1].item()
                margin = (logits[0, 1] - logits[0, 0]).item()
            print(f"    {m.upper():8s} -> P(Fake) = {prob_fake:.4f} | Margin = {margin:+.4f}")

        # ResNet Grad-CAM
        print("  -> Calculando ResNet Grad-CAM (camada layer4.1.conv2)...", flush=True)
        attr_cam, _ = attribute(
            models["resnet"], x_norm, "gradcam",
            baseline=torch.zeros_like(x_norm), layer="layer4.1.conv2"
        )
        cam_png = overlay_bytes(raw_tensor, attr_cam[0])
        with open(out_dir / f"{sample_key}_resnet_gradcam.png", "wb") as f:
            f.write(cam_png)
        results_grid[f"{sample_key}_resnet_gradcam"] = Image.open(out_dir / f"{sample_key}_resnet_gradcam.png")

        # ResNet Integrated Gradients
        print("  -> Calculando ResNet Integrated Gradients (64 passos)...", flush=True)
        attr_res_ig, _ = attribute(
            models["resnet"], x_norm, "integrated_gradients",
            baseline=torch.zeros_like(x_norm), steps=64
        )
        res_ig_png = overlay_bytes(raw_tensor, attr_res_ig[0])
        with open(out_dir / f"{sample_key}_resnet_ig.png", "wb") as f:
            f.write(res_ig_png)
        results_grid[f"{sample_key}_resnet_ig"] = Image.open(out_dir / f"{sample_key}_resnet_ig.png")

        # DINO Integrated Gradients
        print("  -> Calculando DINO Integrated Gradients (64 passos)...", flush=True)
        attr_dino_ig, _ = attribute(
            models["dino"], x_norm, "integrated_gradients",
            baseline=torch.zeros_like(x_norm), steps=64
        )
        dino_ig_png = overlay_bytes(raw_tensor, attr_dino_ig[0])
        with open(out_dir / f"{sample_key}_dino_ig.png", "wb") as f:
            f.write(dino_ig_png)
        results_grid[f"{sample_key}_dino_ig"] = Image.open(out_dir / f"{sample_key}_dino_ig.png")

        # CLIP Integrated Gradients
        print("  -> Calculando CLIP Integrated Gradients (64 passos)...", flush=True)
        attr_clip_ig, _ = attribute(
            models["clip"], x_norm, "integrated_gradients",
            baseline=torch.zeros_like(x_norm), steps=64
        )
        clip_ig_png = overlay_bytes(raw_tensor, attr_clip_ig[0])
        with open(out_dir / f"{sample_key}_clip_ig.png", "wb") as f:
            f.write(clip_ig_png)
        results_grid[f"{sample_key}_clip_ig"] = Image.open(out_dir / f"{sample_key}_clip_ig.png")

    print("\n--- 3. Montando Painel Comparativo Final (xai_comparison_grid.png) ---", flush=True)
    fig, axes = plt.subplots(2, 5, figsize=(20, 8.5))
    col_titles = [
        "Imagem Original",
        "ResNet-18 (Grad-CAM)",
        "ResNet-18 (Integrated Gradients)",
        "DINO ConvNeXt (Integrated Gradients)",
        "CLIP ViT (Integrated Gradients)",
    ]

    for col, title in enumerate(col_titles):
        axes[0, col].set_title(title, fontsize=12, fontweight="bold", pad=10)

    rows = [
        ("real", "Face Real\n(Celeb-DF YouTube-real)", 0),
        ("fake", "Face Manipulada\n(Celeb-synthesis)", 1),
    ]

    keys = [
        ["{r}_original", "{r}_resnet_gradcam", "{r}_resnet_ig", "{r}_dino_ig", "{r}_clip_ig"]
    ]

    for r_key, r_label, row_idx in rows:
        axes[row_idx, 0].set_ylabel(r_label, fontsize=12, fontweight="bold", labelpad=10)
        row_keys = [
            f"{r_key}_original",
            f"{r_key}_resnet_gradcam",
            f"{r_key}_resnet_ig",
            f"{r_key}_dino_ig",
            f"{r_key}_clip_ig",
        ]
        for col_idx, k in enumerate(row_keys):
            ax = axes[row_idx, col_idx]
            img = results_grid[k]
            ax.imshow(img)
            ax.set_xticks([])
            ax.set_yticks([])

    plt.tight_layout()
    grid_path = out_dir / "xai_comparison_grid.png"
    plt.savefig(grid_path, dpi=300, bbox_inches="tight")
    plt.close()
    print(f" [✓] Painel comparativo salvo com sucesso em: {grid_path}", flush=True)


if __name__ == "__main__":
    main()
