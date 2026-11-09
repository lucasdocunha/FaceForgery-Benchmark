"""Gera o painel mestre de explicabilidade (XAI) para todos os 6 modelos robustos na GPU.

Modelos avaliados:
1. ResNet-18 (Grad-CAM + IG)
2. MobileNetV3 (Grad-CAM + IG)
3. Xception (Grad-CAM + IG)
4. ViT-B/16 (IG)
5. CLIP ViT-B/16 (IG)
6. DINO ConvNeXt-B (IG)

Amostras:
- Real 1: YouTube-real (00208_f00028.jpg)
- Real 2: Celeb-real (00011_f00088.jpg)
- Fake 1: Celeb-synthesis Face Swap (id1_id0_0007_f00020.jpg)
- Fake 2: Celeb-synthesis Face Swap 2 (id0_id16_0003_f00066.jpg)
"""

import os
import io
from pathlib import Path
import torch
from torch import nn
import numpy as np
from PIL import Image
import matplotlib.pyplot as plt
from torchvision import transforms

from src.data.paths import models_root
from src.pipelines.checkpoints import discover_trained_runs, load_model_from_run, config_from_run
from src.robustness.explain import attribute, overlay_bytes


def main():
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    print(f"--- Iniciando Pipeline Completo de Explicabilidade (XAI) na GPU: {device} ---", flush=True)

    out_dir = Path("results/xai_heatmaps")
    out_dir.mkdir(parents=True, exist_ok=True)

    crops_dir = Path("/media/ssd2/lucas.ocunha/datasets/celeb_df_crops")
    samples = {
        "real_1": {"path": crops_dir / "00208_f00028.jpg", "label": "Real 1 (YouTube-real)", "true_cls": 0},
        "real_2": {"path": crops_dir / "00011_f00088.jpg", "label": "Real 2 (Celeb-real)", "true_cls": 0},
        "fake_1": {"path": crops_dir / "id1_id0_0007_f00020.jpg", "label": "Fake 1 (Celeb-synthesis)", "true_cls": 1},
        "fake_2": {"path": crops_dir / "id0_id16_0003_f00066.jpg", "label": "Fake 2 (Celeb-synthesis)", "true_cls": 1},
    }

    # Descoberta dos 6 modelos robustos
    runs = discover_trained_runs(models_root())
    
    def get_run(family):
        cands = [r for r in runs if r.model_family == family and ("robust" in str(r.run_dir) or r.seed == 987)]
        if not cands:
            raise RuntimeError(f"Nenhum run encontrado para {family}")
        return cands[0]

    model_names = ["resnet", "mobilenet", "xception", "vit", "clip", "dino"]
    model_labels = {
        "resnet": "ResNet-18",
        "mobilenet": "MobileNetV3",
        "xception": "Xception",
        "vit": "ViT-B/16",
        "clip": "CLIP ViT",
        "dino": "DINO ConvNeXt",
    }
    
    cnn_layers = {
        "resnet": "layer4.1.conv2",
        "mobilenet": "features.16.0",
        "xception": "backbone.conv4.pointwise",
    }

    models = {}
    print("\n[Passo 1/4] Carregando todos os 6 modelos robustos na GPU...", flush=True)
    for m in model_names:
        r = get_run(m)
        models[m] = load_model_from_run(r, device).eval()
        print(f" [✓] {model_labels[m]} carregado com sucesso!", flush=True)

    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
    ])

    images_store = {}
    predictions_store = {}

    print("\n[Passo 2/4] Executando Inferência e Atribuições XAI...", flush=True)
    for s_key, s_info in samples.items():
        print(f"\n>> Processando Amostra: {s_info['label']} ...", flush=True)
        img_raw = Image.open(s_info["path"]).convert("RGB").resize((224, 224))
        raw_tensor = transforms.ToTensor()(img_raw)
        x_norm = transform(img_raw).unsqueeze(0).to(device)

        # Salva original
        orig_path = out_dir / f"{s_key}_orig.png"
        img_raw.save(orig_path)
        images_store[f"{s_key}_orig"] = img_raw

        # Predições
        predictions_store[s_key] = {}
        for m in model_names:
            with torch.no_grad():
                logits = models[m](x_norm)
                prob_fake = torch.softmax(logits, dim=-1)[0, 1].item()
                margin = (logits[0, 1] - logits[0, 0]).item()
                pred_label = "Fake" if prob_fake >= 0.5 else "Real"
                predictions_store[s_key][m] = {
                    "p_fake": prob_fake,
                    "margin": margin,
                    "pred": pred_label,
                    "correct": (prob_fake >= 0.5) == (s_info["true_cls"] == 1)
                }
            status_symbol = "✓" if predictions_store[s_key][m]["correct"] else "✗"
            print(f"   {model_labels[m]:14s}: P(Fake) = {prob_fake:.4f} [{pred_label}] {status_symbol}")

        # Integrated Gradients para os 6 modelos
        for m in model_names:
            attr_ig, _ = attribute(
                models[m], x_norm, "integrated_gradients",
                baseline=torch.zeros_like(x_norm), steps=64
            )
            ig_png = overlay_bytes(raw_tensor, attr_ig[0])
            path = out_dir / f"{s_key}_{m}_ig.png"
            with open(path, "wb") as f:
                f.write(ig_png)
            images_store[f"{s_key}_{m}_ig"] = Image.open(io.BytesIO(ig_png))

        # Grad-CAM para CNNs
        for m, layer in cnn_layers.items():
            attr_cam, _ = attribute(
                models[m], x_norm, "gradcam",
                baseline=torch.zeros_like(x_norm), layer=layer
            )
            cam_png = overlay_bytes(raw_tensor, attr_cam[0])
            path = out_dir / f"{s_key}_{m}_gradcam.png"
            with open(path, "wb") as f:
                f.write(cam_png)
            images_store[f"{s_key}_{m}_gradcam"] = Image.open(io.BytesIO(cam_png))

    # [Passo 3/4] Gerando Grid 1: Comparativo Mestre dos 6 Modelos com Integrated Gradients
    print("\n[Passo 3/4] Renderizando Grid Mestre 6 Modelos (Integrated Gradients)...", flush=True)
    fig, axes = plt.subplots(4, 7, figsize=(26, 16))
    
    col_headers = [
        "Amostra Original",
        "ResNet-18 (IG)",
        "MobileNetV3 (IG)",
        "Xception (IG)",
        "ViT-B/16 (IG)",
        "CLIP ViT (IG)",
        "DINO ConvNeXt (IG)"
    ]
    for c, h in enumerate(col_headers):
        axes[0, c].set_title(h, fontsize=13, fontweight="bold", pad=12)

    sample_rows = [
        ("real_1", "Real 1 (YouTube-real)", 0),
        ("real_2", "Real 2 (Celeb-real)", 1),
        ("fake_1", "DeepFake 1 (Celeb-synth)", 2),
        ("fake_2", "DeepFake 2 (Celeb-synth)", 3),
    ]

    for s_key, s_title, r_idx in sample_rows:
        # Coluna 0: Original
        ax0 = axes[r_idx, 0]
        ax0.imshow(images_store[f"{s_key}_orig"])
        ax0.set_ylabel(s_title, fontsize=12, fontweight="bold", labelpad=10)
        ax0.set_xticks([])
        ax0.set_yticks([])

        # Colunas 1 a 6: Modelos
        for c_idx, m in enumerate(model_names, start=1):
            ax = axes[r_idx, c_idx]
            img = images_store[f"{s_key}_{m}_ig"]
            ax.imshow(img)
            p_info = predictions_store[s_key][m]
            status = "ACERTO" if p_info["correct"] else "ERRO"
            color = "#007700" if p_info["correct"] else "#cc0000"
            ax.set_xlabel(f"P(Fake): {p_info['p_fake']:.1%}\n[{status}]", fontsize=10, fontweight="bold", color=color)
            ax.set_xticks([])
            ax.set_yticks([])

    plt.suptitle("Explicabilidade Forense de DeepFakes: Atribuição por Integrated Gradients nos 6 Modelos Robustos",
                 fontsize=16, fontweight="bold", y=0.995)
    plt.tight_layout()
    grid1_path = out_dir / "xai_master_6models_ig_grid.png"
    plt.savefig(grid1_path, dpi=300, bbox_inches="tight")
    plt.close()
    print(f" [✓] Painel Mestre salvo em: {grid1_path}", flush=True)

    # [Passo 4/4] Gerando Grid 2: CNNs - Grad-CAM vs Integrated Gradients
    print("\n[Passo 4/4] Renderizando Grid CNNs (Grad-CAM vs Integrated Gradients)...", flush=True)
    fig, axes = plt.subplots(4, 7, figsize=(26, 16))
    
    col_headers_cnn = [
        "Amostra Original",
        "ResNet-18 (Grad-CAM)",
        "ResNet-18 (IG)",
        "MobileNetV3 (Grad-CAM)",
        "MobileNetV3 (IG)",
        "Xception (Grad-CAM)",
        "Xception (IG)"
    ]
    for c, h in enumerate(col_headers_cnn):
        axes[0, c].set_title(h, fontsize=13, fontweight="bold", pad=12)

    for s_key, s_title, r_idx in sample_rows:
        ax0 = axes[r_idx, 0]
        ax0.imshow(images_store[f"{s_key}_orig"])
        ax0.set_ylabel(s_title, fontsize=12, fontweight="bold", labelpad=10)
        ax0.set_xticks([])
        ax0.set_yticks([])

        cnn_pairs = [
            ("resnet", f"{s_key}_resnet_gradcam", f"{s_key}_resnet_ig"),
            ("mobilenet", f"{s_key}_mobilenet_gradcam", f"{s_key}_mobilenet_ig"),
            ("xception", f"{s_key}_xception_gradcam", f"{s_key}_xception_ig"),
        ]

        col_pos = 1
        for m, cam_k, ig_k in cnn_pairs:
            # Grad-CAM
            ax_cam = axes[r_idx, col_pos]
            ax_cam.imshow(images_store[cam_k])
            ax_cam.set_xticks([])
            ax_cam.set_yticks([])
            ax_cam.set_xlabel("Ativação Semântica", fontsize=10)
            col_pos += 1

            # IG
            ax_ig = axes[r_idx, col_pos]
            ax_ig.imshow(images_store[ig_k])
            ax_ig.set_xticks([])
            ax_ig.set_yticks([])
            p_info = predictions_store[s_key][m]
            color = "#007700" if p_info["correct"] else "#cc0000"
            ax_ig.set_xlabel(f"P(Fk): {p_info['p_fake']:.1%}", fontsize=10, fontweight="bold", color=color)
            col_pos += 1

    plt.suptitle("Comparativo de Explicabilidade em CNNs: Grad-CAM (Ativação de Camada) vs Integrated Gradients (Atribuição de Pixel)",
                 fontsize=16, fontweight="bold", y=0.995)
    plt.tight_layout()
    grid2_path = out_dir / "xai_cnns_gradcam_vs_ig_grid.png"
    plt.savefig(grid2_path, dpi=300, bbox_inches="tight")
    plt.close()
    print(f" [✓] Painel CNNs salvo em: {grid2_path}", flush=True)

    print("\n=======================================================")
    print(" [✓] TODOS OS PAINÉIS E HEATMAPS FORAM GERADOS COM SUCESSO!")
    print(f"  - Painel 1 (6 Modelos IG): {grid1_path}")
    print(f"  - Painel 2 (CNNs Grad-CAM vs IG): {grid2_path}")
    print("=======================================================", flush=True)


if __name__ == "__main__":
    main()
