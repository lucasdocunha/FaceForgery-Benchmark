"""
Script to generate high-resolution publication-quality heatmaps and attention rollouts
for CLIP (SRM), DINO (SRM), and their Multimodal Fusion across a curated cohort of real
and deepfake faces.
"""
from __future__ import annotations

import sys
from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.cm as cm
from matplotlib.colors import Normalize
from mpl_toolkits.axes_grid1 import make_axes_locatable
from PIL import Image
import torch
import torch.nn.functional as F

from src.pipelines.checkpoints import run_from_checkpoint, load_model_from_run
from src.data.data import encode_pil_image
from src.plots.heatmap import attention_rollout, grad_cam, overlay

device = 'cuda:0' if torch.cuda.is_available() else 'cpu'

# 1. Load checkpoints (SRM finetune_robust, seed 123)
clip_ckpt = Path('/media/ssd2/lucas.ocunha/models-tcc/clip/srm/finetune_robust/seed_123/weights/best.pth')
dino_ckpt = Path('/media/ssd2/lucas.ocunha/models-tcc/dino/srm/finetune_robust/seed_123/weights/best.pth')

print(f"Loading CLIP SRM checkpoint: {clip_ckpt}")
m_clip = load_model_from_run(run_from_checkpoint(clip_ckpt), device)

print(f"Loading DINO SRM checkpoint: {dino_ckpt}")
m_dino = load_model_from_run(run_from_checkpoint(dino_ckpt), device)

# 2. Curated Cohort of 6 representative images
ff = pd.read_csv('data/raw/test.csv')
ff.columns = ff.columns.str.strip()
real_ff = Path('/media/ssd2/lucas.ocunha/datasets/phase1/testset') / ff[ff['target'] == 0].iloc[10]['img_name']
real_celeba = Path('/media/ssd2/lucas.ocunha/datasets/df40_extracted/styleclip/real/105352.jpg')
fake_ff = Path('/media/ssd2/lucas.ocunha/datasets/phase1/testset') / ff[ff['target'] == 1].iloc[42]['img_name']

df40 = pd.read_csv('data/df40/test.csv')
fake_uniface = Path([r['img_name'] for _, r in df40[df40['paradigm']=='face_swap'].iterrows() if '262.png' in r['img_name']][0])
fake_cdf = Path('/media/ssd2/lucas.ocunha/datasets/celeb_df_crops/id1_id9_0007_f00020.jpg')
fake_t2i = Path([r['img_name'] for _, r in df40[df40['target']==1].iterrows() if '422.png' in r['img_name']][0])

cohort = [
    {
        "name": "Real Face (FF++)",
        "category": "Pristine Facial Frame",
        "path": real_ff,
        "target": 0
    },
    {
        "name": "Real Face (CelebA)",
        "category": "Pristine Portrait",
        "path": real_celeba,
        "target": 0
    },
    {
        "name": "DeepFake (FF++)",
        "category": "Autoencoder Face Swap",
        "path": fake_ff,
        "target": 1
    },
    {
        "name": "FaceSwap (DF-40)",
        "category": "UniFace High-Res Swap",
        "path": fake_uniface,
        "target": 1
    },
    {
        "name": "DeepFake (Celeb-DF v2)",
        "category": "Advanced Cross-Domain",
        "path": fake_cdf,
        "target": 1
    },
    {
        "name": "Generative (DF-40)",
        "category": "WhichFaceIsReal (T2I/GAN)",
        "path": fake_t2i,
        "target": 1
    }
]

def min_max_norm(t: torch.Tensor) -> torch.Tensor:
    t_min = t.amin(dim=(-2, -1), keepdim=True)
    t_max = t.amax(dim=(-2, -1), keepdim=True)
    return (t - t_min) / (t_max - t_min).clamp_min(1e-8)

processed_items = []
for item in cohort:
    p = item["path"]
    target = item["target"]
    assert p.exists(), f"Image path does not exist: {p}"
    
    pil_img = Image.open(p).convert('RGB')
    pil_resized = pil_img.resize((224, 224), Image.Resampling.BILINEAR)
    img_tensor = torch.from_numpy(np.array(pil_resized)).float().permute(2, 0, 1) / 255.0  # [3, 224, 224]
    
    # 6-channel SRM input: RGB + 3 SRM noise residuals
    x = encode_pil_image(pil_img, 'srm', 224).unsqueeze(0).to(device)
    
    with torch.no_grad():
        p_clip = torch.softmax(m_clip(x), dim=1)[0, 1].item()
        p_dino = torch.softmax(m_dino(x), dim=1)[0, 1].item()
        p_fus = (p_clip + p_dino) / 2.0
    
    # 1. CLIP: Attention Rollout (self-attention propagation across ViT-B/16 layers)
    clip_roll = attention_rollout(m_clip, x).cpu() # [1, 1, 224, 224]
    
    # 2. DINO: Grad-CAM (convolutional feature maps from ConvNeXt-B stage 3)
    dino_cam = grad_cam(m_dino, x, target_class=1).cpu() # [1, 1, 224, 224]
    
    # 3. Fusion: Geometric and arithmetic multimodal consensus
    # Arithmetic mean captures full union of spatial + frequency artifacts
    fus_map = min_max_norm(0.5 * (clip_roll + dino_cam))
    
    # Blended Overlays with underlying face (alpha=0.52 for optimal visibility)
    disp = img_tensor.unsqueeze(0)
    over_clip = overlay(disp, clip_roll, alpha=0.52)[0].permute(1, 2, 0).numpy()
    over_dino = overlay(disp, dino_cam, alpha=0.52)[0].permute(1, 2, 0).numpy()
    over_fus = overlay(disp, fus_map, alpha=0.52)[0].permute(1, 2, 0).numpy()
    orig_np = img_tensor.permute(1, 2, 0).numpy()
    
    processed_items.append({
        "item": item,
        "orig": orig_np,
        "clip": over_clip,
        "dino": over_dino,
        "fus": over_fus,
        "p_clip": p_clip,
        "p_dino": p_dino,
        "p_fus": p_fus,
        "fus_map": fus_map.squeeze().detach().numpy()
    })
    print(f"Processed: {item['name']:25s} | Target={target} | CLIP={p_clip:.1%} | DINO={p_dino:.1%} | Fusion={p_fus:.1%}")

# -------------------------------------------------------------
# FIGURE A: Exactly 3 rows as explicitly requested by user:
# 1st row: CLIP ViT-B/16 (SRM) Attention Rollout
# 2nd row: DINO ConvNeXt-B (SRM) Grad-CAM
# 3rd row: Fusão Multimodal (CLIP + DINO)
# -------------------------------------------------------------
n_cols = len(processed_items)
fig3, axes3 = plt.subplots(3, n_cols, figsize=(3.2 * n_cols + 1.2, 10.2), dpi=300)

row_headers_3 = [
    "CLIP ViT-B/16 (SRM)\nAttention Rollout",
    "DINO ConvNeXt-B (SRM)\nGrad-CAM",
    "Fusão Multimodal\n(CLIP + DINO SRM)"
]

for col_idx, data in enumerate(processed_items):
    meta = data["item"]
    
    # Row 1: CLIP
    ax_clip = axes3[0, col_idx]
    ax_clip.imshow(data["clip"])
    target_str = "Real (y=0)" if meta["target"] == 0 else "Fake (y=1)"
    col_title = f"{meta['name']}\n[{target_str}]\nP(Fake) = {data['p_clip']:.1%}"
    ax_clip.set_title(col_title, fontsize=10.5, fontweight='bold', pad=8)
    ax_clip.axis('off')
    
    # Row 2: DINO
    ax_dino = axes3[1, col_idx]
    ax_dino.imshow(data["dino"])
    ax_dino.set_title(f"P(Fake) = {data['p_dino']:.1%}", fontsize=10.5, pad=6)
    ax_dino.axis('off')
    
    # Row 3: Fusion
    ax_fus = axes3[2, col_idx]
    ax_fus.imshow(data["fus"])
    ax_fus.set_title(f"P_ens = {data['p_fus']:.1%}", fontsize=10.5, fontweight='bold', pad=6,
                     color="darkred" if data['p_fus'] > 0.5 else "darkgreen")
    ax_fus.axis('off')

for row_idx, title in enumerate(row_headers_3):
    axes3[row_idx, 0].text(-0.18, 0.5, title, transform=axes3[row_idx, 0].transAxes,
                           fontsize=11.5, fontweight='bold', va='center', ha='right', linespacing=1.2)

# Add colorbar on the right
cbar_ax = fig3.add_axes([0.94, 0.15, 0.015, 0.70])
norm = Normalize(vmin=0.0, vmax=1.0)
sm = cm.ScalarMappable(cmap="jet", norm=norm)
sm.set_array([])
cbar = fig3.colorbar(sm, cax=cbar_ax)
cbar.set_label("Intensidade da Ativação / Atenção Relativa (0.0 = Mínima, 1.0 = Máxima)", fontsize=10, labelpad=10)
cbar.ax.tick_params(labelsize=9)

plt.subplots_adjust(left=0.17, right=0.92, top=0.91, bottom=0.05, wspace=0.08, hspace=0.18)

out_fig3_a = Path('figures/heatmap_clip_dino_fusion_3rows.png')
out_fig3_b = Path('results/mostrar_rayson/heatmap_clip_dino_fusion_3rows.png')
out_fig3_a.parent.mkdir(parents=True, exist_ok=True)
out_fig3_b.parent.mkdir(parents=True, exist_ok=True)
fig3.savefig(out_fig3_a, bbox_inches='tight')
fig3.savefig(out_fig3_b, bbox_inches='tight')
plt.close(fig3)
print(f"Saved: {out_fig3_a} & {out_fig3_b}")


# -------------------------------------------------------------
# FIGURE B: Complete 4 rows for academic presentation:
# Row 1: Original Face (RGB)
# Row 2: CLIP ViT-B/16 (SRM) Attention Rollout
# Row 3: DINO ConvNeXt-B (SRM) Grad-CAM
# Row 4: Fusão Multimodal (CLIP + DINO SRM)
# -------------------------------------------------------------
fig4, axes4 = plt.subplots(4, n_cols, figsize=(3.2 * n_cols + 1.2, 13.0), dpi=300)

row_headers_4 = [
    "Face Original\n(RGB Entrada)",
    "CLIP ViT-B/16 (SRM)\nAttention Rollout",
    "DINO ConvNeXt-B (SRM)\nGrad-CAM",
    "Fusão Multimodal\n(CLIP + DINO SRM)"
]

for col_idx, data in enumerate(processed_items):
    meta = data["item"]
    
    # Row 1: Original Face
    ax_orig = axes4[0, col_idx]
    ax_orig.imshow(data["orig"])
    target_str = "Real (y=0)" if meta["target"] == 0 else "Fake (y=1)"
    ax_orig.set_title(f"{meta['name']}\n[{target_str}]", fontsize=10.5, fontweight='bold', pad=8)
    ax_orig.axis('off')
    
    # Row 2: CLIP
    ax_clip = axes4[1, col_idx]
    ax_clip.imshow(data["clip"])
    ax_clip.set_title(f"P(Fake) = {data['p_clip']:.1%}", fontsize=10.5, pad=6)
    ax_clip.axis('off')
    
    # Row 3: DINO
    ax_dino = axes4[2, col_idx]
    ax_dino.imshow(data["dino"])
    ax_dino.set_title(f"P(Fake) = {data['p_dino']:.1%}", fontsize=10.5, pad=6)
    ax_dino.axis('off')
    
    # Row 4: Fusion
    ax_fus = axes4[3, col_idx]
    ax_fus.imshow(data["fus"])
    ax_fus.set_title(f"P_ens = {data['p_fus']:.1%}", fontsize=10.5, fontweight='bold', pad=6,
                     color="darkred" if data['p_fus'] > 0.5 else "darkgreen")
    ax_fus.axis('off')

for row_idx, title in enumerate(row_headers_4):
    axes4[row_idx, 0].text(-0.18, 0.5, title, transform=axes4[row_idx, 0].transAxes,
                           fontsize=11.5, fontweight='bold', va='center', ha='right', linespacing=1.2)

# Add colorbar on the right
cbar_ax4 = fig4.add_axes([0.94, 0.15, 0.015, 0.70])
cbar4 = fig4.colorbar(sm, cax=cbar_ax4)
cbar4.set_label("Intensidade da Ativação / Atenção Relativa (0.0 = Mínima, 1.0 = Máxima)", fontsize=10, labelpad=10)
cbar4.ax.tick_params(labelsize=9)

plt.subplots_adjust(left=0.17, right=0.92, top=0.91, bottom=0.05, wspace=0.08, hspace=0.18)

out_fig4_a = Path('figures/heatmap_clip_dino_fusion_4rows.png')
out_fig4_b = Path('results/mostrar_rayson/heatmap_clip_dino_fusion_4rows.png')
fig4.savefig(out_fig4_a, bbox_inches='tight')
fig4.savefig(out_fig4_b, bbox_inches='tight')
plt.close(fig4)
print(f"Saved: {out_fig4_a} & {out_fig4_b}")

print("All heatmaps and figures successfully generated and saved!")
