from pathlib import Path
import numpy as np
import pandas as pd
from sklearn.metrics import roc_curve

manifest = pd.read_csv("data/df40/test.csv")
tech_order = [
    ("Difusão", "sd2.1", "SD 2.1"),
    ("Difusão", "pixart", "PixArt-alpha"),
    ("Difusão", "SiT", "SiT"),
    ("Difusão", "DiT", "DiT"),
    ("Difusão", "RDDM", "RDDM"),
    ("Difusão", "CollabDiff", "CollabDiff"),
    ("Difusão", "ddim", "DDIM"),
    ("Síntese GAN", "StyleGAN3", "StyleGAN3"),
    ("Síntese GAN", "StyleGAN2", "StyleGAN2"),
    ("Síntese GAN", "StyleGANXL", "StyleGAN-XL"),
    ("Síntese GAN", "VQGAN", "VQGAN"),
    ("Síntese GAN", "stargan", "StarGAN"),
    ("Síntese GAN", "starganv2", "StarGAN v2"),
    ("Troca Facial", "uniface", "UniFace"),
    ("Troca Facial", "faceswap", "FaceSwap"),
    ("Troca Facial", "blendface", "BlendFace"),
    ("Troca Facial", "mobileswap", "MobileSwap"),
    ("Troca Facial", "deepfacelab", "DeepFaceLab"),
    ("Edição / T2I", "whichfaceisreal", "WhichFaceIsReal"),
    ("Edição / T2I", "MidJourney", "MidJourney"),
    ("Edição / T2I", "e4e", "e4e"),
    ("Edição / T2I", "styleclip", "StyleCLIP"),
    ("Misto", "cdf_reenactment", "Celeb-DF (misto)"),
    ("Avatar", "heygen", "HeyGen")
]

# Run Top-3 Revocacao ensemble
from scripts.test_df40_top3_ensemble import seed_results

# Load Top-3 Hibrido Val Champion
p_hib = [
    Path("/media/ssd2/lucas.ocunha/models-tcc/dino/srm/finetune_robust/seed_7"),
    Path("/media/ssd2/lucas.ocunha/models-tcc/dino/concat/finetune/seed_42"),
    Path("/media/ssd2/lucas.ocunha/models-tcc/clip/srm/finetune_robust/seed_2024"),
]
val_p_hib = [np.load(p / "results" / "outputs_val.npz")["probs"][:, 1] if np.load(p / "results" / "outputs_val.npz")["probs"].ndim > 1 else np.load(p / "results" / "outputs_val.npz")["probs"].ravel() for p in p_hib]
df40_p_hib = [np.load(p / "results" / "outputs_df40.npz")["probs"] for p in p_hib]
val_y = np.load(p_hib[0] / "results" / "outputs_val.npz")["y_true"]
fused_val_hib = (val_p_hib[0] * val_p_hib[1] * val_p_hib[2]) ** (1.0 / 3.0)
fpr, tpr, ths = roc_curve(val_y, fused_val_hib)
th_hib = ths[np.argmax(tpr - fpr)]
fused_df40_hib = (df40_p_hib[0] * df40_p_hib[1] * df40_p_hib[2]) ** (1.0 / 3.0)
pred_hib = (fused_df40_hib >= th_hib).astype(int)

# Load Top-3 Campeao Geral Val (CLIP Robusto + CLIP Concat + ResNet Concat)
p_camp = [
    Path("/media/ssd2/lucas.ocunha/models-tcc/clip/none/finetune_robust/seed_987"),
    Path("/media/ssd2/lucas.ocunha/models-tcc/clip/concat/finetune/seed_2025"),
    Path("/media/ssd2/lucas.ocunha/models-tcc/resnet/concat/finetune/seed_123"),
]
val_p_camp = [np.load(p / "results" / "outputs_val.npz")["probs"][:, 1] if np.load(p / "results" / "outputs_val.npz")["probs"].ndim > 1 else np.load(p / "results" / "outputs_val.npz")["probs"].ravel() for p in p_camp]
df40_p_camp = [np.load(p / "results" / "outputs_df40.npz")["probs"] for p in p_camp]
fused_val_camp = (val_p_camp[0] * val_p_camp[1] * val_p_camp[2]) ** (1.0 / 3.0)
fpr, tpr, ths = roc_curve(val_y, fused_val_camp)
th_camp = ths[np.argmax(tpr - fpr)]
fused_df40_camp = (df40_p_camp[0] * df40_p_camp[1] * df40_p_camp[2]) ** (1.0 / 3.0)
pred_camp = (fused_df40_camp >= th_camp).astype(int)

print(f"{'Paradigma':<12} | {'Tecnica':<16} | {'DINOv3 Freq':>11} | {'Xcep SRM':>8} | {'Xcep RGB':>8} | {'Ens Top-3':>9} | {'Top-3 Hibrido':>13} | {'Top-3 Campeao':>13}")
print("-" * 115)

for parad, tech, label in tech_order:
    idx = (manifest["target"] == 1) & (manifest["method"] == tech)
    sub_idx = np.where(idx)[0]
    
    r_d = [seed_results[s]["dino_pred"][sub_idx].mean() * 100 for s in range(3)]
    r_xso = [seed_results[s]["xso_pred"][sub_idx].mean() * 100 for s in range(3)]
    r_xs = [seed_results[s]["xs_pred"][sub_idx].mean() * 100 for s in range(3)]
    r_ens_top3 = [seed_results[s]["pred_soft_cal"][sub_idx].mean() * 100 for s in range(3)]
    r_hib = pred_hib[sub_idx].mean() * 100
    r_camp = pred_camp[sub_idx].mean() * 100
    
    print(f"{parad:<12} | {label:<16} | {np.mean(r_d):>10.1f}% | {np.mean(r_xso):>7.1f}% | {np.mean(r_xs):>7.1f}% | {np.mean(r_ens_top3):>8.1f}% | {r_hib:>12.1f}% | {r_camp:>12.1f}%")

fakes_idx = np.where(manifest["target"] == 1)[0]
tot_d = np.mean([seed_results[s]["dino_pred"][fakes_idx].mean() * 100 for s in range(3)])
tot_xso = np.mean([seed_results[s]["xso_pred"][fakes_idx].mean() * 100 for s in range(3)])
tot_xs = np.mean([seed_results[s]["xs_pred"][fakes_idx].mean() * 100 for s in range(3)])
tot_ens_top3 = np.mean([seed_results[s]["pred_soft_cal"][fakes_idx].mean() * 100 for s in range(3)])
tot_hib = pred_hib[fakes_idx].mean() * 100
tot_camp = pred_camp[fakes_idx].mean() * 100

print("-" * 115)
print(f"{'TOTAL GERAL':<12} | {'6.147 Fakes':<16} | {tot_d:>10.1f}% | {tot_xso:>7.1f}% | {tot_xs:>7.1f}% | {tot_ens_top3:>8.1f}% | {tot_hib:>12.1f}% | {tot_camp:>12.1f}%")
