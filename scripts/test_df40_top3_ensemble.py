from pathlib import Path
import numpy as np
import pandas as pd
from sklearn.metrics import roc_curve

manifest = pd.read_csv("data/df40/test.csv")
base = Path("/media/ssd2/lucas.ocunha/models-tcc")

seeds = [42, 123, 2024]
models_info = [
    ("dino_freq", "dino/concat_frequency/finetune"),
    ("xcep_srm_only", "xception/srm_only/finetune_robust"),
    ("xcep_srm", "xception/srm/finetune_robust"),
]

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

seed_results = []
for s in seeds:
    val_probs_dict = {}
    df40_probs_dict = {}
    val_y = None
    
    for short, rel in models_info:
        v_npz = np.load(base / rel / f"seed_{s}" / "results" / "outputs_val.npz")
        val_probs_dict[short] = v_npz["probs"][:, 1] if v_npz["probs"].ndim > 1 else v_npz["probs"].ravel()
        if val_y is None: val_y = v_npz["y_true"]
        
        d_npz = np.load(base / rel / f"seed_{s}" / "results" / "outputs_df40.npz")
        df40_probs_dict[short] = d_npz["probs"]
        
    # Média dos 3 modelos na validação
    fused_val = np.mean([val_probs_dict[m] for m in ["dino_freq", "xcep_srm_only", "xcep_srm"]], axis=0)
    fpr, tpr, thresholds = roc_curve(val_y, fused_val)
    j_scores = tpr - fpr
    best_idx = np.argmax(j_scores)
    ens_thresh = thresholds[best_idx]
    
    # Limiares individuais na validação
    ind_thresh = {}
    for short, rel in models_info:
        v_npz = np.load(base / rel / f"seed_{s}" / "results" / "outputs_val.npz")
        ind_thresh[short] = float(v_npz["threshold"])
        
    fused_df40 = np.mean([df40_probs_dict[m] for m in ["dino_freq", "xcep_srm_only", "xcep_srm"]], axis=0)
    
    pred_soft_cal = (fused_df40 >= ens_thresh).astype(int)
    pred_soft_05 = (fused_df40 >= 0.50).astype(int)
    
    pred_or = np.zeros(len(fused_df40), dtype=int)
    for short, _ in models_info:
        pred_or = pred_or | (df40_probs_dict[short] >= ind_thresh[short]).astype(int)
        
    seed_results.append({
        "ens_thresh": ens_thresh,
        "pred_soft_cal": pred_soft_cal,
        "pred_soft_05": pred_soft_05,
        "pred_or": pred_or,
        "dino_pred": (df40_probs_dict["dino_freq"] >= ind_thresh["dino_freq"]).astype(int),
        "xso_pred": (df40_probs_dict["xcep_srm_only"] >= ind_thresh["xcep_srm_only"]).astype(int),
        "xs_pred": (df40_probs_dict["xcep_srm"] >= ind_thresh["xcep_srm"]).astype(int),
    })

print("Ensemble thresholds on MFFI Val across seeds:", [f"{r['ens_thresh']:.4f}" for r in seed_results])

print("\n=== COMPARATIVO POR TECNICA (Media 3 Seeds) ===")
header = f"{'Paradigma':<12} | {'Tecnica':<16} | {'DINOv3':>7} | {'XcepSRM':>7} | {'XcepRGB':>7} | {'Ens(Mean)':>9} | {'Ens(OR)':>7}"
print(header)
print("-" * len(header))

for parad, tech, label in tech_order:
    idx_tech = (manifest["target"] == 1) & (manifest["method"] == tech)
    sub_indices = np.where(idx_tech)[0]
    
    rates_d = [seed_results[s]["dino_pred"][sub_indices].mean() * 100 for s in range(3)]
    rates_xso = [seed_results[s]["xso_pred"][sub_indices].mean() * 100 for s in range(3)]
    rates_xs = [seed_results[s]["xs_pred"][sub_indices].mean() * 100 for s in range(3)]
    rates_mean = [seed_results[s]["pred_soft_cal"][sub_indices].mean() * 100 for s in range(3)]
    rates_or = [seed_results[s]["pred_or"][sub_indices].mean() * 100 for s in range(3)]
    
    print(f"{parad:<12} | {label:<16} | {np.mean(rates_d):>6.1f}% | {np.mean(rates_xso):>6.1f}% | {np.mean(rates_xs):>6.1f}% | {np.mean(rates_mean):>8.1f}% | {np.mean(rates_or):>6.1f}%")

fake_indices = np.where(manifest["target"] == 1)[0]
total_dino = [seed_results[s]["dino_pred"][fake_indices].mean() * 100 for s in range(3)]
total_xso = [seed_results[s]["xso_pred"][fake_indices].mean() * 100 for s in range(3)]
total_xs = [seed_results[s]["xs_pred"][fake_indices].mean() * 100 for s in range(3)]
total_mean = [seed_results[s]["pred_soft_cal"][fake_indices].mean() * 100 for s in range(3)]
total_or = [seed_results[s]["pred_or"][fake_indices].mean() * 100 for s in range(3)]

print("-" * len(header))
print(f"{'TOTAL GERAL':<12} | {'6.147 Fakes':<16} | {np.mean(total_dino):>6.1f}% | {np.mean(total_xso):>6.1f}% | {np.mean(total_xs):>6.1f}% | {np.mean(total_mean):>8.1f}% | {np.mean(total_or):>6.1f}%")

real_indices = np.where(manifest["target"] == 0)[0]
spec_mean = [100 - seed_results[s]["pred_soft_cal"][real_indices].mean() * 100 for s in range(3)]
spec_or = [100 - seed_results[s]["pred_or"][real_indices].mean() * 100 for s in range(3)]
print(f"{'ESPECIFIC.':<12} | {'4.999 Reais':<16} | {'-':>7} | {'-':>7} | {'-':>7} | {np.mean(spec_mean):>8.1f}% | {np.mean(spec_or):>6.1f}%")
