import sys
from pathlib import Path
ROOT_DIR = Path.cwd()
sys.path.insert(0, str(ROOT_DIR))

import numpy as np
import pandas as pd
from sklearn.metrics import roc_curve, roc_auc_score
from scripts.benchmark_ensembles_3groups import (
    scan_all_models,
    load_model_data,
    fit_stacking,
    fit_optimal_weights,
    fuse,
    safe_auc,
    STRAT_PT,
    format_member_display
)

df_all = scan_all_models()
path_by_id = df_all.set_index("id")["path"].to_dict()

winning_configs = [
    # Espacial Robusto
    ("Espacial Robusto", 2, ["dino_none_finetune_robust_seed_123", "clip_none_finetune_robust_seed_42"], "geometric"),
    ("Espacial Robusto", 3, ["dino_none_finetune_robust_seed_123", "clip_none_finetune_robust_seed_42", "mobilenet_none_finetune_robust_seed_987"], "optimal_weights"),
    ("Espacial Robusto", 4, ["dino_none_finetune_robust_seed_123", "clip_none_finetune_robust_seed_42", "mobilenet_none_finetune_robust_seed_987", "xception_none_finetune_robust_seed_123"], "optimal_weights"),
    
    # Espacial Convencional
    ("Espacial Convencional", 2, ["dino_none_finetune_seed_123", "clip_none_finetune_seed_987"], "geometric"),
    ("Espacial Convencional", 3, ["dino_none_finetune_seed_123", "clip_none_finetune_seed_987", "resnet_none_finetune_seed_2025"], "stacking"),
    ("Espacial Convencional", 4, ["dino_none_finetune_seed_123", "clip_none_finetune_seed_987", "resnet_none_finetune_seed_2025", "mobilenet_none_finetune_seed_7"], "stacking"),

    # Híbrido Multidomínio
    ("Híbrido Multidomínio", 2, ["dino_srm_finetune_robust_seed_7", "dino_concat_finetune_seed_42"], "stacking"),
    ("Híbrido Multidomínio", 3, ["dino_srm_finetune_robust_seed_7", "dino_concat_finetune_seed_42", "clip_srm_finetune_robust_seed_2024"], "geometric"),
    ("Híbrido Multidomínio", 4, ["dino_srm_finetune_robust_seed_7", "dino_concat_finetune_seed_42", "clip_srm_finetune_robust_seed_2024", "resnet_concat_finetune_seed_2024"], "optimal_weights"),

    # Não-RGB (SRM e Frequência)
    ("Não-RGB (SRM e Frequência)", 2, ["resnet_srm_only_finetune_robust_seed_123", "dino_frequency_3_finetune_seed_2025"], "stacking"),
    ("Não-RGB (SRM e Frequência)", 3, ["resnet_srm_only_finetune_robust_seed_123", "dino_srm_only_finetune_robust_seed_42", "dino_magnitude_finetune_seed_7"], "stacking"),
    ("Não-RGB (SRM e Frequência)", 4, ["resnet_srm_only_finetune_robust_seed_123", "dino_srm_only_finetune_robust_seed_42", "dino_magnitude_finetune_seed_7", "dino_frequency_3_finetune_seed_2025"], "stacking"),
]

# Preload only the unique models
needed_ids = set()
for _, _, combo, _ in winning_configs:
    needed_ids.update(combo)

print(f"Pre-loading {len(needed_ids)} distinct models...")
cache = {m_id: load_model_data(path_by_id[m_id]) for m_id in needed_ids}
print("Models loaded successfully!")

manifest = pd.read_csv('data/df40/test.csv')
p2_df = pd.read_csv('tables/tabela_tecnicas_df40_top3_revocacao.csv')
p2_weights = p2_df.set_index('tecnica')['n_amostras'].to_dict()

tech_map = {
    'sd2.1': 'SD 2.1', 'pixart': 'PixArt-alpha', 'SiT': 'SiT', 'DiT': 'DiT',
    'RDDM': 'RDDM', 'CollabDiff': 'CollabDiff', 'ddim': 'DDIM',
    'StyleGAN3': 'StyleGAN3', 'StyleGAN2': 'StyleGAN2', 'StyleGANXL': 'StyleGAN-XL',
    'VQGAN': 'VQGAN', 'stargan': 'StarGAN', 'starganv2': 'StarGAN v2',
    'uniface': 'UniFace', 'faceswap': 'FaceSwap', 'blendface': 'BlendFace',
    'mobileswap': 'MobileSwap', 'deepfacelab': 'DeepFaceLab',
    'whichfaceisreal': 'WhichFaceIsReal', 'MidJourney': 'MidJourney',
    'e4e': 'e4e', 'styleclip': 'StyleCLIP', 'cdf_reenactment': 'Celeb-DF (misto)',
    'heygen': 'HeyGen'
}

rows = []
for g_name, K, combo, strat in winning_configs:
    val_probs = [cache[m]["val_p"] for m in combo]
    val_y = cache[combo[0]]["val_y"]
    val_aucs = [safe_auc(val_y, p) for p in val_probs]
    val_mat = np.stack(val_probs, axis=1)

    stacking_clf = fit_stacking(val_mat, val_y)
    optimal_w = fit_optimal_weights(val_mat, val_y)

    m_str = " + ".join([format_member_display(m) for m in combo])
    val_fused = fuse(val_probs, strat, val_aucs=val_aucs, stacking_clf=stacking_clf, optimal_w=optimal_w)
    val_auc = safe_auc(val_y, val_fused) * 100.0

    # Calibrar limiar ótimo no MFFI Val (Youden's J)
    fpr, tpr, ths = roc_curve(val_y, val_fused)
    opt_th = ths[np.argmax(tpr - fpr)]

    test_probs = [cache[m]["test_p"] for m in combo]
    test_y = cache[combo[0]]["test_y"]
    test_fused = fuse(test_probs, strat, val_aucs=val_aucs, stacking_clf=stacking_clf, optimal_w=optimal_w)
    test_auc = safe_auc(test_y, test_fused) * 100.0

    hard_probs = [cache[m]["test_d_p"] for m in combo]
    hard_y = cache[combo[0]]["test_d_y"]
    hard_fused = fuse(hard_probs, strat, val_aucs=val_aucs, stacking_clf=stacking_clf, optimal_w=optimal_w)
    hard_auc = safe_auc(hard_y, hard_fused) * 100.0

    celeb_vid_probs = [cache[m]["celeb_video_p"] for m in combo]
    celeb_vid_y = cache[combo[0]]["celeb_video_y"]
    celeb_vid_fused = fuse(celeb_vid_probs, strat, val_aucs=val_aucs, stacking_clf=stacking_clf, optimal_w=optimal_w)
    celeb_vid_auc = safe_auc(celeb_vid_y, celeb_vid_fused) * 100.0

    df40_probs = [cache[m]["df40_p"] for m in combo]
    df40_y = cache[combo[0]]["df40_y"]
    df40_fused = fuse(df40_probs, strat, val_aucs=val_aucs, stacking_clf=stacking_clf, optimal_w=optimal_w)
    df40_auc = safe_auc(df40_y, df40_fused) * 100.0

    # Revocação Protocolo 1 (6.147 imagens falsas)
    fake_mask = (manifest["target"] == 1)
    pred_fake = (df40_fused[fake_mask] >= opt_th).astype(int)
    df40_p1_rec = np.mean(pred_fake) * 100.0

    # Revocação Protocolo 2 ponderada (650.549 imagens falsas)
    tot_samples = 0
    tot_hits = 0
    for method_key, tech_name in tech_map.items():
        sub_mask = (manifest["target"] == 1) & (manifest["method"] == method_key)
        sub_preds = (df40_fused[sub_mask] >= opt_th).astype(int)
        tech_rec = np.mean(sub_preds) if len(sub_preds) > 0 else 0.0
        n_tech = p2_weights[tech_name]
        tot_samples += n_tech
        tot_hits += n_tech * tech_rec

    df40_p2_rec = (tot_hits / tot_samples) * 100.0

    rows.append({
        "Grupo": g_name,
        "K": K,
        "Comitê": f"Top-{K}",
        "Membros": m_str,
        "Estratégia": strat,
        "Estratégia_PT": STRAT_PT.get(strat, strat),
        "Val_AUC": val_auc,
        "Test_AUC": test_auc,
        "Test_Hard_AUC": hard_auc,
        "DF40_AUC": df40_auc,
        "Celeb_Video_AUC": celeb_vid_auc,
        "DF40_P1_Rec": df40_p1_rec,
        "DF40_P2_Rec": df40_p2_rec,
        "Threshold_Val": opt_th
    })

df_res = pd.DataFrame(rows)
df_res.to_csv("tables/benchmark_4groups_top2_top4.csv", index=False)
print("\n=== RESULTADOS TOP-2 A TOP-4 NAS 4 CATEGORIAS ===")
print(df_res[["Grupo", "K", "Estratégia_PT", "Val_AUC", "Test_AUC", "Test_Hard_AUC", "DF40_AUC", "Celeb_Video_AUC", "DF40_P2_Rec"]].to_string())
