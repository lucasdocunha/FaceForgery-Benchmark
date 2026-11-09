"""
Gera relatórios consolidados do DF-40 por categoria (paradigmas) e por técnica individual
para a campanha SRM (5 seeds canônicas).
"""
import numpy as np
import pandas as pd
from pathlib import Path
from sklearn.metrics import roc_auc_score, accuracy_score, balanced_accuracy_score

manifest = pd.read_csv('data/df40/test.csv')
y_true = manifest['target'].values
methods = manifest['method'].values
paradigms = manifest['paradigm'].values
real_mask = (manifest['target'] == 0).values

import os

root = Path(os.environ.get("TCC_MODELS_ROOT", os.environ.get("CISIA_MODELS_ROOT", "/media/ssd2/lucas.ocunha/models-tcc")))
if not root.exists():
    for candidate in [
        Path(f"/projects/models/{os.environ.get('USER', 'lucas.ocunha')}/models-tcc"),
        Path(f"/projects/models/{os.environ.get('USER', 'lucas.ocunha')}/faceforgery"),
        Path("/media/ssd2/lucas.ocunha/models-tcc"),
    ]:
        if candidate.exists():
            root = candidate
            break

families = ['clip', 'dino', 'vit', 'resnet', 'mobilenet', 'xception']
seeds = [42, 123, 2024, 7, 2025]

# 1. Carregar predições das 5 seeds
data = {}
for fam in families:
    for s in seeds:
        p = root / fam / 'srm' / 'finetune_robust' / f'seed_{s}' / 'results' / 'outputs_df40.npz'
        d = np.load(p)
        data[f'{fam}_s{s}'] = {'probs': d['probs'], 'threshold': float(d['threshold'])}

# 2. Deep ensembles por família
for fam in families:
    fam_probs = np.mean([data[f'{fam}_s{s}']['probs'] for s in seeds], axis=0)
    fam_thresh = np.mean([data[f'{fam}_s{s}']['threshold'] for s in seeds])
    data[f'{fam}_5seeds'] = {'probs': fam_probs, 'threshold': fam_thresh}

# 3. Ensembles principais
clip_dino_probs = 0.5 * (data['clip_5seeds']['probs'] + data['dino_5seeds']['probs'])
clip_dino_thresh = 0.5 * (data['clip_5seeds']['threshold'] + data['dino_5seeds']['threshold'])
data['clip_dino_5seeds'] = {'probs': clip_dino_probs, 'threshold': clip_dino_thresh}

mega_probs = np.mean([data[f'{fam}_s{s}']['probs'] for fam in families for s in seeds], axis=0)
mega_thresh = np.mean([data[f'{fam}_s{s}']['threshold'] for fam in families for s in seeds])
data['mega_30models'] = {'probs': mega_probs, 'threshold': mega_thresh}

# -------------------------------------------------------------
# TABELA 1: Desempenho por Paradigma (Macro-Categorias)
# -------------------------------------------------------------
target_models = ['clip_5seeds', 'dino_5seeds', 'clip_dino_5seeds', 'vit_5seeds', 'resnet_5seeds', 'mobilenet_5seeds', 'xception_5seeds', 'mega_30models']
model_display_names = {
    'clip_5seeds': 'CLIP ViT-B/16 (SRM)',
    'dino_5seeds': 'DINO ConvNeXt-B (SRM)',
    'clip_dino_5seeds': 'Fusão CLIP + DINO (SRM)',
    'vit_5seeds': 'ViT-B/16 (SRM)',
    'resnet_5seeds': 'ResNet-50 (SRM)',
    'mobilenet_5seeds': 'MobileNet-v2 (SRM)',
    'xception_5seeds': 'Xception (SRM)',
    'mega_30models': 'Mega-Ensemble (30 Modelos SRM)'
}

fake_paradigms = ['face_swap', 'editing_t2i', 'talking_reenactment', 'gan', 'diffusion', 'commercial_avatar']
paradigm_pt_names = {
    'real': 'Rostos Reais (Pristine)',
    'face_swap': 'Troca Facial (Face Swap)',
    'editing_t2i': 'Edição Text-to-Image (T2I)',
    'talking_reenactment': 'Reencenação Facial (Talking)',
    'gan': 'Síntese GAN',
    'diffusion': 'Modelos de Difusão (Diffusion)',
    'commercial_avatar': 'Avatares Comerciais (HeyGen)'
}

paradigm_rows = []
for m_key in target_models:
    probs = data[m_key]['probs']
    thresh = data[m_key]['threshold']
    y_pred = (probs >= thresh).astype(int)
    
    real_acc = (y_pred[real_mask] == 0).mean() * 100.0
    
    for para in fake_paradigms:
        p_mask = (paradigms == para)
        sub_mask = real_mask | p_mask
        sub_y = y_true[sub_mask]
        sub_probs = probs[sub_mask]
        sub_pred = y_pred[sub_mask]
        
        auc = roc_auc_score(sub_y, sub_probs) * 100.0
        acc_total = accuracy_score(sub_y, sub_pred) * 100.0
        bacc = balanced_accuracy_score(sub_y, sub_pred) * 100.0
        rec_fake = (sub_pred[sub_y == 1] == 1).mean() * 100.0
        
        paradigm_rows.append({
            'Modelo': model_display_names[m_key],
            'Paradigma': paradigm_pt_names[para],
            'Paradigma_Raw': para,
            'N_Fake': int(p_mask.sum()),
            'Taxa_Acerto_Fake_Recall_%': round(rec_fake, 2),
            'Taxa_Acerto_Real_Espec_%': round(real_acc, 2),
            'Acuracia_Balanceada_%': round(bacc, 2),
            'AUC_%': round(auc, 2),
            'Threshold_Usado': round(thresh, 3)
        })

df_paradigms = pd.DataFrame(paradigm_rows)
out_csv_para = Path('tables/df40_srm_breakdown_by_paradigm.csv')
out_csv_para.parent.mkdir(parents=True, exist_ok=True)
df_paradigms.to_csv(out_csv_para, index=False)
print(f"Salvo: {out_csv_para}")

# -------------------------------------------------------------
# TABELA 2: Desempenho por Técnica Específica (24 métodos)
# -------------------------------------------------------------
fake_methods = sorted(manifest[manifest['target'] == 1]['method'].unique())
tech_rows = []

for m in fake_methods:
    m_mask = (methods == m)
    para = manifest[manifest['method'] == m]['paradigm'].iloc[0]
    n = int(m_mask.sum())
    sub_mask = real_mask | m_mask
    sub_y = y_true[sub_mask]
    
    row = {
        'Paradigma': paradigm_pt_names[para],
        'Tecnica': m,
        'N_Amostras': n,
    }
    
    for m_key in ['clip_5seeds', 'dino_5seeds', 'clip_dino_5seeds', 'mega_30models']:
        probs = data[m_key]['probs']
        thresh = data[m_key]['threshold']
        pred = (probs >= thresh).astype(int)
        
        rec = (pred[m_mask] == 1).mean() * 100.0
        auc = roc_auc_score(sub_y, probs[sub_mask]) * 100.0
        
        prefix = m_key.replace('_5seeds', '').replace('_30models', '')
        row[f'{prefix}_Acerto_%'] = round(rec, 1)
        row[f'{prefix}_AUC_%'] = round(auc, 1)
        
    tech_rows.append(row)

df_tech = pd.DataFrame(tech_rows).sort_values(['Paradigma', 'clip_dino_AUC_%'], ascending=[True, False])
out_csv_tech = Path('tables/df40_srm_breakdown_by_technique.csv')
df_tech.to_csv(out_csv_tech, index=False)
print(f"Salvo: {out_csv_tech}")

print("Tabelas DF-40 geradas com sucesso!")
