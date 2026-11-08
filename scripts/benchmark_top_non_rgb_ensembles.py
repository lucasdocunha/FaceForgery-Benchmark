#!/usr/bin/env python3
"""Executa a avaliação exaustiva de TODOS os métodos de fusão para comitês Top-2 até Top-5
considerando EXCLUSIVAMENTE modelos SEM RGB normal (usando representações SRM e Fourier Concat).
"""

from pathlib import Path
import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score
from sklearn.linear_model import LogisticRegression
from scipy.optimize import minimize

BASE_DIR = Path("/media/ssd2/lucas.ocunha/models-tcc")

MODELS = {
    # Resíduos Forenses (SRM)
    "dino_srm_s123": BASE_DIR / "dino/srm/finetune_robust/seed_123/results",
    "clip_srm_s2024": BASE_DIR / "clip/srm/finetune_robust/seed_2024/results",
    "resnet_srm_s123": BASE_DIR / "resnet/srm/finetune_robust/seed_123/results",
    "mobilenet_srm_s7": BASE_DIR / "mobilenet/srm/finetune_robust/seed_7/results",
    "vit_srm_s123": BASE_DIR / "vit/srm/finetune_robust/seed_123/results",
    # Espectrais de Fourier (Concat)
    "resnet_concat_s123": BASE_DIR / "resnet/concat/finetune/seed_123/results",
    "clip_concat_s2025": BASE_DIR / "clip/concat/finetune/seed_2025/results",
    "dino_concat_s123": BASE_DIR / "dino/concat/finetune/seed_123/results",
    "xception_concat_s42": BASE_DIR / "xception/concat/finetune/seed_42/results",
}

COMITES = {
    # 1. Comitês Campeões Não-RGB (SRM + Fourier Concat)
    "Top-2 Não-RGB (DINO SRM + CLIP SRM)": [
        "dino_srm_s123", "clip_srm_s2024"
    ],
    "Top-3 Não-RGB (DINO SRM + CLIP SRM + RESNET Concat)": [
        "dino_srm_s123", "clip_srm_s2024", "resnet_concat_s123"
    ],
    "Top-4 Não-RGB (DINO SRM + CLIP SRM + RESNET Concat + MOBILENET SRM)": [
        "dino_srm_s123", "clip_srm_s2024", "resnet_concat_s123", "mobilenet_srm_s7"
    ],
    "Top-5 Não-RGB (DINO SRM + CLIP SRM + RESNET Concat + MOBILENET SRM + VIT SRM)": [
        "dino_srm_s123", "clip_srm_s2024", "resnet_concat_s123", "mobilenet_srm_s7", "vit_srm_s123"
    ],

    # 2. Comitês Exclusivamente Fourier Concat (Sem SRM, Sem RGB Normal)
    "Top-2 Fourier Concat (DINO Concat + RESNET Concat)": [
        "dino_concat_s123", "resnet_concat_s123"
    ],
    "Top-3 Fourier Concat (DINO Concat + RESNET Concat + CLIP Concat)": [
        "dino_concat_s123", "resnet_concat_s123", "clip_concat_s2025"
    ],
    "Top-4 Fourier Concat (DINO Concat + RESNET Concat + CLIP Concat + XCEPTION Concat)": [
        "dino_concat_s123", "resnet_concat_s123", "clip_concat_s2025", "xception_concat_s42"
    ],
}

def safe_auc(y_true, y_score):
    try:
        y_score = np.nan_to_num(y_score, nan=0.5, posinf=1.0, neginf=0.0)
        if len(np.unique(y_true)) < 2:
            return 0.5
        return float(roc_auc_score(y_true, y_score))
    except Exception:
        return 0.5

def load_data():
    loaded = {}
    for name, path in MODELS.items():
        data = {}
        with np.load(path / "outputs_val.npz") as d:
            data["val_probs"] = d["probs"]
            data["val_y"] = d["y_true"]
        with np.load(path / "outputs_test.npz") as d:
            data["test_probs"] = d["probs"]
            data["test_y"] = d["y_true"]
        with np.load(path / "outputs_test_d.npz") as d:
            data["test_d_probs"] = d["probs"]
            data["test_d_y"] = d["y_true"]
        with np.load(path / "outputs_celeb_df.npz") as d:
            data["celeb_frame_p"] = d["p_fake"]
            data["celeb_frame_y"] = d["y_true"]
            data["celeb_video_p"] = d["video_p"]
            data["celeb_video_y"] = d["video_y"]
        with np.load(path / "outputs_df40.npz") as d:
            data["df40_probs"] = d["probs"]
            data["df40_y"] = d["y_true"]
        loaded[name] = data
    return loaded

def fit_stacking(val_probs_matrix, val_y):
    eps = 1e-6
    X_val = np.clip(val_probs_matrix, eps, 1.0 - eps)
    clf = LogisticRegression(random_state=42, max_iter=1000, C=1.0)
    clf.fit(X_val, val_y)
    return clf

def fit_optimal_weights(val_probs_matrix, val_y):
    K = val_probs_matrix.shape[1]
    eps = 1e-7
    def loss(w):
        w_norm = w / (np.sum(w) + eps)
        p = np.dot(val_probs_matrix, w_norm)
        p = np.clip(p, eps, 1.0 - eps)
        return -np.mean(val_y * np.log(p) + (1.0 - val_y) * np.log(1.0 - p))
    w0 = np.ones(K) / K
    bnds = [(0, 1) for _ in range(K)]
    cons = ({'type': 'eq', 'fun': lambda w: np.sum(w) - 1.0})
    res = minimize(loss, w0, method='SLSQP', bounds=bnds, constraints=cons)
    return res.x / np.sum(res.x)

def fuse(probs_list, strategy, val_aucs=None, stacking_clf=None, optimal_w=None):
    stack = np.stack(probs_list, axis=0) # [K, N]
    eps = 1e-7
    if strategy == "mean":
        return np.mean(stack, axis=0)
    elif strategy == "geometric":
        clipped = np.clip(stack, eps, 1.0 - eps)
        return np.exp(np.mean(np.log(clipped), axis=0))
    elif strategy == "harmonic":
        clipped = np.clip(stack, eps, 1.0)
        return len(probs_list) / np.sum(1.0 / clipped, axis=0)
    elif strategy == "max":
        return np.max(stack, axis=0)
    elif strategy == "weighted":
        weights = np.maximum(np.array(val_aucs) - 0.5, 0.0)
        if weights.sum() == 0:
            return np.mean(stack, axis=0)
        return np.average(stack, axis=0, weights=weights)
    elif strategy == "optimal_weights":
        return np.dot(optimal_w, stack)
    elif strategy == "stacking":
        X = np.clip(stack.T, eps, 1.0 - eps)
        return stacking_clf.predict_proba(X)[:, 1]
    elif strategy == "majority":
        return np.mean((stack >= 0.5).astype(float), axis=0)
    else:
        raise ValueError(f"Unknown strategy: {strategy}")

def main():
    print("Carregando modelos Não-RGB (SRM e Fourier)...")
    loaded = load_data()
    print("Modelos carregados com sucesso!")

    strategies = [
        "mean",
        "geometric",
        "harmonic",
        "max",
        "weighted",
        "optimal_weights",
        "stacking",
        "majority",
    ]

    records = []

    for c_name, members in COMITES.items():
        K = len(members)
        print(f"\nProcessando {c_name} (K={K})...")
        
        val_probs = [loaded[m]["val_probs"] for m in members]
        val_y = loaded[members[0]]["val_y"]
        val_aucs = [safe_auc(val_y, p) for p in val_probs]
        val_mat = np.stack(val_probs, axis=1)

        stacking_clf = fit_stacking(val_mat, val_y)
        optimal_w = fit_optimal_weights(val_mat, val_y)

        splits_data = {
            "val": (val_probs, val_y),
            "test": ([loaded[m]["test_probs"] for m in members], loaded[members[0]]["test_y"]),
            "test_d": ([loaded[m]["test_d_probs"] for m in members], loaded[members[0]]["test_d_y"]),
            "celeb_frame": ([loaded[m]["celeb_frame_p"] for m in members], loaded[members[0]]["celeb_frame_y"]),
            "celeb_video": ([loaded[m]["celeb_video_p"] for m in members], loaded[members[0]]["celeb_video_y"]),
            "df40": ([loaded[m]["df40_probs"] for m in members], loaded[members[0]]["df40_y"]),
        }

        for strat in strategies:
            row = {
                "Comitê": c_name,
                "K": K,
                "Estratégia": strat,
            }
            for split_name, (p_list, y_true) in splits_data.items():
                p_fused = fuse(p_list, strat, val_aucs=val_aucs, stacking_clf=stacking_clf, optimal_w=optimal_w)
                auc = safe_auc(y_true, p_fused)
                row[f"{split_name}_auc"] = round(auc * 100, 2)
            
            records.append(row)

    df = pd.DataFrame(records)
    out_csv = Path("tables/benchmark_non_rgb_ensembles_all_methods.csv")
    out_csv.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(out_csv, index=False)
    print(f"\nSalvo com sucesso em {out_csv}!")

    pd.set_option('display.max_columns', 15)
    pd.set_option('display.width', 1000)
    print("\nResultados Consolidados dos Ensembles SEM RGB Normal:")
    print(df[['Comitê', 'Estratégia', 'val_auc', 'test_auc', 'test_d_auc', 'df40_auc', 'celeb_video_auc']])

if __name__ == "__main__":
    main()
