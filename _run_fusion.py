import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import roc_auc_score
from src.pipelines.evaluation import best_threshold, binary_metrics, clean_probabilities

TOP3_VAL = [
    "models/dino/dinov3_base/rgb/results/predictions_val.csv",
    "models/vit/vit_scratch/concat/results/predictions_val.csv",
    "models/resnet/concat_frequency/results/predictions_val.csv",
]

val_dfs = [pd.read_csv(p) for p in TOP3_VAL]
val_aucs = [roc_auc_score(df["y_true"].values, df["prob_pos"].values) for df in val_dfs]
print("Val AUC individuais:", [round(a, 4) for a in val_aucs])

def load_split(split):
    return [pd.read_csv(p.replace("predictions_val.csv", f"predictions_{split}.csv")) for p in TOP3_VAL]

def pmat(dfs):
    return np.stack([clean_probabilities(df["prob_pos"].values) for df in dfs], axis=1)

def fast_best_threshold(y_true, probs, n=300):
    """Busca threshold em n pontos uniformes — O(n*N) em vez de O(N²)."""
    candidates = np.linspace(probs.min(), probs.max(), n)
    candidates = np.concatenate([[0.5], candidates])
    best_thr, best_acc = 0.5, -1.0
    for thr in candidates:
        acc = (y_true == (probs >= thr).astype(int)).mean()
        if acc > best_acc:
            best_acc, best_thr = acc, float(thr)
    return best_thr


def run_strategy(name, Pv, Ps, y_val, y_split):
    if name == "soft_voting":
        fv = Pv.mean(axis=1)
        fs = Ps.mean(axis=1)
    elif name == "weighted_avg":
        w = np.array(val_aucs); w = w / w.sum()
        fv = (Pv * w).sum(axis=1)
        fs = (Ps * w).sum(axis=1)
    elif name == "majority_vote":
        thrs = [fast_best_threshold(y_val, Pv[:, i]) for i in range(3)]
        hv = np.stack([(Pv[:, i] >= thrs[i]).astype(int) for i in range(3)], axis=1)
        hs = np.stack([(Ps[:, i] >= thrs[i]).astype(int) for i in range(3)], axis=1)
        fv = hv.mean(axis=1)
        fs = hs.mean(axis=1)
    elif name == "stacking_lr":
        sc = StandardScaler()
        Xv = sc.fit_transform(Pv)
        lr = LogisticRegression(max_iter=1000, C=1.0, random_state=42)
        lr.fit(Xv, y_val)
        fv = lr.predict_proba(Xv)[:, 1]
        fs = lr.predict_proba(sc.transform(Ps))[:, 1]
    elif name == "max_prob":
        fv = Pv.max(axis=1)
        fs = Ps.max(axis=1)
    elif name == "geometric_mean":
        eps = 1e-7
        fv = np.exp(np.log(np.clip(Pv, eps, 1 - eps)).mean(axis=1))
        fs = np.exp(np.log(np.clip(Ps, eps, 1 - eps)).mean(axis=1))

    thr = fast_best_threshold(y_val, fv)
    vm = binary_metrics(y_val, fv, threshold=thr)
    tm = binary_metrics(y_split, fs, threshold=thr)
    return thr, vm, tm


y_val = val_dfs[0]["y_true"].values.astype(int)
Pv = pmat(val_dfs)
strategies = ["soft_voting", "weighted_avg", "majority_vote", "stacking_lr", "max_prob", "geometric_mean"]

for split in ["test", "test_d"]:
    split_dfs = load_split(split)
    y_split = split_dfs[0]["y_true"].values.astype(int)
    Ps = pmat(split_dfs)
    print(f"\n=== SPLIT: {split.upper()} ===")
    print(f"{'estrategia':<16} {'thr':>5} {'val_auc':>8} {'auc':>8} {'acc':>7} {'f1':>7} {'recall':>8} {'spec':>8}")
    print("-" * 75)
    for s in strategies:
        thr, vm, tm = run_strategy(s, Pv, Ps, y_val, y_split)
        print(f"{s:<16} {thr:>5.3f} {vm['auc']:>8.4f} {tm['auc']:>8.4f} {tm['acc']:>7.4f} {tm['f1']:>7.4f} {tm['recall']:>8.4f} {tm['specificity']:>8.4f}")
