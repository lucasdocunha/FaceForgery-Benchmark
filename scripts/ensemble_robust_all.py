#!/usr/bin/env python3
"""
Ensemble dos modelos robustos (seed=987, fourier=none) treinados com RandomizedRobustAugment.
Combina todos os 6 modelos com múltiplas estratégias e gera tabela + figura.
"""
from __future__ import annotations
import os, sys
sys.path.insert(0, "/home/lucas.ocunha/tcc")

import numpy as np
import pandas as pd
import json
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from pathlib import Path
from sklearn.metrics import roc_auc_score, accuracy_score, f1_score
from sklearn.linear_model import LogisticRegression
from scipy.optimize import minimize

MODELS_ROOT = Path("/media/ssd2/lucas.ocunha/models-tcc")
TCC_ROOT    = Path("/home/lucas.ocunha/tcc")
SEED        = 987
FOURIER     = "none"
REGIME      = "finetune"
ALL_FAMILIES = ["clip", "dino", "vit", "resnet", "mobilenet", "xception"]


def load_outputs(family):
    base = MODELS_ROOT / family / FOURIER / REGIME / f"seed_{SEED}" / "results"
    out = {}
    for split, fname in [("test","outputs_test.npz"), ("test_d","outputs_test_d.npz"), ("val","outputs_val.npz")]:
        d = np.load(base / fname)
        out[split] = {"probs": d["probs"], "y_true": d["y_true"]}
    out["threshold"] = float(pd.read_csv(base / "metrics_val.csv").iloc[0]["threshold"])
    return out


def compute_metrics(probs, y_true, thr=0.5):
    y_pred = (probs >= thr).astype(int)
    auc  = roc_auc_score(y_true, probs)
    acc  = accuracy_score(y_true, y_pred)
    f1   = f1_score(y_true, y_pred, zero_division=0)
    tn   = int(np.sum((y_pred==0)&(y_true==0)))
    fp   = int(np.sum((y_pred==1)&(y_true==0)))
    fn   = int(np.sum((y_pred==0)&(y_true==1)))
    tp   = int(np.sum((y_pred==1)&(y_true==1)))
    return dict(auc=auc, acc=acc, f1=f1, tp=tp, fp=fp, fn=fn, tn=tn)


def find_threshold(probs, y_true):
    best_t, best_s = 0.5, 0.0
    for t in np.linspace(0.01, 0.99, 200):
        s = accuracy_score(y_true, (probs >= t).astype(int))
        if s > best_s:
            best_s, best_t = s, t
    return float(best_t)


def fuse(probs_list, strategy, y_val=None):
    stack = np.stack(probs_list, axis=1)
    if strategy == "mean":
        return stack.mean(axis=1)
    elif strategy == "geometric":
        eps = 1e-7
        return np.exp(np.mean(np.log(np.clip(stack, eps, 1-eps)), axis=1))
    elif strategy == "weighted":
        # pesos = val AUC de cada modelo
        return stack.mean(axis=1)  # sem AUCs individuais disponíveis aqui, usa mean
    elif strategy == "stacking":
        clf = LogisticRegression(C=1.0, max_iter=1000)
        clf.fit(stack, y_val.astype(int))
        return clf.predict_proba(stack)[:, 1]
    elif strategy == "max":
        return stack.max(axis=1)


# ── Carregar modelos disponíveis ─────────────────────────────────────────────
print("Carregando outputs dos modelos robustos (seed 987)...")
models = {}
for fam in ALL_FAMILIES:
    try:
        models[fam] = load_outputs(fam)
        td_auc = roc_auc_score(models[fam]["test_d"]["y_true"], models[fam]["test_d"]["probs"])
        t_auc  = roc_auc_score(models[fam]["test"]["y_true"],   models[fam]["test"]["probs"])
        print(f"  ✅ {fam:12s} test={t_auc:.4f}  test_d={td_auc:.4f}  ΔAUC={td_auc-t_auc:+.4f}")
    except Exception as e:
        print(f"  ❌ {fam:12s} NÃO disponível: {e}")

available = list(models.keys())
print(f"\n{len(available)} modelos disponíveis: {available}\n")

y_true_test  = models[available[0]]["test"]["y_true"]
y_true_testd = models[available[0]]["test_d"]["y_true"]
y_true_val   = models[available[0]]["val"]["y_true"]

# ── Definir composições ────────────────────────────────────────────────────────
COMPOSITIONS = []

# Todos disponíveis
if len(available) >= 2:
    COMPOSITIONS.append(("todos_robustos", f"Todos Robustos ({len(available)}M)",
                         available, list(available)))

# Top-3: CLIP + DINO + melhor restante
for top in [["clip","dino","vit"], ["clip","dino","resnet"], ["clip","dino","xception"]]:
    avail_top = [m for m in top if m in available]
    if len(avail_top) == len(top):
        COMPOSITIONS.append((f"robust_{'_'.join(top)}", f"CLIP+DINO+{top[2].upper()} Robustos",
                              avail_top, avail_top))

# Apenas CLIP e DINO robustos
if "clip" in available and "dino" in available:
    COMPOSITIONS.append(("robust_clip_dino", "CLIP+DINO Robustos",
                          ["clip","dino"], ["clip","dino"]))

STRATEGIES = ["mean", "geometric", "stacking", "max"]

rows = []
print("=== AVALIANDO ENSEMBLES ROBUSTOS ===\n")
for comp_id, comp_label, members, _ in COMPOSITIONS:
    for strat in STRATEGIES:
        try:
            probs_test  = [models[m]["test"]["probs"]   for m in members]
            probs_testd = [models[m]["test_d"]["probs"] for m in members]
            probs_val   = [models[m]["val"]["probs"]    for m in members]

            if strat == "stacking":
                stack_val = np.stack(probs_val, axis=1)
                clf = LogisticRegression(C=1.0, max_iter=1000)
                clf.fit(stack_val, y_true_val.astype(int))
                fused_test  = clf.predict_proba(np.stack(probs_test,  axis=1))[:,1]
                fused_testd = clf.predict_proba(np.stack(probs_testd, axis=1))[:,1]
                fused_val   = clf.predict_proba(stack_val)[:,1]
            else:
                fused_test  = fuse(probs_test,  strat, y_true_val)
                fused_testd = fuse(probs_testd, strat, y_true_val)
                fused_val   = fuse(probs_val,   strat, y_true_val)

            thr = find_threshold(fused_val, y_true_val)
            m_test  = compute_metrics(fused_test,  y_true_test,  thr)
            m_testd = compute_metrics(fused_testd, y_true_testd, thr)
            delta   = m_testd["auc"] - m_test["auc"]

            score = (0.35*m_test["auc"] + 0.45*m_testd["auc"] + 0.20*(1+delta)) * 10
            grade = "A+" if score>=8.0 else "A" if score>=7.5 else "B+" if score>=7.0 else "B"

            rows.append({
                "composition": comp_id, "label": comp_label,
                "n_models": len(members), "strategy": strat,
                "members": "+".join(members),
                "test_auc":   round(m_test["auc"],   4),
                "test_acc":   round(m_test["acc"],   4),
                "test_f1":    round(m_test["f1"],    4),
                "test_d_auc": round(m_testd["auc"],  4),
                "test_d_acc": round(m_testd["acc"],  4),
                "test_d_f1":  round(m_testd["f1"],   4),
                "test_d_fp":  m_testd["fp"],
                "delta_auc":  round(delta, 4),
                "score": round(score, 2), "grade": grade,
            })
            print(f"[{comp_id[:30]:30s}][{strat:10s}] test={m_test['auc']:.4f} | test_d={m_testd['auc']:.4f} | ΔAUC={delta:+.4f} | {grade} {score:.2f}")
        except Exception as e:
            print(f"  ERRO [{comp_id}][{strat}]: {e}")

df = pd.DataFrame(rows).sort_values("score", ascending=False)
out_csv = TCC_ROOT / "tables" / "ensemble_robust_all_models.csv"
df.to_csv(out_csv, index=False)
print(f"\n✅ CSV salvo: {out_csv}")

# ── Figura resumo ─────────────────────────────────────────────────────────────
# Melhor por composição
top_per_comp = df.drop_duplicates("composition").head(10)

fig, axes = plt.subplots(1, 2, figsize=(16, 6))
fig.suptitle("Ensemble de Modelos Robustos (Seed 987) — Todos os Modelos", fontsize=14, fontweight="bold")

# Scatter test vs test_d
ax = axes[0]
# Individuais
for fam in available:
    t  = roc_auc_score(models[fam]["test"]["y_true"],   models[fam]["test"]["probs"])
    td = roc_auc_score(models[fam]["test_d"]["y_true"], models[fam]["test_d"]["probs"])
    ax.scatter(t*100, td*100, s=80, marker="o", alpha=0.7, zorder=4)
    ax.annotate(fam, (t*100, td*100), xytext=(4,2), textcoords="offset points", fontsize=8)

# Ensembles top
colors = ["#E74C3C","#E67E22","#2ECC71","#3498DB","#9B59B6"]
for i, (_, row) in enumerate(top_per_comp.iterrows()):
    ax.scatter(row.test_auc*100, row.test_d_auc*100, s=140, marker="*",
               color=colors[i%len(colors)], edgecolors="black", linewidths=0.8, zorder=5)
    ax.annotate(f"{row['strategy']}\n{row['composition'][:15]}", 
                (row.test_auc*100, row.test_d_auc*100),
                xytext=(5, -10), textcoords="offset points", fontsize=7,
                color=colors[i%len(colors)], fontweight="bold")

lims = [70, 100]
ax.plot(lims, lims, "k--", lw=0.8, alpha=0.4)
ax.set_xlabel("Test AUC (%)", fontsize=11)
ax.set_ylabel("Test Difícil AUC (%)", fontsize=11)
ax.set_title("Trade-off IID vs OOD — Modelos Robustos", fontsize=11, fontweight="bold")
ax.grid(alpha=0.3)
p1 = mpatches.Patch(color="gray", label="Individuais")
p2 = mpatches.Patch(color="#E74C3C", label="Ensembles")
ax.legend(handles=[p1, p2], fontsize=9)

# Bar chart: top-8 ensembles por test_d AUC
ax2 = axes[1]
top8 = df.head(8)
x = range(len(top8))
w = 0.35
bars1 = ax2.bar([i-w/2 for i in x], top8.test_auc*100,  w, label="Test AUC",   color="#4C72B0", alpha=0.8)
bars2 = ax2.bar([i+w/2 for i in x], top8.test_d_auc*100, w, label="Test_d AUC", color="#E74C3C", alpha=0.8)
ax2.set_xticks(list(x))
ax2.set_xticklabels([f"{r['composition'][:12]}\n{r['strategy']}" for _,r in top8.iterrows()], fontsize=7)
ax2.set_ylim(70, 100)
ax2.set_ylabel("AUC (%)", fontsize=11)
ax2.set_title("Top-8 Ensembles Robustos", fontsize=11, fontweight="bold")
ax2.legend(fontsize=9)
ax2.grid(axis="y", alpha=0.3)
for bar in bars2:
    h = bar.get_height()
    ax2.text(bar.get_x()+bar.get_width()/2, h+0.2, f"{h:.1f}", ha="center", va="bottom", fontsize=7, fontweight="bold")

plt.tight_layout()
out_fig = TCC_ROOT / "figures" / "ensemble_robust_all_models.png"
fig.savefig(out_fig, dpi=150, bbox_inches="tight")
plt.close()
print(f"✅ Figura salva: {out_fig}")

print("\n=== TOP-10 ENSEMBLES ROBUSTOS ===")
print(df[["label","strategy","test_auc","test_d_auc","delta_auc","score","grade"]].head(10).to_string(index=False))
