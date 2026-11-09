from pathlib import Path
import itertools
import numpy as np
import pandas as pd
from sklearn.metrics import (
    roc_auc_score, accuracy_score, f1_score, precision_score, recall_score, confusion_matrix
)
from sklearn.linear_model import LogisticRegression

import os

MODELS_ROOT = Path(os.environ.get("TCC_MODELS_ROOT", os.environ.get("CISIA_MODELS_ROOT", "/media/ssd2/lucas.ocunha/models-tcc")))
if not MODELS_ROOT.exists():
    for candidate in [
        Path(f"/projects/models/{os.environ.get('USER', 'lucas.ocunha')}/models-tcc"),
        Path(f"/projects/models/{os.environ.get('USER', 'lucas.ocunha')}/faceforgery"),
        Path("/media/ssd2/lucas.ocunha/models-tcc"),
    ]:
        if candidate.exists():
            MODELS_ROOT = candidate
            break

PROJECT_ROOT = Path(os.environ.get("TCC_PROJECT_ROOT", Path(__file__).resolve().parent.parent))
OUT_DIR = PROJECT_ROOT / "results" / "mostrar_rayson"
TABLES_DIR = PROJECT_ROOT / "tables"
OUT_DIR.mkdir(parents=True, exist_ok=True)
TABLES_DIR.mkdir(parents=True, exist_ok=True)

FAMILIES = ["clip", "dino", "vit", "resnet", "mobilenet", "xception"]
FAMILY_NAMES = {
    "clip": "CLIP ViT-B/16",
    "dino": "DINO (ConvNeXt-B)",
    "vit": "ViT-B/16",
    "resnet": "ResNet-18",
    "mobilenet": "MobileNetV3",
    "xception": "Xception",
}
SEEDS = [42, 123, 2024, 7, 2025]

def safe_auc(y_true, probs):
    try:
        return float(roc_auc_score(y_true, probs))
    except Exception:
        return 0.5

def find_optimal_threshold(val_probs, val_y):
    best_t, best_acc = 0.5, 0.0
    for t in np.linspace(0.01, 0.99, 100):
        acc = accuracy_score(val_y, (val_probs >= t).astype(int))
        if acc > best_acc:
            best_acc, best_t = acc, t
    return float(best_t)

def full_metrics(y_true, probs, threshold=0.5):
    preds = (probs >= threshold).astype(int)
    auc = safe_auc(y_true, probs)
    acc = float(accuracy_score(y_true, preds))
    f1 = float(f1_score(y_true, preds, zero_division=0))
    prec = float(precision_score(y_true, preds, zero_division=0))
    rec = float(recall_score(y_true, preds, zero_division=0))
    tn, fp, fn, tp = confusion_matrix(y_true, preds, labels=[0, 1]).ravel()
    spec = float(tn / (tn + fp)) if (tn + fp) > 0 else 0.0
    return {
        "auc": auc, "acc": acc, "f1": f1, "prec": prec, "rec": rec, "spec": spec,
        "tp": int(tp), "fp": int(fp), "fn": int(fn), "tn": int(tn), "threshold": threshold
    }

def fuse_probs(probs_list, strategy):
    stack = np.stack(probs_list, axis=1)
    if strategy == "mean":
        return stack.mean(axis=1)
    elif strategy == "geometric":
        eps = 1e-7
        return np.exp(np.mean(np.log(np.clip(stack, eps, 1.0 - eps)), axis=1))
    elif strategy == "max":
        return stack.max(axis=1)
    raise ValueError(f"Unknown strategy: {strategy}")

def load_all_predictions():
    print("Loading predictions for all 30 models (6 families x 5 seeds)...")
    data = {}
    for fam in FAMILIES:
        data[fam] = {}
        for s in SEEDS:
            res_dir = MODELS_ROOT / fam / "srm" / "finetune_robust" / f"seed_{s}" / "results"
            val = np.load(res_dir / "outputs_val.npz")
            test = np.load(res_dir / "outputs_test.npz")
            test_d = np.load(res_dir / "outputs_test_d.npz")
            df40 = np.load(res_dir / "outputs_df40.npz")
            celeb = np.load(res_dir / "outputs_celeb_df.npz")
            thr = float(pd.read_csv(res_dir / "metrics_val.csv").iloc[0]["threshold"])
            
            data[fam][s] = {
                "val": val["probs"],
                "test": test["probs"],
                "test_d": test_d["probs"],
                "df40": df40["probs"],
                "celeb_frame": celeb["p_fake"],
                "celeb_video": celeb["video_p"],
                "threshold": thr,
            }
    
    # Also get ground truth vectors (identical across models)
    sample_res = MODELS_ROOT / "clip" / "srm" / "finetune_robust" / "seed_42" / "results"
    ground_truth = {
        "val": np.load(sample_res / "outputs_val.npz")["y_true"],
        "test": np.load(sample_res / "outputs_test.npz")["y_true"],
        "test_d": np.load(sample_res / "outputs_test_d.npz")["y_true"],
        "df40": np.load(sample_res / "outputs_df40.npz")["y_true"],
        "celeb_frame": np.load(sample_res / "outputs_celeb_df.npz")["y_true"],
        "celeb_video": np.load(sample_res / "outputs_celeb_df.npz")["video_y"],
    }
    return data, ground_truth

def consolidate_5seeds_summary():
    print("\nConsolidating 5-seed ensemble tables...")
    seed_dfs = []
    for s in SEEDS:
        p = OUT_DIR / f"ensembles_srm_seed_{s}.csv"
        if not p.exists():
            print(f"Waiting for {p}...")
            return None
        df = pd.read_csv(p)
        df["seed"] = s
        seed_dfs.append(df)
        
    combined = pd.concat(seed_dfs, ignore_index=True)
    combined.to_csv(OUT_DIR / "ensembles_srm_5seeds_raw_all_runs.csv", index=False)
    
    # Group by composition, models, strategy, k
    group_cols = ["composition", "strategy", "k"]
    metric_cols = [
        "test_auc", "test_acc", "test_f1",
        "test_d_auc", "test_d_acc", "test_d_f1", "delta_auc",
        "df40_auc", "df40_acc",
        "celeb_video_auc", "celeb_video_acc", "threshold"
    ]
    
    records = []
    for (comp, strat, k), grp in combined.groupby(group_cols):
        rec = {
            "composition": comp,
            "strategy": strat,
            "k": k,
            "num_seeds": len(grp),
        }
        for col in metric_cols:
            rec[f"{col}_mean"] = float(grp[col].mean())
            rec[f"{col}_std"] = float(grp[col].std()) if len(grp) > 1 else 0.0
        records.append(rec)
        
    summary_df = pd.DataFrame(records)
    summary_df.sort_values(by="test_d_auc_mean", ascending=False, inplace=True)
    summary_df.to_csv(OUT_DIR / "ensembles_srm_5seeds_consolidated.csv", index=False)
    summary_df.to_csv(TABLES_DIR / "ensembles_srm_5seeds_summary.csv", index=False)
    print(f"Summary saved: {len(summary_df)} ensemble configurations across all 5 seeds.")
    return summary_df

def compute_mega_ensembles(data, gt):
    print("\nComputing Mega-Ensembles of ALL models...")
    records = []
    
    # 1. Intra-architecture cross-seed bagging: for each family, average its 5 seeds
    family_meta_probs = {split: {} for split in gt.keys()}
    for fam in FAMILIES:
        for split in gt.keys():
            fam_stack = [data[fam][s][split] for s in SEEDS]
            family_meta_probs[split][fam] = np.mean(fam_stack, axis=0)
            
    # Also evaluate the 5-seed bagged model for each architecture alone
    for fam in FAMILIES:
        val_p = family_meta_probs["val"][fam]
        thr = find_optimal_threshold(val_p, gt["val"])
        m_test = full_metrics(gt["test"], family_meta_probs["test"][fam], thr)
        m_test_d = full_metrics(gt["test_d"], family_meta_probs["test_d"][fam], thr)
        m_df40 = full_metrics(gt["df40"], family_meta_probs["df40"][fam], thr)
        m_celeb_fr = full_metrics(gt["celeb_frame"], family_meta_probs["celeb_frame"][fam], thr)
        m_celeb_vd = full_metrics(gt["celeb_video"], family_meta_probs["celeb_video"][fam], thr)
        records.append({
            "ensemble_name": f"Bagged 5-Seeds: {FAMILY_NAMES[fam]}",
            "type": "bagged_architecture",
            "k_architectures": 1,
            "total_models": 5,
            "strategy": "mean_seeds",
            "threshold": thr,
            "test_auc": m_test["auc"], "test_acc": m_test["acc"], "test_f1": m_test["f1"],
            "test_d_auc": m_test_d["auc"], "test_d_acc": m_test_d["acc"], "test_d_f1": m_test_d["f1"],
            "delta_auc": m_test_d["auc"] - m_test["auc"],
            "df40_auc": m_df40["auc"], "df40_acc": m_df40["acc"], "df40_f1": m_df40["f1"],
            "celeb_frame_auc": m_celeb_fr["auc"], "celeb_frame_acc": m_celeb_fr["acc"],
            "celeb_video_auc": m_celeb_vd["auc"], "celeb_video_acc": m_celeb_vd["acc"], "celeb_video_f1": m_celeb_vd["f1"],
        })

    # 2. Mega-Ensemble: ALL 6 Architectures (using their 5-seed bagged predictions)
    # Strategies: geometric, mean, stacking
    for strat in ["geometric", "mean", "stacking"]:
        arch_list = [family_meta_probs["val"][f] for f in FAMILIES]
        val_stack = np.stack(arch_list, axis=1)
        
        if strat == "stacking":
            clf = LogisticRegression(C=1.0, max_iter=1000)
            clf.fit(val_stack, gt["val"].astype(int))
            fused_val = clf.predict_proba(val_stack)[:, 1]
            fused_test = clf.predict_proba(np.stack([family_meta_probs["test"][f] for f in FAMILIES], axis=1))[:, 1]
            fused_test_d = clf.predict_proba(np.stack([family_meta_probs["test_d"][f] for f in FAMILIES], axis=1))[:, 1]
            fused_df40 = clf.predict_proba(np.stack([family_meta_probs["df40"][f] for f in FAMILIES], axis=1))[:, 1]
            fused_celeb_fr = clf.predict_proba(np.stack([family_meta_probs["celeb_frame"][f] for f in FAMILIES], axis=1))[:, 1]
            fused_celeb_vd = clf.predict_proba(np.stack([family_meta_probs["celeb_video"][f] for f in FAMILIES], axis=1))[:, 1]
        else:
            fused_val = fuse_probs(arch_list, strat)
            fused_test = fuse_probs([family_meta_probs["test"][f] for f in FAMILIES], strat)
            fused_test_d = fuse_probs([family_meta_probs["test_d"][f] for f in FAMILIES], strat)
            fused_df40 = fuse_probs([family_meta_probs["df40"][f] for f in FAMILIES], strat)
            fused_celeb_fr = fuse_probs([family_meta_probs["celeb_frame"][f] for f in FAMILIES], strat)
            fused_celeb_vd = fuse_probs([family_meta_probs["celeb_video"][f] for f in FAMILIES], strat)
            
        thr = find_optimal_threshold(fused_val, gt["val"])
        m_test = full_metrics(gt["test"], fused_test, thr)
        m_test_d = full_metrics(gt["test_d"], fused_test_d, thr)
        m_df40 = full_metrics(gt["df40"], fused_df40, thr)
        m_celeb_fr = full_metrics(gt["celeb_frame"], fused_celeb_fr, thr)
        m_celeb_vd = full_metrics(gt["celeb_video"], fused_celeb_vd, thr)
        
        records.append({
            "ensemble_name": "Mega-Ensemble (All 6 Architectures x 5 Seeds)",
            "type": "all_6_architectures_bagged",
            "k_architectures": 6,
            "total_models": 30,
            "strategy": strat,
            "threshold": thr,
            "test_auc": m_test["auc"], "test_acc": m_test["acc"], "test_f1": m_test["f1"],
            "test_d_auc": m_test_d["auc"], "test_d_acc": m_test_d["acc"], "test_d_f1": m_test_d["f1"],
            "delta_auc": m_test_d["auc"] - m_test["auc"],
            "df40_auc": m_df40["auc"], "df40_acc": m_df40["acc"], "df40_f1": m_df40["f1"],
            "celeb_frame_auc": m_celeb_fr["auc"], "celeb_frame_acc": m_celeb_fr["acc"],
            "celeb_video_auc": m_celeb_vd["auc"], "celeb_video_acc": m_celeb_vd["acc"], "celeb_video_f1": m_celeb_vd["f1"],
        })

    # 3. Direct Flattened Ensemble of all 30 individual models simultaneously
    for strat in ["geometric", "mean", "stacking"]:
        all_val_list = [data[f][s]["val"] for f in FAMILIES for s in SEEDS]
        val_stack_30 = np.stack(all_val_list, axis=1)
        
        if strat == "stacking":
            clf = LogisticRegression(C=1.0, max_iter=1000)
            clf.fit(val_stack_30, gt["val"].astype(int))
            fused_val = clf.predict_proba(val_stack_30)[:, 1]
            fused_test = clf.predict_proba(np.stack([data[f][s]["test"] for f in FAMILIES for s in SEEDS], axis=1))[:, 1]
            fused_test_d = clf.predict_proba(np.stack([data[f][s]["test_d"] for f in FAMILIES for s in SEEDS], axis=1))[:, 1]
            fused_df40 = clf.predict_proba(np.stack([data[f][s]["df40"] for f in FAMILIES for s in SEEDS], axis=1))[:, 1]
            fused_celeb_fr = clf.predict_proba(np.stack([data[f][s]["celeb_frame"] for f in FAMILIES for s in SEEDS], axis=1))[:, 1]
            fused_celeb_vd = clf.predict_proba(np.stack([data[f][s]["celeb_video"] for f in FAMILIES for s in SEEDS], axis=1))[:, 1]
        else:
            fused_val = fuse_probs(all_val_list, strat)
            fused_test = fuse_probs([data[f][s]["test"] for f in FAMILIES for s in SEEDS], strat)
            fused_test_d = fuse_probs([data[f][s]["test_d"] for f in FAMILIES for s in SEEDS], strat)
            fused_df40 = fuse_probs([data[f][s]["df40"] for f in FAMILIES for s in SEEDS], strat)
            fused_celeb_fr = fuse_probs([data[f][s]["celeb_frame"] for f in FAMILIES for s in SEEDS], strat)
            fused_celeb_vd = fuse_probs([data[f][s]["celeb_video"] for f in FAMILIES for s in SEEDS], strat)

        thr = find_optimal_threshold(fused_val, gt["val"])
        m_test = full_metrics(gt["test"], fused_test, thr)
        m_test_d = full_metrics(gt["test_d"], fused_test_d, thr)
        m_df40 = full_metrics(gt["df40"], fused_df40, thr)
        m_celeb_fr = full_metrics(gt["celeb_frame"], fused_celeb_fr, thr)
        m_celeb_vd = full_metrics(gt["celeb_video"], fused_celeb_vd, thr)

        records.append({
            "ensemble_name": "Direct Flat 30-Model Fusion",
            "type": "direct_30_models",
            "k_architectures": 6,
            "total_models": 30,
            "strategy": strat,
            "threshold": thr,
            "test_auc": m_test["auc"], "test_acc": m_test["acc"], "test_f1": m_test["f1"],
            "test_d_auc": m_test_d["auc"], "test_d_acc": m_test_d["acc"], "test_d_f1": m_test_d["f1"],
            "delta_auc": m_test_d["auc"] - m_test["auc"],
            "df40_auc": m_df40["auc"], "df40_acc": m_df40["acc"], "df40_f1": m_df40["f1"],
            "celeb_frame_auc": m_celeb_fr["auc"], "celeb_frame_acc": m_celeb_fr["acc"],
            "celeb_video_auc": m_celeb_vd["auc"], "celeb_video_acc": m_celeb_vd["acc"], "celeb_video_f1": m_celeb_vd["f1"],
        })

    # 4. Top Specialized Ensembles (e.g. DINO + CLIP, DINO + MobileNet, Transformers vs CNNs)
    top_combos = [
        ("DINO + CLIP (Vision-Language & Self-Supervised)", ["dino", "clip"]),
        ("DINO + MobileNet (Top Celeb-DF Hybrid)", ["dino", "mobilenet"]),
        ("Transformers Committee (CLIP + DINO + ViT)", ["clip", "dino", "vit"]),
        ("CNNs Committee (ResNet + MobileNet + Xception)", ["resnet", "mobilenet", "xception"]),
        ("DINO + CLIP + Xception (Multi-Spectral Champion)", ["dino", "clip", "xception"]),
    ]
    for combo_name, sub_fams in top_combos:
        for strat in ["geometric", "mean", "stacking"]:
            arch_list = [family_meta_probs["val"][f] for f in sub_fams]
            val_stack_sub = np.stack(arch_list, axis=1)
            
            if strat == "stacking":
                clf = LogisticRegression(C=1.0, max_iter=1000)
                clf.fit(val_stack_sub, gt["val"].astype(int))
                fused_val = clf.predict_proba(val_stack_sub)[:, 1]
                fused_test = clf.predict_proba(np.stack([family_meta_probs["test"][f] for f in sub_fams], axis=1))[:, 1]
                fused_test_d = clf.predict_proba(np.stack([family_meta_probs["test_d"][f] for f in sub_fams], axis=1))[:, 1]
                fused_df40 = clf.predict_proba(np.stack([family_meta_probs["df40"][f] for f in sub_fams], axis=1))[:, 1]
                fused_celeb_fr = clf.predict_proba(np.stack([family_meta_probs["celeb_frame"][f] for f in sub_fams], axis=1))[:, 1]
                fused_celeb_vd = clf.predict_proba(np.stack([family_meta_probs["celeb_video"][f] for f in sub_fams], axis=1))[:, 1]
            else:
                fused_val = fuse_probs(arch_list, strat)
                fused_test = fuse_probs([family_meta_probs["test"][f] for f in sub_fams], strat)
                fused_test_d = fuse_probs([family_meta_probs["test_d"][f] for f in sub_fams], strat)
                fused_df40 = fuse_probs([family_meta_probs["df40"][f] for f in sub_fams], strat)
                fused_celeb_fr = fuse_probs([family_meta_probs["celeb_frame"][f] for f in sub_fams], strat)
                fused_celeb_vd = fuse_probs([family_meta_probs["celeb_video"][f] for f in sub_fams], strat)

            thr = find_optimal_threshold(fused_val, gt["val"])
            m_test = full_metrics(gt["test"], fused_test, thr)
            m_test_d = full_metrics(gt["test_d"], fused_test_d, thr)
            m_df40 = full_metrics(gt["df40"], fused_df40, thr)
            m_celeb_fr = full_metrics(gt["celeb_frame"], fused_celeb_fr, thr)
            m_celeb_vd = full_metrics(gt["celeb_video"], fused_celeb_vd, thr)

            records.append({
                "ensemble_name": combo_name,
                "type": "specialized_meta_ensemble",
                "k_architectures": len(sub_fams),
                "total_models": len(sub_fams) * len(SEEDS),
                "strategy": strat,
                "threshold": thr,
                "test_auc": m_test["auc"], "test_acc": m_test["acc"], "test_f1": m_test["f1"],
                "test_d_auc": m_test_d["auc"], "test_d_acc": m_test_d["acc"], "test_d_f1": m_test_d["f1"],
                "delta_auc": m_test_d["auc"] - m_test["auc"],
                "df40_auc": m_df40["auc"], "df40_acc": m_df40["acc"], "df40_f1": m_df40["f1"],
                "celeb_frame_auc": m_celeb_fr["auc"], "celeb_frame_acc": m_celeb_fr["acc"],
                "celeb_video_auc": m_celeb_vd["auc"], "celeb_video_acc": m_celeb_vd["acc"], "celeb_video_f1": m_celeb_vd["f1"],
            })

    mega_df = pd.DataFrame(records)
    mega_df.sort_values(by="test_d_auc", ascending=False, inplace=True)
    mega_df.to_csv(OUT_DIR / "mega_ensembles_srm_all_models.csv", index=False)
    mega_df.to_csv(TABLES_DIR / "mega_ensembles_srm_all_models.csv", index=False)
    print(f"Mega-Ensembles saved: {len(mega_df)} configurations.")
    return mega_df

def generate_tabela9_markdown():
    p_5s = TABLES_DIR / "ensembles_srm_5seeds_summary.csv"
    p_mega = TABLES_DIR / "mega_ensembles_srm_all_models.csv"
    if not p_5s.exists() or not p_mega.exists():
        return
    df_5s = pd.read_csv(p_5s)
    df_mega = pd.read_csv(p_mega)
    
    lines = []
    lines.append("# Tabela 9: Resultados Consolidados de Ensembles SRM (Matriz Canônica de 5 Sementes)")
    lines.append("")
    lines.append("**Documento:** `tabela9-ensemble-srm-5seeds.md`  ")
    lines.append("**Ambiente:** Dual NVIDIA GeForce RTX 3090 | Workstation Local (`sicret2`)  ")
    lines.append("**Sementes Avaliadas:** 5 Sementes Canônicas (`42`, `123`, `2024`, `7`, `2025`)  ")
    lines.append("**Total de Modelos Treinados:** 30 modelos (6 arquiteturas × 5 seeds)  ")
    lines.append("**Estratégias de Fusão:** `geometric` (Média Geométrica), `stacking` (Regressão Logística em Validação), `mean` (Média Aritmética), `max` (Max Pooling)  ")
    lines.append("")
    lines.append("---")
    lines.append("")
    lines.append("## 1. 🌟 Mega-Ensembles Globais (Fusão de Metas e dos 30 Modelos)")
    lines.append("")
    lines.append("| Composição / Comitê | Estratégia | N° Modelos | Test Limpo (AUC) | Test Acc | Test-D Difícil (AUC) | Test-D Acc | ΔAUC | DF-40 (AUC) | Celeb-DF Vídeo (AUC) |")
    lines.append("| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |")
    
    for _, r in df_mega.head(10).iterrows():
        name = r['ensemble_name']
        strat = r['strategy']
        n_mod = f"{r['total_models']}x"
        t_auc = f"{r['test_auc']*100:.2f}%"
        t_acc = f"{r['test_acc']*100:.2f}%"
        td_auc = f"**{r['test_d_auc']*100:.2f}%**"
        td_acc = f"{r['test_d_acc']*100:.2f}%"
        d_auc = f"{r['delta_auc']*100:+.2f}%"
        df40 = f"{r['df40_auc']*100:.2f}%"
        cvd = f"**{r['celeb_video_auc']*100:.2f}%**"
        lines.append(f"| **{name}** | `{strat}` | {n_mod} | {t_auc} | {t_acc} | {td_auc} | {td_acc} | {d_auc} | {df40} | {cvd} |")
    
    lines.append("")
    lines.append("---")
    lines.append("")
    lines.append("## 2. 🥇 Top 15 Ensembles Canônicos (Média das 5 Sementes ± Desvio Padrão)")
    lines.append("")
    lines.append("Ordenado por **Test-D AUC** (Robustez sob corrupções não vistas):")
    lines.append("")
    lines.append("| Rank | Composição da Fusão | Estratégia | K | Test Limpo (AUC) | Test-D Difícil (AUC) | ΔAUC | DF-40 Cross-Data (AUC) | Celeb-DF Vídeo (AUC) |")
    lines.append("| :---: | :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: |")
    
    sub15 = df_5s[df_5s["k"] > 1].head(15)
    for idx, (_, r) in enumerate(sub15.iterrows(), start=1):
        medal = "🥇" if idx == 1 else ("🥈" if idx == 2 else ("🥉" if idx == 3 else str(idx)))
        comp = r['composition']
        strat = r['strategy']
        k = f"{r['k']}x"
        t_auc = f"{r['test_auc_mean']*100:.2f} ± {r['test_auc_std']*100:.2f}%"
        td_auc = f"**{r['test_d_auc_mean']*100:.2f} ± {r['test_d_auc_std']*100:.2f}%**"
        d_auc = f"{r['delta_auc_mean']*100:+.2f}%"
        df40 = f"{r['df40_auc_mean']*100:.2f} ± {r['df40_auc_std']*100:.2f}%"
        cvd = f"**{r['celeb_video_auc_mean']*100:.2f} ± {r['celeb_video_auc_std']*100:.2f}%**"
        lines.append(f"| {medal} | **{comp}** | `{strat}` | {k} | {t_auc} | {td_auc} | {d_auc} | {df40} | {cvd} |")
    
    lines.append("")
    lines.append("---")
    lines.append("")
    lines.append("## 3. 🎯 Comparativo do Comitê de Todos os 6 Modelos (K=6) nas 5 Sementes")
    lines.append("")
    lines.append("| Estratégia | Test Limpo (AUC) | Test-D Difícil (AUC) | ΔAUC | DF-40 (AUC) | Celeb-DF Vídeo (AUC) |")
    lines.append("| :---: | :---: | :---: | :---: | :---: | :---: |")
    k6_sub = df_5s[df_5s["k"] == 6]
    for _, r in k6_sub.iterrows():
        strat = r['strategy']
        t_auc = f"{r['test_auc_mean']*100:.2f} ± {r['test_auc_std']*100:.2f}%"
        td_auc = f"**{r['test_d_auc_mean']*100:.2f} ± {r['test_d_auc_std']*100:.2f}%**"
        d_auc = f"{r['delta_auc_mean']*100:+.2f}%"
        df40 = f"{r['df40_auc_mean']*100:.2f} ± {r['df40_auc_std']*100:.2f}%"
        cvd = f"**{r['celeb_video_auc_mean']*100:.2f} ± {r['celeb_video_auc_std']*100:.2f}%**"
        lines.append(f"| `{strat}` | {t_auc} | {td_auc} | {d_auc} | {df40} | {cvd} |")
    
    out_md = OUT_DIR / "tabela9-ensemble-srm-5seeds.md"
    out_md.write_text("\n".join(lines), encoding="utf-8")
    print(f"Markdown salvo em: {out_md}")

if __name__ == "__main__":
    summary_df = consolidate_5seeds_summary()
    data, gt = load_all_predictions()
    mega_df = compute_mega_ensembles(data, gt)
    generate_tabela9_markdown()
    print("\nDONE!")

