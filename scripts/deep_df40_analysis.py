"""Análise profunda e exaustiva do benchmark DF40 (DeepFake-40).

Investiga:
1. Desempenho por Técnica Generativa Individual (24 métodos fakes vs reais)
2. Modelo Campeão Individual e Ensemble Campeão para cada técnica
3. Ganho relativo de Fourier (concat vs none) por técnica
4. Matriz de Correlação de Predições e Diversidade entre Famílias
5. Ranking de Evasão Forense (técnicas mais evasivas vs mais fáceis de detectar)
6. Comparativo cruzado entre gerações (StyleGAN2 vs 3 vs XL; DiT vs SiT vs UNet; DFL vs UniFace vs MobileSwap)
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score, roc_auc_score

from src.data.paths import models_root


def safe_auc(y_true: np.ndarray, y_score: np.ndarray) -> float:
    try:
        if len(np.unique(y_true)) < 2:
            return 50.0
        return float(roc_auc_score(y_true, y_score)) * 100.0
    except Exception:
        return 50.0


def compute_metrics(y_true: np.ndarray, probs: np.ndarray, threshold: float = 0.5) -> dict:
    y_pred = (probs >= threshold).astype(int)
    acc = float(accuracy_score(y_true, y_pred)) * 100.0
    prec = float(precision_score(y_true, y_pred, zero_division=0)) * 100.0
    rec = float(recall_score(y_true, y_pred, zero_division=0)) * 100.0
    f1 = float(f1_score(y_true, y_pred, zero_division=0)) * 100.0
    auc = safe_auc(y_true, probs)
    return {
        "auc": round(auc, 2),
        "acc": round(acc, 2),
        "f1": round(f1, 2),
        "precision": round(prec, 2),
        "recall": round(rec, 2),
    }


def load_model_probs(root: Path, family: str, mode: str, seed: int | str) -> np.ndarray | None:
    p = root / family / mode / "finetune" / f"seed_{seed}" / "results" / "outputs_df40.npz"
    if not p.exists():
        return None
    d = np.load(p)
    return d["probs"]


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--manifest-csv", type=Path, default=Path("data/df40/test.csv"))
    parser.add_argument("--output-dir", type=Path, default=Path("tables"))
    args = parser.parse_args()

    args.output_dir.mkdir(parents=True, exist_ok=True)
    manifest = pd.read_csv(args.manifest_csv)
    y_true_all = manifest["target"].values
    real_mask = (manifest["target"] == 0).values

    root = models_root()
    families = ["clip", "dino", "vit", "resnet", "mobilenet", "xception"]
    modes = ["none", "concat", "concat_frequency"]
    seeds_std = [7, 42, 123, 2024, 2025]
    all_seeds = [7, 42, 123, 2024, 2025, 987]

    # 1. Carregar predições disponíveis
    print("[1/6] Carregando predições dos modelos em memória...")
    pool: dict[str, np.ndarray] = {}
    for fam in families:
        for mode in modes:
            for s in all_seeds:
                p = load_model_probs(root, fam, mode, s)
                if p is not None:
                    pool[f"{fam}_{mode}_s{s}"] = p

    print(f"Total de modelos disponíveis carregados: {len(pool)}")

    # Construir médias de seeds (Deep Ensembles por família/modo)
    for fam in families:
        for mode in modes:
            sub = [pool[f"{fam}_{mode}_s{s}"] for s in seeds_std if f"{fam}_{mode}_s{s}" in pool]
            if sub:
                pool[f"{fam}_{mode}_mean"] = np.mean(sub, axis=0)

    # Definir Ensembles Campeões
    ensembles = {
        "Trio_Campeao": np.mean([pool["clip_none_s987"], pool["clip_concat_s2025"], pool["resnet_concat_s123"]], axis=0),
        "Quarteto_Campeao": np.mean([pool["clip_none_s987"], pool["clip_concat_s2025"], pool["resnet_concat_s123"], pool["dino_concat_s123"]], axis=0),
        "CLIP_Rob_RESNET_Concat": np.mean([pool["clip_none_s987"], pool["resnet_concat_s123"]], axis=0),
        "CLIP_DINO_Robusto": np.mean([pool["clip_none_s987"], pool["dino_none_s987"]], axis=0),
        "Super_Top5": np.mean([pool["clip_none_s987"], pool["dino_none_s987"], pool["resnet_concat_s123"], pool["dino_concat_s123"], pool["xception_concat_s42"]], axis=0),
        "Deep_Resnet_Concat": pool["resnet_concat_mean"],
        "Deep_Clip_None": pool["clip_none_mean"],
        "Todos_6_Padrao": np.mean([pool[f"{f}_none_s42"] for f in families if f"{f}_none_s42" in pool], axis=0),
    }

    # 2. Identificar métodos sintéticos únicos
    fake_methods = sorted(manifest[manifest["target"] == 1]["method"].unique())
    paradigms = sorted(manifest[manifest["target"] == 1]["paradigm"].unique())

    print(f"[2/6] Analisando {len(fake_methods)} técnicas fakes em {len(paradigms)} paradigmas...")

    # Mapeamento técnica -> paradigma
    method_to_paradigm = manifest[manifest["target"] == 1].groupby("method")["paradigm"].first().to_dict()

    # 3. Análise Detalhada por Técnica
    technique_records = []
    fourier_gain_records = []

    for m in fake_methods:
        para = method_to_paradigm[m]
        fake_mask = (manifest["method"] == m).values
        # Subconjunto: todos os reais + os fakes da técnica m
        sub_idx = real_mask | fake_mask
        sub_y = y_true_all[sub_idx]

        # Avaliar todos os modelos individuais disponíveis
        model_scores = {}
        for k, probs in pool.items():
            if k.endswith("_mean"):
                continue  # tratamos depois
            sub_probs = probs[sub_idx]
            model_scores[k] = safe_auc(sub_y, sub_probs)

        # Melhor modelo individual
        best_model_key = max(model_scores, key=model_scores.get)
        best_model_auc = model_scores[best_model_key]

        # Avaliar Ensembles
        ens_scores = {}
        for ens_name, ens_probs in ensembles.items():
            sub_probs = ens_probs[sub_idx]
            ens_scores[ens_name] = safe_auc(sub_y, sub_probs)

        best_ens_key = max(ens_scores, key=ens_scores.get)
        best_ens_auc = ens_scores[best_ens_key]

        # Métricas do Trio Campeão especificamente
        trio_m = compute_metrics(sub_y, ensembles["Trio_Campeao"][sub_idx])

        # Média de todos os modelos (medida de dificuldade da técnica)
        all_auc_vals = list(model_scores.values())
        avg_technique_auc = float(np.mean(all_auc_vals))

        technique_records.append({
            "Paradigma": para,
            "Tecnica": m,
            "N_Amostras_Fake": int(fake_mask.sum()),
            "Dificuldade_Media_AUC": round(avg_technique_auc, 2),
            "Melhor_Modelo_Individual": best_model_key,
            "Melhor_Indiv_AUC": round(best_model_auc, 2),
            "Melhor_Ensemble": best_ens_key,
            "Melhor_Ens_AUC": round(best_ens_auc, 2),
            "Trio_Campeao_AUC": trio_m["auc"],
            "Trio_Campeao_Acc": trio_m["acc"],
            "Trio_Campeao_Recall": trio_m["recall"],
            "CLIP_none_s987_AUC": round(model_scores.get("clip_none_s987", 50.0), 2),
            "RESNET_concat_s123_AUC": round(model_scores.get("resnet_concat_s123", 50.0), 2),
            "DINO_concat_s123_AUC": round(model_scores.get("dino_concat_s123", 50.0), 2),
            "XCEPTION_concat_s42_AUC": round(model_scores.get("xception_concat_s42", 50.0), 2),
        })

        # Análise do Ganho de Fourier por Família nesta técnica
        for fam in families:
            none_keys = [f"{fam}_none_s{s}" for s in seeds_std if f"{fam}_none_s{s}" in pool]
            concat_keys = [f"{fam}_concat_s{s}" for s in seeds_std if f"{fam}_concat_s{s}" in pool]
            if none_keys and concat_keys:
                auc_none = np.mean([model_scores[k] for k in none_keys])
                auc_concat = np.mean([model_scores[k] for k in concat_keys])
                delta = auc_concat - auc_none
                fourier_gain_records.append({
                    "Paradigma": para,
                    "Tecnica": m,
                    "Familia": fam.upper(),
                    "AUC_Espacial_none": round(auc_none, 2),
                    "AUC_Hibrido_concat": round(auc_concat, 2),
                    "Delta_AUC_Fourier": round(delta, 2),
                })

    df_tech = pd.DataFrame(technique_records).sort_values("Dificuldade_Media_AUC")
    out_tech = args.output_dir / "df40_techniques_rankings.csv"
    df_tech.to_csv(out_tech, index=False)
    print(f"Salvo: {out_tech}")

    df_fourier = pd.DataFrame(fourier_gain_records)
    out_fourier = args.output_dir / "df40_fourier_gain_per_technique.csv"
    df_fourier.to_csv(out_fourier, index=False)
    print(f"Salvo: {out_fourier}")

    # 4. Matriz de Paradigma vs Família Arquitetural
    print("[3/6] Calculando Matriz Paradigma vs Família Arquitetural...")
    matrix_records = []
    for para in paradigms:
        p_mask = (manifest["paradigm"] == para).values
        sub_idx = real_mask | p_mask
        sub_y = y_true_all[sub_idx]

        rec = {"Paradigma": para}
        for fam in families:
            # none
            keys_none = [f"{fam}_none_s{s}" for s in seeds_std if f"{fam}_none_s{s}" in pool]
            if keys_none:
                rec[f"{fam.upper()}_none"] = round(float(np.mean([safe_auc(sub_y, pool[k][sub_idx]) for k in keys_none])), 2)
            # concat
            keys_concat = [f"{fam}_concat_s{s}" for s in seeds_std if f"{fam}_concat_s{s}" in pool]
            if keys_concat:
                rec[f"{fam.upper()}_concat"] = round(float(np.mean([safe_auc(sub_y, pool[k][sub_idx]) for k in keys_concat])), 2)

        matrix_records.append(rec)

    df_matrix = pd.DataFrame(matrix_records)
    out_matrix = args.output_dir / "df40_paradigm_architecture_matrix.csv"
    df_matrix.to_csv(out_matrix, index=False)
    print(f"Salvo: {out_matrix}")

    # 5. Análise de Correlação e Diversidade dos Detectores Campeões
    print("[4/6] Calculando Matriz de Correlação entre Detectores Campeões...")
    champ_keys = [
        "clip_none_s987",
        "clip_concat_s2025",
        "resnet_concat_s123",
        "dino_concat_s123",
        "dino_none_s987",
        "xception_concat_s42",
        "mobilenet_none_s987",
        "vit_none_s987",
    ]
    corr_data = {}
    for k in champ_keys:
        if k in pool:
            corr_data[k] = pool[k]

    df_corr_input = pd.DataFrame(corr_data)
    corr_matrix = df_corr_input.corr(method="pearson").round(4)
    out_corr = args.output_dir / "df40_champions_correlation_matrix.csv"
    corr_matrix.to_csv(out_corr)
    print(f"Salvo: {out_corr}")

    # 6. Easiest vs Hardest Techniques Ranking
    print("[5/6] Gerando ranking de Vulnerabilidade e Evasão...")
    df_vulnerability = df_tech[["Paradigma", "Tecnica", "N_Amostras_Fake", "Dificuldade_Media_AUC", "Trio_Campeao_AUC", "Melhor_Indiv_AUC", "Melhor_Modelo_Individual"]].copy()
    df_vulnerability["Status"] = np.where(
        df_vulnerability["Dificuldade_Media_AUC"] < 65.0,
        "Altamente Evasivo (Crítico)",
        np.where(df_vulnerability["Dificuldade_Media_AUC"] < 75.0, "Moderadamente Evasivo", "Vulnerável à Detecção")
    )
    out_vuln = args.output_dir / "df40_evasion_and_vulnerability_ranking.csv"
    df_vulnerability.to_csv(out_vuln, index=False)
    print(f"Salvo: {out_vuln}")

    # 7. Comparativo Especial de Gerações da Mesma Família Generativa
    print("[6/6] Calculando Comparações Intra-Família Generativa...")
    comparisons = [
        ("GANs: StyleGAN2 vs StyleGAN3 vs StyleGANXL", ["StyleGAN2", "StyleGAN3", "StyleGANXL"]),
        ("Diffusion: DiT (Transformer) vs SiT (Interpolant) vs PixArt vs SD 2.1 vs DDIM", ["DiT", "SiT", "pixart", "sd2.1", "ddim"]),
        ("Face Swap: DeepFaceLab vs FaceSwap vs MobileSwap vs UniFace vs BlendFace", ["deepfacelab", "faceswap", "mobileswap", "uniface", "blendface"]),
        ("Editing/T2I: MidJourney vs WhichFaceIsReal vs StyleCLIP vs E4E", ["MidJourney", "whichfaceisreal", "styleclip", "e4e"]),
    ]

    comp_records = []
    for title, group_methods in comparisons:
        for m in group_methods:
            row = df_tech[df_tech["Tecnica"] == m]
            if not row.empty:
                r = row.iloc[0]
                comp_records.append({
                    "Grupo": title,
                    "Tecnica": m,
                    "Dificuldade_Media_AUC": r["Dificuldade_Media_AUC"],
                    "Trio_Campeao_AUC": r["Trio_Campeao_AUC"],
                    "Melhor_Indiv_AUC": r["Melhor_Indiv_AUC"],
                    "Melhor_Modelo": r["Melhor_Modelo_Individual"],
                })

    df_comp = pd.DataFrame(comp_records)
    out_comp = args.output_dir / "df40_generational_comparisons.csv"
    df_comp.to_csv(out_comp, index=False)
    print(f"Salvo: {out_comp}")

    print("\n=== TODAS AS ANÁLISES PROFUNDAS FORAM CONCLUÍDAS COM SUCESSO! ===")


if __name__ == "__main__":
    main()
