"""Avaliação sistemática dos melhores Ensembles nos benchmarks Cross-Dataset: DF-40 e Celeb-DF v2.

Carrega as predições pré-computadas (outputs_df40.npz e outputs_celeb_df.npz)
e avalia comitês sob múltiplas estratégias de fusão (mean, geom, max).
"""

from __future__ import annotations

import argparse
from pathlib import Path
import numpy as np
import pandas as pd
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score, roc_auc_score

from src.data.paths import models_root, output_root


def safe_auc(y_true: np.ndarray, y_score: np.ndarray) -> float:
    try:
        if len(np.unique(y_true)) < 2:
            return 0.5
        y_score = np.nan_to_num(y_score, nan=0.5, posinf=1.0, neginf=0.0)
        return float(roc_auc_score(y_true, y_score))
    except Exception:
        return 0.5


def compute_metrics(y_true: np.ndarray, probs: np.ndarray, threshold: float = 0.5) -> dict:
    probs = np.nan_to_num(probs, nan=0.5, posinf=1.0, neginf=0.0)
    y_pred = (probs >= threshold).astype(int)
    acc = float(accuracy_score(y_true, y_pred))
    f1 = float(f1_score(y_true, y_pred, zero_division=0))
    auc = safe_auc(y_true, probs)
    return {"auc": auc, "acc": acc, "f1": f1}


def fuse_predictions(probs_list: list[np.ndarray], strategy: str) -> np.ndarray:
    stack = np.stack(probs_list, axis=0)
    if strategy == "mean":
        return np.mean(stack, axis=0)
    elif strategy == "geom":
        eps = 1e-6
        clipped = np.clip(stack, eps, 1.0 - eps)
        return np.exp(np.mean(np.log(clipped), axis=0))
    elif strategy == "max":
        return np.max(stack, axis=0)
    else:
        raise ValueError(f"Estratégia desconhecida: {strategy}")


def load_model_data(root: Path, family: str, mode: str, regime: str, seed: int | str):
    s_str = f"seed_{seed}" if not str(seed).startswith("seed_") else str(seed)
    res_dir = root / family / mode / regime / s_str / "results"
    
    p_df40 = res_dir / "outputs_df40.npz"
    p_celeb = res_dir / "outputs_celeb_df.npz"
    
    data = {}
    if p_df40.exists():
        with np.load(p_df40) as d:
            data["df40_probs"] = d["probs"]
            data["df40_y"] = d["y_true"]
            
    if p_celeb.exists():
        with np.load(p_celeb) as d:
            data["celeb_p"] = d["p_fake"]
            data["celeb_y"] = d["y_true"]
            if "video_p" in d:
                data["celeb_vid_p"] = d["video_p"]
                data["celeb_vid_y"] = d["video_y"]
                
    return data if ("df40_probs" in data or "celeb_p" in data) else None


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--celeb-csv", type=Path, default=Path("data/celeb_df/test.csv"))
    parser.add_argument("--output-dir", type=Path, default=None)
    args = parser.parse_args()

    root = models_root()
    dest = Path(args.output_dir or output_root() / "results" / "tables")
    dest.mkdir(parents=True, exist_ok=True)

    celeb_meta = pd.read_csv(args.celeb_csv)
    celeb_vid_y = celeb_meta.groupby("video_id")["target"].first().to_numpy().astype(int)

    print("🔍 Carregando pool de predições dos modelos...", flush=True)

    # Dicionário de modelos disponíveis
    models_pool = {}

    families = ["clip", "dino", "vit", "resnet", "mobilenet", "xception"]
    canonical_seeds = [42, 123, 2024, 7, 2025]
    all_seeds = [42, 123, 2024, 7, 2025, 987]

    for fam in families:
        for mode in ["none", "concat", "srm", "dtcwt"]:
            for regime in ["finetune", "finetune_robust"]:
                for s in all_seeds:
                    key = f"{fam}_{mode}_{regime}_s{s}"
                    res = load_model_data(root, fam, mode, regime, s)
                    if res is not None:
                        models_pool[key] = res

    print(f"✅ Pool carregado com {len(models_pool)} modelos disponíveis.", flush=True)

    ensembles_def = [
        # --- A. Ensembles dos Modelos Robustos (Seed 987) ---
        ("Ensemble Robusto: CLIP + DINO (s987)", [
            "clip_none_finetune_robust_s987", "dino_none_finetune_robust_s987"
        ]),
        ("Ensemble Robusto: CLIP + DINO + VIT (s987)", [
            "clip_none_finetune_robust_s987", "dino_none_finetune_robust_s987", "vit_none_finetune_robust_s987"
        ]),
        ("Ensemble Robusto: CLIP + DINO + RESNET (s987)", [
            "clip_none_finetune_robust_s987", "dino_none_finetune_robust_s987", "resnet_none_finetune_robust_s987"
        ]),
        ("Ensemble Robusto: CLIP + DINO + XCEPTION (s987)", [
            "clip_none_finetune_robust_s987", "dino_none_finetune_robust_s987", "xception_none_finetune_robust_s987"
        ]),
        ("Ensemble Robusto: TODOS OS 6 (s987)", [
            f"{fam}_none_finetune_robust_s987" for fam in families
        ]),

        # --- B. Ensembles dos Modelos Clean Padrão (Seed 42) ---
        ("Ensemble Clean: CLIP + DINO (s42)", [
            "clip_none_finetune_s42", "dino_none_finetune_s42"
        ]),
        ("Ensemble Clean: CLIP + DINO + RESNET (s42)", [
            "clip_none_finetune_s42", "dino_none_finetune_s42", "resnet_none_finetune_s42"
        ]),
        ("Ensemble Clean: TODOS OS 6 CLEAN (s42)", [
            f"{fam}_none_finetune_s42" for fam in families
        ]),

        # --- C. Ensembles Híbridos Espectrais (RGB + Fourier Concat) ---
        ("Híbrido Espectral: RESNET + CLIP (Concat s42)", [
            "resnet_concat_finetune_s42", "clip_concat_finetune_s42"
        ]),
        ("Híbrido Espectral: RESNET + DINO (Concat s42)", [
            "resnet_concat_finetune_s42", "dino_concat_finetune_s42"
        ]),
        ("Híbrido Espectral: RESNET + CLIP + DINO (Concat s42)", [
            "resnet_concat_finetune_s42", "clip_concat_finetune_s42", "dino_concat_finetune_s42"
        ]),
        ("Híbrido Espectral: TODOS OS 6 CONCAT (s42)", [
            f"{fam}_concat_finetune_s42" for fam in families
        ]),

        # --- D. Self-Ensembles (5 Seeds Canônicas de cada Arquitetura) ---
        ("Self-Ensemble: DINO (5 Seeds Canônicas)", [
            f"dino_none_finetune_robust_s{s}" for s in canonical_seeds
        ]),
        ("Self-Ensemble: CLIP (5 Seeds Canônicas)", [
            f"clip_none_finetune_robust_s{s}" for s in canonical_seeds
        ]),
        ("Self-Ensemble: RESNET (5 Seeds Canônicas)", [
            f"resnet_none_finetune_robust_s{s}" for s in canonical_seeds
        ]),
        ("Self-Ensemble: MOBILENET (5 Seeds Canônicas)", [
            f"mobilenet_none_finetune_robust_s{s}" for s in canonical_seeds
        ]),
        ("Self-Ensemble: VIT (5 Seeds Canônicas)", [
            f"vit_none_finetune_robust_s{s}" for s in canonical_seeds
        ]),
        ("Self-Ensemble: XCEPTION (5 Seeds Canônicas)", [
            f"xception_none_finetune_robust_s{s}" for s in canonical_seeds
        ]),

        # --- E. Super-Ensemble Global (30 Redes: 6 Arquiteturas x 5 Seeds) ---
        ("Super-Ensemble Robusto Global (30 Redes: 6 Arqs x 5 Seeds)", [
            f"{fam}_none_finetune_robust_s{s}" for fam in families for s in canonical_seeds
        ]),

        # --- F. Fusões de Alta Performance Espaço-Frequência-Ruído ---
        ("Fusão Espaço-Frequência: CLIP Robusto + RESNET Concat + DINO Concat", [
            "clip_none_finetune_robust_s987", "resnet_concat_finetune_s42", "dino_concat_finetune_s42"
        ]),
        ("Fusão Tri-Domínio: CLIP Robusto + RESNET Concat + DINO Concat + MOBILENET SRM", [
            "clip_none_finetune_robust_s987", "resnet_concat_finetune_s42", "dino_concat_finetune_s42", "mobilenet_srm_finetune_robust_s42"
        ]),
        ("Super Comitê Campeão: DINO Clean + RESNET Clean + CLIP Robusto + MOBILENET SRM + RESNET Concat", [
            "dino_none_finetune_s42", "resnet_none_finetune_s42", "clip_none_finetune_robust_s987", "mobilenet_srm_finetune_robust_s42", "resnet_concat_finetune_s42"
        ]),
    ]

    results = []

    for name, m_keys in ensembles_def:
        valid_keys = [k for k in m_keys if k in models_pool]
        if not valid_keys:
            continue

        for strat in ["mean", "geom", "max"]:
            row = {
                "ensemble_name": name,
                "strategy": strat,
                "n_models": len(valid_keys),
            }

            # 1. DF-40
            df40_list = [models_pool[k]["df40_probs"] for k in valid_keys if "df40_probs" in models_pool[k]]
            if df40_list and len(df40_list) == len(valid_keys):
                y_df40 = models_pool[valid_keys[0]]["df40_y"]
                fused_df40 = fuse_predictions(df40_list, strat)
                m40 = compute_metrics(y_df40, fused_df40)
                row["df40_auc"] = m40["auc"]
                row["df40_acc"] = m40["acc"]
                row["df40_f1"] = m40["f1"]
            else:
                row["df40_auc"] = np.nan
                row["df40_acc"] = np.nan
                row["df40_f1"] = np.nan

            # 2. Celeb-DF v2
            celeb_list = [models_pool[k]["celeb_p"] for k in valid_keys if "celeb_p" in models_pool[k]]
            if celeb_list and len(celeb_list) == len(valid_keys):
                y_celeb = models_pool[valid_keys[0]]["celeb_y"]
                fused_celeb_frame = fuse_predictions(celeb_list, strat)
                m_frame = compute_metrics(y_celeb, fused_celeb_frame)
                row["celeb_frame_auc"] = m_frame["auc"]
                row["celeb_frame_acc"] = m_frame["acc"]
                row["celeb_frame_f1"] = m_frame["f1"]

                # Agregação por vídeo
                df_temp = celeb_meta.copy()
                df_temp["prob"] = fused_celeb_frame
                v_probs = df_temp.groupby("video_id")["prob"].mean().to_numpy()
                m_video = compute_metrics(celeb_vid_y, v_probs)
                row["celeb_video_auc"] = m_video["auc"]
                row["celeb_video_acc"] = m_video["acc"]
                row["celeb_video_f1"] = m_video["f1"]
            else:
                row["celeb_frame_auc"] = np.nan
                row["celeb_frame_acc"] = np.nan
                row["celeb_frame_f1"] = np.nan
                row["celeb_video_auc"] = np.nan
                row["celeb_video_acc"] = np.nan
                row["celeb_video_f1"] = np.nan

            results.append(row)

    df_res = pd.DataFrame(results)
    
    csv_path = dest / "cross_dataset_ensembles_benchmark.csv"
    md_path = dest / "cross_dataset_ensembles_benchmark.md"
    df_res.to_csv(csv_path, index=False)
    df_res.to_markdown(md_path, index=False)

    print(f"\n🎉 Avaliação concluída com sucesso! Resultados salvos em:\n  - {csv_path}\n  - {md_path}", flush=True)


if __name__ == "__main__":
    main()
