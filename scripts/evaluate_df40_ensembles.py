"""Avaliação abrangente de Ensembles no benchmark DF40 (DeepFake-40).

Constrói e avalia:
1. Ensembles Padrão (Seed 42 - Espacial none): CLIP, DINO, ViT, ResNet, MobileNet, Xception.
2. Ensembles Híbridos Espectrais (Modo concat - Espaço + Frequência): ResNet, CLIP, DINO, Xception, etc.
3. Super-Ensembles Campeões (Fusão dos melhores modelos espaciais e espectrais de cada família).
4. Deep Ensembles Intra-Arquitetura (Averaging das 5 sementes de treino por família).
5. Estratégias de fusão: mean, geom, max.
6. Breakdown por paradigma generativo (Diffusion, GANs, Face Swap, Edição T2I, Talking Head, Avatares Comerciais).
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
            return 0.5
        return float(roc_auc_score(y_true, y_score))
    except Exception:
        return 0.5


def compute_metrics(y_true: np.ndarray, probs: np.ndarray, threshold: float = 0.5) -> dict:
    y_pred = (probs >= threshold).astype(int)
    tn = int(((y_true == 0) & (y_pred == 0)).sum())
    fp = int(((y_true == 0) & (y_pred == 1)).sum())
    fn = int(((y_true == 1) & (y_pred == 0)).sum())
    tp = int(((y_true == 1) & (y_pred == 1)).sum())

    acc = float(accuracy_score(y_true, y_pred))
    prec = float(precision_score(y_true, y_pred, zero_division=0))
    rec = float(recall_score(y_true, y_pred, zero_division=0))
    f1 = float(f1_score(y_true, y_pred, zero_division=0))
    auc = safe_auc(y_true, probs)
    spec = float(tn / (tn + fp)) if (tn + fp) > 0 else 0.0

    return {
        "auc": round(auc * 100, 2),
        "acc": round(acc * 100, 2),
        "f1": round(f1 * 100, 2),
        "precision": round(prec * 100, 2),
        "recall": round(rec * 100, 2),
        "specificity": round(spec * 100, 2),
    }


def fuse_predictions(probs_list: list[np.ndarray], strategy: str) -> np.ndarray:
    stack = np.stack(probs_list, axis=0)
    if strategy == "mean":
        return np.mean(stack, axis=0)
    elif strategy == "geom":
        eps = 1e-7
        log_mean = np.mean(np.log(np.clip(stack, eps, 1.0 - eps)), axis=0)
        return np.exp(log_mean)
    elif strategy == "max":
        return np.max(stack, axis=0)
    elif strategy == "harmonic":
        eps = 1e-7
        return len(probs_list) / np.sum(1.0 / np.clip(stack, eps, 1.0), axis=0)
    else:
        raise ValueError(f"Estratégia desconhecida: {strategy}")


def load_model_probs(root: Path, family: str, mode: str, seed: int | str) -> np.ndarray | None:
    path = root / family / mode / "finetune" / f"seed_{seed}" / "results" / "outputs_df40.npz"
    if not path.exists():
        return None
    d = np.load(path)
    return d["probs"]


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--manifest-csv", type=Path, default=Path("data/df40/test.csv"))
    parser.add_argument("--output-dir", type=Path, default=Path("tables"))
    args = parser.parse_args()

    root = models_root()
    manifest_df = pd.read_csv(args.manifest_csv)
    y_true = manifest_df["target"].values

    print(f"Manifest carregado: {len(manifest_df)} amostras (Real: {(y_true==0).sum()}, Fake: {(y_true==1).sum()})")

    # 1. Carregar predições disponíveis
    families = ["clip", "dino", "vit", "resnet", "mobilenet", "xception"]
    seeds_std = [7, 42, 123, 2024, 2025]
    all_seeds = [7, 42, 123, 2024, 2025, 987]

    pool: dict[str, np.ndarray] = {}
    for fam in families:
        for mode in ["none", "concat"]:
            for s in all_seeds:
                p = load_model_probs(root, fam, mode, s)
                if p is not None:
                    pool[f"{fam}_{mode}_s{s}"] = p

    print(f"Predições carregadas na memória: {len(pool)} modelos.")

    # 2. Definição das Combinações de Ensembles
    categories: dict[str, list[tuple[str, list[str]]]] = {}

    # Categoria A: Ensembles dos Modelos Padrão (Seed 42 - Espacial `none`)
    categories["Ensemble Padrão (Seed 42 - Espacial)"] = [
        ("CLIP + DINO (Padrão)", ["clip_none_s42", "dino_none_s42"]),
        ("CLIP + DINO + VIT (Padrão)", ["clip_none_s42", "dino_none_s42", "vit_none_s42"]),
        ("CLIP + DINO + XCEPTION (Padrão)", ["clip_none_s42", "dino_none_s42", "xception_none_s42"]),
        ("CLIP + DINO + RESNET (Padrão)", ["clip_none_s42", "dino_none_s42", "resnet_none_s42"]),
        ("DINO + XCEPTION + VIT (Padrão)", ["dino_none_s42", "xception_none_s42", "vit_none_s42"]),
        ("TODOS OS 6 PADRÃO", ["clip_none_s42", "dino_none_s42", "vit_none_s42", "resnet_none_s42", "mobilenet_none_s42", "xception_none_s42"]),
    ]

    # Categoria B: Ensembles dos Modelos Robustos (Seed 987 - Espacial `none`)
    categories["Ensemble Robusto (Seed 987 - Espacial)"] = [
        ("CLIP + DINO (Robusto)", ["clip_none_s987", "dino_none_s987"]),
        ("CLIP + DINO + VIT (Robusto)", ["clip_none_s987", "dino_none_s987", "vit_none_s987"]),
        ("CLIP + DINO + XCEPTION (Robusto)", ["clip_none_s987", "dino_none_s987", "xception_none_s987"]),
        ("CLIP + DINO + RESNET (Robusto)", ["clip_none_s987", "dino_none_s987", "resnet_none_s987"]),
        ("DINO + XCEPTION + VIT (Robusto)", ["dino_none_s987", "xception_none_s987", "vit_none_s987"]),
        ("TODOS OS 6 ROBUSTOS", ["clip_none_s987", "dino_none_s987", "vit_none_s987", "resnet_none_s987", "mobilenet_none_s987", "xception_none_s987"]),
    ]

    # Categoria C: Ensembles Híbridos Espectrais (`concat` - RGB + Magnitude Fourier)
    categories["Ensemble Híbrido Concat (RGB + Magnitude)"] = [
        ("RESNET + CLIP (Concat s42)", ["resnet_concat_s42", "clip_concat_s42"]),
        ("RESNET + DINO (Concat s42)", ["resnet_concat_s42", "dino_concat_s42"]),
        ("CLIP + DINO (Concat s42)", ["clip_concat_s42", "dino_concat_s42"]),
        ("RESNET + CLIP + DINO (Concat s42)", ["resnet_concat_s42", "clip_concat_s42", "dino_concat_s42"]),
        ("RESNET + CLIP + DINO + XCEPTION (Concat s42)", ["resnet_concat_s42", "clip_concat_s42", "dino_concat_s42", "xception_concat_s42"]),
        ("TODOS OS 6 CONCAT (s42)", ["resnet_concat_s42", "clip_concat_s42", "dino_concat_s42", "xception_concat_s42", "mobilenet_concat_s42", "vit_concat_s42"]),
    ]

    # Categoria D: Super-Ensembles Inter-Domínio (Melhores de Cada Família)
    # ResNet: concat s123 (80.65% AUC)
    # CLIP: none s987 (82.63% AUC) e concat s2025 (79.72% AUC)
    # DINO: concat s123 (78.10% AUC) e none s987 (75.18% AUC)
    # Xception: concat s42 (74.21% AUC)
    # MobileNet: none s987 (73.05% AUC)
    # ViT: none s987 (72.72% AUC)
    categories["Super-Ensemble Campeões (Melhores Modelos)"] = [
        ("CLIP Robusto + CLIP Concat (s2025) + RESNET Concat (s123)", ["clip_none_s987", "clip_concat_s2025", "resnet_concat_s123"]),
        ("CLIP Robusto + CLIP Concat (s2025) + RESNET Concat (s123) + DINO Concat (s123)", ["clip_none_s987", "clip_concat_s2025", "resnet_concat_s123", "dino_concat_s123"]),
        ("CLIP Robusto + RESNET Concat (s123)", ["clip_none_s987", "resnet_concat_s123"]),
        ("CLIP Robusto + DINO Robusto + RESNET Concat (s123)", ["clip_none_s987", "dino_none_s987", "resnet_concat_s123"]),
        ("CLIP Robusto + DINO Concat (s123) + RESNET Concat (s123)", ["clip_none_s987", "dino_concat_s123", "resnet_concat_s123"]),
        ("CLIP Robusto + RESNET Concat + DINO Concat + XCEPTION Concat", ["clip_none_s987", "resnet_concat_s123", "dino_concat_s123", "xception_concat_s42"]),
        ("SUPER-ENSEMBLE TOP-5 (CLIP Robusto + DINO Robusto + RESNET Concat + DINO Concat + XCEPTION Concat)", [
            "clip_none_s987", "dino_none_s987", "resnet_concat_s123", "dino_concat_s123", "xception_concat_s42"
        ]),
        ("SUPER-ENSEMBLE COMPLETO (Melhor de Cada Família)", [
            "clip_none_s987", "resnet_concat_s123", "dino_concat_s123", "xception_concat_s42", "mobilenet_none_s987", "vit_none_s987"
        ]),
    ]

    # Categoria E: Deep Ensembles Intra-Arquitetura (5 sementes de cada família)
    deep_ens = []
    for fam in families:
        for mode in ["none", "concat"]:
            seeds_available = [s for s in seeds_std if f"{fam}_{mode}_s{s}" in pool]
            if len(seeds_available) == 5:
                deep_ens.append((f"Deep Ensemble: {fam.upper()} `{mode}` (5 seeds)", [f"{fam}_{mode}_s{s}" for s in seeds_available]))
    categories["Deep Ensembles Intra-Arquitetura (5 Seeds)"] = deep_ens

    # 3. Avaliar Ensembles
    results = []
    fusions = ["mean", "geom", "max"]

    for cat_name, ens_list in categories.items():
        for ens_name, model_keys in ens_list:
            # Checar se todos os modelos necessários estão disponíveis
            missing = [k for k in model_keys if k not in pool]
            if missing:
                print(f"Pulando {ens_name}: faltam predições ({missing})")
                continue

            probs_list = [pool[k] for k in model_keys]

            for fusion in fusions:
                fused = fuse_predictions(probs_list, fusion)
                m = compute_metrics(y_true, fused, threshold=0.5)

                results.append({
                    "Categoria": cat_name,
                    "Ensemble": ens_name,
                    "Fusão": fusion,
                    "Modelos": len(model_keys),
                    "AUC (%)": m["auc"],
                    "ACC (%)": m["acc"],
                    "F1 (%)": m["f1"],
                    "Precision (%)": m["precision"],
                    "Recall (%)": m["recall"],
                    "Specificity (%)": m["specificity"],
                })

    df_results = pd.DataFrame(results).sort_values("AUC (%)", ascending=False)
    args.output_dir.mkdir(parents=True, exist_ok=True)

    out_csv = args.output_dir / "df40_all_ensembles_benchmark.csv"
    df_results.to_csv(out_csv, index=False)
    print(f"\nResultados salvos em {out_csv} ({len(df_results)} ensembles avaliados)")

    # 4. Avaliar Paradigmas Generativos para os Top Ensembles
    paradigms = ["diffusion", "gan", "face_swap", "editing_t2i", "talking_reenactment", "commercial_avatar"]
    real_mask = (manifest_df["paradigm"] == "real").values
    top_ensembles = [
        ("CLIP Robusto + CLIP Concat (s2025) + RESNET Concat (s123)", ["clip_none_s987", "clip_concat_s2025", "resnet_concat_s123"], "mean"),
        ("CLIP Robusto + CLIP Concat + RESNET Concat + DINO Concat", ["clip_none_s987", "clip_concat_s2025", "resnet_concat_s123", "dino_concat_s123"], "mean"),
        ("CLIP Robusto + RESNET Concat (s123)", ["clip_none_s987", "resnet_concat_s123"], "mean"),
        ("CLIP Robusto + DINO Robusto + RESNET Concat (s123)", ["clip_none_s987", "dino_none_s987", "resnet_concat_s123"], "mean"),
        ("SUPER-ENSEMBLE TOP-5", ["clip_none_s987", "dino_none_s987", "resnet_concat_s123", "dino_concat_s123", "xception_concat_s42"], "mean"),
        ("CLIP + DINO (Robusto)", ["clip_none_s987", "dino_none_s987"], "mean"),
        ("RESNET + CLIP + DINO (Concat s42)", ["resnet_concat_s42", "clip_concat_s42", "dino_concat_s42"], "mean"),
        ("TODOS OS 6 PADRÃO", ["clip_none_s42", "dino_none_s42", "vit_none_s42", "resnet_none_s42", "mobilenet_none_s42", "xception_none_s42"], "mean"),
        ("Deep Ensemble: RESNET `concat` (5 seeds)", [f"resnet_concat_s{s}" for s in seeds_std], "mean"),
    ]

    paradigm_records = []
    for ens_name, model_keys, fusion in top_ensembles:
        missing = [k for k in model_keys if k not in pool]
        if missing:
            continue
        probs_list = [pool[k] for k in model_keys]
        fused = fuse_predictions(probs_list, fusion)

        rec = {"Ensemble": ens_name, "Fusão": fusion}
        for p in paradigms:
            p_mask = (manifest_df["paradigm"] == p).values
            sub_idx = real_mask | p_mask
            sub_y = y_true[sub_idx]
            sub_probs = fused[sub_idx]
            sub_m = compute_metrics(sub_y, sub_probs, threshold=0.5)
            rec[p] = sub_m["auc"]
        paradigm_records.append(rec)

    df_para = pd.DataFrame(paradigm_records)
    out_para = args.output_dir / "df40_ensembles_by_paradigm.csv"
    df_para.to_csv(out_para, index=False)
    print(f"Breakdown por paradigma salvo em {out_para}")

    # 5. Gerar versão Markdown e LaTeX
    dest = Path("results/tables")
    dest.mkdir(parents=True, exist_ok=True)

    (dest / "results_df40_ensembles.md").write_text(df_results.to_markdown(index=False), encoding="utf-8")
    tex = df_results.head(25).to_latex(index=False, caption="Top Ensembles no DF40 Benchmark", label="tab:df40_ensembles")
    (dest / "results_df40_ensembles.tex").write_text(tex, encoding="utf-8")

    print("\n" + df_results.head(20).to_markdown(index=False))


if __name__ == "__main__":
    main()
