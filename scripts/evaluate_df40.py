"""Avaliação dos modelos no benchmark DF40 (DeepFake-40)."""

from __future__ import annotations

import time
from pathlib import Path
import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader
from torchvision import transforms

from src.data.data import ImageDataset
from src.data.paths import models_root
from src.pipelines.checkpoints import discover_trained_runs, load_model_from_run, config_from_run
from src.pipelines.evaluation import binary_metrics, evaluate_classifier


def main():
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    root = models_root()
    manifest_csv = Path("data/df40/test.csv")
    manifest_df = pd.read_csv(manifest_csv)

    models = ["clip", "dino", "vit", "resnet", "mobilenet", "xception"]

    print(f"Iniciando benchmark DF40 com {len(manifest_df)} imagens...", flush=True)
    print(f"Distribuição Real/Fake:\n{manifest_df['target'].value_counts()}", flush=True)

    predictions = {}
    metrics_summary = []

    # 1. AVALIAÇÃO DOS MODELOS ROBUSTOS (SEED 987)
    print("\n=== AVALIANDO MODELOS ROBUSTOS (SEED 987) ===", flush=True)
    for m in models:
        runs = [
            r for r in discover_trained_runs(root, only_model_family=m)
            if r.seed == 987 and r.fourier_mode == "none" and r.regime == "finetune"
        ]
        if not runs:
            print(f"Run não encontrado para {m} (987)", flush=True)
            continue
        run = runs[0]
        config = config_from_run(run)
        model = load_model_from_run(run, device)

        transform = transforms.Compose([
            transforms.Resize((config.image_size, config.image_size)),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
        ])

        dataset = ImageDataset(
            manifest_csv,
            Path(""),
            transform=transform,
            fourier="none",
            spatial_size=(config.image_size, config.image_size),
            in_channels=3,
        )
        loader = DataLoader(dataset, batch_size=128, shuffle=False, num_workers=4, pin_memory=True)

        t0 = time.time()
        res = evaluate_classifier(
            model, loader, torch.nn.CrossEntropyLoss(), device,
            threshold=run.threshold, use_amp=True, desc=f"{m}_987"
        )
        dt = time.time() - t0
        predictions[m] = res["probs"]

        print(f"[{m.upper()} 987] AUC: {res['auc']*100:.2f}% | ACC: {res['acc']*100:.2f}% | F1: {res['f1']*100:.2f}% ({dt:.1f}s)", flush=True)

        metrics_summary.append({
            "Modelo": m.upper(),
            "Tipo": "Robusto (987)",
            "AUC (%)": round(res["auc"] * 100, 2),
            "ACC (%)": round(res["acc"] * 100, 2),
            "F1 (%)": round(res["f1"] * 100, 2),
            "Precision (%)": round(res["precision"] * 100, 2),
            "Recall (%)": round(res["recall"] * 100, 2),
            "Specificity (%)": round(res["specificity"] * 100, 2),
        })

        del model
        torch.cuda.empty_cache()

    # 2. AVALIAÇÃO DOS MODELOS PADRÃO (SEED 42 PARA COMPARAÇÃO)
    print("\n=== AVALIANDO MODELOS PADRÃO (SEED 42) PARA COMPARAÇÃO ===", flush=True)
    std_predictions = {}
    for m in models:
        runs = [
            r for r in discover_trained_runs(root, only_model_family=m)
            if r.seed == 42 and r.fourier_mode == "none" and r.regime == "finetune"
        ]
        if not runs:
            continue
        run = runs[0]
        config = config_from_run(run)
        model = load_model_from_run(run, device)

        transform = transforms.Compose([
            transforms.Resize((config.image_size, config.image_size)),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
        ])

        dataset = ImageDataset(
            manifest_csv,
            Path(""),
            transform=transform,
            fourier="none",
            spatial_size=(config.image_size, config.image_size),
            in_channels=3,
        )
        loader = DataLoader(dataset, batch_size=128, shuffle=False, num_workers=4, pin_memory=True)

        res = evaluate_classifier(
            model, loader, torch.nn.CrossEntropyLoss(), device,
            threshold=run.threshold, use_amp=True, desc=f"{m}_42"
        )
        std_predictions[m] = res["probs"]

        print(f"[{m.upper()} 42]  AUC: {res['auc']*100:.2f}% | ACC: {res['acc']*100:.2f}% | F1: {res['f1']*100:.2f}%", flush=True)

        metrics_summary.append({
            "Modelo": m.upper(),
            "Tipo": "Padrão (42)",
            "AUC (%)": round(res["auc"] * 100, 2),
            "ACC (%)": round(res["acc"] * 100, 2),
            "F1 (%)": round(res["f1"] * 100, 2),
            "Precision (%)": round(res["precision"] * 100, 2),
            "Recall (%)": round(res["recall"] * 100, 2),
            "Specificity (%)": round(res["specificity"] * 100, 2),
        })

        del model
        torch.cuda.empty_cache()

    # 3. ENSEMBLES DOS MODELOS ROBUSTOS
    y_true = manifest_df["target"].values
    ensembles = [
        ("DINO + XCEPTION + VIT", ["dino", "xception", "vit"]),
        ("CLIP + DINO", ["clip", "dino"]),
        ("CLIP + DINO + XCEPTION", ["clip", "dino", "xception"]),
        ("CLIP + DINO + VIT", ["clip", "dino", "vit"]),
        ("CLIP + DINO + RESNET", ["clip", "dino", "resnet"]),
        ("TODOS OS 6 ROBUSTOS", ["clip", "dino", "vit", "resnet", "mobilenet", "xception"]),
    ]

    print("\n=== AVALIANDO ENSEMBLES ROBUSTOS NO DF40 ===", flush=True)
    ensemble_results = []
    for ens_name, m_keys in ensembles:
        stacked = np.stack([predictions[k] for k in m_keys], axis=0)
        for fusion in ["max", "mean", "geom"]:
            if fusion == "mean":
                fused = np.mean(stacked, axis=0)
            elif fusion == "max":
                fused = np.max(stacked, axis=0)
            elif fusion == "geom":
                eps = 1e-7
                log_fused = np.mean(np.log(np.clip(stacked, eps, 1.0)), axis=0)
                fused = np.exp(log_fused)

            m = binary_metrics(y_true, fused, threshold=0.5)
            ensemble_results.append({
                "Ensemble": ens_name,
                "Fusão": fusion,
                "AUC (%)": round(m["auc"] * 100, 2),
                "ACC (%)": round(m["acc"] * 100, 2),
                "F1 (%)": round(m["f1"] * 100, 2),
                "Precision (%)": round(m["precision"] * 100, 2),
                "Recall (%)": round(m["recall"] * 100, 2),
                "Specificity (%)": round(m["specificity"] * 100, 2),
            })
            metrics_summary.append({
                "Modelo": f"ENS: {ens_name} ({fusion})",
                "Tipo": f"Ensemble Robusto ({fusion})",
                "AUC (%)": round(m["auc"] * 100, 2),
                "ACC (%)": round(m["acc"] * 100, 2),
                "F1 (%)": round(m["f1"] * 100, 2),
                "Precision (%)": round(m["precision"] * 100, 2),
                "Recall (%)": round(m["recall"] * 100, 2),
                "Specificity (%)": round(m["specificity"] * 100, 2),
            })

    # 4. DESEMPENHO POR PARADIGMA GENERATIVO
    print("\n=== DESEMPENHO POR PARADIGMA GENERATIVO NO DF40 ===", flush=True)
    paradigms = ["diffusion", "gan", "face_swap", "editing_t2i", "talking_reenactment", "commercial_avatar"]
    real_mask = (manifest_df["paradigm"] == "real").values
    paradigm_records = []

    for p in paradigms:
        p_mask = (manifest_df["paradigm"] == p).values
        sub_idx = real_mask | p_mask
        sub_y = y_true[sub_idx]

        for m in models:
            sub_probs = predictions[m][sub_idx]
            m_res = binary_metrics(sub_y, sub_probs, threshold=0.5)
            paradigm_records.append({
                "Paradigma": p,
                "Modelo": m.upper(),
                "Tipo": "Robusto",
                "AUC (%)": round(m_res["auc"] * 100, 2),
                "ACC (%)": round(m_res["acc"] * 100, 2),
                "F1 (%)": round(m_res["f1"] * 100, 2),
            })

    # Salvar tabelas
    Path("tables").mkdir(exist_ok=True)
    df_metrics = pd.DataFrame(metrics_summary)
    df_metrics.to_csv("tables/df40_models_benchmark.csv", index=False)

    df_ens = pd.DataFrame(ensemble_results).sort_values("AUC (%)", ascending=False)
    df_ens.to_csv("tables/df40_ensembles_benchmark.csv", index=False)

    df_paradigms = pd.DataFrame(paradigm_records)
    df_paradigms.to_csv("tables/df40_paradigms_benchmark.csv", index=False)

    print("\nResultados salvos em:")
    print(" - tables/df40_models_benchmark.csv")
    print(" - tables/df40_ensembles_benchmark.csv")
    print(" - tables/df40_paradigms_benchmark.csv")


if __name__ == "__main__":
    main()
