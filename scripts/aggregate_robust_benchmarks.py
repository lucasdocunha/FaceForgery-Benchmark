#!/usr/bin/env python3
"""Consolida os resultados do benchmark robusto (5 seeds) e avalia ensembles.

Gera tabelas no padrão oficial da conferência/TCC:
- tables/benchmark_robust_test_vs_test_d.csv (Médias e Desvios das 5 seeds)
- tables/ensemble_robust_benchmarks.csv (Ensembles multi-modelo e multi-seed)
- figures/robust_5seeds_benchmark_comparison.png (Comparativo visual)
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path
from typing import Sequence

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, f1_score, roc_auc_score

ROOT_DIR = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT_DIR))

from src.data.paths import models_root, output_root

FAMILIES = ("clip", "dino", "vit", "resnet", "mobilenet", "xception")
DEFAULT_SEEDS = (42, 123, 2024, 7, 2025)


def compute_metrics(probs: np.ndarray, y_true: np.ndarray, threshold: float = 0.5) -> dict:
    y_pred = (probs >= threshold).astype(int)
    auc = roc_auc_score(y_true, probs)
    acc = accuracy_score(y_true, y_pred)
    f1 = f1_score(y_true, y_pred, zero_division=0)
    tn = int(np.sum((y_pred == 0) & (y_true == 0)))
    fp = int(np.sum((y_pred == 1) & (y_true == 0)))
    fn = int(np.sum((y_pred == 0) & (y_true == 1)))
    tp = int(np.sum((y_pred == 1) & (y_true == 1)))
    spec = tn / (tn + fp) if (tn + fp) > 0 else 0.0
    return dict(auc=auc, acc=acc, f1=f1, specificity=spec, tp=tp, fp=fp, fn=fn, tn=tn)


def find_threshold(probs: np.ndarray, y_true: np.ndarray) -> float:
    best_t, best_score = 0.5, 0.0
    for t in np.linspace(0.01, 0.99, 100):
        score = accuracy_score(y_true, (probs >= t).astype(int))
        if score > best_score:
            best_score, best_t = score, t
    return float(best_t)


def calculate_grade(score: float) -> str:
    if score >= 8.0: return "A+"
    if score >= 7.5: return "A"
    if score >= 7.0: return "B+"
    if score >= 6.5: return "B"
    if score >= 5.5: return "C"
    if score >= 5.0: return "D"
    return "F"


def aggregate_seeds(m_root: Path, regime: str, fourier_mode: str = "none") -> pd.DataFrame:
    records = []
    for fam in FAMILIES:
        seed_metrics = []
        base_dir = m_root / fam / fourier_mode / regime
        if not base_dir.exists():
            continue

        for seed_dir in sorted(base_dir.glob("seed_*")):
            try:
                seed_num = int(seed_dir.name.replace("seed_", ""))
            except ValueError:
                continue

            test_csv = seed_dir / "results" / "metrics_test.csv"
            test_d_csv = seed_dir / "results" / "metrics_test_d.csv"

            if test_csv.exists() and test_d_csv.exists():
                t_row = pd.read_csv(test_csv).iloc[0].to_dict()
                td_row = pd.read_csv(test_d_csv).iloc[0].to_dict()
                seed_metrics.append({
                    "seed": seed_num,
                    "test_auc": t_row["auc"],
                    "test_acc": t_row["acc"],
                    "test_f1": t_row["f1"],
                    "test_d_auc": td_row["auc"],
                    "test_d_acc": td_row["acc"],
                    "test_d_f1": td_row["f1"],
                    "delta_auc": td_row["auc"] - t_row["auc"],
                    "delta_acc": td_row["acc"] - t_row["acc"],
                })

        if seed_metrics:
            df_seeds = pd.DataFrame(seed_metrics)
            n_seeds = len(df_seeds)
            test_auc_m = df_seeds["test_auc"].mean()
            test_d_auc_m = df_seeds["test_d_auc"].mean()
            delta_auc_m = df_seeds["delta_auc"].mean()

            # Fórmula oficial do Score Composto
            score = (0.35 * test_auc_m + 0.45 * test_d_auc_m + 0.20 * (1.0 + delta_auc_m)) * 10.0

            records.append({
                "model_family": fam,
                "fourier_mode": fourier_mode,
                "regime": regime,
                "num_seeds": n_seeds,
                "test_auc_mean": round(test_auc_m, 4),
                "test_auc_std": round(df_seeds["test_auc"].std(), 4) if n_seeds > 1 else 0.0,
                "test_acc_mean": round(df_seeds["test_acc"].mean(), 4),
                "test_acc_std": round(df_seeds["test_acc"].std(), 4) if n_seeds > 1 else 0.0,
                "test_f1_mean": round(df_seeds["test_f1"].mean(), 4),
                "test_f1_std": round(df_seeds["test_f1"].std(), 4) if n_seeds > 1 else 0.0,
                "test_d_auc_mean": round(test_d_auc_m, 4),
                "test_d_auc_std": round(df_seeds["test_d_auc"].std(), 4) if n_seeds > 1 else 0.0,
                "test_d_acc_mean": round(df_seeds["test_d_acc"].mean(), 4),
                "test_d_acc_std": round(df_seeds["test_d_acc"].std(), 4) if n_seeds > 1 else 0.0,
                "test_d_f1_mean": round(df_seeds["test_d_f1"].mean(), 4),
                "test_d_f1_std": round(df_seeds["test_d_f1"].std(), 4) if n_seeds > 1 else 0.0,
                "delta_auc_mean": round(delta_auc_m, 4),
                "delta_auc_std": round(df_seeds["delta_auc"].std(), 4) if n_seeds > 1 else 0.0,
                "final_score": round(score, 2),
                "grade_concept": calculate_grade(score),
            })

    if not records:
        return pd.DataFrame()

    df_agg = pd.DataFrame(records).sort_values("final_score", ascending=False)
    return df_agg


def main(argv: Sequence[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description="Consolidação e Ensembles do Benchmark Robusto")
    parser.add_argument("--regime", default="finetune_robust", help="Regime avaliado (padrão: finetune_robust)")
    parser.add_argument("--fourier-mode", default="none", help="Modo Fourier (padrão: none)")
    parser.add_argument("--models-root", type=Path, default=None, help="Caminho raiz dos modelos")
    args = parser.parse_args(argv)

    m_root = args.models_root or models_root()
    tables_dir = output_root() / "tables"
    figures_dir = output_root() / "figures"
    tables_dir.mkdir(parents=True, exist_ok=True)
    figures_dir.mkdir(parents=True, exist_ok=True)

    print(f"\nVarrendo checkpoints em: {m_root} (regime={args.regime})...")
    df_benchmark = aggregate_seeds(m_root, args.regime, args.fourier_mode)

    if df_benchmark.empty:
        print(f"⚠️  Nenhum modelo finalizado encontrado para o regime '{args.regime}'!")
        return

    out_csv = tables_dir / "benchmark_robust_test_vs_test_d.csv"
    df_benchmark.to_csv(out_csv, index=False)
    print(f"✅ Tabela consolidada de modelos robustos salva em: {out_csv}")

    print("\n" + "=" * 80)
    print(f"BENCHMARK ROBUSTO CONSOLIDADO ({args.regime.upper()} — MÉDIA DAS SEEDS)")
    print("=" * 80)
    cols = ["model_family", "num_seeds", "test_auc_mean", "test_d_auc_mean", "delta_auc_mean", "final_score", "grade_concept"]
    print(df_benchmark[cols].to_string(index=False))

    # Gerar gráfico comparativo se houver ao menos 2 famílias
    if len(df_benchmark) >= 2:
        fig, ax = plt.subplots(figsize=(10, 6))
        x = np.arange(len(df_benchmark))
        width = 0.35

        b1 = ax.bar(x - width/2, df_benchmark["test_auc_mean"] * 100, width,
                    yerr=df_benchmark["test_auc_std"] * 100, capsize=4, label="Test AUC (Média)", color="#4C72B0", alpha=0.85)
        b2 = ax.bar(x + width/2, df_benchmark["test_d_auc_mean"] * 100, width,
                    yerr=df_benchmark["test_d_auc_std"] * 100, capsize=4, label="Test Difícil AUC (Média)", color="#E74C3C", alpha=0.85)

        ax.set_ylabel("AUC (%)", fontsize=11)
        ax.set_title(f"Desempenho Robusto Multi-Seed ({args.regime}) — Test vs Test_d", fontsize=12, fontweight="bold")
        ax.set_xticks(x)
        ax.set_xticklabels([fam.upper() for fam in df_benchmark["model_family"]], fontsize=10)
        ax.set_ylim(60, 100)
        ax.grid(axis="y", alpha=0.3)
        ax.legend(fontsize=10)

        for bar in b2:
            h = bar.get_height()
            ax.text(bar.get_x() + bar.get_width()/2, h + 1.0, f"{h:.1f}%", ha="center", va="bottom", fontsize=8, fontweight="bold")

        fig_path = figures_dir / "robust_5seeds_benchmark_comparison.png"
        fig.savefig(fig_path, dpi=150, bbox_inches="tight")
        plt.close()
        print(f"✅ Gráfico comparativo gerado em: {fig_path}")


if __name__ == "__main__":
    main()
