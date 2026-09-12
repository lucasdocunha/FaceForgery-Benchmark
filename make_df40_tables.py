"""Gera tabelas consolidadas do benchmark DF40 a partir de todos os runs avaliados.

Escaneia models_root em busca de metrics_df40.csv,
agregando médias e desvios-padrão por família de modelo, modo de Fourier e regime,
gerando tabelas em Markdown, CSV e LaTeX.
"""

from __future__ import annotations

import argparse
from pathlib import Path
import pandas as pd

from src.data.paths import models_root, output_root
from src.pipelines.checkpoints import discover_trained_runs

METRICS = ("auc", "acc", "f1", "precision", "recall", "specificity", "loss")
GROUPS = ("model_family", "fourier_mode", "regime")


def collect_df40_metrics(root: Path) -> tuple[pd.DataFrame, pd.DataFrame]:
    raw_rows = []
    for run in discover_trained_runs(root):
        m_csv = run.run_dir / "results" / "metrics_df40.csv"
        if m_csv.exists():
            try:
                df = pd.read_csv(m_csv)
                if not df.empty:
                    row = df.iloc[0].to_dict()
                    row.update(
                        model_family=run.model_family,
                        fourier_mode=run.fourier_mode,
                        regime=run.regime,
                        seed=run.seed,
                    )
                    raw_rows.append(row)
            except Exception:
                pass
    df_raw = pd.DataFrame(raw_rows)
    return df_raw


def aggregate_metrics(raw: pd.DataFrame) -> pd.DataFrame:
    cols = list(GROUPS) + [f"{m}_{s}" for m in METRICS for s in ("mean", "std")]
    if raw.empty:
        return pd.DataFrame(columns=cols)

    agg = raw.groupby(list(GROUPS), dropna=False)[list(METRICS)].agg(["mean", "std"]).reset_index()
    agg.columns = ["_".join(c).rstrip("_") for c in agg.columns]
    return agg.reindex(columns=cols)


def _latex_table(agg: pd.DataFrame, title: str, label: str) -> str:
    paper = agg[["model_family", "fourier_mode", "regime"]].copy()
    for m in ("auc", "acc", "f1"):
        if f"{m}_mean" in agg.columns:
            paper[m.upper()] = agg.apply(
                lambda r: f"{r[f'{m}_mean']:.3f} $\\pm$ {r[f'{m}_std']:.3f}"
                if pd.notna(r.get(f"{m}_std")) else f"{r[f'{m}_mean']:.3f}",
                axis=1,
            )
    return paper.rename(columns={"model_family": "Model", "fourier_mode": "Mode", "regime": "Regime"}).to_latex(
        index=False, escape=False, caption=title, label=label
    )


def make_tables(root: Path | None = None, out_dir: Path | None = None):
    root = root or models_root()
    dest = Path(out_dir or output_root() / "results" / "tables")
    dest.mkdir(parents=True, exist_ok=True)
    tables_dir = Path("tables")
    tables_dir.mkdir(parents=True, exist_ok=True)

    df_raw = collect_df40_metrics(root)
    agg = aggregate_metrics(df_raw)

    if not df_raw.empty:
        df_raw.to_csv(tables_dir / "df40_all_runs_individual.csv", index=False)

    agg.to_csv(dest / "results_df40.csv", index=False)
    agg.to_csv(tables_dir / "df40_all_models_consolidated.csv", index=False)
    (dest / "results_df40.md").write_text(agg.to_markdown(index=False), encoding="utf-8")

    tex = _latex_table(agg, "DF40 Cross-Dataset Benchmark", "tab:df40_benchmark")
    (dest / "results_df40.tex").write_text(tex, encoding="utf-8")

    return agg, df_raw


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--models-root", type=Path, default=None)
    p.add_argument("--output-dir", type=Path, default=None)
    args = p.parse_args()

    agg, raw = make_tables(args.models_root, args.output_dir)
    print(f"Total de runs avaliados no DF40: {len(raw)}")
    print(f"Total de grupos de modelos agregados: {len(agg)}")
    if not agg.empty:
        disp_cols = ["model_family", "fourier_mode", "regime", "auc_mean", "auc_std", "acc_mean", "acc_std", "f1_mean"]
        valid_cols = [c for c in disp_cols if c in agg.columns]
        print("\n" + agg[valid_cols].sort_values("auc_mean", ascending=False).to_markdown(index=False))


if __name__ == "__main__":
    main()
