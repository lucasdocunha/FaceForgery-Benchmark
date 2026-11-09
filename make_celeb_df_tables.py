"""Gera tabelas consolidadas do benchmark Celeb-DF v2 a partir de todos os runs avaliados.

Escaneia models_root em busca de metrics_celeb_df.csv e metrics_celeb_df_video.csv,
agregando médias e desvios-padrão por família de modelo, modo de Fourier e regime,
gerando tabelas em Markdown, CSV e LaTeX no exato padrão do repositório.
"""

from __future__ import annotations

import argparse
from pathlib import Path
import numpy as np
import pandas as pd

from src.data.paths import models_root, output_root
from src.pipelines.checkpoints import discover_trained_runs

METRICS = ("auc", "acc", "f1", "precision", "recall", "specificity", "loss")
GROUPS = ("model_family", "fourier_mode", "regime")


def collect_celeb_metrics(root: Path) -> tuple[pd.DataFrame, pd.DataFrame]:
    frame_rows = []
    video_rows = []

    for run in discover_trained_runs(root):
        f_csv = run.run_dir / "results" / "metrics_celeb_df.csv"
        v_csv = run.run_dir / "results" / "metrics_celeb_df_video.csv"

        if f_csv.exists():
            try:
                df_f = pd.read_csv(f_csv)
                if not df_f.empty:
                    row = df_f.iloc[0].to_dict()
                    row.update(model_family=run.model_family, fourier_mode=run.fourier_mode,
                               regime=run.regime, seed=run.seed, split="celeb_df")
                    frame_rows.append(row)
            except Exception:
                pass

        if v_csv.exists():
            try:
                df_v = pd.read_csv(v_csv)
                if not df_v.empty:
                    row = df_v.iloc[0].to_dict()
                    row.update(model_family=run.model_family, fourier_mode=run.fourier_mode,
                               regime=run.regime, seed=run.seed, split="celeb_df_video")
                    video_rows.append(row)
            except Exception:
                pass

    return pd.DataFrame(frame_rows), pd.DataFrame(video_rows)


def aggregate_metrics(raw: pd.DataFrame, split_name: str) -> pd.DataFrame:
    cols = list(GROUPS) + ["split"] + [f"{m}_{s}" for m in METRICS for s in ("mean", "std")]
    if raw.empty:
        return pd.DataFrame(columns=cols)

    agg = raw.groupby(list(GROUPS), dropna=False)[list(METRICS)].agg(["mean", "std"]).reset_index()
    agg.columns = ["_".join(c).rstrip("_") for c in agg.columns]
    agg["split"] = split_name
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

    df_frame_raw, df_video_raw = collect_celeb_metrics(root)

    agg_frame = aggregate_metrics(df_frame_raw, "celeb_df_frame")
    agg_video = aggregate_metrics(df_video_raw, "celeb_df_video")

    agg_frame.to_csv(dest / "results_celeb_df_frame.csv", index=False)
    (dest / "results_celeb_df_frame.md").write_text(agg_frame.to_markdown(index=False), encoding="utf-8")

    agg_video.to_csv(dest / "results_celeb_df_video.csv", index=False)
    (dest / "results_celeb_df_video.md").write_text(agg_video.to_markdown(index=False), encoding="utf-8")

    tex_frame = _latex_table(agg_frame, "Celeb-DF Cross-Dataset Benchmark (Frame-level)", "tab:celebdf_frame")
    tex_video = _latex_table(agg_video, "Celeb-DF Cross-Dataset Benchmark (Video-level)", "tab:celebdf_video")
    (dest / "results_celeb_df.tex").write_text(tex_frame + "\n\n" + tex_video, encoding="utf-8")

    return agg_frame, agg_video


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--models-root", type=Path, default=None)
    p.add_argument("--output-dir", type=Path, default=None)
    args = p.parse_args()

    agg_frame, agg_video = make_tables(args.models_root, args.output_dir)
    print(f"Modelos com métricas de Frame: {len(agg_frame)}")
    print(f"Modelos com métricas de Vídeo: {len(agg_video)}")
    if not agg_video.empty:
        print("\n" + agg_video[["model_family", "fourier_mode", "regime", "auc_mean", "auc_std", "acc_mean", "acc_std", "f1_mean"]].to_markdown(index=False))


if __name__ == "__main__":
    main()
