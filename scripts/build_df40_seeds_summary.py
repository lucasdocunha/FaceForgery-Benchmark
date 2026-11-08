#!/usr/bin/env python3
"""
Consolida os resultados de inferência de todas as sementes do DF-40
em tabelas estatísticas com Média ± Desvio-Padrão por Paradigma e Técnica.
"""

from __future__ import annotations
import re
from pathlib import Path
import numpy as np
import pandas as pd

ROOT_DIR = Path(__file__).resolve().parent.parent
TABLES_DIR = ROOT_DIR / "tables" / "df40_all_fakes_models"
OUT_DIR = ROOT_DIR / "tables"


def parse_filename(p: Path):
    # Formatos esperados:
    # df40_all_fakes_{prefix}_seed_{seed}.csv
    # df40_all_fakes_{prefix}.csv (assume seed_42)
    name = p.stem.replace("df40_all_fakes_", "")
    m = re.match(r"^(.*)_(seed_\d+)$", name)
    if m:
        return m.group(1), m.group(2)
    return name, "seed_42"


def main():
    if not TABLES_DIR.exists():
        print(f"Diretório não encontrado: {TABLES_DIR}")
        return

    csvs = list(TABLES_DIR.glob("df40_all_fakes_*.csv"))
    if not csvs:
        print("Nenhum CSV encontrado em", TABLES_DIR)
        return

    records = []
    for p in csvs:
        prefix, seed = parse_filename(p)
        try:
            df = pd.read_csv(p)
            df["config_prefix"] = prefix
            df["seed_id"] = seed
            records.append(df)
        except Exception as e:
            print(f"Aviso ao ler {p.name}: {e}")

    if not records:
        return

    df_all = pd.concat(records, ignore_index=True)
    # Evitar duplicatas caso exista arquivo com e sem sufixo seed_42
    df_all = df_all.drop_duplicates(subset=["config_prefix", "seed_id", "paradigma", "tecnica"])

    print(f"Total de registros lidos: {len(df_all)}")
    configs = df_all["config_prefix"].unique()
    print(f"Modelos encontrados ({len(configs)}): {list(configs)}")

    # 1. Resumo por Paradigma
    # Para cada modelo, seed e paradigma, computar a taxa ponderada de acerto
    paradigm_rows = []
    for (prefix, seed, parad), grp in df_all.groupby(["config_prefix", "seed_id", "paradigma"]):
        total_samples = grp["n_amostras_total"].sum()
        total_correct = grp["acertos_threshold_val"].sum()
        rate = (total_correct / total_samples) * 100.0 if total_samples > 0 else 0.0
        paradigm_rows.append({
            "config_prefix": prefix,
            "seed_id": seed,
            "paradigma": parad,
            "taxa_acerto_pct": rate
        })

    # Adicionar TOTAL GERAL por seed
    for (prefix, seed), grp in df_all.groupby(["config_prefix", "seed_id"]):
        total_samples = grp["n_amostras_total"].sum()
        total_correct = grp["acertos_threshold_val"].sum()
        rate = (total_correct / total_samples) * 100.0 if total_samples > 0 else 0.0
        paradigm_rows.append({
            "config_prefix": prefix,
            "seed_id": seed,
            "paradigma": "TOTAL_GERAL",
            "taxa_acerto_pct": rate
        })

    df_par_seeds = pd.DataFrame(paradigm_rows)

    # Pivotar para colunas de paradigmas
    pivot_seeds = df_par_seeds.pivot_table(
        index=["config_prefix", "seed_id"],
        columns="paradigma",
        values="taxa_acerto_pct"
    ).reset_index()

    # Formatar como string "Média ± Std"
    paradigms = [c for c in df_par_seeds["paradigma"].unique()]
    formatted_rows = []
    for prefix in sorted(configs):
        sub = pivot_seeds[pivot_seeds["config_prefix"] == prefix]
        n_seeds = len(sub)
        row = {"model_name": prefix, "n_seeds": n_seeds}
        for col in sorted(paradigms):
            vals = sub[col].dropna()
            if len(vals) == 0:
                row[col] = "N/A"
            elif len(vals) == 1:
                row[col] = f"{vals.iloc[0]:.2f}%"
            else:
                row[col] = f"{vals.mean():.2f} ± {vals.std():.2f}%"
        formatted_rows.append(row)

    df_out_par = pd.DataFrame(formatted_rows)
    out_par_csv = OUT_DIR / "df40_all_fakes_seeds_summary_by_paradigm.csv"
    df_out_par.to_csv(out_par_csv, index=False)
    print(f"Salvo resumo por paradigma em: {out_par_csv}")

    # 2. Resumo por Técnica
    tech_rows = []
    for prefix in sorted(configs):
        sub = df_all[df_all["config_prefix"] == prefix]
        seeds = sub["seed_id"].unique()
        row = {"model_name": prefix, "n_seeds": len(seeds)}
        for (parad, tech), grp in sub.groupby(["paradigma", "tecnica"]):
            col_name = f"{tech} ({parad})"
            vals = grp["taxa_acerto_pct"].dropna()
            if len(vals) == 0:
                row[col_name] = "N/A"
            elif len(vals) == 1:
                row[col_name] = f"{vals.iloc[0]:.2f}%"
            else:
                row[col_name] = f"{vals.mean():.2f} ± {vals.std():.2f}%"
        tech_rows.append(row)

    df_out_tech = pd.DataFrame(tech_rows)
    out_tech_csv = OUT_DIR / "df40_all_fakes_seeds_summary_by_technique.csv"
    df_out_tech.to_csv(out_tech_csv, index=False)
    print(f"Salvo resumo por técnica em: {out_tech_csv}")

    print("\nVisualização do Resumo por Paradigma:")
    print(df_out_par.to_string(index=False))


if __name__ == "__main__":
    main()
