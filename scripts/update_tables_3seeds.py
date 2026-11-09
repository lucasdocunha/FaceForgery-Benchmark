#!/usr/bin/env python3
"""
Atualiza todas as tabelas LaTeX e resumos CSV considerando as 3 sementes completas:
seed_42, seed_123 e seed_2024.
"""

from __future__ import annotations
import re
from pathlib import Path
import pandas as pd
import numpy as np

ROOT_DIR = Path(__file__).resolve().parent.parent
TABLES_DIR = ROOT_DIR / "tables" / "df40_all_fakes_models"
OUT_TABLES_DIR = ROOT_DIR / "tables"
RESULTS_DIR = ROOT_DIR / "results" / "tables"
PAPER_DIR = ROOT_DIR / "paper"

SEEDS = ["seed_42", "seed_123", "seed_2024"]

FAMILIES = ["xception", "resnet", "mobilenet", "vit", "dino", "clip"]
METHODS = [
    ("rgb", "RGB"),
    ("magnitude", "Log-Magnitude"),
    ("phase", "Phase"),
    ("complex", "Complex Spectrum"),
    ("concat", "RGB+Mag"),
    ("frequency_3", "High-Pass Spectrum"),
    ("concat_freq", "RGB+Freq"),
    ("rgb_robust", "RGB"),          # Protocolo 2 / c/DA
    ("srm", "RGB+SRM"),             # Protocolo 2 / c/DA
    ("srm_only", "SRM-only"),       # Protocolo 2 / c/DA
]

def load_3seeds_data():
    records = []
    for p in TABLES_DIR.glob("df40_all_fakes_*.csv"):
        name = p.stem.replace("df40_all_fakes_", "")
        m = re.match(r"^(.*)_(seed_\d+)$", name)
        if m:
            prefix, seed = m.group(1), m.group(2)
        else:
            prefix, seed = name, "seed_42"
        
        if seed in SEEDS:
            try:
                df = pd.read_csv(p)
                df["config_prefix"] = prefix
                df["seed_id"] = seed
                records.append(df)
            except Exception as e:
                print(f"Erro ao ler {p.name}: {e}")

    df_all = pd.concat(records, ignore_index=True).drop_duplicates(
        subset=["config_prefix", "seed_id", "paradigma", "tecnica"]
    )
    return df_all

def compute_paradigm_and_total(df_all: pd.DataFrame):
    # 1. Paradigmas por seed
    par_rows = []
    for (prefix, seed, parad), grp in df_all.groupby(["config_prefix", "seed_id", "paradigma"]):
        tot = grp["n_amostras_total"].sum()
        corr = grp["acertos_threshold_val"].sum()
        rate = (corr / tot * 100.0) if tot > 0 else 0.0
        par_rows.append({"config_prefix": prefix, "seed_id": seed, "paradigma": parad, "rate": rate})

    # Total geral por seed
    for (prefix, seed), grp in df_all.groupby(["config_prefix", "seed_id"]):
        tot = grp["n_amostras_total"].sum()
        corr = grp["acertos_threshold_val"].sum()
        rate = (corr / tot * 100.0) if tot > 0 else 0.0
        par_rows.append({"config_prefix": prefix, "seed_id": seed, "paradigma": "TOTAL_GERAL", "rate": rate})

    df_par = pd.DataFrame(par_rows)
    pivot = df_par.pivot_table(index=["config_prefix", "seed_id"], columns="paradigma", values="rate").reset_index()

    # Estatísticas
    stats = {}
    for prefix, grp in pivot.groupby("config_prefix"):
        stats[prefix] = {}
        for col in grp.columns:
            if col in ["config_prefix", "seed_id"]:
                continue
            vals = grp[col].dropna()
            stats[prefix][f"{col}_mean"] = vals.mean()
            stats[prefix][f"{col}_std"] = vals.std()

    return stats, pivot

def update_summary_csvs(df_all: pd.DataFrame, pivot: pd.DataFrame):
    paradigms = sorted([c for c in pivot.columns if c not in ["config_prefix", "seed_id"]])
    configs = sorted(df_all["config_prefix"].unique())

    # 1. Por Paradigma
    rows_par = []
    for prefix in configs:
        sub = pivot[pivot["config_prefix"] == prefix]
        row = {"model_name": prefix, "n_seeds": len(sub)}
        for p in paradigms:
            vals = sub[p].dropna()
            if len(vals) == 0:
                row[p] = "N/A"
            elif len(vals) == 1:
                row[p] = f"{vals.iloc[0]:.2f}%"
            else:
                row[p] = f"{vals.mean():.2f} ± {vals.std():.2f}%"
        rows_par.append(row)

    df_out_par = pd.DataFrame(rows_par)
    out_par_csv = OUT_TABLES_DIR / "df40_all_fakes_seeds_summary_by_paradigm.csv"
    df_out_par.to_csv(out_par_csv, index=False)
    print(f"Salvo resumo por paradigma em: {out_par_csv}")

    # 2. Por Técnica
    rows_tech = []
    for prefix in configs:
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
        rows_tech.append(row)

    df_out_tech = pd.DataFrame(rows_tech)
    out_tech_csv = OUT_TABLES_DIR / "df40_all_fakes_seeds_summary_by_technique.csv"
    df_out_tech.to_csv(out_tech_csv, index=False)
    print(f"Salvo resumo por técnica em: {out_tech_csv}")

def update_familias_representacoes_tex(stats: dict):
    tex_path = RESULTS_DIR / "familias_representacoes.tex"
    with open(tex_path, "r", encoding="utf-8") as f:
        lines = f.readlines()

    current_fam = None
    new_lines = []

    # Mapeamento de ordem de representações para cada família
    rep_order = [
        "rgb", "magnitude", "phase", "complex", "concat",
        "frequency_3", "concat_freq", "rgb_robust", "srm", "srm_only"
    ]

    fam_counts = {fam: 0 for fam in FAMILIES}

    for line in lines:
        stripped = line.strip()
        # Detectar família pelo início da linha
        for fam in FAMILIES:
            if stripped.lower().startswith(fam) or (fam == "mobilenet" and "mobilenetv3" in stripped.lower()) or (fam == "resnet" and "resnet-18" in stripped.lower()):
                current_fam = fam
                fam_counts[current_fam] = 0
                break

        if "&" in line and current_fam is not None and fam_counts[current_fam] < len(rep_order):
            parts = line.split("&")
            if len(parts) >= 12:
                rep_idx = fam_counts[current_fam]
                method_key = rep_order[rep_idx]
                model_name = f"{current_fam}_{method_key}"

                if model_name in stats:
                    m = stats[model_name]["TOTAL_GERAL_mean"] / 100.0
                    s = stats[model_name]["TOTAL_GERAL_std"] / 100.0

                    m_str = f"{m:.3f}".replace(".", ",")
                    s_str = f"{s:.3f}".replace(".", ",")

                    # Destaque em negrito para o campeão válido: dino_concat_freq (0,934)
                    if model_name == "dino_concat_freq":
                        new_cell = f" \\pmsd{{\\textbf{{{m_str}}}}}{{{s_str}}} "
                    else:
                        new_cell = f" \\pmsd{{{m_str}}}{{{s_str}}} "

                    parts[9] = new_cell
                    line = "&".join(parts)
                    fam_counts[current_fam] += 1

        new_lines.append(line)

    with open(tex_path, "w", encoding="utf-8") as f:
        f.writelines(new_lines)
    print(f"Atualizado: {tex_path}")

def update_paper_tcc_versao_final(stats: dict):
    tex_path = PAPER_DIR / "TCC_versao_final.tex"
    with open(tex_path, "r", encoding="utf-8") as f:
        lines = f.readlines()

    current_fam = None
    new_lines = []
    in_target_table = False

    rep_order = [
        "rgb", "magnitude", "phase", "complex", "concat",
        "frequency_3", "concat_freq", "rgb_robust", "srm", "srm_only"
    ]
    fam_counts = {fam: 0 for fam in FAMILIES}

    for line in lines:
        if "\\label{tab:familias_representacoes}" in line:
            in_target_table = True

        if in_target_table and "\\bottomrule" in line:
            in_target_table = False

        if in_target_table:
            stripped = line.strip()
            for fam in FAMILIES:
                if stripped.lower().startswith(fam) or (fam == "mobilenet" and "mobilenetv3" in stripped.lower()) or (fam == "resnet" and "resnet-18" in stripped.lower()):
                    current_fam = fam
                    fam_counts[current_fam] = 0
                    break

            if "&" in line and current_fam is not None and fam_counts[current_fam] < len(rep_order):
                parts = line.split("&")
                if len(parts) >= 8:
                    rep_idx = fam_counts[current_fam]
                    method_key = rep_order[rep_idx]
                    model_name = f"{current_fam}_{method_key}"

                    if model_name in stats:
                        m = stats[model_name]["TOTAL_GERAL_mean"]
                        s = stats[model_name]["TOTAL_GERAL_std"]

                        m_str = f"{m:.1f}".replace(".", ",")
                        s_str = f"{s:.1f}".replace(".", ",")

                        if model_name == "dino_concat_freq":
                            new_cell = f" \\pmsd{{\\textbf{{{m_str}}}}}{{{s_str}}} "
                        else:
                            new_cell = f" \\pmsd{{{m_str}}}{{{s_str}}} "

                        parts[6] = new_cell
                        line = "&".join(parts)
                        fam_counts[current_fam] += 1

        new_lines.append(line)

    with open(tex_path, "w", encoding="utf-8") as f:
        f.writelines(new_lines)
    print(f"Atualizado: {tex_path}")

def update_df40_paradigmas_tex(stats: dict):
    tex_path = RESULTS_DIR / "df40_paradigmas.tex"
    with open(tex_path, "r", encoding="utf-8") as f:
        lines = f.readlines()

    current_fam = None
    new_lines = []

    rep_order = [
        "rgb", "magnitude", "phase", "complex", "concat",
        "frequency_3", "concat_freq", "rgb_robust", "srm", "srm_only"
    ]
    fam_counts = {fam: 0 for fam in FAMILIES}

    # Paradigmas correspondentes às colunas 3 a 8 (Difusão, Síntese GAN, Troca Facial, Edição / T2I, Avatares, Total Geral)
    cols = ["Difusão", "Síntese GAN", "Troca Facial", "Edição / T2I", "Avatares & Reencenação", "TOTAL_GERAL"]

    for line in lines:
        stripped = line.strip()
        for fam in FAMILIES:
            if stripped.lower().startswith(fam) or (fam == "mobilenet" and "mobilenetv3" in stripped.lower()) or (fam == "resnet" and "resnet-18" in stripped.lower()):
                current_fam = fam
                fam_counts[current_fam] = 0
                break

        if "&" in line and current_fam is not None and fam_counts[current_fam] < len(rep_order):
            parts = line.split("&")
            if len(parts) >= 9:
                rep_idx = fam_counts[current_fam]
                method_key = rep_order[rep_idx]
                model_name = f"{current_fam}_{method_key}"

                if model_name in stats:
                    for i, col in enumerate(cols):
                        m = stats[model_name][f"{col}_mean"]
                        s = stats[model_name][f"{col}_std"]
                        m_str = f"{m:.1f}".replace(".", ",")
                        s_str = f"{s:.1f}".replace(".", ",")

                        # Destaques em negrito dos campeões válidos
                        # Difusão: dino_concat_freq (90.7%) ou xception_srm_only (91.7%)
                        # Síntese GAN: dino_concat_freq (96.1%)
                        # Troca Facial: dino_rgb_robust (95.1%) / dino_concat_freq (95.1%)
                        # Edição / T2I: dino_rgb (100.0%) / vit_magnitude (99.4%)
                        # Avatares: xception_srm_only (94.8%)
                        # Total Geral: dino_concat_freq (93.4%)
                        bold = False
                        if col == "TOTAL_GERAL" and model_name == "dino_concat_freq":
                            bold = True
                        elif col == "Síntese GAN" and model_name == "dino_concat_freq":
                            bold = True
                        elif col == "Difusão" and model_name == "xception_srm_only":
                            bold = True
                        elif col == "Avatares & Reencenação" and model_name == "xception_srm_only":
                            bold = True

                        if bold:
                            cell_text = f" \\pmsd{{\\textbf{{{m_str}\\%}}}}{{{s_str}\\%}} "
                        else:
                            cell_text = f" \\pmsd{{{m_str}\\%}}{{{s_str}\\%}} "

                        if i == len(cols) - 1:
                            parts[3 + i] = cell_text + "\\\\\n"
                        else:
                            parts[3 + i] = cell_text

                    line = "&".join(parts)
                    fam_counts[current_fam] += 1

        new_lines.append(line)

    with open(tex_path, "w", encoding="utf-8") as f:
        f.writelines(new_lines)
    print(f"Atualizado: {tex_path}")


def main():
    print("Carregando resultados das 3 primeiras sementes (seed_42, seed_123, seed_2024)...")
    df_all = load_3seeds_data()
    print(f"Total registros: {len(df_all)}")
    stats, pivot = compute_paradigm_and_total(df_all)

    print("\n1. Atualizando CSVs de resumo...")
    update_summary_csvs(df_all, pivot)

    print("\n2. Atualizando results/tables/familias_representacoes.tex...")
    update_familias_representacoes_tex(stats)

    print("\n3. Atualizando paper/TCC_versao_final.tex...")
    update_paper_tcc_versao_final(stats)

    print("\n4. Atualizando results/tables/df40_paradigmas.tex...")
    update_df40_paradigmas_tex(stats)

    print("\nSucesso!")

if __name__ == "__main__":
    main()
