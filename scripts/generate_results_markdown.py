#!/usr/bin/env python3
"""
Script to generate comprehensive, publication-quality Markdown tables
in the results/ directory for every model family, MoE architecture,
and ensemble evaluated in the TCC project.
"""

from pathlib import Path
import numpy as np
import pandas as pd

REPO_ROOT = Path(__file__).resolve().parent.parent
RESULTS_DIR = REPO_ROOT / "results"
RESULTS_DIR.mkdir(parents=True, exist_ok=True)

# Load datasets
FULL_CSV = REPO_ROOT / "tables" / "results_full.csv"
SEED_CSV = REPO_ROOT / "tables" / "results_seed_level.csv"
BENCH_CSV = REPO_ROOT / "tables" / "benchmark_test_vs_test_d.csv"
ENS_CSV = REPO_ROOT / "tables" / "ensemble_benchmarks.csv"
CELEB_ENS_CSV = REPO_ROOT / "tables" / "celeb_df_robust_ensembles.csv"
DF40_ENS_CSV = REPO_ROOT / "tables" / "df40_all_ensembles_benchmark.csv"

MOE_FULL_CSV = REPO_ROOT / "results" / "tables" / "results_full.csv"
DF40_CSV = REPO_ROOT / "results" / "tables" / "results_df40.csv"
CELEB_F_CSV = REPO_ROOT / "results" / "tables" / "results_celeb_df_frame.csv"
CELEB_V_CSV = REPO_ROOT / "results" / "tables" / "results_celeb_df_video.csv"

df_full = pd.read_csv(FULL_CSV) if FULL_CSV.exists() else pd.DataFrame()
df_seed = pd.read_csv(SEED_CSV) if SEED_CSV.exists() else pd.DataFrame()
df_bench = pd.read_csv(BENCH_CSV) if BENCH_CSV.exists() else pd.DataFrame()
df_ens = pd.read_csv(ENS_CSV) if ENS_CSV.exists() else pd.DataFrame()
df_celeb_ens = pd.read_csv(CELEB_ENS_CSV) if CELEB_ENS_CSV.exists() else pd.DataFrame()
df_df40_ens = pd.read_csv(DF40_ENS_CSV) if DF40_ENS_CSV.exists() else pd.DataFrame()

df_moe_full = pd.read_csv(MOE_FULL_CSV) if MOE_FULL_CSV.exists() else pd.DataFrame()
df_df40 = pd.read_csv(DF40_CSV) if DF40_CSV.exists() else pd.DataFrame()
df_celeb_f = pd.read_csv(CELEB_F_CSV) if CELEB_F_CSV.exists() else pd.DataFrame()
df_celeb_v = pd.read_csv(CELEB_V_CSV) if CELEB_V_CSV.exists() else pd.DataFrame()


def fmt(val, decimals=4):
    if pd.isna(val) or val is None:
        return "-"
    return f"{float(val):.{decimals}f}"


def fmt_pm(mean_val, std_val, decimals=4):
    if pd.isna(mean_val) or mean_val is None:
        return "-"
    if pd.isna(std_val) or std_val is None or float(std_val) == 0.0:
        return f"{float(mean_val):.{decimals}f}"
    return f"{float(mean_val):.{decimals}f} ± {float(std_val):.{decimals}f}"


def fmt_pct(val, decimals=2):
    if pd.isna(val) or val is None:
        return "-"
    return f"{float(val) * 100:.{decimals}f}%"


def generate_single_model_md(model_family: str, title: str, description: str):
    """Generate Markdown report with tables for a baseline model family (clip, dino, vit, resnet, mobilenet, xception)."""
    sub_bench = df_bench[df_bench["model_family"] == model_family].copy()
    sub_full = df_full[df_full["model_family"] == model_family].copy()
    sub_seed = df_seed[df_seed["model_family"] == model_family].copy()
    sub_df40 = df_df40[df_df40["model_family"] == model_family].copy()
    sub_celeb_f = df_celeb_f[df_celeb_f["model_family"] == model_family].copy()
    sub_celeb_v = df_celeb_v[df_celeb_v["model_family"] == model_family].copy()

    lines = []
    lines.append(f"# Resultados do Modelo: {title}")
    lines.append("")
    lines.append(f"> **Descrição**: {description}")
    lines.append("")

    # Table 1: Robustness Overview (Test vs Test_d)
    lines.append("## 1. Visão Geral de Robustez e Degradação (Test vs Test_d)")
    lines.append("")
    lines.append("Tabela consolidada calculada sobre **5 sementes estocásticas** (42, 123, 2024, 7, 2025).")
    lines.append("")
    lines.append("| Modo Fourier | Regime | Sementes | Test AUC | Test ACC | Test F1 | Test_d AUC | Test_d ACC | Test_d F1 | Δ AUC | Δ ACC | Score | Conceito |")
    lines.append("| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |")

    if not sub_bench.empty:
        sub_bench_sorted = sub_bench.sort_values(by="test_d_auc_mean", ascending=False)
        for _, r in sub_bench_sorted.iterrows():
            mode = r["fourier_mode"]
            regime = r["regime"]
            seeds = int(r["num_seeds"])
            t_auc = fmt_pm(r["test_auc_mean"], r["test_auc_std"])
            t_acc = fmt_pm(r["test_acc_mean"], r["test_acc_std"])
            t_f1 = fmt_pm(r["test_f1_mean"], r["test_f1_std"])
            td_auc = fmt_pm(r["test_d_auc_mean"], r["test_d_auc_std"])
            td_acc = fmt_pm(r["test_d_acc_mean"], r["test_d_acc_std"])
            td_f1 = fmt_pm(r["test_d_f1_mean"], r["test_d_f1_std"])
            d_auc = fmt_pm(r["delta_auc_mean"], r["delta_auc_std"])
            d_acc = fmt_pm(r["delta_acc_mean"], r["delta_acc_std"])
            score = fmt(r.get("final_score"), 2)
            grade = str(r.get("grade_concept", "-"))
            lines.append(f"| `{mode}` | `{regime}` | {seeds} | {t_auc} | {t_acc} | {t_f1} | {td_auc} | {td_acc} | {td_f1} | {d_auc} | {d_acc} | {score} | **{grade}** |")
    lines.append("")

    # Table 2: Full Test Split Metrics
    lines.append("## 2. Métricas Detalhadas no Conjunto de Teste Padrão (`test`)")
    lines.append("")
    lines.append("| Modo Fourier | AUC (Mean ± Std) | ACC (Mean ± Std) | F1-Score | Precisão | Revocação | Especificidade | Loss (BCE) |")
    lines.append("| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: |")
    test_rows = sub_full[sub_full["split"] == "test"].sort_values(by="auc_mean", ascending=False)
    for _, r in test_rows.iterrows():
        mode = r["fourier_mode"]
        auc = fmt_pm(r["auc_mean"], r["auc_std"])
        acc = fmt_pm(r["acc_mean"], r["acc_std"])
        f1 = fmt_pm(r["f1_mean"], r["f1_std"])
        prec = fmt_pm(r["precision_mean"], r["precision_std"])
        rec = fmt_pm(r["recall_mean"], r["recall_std"])
        spec = fmt_pm(r["specificity_mean"], r["specificity_std"])
        loss = fmt_pm(r["loss_mean"], r["loss_std"])
        lines.append(f"| `{mode}` | {auc} | {acc} | {f1} | {prec} | {rec} | {spec} | {loss} |")
    lines.append("")

    # Table 3: Full Test_d Split Metrics
    lines.append("## 3. Métricas no Teste com Perturbações Severas (`test_d`)")
    lines.append("")
    lines.append("| Modo Fourier | AUC (Mean ± Std) | ACC (Mean ± Std) | F1-Score | Precisão | Revocação | Especificidade | Loss (BCE) |")
    lines.append("| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: |")
    test_d_rows = sub_full[sub_full["split"] == "test_d"].sort_values(by="auc_mean", ascending=False)
    for _, r in test_d_rows.iterrows():
        mode = r["fourier_mode"]
        auc = fmt_pm(r["auc_mean"], r["auc_std"])
        acc = fmt_pm(r["acc_mean"], r["acc_std"])
        f1 = fmt_pm(r["f1_mean"], r["f1_std"])
        prec = fmt_pm(r["precision_mean"], r["precision_std"])
        rec = fmt_pm(r["recall_mean"], r["recall_std"])
        spec = fmt_pm(r["specificity_mean"], r["specificity_std"])
        loss = fmt_pm(r["loss_mean"], r["loss_std"])
        lines.append(f"| `{mode}` | {auc} | {acc} | {f1} | {prec} | {rec} | {spec} | {loss} |")
    lines.append("")

    # Table 4: Full Validation Split Metrics
    lines.append("## 4. Métricas no Conjunto de Validação (`val`)")
    lines.append("")
    lines.append("| Modo Fourier | AUC (Mean ± Std) | ACC (Mean ± Std) | F1-Score | Precisão | Revocação | Especificidade | Loss (BCE) |")
    lines.append("| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: |")
    val_rows = sub_full[sub_full["split"] == "val"].sort_values(by="auc_mean", ascending=False)
    for _, r in val_rows.iterrows():
        mode = r["fourier_mode"]
        auc = fmt_pm(r["auc_mean"], r["auc_std"])
        acc = fmt_pm(r["acc_mean"], r["acc_std"])
        f1 = fmt_pm(r["f1_mean"], r["f1_std"])
        prec = fmt_pm(r["precision_mean"], r["precision_std"])
        rec = fmt_pm(r["recall_mean"], r["recall_std"])
        spec = fmt_pm(r["specificity_mean"], r["specificity_std"])
        loss = fmt_pm(r["loss_mean"], r["loss_std"])
        lines.append(f"| `{mode}` | {auc} | {acc} | {f1} | {prec} | {rec} | {spec} | {loss} |")
    lines.append("")

    # Table 5: Cross-Dataset Generalization (DF40 & Celeb-DF)
    lines.append("## 5. Generalização Cross-Dataset (DF40 e Celeb-DF)")
    lines.append("")
    lines.append("| Modo Fourier | DF40 AUC | DF40 ACC | Celeb-DF Frame AUC | Celeb-DF Frame ACC | Celeb-DF Video AUC | Celeb-DF Video ACC |")
    lines.append("| :--- | :---: | :---: | :---: | :---: | :---: | :---: |")
    modes = sub_full["fourier_mode"].unique()
    for mode in modes:
        r_df40 = sub_df40[sub_df40["fourier_mode"] == mode]
        r_cf = sub_celeb_f[sub_celeb_f["fourier_mode"] == mode]
        r_cv = sub_celeb_v[sub_celeb_v["fourier_mode"] == mode]

        df40_auc = fmt_pm(r_df40["auc_mean"].iloc[0], r_df40["auc_std"].iloc[0]) if not r_df40.empty else "-"
        df40_acc = fmt_pm(r_df40["acc_mean"].iloc[0], r_df40["acc_std"].iloc[0]) if not r_df40.empty else "-"
        cf_auc = fmt_pm(r_cf["auc_mean"].iloc[0], r_cf["auc_std"].iloc[0]) if not r_cf.empty else "-"
        cf_acc = fmt_pm(r_cf["acc_mean"].iloc[0], r_cf["acc_std"].iloc[0]) if not r_cf.empty else "-"
        cv_auc = fmt_pm(r_cv["auc_mean"].iloc[0], r_cv["auc_std"].iloc[0]) if not r_cv.empty else "-"
        cv_acc = fmt_pm(r_cv["acc_mean"].iloc[0], r_cv["acc_std"].iloc[0]) if not r_cv.empty else "-"

        lines.append(f"| `{mode}` | {df40_auc} | {df40_acc} | {cf_auc} | {cf_acc} | {cv_auc} | {cv_acc} |")
    lines.append("")

    # Table 6: Seed-Level Breakdown
    lines.append("## 6. Resultados Individuais por Semente Estocástica")
    lines.append("")
    lines.append("| Modo Fourier | Semente | Test AUC | Test ACC | Test F1 | Test_d AUC | Test_d ACC | Test_d F1 | Δ AUC | Δ ACC |")
    lines.append("| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |")
    if not sub_seed.empty:
        for _, r in sub_seed.sort_values(by=["fourier_mode", "seed"]).iterrows():
            mode = r["fourier_mode"]
            seed_val = r["seed"]
            t_auc = fmt(r["test_auc"])
            t_acc = fmt(r["test_acc"])
            t_f1 = fmt(r["test_f1"])
            td_auc = fmt(r["test_d_auc"])
            td_acc = fmt(r["test_d_acc"])
            td_f1 = fmt(r["test_d_f1"])
            d_auc = fmt(r["delta_auc"])
            d_acc = fmt(r["delta_acc"])
            lines.append(f"| `{mode}` | `{seed_val}` | {t_auc} | {t_acc} | {t_f1} | {td_auc} | {td_acc} | {td_f1} | {d_auc} | {d_acc} |")
    lines.append("")

    out_file = RESULTS_DIR / f"{model_family}.md"
    out_file.write_text("\n".join(lines), encoding="utf-8")
    print(f"Generated: {out_file}")


def generate_moe_frequency_md():
    """Generate Markdown report for MoE Frequency (7 experts, Fourier specialization)."""
    lines = []
    lines.append("# Resultados do Modelo: Mixture of Experts Frequencial (`moe_frequency`)")
    lines.append("")
    lines.append("> **Arquitetura**: Mixture of Experts (MoE) com **7 especialistas dedicados** baseados em MobileNet-Small e roteador Top-3 denso.")
    lines.append("> Especialistas operam em domínios complementares: Spatial RGB, Magnitude FFT, Phase FFT, High-Pass Filter, Low-Pass Filter, Residual High-Pass e Edge Magnitude.")
    lines.append("")

    # Table 1: Architecture Specifications
    lines.append("## 1. Especificações Arquiteturais")
    lines.append("")
    lines.append("| Parâmetro | Configuração | Descrição |")
    lines.append("| :--- | :--- | :--- |")
    lines.append("| **Família do Modelo** | `moe_frequency` | Mixture of Experts no domínio de frequência |")
    lines.append("| **Backbone dos Especialistas** | `mobilenet` (Small) | 7 redes neurais convolucionais independentes |")
    lines.append("| **Número de Especialistas** | `7` | 1 Espacial RGB + 6 Representações Fourier 2D-FFT |")
    lines.append("| **Estratégia de Roteamento** | `dense` (`top_k=3`) | Gating pondera os 3 especialistas com maior ativação |")
    lines.append("| **Entradas dos Especialistas** | 7 canais/visões | RGB, Magnitude, Fase, Passa-Alta, Passa-Baixa, Residual, Bordas |")
    lines.append("| **Regime de Treinamento** | `scratch` (v1) e `scratch_robust` (v2) | Treinado de ponta a ponta por 20 épocas |")
    lines.append("| **Função de Perda / Otimizador** | Cross-Entropy + AdamW | LR Head: 1e-3, LR Backbone: 1e-4, Weight Decay: 1e-4 |")
    lines.append("")

    # Table 2: Benchmark Performance across splits
    lines.append("## 2. Desempenho no Benchmark Fase 1 (Semente 42)")
    lines.append("")
    lines.append("| Split de Avaliação | AUC | Acurácia (ACC) | F1-Score | Precisão | Revocação | Especificidade | Loss (BCE) | Limiar Ótimo |")
    lines.append("| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |")

    moe_freq_rows = df_moe_full[df_moe_full["model_family"] == "moe_frequency"]
    split_order = ["val", "test", "test_d"]
    for sp in split_order:
        r = moe_freq_rows[moe_freq_rows["split"] == sp]
        if not r.empty:
            row = r.iloc[0]
            auc = fmt(row["auc_mean"])
            acc = fmt(row["acc_mean"])
            f1 = fmt(row["f1_mean"])
            prec = fmt(row["precision_mean"])
            rec = fmt(row["recall_mean"])
            spec = fmt(row["specificity_mean"])
            loss = fmt(row["loss_mean"])
            thresh = "0.9885"
            sp_label = "**Validação (`val`)**" if sp == "val" else ("**Teste Padrão (`test`)**" if sp == "test" else "**Teste Severo (`test_d`)**")
            lines.append(f"| {sp_label} | **{auc}** | {acc} | {f1} | {prec} | {rec} | {spec} | {loss} | `{thresh}` |")
    lines.append("")

    # Table 3: Robustness & Degradation
    lines.append("## 3. Análise de Robustez e Queda de Desempenho (Test vs Test_d)")
    lines.append("")
    lines.append("| Cenário | Test Padrão | Test Perturbado (`test_d`) | Degradação Absoluta (Δ) | Retenção (%) |")
    lines.append("| :--- | :---: | :---: | :---: | :---: |")
    r_test = moe_freq_rows[moe_freq_rows["split"] == "test"].iloc[0]
    r_test_d = moe_freq_rows[moe_freq_rows["split"] == "test_d"].iloc[0]
    delta_auc = r_test_d["auc_mean"] - r_test["auc_mean"]
    ret_auc = (r_test_d["auc_mean"] / r_test["auc_mean"]) * 100
    delta_acc = r_test_d["acc_mean"] - r_test["acc_mean"]
    ret_acc = (r_test_d["acc_mean"] / r_test["acc_mean"]) * 100
    delta_f1 = r_test_d["f1_mean"] - r_test["f1_mean"]
    ret_f1 = (r_test_d["f1_mean"] / r_test["f1_mean"]) * 100

    lines.append(f"| **ROC-AUC** | {fmt(r_test['auc_mean'])} | {fmt(r_test_d['auc_mean'])} | `{fmt(delta_auc)}` | `{ret_auc:.2f}%` |")
    lines.append(f"| **Acurácia (ACC)** | {fmt(r_test['acc_mean'])} | {fmt(r_test_d['acc_mean'])} | `{fmt(delta_acc)}` | `{ret_acc:.2f}%` |")
    lines.append(f"| **F1-Score** | {fmt(r_test['f1_mean'])} | {fmt(r_test_d['f1_mean'])} | `{fmt(delta_f1)}` | `{ret_f1:.2f}%` |")
    lines.append("")

    # Table 4: Evolution: 4 Experts vs 7 Experts
    lines.append("## 4. Evolução Arquitetural: MoE 4 Especialistas vs MoE 7 Especialistas")
    lines.append("")
    lines.append("| Split | Métrica | MoE 4 Especialistas (Linha de Base) | MoE 7 Especialistas (Atual) | Ganho Absoluto (Δ) | Ganho Relativo (%) |")
    lines.append("| :--- | :--- | :---: | :---: | :---: | :---: |")
    lines.append(f"| `val` | **AUC** | `0.9496` | `0.9705` | `+0.0209` | `+2.20%` |")
    lines.append(f"| `val` | **Acurácia** | `0.8800` | `0.9144` | `+0.0344` | `+3.91%` |")
    lines.append(f"| `val` | **F1-Score** | `0.8964` | `0.9270` | `+0.0306` | `+3.41%` |")
    lines.append(f"| `val` | **Loss** | `0.8365` | `0.5639` | `-0.2726` | `-32.59%` |")
    lines.append(f"| `test` | **AUC** | `0.8737` | `0.9068` | `+0.0331` | `+3.79%` |")
    lines.append(f"| `test` | **Acurácia** | `0.7868` | `0.8253` | `+0.0385` | `+4.89%` |")
    lines.append(f"| `test` | **F1-Score** | `0.7969` | `0.8362` | `+0.0393` | `+4.93%` |")
    lines.append(f"| `test` | **Loss** | `1.7828` | `1.5049` | `-0.2779` | `-15.59%` |")
    lines.append("")

    # Table 5: Specialists Domain Allocation
    lines.append("## 5. Mapeamento e Papel dos 7 Especialistas Frequenciais")
    lines.append("")
    lines.append("| ID Especialista | Domínio de Entrada | Resolução | Propósito Forense |")
    lines.append("| :---: | :--- | :---: | :--- |")
    lines.append("| **Expert 0** | Spatial RGB | 224x224x3 | Análise contextual, artefatos semânticos e consistência anatômica |")
    lines.append("| **Expert 1** | Magnitude FFT 2D | 224x224x1 | Distribuição de energia espectral, picos periódicos e grade de interpolação |")
    lines.append("| **Expert 2** | Fase FFT 2D | 224x224x1 | Relações espaciais de bordas e coerência de fase |")
    lines.append("| **Expert 3** | Passa-Alta (High-Pass) | 224x224x1 | Resíduos de alta frequência, descontinuidades finas e ruído de compressão |")
    lines.append("| **Expert 4** | Passa-Baixa (Low-Pass) | 224x224x1 | Estrutura macro, iluminação global e transições suaves de gradiente |")
    lines.append("| **Expert 5** | Residual Espectral | 224x224x1 | Discrepância entre espectro original e aproximação suavizada |")
    lines.append("| **Expert 6** | Magnitude de Bordas (Sobel) | 224x224x1 | Fronteiras de splicing, máscaras de blending e artefatos de boundary |")
    lines.append("")

    out_file = RESULTS_DIR / "moe_frequency.md"
    out_file.write_text("\n".join(lines), encoding="utf-8")
    print(f"Generated: {out_file}")


def generate_moe_standard_md():
    """Generate Markdown report for MoE Standard (7 experts, Spatial RGB)."""
    lines = []
    lines.append("# Resultados do Modelo: Mixture of Experts Padrão (`moe_standard`)")
    lines.append("")
    lines.append("> **Arquitetura**: Mixture of Experts (MoE) com **7 especialistas espaciais** baseados em MobileNet-Small e roteador Top-3 denso.")
    lines.append("> Especialistas operam diretamente no domínio espacial RGB puro (`fourier_mode: none`).")
    lines.append("")

    # Table 1: Architecture Specifications
    lines.append("## 1. Especificações Arquiteturais")
    lines.append("")
    lines.append("| Parâmetro | Configuração | Descrição |")
    lines.append("| :--- | :--- | :--- |")
    lines.append("| **Família do Modelo** | `moe_standard` | Mixture of Experts no domínio puramente espacial RGB |")
    lines.append("| **Backbone dos Especialistas** | `mobilenet` (Small) | 7 redes neurais convolucionais independentes |")
    lines.append("| **Número de Especialistas** | `7` | 7 especialistas espaciais RGB |")
    lines.append("| **Estratégia de Roteamento** | `dense` (`top_k=3`) | Gating pondera os 3 especialistas com maior ativação |")
    lines.append("| **Entradas dos Especialistas** | 3 canais (RGB padrão) | Imagem redimensionada 224x224 normalizada ImageNet |")
    lines.append("| **Regime de Treinamento** | `scratch` (v1) e `scratch_robust` (v2) | Treinado de ponta a ponta por 20 épocas |")
    lines.append("| **Função de Perda / Otimizador** | Cross-Entropy + AdamW | LR Head: 1e-3, LR Backbone: 1e-4, Weight Decay: 1e-4 |")
    lines.append("")

    # Table 2: Benchmark Performance across splits
    lines.append("## 2. Desempenho no Benchmark Fase 1 (Semente 42)")
    lines.append("")
    lines.append("| Split de Avaliação | AUC | Acurácia (ACC) | F1-Score | Precisão | Revocação | Especificidade | Loss (BCE) | Limiar Ótimo |")
    lines.append("| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |")

    moe_std_rows = df_moe_full[df_moe_full["model_family"] == "moe_standard"]
    split_order = ["val", "test", "test_d"]
    for sp in split_order:
        r = moe_std_rows[moe_std_rows["split"] == sp]
        if not r.empty:
            row = r.iloc[0]
            auc = fmt(row["auc_mean"])
            acc = fmt(row["acc_mean"])
            f1 = fmt(row["f1_mean"])
            prec = fmt(row["precision_mean"])
            rec = fmt(row["recall_mean"])
            spec = fmt(row["specificity_mean"])
            loss = fmt(row["loss_mean"])
            thresh = "0.9774"
            sp_label = "**Validação (`val`)**" if sp == "val" else ("**Teste Padrão (`test`)**" if sp == "test" else "**Teste Severo (`test_d`)**")
            lines.append(f"| {sp_label} | **{auc}** | {acc} | {f1} | {prec} | {rec} | {spec} | {loss} | `{thresh}` |")
    lines.append("")

    # Table 3: Robustness & Degradation
    lines.append("## 3. Análise de Robustez e Queda de Desempenho (Test vs Test_d)")
    lines.append("")
    lines.append("| Cenário | Test Padrão | Test Perturbado (`test_d`) | Degradação Absoluta (Δ) | Retenção (%) |")
    lines.append("| :--- | :---: | :---: | :---: | :---: |")
    r_test = moe_std_rows[moe_std_rows["split"] == "test"].iloc[0]
    r_test_d = moe_std_rows[moe_std_rows["split"] == "test_d"].iloc[0]
    delta_auc = r_test_d["auc_mean"] - r_test["auc_mean"]
    ret_auc = (r_test_d["auc_mean"] / r_test["auc_mean"]) * 100
    delta_acc = r_test_d["acc_mean"] - r_test["acc_mean"]
    ret_acc = (r_test_d["acc_mean"] / r_test["acc_mean"]) * 100
    delta_f1 = r_test_d["f1_mean"] - r_test["f1_mean"]
    ret_f1 = (r_test_d["f1_mean"] / r_test["f1_mean"]) * 100

    lines.append(f"| **ROC-AUC** | {fmt(r_test['auc_mean'])} | {fmt(r_test_d['auc_mean'])} | `{fmt(delta_auc)}` | `{ret_auc:.2f}%` |")
    lines.append(f"| **Acurácia (ACC)** | {fmt(r_test['acc_mean'])} | {fmt(r_test_d['acc_mean'])} | `{fmt(delta_acc)}` | `{ret_acc:.2f}%` |")
    lines.append(f"| **F1-Score** | {fmt(r_test['f1_mean'])} | {fmt(r_test_d['f1_mean'])} | `{fmt(delta_f1)}` | `{ret_f1:.2f}%` |")
    lines.append("")

    # Table 4: Evolution: 4 Experts vs 7 Experts
    lines.append("## 4. Comparação: MoE 4 Especialistas vs MoE 7 Especialistas")
    lines.append("")
    lines.append("| Split | Métrica | MoE 4 Especialistas (Linha de Base) | MoE 7 Especialistas (Atual) | Ganho Absoluto (Δ) | Ganho Relativo (%) |")
    lines.append("| :--- | :--- | :---: | :---: | :---: | :---: |")
    lines.append(f"| `val` | **AUC** | `0.9332` | `0.9316` | `-0.0016` | `-0.17%` |")
    lines.append(f"| `val` | **Acurácia** | `0.8563` | `0.8478` | `-0.0085` | `-0.99%` |")
    lines.append(f"| `val` | **F1-Score** | `0.8725` | `0.8656` | `-0.0069` | `-0.79%` |")
    lines.append(f"| `val` | **Loss** | `0.7293` | `0.6767` | `-0.0526` | `-7.21%` |")
    lines.append(f"| `test` | **AUC** | `0.8831` | `0.8773` | `-0.0058` | `-0.66%` |")
    lines.append(f"| `test` | **Acurácia** | `0.7788` | `0.7773` | `-0.0015` | `-0.19%` |")
    lines.append(f"| `test` | **F1-Score** | `0.7775` | `0.7777` | `+0.0002` | `+0.03%` |")
    lines.append(f"| `test` | **Loss** | `1.1791` | `1.1193` | `-0.0598` | `-5.07%` |")
    lines.append("")

    # Table 5: MoE Standard vs MoE Frequency Comparison
    lines.append("## 5. Comparativo Direto: MoE Standard (Espacial) vs MoE Frequency (Fourier)")
    lines.append("")
    lines.append("| Split de Avaliação | Métrica | MoE Standard (RGB) | MoE Frequency (Fourier) | Vencedor | Vantagem Frequencial |")
    lines.append("| :--- | :--- | :---: | :---: | :---: | :---: |")
    lines.append(f"| `val` | **AUC** | `0.9316` | `0.9705` | **MoE Frequency** | `+0.0389 (+4.18%)` |")
    lines.append(f"| `val` | **Acurácia** | `0.8478` | `0.9144` | **MoE Frequency** | `+0.0666 (+7.86%)` |")
    lines.append(f"| `val` | **Loss** | `0.6767` | `0.5639` | **MoE Frequency** | `-0.1128 (-16.67%)` |")
    lines.append(f"| `test` | **AUC** | `0.8773` | `0.9068` | **MoE Frequency** | `+0.0295 (+3.36%)` |")
    lines.append(f"| `test` | **Acurácia** | `0.7773` | `0.8253` | **MoE Frequency** | `+0.0480 (+6.17%)` |")
    lines.append(f"| `test_d` | **AUC** | `0.6796` | `0.6512` | **MoE Standard** | `-0.0284 (-4.18%)` |")
    lines.append(f"| `test_d` | **Loss** | `2.3879` | `5.6885` | **MoE Standard** | `+3.3006 (+138.2%)` |")
    lines.append("")

    out_file = RESULTS_DIR / "moe_standard.md"
    out_file.write_text("\n".join(lines), encoding="utf-8")
    print(f"Generated: {out_file}")


def generate_moe_4experts_md():
    """Generate Markdown report for archive 4-experts baseline."""
    lines = []
    lines.append("# Resultados dos Modelos MoE Históricos: Linha de Base com 4 Especialistas")
    lines.append("")
    lines.append("> **Histórico Arquitetural**: Primeira geração dos modelos Mixture of Experts avaliados no projeto, configurados com **4 especialistas** (MobileNet-Small) e roteador Top-2 denso.")
    lines.append("> Armazenados em `models/archive_4experts/` como ponto de referência para a expansão para 7 especialistas.")
    lines.append("")

    # Table 1: Baseline Summary
    lines.append("## 1. Tabela Comparativa de Desempenho dos Modelos com 4 Especialistas")
    lines.append("")
    lines.append("| Modelo MoE | Modo | Split | AUC | Acurácia (ACC) | F1-Score | Precisão | Revocação | Especificidade | Loss (BCE) |")
    lines.append("| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |")
    lines.append("| `moe_frequency` (4 exp) | `concat_frequency` | `val` | **0.9496** | 0.8800 | 0.8964 | 0.9282 | 0.8668 | 0.8998 | 0.8365 |")
    lines.append("| `moe_frequency` (4 exp) | `concat_frequency` | `test` | **0.8737** | 0.7868 | 0.7969 | 0.8784 | 0.7292 | 0.8642 | 1.7828 |")
    lines.append("| `moe_standard` (4 exp) | `none` | `val` | **0.9332** | 0.8563 | 0.8725 | 0.9313 | 0.8208 | 0.9095 | 0.7293 |")
    lines.append("| `moe_standard` (4 exp) | `none` | `test` | **0.8831** | 0.7788 | 0.7775 | 0.9187 | 0.6739 | 0.9198 | 1.1791 |")
    lines.append("")

    # Table 2: Direct Comparison
    lines.append("## 2. Comparação Direta entre MoE Frequencial e Espacial (4 Especialistas)")
    lines.append("")
    lines.append("| Split | Métrica | MoE Frequency (4 exp) | MoE Standard (4 exp) | Vantagem Frequencial (Δ) |")
    lines.append("| :--- | :--- | :---: | :---: | :---: |")
    lines.append("| `val` | **AUC** | `0.9496` | `0.9332` | `+0.0164 (+1.76%)` |")
    lines.append("| `val` | **Acurácia** | `0.8800` | `0.8563` | `+0.0237 (+2.77%)` |")
    lines.append("| `val` | **F1-Score** | `0.8964` | `0.8725` | `+0.0239 (+2.74%)` |")
    lines.append("| `test` | **AUC** | `0.8737` | `0.8831` | `-0.0094 (-1.06%)` |")
    lines.append("| `test` | **Acurácia** | `0.7868` | `0.7788` | `+0.0080 (+1.03%)` |")
    lines.append("| `test` | **F1-Score** | `0.7969` | `0.7775` | `+0.0194 (+2.50%)` |")
    lines.append("")

    # Table 3: Hyperparameters
    lines.append("## 3. Configurações de Hiperparâmetros (MoE 4 Especialistas)")
    lines.append("")
    lines.append("| Parâmetro | `moe_frequency` (4 exp) | `moe_standard` (4 exp) |")
    lines.append("| :--- | :---: | :---: |")
    lines.append("| **Número de Especialistas** | 4 | 4 |")
    lines.append("| **Top-K Roteamento** | 2 | 2 |")
    lines.append("| **Canais de Entrada** | 6 (RGB + Magnitude + Fase + HPF) | 3 (RGB) |")
    lines.append("| **Backbone** | MobileNetV3-Small | MobileNetV3-Small |")
    lines.append("| **Épocas** | 20 | 20 |")
    lines.append("| **Batch Size** | 16 | 16 |")
    lines.append("| **Limiar Ótimo de Decisão** | `0.9885` | `0.9774` |")
    lines.append("")

    out_file = RESULTS_DIR / "moe_4experts.md"
    out_file.write_text("\n".join(lines), encoding="utf-8")
    print(f"Generated: {out_file}")


def generate_ensembles_md():
    """Generate Markdown report for Ensembles benchmark."""
    lines = []
    lines.append("# Resultados dos Modelos em Ensemble (Comitês)")
    lines.append("")
    lines.append("> **Visão Geral**: Avaliação sistemática de arquiteturas combinadas por fusão probabilística, votação e empilhamento (stacking), visando maximizar a robustez e mitigar o overfitting aos artefatos de treino.")
    lines.append("")

    # Table 1: Master Ensemble Benchmark
    lines.append("## 1. Tabela Principal de Desempenho dos Ensembles no Teste Padrão e Teste Difícil")
    lines.append("")
    lines.append("| Ranking | ID do Ensemble | Composição | Estratégia de Fusão | Val AUC | Test AUC | Test ACC | Test F1 | Test_d AUC | Test_d ACC | Test_d F1 | Δ AUC | Score | Conceito |")
    lines.append("| :---: | :--- | :--- | :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |")

    if not df_ens.empty:
        sorted_ens = df_ens.sort_values(by="final_score", ascending=False)
        for idx, (_, r) in enumerate(sorted_ens.iterrows(), start=1):
            ens_id = r["ensemble_id"]
            name = r["composition_name"]
            strat = r["strategy"]
            v_auc = fmt(r["val_auc"])
            t_auc = fmt(r["test_auc"])
            t_acc = fmt(r["test_acc"])
            t_f1 = fmt(r["test_f1"])
            td_auc = fmt(r["test_d_auc"])
            td_acc = fmt(r["test_d_acc"])
            td_f1 = fmt(r["test_d_f1"])
            d_auc = fmt(r["delta_auc"])
            score = fmt(r["final_score"], 2)
            grade = r["grade_concept"]
            lines.append(f"| {idx} | `{ens_id}` | {name} | `{strat}` | {v_auc} | {t_auc} | {t_acc} | {t_f1} | **{td_auc}** | {td_acc} | {td_f1} | `{d_auc}` | {score} | **{grade}** |")
    lines.append("")

    # Table 2: Generalization on Celeb-DF
    lines.append("## 2. Generalização Cross-Dataset no Celeb-DF (In-the-Wild)")
    lines.append("")
    lines.append("| Ensemble | Estratégia de Fusão | Frame AUC (%) | Frame ACC (%) | Video AUC (%) | Video ACC (%) |")
    lines.append("| :--- | :---: | :---: | :---: | :---: | :---: |")
    if not df_celeb_ens.empty:
        for _, r in df_celeb_ens.iterrows():
            ens_name = r.get("Ensemble", "-")
            fus = r.get("Fusão", "-")
            f_auc = fmt(r.get("Frame AUC (%)"), 2)
            f_acc = fmt(r.get("Frame ACC (%)"), 2)
            v_auc = fmt(r.get("Video AUC (%)"), 2)
            v_acc = fmt(r.get("Video ACC (%)"), 2)
            lines.append(f"| **{ens_name}** | `{fus}` | {f_auc}% | {f_acc}% | **{v_auc}%** | {v_acc}% |")
    lines.append("")

    # Table 3: Generalization on DF40
    lines.append("## 3. Generalização no Benchmark DF40 por Paradigma Gerativo")
    lines.append("")
    lines.append("| Categoria | Ensemble | Estratégia de Fusão | Modelos Integrados | AUC (%) | ACC (%) |")
    lines.append("| :--- | :--- | :---: | :--- | :---: | :---: |")
    if not df_df40_ens.empty:
        top_df40 = df_df40_ens.sort_values(by="AUC (%)", ascending=False).head(20)
        for _, r in top_df40.iterrows():
            cat = r.get("Categoria", "-")
            ens_name = r.get("Ensemble", "-")
            fus = r.get("Fusão", "-")
            models = r.get("Modelos", "-")
            auc = fmt(r.get("AUC (%)"), 2)
            acc = fmt(r.get("ACC (%)"), 2)
            lines.append(f"| {cat} | **{ens_name}** | `{fus}` | {models} | **{auc}%** | {acc}% |")
    lines.append("")

    out_file = RESULTS_DIR / "ensembles.md"
    out_file.write_text("\n".join(lines), encoding="utf-8")
    print(f"Generated: {out_file}")


def generate_readme_md():
    """Generate master README.md in results/ with comprehensive leaderboard and comparative tables."""
    lines = []
    lines.append("# Sumário Geral e Tabela Consolidada de Resultados")
    lines.append("")
    lines.append("> Este repositório documenta a avaliação empírica de **6 famílias arquiteturais de redes neurais**, **modelos Mixture of Experts (MoE)** e **comitês (Ensembles)** sob **7 regimes de representação no domínio de frequência (2D-FFT)** contra imagens autênticas e manipuladas por Deepfakes.")
    lines.append("")

    # Table 1: Global Benchmark Leaderboard
    lines.append("## 1. Tabela de Classificação Global de Robustez (Test vs Test_d)")
    lines.append("")
    lines.append("Classificação de modelos individuais e ensembles ordenados pelo desempenho em dados com perturbações adversas e compressão (`test_d`).")
    lines.append("")
    lines.append("| Posição | Família do Modelo | Modo Fourier / Fusão | Regime | Sementes | Test AUC | Test ACC | Test_d AUC | Test_d ACC | Queda Δ AUC | Score Geral | Conceito |")
    lines.append("| :---: | :--- | :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |")

    leaderboard_rows = []

    if not df_ens.empty:
        for _, r in df_ens.iterrows():
            leaderboard_rows.append({
                "family": f"ENSEMBLE ({r['num_models']}M)",
                "mode": f"{r['ensemble_id']} [{r['strategy']}]",
                "regime": "ensemble",
                "seeds": 5,
                "test_auc": r["test_auc"],
                "test_acc": r["test_acc"],
                "test_d_auc": r["test_d_auc"],
                "test_d_acc": r["test_d_acc"],
                "delta_auc": r["delta_auc"],
                "score": r["final_score"],
                "grade": r["grade_concept"],
            })

    if not df_bench.empty:
        for _, r in df_bench.iterrows():
            if str(r["model_family"]).startswith("ENSEMBLE"):
                continue
            leaderboard_rows.append({
                "family": r["model_family"],
                "mode": r["fourier_mode"],
                "regime": r["regime"],
                "seeds": int(r["num_seeds"]),
                "test_auc": r["test_auc_mean"],
                "test_acc": r["test_acc_mean"],
                "test_d_auc": r["test_d_auc_mean"],
                "test_d_acc": r["test_d_acc_mean"],
                "delta_auc": r["delta_auc_mean"],
                "score": r.get("final_score", 0.0),
                "grade": r.get("grade_concept", "-"),
            })

    if not df_moe_full.empty:
        for fam in ["moe_frequency", "moe_standard"]:
            sub = df_moe_full[df_moe_full["model_family"] == fam]
            r_t = sub[sub["split"] == "test"]
            r_td = sub[sub["split"] == "test_d"]
            if not r_t.empty and not r_td.empty:
                t_auc = r_t.iloc[0]["auc_mean"]
                t_acc = r_t.iloc[0]["acc_mean"]
                td_auc = r_td.iloc[0]["auc_mean"]
                td_acc = r_td.iloc[0]["acc_mean"]
                d_auc = td_auc - t_auc
                score = (t_auc * 4.0 + td_auc * 6.0)
                grade = "A" if score >= 7.0 else ("B+" if score >= 6.5 else ("B" if score >= 6.0 else ("C" if score >= 5.5 else "D")))
                mode = "concat_frequency" if fam == "moe_frequency" else "none"
                leaderboard_rows.append({
                    "family": f"MoE (7 experts) - {fam}",
                    "mode": mode,
                    "regime": "scratch",
                    "seeds": 1,
                    "test_auc": t_auc,
                    "test_acc": t_acc,
                    "test_d_auc": td_auc,
                    "test_d_acc": td_acc,
                    "delta_auc": d_auc,
                    "score": score,
                    "grade": grade,
                })

    df_lb = pd.DataFrame(leaderboard_rows)
    df_lb_sorted = df_lb.sort_values(by="test_d_auc", ascending=False)

    for idx, (_, r) in enumerate(df_lb_sorted.iterrows(), start=1):
        fam = r["family"]
        mode = r["mode"]
        reg = r["regime"]
        seeds = r["seeds"]
        t_auc = fmt(r["test_auc"])
        t_acc = fmt(r["test_acc"])
        td_auc = fmt(r["test_d_auc"])
        td_acc = fmt(r["test_d_acc"])
        d_auc = fmt(r["delta_auc"])
        score = fmt(r["score"], 2)
        grade = r["grade"]
        lines.append(f"| {idx} | **{fam}** | `{mode}` | `{reg}` | {seeds} | {t_auc} | {t_acc} | **{td_auc}** | {td_acc} | `{d_auc}` | {score} | **{grade}** |")
    lines.append("")

    # Table 2: Champion of Each Model Family
    lines.append("## 2. Tabela de Campeões por Família Arquitetural")
    lines.append("")
    lines.append("| Família Arquitetural | Melhor Modo Fourier | Test AUC | Test ACC | Test_d AUC | Test_d ACC | Δ AUC (Robustez) | DF40 AUC | Celeb-DF Frame AUC |")
    lines.append("| :--- | :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: |")

    single_fams = ["clip", "dino", "vit", "resnet", "mobilenet", "xception"]
    for fam in single_fams:
        sub = df_bench[df_bench["model_family"] == fam]
        if sub.empty:
            continue
        best_row = sub.sort_values(by="test_d_auc_mean", ascending=False).iloc[0]
        mode = best_row["fourier_mode"]
        t_auc = fmt_pm(best_row["test_auc_mean"], best_row["test_auc_std"])
        t_acc = fmt_pm(best_row["test_acc_mean"], best_row["test_acc_std"])
        td_auc = fmt_pm(best_row["test_d_auc_mean"], best_row["test_d_auc_std"])
        td_acc = fmt_pm(best_row["test_d_acc_mean"], best_row["test_d_acc_std"])
        d_auc = fmt_pm(best_row["delta_auc_mean"], best_row["delta_auc_std"])

        sub_df40 = df_df40[(df_df40["model_family"] == fam) & (df_df40["fourier_mode"] == mode)]
        sub_cf = df_celeb_f[(df_celeb_f["model_family"] == fam) & (df_celeb_f["fourier_mode"] == mode)]

        df40_auc = fmt_pm(sub_df40["auc_mean"].iloc[0], sub_df40["auc_std"].iloc[0]) if not sub_df40.empty else "-"
        cf_auc = fmt_pm(sub_cf["auc_mean"].iloc[0], sub_cf["auc_std"].iloc[0]) if not sub_cf.empty else "-"

        lines.append(f"| **{fam.upper()}** | `{mode}` | {t_auc} | {t_acc} | **{td_auc}** | {td_acc} | `{d_auc}` | {df40_auc} | {cf_auc} |")

    lines.append(f"| **MoE (Frequencial, 7 exp)** | `concat_frequency` | `0.9068` | `0.8253` | **`0.6512`** | `0.6240` | `-0.2556` | Em andamento | Em andamento |")
    lines.append(f"| **MoE (Espacial, 7 exp)** | `none` | `0.8773` | `0.7773` | **`0.6796`** | `0.6199` | `-0.1976` | Em andamento | Em andamento |")
    lines.append("")

    # Table 3: Impact of Fourier Domain Representation
    lines.append("## 3. Tabela de Impacto da Representação Frequencial (Espacial vs Híbrido Fourier)")
    lines.append("")
    lines.append("Comparação direta entre o modelo de linha de base puramente espacial (`none`), o modelo híbrido concatenado (`concat`) e a fusão frequencial pura (`concat_frequency`).")
    lines.append("")
    lines.append("| Família | Test AUC (`none`) | Test AUC (`concat`) | Test AUC (`concat_freq`) | Test_d AUC (`none`) | Test_d AUC (`concat`) | Test_d AUC (`concat_freq`) | Domínio Mais Robusto |")
    lines.append("| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: |")

    for fam in single_fams:
        r_none = df_bench[(df_bench["model_family"] == fam) & (df_bench["fourier_mode"] == "none")]
        r_concat = df_bench[(df_bench["model_family"] == fam) & (df_bench["fourier_mode"] == "concat")]
        r_cf = df_bench[(df_bench["model_family"] == fam) & (df_bench["fourier_mode"] == "concat_frequency")]

        t_none = fmt(r_none["test_auc_mean"].iloc[0]) if not r_none.empty else "-"
        t_concat = fmt(r_concat["test_auc_mean"].iloc[0]) if not r_concat.empty else "-"
        t_cf = fmt(r_cf["test_auc_mean"].iloc[0]) if not r_cf.empty else "-"

        td_none = fmt(r_none["test_d_auc_mean"].iloc[0]) if not r_none.empty else "-"
        td_concat = fmt(r_concat["test_d_auc_mean"].iloc[0]) if not r_concat.empty else "-"
        td_cf = fmt(r_cf["test_d_auc_mean"].iloc[0]) if not r_cf.empty else "-"

        best_td = "-"
        if not r_none.empty and not r_concat.empty:
            v_none = r_none["test_d_auc_mean"].iloc[0]
            v_concat = r_concat["test_d_auc_mean"].iloc[0]
            v_cf = r_cf["test_d_auc_mean"].iloc[0] if not r_cf.empty else -1
            best_val = max(v_none, v_concat, v_cf)
            if best_val == v_concat:
                best_td = "**concat (Híbrido)**"
            elif best_val == v_cf:
                best_td = "**concat_freq (Fourier)**"
            else:
                best_td = "none (Espacial)"

        lines.append(f"| **{fam.upper()}** | {t_none} | {t_concat} | {t_cf} | {td_none} | {td_concat} | {td_cf} | {best_td} |")
    lines.append("")

    # Table 4: Repository Navigation Index
    lines.append("## 4. Índice e Navegação dos Relatórios de Modelos")
    lines.append("")
    lines.append("| Relatório (.md) | Família / Categoria | Número de Especialistas / Variantes | Métricas Contidas |")
    lines.append("| :--- | :--- | :---: | :--- |")
    lines.append("| [`moe_frequency.md`](./moe_frequency.md) | Mixture of Experts Frequencial | 7 Especialistas | Val, Test, Test_d, Evolução 4 exp vs 7 exp, Alocação de Especialistas |")
    lines.append("| [`moe_standard.md`](./moe_standard.md) | Mixture of Experts Padrão | 7 Especialistas | Val, Test, Test_d, Comparativo com MoE Frequencial, Evolução 4 exp vs 7 exp |")
    lines.append("| [`moe_4experts.md`](./moe_4experts.md) | MoE Histórico (Linha de Base) | 4 Especialistas | Val, Test, Comparativo Frequency vs Standard |")
    lines.append("| [`clip.md`](./clip.md) | CLIP ViT-B/16 | 7 Modos Fourier x 5 Sementes | Test, Test_d, Val, DF40, Celeb-DF, Desagregação por Semente |")
    lines.append("| [`dino.md`](./dino.md) | DINO ViT-S/16 | 7 Modos Fourier x 5 Sementes | Test, Test_d, Val, DF40, Celeb-DF, Desagregação por Semente |")
    lines.append("| [`vit.md`](./vit.md) | Vision Transformer (ViT-B/16) | 7 Modos Fourier x 5 Sementes | Test, Test_d, Val, DF40, Celeb-DF, Desagregação por Semente |")
    lines.append("| [`resnet.md`](./resnet.md) | ResNet-50 | 7 Modos Fourier x 5 Sementes | Test, Test_d, Val, DF40, Celeb-DF, Desagregação por Semente |")
    lines.append("| [`mobilenet.md`](./mobilenet.md) | MobileNetV3-Small | 7 Modos Fourier x 5 Sementes | Test, Test_d, Val, DF40, Celeb-DF, Desagregação por Semente |")
    lines.append("| [`xception.md`](./xception.md) | XceptionNet | 7 Modos Fourier x 5 Sementes | Test, Test_d, Val, DF40, Celeb-DF, Desagregação por Semente |")
    lines.append("| [`ensembles.md`](./ensembles.md) | Comitês de Modelos (Ensembles) | Top-4, Top-5, Top-3 e Val Optimal | Fusões Ponderadas, Médias, Stacking, Celeb-DF e DF40 |")
    lines.append("")

    out_file = RESULTS_DIR / "README.md"
    out_file.write_text("\n".join(lines), encoding="utf-8")
    print(f"Generated: {out_file}")


def main():
    print("Iniciando geração dos relatórios markdown com tabelas em results/...")

    models_meta = [
        ("clip", "CLIP (ViT-B/16 - Contrastive Language-Image Pretraining)", "Modelo multimodal de visão-linguagem pré-treinado no OpenAI WebImageText (400M pares)."),
        ("dino", "DINO (ViT-S/16 - Self-distillation with no labels)", "Visão auto-supervisionada pré-treinada por auto-destilação, capturando representações ricas de fronteira semântica."),
        ("vit", "Vision Transformer (ViT-B/16)", "Arquitetura baseada puramente em atenção multi-cabeça aplicada a patches espaciais e espectrais de imagem."),
        ("resnet", "ResNet-50 (Deep Residual Learning)", "Arquitetura convolucional profunda com conexões residuais (skip connections), amplamente utilizada em detecção forense."),
        ("mobilenet", "MobileNetV3-Small", "Rede convolucional ultraleve otimizada com blocos invertidos e atenção Squeeze-and-Excitation, servindo como base dos especialistas MoE."),
        ("xception", "Xception (Extreme Inception)", "Rede convolucional baseada em convoluções separáveis em profundidade (Depthwise Separable), padrão clássico da literatura de Deepfake."),
    ]

    for model_family, title, description in models_meta:
        generate_single_model_md(model_family, title, description)

    generate_moe_frequency_md()
    generate_moe_standard_md()
    generate_moe_4experts_md()
    generate_ensembles_md()
    generate_readme_md()

    print("Todos os relatórios em tabela foram gerados com sucesso na pasta results/!")


if __name__ == "__main__":
    main()
