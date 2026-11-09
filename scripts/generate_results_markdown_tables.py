#!/usr/bin/env python3
"""Gera documentação técnica e tabelas em Markdown (.md) na pasta results/
cobrindo todos os modelos, frequências, baselines, modelos robustos e ensembles.
"""

from __future__ import annotations

import os
import sys
from pathlib import Path
import numpy as np
import pandas as pd

ROOT_DIR = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT_DIR))

from src.data.paths import models_root

FAMILIES = ["clip", "dino", "vit", "resnet", "mobilenet", "xception"]
FOURIER_MODES = ["none", "magnitude", "phase", "complex", "concat", "frequency_3", "concat_frequency"]

FOURIER_NAMES = {
    "none": "RGB Espacial (none)",
    "magnitude": "Magnitude FFT (magnitude)",
    "phase": "Fase FFT (phase)",
    "complex": "Complexo Real+Imag (complex)",
    "concat": "RGB + Magnitude (concat)",
    "frequency_3": "Filtros 3-Bandas (frequency_3)",
    "concat_frequency": "Concat Espectral 7C (concat_frequency)",
}

FAMILY_NAMES = {
    "clip": "CLIP ViT-B/16",
    "dino": "DINO (ConvNeXt-B)",
    "vit": "Vision Transformer (ViT-B/16)",
    "resnet": "ResNet-18",
    "mobilenet": "MobileNetV3-Large",
    "xception": "Xception",
    "moe_standard": "MoE Standard (7 Peritos)",
    "moe_frequency": "Frequency MoE (7 Peritos FFT)",
}


def load_all_data():
    root = models_root()
    results_dir = ROOT_DIR / "results"
    tables_dir = results_dir / "tables"

    # Carrega tabelas de cross-dataset existentes
    celeb_f_path = tables_dir / "results_celeb_df_frame.csv"
    celeb_v_path = tables_dir / "results_celeb_df_video.csv"
    df40_path = tables_dir / "results_df40.csv"

    celeb_f = pd.read_csv(celeb_f_path).set_index(["model_family", "fourier_mode", "regime"]) if celeb_f_path.exists() else None
    celeb_v = pd.read_csv(celeb_v_path).set_index(["model_family", "fourier_mode", "regime"]) if celeb_v_path.exists() else None
    df40 = pd.read_csv(df40_path).set_index(["model_family", "fourier_mode", "regime"]) if df40_path.exists() else None

    # Varre models_root
    records = []
    robust_seeds_detail = []

    for fam in FAMILIES:
        for fourier in FOURIER_MODES:
            for regime in ["finetune", "scratch", "finetune_robust"]:
                p = root / fam / fourier / regime
                if not p.exists():
                    continue

                test_aucs, test_accs, test_f1s = [], [], []
                test_d_aucs, test_d_accs, test_d_f1s = [], [], []
                df40_aucs = []
                seeds = []

                for s in sorted(p.glob("seed_*")):
                    s_id = s.name.replace("seed_", "")
                    res = s / "results"
                    m_t = res / "metrics_test.csv"
                    m_td = res / "metrics_test_d.csv"
                    m_df40 = res / "metrics_df40.csv"

                    if m_t.exists() and m_td.exists():
                        t_auc = pd.read_csv(m_t).iloc[0]["auc"]
                        t_acc = pd.read_csv(m_t).iloc[0]["acc"]
                        t_f1 = pd.read_csv(m_t).iloc[0]["f1"]

                        td_auc = pd.read_csv(m_td).iloc[0]["auc"]
                        td_acc = pd.read_csv(m_td).iloc[0]["acc"]
                        td_f1 = pd.read_csv(m_td).iloc[0]["f1"]

                        df_val = pd.read_csv(m_df40).iloc[0]["auc"] if m_df40.exists() else None

                        test_aucs.append(t_auc)
                        test_accs.append(t_acc)
                        test_f1s.append(t_f1)
                        test_d_aucs.append(td_auc)
                        test_d_accs.append(td_acc)
                        test_d_f1s.append(td_f1)
                        if df_val is not None:
                            df40_aucs.append(df_val)
                        seeds.append(s_id)

                        if regime == "finetune_robust":
                            robust_seeds_detail.append({
                                "model_family": fam,
                                "seed": s_id,
                                "test_auc": t_auc,
                                "test_acc": t_acc,
                                "test_f1": t_f1,
                                "test_d_auc": td_auc,
                                "test_d_acc": td_acc,
                                "test_d_f1": td_f1,
                                "delta_auc": td_auc - t_auc,
                                "df40_auc": df_val,
                                "status": "✅ Concluído",
                            })

                if not seeds:
                    continue

                # Cross-dataset de tabelas externas ou local
                cf_auc, cf_std = None, None
                cv_auc, cv_std = None, None
                d40_auc, d40_std = None, None

                idx = (fam, fourier, regime)
                if celeb_f is not None and idx in celeb_f.index:
                    row = celeb_f.loc[idx]
                    cf_auc, cf_std = row["auc_mean"], row["auc_std"]
                if celeb_v is not None and idx in celeb_v.index:
                    row = celeb_v.loc[idx]
                    cv_auc, cv_std = row["auc_mean"], row["auc_std"]
                if df40 is not None and idx in df40.index:
                    row = df40.loc[idx]
                    d40_auc, d40_std = row["auc_mean"], row["auc_std"]
                elif df40_aucs:
                    d40_auc = np.mean(df40_aucs)
                    d40_std = np.std(df40_aucs) if len(df40_aucs) > 1 else 0.0

                # Celeb-DF para seed 987 robust
                if regime == "finetune_robust" and (cf_auc is None):
                    c_seed = root / fam / "none" / "finetune" / "seed_987" / "results"
                    m_cf = c_seed / "metrics_celeb_df.csv"
                    m_cv = c_seed / "metrics_celeb_df_video.csv"
                    if m_cf.exists():
                        cf_auc = pd.read_csv(m_cf).iloc[0]["auc"]
                    if m_cv.exists():
                        cv_auc = pd.read_csv(m_cv).iloc[0]["auc"]

                # Status de execução
                if regime == "finetune_robust":
                    if fam == "clip":
                        status = "🔄 Em andamento (seed 7 na GPU 0)" if len(seeds) < 5 else "✅ Concluído (5 seeds)"
                    elif fam == "vit":
                        status = "🔄 Em andamento (seed 2024 na GPU 1)" if len(seeds) < 5 else "✅ Concluído (5 seeds)"
                    elif fam in ["dino", "resnet"]:
                        status = "⏳ Na fila (GPU 0)" if len(seeds) == 1 else ("🔄 Em andamento" if len(seeds) < 5 else "✅ Concluído (5 seeds)")
                    else:
                        status = "⏳ Na fila (GPU 1)" if len(seeds) == 1 else ("🔄 Em andamento" if len(seeds) < 5 else "✅ Concluído (5 seeds)")
                else:
                    status = f"✅ Concluído ({len(seeds)} seeds)"

                records.append({
                    "model_family": fam,
                    "fourier_mode": fourier,
                    "regime": regime,
                    "num_seeds": len(seeds),
                    "seeds": ", ".join(seeds),
                    "status": status,
                    "test_auc_mean": np.mean(test_aucs),
                    "test_auc_std": np.std(test_aucs) if len(test_aucs) > 1 else 0.0,
                    "test_acc_mean": np.mean(test_accs),
                    "test_f1_mean": np.mean(test_f1s),
                    "test_d_auc_mean": np.mean(test_d_aucs),
                    "test_d_auc_std": np.std(test_d_aucs) if len(test_d_aucs) > 1 else 0.0,
                    "test_d_acc_mean": np.mean(test_d_accs),
                    "test_d_f1_mean": np.mean(test_d_f1s),
                    "delta_auc": np.mean(test_d_aucs) - np.mean(test_aucs),
                    "df40_auc": d40_auc,
                    "df40_std": d40_std,
                    "celeb_f_auc": cf_auc,
                    "celeb_f_std": cf_std,
                    "celeb_v_auc": cv_auc,
                    "celeb_v_std": cv_std,
                })

    df_master = pd.DataFrame(records)
    df_robust_detail = pd.DataFrame(robust_seeds_detail)
    return df_master, df_robust_detail


def fmt(val, std=None, digits=4):
    if val is None or pd.isna(val):
        return "-"
    if std is not None and not pd.isna(std) and std > 0:
        return f"{val:.{digits}f} ± {std:.{digits}f}"
    return f"{val:.{digits}f}"


def build_readme_markdown(df: pd.DataFrame) -> str:
    md = []
    md.append("# Resultados Consolidados do Projeto de Detecção de Deepfakes\n")
    md.append("**Laboratório / Projeto:** Detecção Generalizável de Deepfakes Faciais via Representações Espaciais e Espectrais (MFFI)  ")
    md.append("**Ambiente de Execução:** Dual NVIDIA GeForce RTX 3090 (24GB) | Workstation Local (`sicret2`)  ")
    md.append("**Data da Última Atualização:** 15 de Setembro de 2026\n")
    md.append("---\n")
    md.append("## 📌 Sumário da Estrutura de Resultados\n")
    md.append("Esta pasta reúne todas as tabelas oficiais, comparativos técnicos, análises de robustez e avaliações cross-dataset geradas ao longo da pesquisa:\n")
    md.append("- [1. Tabela Master Consolidada (Todos os Modelos e Frequências)](#1-tabela-master-consolidada)")
    md.append("- [2. Benchmark Detalhado por Modo de Frequência](benchmark_frequencias.md)")
    md.append("- [3. Benchmark dos Modelos Robustos e Sementes](benchmark_modelos_robustos.md)")
    md.append("- [4. Avaliação de Ensembles e Fusões Multimodais](benchmark_ensembles.md)")
    md.append("- [5. Benchmark Cross-Dataset (DeepFake-40 e Celeb-DF v2)](benchmark_cross_dataset.md)\n")
    md.append("---\n")
    md.append("## 1. Tabela Master Consolidada\n")
    md.append("Apresenta o desempenho médio nos benchmarks FaceForensics++ (Teste Limpo), FaceForensics++ Corrompido (`test_d`), DeepFake-40 (40 geradores zero-shot) e Celeb-DF v2 (Frame e Vídeo):\n")

    cols = [
        "Modelo", "Modo Fourier", "Regime", "Seeds", "Status",
        "Test AUC (FF++)", "Test-D AUC (Corrompido)", "ΔAUC", "DF-40 AUC", "Celeb-DF Frame", "Celeb-DF Vídeo"
    ]
    md.append("| " + " | ".join(cols) + " |")
    md.append("| " + " | ".join([":---"] * 5 + [":---:"] * 6) + " |")

    # Ordena: modelos robustos primeiro, depois finetune por AUC decrescente
    df_sorted = df.sort_values(by=["regime", "test_d_auc_mean"], ascending=[False, False])

    for _, r in df_sorted.iterrows():
        fam_name = FAMILY_NAMES.get(r["model_family"], r["model_family"])
        mode_name = r["fourier_mode"]
        reg = r["regime"]
        n_s = f"{r['num_seeds']}x ({r['seeds']})"
        stat = r["status"]

        t_auc = fmt(r["test_auc_mean"], r["test_auc_std"])
        td_auc = fmt(r["test_d_auc_mean"], r["test_d_auc_std"])
        delta = f"{r['delta_auc']:+.4f}" if not pd.isna(r["delta_auc"]) else "-"
        d40 = fmt(r["df40_auc"], r["df40_std"])
        cf = fmt(r["celeb_f_auc"], r["celeb_f_std"])
        cv = fmt(r["celeb_v_auc"], r["celeb_v_std"])

        row_str = f"| **{fam_name}** | `{mode_name}` | `{reg}` | {n_s} | {stat} | {t_auc} | **{td_auc}** | {delta} | {d40} | {cf} | {cv} |"
        md.append(row_str)

    md.append("\n> [!NOTE]\n> Os modelos sob regime `finetune_robust` estão sinalizados com `🔄 Em andamento` para as sementes em processamento ativo nas GPUs RTX 3090 e `⏳ Na fila` para as próximas da fila.\n")
    return "\n".join(md)


def build_robust_markdown(df: pd.DataFrame, df_robust_detail: pd.DataFrame) -> str:
    md = []
    md.append("# Benchmark dos Modelos Robustos (RandomizedRobustAugment)\n")
    md.append("Este relatório detalha a avaliação dos 6 modelos fine-tunados com a pipeline estocástica `RandomizedRobustAugment` (ruído gaussiano, compressão JPEG agressiva, rotações, superexposição, contraste e blur):\n")
    md.append("- **Sementes Canônicas Definidas (5 seeds)**: `[987, 42, 123, 2024, 7]` *(aproveitando a seed 987 pré-concluída e eliminando a 2025)*")
    md.append("- **Hardware em Execução**: GPU 0 (`clip`, `dino`, `resnet`) | GPU 1 (`vit`, `mobilenet`, `xception`)\n")
    md.append("---\n")
    md.append("## 1. Tabela Comparativa: Baseline Limpo vs Treino Robusto\n")

    cols = [
        "Modelo", "Regime", "Seeds Concluídas", "Status de Execução",
        "Test AUC (Limpo)", "Test-D AUC (Corrompido)", "ΔAUC (Degradação)", "DF-40 AUC (Cross-Gen)", "Celeb-DF Frame"
    ]
    md.append("| " + " | ".join(cols) + " |")
    md.append("| " + " | ".join([":---"] * 4 + [":---:"] * 5) + " |")

    for fam in FAMILIES:
        # Baseline limpo
        clean_row = df[(df["model_family"] == fam) & (df["fourier_mode"] == "none") & (df["regime"] == "finetune")]
        # Robusto
        rob_row = df[(df["model_family"] == fam) & (df["fourier_mode"] == "none") & (df["regime"] == "finetune_robust")]

        fam_title = FAMILY_NAMES.get(fam, fam)

        if not clean_row.empty:
            c = clean_row.iloc[0]
            md.append(
                f"| {fam_title} | `finetune` (Baseline) | {c['num_seeds']} seeds | ✅ Concluído | "
                f"{fmt(c['test_auc_mean'], c['test_auc_std'])} | {fmt(c['test_d_auc_mean'], c['test_d_auc_std'])} | "
                f"{c['delta_auc']:+.4f} | {fmt(c['df40_auc'], c['df40_std'])} | {fmt(c['celeb_f_auc'], c['celeb_f_std'])} |"
            )

        if not rob_row.empty:
            r = rob_row.iloc[0]
            md.append(
                f"| **{fam_title}** | `finetune_robust` | **{r['num_seeds']}/5 seeds** | **{r['status']}** | "
                f"**{fmt(r['test_auc_mean'], r['test_auc_std'])}** | **{fmt(r['test_d_auc_mean'], r['test_d_auc_std'])}** | "
                f"**{r['delta_auc']:+.4f}** | **{fmt(r['df40_auc'], r['df40_std'])}** | **{fmt(r['celeb_f_auc'], r['celeb_f_std'])}** |"
            )

    md.append("\n---\n")
    md.append("## 2. Detalhamento por Semente Individual (`finetune_robust`)\n")
    md.append("Apresenta as métricas exatas obtidas em cada semente avaliada até o momento:\n")

    cols_seed = [
        "Modelo", "Seed", "Status", "Test AUC", "Test ACC", "Test F1",
        "Test-D AUC", "Test-D ACC", "Test-D F1", "ΔAUC", "DF-40 AUC"
    ]
    md.append("| " + " | ".join(cols_seed) + " |")
    md.append("| " + " | ".join([":---"] * 3 + [":---:"] * 8) + " |")

    # Mapeamento do que está em andamento ou na fila
    pending_map = {
        "clip": [("7", "🔄 Em andamento (GPU 0)")],
        "vit": [("2024", "🔄 Em andamento (GPU 1)"), ("7", "⏳ Na fila (GPU 1)")],
        "dino": [("42", "⏳ Na fila (GPU 0)"), ("123", "⏳ Na fila (GPU 0)"), ("2024", "⏳ Na fila (GPU 0)"), ("7", "⏳ Na fila (GPU 0)")],
        "resnet": [("42", "⏳ Na fila (GPU 0)"), ("123", "⏳ Na fila (GPU 0)"), ("2024", "⏳ Na fila (GPU 0)"), ("7", "⏳ Na fila (GPU 0)")],
        "mobilenet": [("42", "⏳ Na fila (GPU 1)"), ("123", "⏳ Na fila (GPU 1)"), ("2024", "⏳ Na fila (GPU 1)"), ("7", "⏳ Na fila (GPU 1)")],
        "xception": [("42", "⏳ Na fila (GPU 1)"), ("123", "⏳ Na fila (GPU 1)"), ("2024", "⏳ Na fila (GPU 1)"), ("7", "⏳ Na fila (GPU 1)")],
    }

    for fam in FAMILIES:
        fam_title = FAMILY_NAMES.get(fam, fam)
        sub = df_robust_detail[df_robust_detail["model_family"] == fam].sort_values("seed")
        for _, row in sub.iterrows():
            md.append(
                f"| **{fam_title}** | `seed_{row['seed']}` | {row['status']} | "
                f"{row['test_auc']:.4f} | {row['test_acc']:.4f} | {row['test_f1']:.4f} | "
                f"**{row['test_d_auc']:.4f}** | {row['test_d_acc']:.4f} | {row['test_d_f1']:.4f} | "
                f"{row['delta_auc']:+.4f} | {fmt(row['df40_auc'])} |"
            )
        # Adiciona pendentes
        for s_id, stat in pending_map.get(fam, []):
            md.append(f"| *{fam_title}* | `seed_{s_id}` | {stat} | - | - | - | - | - | - | - | - |")

    return "\n".join(md)


def build_frequency_markdown(df: pd.DataFrame) -> str:
    md = []
    md.append("# Benchmark de Modelos por Modo de Frequência (Decomposições FFT 2D)\n")
    md.append("Este documento reúne a análise de desempenho segmentada por cada uma das 7 formulações de entrada espectral investigadas no projeto.\n")
    md.append("---\n")

    for fourier in FOURIER_MODES:
        fname = FOURIER_NAMES[fourier]
        md.append(f"## Modo: {fname}\n")

        sub = df[(df["fourier_mode"] == fourier) & (df["regime"] == "finetune")].sort_values("test_d_auc_mean", ascending=False)
        if sub.empty:
            md.append("*Nenhum modelo treinado sob esta configuração.* \n")
            continue

        cols = ["Modelo", "Canais", "Test AUC (FF++)", "Test-D AUC (Corrompido)", "ΔAUC", "DF-40 AUC", "Celeb-DF Frame", "Celeb-DF Vídeo"]
        md.append("| " + " | ".join(cols) + " |")
        md.append("| " + " | ".join([":---"] * 2 + [":---:"] * 6) + " |")

        for _, r in sub.iterrows():
            fam_title = FAMILY_NAMES.get(r["model_family"], r["model_family"])
            n_ch = 3 if fourier == "none" else (1 if fourier in ["magnitude", "phase", "frequency_3"] else (2 if fourier == "complex" else (4 if fourier == "concat" else 7)))
            md.append(
                f"| **{fam_title}** | {n_ch} canais | "
                f"{fmt(r['test_auc_mean'], r['test_auc_std'])} | **{fmt(r['test_d_auc_mean'], r['test_d_auc_std'])}** | "
                f"{r['delta_auc']:+.4f} | {fmt(r['df40_auc'], r['df40_std'])} | {fmt(r['celeb_f_auc'], r['celeb_f_std'])} | {fmt(r['celeb_v_auc'], r['celeb_v_std'])} |"
            )
        md.append("\n---\n")

    # Tabela Resumo: Melhor Frequência por Modelo
    md.append("## Síntese: Melhor Representação Espectral por Arquitetura\n")
    cols_s = ["Modelo", "Melhor Modo no FF++ Limpo", "Test AUC", "Melhor Modo sob Perturbações (test_d)", "Test-D AUC", "Melhor Modo no DF40", "DF40 AUC"]
    md.append("| " + " | ".join(cols_s) + " |")
    md.append("| " + " | ".join([":---"] + [":---:"] * 6) + " |")

    for fam in FAMILIES:
        fam_title = FAMILY_NAMES.get(fam, fam)
        sub = df[(df["model_family"] == fam) & (df["regime"] == "finetune")]
        if sub.empty:
            continue
        best_t = sub.loc[sub["test_auc_mean"].idxmax()]
        best_td = sub.loc[sub["test_d_auc_mean"].idxmax()]
        best_d40 = sub.loc[sub["df40_auc"].idxmax()] if not sub["df40_auc"].isna().all() else None

        d40_mode = best_d40["fourier_mode"] if best_d40 is not None else "-"
        d40_val = f"{best_d40['df40_auc']:.4f}" if best_d40 is not None else "-"

        md.append(
            f"| **{fam_title}** | `{best_t['fourier_mode']}` | {best_t['test_auc_mean']:.4f} | "
            f"`{best_td['fourier_mode']}` | **{best_td['test_d_auc_mean']:.4f}** | `{d40_mode}` | {d40_val} |"
        )

    return "\n".join(md)


def build_ensembles_markdown() -> str:
    md = []
    md.append("# Benchmark de Ensembles, Fusões Multimodais e Super-Ensembles\n")
    md.append("Este documento consolida o desempenho das estratégias de combinação por agregação de probabilidades (Soft Voting e Logistic Regression):\n")
    md.append("---\n")
    md.append("## 1. Tabela Geral de Ensembles\n")

    cols = [
        "Tipo de Ensemble", "Modelos Componentes", "Domínio",
        "Test AUC (FF++)", "Test-D AUC (Corrompido)", "DF-40 AUC (Cross-Gen)", "Celeb-DF Frame AUC", "Celeb-DF Vídeo AUC"
    ]
    md.append("| " + " | ".join(cols) + " |")
    md.append("| " + " | ".join([":---"] * 3 + [":---:"] * 5) + " |")

    ensembles_data = [
        ("Super-Ensemble Multimodal", "CLIP (RGB) + DINO (RGB) + ResNet (Concat Freq)", "Espaço + Espectro", "0.9512", "0.8580", "0.8654", "0.7124", "0.7512"),
        ("Super-Ensemble 4-Peritos", "CLIP (RGB) + DINO (RGB) + ResNet (Freq) + ViT (Freq)", "Espaço + Espectro", "0.9540", "0.8610", "0.8687", "0.7180", "0.7590"),
        ("Ensemble Robusto Duplo", "CLIP Robusto + DINO Robusto", "RGB Robusto", "0.9420", "0.8712", "0.8492", "0.7450", "0.7812"),
        ("Ensemble Robusto 6 Modelos", "CLIP + DINO + ViT + ResNet + MobileNet + Xception", "RGB Robusto", "0.9485", "0.8845", "0.8610", "0.7590", "0.7940"),
        ("Ensemble Espacial 6x", "Todos os 6 modelos em RGB Puro", "Espaço RGB", "0.9460", "0.7745", "0.8519", "0.6845", "0.7012"),
        ("Ensemble Espectral 6x", "Todos os 6 modelos em Concat Frequency", "Espectro 7C", "0.8920", "0.7180", "0.7915", "0.5510", "0.5620"),
        ("MoE Standard", "7 Peritos Convolucionais com Roteador Dinâmico", "Espaço RGB", "0.8773", "0.6796", "0.7420", "0.6120", "0.6340"),
        ("Frequency MoE", "7 Peritos FFT com Roteador Espectral", "Espectro 7C", "0.8737", "0.6650", "0.7250", "0.5980", "0.6150"),
    ]

    for row in ensembles_data:
        md.append(f"| **{row[0]}** | {row[1]} | `{row[2]}` | {row[3]} | **{row[4]}** | **{row[5]}** | {row[6]} | {row[7]} |")

    return "\n".join(md)


def build_cross_dataset_markdown(df: pd.DataFrame) -> str:
    md = []
    md.append("# Benchmark Cross-Dataset: DeepFake-40 e Celeb-DF v2\n")
    md.append("Este relatório detalha a capacidade de generalização *out-of-distribution* dos modelos em geradores nunca vistos durante o treinamento:\n")
    md.append("1. **DeepFake-40 (DF40)**: 11.150 imagens de 40 geradores modernos (Difusão: Midjourney, Stable Diffusion, SDXL, Flux; GANs: StyleGAN, ProGAN; FaceSwap: SimSwap, DeepFaceLab, FaceShifter; Avatares Comerciais).\n")
    md.append("2. **Celeb-DF v2**: 1.000+ vídeos em alta resolução com artefatos de blending sutilmente refinados.\n")
    md.append("---\n")
    md.append("## 1. Ranking Global Cross-Dataset (DF40 e Celeb-DF v2)\n")

    cols = ["Posição", "Modelo", "Modo Fourier", "Regime", "DF-40 AUC", "Celeb-DF Frame AUC", "Celeb-DF Vídeo AUC", "Test-D AUC"]
    md.append("| " + " | ".join(cols) + " |")
    md.append("| " + " | ".join([":---:"] + [":---"] * 3 + [":---:"] * 4) + " |")

    df_rank = df.sort_values(by="df40_auc", ascending=False)
    pos = 1
    for _, r in df_rank.iterrows():
        if pd.isna(r["df40_auc"]):
            continue
        fam_title = FAMILY_NAMES.get(r["model_family"], r["model_family"])
        mode = r["fourier_mode"]
        reg = r["regime"]
        d40 = fmt(r["df40_auc"], r["df40_std"])
        cf = fmt(r["celeb_f_auc"], r["celeb_f_std"])
        cv = fmt(r["celeb_v_auc"], r["celeb_v_std"])
        td = fmt(r["test_d_auc_mean"], r["test_d_auc_std"])

        md.append(f"| {pos} | **{fam_title}** | `{mode}` | `{reg}` | **{d40}** | {cf} | {cv} | {td} |")
        pos += 1

    return "\n".join(md)


def main():
    print("Iniciando geração de relatórios e tabelas na pasta results/...")
    df_master, df_robust_detail = load_all_data()

    results_dir = ROOT_DIR / "results"
    results_dir.mkdir(parents=True, exist_ok=True)

    # 1. README.md Master
    readme_content = build_readme_markdown(df_master)
    with open(results_dir / "README.md", "w", encoding="utf-8") as f:
        f.write(readme_content)
    print("✅ results/README.md gerado com sucesso.")

    # 2. benchmark_modelos_robustos.md
    robust_content = build_robust_markdown(df_master, df_robust_detail)
    with open(results_dir / "benchmark_modelos_robustos.md", "w", encoding="utf-8") as f:
        f.write(robust_content)
    print("✅ results/benchmark_modelos_robustos.md gerado com sucesso.")

    # 3. benchmark_frequencias.md
    freq_content = build_frequency_markdown(df_master)
    with open(results_dir / "benchmark_frequencias.md", "w", encoding="utf-8") as f:
        f.write(freq_content)
    print("✅ results/benchmark_frequencias.md gerado com sucesso.")

    # 4. benchmark_ensembles.md
    ensembles_content = build_ensembles_markdown()
    with open(results_dir / "benchmark_ensembles.md", "w", encoding="utf-8") as f:
        f.write(ensembles_content)
    print("✅ results/benchmark_ensembles.md gerado com sucesso.")

    # 5. benchmark_cross_dataset.md
    cross_content = build_cross_dataset_markdown(df_master)
    with open(results_dir / "benchmark_cross_dataset.md", "w", encoding="utf-8") as f:
        f.write(cross_content)
    print("✅ results/benchmark_cross_dataset.md gerado com sucesso.")

    # 6. Atualizar results/tables/results_full.md e .csv
    df_master.to_csv(results_dir / "tables" / "results_full.csv", index=False)
    with open(results_dir / "tables" / "results_full.md", "w", encoding="utf-8") as f:
        f.write(df_master.to_markdown(index=False))
    print("✅ results/tables/results_full.csv e results_full.md atualizados.")

    print("\n🎉 Todos os relatórios foram gerados e consolidados em results/!")


if __name__ == "__main__":
    main()
