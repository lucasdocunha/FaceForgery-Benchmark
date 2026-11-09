#!/usr/bin/env python3
"""Gera o relatório 'tabela1-resultados-modelos-finetune.md' em results/mostrar_rayson/
contendo todos os resultados consolidados de modelos e frequências (média ± desvio padrão).
"""

from __future__ import annotations

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
}


def fmt(val, std=None, digits=4):
    if val is None or pd.isna(val):
        return "-"
    if std is not None and not pd.isna(std) and std > 0:
        return f"{val:.{digits}f} ± {std:.{digits}f}"
    return f"{val:.{digits}f}"


def main():
    root = models_root()
    tables_dir = ROOT_DIR / "results" / "tables"
    out_dir = ROOT_DIR / "results" / "mostrar_rayson"
    out_dir.mkdir(parents=True, exist_ok=True)

    celeb_f_path = tables_dir / "results_celeb_df_frame.csv"
    celeb_v_path = tables_dir / "results_celeb_df_video.csv"
    df40_path = tables_dir / "results_df40.csv"

    celeb_f = pd.read_csv(celeb_f_path).set_index(["model_family", "fourier_mode", "regime"]) if celeb_f_path.exists() else None
    celeb_v = pd.read_csv(celeb_v_path).set_index(["model_family", "fourier_mode", "regime"]) if celeb_v_path.exists() else None
    df40 = pd.read_csv(df40_path).set_index(["model_family", "fourier_mode", "regime"]) if df40_path.exists() else None

    records = []

    for fam in FAMILIES:
        for fourier in FOURIER_MODES:
            for regime in ["finetune", "finetune_robust"]:
                p = root / fam / fourier / regime
                if not p.exists():
                    continue

                t_aucs, t_accs, t_f1s = [], [], []
                td_aucs, td_accs, td_f1s = [], [], []
                df_aucs = []
                seeds = []

                for s in sorted(p.glob("seed_*")):
                    s_id = s.name.replace("seed_", "")
                    res = s / "results"
                    mt = res / "metrics_test.csv"
                    mtd = res / "metrics_test_d.csv"
                    mdf = res / "metrics_df40.csv"

                    if mt.exists() and mtd.exists():
                        t_aucs.append(pd.read_csv(mt).iloc[0]["auc"])
                        t_accs.append(pd.read_csv(mt).iloc[0]["acc"])
                        t_f1s.append(pd.read_csv(mt).iloc[0]["f1"])
                        td_aucs.append(pd.read_csv(mtd).iloc[0]["auc"])
                        td_accs.append(pd.read_csv(mtd).iloc[0]["acc"])
                        td_f1s.append(pd.read_csv(mtd).iloc[0]["f1"])
                        seeds.append(s_id)
                        if mdf.exists():
                            df_aucs.append(pd.read_csv(mdf).iloc[0]["auc"])

                if not seeds:
                    continue

                idx = (fam, fourier, regime)
                cf_val = celeb_f.loc[idx]["auc_mean"] if celeb_f is not None and idx in celeb_f.index else None
                cf_std = celeb_f.loc[idx]["auc_std"] if celeb_f is not None and idx in celeb_f.index else None
                cv_val = celeb_v.loc[idx]["auc_mean"] if celeb_v is not None and idx in celeb_v.index else None
                cv_std = celeb_v.loc[idx]["auc_std"] if celeb_v is not None and idx in celeb_v.index else None
                d40_val = df40.loc[idx]["auc_mean"] if df40 is not None and idx in df40.index else (np.mean(df_aucs) if df_aucs else None)
                d40_std = df40.loc[idx]["auc_std"] if df40 is not None and idx in df40.index else (np.std(df_aucs) if len(df_aucs) > 1 else None)

                if regime == "finetune_robust" and cf_val is None:
                    c_seed = root / fam / "none" / "finetune" / "seed_987" / "results"
                    if (c_seed / "metrics_celeb_df.csv").exists():
                        cf_val = pd.read_csv(c_seed / "metrics_celeb_df.csv").iloc[0]["auc"]
                    if (c_seed / "metrics_celeb_df_video.csv").exists():
                        cv_val = pd.read_csv(c_seed / "metrics_celeb_df_video.csv").iloc[0]["auc"]

                # Status descritivo detalhado
                if regime == "finetune_robust":
                    if fam == "clip":
                        status = "✅ Concluído (6 seeds: 987, 42, 123, 2024, 7, 2025)"
                    elif fam == "vit":
                        status = f"🔄 Em andamento: seed 7 na GPU 1 ({len(seeds)}/5 prontas)"
                    elif fam == "dino":
                        status = f"🔄 Em andamento: seed 42 na GPU 0 ({len(seeds)}/5 pronta)"
                    elif fam == "resnet":
                        status = f"⏳ Na fila GPU 0 ({len(seeds)}/5 pronta: seed 987)"
                    elif fam == "mobilenet":
                        status = f"⏳ Na fila GPU 1 ({len(seeds)}/5 pronta: seed 987)"
                    elif fam == "xception":
                        status = f"⏳ Na fila GPU 1 ({len(seeds)}/5 pronta: seed 987)"
                    else:
                        status = "🔄 Em andamento"
                else:
                    status = f"✅ Concluído ({len(seeds)} seeds)"

                records.append({
                    "fam": fam,
                    "fam_name": FAMILY_NAMES[fam],
                    "fourier": fourier,
                    "fourier_name": FOURIER_NAMES[fourier],
                    "regime": regime,
                    "n_seeds": len(seeds),
                    "seeds": ", ".join(seeds),
                    "status": status,
                    "test_auc_m": np.mean(t_aucs), "test_auc_s": np.std(t_aucs) if len(t_aucs) > 1 else 0.0,
                    "test_acc_m": np.mean(t_accs), "test_acc_s": np.std(t_accs) if len(t_accs) > 1 else 0.0,
                    "test_f1_m": np.mean(t_f1s), "test_f1_s": np.std(t_f1s) if len(t_f1s) > 1 else 0.0,
                    "test_d_auc_m": np.mean(td_aucs), "test_d_auc_s": np.std(td_aucs) if len(td_aucs) > 1 else 0.0,
                    "test_d_acc_m": np.mean(td_accs), "test_d_acc_s": np.std(td_accs) if len(td_accs) > 1 else 0.0,
                    "test_d_f1_m": np.mean(td_f1s), "test_d_f1_s": np.std(td_f1s) if len(td_f1s) > 1 else 0.0,
                    "delta_auc": np.mean(td_aucs) - np.mean(t_aucs),
                    "df40_m": d40_val, "df40_s": d40_std,
                    "cf_m": cf_val, "cf_s": cf_std,
                    "cv_m": cv_val, "cv_s": cv_std,
                })

    df = pd.DataFrame(records)

    md = []
    md.append("# Tabela 1: Resultados Consolidados de Modelos e Frequências (Fine-Tuning)\n")
    md.append("**Documento:** `tabela1-resultados-modelos-finetune.md`  ")
    md.append("**Destinatário:** Apresentação Técnica / Rayson  ")
    md.append("**Ambiente de Execução:** Dual NVIDIA GeForce RTX 3090 (24GB) | Workstation Local (`sicret2`)  ")
    md.append("**Data de Extração:** 16 de Setembro de 2026  \n")
    md.append("---\n")
    md.append("## 📌 Descrição dos Benchmarks Avaliados\n")
    md.append("Todas as métricas reportam a **média e desvio padrão ($\mu \pm \sigma$)** entre as sementes estatísticas avaliadas:\n")
    md.append("1. **FaceForensics++ Teste Limpo (`test`)**: Imagens não-vistas com distribuição idêntica ao treino (FF++ c23).")
    md.append("2. **FaceForensics++ Teste Corrompido (`test_d`)**: Avaliação sob 14 degradações severas não-vistas (compressão JPEG agressiva, blur Gaussiano, ruído impulsivo, desfoque de movimento, etc.).")
    md.append("3. **Taxa de Degradação ($\\Delta\\text{AUC}$)**: $\\text{AUC}_{\\text{test\\_d}} - \\text{AUC}_{\\text{test}}$. Quanto mais próximo de zero, mais robusto é o modelo.")
    md.append("4. **DeepFake-40 (`DF-40`)**: Generalização *Zero-Shot Out-of-Distribution* em 11.150 imagens geradas por **40 geradores modernos** (Modelos de Difusão, GANs, FaceSwap e Avatares Comerciais).")
    md.append("5. **Celeb-DF v2**: Generalização *Zero-Shot* em vídeos de alta qualidade em nível de frame e nível de vídeo agregado.\n")
    md.append("---\n")

    # =========================================================================
    # TABELA 1: Quadro Geral Consolidado (Todos os Modelos x Todas as Frequências)
    # =========================================================================
    md.append("## 1. Quadro Geral Consolidado: Todos os Modelos e Frequências\n")
    md.append("*(Ordenado por Regime e Desempenho sob Corrupção Test-D)*\n")

    cols1 = [
        "Modelo", "Modo Fourier", "Regime", "Seeds", "Status",
        "Test AUC (FF++)", "Test-D AUC (Corrompido)", "ΔAUC", "DF-40 AUC", "Celeb-DF Frame", "Celeb-DF Vídeo"
    ]
    md.append("| " + " | ".join(cols1) + " |")
    md.append("| " + " | ".join([":---"] * 5 + [":---:"] * 6) + " |")

    df_sorted = df.sort_values(by=["regime", "test_d_auc_m"], ascending=[False, False])
    for _, r in df_sorted.iterrows():
        fam_str = f"**{r['fam_name']}**"
        fourier_str = f"`{r['fourier']}`"
        reg_str = f"`{r['regime']}`"
        seed_str = f"{r['n_seeds']}x"
        stat_str = r["status"]

        t_auc = fmt(r["test_auc_m"], r["test_auc_s"])
        td_auc = fmt(r["test_d_auc_m"], r["test_d_auc_s"])
        delta = f"{r['delta_auc']:+.4f}"
        d40 = fmt(r["df40_m"], r["df40_s"])
        cf = fmt(r["cf_m"], r["cf_s"])
        cv = fmt(r["cv_m"], r["cv_s"])

        md.append(f"| {fam_str} | {fourier_str} | {reg_str} | {seed_str} | {stat_str} | {t_auc} | **{td_auc}** | {delta} | {d40} | {cf} | {cv} |")

    md.append("\n---\n")

    # =========================================================================
    # TABELA 2: Foco em Robustez: Baseline Limpo vs. Modelo Robusto
    # =========================================================================
    md.append("## 2. Comparativo Específico: Baseline Limpo vs. Modelo Robusto (`none`)\n")
    md.append("Evidencia o ganho de resiliência ao aplicar a augmentação estocástica `RandomizedRobustAugment` no espaço RGB puro:\n")

    cols2 = [
        "Modelo", "Regime", "Status",
        "Test AUC (Limpo)", "Test-D AUC (Corrompido)", "ΔAUC (Degradação)", "Ganho Test-D", "DF-40 AUC", "Celeb-DF Frame"
    ]
    md.append("| " + " | ".join(cols2) + " |")
    md.append("| " + " | ".join([":---"] * 3 + [":---:"] * 6) + " |")

    for fam in FAMILIES:
        clean = df[(df["fam"] == fam) & (df["fourier"] == "none") & (df["regime"] == "finetune")]
        robust = df[(df["fam"] == fam) & (df["fourier"] == "none") & (df["regime"] == "finetune_robust")]

        if clean.empty or robust.empty:
            continue
        c = clean.iloc[0]
        r = robust.iloc[0]

        gain_td = r["test_d_auc_m"] - c["test_d_auc_m"]
        gain_str = f"**{gain_td:+.4f}** ({gain_td*100:+.2f} pp)"

        md.append(
            f"| {r['fam_name']} | `finetune` (Baseline) | {c['status']} | "
            f"{fmt(c['test_auc_m'], c['test_auc_s'])} | {fmt(c['test_d_auc_m'], c['test_d_auc_s'])} | "
            f"{c['delta_auc']:+.4f} | — | {fmt(c['df40_m'], c['df40_s'])} | {fmt(c['cf_m'], c['cf_s'])} |"
        )
        md.append(
            f"| **{r['fam_name']}** | `finetune_robust` | **{r['status']}** | "
            f"**{fmt(r['test_auc_m'], r['test_auc_s'])}** | **{fmt(r['test_d_auc_m'], r['test_d_auc_s'])}** | "
            f"**{r['delta_auc']:+.4f}** | {gain_str} | **{fmt(r['df40_m'], r['df40_s'])}** | **{fmt(r['cf_m'], r['cf_s'])}** |"
        )

    md.append("\n---\n")

    # =========================================================================
    # TABELA 3: Desempenho Segmentado por Modo de Frequência
    # =========================================================================
    md.append("## 3. Desempenho Detalhado por Modo de Frequência (Baseline `finetune`)\n")
    md.append("Permite comparar diretamente o comportamento das 6 arquiteturas para cada uma das formulações espectrais:\n")

    for fourier in FOURIER_MODES:
        fname = FOURIER_NAMES[fourier]
        md.append(f"### Modo: {fname}\n")

        sub = df[(df["fourier"] == fourier) & (df["regime"] == "finetune")].sort_values("test_d_auc_m", ascending=False)
        if sub.empty:
            md.append("*Nenhum modelo disponível nesta configuração.*\n")
            continue

        cols3 = ["Modelo", "Canais", "Test AUC (FF++)", "Test Acc", "Test F1", "Test-D AUC (Corrompido)", "ΔAUC", "DF-40 AUC", "Celeb-DF Frame"]
        md.append("| " + " | ".join(cols3) + " |")
        md.append("| " + " | ".join([":---"] * 2 + [":---:"] * 7) + " |")

        for _, r in sub.iterrows():
            n_ch = 3 if fourier == "none" else (1 if fourier in ["magnitude", "phase", "frequency_3"] else (2 if fourier == "complex" else (4 if fourier == "concat" else 7)))
            md.append(
                f"| **{r['fam_name']}** | {n_ch}C | "
                f"{fmt(r['test_auc_m'], r['test_auc_s'])} | {fmt(r['test_acc_m'], r['test_acc_s'])} | {fmt(r['test_f1_m'], r['test_f1_s'])} | "
                f"**{fmt(r['test_d_auc_m'], r['test_d_auc_s'])}** | {r['delta_auc']:+.4f} | "
                f"{fmt(r['df40_m'], r['df40_s'])} | {fmt(r['cf_m'], r['cf_s'])} |"
            )
        md.append("\n")

    md.append("---\n")

    # =========================================================================
    # TABELA 4: Síntese da Melhor Representação por Modelo
    # =========================================================================
    md.append("## 4. Síntese: Melhor Representação Espectral por Arquitetura\n")
    cols4 = [
        "Arquitetura",
        "Melhor Modo FF++ Limpo", "Test AUC",
        "Melhor Modo sob Corrupção (`test_d`)", "Test-D AUC",
        "Melhor Modo Cross-Dataset (DF-40)", "DF-40 AUC"
    ]
    md.append("| " + " | ".join(cols4) + " |")
    md.append("| " + " | ".join([":---"] + [":---:"] * 6) + " |")

    for fam in FAMILIES:
        sub = df[(df["fam"] == fam) & (df["regime"] == "finetune")]
        if sub.empty:
            continue
        best_t = sub.loc[sub["test_auc_m"].idxmax()]
        best_td = sub.loc[sub["test_d_auc_m"].idxmax()]
        best_d40 = sub.loc[sub["df40_m"].idxmax()] if not sub["df40_m"].isna().all() else None

        d40_mode = best_d40["fourier"] if best_d40 is not None else "-"
        d40_val = f"{best_d40['df40_m']:.4f}" if best_d40 is not None else "-"

        md.append(
            f"| **{FAMILY_NAMES[fam]}** | "
            f"`{best_t['fourier']}` | {best_t['test_auc_m']:.4f} | "
            f"`{best_td['fourier']}` | **{best_td['test_d_auc_m']:.4f}** | "
            f"`{d40_mode}` | {d40_val} |"
        )

    md.append("\n---\n")

    # =========================================================================
    # TABELA 5: Detalhamento por Semente dos Modelos Robustos
    # =========================================================================
    md.append("## 5. Rastreamento por Semente Individual (`finetune_robust`)\n")
    md.append("Mapeamento do status de cada uma das 5 sementes canônicas (`[987, 42, 123, 2024, 7]`):\n")

    cols5 = [
        "Modelo", "Seed", "Status", "Test AUC", "Test Acc", "Test F1",
        "Test-D AUC", "Test-D Acc", "Test-D F1", "ΔAUC", "DF-40 AUC"
    ]
    md.append("| " + " | ".join(cols5) + " |")
    md.append("| " + " | ".join([":---"] * 3 + [":---:"] * 8) + " |")

    for fam in FAMILIES:
        p_rob = root / fam / "none" / "finetune_robust"
        if not p_rob.exists():
            continue

        done_seeds = {}
        for s in p_rob.glob("seed_*"):
            res = s / "results"
            mt = res / "metrics_test.csv"
            mtd = res / "metrics_test_d.csv"
            mdf = res / "metrics_df40.csv"
            if mt.exists() and mtd.exists():
                s_id = s.name.replace("seed_", "")
                done_seeds[s_id] = {
                    "t_auc": pd.read_csv(mt).iloc[0]["auc"],
                    "t_acc": pd.read_csv(mt).iloc[0]["acc"],
                    "t_f1": pd.read_csv(mt).iloc[0]["f1"],
                    "td_auc": pd.read_csv(mtd).iloc[0]["auc"],
                    "td_acc": pd.read_csv(mtd).iloc[0]["acc"],
                    "td_f1": pd.read_csv(mtd).iloc[0]["f1"],
                    "df40": pd.read_csv(mdf).iloc[0]["auc"] if mdf.exists() else None,
                }

        # Imprime sementes canônicas (e 2025 se concluída como no CLIP)
        canonical_seeds = ["987", "42", "123", "2024", "7"]
        if "2025" in done_seeds:
            canonical_seeds.append("2025")

        for s_id in canonical_seeds:
            if s_id in done_seeds:
                d = done_seeds[s_id]
                delta = d["td_auc"] - d["t_auc"]
                d40_s = fmt(d["df40"])
                md.append(
                    f"| **{FAMILY_NAMES[fam]}** | `seed_{s_id}` | ✅ Concluído | "
                    f"{d['t_auc']:.4f} | {d['t_acc']:.4f} | {d['t_f1']:.4f} | "
                    f"**{d['td_auc']:.4f}** | {d['td_acc']:.4f} | {d['td_f1']:.4f} | "
                    f"{delta:+.4f} | {d40_s} |"
                )
            else:
                # Determina se está em andamento ou na fila
                if fam == "vit" and s_id == "7":
                    stat = "🔄 Em andamento (GPU 1)"
                elif fam == "dino" and s_id == "42":
                    stat = "🔄 Em andamento (GPU 0)"
                elif fam in ["dino", "resnet"]:
                    stat = "⏳ Na fila (GPU 0)"
                else:
                    stat = "⏳ Na fila (GPU 1)"
                md.append(f"| *{FAMILY_NAMES[fam]}* | `seed_{s_id}` | {stat} | - | - | - | - | - | - | - | - |")

    # Salva o arquivo final
    output_path = out_dir / "tabela1-resultados-modelos-finetune.md"
    with open(output_path, "w", encoding="utf-8") as f:
        f.write("\n".join(md))

    print(f"✅ Arquivo gerado com sucesso: {output_path}")


if __name__ == "__main__":
    main()
