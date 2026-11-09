#!/usr/bin/env python3
"""Benchmark e Seleção Exaustiva de Ensembles Top-2 a Top-5 para os conjuntos:
1. Espacial Robusto: só RGB com augmentations robustas (avalia o teto anti-degradação).
2. Espacial Convencional: só RGB padrão (mostra o que a imagem sozinha sem robustez consegue).
3. Híbrido: RGB+SRM e RGB+Fourier (mostra se juntar imagem e frequência no mesmo modelo ajuda).
4. Frequência: só SRM e só Fourier (mostra se a frequência sozinha se sustenta).

Metodologia:
- Seleção estrita guiada pela Validação (MFFI Val):
  Para cada tamanho K (2, 3, 4, 5), seleciona-se a combinação de membros e a estratégia de fusão
  (entre mean, geometric, harmonic, max, weighted, optimal_weights, stacking, majority)
  que maximiza a AUC na partição de validação.
- Avaliação subsequente nos splits de teste:
  - MFFI Val (AUC)
  - MFFI Test (AUC)
  - MFFI Test-Hard / Test-D (AUC)
  - DF-40 Protocolo 1 (AUC)
  - Celeb-DF v2 Vídeo (AUC)
  - Celeb-DF v2 Frame (AUC)
"""

from __future__ import annotations
import itertools
from pathlib import Path
import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score
from sklearn.linear_model import LogisticRegression
from scipy.optimize import minimize

ROOT_DIR = Path(__file__).resolve().parent.parent
BASE_DIR = Path("/media/ssd2/lucas.ocunha/models-tcc")
TABLES_DIR = ROOT_DIR / "tables"

STRATEGIES = [
    "mean",
    "geometric",
    "harmonic",
    "max",
    "weighted",
    "optimal_weights",
    "stacking",
    "majority",
]

STRAT_PT = {
    "mean": "média aritmética",
    "geometric": "média geométrica",
    "harmonic": "média harmônica",
    "max": "máximo",
    "weighted": "ponderada (val)",
    "optimal_weights": "pesos ótimos (SLSQP)",
    "stacking": "stacking (reg. log.)",
    "majority": "voto majoritário",
}


def safe_auc(y_true, y_score):
    try:
        y_score = np.nan_to_num(y_score, nan=0.5, posinf=1.0, neginf=0.0)
        if len(np.unique(y_true)) < 2:
            return 0.5
        return float(roc_auc_score(y_true, y_score))
    except Exception:
        return 0.5


def fit_stacking(val_probs_matrix, val_y):
    eps = 1e-6
    X_val = np.clip(val_probs_matrix, eps, 1.0 - eps)
    clf = LogisticRegression(random_state=42, max_iter=1000, C=1.0)
    clf.fit(X_val, val_y)
    return clf


def fit_optimal_weights(val_probs_matrix, val_y):
    K = val_probs_matrix.shape[1]
    eps = 1e-7
    def loss(w):
        w_norm = w / (np.sum(w) + eps)
        p = np.dot(val_probs_matrix, w_norm)
        p = np.clip(p, eps, 1.0 - eps)
        return -np.mean(val_y * np.log(p) + (1.0 - val_y) * np.log(1.0 - p))
    w0 = np.ones(K) / K
    bnds = [(0, 1) for _ in range(K)]
    cons = ({"type": "eq", "fun": lambda w: np.sum(w) - 1.0})
    res = minimize(loss, w0, method="SLSQP", bounds=bnds, constraints=cons)
    return res.x / np.sum(res.x)


def fuse(probs_list, strategy, val_aucs=None, stacking_clf=None, optimal_w=None):
    stack = np.stack(probs_list, axis=0) # [K, N]
    eps = 1e-7
    if strategy == "mean":
        return np.mean(stack, axis=0)
    elif strategy == "geometric":
        clipped = np.clip(stack, eps, 1.0 - eps)
        return np.exp(np.mean(np.log(clipped), axis=0))
    elif strategy == "harmonic":
        clipped = np.clip(stack, eps, 1.0)
        return len(probs_list) / np.sum(1.0 / clipped, axis=0)
    elif strategy == "max":
        return np.max(stack, axis=0)
    elif strategy == "weighted":
        weights = np.maximum(np.array(val_aucs) - 0.5, 0.0)
        if weights.sum() == 0:
            return np.mean(stack, axis=0)
        return np.average(stack, axis=0, weights=weights)
    elif strategy == "optimal_weights":
        return np.dot(optimal_w, stack)
    elif strategy == "stacking":
        X = np.clip(stack.T, eps, 1.0 - eps)
        return stacking_clf.predict_proba(X)[:, 1]
    elif strategy == "majority":
        return np.mean((stack >= 0.5).astype(float), axis=0)
    return np.mean(stack, axis=0)


def load_model_data(path: Path):
    data = {}
    with np.load(path / "outputs_val.npz") as d:
        p = d["probs"]
        data["val_p"] = p[:, 1] if p.ndim > 1 and p.shape[1] > 1 else p.ravel()
        data["val_y"] = d["y_true"]
    with np.load(path / "outputs_test.npz") as d:
        p = d["probs"]
        data["test_p"] = p[:, 1] if p.ndim > 1 and p.shape[1] > 1 else p.ravel()
        data["test_y"] = d["y_true"]
    with np.load(path / "outputs_test_d.npz") as d:
        p = d["probs"]
        data["test_d_p"] = p[:, 1] if p.ndim > 1 and p.shape[1] > 1 else p.ravel()
        data["test_d_y"] = d["y_true"]
    with np.load(path / "outputs_celeb_df.npz") as d:
        data["celeb_frame_p"] = d["p_fake"]
        data["celeb_frame_y"] = d["y_true"]
        data["celeb_video_p"] = d["video_p"]
        data["celeb_video_y"] = d["video_y"]
    with np.load(path / "outputs_df40.npz") as d:
        data["df40_p"] = d["probs"]
        data["df40_y"] = d["y_true"]
    return data


def scan_all_models():
    print("Escaneando modelos disponíveis em /media/ssd2/lucas.ocunha/models-tcc...")
    models = []
    for p in BASE_DIR.glob("**/outputs_val.npz"):
        res = p.parent
        rel = res.relative_to(BASE_DIR)
        parts = rel.parts
        if len(parts) >= 4:
            fam, mode, reg, sd = parts[0], parts[1], parts[2], parts[3]
            # Verificar se todos os 4 splits existem
            if not ((res / "outputs_test.npz").exists() and 
                    (res / "outputs_test_d.npz").exists() and 
                    (res / "outputs_celeb_df.npz").exists() and 
                    (res / "outputs_df40.npz").exists()):
                continue
            try:
                with np.load(p) as d:
                    probs = d["probs"]
                    probs = probs[:, 1] if probs.ndim > 1 and probs.shape[1] > 1 else probs.ravel()
                    auc = safe_auc(d["y_true"], probs)
                models.append({
                    "family": fam,
                    "mode": mode,
                    "regime": reg,
                    "seed": sd,
                    "val_auc": auc,
                    "path": res,
                    "id": f"{fam}_{mode}_{reg}_{sd}",
                })
            except Exception:
                pass
    df = pd.DataFrame(models)
    print(f"Total de modelos válidos com todos os splits: {len(df)}")
    return df


def evaluate_committee_on_val(members, loaded_cache):
    val_probs = [loaded_cache[m]["val_p"] for m in members]
    val_y = loaded_cache[members[0]]["val_y"]
    val_aucs = [safe_auc(val_y, p) for p in val_probs]
    val_mat = np.stack(val_probs, axis=1)

    stacking_clf = fit_stacking(val_mat, val_y)
    optimal_w = fit_optimal_weights(val_mat, val_y)

    best_strat = None
    best_val_auc = -1.0

    for strat in STRATEGIES:
        p_fused = fuse(val_probs, strat, val_aucs=val_aucs, stacking_clf=stacking_clf, optimal_w=optimal_w)
        auc = safe_auc(val_y, p_fused)
        if auc > best_val_auc:
            best_val_auc = auc
            best_strat = strat

    return best_strat, best_val_auc, stacking_clf, optimal_w, val_aucs


def evaluate_committee_all_splits(members, strat, loaded_cache, stacking_clf, optimal_w, val_aucs):
    splits = {
        "val": ([loaded_cache[m]["val_p"] for m in members], loaded_cache[members[0]]["val_y"]),
        "test": ([loaded_cache[m]["test_p"] for m in members], loaded_cache[members[0]]["test_y"]),
        "test_d": ([loaded_cache[m]["test_d_p"] for m in members], loaded_cache[members[0]]["test_d_y"]),
        "df40": ([loaded_cache[m]["df40_p"] for m in members], loaded_cache[members[0]]["df40_y"]),
        "celeb_video": ([loaded_cache[m]["celeb_video_p"] for m in members], loaded_cache[members[0]]["celeb_video_y"]),
        "celeb_frame": ([loaded_cache[m]["celeb_frame_p"] for m in members], loaded_cache[members[0]]["celeb_frame_y"]),
    }

    res = {}
    for s_name, (p_list, y_true) in splits.items():
        p_fused = fuse(p_list, strat, val_aucs=val_aucs, stacking_clf=stacking_clf, optimal_w=optimal_w)
        res[f"{s_name}_auc"] = safe_auc(y_true, p_fused) * 100.0

    return res


def format_member_display(m_id):
    # Formato: {fam}_{mode}_{reg}_{sd}
    fam = m_id.split("_")[0].capitalize()
    if fam == "Vit": fam = "ViT"
    elif fam == "Mobilenet": fam = "MobileNetV3"
    elif fam == "Resnet": fam = "ResNet-18"
    elif fam == "Dino": fam = "DINOv3"
    elif fam == "Clip": fam = "CLIP"
    elif fam == "Xception": fam = "Xception"

    if "_srm_only_" in m_id: mode_str = "SRM-only"
    elif "_concat_frequency_" in m_id: mode_str = "RGB+Freq3"
    elif "_frequency_3_" in m_id: mode_str = "Fourier 3-Bandas"
    elif "_srm_" in m_id: mode_str = "RGB+SRM"
    elif "_concat_" in m_id: mode_str = "RGB+Fourier"
    elif "_none_" in m_id: mode_str = "RGB"
    elif "_magnitude_" in m_id: mode_str = "Magnitude"
    elif "_phase_" in m_id: mode_str = "Fase"
    elif "_complex_" in m_id: mode_str = "Espectro Compl."
    else: mode_str = ""

    return f"{fam} {mode_str}"


def run_benchmark():
    df_all = scan_all_models()

    # 1. Definição das candidaturas
    # Espacial Robusto: só RGB com regime finetune_robust
    df_esp_rob = df_all[(df_all["mode"] == "none") & (df_all["regime"] == "finetune_robust")].sort_values("val_auc", ascending=False)
    esp_rob_candidates = df_esp_rob.drop_duplicates(subset=["family"]).head(6)

    # Espacial Convencional: só RGB com regime finetune
    df_esp_conv = df_all[(df_all["mode"] == "none") & (df_all["regime"] == "finetune")].sort_values("val_auc", ascending=False)
    esp_conv_candidates = df_esp_conv.drop_duplicates(subset=["family"]).head(6)

    # Híbrido: RGB+SRM e RGB+Fourier (mode in ["srm", "concat", "concat_frequency"])
    # Selecionar os melhores representantes por família e modo
    df_hib = df_all[df_all["mode"].isin(["srm", "concat", "concat_frequency"])].sort_values("val_auc", ascending=False)
    hib_candidates = df_hib.drop_duplicates(subset=["family", "mode"]).head(10)

    # Frequência: só SRM e só Fourier (mode in ["srm_only", "magnitude", "phase", "complex", "frequency_3"])
    df_freq = df_all[df_all["mode"].isin(["srm_only", "magnitude", "phase", "complex", "frequency_3"])].sort_values("val_auc", ascending=False)
    freq_candidates = df_freq.drop_duplicates(subset=["family", "mode"]).head(10)

    print("\n--- Candidatos Espacial Robusto Selecionados (Top por Família) ---")
    for _, r in esp_rob_candidates.iterrows():
        print(f"  {r['family']:12s} | Val AUC: {r['val_auc']:.4f} | {r['id']}")

    print("\n--- Candidatos Espacial Convencional Selecionados (Top por Família) ---")
    for _, r in esp_conv_candidates.iterrows():
        print(f"  {r['family']:12s} | Val AUC: {r['val_auc']:.4f} | {r['id']}")

    print("\n--- Candidatos Híbridos Selecionados (Top por Família + Modo) ---")
    for _, r in hib_candidates.iterrows():
        print(f"  {r['family']:12s} | {r['mode']:15s} | Val AUC: {r['val_auc']:.4f} | {r['id']}")

    print("\n--- Candidatos Frequência Selecionados (Top por Família + Modo) ---")
    for _, r in freq_candidates.iterrows():
        print(f"  {r['family']:12s} | {r['mode']:15s} | Val AUC: {r['val_auc']:.4f} | {r['id']}")

    # Pré-carregar em memória os modelos necessários
    all_needed_paths = set(esp_rob_candidates["path"]).union(esp_conv_candidates["path"]).union(hib_candidates["path"]).union(freq_candidates["path"])
    print(f"\nCarregando dados de {len(all_needed_paths)} modelos distintos...")
    cache = {}
    path_to_id = {}
    for _, r in df_all.iterrows():
        if r["path"] in all_needed_paths:
            path_to_id[r["path"]] = r["id"]
            if r["id"] not in cache:
                cache[r["id"]] = load_model_data(r["path"])
    print("Cache pronto!")

    # Processar cada grupo para K = 2, 3, 4, 5
    groups = [
        ("Espacial Robusto (RGB Robusto)", esp_rob_candidates["id"].tolist()),
        ("Espacial Convencional (RGB Convencional)", esp_conv_candidates["id"].tolist()),
        ("Híbrido (RGB+SRM e RGB+Fourier)", hib_candidates["id"].tolist()),
        ("Frequência (só SRM e só Fourier)", freq_candidates["id"].tolist()),
    ]

    results_table = []

    for g_name, cand_ids in groups:
        print(f"\n=======================================================")
        print(f"🔍 Otimizando Ensembles para o Grupo: {g_name}")
        print(f"=======================================================")

        for K in [2, 3, 4, 5]:
            best_val = -1.0
            best_combo = None
            best_strat = None
            best_clf = None
            best_w = None
            best_aucs = None

            # Testar combinações de K membros
            for combo in itertools.combinations(cand_ids, K):
                # Restrição opcional para híbrido: garantir que tenha pelo menos 1 SRM e 1 Fourier se K >= 3
                if "Híbrido" in g_name and K >= 3:
                    has_srm = any("srm" in m for m in combo)
                    has_fourier = any("concat" in m for m in combo)
                    if not (has_srm and has_fourier):
                        continue

                # Restrição opcional para frequência: garantir pelo menos 1 SRM-only e 1 Fourier se K >= 3
                if "Frequência" in g_name and K >= 3:
                    has_srm_only = any("srm_only" in m for m in combo)
                    has_fourier = any(m.split("_")[1] in ["magnitude", "phase", "complex", "frequency_3"] for m in combo)
                    if not (has_srm_only and has_fourier):
                        continue

                strat, val_auc, clf, opt_w, val_aucs = evaluate_committee_on_val(combo, cache)
                if val_auc > best_val:
                    best_val = val_auc
                    best_combo = combo
                    best_strat = strat
                    best_clf = clf
                    best_w = opt_w
                    best_aucs = val_aucs

            # Avaliar nos splits de teste com a melhor estratégia escolhida na validação
            metrics = evaluate_committee_all_splits(best_combo, best_strat, cache, best_clf, best_w, best_aucs)

            members_display = " + ".join([format_member_display(m) for m in best_combo])
            print(f"Top-{K} | Val: {metrics['val_auc']:.2f}% | Test: {metrics['test_auc']:.2f}% | Test-D: {metrics['test_d_auc']:.2f}% | DF-40: {metrics['df40_auc']:.2f}% | Celeb-DF: {metrics['celeb_video_auc']:.2f}% | Fusão: {best_strat}")
            print(f"   Membros: {members_display}")

            results_table.append({
                "Grupo": g_name,
                "K": K,
                "Comitê": f"Top-{K} {g_name.split('(')[0].strip()}",
                "Membros": members_display,
                "Estratégia": best_strat,
                "Estratégia_PT": STRAT_PT.get(best_strat, best_strat),
                "Val_AUC": metrics["val_auc"],
                "Test_AUC": metrics["test_auc"],
                "Test_Hard_AUC": metrics["test_d_auc"],
                "DF40_AUC": metrics["df40_auc"],
                "Celeb_Video_AUC": metrics["celeb_video_auc"],
                "Celeb_Frame_AUC": metrics["celeb_frame_auc"],
            })

    df_res = pd.DataFrame(results_table)
    out_csv = TABLES_DIR / "benchmark_ensembles_espacial_hibrido_frequencia.csv"
    df_res.to_csv(out_csv, index=False)
    print(f"\n💾 Tabela salva com sucesso em: {out_csv}")

    # Gerar LaTeX
    generate_latex_table(df_res)
    generate_markdown_table(df_res)


def generate_latex_table(df: pd.DataFrame):
    tex_path = TABLES_DIR / "benchmark_ensembles_espacial_hibrido_frequencia.tex"
    lines = [
        r"\begin{table*}[!htb]",
        r"\centering",
        r"\small",
        r"\caption{Comparação dos Ensembles Top-2 ao Top-5 por Paradigma de Representação (Espacial Robusto, Espacial Convencional, Híbrido e Frequência), sempre selecionando o melhor comitê e a melhor estratégia de fusão na validação do MFFI.}",
        r"\label{tab:ensembles_paradigmas}",
        r"\vspace{-2mm}",
        r"\resizebox{0.95\textwidth}{!}{",
        r"\begin{tabular}{@{}ll l c c c c c@{}}",
        r"\toprule",
        r"\textbf{Comitê} & \textbf{Membros} & \textbf{Fusão} & \textbf{Val (\%)} & \textbf{Test (\%)} & \textbf{Test-Hard (\%)} & \textbf{DF-40 (\%)} & \textbf{Celeb-DF Vídeo (\%)} \\",
        r"\midrule",
    ]

    curr_group = None
    for _, r in df.iterrows():
        g = r["Grupo"]
        if g != curr_group:
            if curr_group is not None:
                lines.append(r"\midrule")
            curr_group = g
            lines.append(f"\\multicolumn{{8}}{{@{{}}l}}{{\\textit{{\\textbf{{{g}}}}}}} \\\\")

        k_str = f"Top-{r['K']}"
        fusao_str = r['Estratégia_PT']
        val = f"{r['Val_AUC']:.2f}"
        test = f"{r['Test_AUC']:.2f}"
        hard = f"{r['Test_Hard_AUC']:.2f}"
        df40 = f"{r['DF40_AUC']:.2f}"
        celeb = f"{r['Celeb_Video_AUC']:.2f}"

        lines.append(f"{k_str} & {r['Membros']} & \\texttt{{{fusao_str}}} & {val} & {test} & {hard} & {df40} & {celeb} \\\\")

    lines.extend([
        r"\bottomrule",
        r"\end{tabular}",
        r"}",
        r"\end{table*}",
    ])

    tex_path.write_text("\n".join(lines))
    print(f"📄 Tabela LaTeX gerada em: {tex_path}")


def generate_markdown_table(df: pd.DataFrame):
    md_path = TABLES_DIR / "benchmark_ensembles_espacial_hibrido_frequencia.md"
    lines = [
        "# Benchmark de Ensembles: Espacial Robusto vs. Espacial Convencional vs. Híbrido vs. Frequência (Top-2 a Top-5)",
        "",
        "Esta tabela consolida os ensembles otimizados na partição de validação do **MFFI** para responder às hipóteses centrais da pesquisa:",
        "1. **Espacial Robusto (RGB Robusto):** Avalia o teto da representação espacial quando submetida a data augmentation robusta (anti-degradação).",
        "2. **Espacial Convencional (RGB Convencional):** Mede o comportamento dos modelos puramente espaciais sem aumento de robustez sob degradações do mundo real.",
        "3. **Híbrido (RGB+SRM e RGB+Fourier):** Avalia se a fusão explícita de aparência espacial com pistas espectrais no mesmo detector melhora a generalização.",
        "4. **Frequência (só SRM e só Fourier):** Testa se os resíduos espectrais e de ruído conseguem se sustentar isoladamente, sem qualquer informação de cor RGB.",
        "",
        "| Grupo | Comitê | Membros | Estratégia de Fusão | Val AUC (%) | Test AUC (%) | Test-Hard AUC (%) | DF-40 AUC (%) | Celeb-DF v2 Vídeo (%) |",
        "| :--- | :---: | :--- | :--- | :---: | :---: | :---: | :---: | :---: |",
    ]

    for _, r in df.iterrows():
        lines.append(
            f"| **{r['Grupo']}** | Top-{r['K']} | {r['Membros']} | `{r['Estratégia_PT']}` | "
            f"**{r['Val_AUC']:.2f}** | {r['Test_AUC']:.2f} | {r['Test_Hard_AUC']:.2f} | {r['DF40_AUC']:.2f} | {r['Celeb_Video_AUC']:.2f} |"
        )

    md_path.write_text("\n".join(lines))
    print(f"📊 Resumo Markdown gerado em: {md_path}")


if __name__ == "__main__":
    run_benchmark()
