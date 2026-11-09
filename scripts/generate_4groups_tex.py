import pandas as pd

df = pd.read_csv("tables/benchmark_4groups_top2_top4.csv")

tex_lines = [
    r"\begin{table*}[!htb]",
    r"\centering",
    r"\caption{Comparativo sistemático de comitês (Top-2 a Top-4) em quatro categorias de representação no benchmark, otimizados na validação do \gls*{mffi}. Métricas em \%: \gls*{auc} nas partições do \gls*{mffi}, no Celeb-DF v2 (por vídeo) e no DF-40 (protocolo~1), acompanhadas da Revocação ponderada no protocolo~2 do DF-40 ($650.549$ imagens falsas, limiar calibrado na validação). Em negrito, o maior valor de cada coluna.}",
    r"\label{tab:ensembles_4categorias_top2_top4}",
    r"\vspace{-2mm}",
    r"\resizebox{\textwidth}{!}{",
    r"\begin{tabular}{@{}ll l l c c c c c c@{}}",
    r"\toprule",
    r"\textbf{Categoria} & \textbf{Comitê} & \textbf{Membros Integrantes} & \textbf{Fusão} & \makecell{\textbf{Val}\\\textbf{(\%)}} & \makecell{\textbf{Test}\\\textbf{(\%)}} & \makecell{\textbf{Test-Hard}\\\textbf{(\%)}} & \makecell{\textbf{Celeb-DF}\\\textbf{Vídeo (\%)}} & \makecell{\textbf{DF-40}\\\textbf{AUC (\%)}} & \makecell{\textbf{DF-40 P2}\\\textbf{Revoc. (\%)}} \\",
    r"\midrule"
]

cat_map = {
    "Espacial Robusto": r"\multirow{3}{*}{\makecell[l]{\textbf{Espacial}\\\textbf{Robusto}\\\textit{(RGB Aumento)}}}",
    "Espacial Convencional": r"\multirow{3}{*}{\makecell[l]{\textbf{Espacial}\\\textbf{Convencional}\\\textit{(RGB Padrão)}}}",
    "Híbrido Multidomínio": r"\multirow{3}{*}{\makecell[l]{\textbf{Híbrido}\\\textbf{Multidomínio}\\\textit{(RGB+SRM/Freq)}}}",
    "Não-RGB (SRM e Frequência)": r"\multirow{3}{*}{\makecell[l]{\textbf{Não-RGB}\\\textbf{Puro}\\\textit{(SRM+Fourier)}}}"
}

max_vals = {
    "Val_AUC": df["Val_AUC"].max(),
    "Test_AUC": df["Test_AUC"].max(),
    "Test_Hard_AUC": df["Test_Hard_AUC"].max(),
    "Celeb_Video_AUC": df["Celeb_Video_AUC"].max(),
    "DF40_AUC": df["DF40_AUC"].max(),
    "DF40_P2_Rec": df["DF40_P2_Rec"].max(),
}

curr_group = None
for idx, r in df.iterrows():
    g = r["Grupo"]
    k_str = f"Top-{r['K']}"
    m_str = r["Membros"].replace(" + ", r" \textbf{+} ")
    fusao = r["Estratégia_PT"].replace(" (SLSQP)", "").replace(" (reg. log.)", "")
    
    def fmt(col):
        v = r[col]
        s = f"{v:.1f}".replace(".", ",")
        if abs(v - max_vals[col]) < 0.05:
            return f"\\textbf{{{s}}}"
        return s

    v_val = fmt("Val_AUC")
    v_test = fmt("Test_AUC")
    v_hard = fmt("Test_Hard_AUC")
    v_celeb = fmt("Celeb_Video_AUC")
    v_df40 = fmt("DF40_AUC")
    v_p2 = fmt("DF40_P2_Rec")

    if g != curr_group:
        if curr_group is not None:
            tex_lines.append(r"\midrule")
        curr_group = g
        c_label = cat_map[g]
        tex_lines.append(f"{c_label} & {k_str} & {m_str} & \\texttt{{{fusao}}} & {v_val} & {v_test} & {v_hard} & {v_celeb} & {v_df40} & {v_p2} \\\\")
    else:
        tex_lines.append(f" & {k_str} & {m_str} & \\texttt{{{fusao}}} & {v_val} & {v_test} & {v_hard} & {v_celeb} & {v_df40} & {v_p2} \\\\")

tex_lines.extend([
    r"\bottomrule",
    r"\end{tabular}",
    r"}",
    r"\end{table*}"
])

with open("tables/benchmark_4groups_top2_top4.tex", "w") as f:
    f.write("\n".join(tex_lines) + "\n")

print("LaTeX table generated successfully in tables/benchmark_4groups_top2_top4.tex")
