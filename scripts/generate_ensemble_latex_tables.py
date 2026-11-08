#!/usr/bin/env python3
"""Gera tabelas LaTeX (.tex) formatadas com booktabs para inclusão direta no Overleaf / TCC."""

from pathlib import Path
import pandas as pd

def generate_latex():
    # 1. Tabela Não-RGB
    csv_non_rgb = Path("tables/benchmark_non_rgb_ensembles_all_methods.csv")
    if csv_non_rgb.exists():
        df = pd.read_csv(csv_non_rgb)
        # Filtrar as melhores estratégias ou tabela completa
        tex_path = Path("tables/benchmark_non_rgb_ensembles.tex")
        lines = [
            r"\begin{table*}[t]",
            r"\centering",
            r"\small",
            r"\caption{Desempenho dos Ensembles Forenses Não-RGB (SRM e Fourier Concat) sob Múltiplas Estratégias de Fusão.}",
            r"\label{tab:non_rgb_ensembles}",
            r"\begin{tabular}{l c l c c c c}",
            r"\toprule",
            r"\textbf{Comitê Não-RGB} & \textbf{$K$} & \textbf{Estratégia} & \textbf{Val AUC (\%)} & \textbf{Test AUC (\%)} & \textbf{Test-D AUC (\%)} & \textbf{DF-40 AUC (\%)} & \textbf{Celeb-DF Vídeo (\%)} \\",
            r"\midrule",
        ]
        
        current_comite = None
        for _, r in df.iterrows():
            c = r["Comitê"]
            if c != current_comite:
                if current_comite is not None:
                    lines.append(r"\midrule")
                current_comite = c
                c_display = f"\\textbf{{{c.split('(')[0].strip()}}}"
            else:
                c_display = ""
            
            strat = r["Estratégia"]
            val_a = f"{r['val_auc']:.2f}"
            test_a = f"{r['test_auc']:.2f}"
            test_d = f"{r['test_d_auc']:.2f}"
            df40 = f"{r['df40_auc']:.2f}"
            celeb = f"{r['celeb_video_auc']:.2f}"
            
            # Negrito nos picos
            if float(test_d) >= 87.30: test_d = f"\\textbf{{{test_d}}}"
            if float(df40) >= 86.30: df40 = f"\\textbf{{{df40}}}"
            if float(celeb) >= 78.00: celeb = f"\\textbf{{{celeb}}}"
            
            lines.append(f"{c_display} & {r['K']} & \\texttt{{{strat}}} & {val_a} & {test_a} & {test_d} & {df40} & {celeb} \\\\")
        
        lines.extend([
            r"\bottomrule",
            r"\end{tabular}",
            r"\end{table*}",
        ])
        
        tex_path.write_text("\n".join(lines))
        print(f"LaTeX gerado: {tex_path}")

    # 2. Tabela Híbrida Top-2 até Top-5
    csv_hybrid = Path("tables/benchmark_top_ensembles_all_methods.csv")
    if csv_hybrid.exists():
        df2 = pd.read_csv(csv_hybrid)
        tex_path2 = Path("tables/benchmark_top_ensembles_all_methods.tex")
        lines2 = [
            r"\begin{table*}[t]",
            r"\centering",
            r"\small",
            r"\caption{Avaliação Exaustiva dos Super-Ensembles Campeões (Top-2 ao Top-5) em Todos os Métodos de Fusão.}",
            r"\label{tab:hybrid_ensembles_all}",
            r"\begin{tabular}{l c l c c c c}",
            r"\toprule",
            r"\textbf{Comitê Híbrido} & \textbf{$K$} & \textbf{Estratégia} & \textbf{Val AUC (\%)} & \textbf{Test AUC (\%)} & \textbf{Test-D AUC (\%)} & \textbf{DF-40 AUC (\%)} & \textbf{Celeb-DF Vídeo (\%)} \\",
            r"\midrule",
        ]
        
        curr2 = None
        for _, r in df2.iterrows():
            c = r["Comitê"]
            if c != curr2:
                if curr2 is not None:
                    lines2.append(r"\midrule")
                curr2 = c
                c_display = f"\\textbf{{{c}}}"
            else:
                c_display = ""
            
            strat = r["Estratégia"]
            val_a = f"{r['val_auc']:.2f}"
            test_a = f"{r['test_auc']:.2f}"
            test_d = f"{r['test_d_auc']:.2f}"
            df40 = f"{r['df40_auc']:.2f}"
            celeb = f"{r['celeb_video_auc']:.2f}"
            
            if float(df40) >= 86.50: df40 = f"\\textbf{{{df40}}}"
            if float(test_d) >= 83.90: test_d = f"\\textbf{{{test_d}}}"
            if float(celeb) >= 74.70: celeb = f"\\textbf{{{celeb}}}"
            if float(test_a) >= 94.60: test_a = f"\\textbf{{{test_a}}}"
            
            lines2.append(f"{c_display} & {r['K']} & \\texttt{{{strat}}} & {val_a} & {test_a} & {test_d} & {df40} & {celeb} \\\\")
            
        lines2.extend([
            r"\bottomrule",
            r"\end{tabular}",
            r"\end{table*}",
        ])
        tex_path2.write_text("\n".join(lines2))
        print(f"LaTeX gerado: {tex_path2}")

if __name__ == "__main__":
    generate_latex()
