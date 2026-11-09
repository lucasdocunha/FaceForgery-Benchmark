import pandas as pd
import numpy as np

df = pd.read_csv('tables/tabela6_top3_e_ensemble_mffi_val.csv')

# Markdown generation
md_lines = [
    '# Tabela 6: Revocação (%) por Técnica no Protocolo 2 do DF-40 (650.549 Imagens)',
    '',
    '> **Protocolo 2 Oficial:** Avaliação exaustiva em **650.549 imagens falsas** distribuídas em 24 técnicas de manipulação facial.',
    '> Modelos individuais e **Ensemble Top-3** (fusão por soft voting / média de probabilidades dos 3 melhores modelos: DINOv3 RGB+Freq, Xception SRM-only e Xception RGB+SRM) com limiar ótimo calibrado na validação do MFFI.',
    '> Médias e desvios-padrão reportados ao longo de 3 sementes (42, 123 e 2024). Em negrito, o maior valor por técnica.',
    '',
    '| Paradigma | Técnica | N (Amostras) | DINOv3 (RGB+Freq) | Xception (SRM-only) | Xception (RGB+SRM) | Ensemble Top-3 (Val. MFFI) |',
    '| :--- | :--- | :---: | :---: | :---: | :---: | :---: |'
]

for idx, r in df.iterrows():
    p = r['paradigma']
    t = r['tecnica']
    n = f"{int(r['n_amostras']):,}".replace(',', '.')
    d = r['dino_freq']
    xso = r['xcep_srm_only']
    xs = r['xcep_srm']
    em = r['ens_top3_val']
    
    vals = [d, xso, xs, em]
    max_val = max(vals)
    
    fmt = []
    for v in vals:
        s = f"{v:.1f}%".replace('.', ',')
        if abs(v - max_val) < 0.05:
            s = f"**{s}**"
        fmt.append(s)
        
    md_lines.append(f"| {p} | {t} | {n} | {fmt[0]} | {fmt[1]} | {fmt[2]} | {fmt[3]} |")

tot_n = int(df['n_amostras'].sum())
tot_d = (df['n_amostras'] * df['dino_freq']).sum() / tot_n
tot_xso = (df['n_amostras'] * df['xcep_srm_only']).sum() / tot_n
tot_xs = (df['n_amostras'] * df['xcep_srm']).sum() / tot_n
tot_em = (df['n_amostras'] * df['ens_top3_val']).sum() / tot_n

max_tot = max([tot_d, tot_xso, tot_xs, tot_em])
tot_fmt = []
for v in [tot_d, tot_xso, tot_xs, tot_em]:
    s = f"{v:.1f}%".replace('.', ',')
    if abs(v - max_tot) < 0.05:
        s = f"**{s}**"
    tot_fmt.append(s)

tot_n_str = f"{tot_n:,}".replace(',', '.')
md_lines.append(f"| **TOTAL GERAL** | **24 técnicas** | **{tot_n_str}** | {tot_fmt[0]} | {tot_fmt[1]} | {tot_fmt[2]} | {tot_fmt[3]} |")


with open('tables/tabela6_top3_e_ensemble_mffi_val.md', 'w') as f:
    f.write('\n'.join(md_lines) + '\n')

print('MD saved successfully!')

# LaTeX generation
tex_lines = [
    r"\begin{table}[!htb]",
    r"\centering",
    r"\caption{Revocação (\%) por técnica no protocolo~2 do DF-40 ($650.549$ imagens falsas, média de 3 sementes) para os três melhores modelos do benchmark e o comitê Top-3 calibrado com limiar ótimo da validação do \gls*{mffi}. Em negrito, o melhor resultado por técnica.}",
    r"\label{tab:tecnicas}",
    r"\vspace{-2mm}",
    r"\resizebox{\linewidth}{!}{",
    r"\begin{tabular}{@{}llr ccc c@{}}",
    r"\toprule",
    r"Paradigma & Técnica & $N$ & \makecell{DINOv3\\(RGB+Freq)} & \makecell{Xception\\(SRM-only)} & \makecell{Xception\\(RGB+SRM)} & \makecell{\textbf{Ensemble Top-3}\\\textbf{(Val.\ MFFI)}} \\",
    r"\midrule"
]

paradigm_map = {
    'Difusão': r"\multirow{7}{*}{Difusão}",
    'Síntese GAN': r"\multirow{6}{*}{\gls*{gan}}",
    'Troca Facial': r"\multirow{5}{*}{\makecell[l]{Troca\\facial}}",
    'Edição / T2I': r"\multirow{4}{*}{\makecell[l]{Edição/\\T2I}}",
    'Misto': "Misto",
    'Avatar': "Avatar"
}

current_paradigm = ''
for idx, r in df.iterrows():
    p = r['paradigma']
    t = r['tecnica']
    if t == 'PixArt-alpha':
        t_tex = r"PixArt-$\alpha$"
    else:
        t_tex = t
        
    n = f"{int(r['n_amostras']):,}".replace(',', '.')
    d = r['dino_freq']
    xso = r['xcep_srm_only']
    xs = r['xcep_srm']
    em = r['ens_top3_val']
    
    vals = [d, xso, xs, em]
    max_val = max(vals)
    
    fmt = []
    for v in vals:
        s = f"{v:.1f}".replace('.', ',')
        if abs(v - max_val) < 0.05:
            s = f"\\textbf{{{s}}}"
        fmt.append(s)
        
    if p != current_paradigm:
        if current_paradigm != '':
            tex_lines.append(r"\midrule")
        current_paradigm = p
        p_label = paradigm_map.get(p, p)
        tex_lines.append(f"{p_label} & {t_tex} & {n} & {fmt[0]} & {fmt[1]} & {fmt[2]} & {fmt[3]} \\\\")
    else:
        tex_lines.append(f" & {t_tex} & {n} & {fmt[0]} & {fmt[1]} & {fmt[2]} & {fmt[3]} \\\\")

tot_fmt_tex = []
for v in [tot_d, tot_xso, tot_xs, tot_em]:
    s = f"{v:.1f}".replace('.', ',')
    if abs(v - max_tot) < 0.05:
        s = f"\\textbf{{{s}}}"
    tot_fmt_tex.append(s)

tex_lines.extend([
    r"\midrule[\heavyrulewidth]",
    f"\\multicolumn{{2}}{{@{{}}l}}{{\\textbf{{TOTAL GERAL ({tot_n_str})}}}} & & {tot_fmt_tex[0]} & {tot_fmt_tex[1]} & {tot_fmt_tex[2]} & {tot_fmt_tex[3]} \\\\",
    r"\bottomrule",
    r"\end{tabular}",
    r"}",
    r"\end{table}"
])

with open('tables/tabela6_top3_e_ensemble_mffi_val.tex', 'w') as f:
    f.write('\n'.join(tex_lines) + '\n')

print('TeX saved successfully!')
