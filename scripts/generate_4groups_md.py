import pandas as pd

df = pd.read_csv("tables/benchmark_4groups_top2_top4.csv")

max_vals = {
    'Val_AUC': df['Val_AUC'].max(),
    'Test_AUC': df['Test_AUC'].max(),
    'Test_Hard_AUC': df['Test_Hard_AUC'].max(),
    'Celeb_Video_AUC': df['Celeb_Video_AUC'].max(),
    'DF40_AUC': df['DF40_AUC'].max(),
    'DF40_P2_Rec': df['DF40_P2_Rec'].max(),
}

md_lines = [
    "# Comparativo Sistemático de Ensembles (Top-2 ao Top-4 em 4 Categorias)",
    "",
    "> Avaliação abrangente comparando comitês otimizados na validação do MFFI.",
    "> Inclui AUC (%) nos 4 benchmarks de teste e a **Revocação no Protocolo 2 do DF-40 (650.549 imagens falsas)** com limiar ótimo da validação.",
    "",
    "| Categoria | Comitê | Membros Integrantes | Fusão | Val AUC (%) | Test AUC (%) | Test-Hard AUC (%) | Celeb-DF Vídeo (%) | DF-40 AUC (%) | DF-40 P2 Revoc. (%) |",
    "| :--- | :---: | :--- | :--- | :---: | :---: | :---: | :---: | :---: | :---: |"
]

for idx, r in df.iterrows():
    g = r['Grupo']
    k = f"Top-{r['K']}"
    m = r['Membros']
    f_pt = r['Estratégia_PT']
    
    def fmt(col):
        v = r[col]
        s = f"{v:.1f}%".replace('.', ',')
        if abs(v - max_vals[col]) < 0.05:
            return f"**{s}**"
        return s
        
    md_lines.append(f"| {g} | {k} | {m} | `{f_pt}` | {fmt('Val_AUC')} | {fmt('Test_AUC')} | {fmt('Test_Hard_AUC')} | {fmt('Celeb_Video_AUC')} | {fmt('DF40_AUC')} | {fmt('DF40_P2_Rec')} |")

with open('tables/benchmark_4groups_top2_top4.md', 'w') as f:
    f.write('\n'.join(md_lines) + '\n')

print("Markdown updated!")
