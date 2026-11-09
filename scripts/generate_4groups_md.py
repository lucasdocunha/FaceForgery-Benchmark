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
    "> **Critério de Seleção Rigoroso:** Todos os comitês foram otimizados estritamente na partição de validação do MFFI (sem conhecimento prévio dos conjuntos de teste).",
    "> Avaliação multivariada compreendendo: AUC (%) no Test Limpo (FF++ c23), Test-Hard (14 corrupções de distorção física), Celeb-DF v2 (por vídeo H.264), DF-40 (protocolo 1 balanceado) e **Revocação no Protocolo 2 do DF-40 ($650.549$ imagens falsas de 24 técnicas)** com limiar ótimo calibrado via índice de Youden na validação.",
    "",
    "## 📊 Tabela Consolidada de Resultados",
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

md_lines.extend([
    "",
    "---",
    "",
    "## 🔬 Justificativa e Análise Científica de Cada Ensemble",
    "",
    "### 1️⃣ Categoria: Espacial Robusto (RGB com Aumento Robusto)",
    "**Premissa e Inductive Bias:** Modelos que operam no domínio de cor RGB, mas foram submetidos durante o treinamento a um protocolo agressivo de perturbações sintéticas (compressão JPEG variável, desfoque gaussiano e ruído aditivo).",
    "",
    "* **Top-2: DINOv3 RGB + CLIP RGB (Fusão: Média Geométrica)**",
    "  * **Por que esses membros?** DINOv3 e CLIP são os dois modelos mais capazes de todo o benchmark. O DINOv3 foi pré-treinado de forma autossupervisionada (foco em coerência estrutural densa e bordas de patch), enquanto o CLIP foi pré-treinado por alinhamento multimodal texto-imagem (foco em semântica global de alto nível). Por partirem de princípios de aprendizado opostos, seus erros são fortemente desacoplados.",
    "  * **Por que Média Geométrica?** A média geométrica $(\\sqrt{p_1 \\cdot p_2})$ penaliza severamente a divergência: a predição só é confiante se ambos os detectores concordarem. Isso elimina falsos positivos causados por ruído ou compressão, resultando no **maior resultado de Test-Hard de todo o trabalho (89,0% de AUC)** e na maior taxa de revocação do DF-40 Protocolo 2 (**88,2%** entre os comitês espaciais).",
    "",
    "* **Top-3 e Top-4 (+ MobileNetV3 / + Xception, Fusão: Pesos Ótimos SLSQP)**",
    "  * **Por que Pesos Ótimos?** Ao buscar a combinação linear que maximiza a verossimilhança na validação, o algoritmo SLSQP atribui peso quase total a DINOv3 e CLIP ($w_1 + w_2 \\approx 0{,}98$), praticamente zerando os pesos dos modelos convolucionais menores ($w \\le 0{,}02$).",
    "  * **O Fenômeno da Saturação (Rendimentos Decrescentes):** Adicionar MobileNetV3 ou Xception não agrega diversidade representacional contra perturbações e introduz ruído marginal, fazendo o Test-Hard cair de **89,0% para 87,3%** e a revocação do DF-40 P2 recuar para **81,8%**. Prova que *dois backbones de grande porte com pré-treinos distintos superam comitês maiores com arquiteturas mais fracas*.",
    "",
    "---",
    "",
    "### 2️⃣ Categoria: Espacial Convencional (RGB Padrão)",
    "**Premissa e Inductive Bias:** O baseline tradicional da literatura de deepfakes: finetuning supervisionado sobre imagens RGB sem aumento robusto contra degradações.",
    "",
    "* **Top-2: DINOv3 RGB + CLIP RGB (Fusão: Média Geométrica)**",
    "  * Apresenta excelente generalização em imagens limpas de laboratório (**94,9% no Test AUC**), superando inclusive a versão robusta nessa partição por reter artefatos sutis de pixels não mascarados pelo aumento de dados.",
    "  * **O Gargalo:** Sob as 14 corrupções físicas do Test-Hard, despenca para **83,4%** (queda de mais de 11 pontos percentuais), evidenciando a fragilidade do RGB convencional no mundo real.",
    "",
    "* **Top-3 e Top-4 (+ ResNet-18 / + MobileNetV3, Fusão: Stacking)**",
    "  * **Por que Stacking?** A validação seleciona um meta-classificador linear (Regressão Logística) para aprender correlações entre as saídas dos modelos.",
    "  * **O Gargalo no DF-40:** O Stacking não é capaz de inventar invariância física que não foi aprendida na extração de features. Como consequência, o Test-Hard cai para **80,6%** e a revocação no DF-40 P2 atinge as menores marcas de todo o estudo (**72,3% a 74,6%**). Demonstra empiricamente que *técnicas de fusão tardia não compensam a ausência de representações robustas*.",
    "",
    "---",
    "",
    "### 3️⃣ Categoria: Híbrido Multidomínio (RGB+SRM e RGB+Fourier)",
    "**Premissa e Inductive Bias:** Fusão precoce de canais (*early fusion*). O modelo recebe simultaneamente a informação visual de aparência (RGB) e um canal complementar explícito: resíduos de ruído de alta frequência (SRM - 6 canais) ou a magnitude 2D da transformada rápida de Fourier (FFT - 4º canal).",
    "",
    "* **Top-2: DINOv3 RGB+SRM + DINOv3 RGB+Fourier (Fusão: Stacking)**",
    "  * Avalia a complementaridade de duas representações distintas sob o mesmo backbone visual forte, atingindo 94,6% no teste limpo e 80,5% de revocação no DF-40 P2.",
    "",
    "* **Top-3: DINOv3 RGB+SRM + DINOv3 RGB+Fourier + CLIP RGB+SRM (Fusão: Média Geométrica)**",
    "  * **Por que é o Campeão no Test Limpo (95,1% AUC) e no DF-40 Protocolo 1 (81,8% AUC)?**",
    "    1. O canal de **Fourier** expõe imediatamente os artefatos de grade periódica (*checkerboard artifacts*) característicos dos blocos de upsampling de redes convolucionais e modelos de Difusão latente.",
    "    2. O canal **SRM** extrai a quebra de correlação espacial entre pixels adjacentes, identificando descontinuidades de interpolação em trocas faciais.",
    "    3. A aparência **RGB** ancora o contexto semântico global da face.",
    "  * A fusão multidomínio atinge **84,5% de revocação no DF-40 P2**, provando que pistas espectrais auxiliam na detecção de geradores desconhecidos (*zero-shot cross-generator generalization*).",
    "",
    "* **Top-4 (+ ResNet-18 RGB+Fourier, Fusão: Pesos Ótimos)**",
    "  * Mantém desempenho próximo ao Top-3 (84,2% de revocação no DF-40 P2 e 74,0% no Celeb-DF), confirmando a robustez da combinação híbrida mesmo com a adição de uma CNN clássica.",
    "",
    "---",
    "",
    "### 4️⃣ Categoria: Não-RGB Puro (SRM-only + Fourier-only)",
    "**Premissa e Inductive Bias:** Supressão total do sinal de cor e textura visual RGB. O modelo é obrigado a tomar decisões baseando-se estritamente em propriedades físicas:",
    "  1. *Filtros SRM:* 30 filtros esteganográficos fixos que extraem ruído de sensor fotográfico e resíduos residuais;",
    "  2. *Fourier:* Decomposição espectral por magnitude, fase e decomposição em três bandas de frequência radial.",
    "",
    "* **Por que Stacking foi a Estratégia Escolhida em Todos os Comitês (Top-2 a Top-4)?**",
    "  * Resíduos de ruído de sensor e coeficientes de Fourier possuem distribuições estatísticas completamente heterogêneas. Médias simples tratam grandezas distintas com o mesmo peso. O Stacking treina hiperplanos ótimos na validação para mapear a confiabilidade de cada espectro físico.",
    "",
    "* **Por que quebra o teto no Celeb-DF v2 (78,4% de AUC por vídeo no Top-3)?**",
    "  * **Eliminação do Atalho de Textura (*Shortcut Learning*):** Modelos RGB aprendem a identificar deepfakes baseando-se na iluminação, textura da pele e bordas de recorte das imagens de treino (FF++). Quando avaliados em vídeos do Celeb-DF v2 gravados em cenários reais e comprimidos pelo codec H.264, esses atalhos visuais desaparecem e os detectores RGB falham ($65\\%$ a $70\\%$ AUC).",
    "  * Os modelos Não-RGB operam no domínio onde a compressão de vídeo e a interpolação do DeepFaceLab deixam perturbações drásticas no ruído de alta frequência. Ao ignorar o rosto e focar no sinal de ruído, o Top-3 Não-RGB alcança **78,4% de AUC por vídeo**, superando todos os ensembles espaciais do benchmark.",
    "",
    "* **Comportamento no DF-40 P2:**",
    "  * Atinge **85,1% de revocação no Top-2 e 83,6% no Top-4** em 650.549 imagens falsas. Isso prova conclusivamente que *artefatos de frequência e ruído residual são universais entre arquiteturas geradoras*, permitindo identificar fakes sem jamais visualizar a cor dos pixels.",
    "",
    "---",
    "",
    "## 🎯 Guia de Aplicação Operacional (Qual Ensemble usar?)",
    "",
    "| Objetivo Forense / Cenário Operacional | Ensemble Recomendado | Métrica Chave | Motivo da Escolha |",
    "| :--- | :--- | :---: | :--- |",
    "| **Alta Resiliência a Degradação (Redes Sociais, WhatsApp)** | **Top-2 Espacial Robusto** | **89,0%** Test-Hard | DINOv3 + CLIP com média geométrica resistem a blur, ruído e compressão JPEG extrema. |",
    "| **Detecção de Novos Geradores e T2I (DF-40)** | **Top-3 Híbrido Multidomínio** | **81,8%** DF-40 / **95,1%** Test | Fourier e SRM expõem artefatos de grade e interpolação de novos modelos de IA generativa. |",
    "| **Auditoria e Perícia de Vídeos Reais (Celeb-DF v2)** | **Top-3 Não-RGB (SRM+Fourier)** | **78,4%** Celeb Vídeo | Remove atalhos visuais e analisa ruído de sensor residual corrompido pela troca de rostos. |",
    "| **Triagem Forense de Máxima Segurança** | **Top-3 Híbrido com Regra OR** | **99,2%** Revocação P2 | Garante que praticamente nenhum deepfake escape simultaneamente dos três domínios. |"
])

with open("tables/benchmark_4groups_top2_top4.md", "w") as f:
    f.write("\n".join(md_lines) + "\n")

print("Generated full scientific report in tables/benchmark_4groups_top2_top4.md")
