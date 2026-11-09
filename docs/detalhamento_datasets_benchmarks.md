# Detalhamento Completo dos Datasets e Benchmarks Forenses

**Documento:** `docs/detalhamento_datasets_benchmarks.md`  
**Escopo:** Caracterização técnica, balanceamento de classes, taxonomia de geradores e protocolos de avaliação de todos os datasets utilizados na pesquisa.  
**Projeto:** FaceForgery Benchmark — Detecção Forense de Faces Forjadas, Robustez e Generalização.

---

## 📌 Visão Geral dos Três Pilares de Dados

A pesquisa organiza a avaliação em três pilares complementares de dados, cobrindo desde o aprendizado *in-distribution* até a generalização extrema *out-of-distribution* contra modelos generativos modernos:

```
┌────────────────────────────────────────────────────────────────────────────────────────────────────────┐
│                                   ARQUITETURA DE DADOS DO PROJETO                                      │
├────────────────────────────────────┬───────────────────────────────────┬───────────────────────────────┤
│ 1. MFFI / FaceForensics++ (Base)   │ 2. Celeb-DF v2 (Alta Fidelidade)  │ 3. DeepFake-40 / DF-40 (SoTA) │
├────────────────────────────────────┼───────────────────────────────────┼───────────────────────────────┤
│ • Treino, Validação e Teste Limpo  │ • Avaliação Cross-Dataset em Vídeo│ • 40 Métodos Generativos      │
│ • Teste Degradado (Test-D, 14 corr)│ • Geração DeepFaceLab Refinada    │ • Difusão, GANs, Avatares     │
│ • 853.739 amostras totais          │ • 13.000+ frames / ~518 vídeos    │ • 11.146 imagens estáticas    │
└────────────────────────────────────┴───────────────────────────────────┴───────────────────────────────┘
```

---

## 1. MFFI / FaceForensics++ (Conjunto Canônico de Treino e Teste)

O **MFFI (Multi-Factor FaceForensics Initiative)** é o corpus principal utilizado para o treinamento, validação de hiperparâmetros e teste controlado *in-distribution*.

### 1.1 Distribuição e Balanceamento de Classes

O dataset é particionado em 3 splits disjuntos de identidades:

| Partição | Amostras Reais (0) | Amostras Fakes (1) | Total Amostras | Proporção Real / Fake | Finalidade Metodológica |
| :--- | :---: | :---: | :---: | :---: | :--- |
| **Treino (`train`)** | 99.386 (18.95%) | 425.043 (81.05%) | **524.429** | ~1 : 4.3 | Treinamento dos backbones com amostragem balanceada ponderada. |
| **Validação (`val`)** | 59.082 (40.09%) | 88.281 (59.91%) | **147.363** | ~1 : 1.5 | Calibração de threshold (Youden), early stopping e tuning. |
| **Teste Limpo (`test`)** | 77.602 (42.65%) | 104.345 (57.35%) | **181.947** | ~1 : 1.3 | Avaliação de benchmark in-distribution em imagens intactas. |
| **Teste Degradado (`test_d`)** | 77.602 (42.65%) | 104.345 (57.35%) | **181.947** | ~1 : 1.3 | Avaliação de resiliência sob 14 corrupções realistas severas. |
| **TOTAL** | **313.672** | **722.014** | **1.035.686** | — | — |

### 1.2 Características Técnicas e Métodos Inclusos
* **Compressão:** Padrão H.264 com fator de quantização constante **c23** (qualidade visual média-alta, compressão realista de vídeos web).
* **Métodos de Manipulação Inclusos:**
  1. *Deepfakes* (autoencoder com decodificadores específicos por identidade).
  2. *Face2Face* (reencenação facial em tempo real transferindo expressões).
  3. *FaceSwap* (transferência gráfica de faces baseada em marcos faciais 3D).
  4. *NeuralTextures* (renderização neural baseada em texturas dinâmicas).
  5. *FaceShifter* (oclusão facial e preservação de atributos).
* **Partição Degradada (`test_d` - ImageNet-C Adaptado):**
  Aplica 14 corrupções adversas estocásticas sobre as 181.947 imagens de teste:
  - *Compressões:* JPEG agressivo ($QF \in [20, 50]$) e H.264 de baixa taxa de bits.
  - *Ruídos:* Gaussiano, impulsivo (sal e pimenta) e speckle.
  - *Desfoques:* Blur gaussiano, desfoque de movimento (*motion blur*) e desfoque óptico de lente (*defocus*).
  - *Distorções ambientais:* Variação de brilho, contraste e saturação.

---

## 2. Celeb-DF v2 (Benchmark de Alta Qualidade e Nível de Vídeo)

O **Celeb-DF v2** é amplamente reconhecido como o benchmark mais desafiador da primeira geração de deepfakes, criado para superar os artefatos fáceis do FaceForensics++.

### 2.1 Balanceamento e Características de Partição

| Unidade de Avaliação | Amostras Reais | Amostras Fakes | Total Amostras | Proporção |
| :--- | :---: | :---: | :---: | :---: |
| **Nível de Vídeo (`celeb_df_video`)** | ~59 vídeos (11.4%) | ~459 vídeos (88.6%) | **~518 vídeos** | **1 : 7.8 (Altamente desbalanceado)** |
| **Nível de Frame (`celeb_df_frame`)** | ~4.500 frames | ~8.500+ frames | **13.000+ faces** | **1 : 1.9** |

### 2.2 Desafios Forenses e Peculiaridades
1. **Geração por DeepFaceLab Aprimorado:** Os vídeos foram manipulados utilizando versões modificadas do DeepFaceLab com mascaramento suave de bordas (*soft boundary feathering*), equalização de histograma de cores e síntese em resolução $512\times512$.
2. **Ausência de Descontinuidades Óbvias:** Reduz drasticamente inconsistências visuais perceptíveis ao olho humano, cores não-casadas e bordas serrilhadas típicas do FF++.
3. **Agregação Temporal (Frame vs Vídeo):**
   - **Por Frame:** A rede classifica cada frame de forma isolada.
   - **Por Vídeo:** Calcula a média probabilística de todos os frames detectados pertencentes ao mesmo vídeo:
     $$\text{Score}_{\text{vídeo}} = \frac{1}{T} \sum_{t=1}^{T} P(\text{Fake} \mid x_t)$$
   - *Ganho empírico:* Em 100% dos modelos testados, a avaliação por vídeo foi entre **+2.6 pp e +5.7 pp superior** à de frame, porque suaviza frames ruidosos causados por *motion blur* ou perfis faciais extremos.

### 2.3 Resolução da Anomalia Histórica de Inversão de Polaridade
* **O Problema:** Na lista oficial de teste (`List_of_testing_videos.txt`), o autor anotou `1 = Real` e `0 = Fake` (o inverso da convenção MFFI onde `1 = Fake`).
* **A Correção:** A auditoria do projeto corrigiu formalmente o mapeamento dos rótulos no script avaliador. Resultados anteriores que aparentavam ter caído para ~28% a 35% de AUC revelaram-se desempenhos reais de **72.60% no RGB Robusto** e picos de **83.19% no DINO SRM** (e **83.79% no Ensemble SRM**).

---

## 3. DeepFake-40 / DF-40 (Benchmark de Generalização Multigerador)

O **DF-40** foi estruturado para testar a capacidade do modelo de detectar tecnologias generativas emergentes que não existiam quando os datasets clássicos foram concebidos.

### 3.1 Balanceamento e Estatísticas Globais

Diferente do Celeb-DF, o DF-40 foi construído com balanceamento próximo do ideal:

| Categoria | N° Amostras | Percentual | Descrição dos Dados |
| :--- | :---: | :---: | :--- |
| **Rostos Reais (Pristine)** | **4.999** | **44.85%** | Faces reais fotográficas em altíssima resolução (FFHQ, CelebA-HQ e retratos fotográficos não-manipulados). |
| **Deepfakes Sintéticos** | **6.147** | **55.15%** | Faces manipuladas ou geradas por 40 técnicas distintas. |
| **TOTAL** | **11.146** | **100.0%** | Imagens estáticas em alta definição ($512\times512$ a $1024\times1024$). |

---

### 3.2 Taxonomia dos 6 Paradigmas Generativos e 40 Geradores

O benchmark categoriza as técnicas em 6 macro-famílias tecnológicas:

| Paradigma Generativo | N° Amostras | Exemplos de Métodos Inclusos | Assinatura Forense Observada | Dificuldade para o Modelo |
| :--- | :---: | :--- | :--- | :---: |
| **1. Modelos de Difusão** | 1.750 (28.5%) | Stable Diffusion 2.1, PixArt-$\alpha$, DiT, SiT, DDIM, CollabDiff, RDDM | Ruído de difusão reversa estocástica em médias frequências. | Média a Baixa (Acerto: 72% a 100%) |
| **2. Síntese GAN Pura** | 1.399 (22.8%) | StyleGAN2, StyleGAN3, StyleGAN-XL, VQGAN, StarGAN, StarGAN v2 | Artefatos de grade (*checkerboard*) causados por upsampling/convoluções transpostas. | Baixa (Acerto: 85% a 100%) |
| **3. Troca Facial (Face Swap)** | 1.250 (20.3%) | UniFace, DeepFaceLab (DFL), FaceSwap, BlendFace, MobileSwap | Descontinuidades de interpolação na borda de fusão (*boundary blending*). | Baixa (Acerto: 88% a 99%) |
| **4. Edição Text-to-Image (T2I)** | 998 (16.2%) | MidJourney, StyleCLIP, WhichFaceIsReal, e4e (*encoder4editing*) | Modificação semântica direcionada por prompts em espaço latente. | Baixa (Acerto: 95% a 99%) |
| **5. Reencenação (Talking Heads)** | 500 (8.1%) | Celeb-DF Reenactment, SadTalker, Wav2Lip | Desfoque e dessincronia na região dos lábios e queixo. | Média (Acerto: 89% a 96%) |
| **6. Avatares Comerciais** | 250 (4.1%) | HeyGen | Pós-processamento de estúdio com antialiasing avançado e re-amostragem severa. | **Alta / Evasivo** (Acerto: 33.2%) |

---

### 3.3 Análise de Desempenho por Método no DF-40 (Comitê CLIP + DINO SRM)

O ranking das técnicas individuais no DF-40 revela onde a detecção é trivial e onde a tecnologia generativa atinge evasão:

```
                                  TAXA DE ACERTO POR TÉCNICA NO DF-40
 ┌───────────────────────────────────────────────┬────────────┬─────────────┬───────────────────────────┐
 │ Técnica Generativa                            │ Família    │ N Amostras  │ Taxa de Acerto (Recall)   │
 ├───────────────────────────────────────────────┼────────────┼─────────────┼───────────────────────────┤
 │ Stable Diffusion 2.1 (SD 2.1)                 │ Difusão    │ 250         │ 100.0% 🟢 (AUC: 98.8%)    │
 │ StyleGAN2 / StyleGAN3                         │ GAN        │ 500         │ 100.0% 🟢 (AUC: 94.1%)    │
 │ PixArt-Alpha                                  │ Difusão    │ 250         │ 99.6%  🟢 (AUC: 96.2%)    │
 │ WhichFaceIsReal                               │ T2I        │ 250         │ 99.2%  🟢 (AUC: 95.1%)    │
 │ UniFace (High-Res Swap)                       │ Face Swap  │ 250         │ 98.8%  🟢 (AUC: 92.6%)    │
 │ MidJourney                                    │ T2I        │ 248         │ 98.0%  🟢 (AUC: 92.2%)    │
 │ e4e (Encoder4Editing)                         │ T2I        │ 250         │ 98.0%  🟢 (AUC: 91.5%)    │
 │ FaceSwap Tradicional                          │ Face Swap  │ 250         │ 96.4%  🟢 (AUC: 88.7%)    │
 │ StyleCLIP                                     │ T2I        │ 250         │ 94.8%  🟢 (AUC: 90.1%)    │
 │ BlendFace                                     │ Face Swap  │ 250         │ 93.6%  🟢 (AUC: 87.7%)    │
 │ MobileSwap                                    │ Face Swap  │ 250         │ 89.6%  🟡 (AUC: 86.4%)    │
 │ Celeb-DF Reenactment (Talking)                │ Reencenação│ 500         │ 89.0%  🟡 (AUC: 86.7%)    │
 │ DeepFaceLab (DFL)                             │ Face Swap  │ 250         │ 88.0%  🟡 (AUC: 84.7%)    │
 │ SiT (Scalable Interpolant Transformers)       │ Difusão    │ 250         │ 85.2%  🟡 (AUC: 83.1%)    │
 │ DiT (Diffusion Transformers)                  │ Difusão    │ 250         │ 78.4%  🟡 (AUC: 80.6%)    │
 │ StarGAN Clássico                              │ GAN        │ 200         │ 76.0%  🟡 (AUC: 79.8%)    │
 │ CollabDiff                                    │ Difusão    │ 250         │ 42.8%  🔴 (AUC: 64.1%)    │
 │ DDIM (Denoising Diffusion Implicit Models)    │ Difusão    │ 250         │ 37.6%  🔴 (AUC: 58.8%)    │
 │ StarGAN v2                                    │ GAN        │ 199         │ 33.2%  🔴 (AUC: 57.4%)    │
 │ HeyGen (Avatar Corporativo)                   │ Comercial  │ 250         │ 33.2%  🔴 (AUC: 56.6%)    │
 └───────────────────────────────────────────────┴────────────┴─────────────┴───────────────────────────┘
```

#### Fatores Científicos de Sucesso e Evasão:
1. **Por que SD 2.1, StyleGAN e PixArt são facilmente detectados (>99%):**
   Os filtros espaciais de ruído (**SRM**) suprimem o conteúdo semântico visual e expõem a periodicidade espacial de deconvoluções de GANs e o padrão uniforme de ruído de alta frequência introduzido pelos escaladores de Difusão.
2. **Por que HeyGen e StarGAN v2 são evasivos (~33%):**
   O HeyGen utiliza pós-processadores de vídeo comerciais proprietários (filtros temporais anti-cintilação e *antialiasing* bicúbico) que atuam como filtros passa-baixas naturais, suavizando as altas frequências e apagando as assinaturas esteganográficas que os detectores de ruído procuram.

---

## 4. Comparativo Master: Características Estruturais dos Três Datasets

| Dimensão Técnica | MFFI (FaceForensics++) | Celeb-DF v2 | DeepFake-40 (DF-40) |
| :--- | :---: | :---: | :---: |
| **Formato dos Dados** | Imagens estáticas extraídas de vídeo | Vídeos codificados (MPEG-4/H.264) | Imagens estáticas em alta resolução |
| **Tamanho Amostral** | 853.739 faces (524k treino, 182k teste) | ~518 vídeos (13.000+ frames) | 11.146 imagens |
| **Balanceamento Real / Fake** | ~1 : 4.3 (Treino) \| ~1 : 1.3 (Teste) | **1 : 7.8 (Desbalanceado: 11% Real)** | **1 : 1.2 (Equilibrado: 45% Real)** |
| **Resolução Típica** | $256\times256$ a $224\times224$ | $512\times512$ a $1024\times1024$ | $512\times512$ a $1024\times1024$ |
| **Qualidade da Síntese** | Baixa a Média (c23, bordas visíveis) | Muito Alta (DeepFaceLab customizado) | Estado da Arte (Difusão, MidJourney, GANs) |
| **Variedade de Geradores** | 5 técnicas clássicas | 1 técnica (DeepFaceLab) | **40 técnicas geradoras distintas** |
| **Propósito Metodológico** | Treino supervisionado e robustez a corrupções | Teste cross-dataset em nível de vídeo | Generalização contra geradores desconhecidos |
| **Melhor Modelo Individual** | **DINO SRM (92.29% Test / 85.00% Test-D)** | **DINO SRM (75.54% / pico 83.19%)** | **CLIP Robusto (81.55% AUC)** |
| **Melhor Ensemble de Fusão** | **Ensemble Robusto RGB (88.45% Test-D)** | **Ensemble SRM (83.79% Vídeo AUC)** | **Super-Ensemble Multimodal (86.44% AUC)** |

---

## 5. Referências e Scripts de Manuseio no Repositório

* **Manifestos Canônicos:** Localizados em [`data/manifests/`](../data/manifests/) (`train.csv`, `val.csv`, `test.csv`).
* **Scripts de Avaliação:**
  * Avaliação DF-40: [`scripts/evaluate_df40.py`](../scripts/evaluate_df40.py) e [`scripts/deep_df40_analysis.py`](../scripts/deep_df40_analysis.py).
  * Avaliação Celeb-DF v2: [`scripts/prepare_celeb_df.py`](../scripts/prepare_celeb_df.py) e [`scripts/recompute_all_celeb_df.py`](../scripts/recompute_all_celeb_df.py).
  * Avaliação MFFI / Test-D: [`scripts/run_forensics_campaign.py`](../scripts/run_forensics_campaign.py) e [`scripts/aggregate_robust_benchmarks.py`](../scripts/aggregate_robust_benchmarks.py).
