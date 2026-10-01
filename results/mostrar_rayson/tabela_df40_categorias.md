# Análise de Desempenho por Categoria no Benchmark DF-40 (DeepFake-40)

Este relatório detalha a taxa de acerto (**Recall / Sensibilidade** para deepfakes e **Especificidade** para rostos reais), a **Acurácia Balanceada** e a área sob a curva ROC (**AUC**) no benchmark **DF-40 (11.146 imagens)** para a campanha canônica SRM (5 seeds: 42, 123, 2024, 7, 2025).

---

## 1. Resumo por Paradigma (Macro-Categorias de Manipulação)

O benchmark DF-40 divide as manipulações faciais em **6 paradigmas generativos** principais, além do conjunto de **Rostos Reais (Pristine)**:

| Paradigma Generativo | Exemplos de Métodos no DF-40 | N Amostras | Descrição Forense |
| :--- | :--- | :---: | :--- |
| **Rostos Reais** | FFHQ, CelebA-HQ, MidJourney Real | 4.999 | Faces autênticas sem manipulação sintética. |
| **Edição Text-to-Image (T2I)** | StyleCLIP, WhichFaceIsReal, e4e, MidJourney | 998 | Edição semântica ou geração guiada por texto/espaço latente. |
| **Troca Facial (Face Swap)** | UniFace, FaceSwap, BlendFace, MobileSwap, DFL | 1.250 | Substituição da face mantendo contorno e iluminação. |
| **Reencenação (Talking)** | Celeb-DF Reenactment, SadTalker, Wav2Lip | 500 | Modificação sincronizada de lábios, fala e expressões. |
| **Síntese GAN** | StyleGAN2, StyleGAN3, StyleGAN-XL, VQGAN, StarGAN | 1.399 | Geração adversarial incondicional ou condicional de alta resolução. |
| **Modelos de Difusão** | Stable Diffusion 2.1, PixArt, SiT, DiT, CollabDiff, DDIM | 1.750 | Processos estocásticos de difusão reversa e denoise. |
| **Avatares Comerciais** | HeyGen | 250 | Vídeos corporativos de avatares sintéticos com pós-processamento industrial. |

---

## 2. Quanto Estamos Acertando por Categoria?

Abaixo apresentamos o desempenho comparativo entre o **CLIP ViT-B/16 (SRM)**, o **DINO ConvNeXt-B (SRM)**, a **Fusão Campeã (CLIP + DINO SRM)** e o **Mega-Ensemble (30 Modelos SRM)**.

### Tabela 1: Taxa de Acerto Direto e AUC por Paradigma

> **Legenda das Métricas:**
> - **Acerto Fake (%):** Proporção de imagens sintéticas da categoria corretamente identificadas como manipulação (*Recall / True Positive Rate*).
> - **Acerto Real (%):** Proporção de imagens autênticas corretamente mantidas como reais (*Especificidade / True Negative Rate*).
> - **BAcc (%):** Acurácia balanceada entre a categoria manipulada e os rostos reais $\frac{\text{Recall} + \text{Especificidade}}{2}$.
> - **AUC (%):** Capacidade intrínseca de discriminação probabilística independente do limiar.

| Paradigma | N | CLIP ViT-B/16 (SRM)<br>Acerto / AUC | DINO ConvNeXt-B (SRM)<br>Acerto / AUC | **Fusão CLIP + DINO (SRM)**<br>**Acerto / AUC / BAcc** | Mega-Ensemble (30 SRM)<br>Acerto / AUC |
| :--- | :---: | :---: | :---: | :---: | :---: |
| **Edição Text-to-Image (T2I)** | 998 | 90.5% / 90.0% | 97.4% / 91.3% | **97.5% / 92.2% (BAcc: 84.9%)** | 97.3% / 89.3% |
| **Troca Facial (Face Swap)** | 1.250 | 83.0% / 84.1% | 95.2% / 88.4% | **93.3% / 88.1% (BAcc: 82.8%)** | 96.6% / 85.4% |
| **Reencenação (Talking)** | 500 | 74.6% / 82.6% | 93.8% / 87.6% | **89.0% / 86.7% (BAcc: 80.6%)** | 96.4% / 86.9% |
| **Síntese GAN** | 1.399 | 78.4% / 83.4% | 80.2% / 80.5% | **85.0% / 83.8% (BAcc: 78.6%)** | 77.8% / 75.9% |
| **Modelos de Difusão** | 1.750 | 68.2% / 79.5% | 68.0% / 76.8% | **72.3% / 79.4% (BAcc: 72.3%)** | 76.4% / 75.9% |
| **Avatares Comerciais (HeyGen)** | 250 | 30.4% / 60.1% | 33.2% / 53.0% | **33.2% / 56.6% (BAcc: 52.7%)** | 47.6% / 59.8% |
| **Rostos Reais (Pristine)** | 4.999 | **75.5%** | **67.8%** | **72.2%** | **62.8%** |

---

## 3. Desempenho Detalhado por Técnica Individual (24 Métodos)

Detalhamento ordenado por taxa de acerto e AUC da **Fusão CLIP + DINO (SRM)**:

| Paradigma | Técnica Específica | Amostras | CLIP Acerto (%) | DINO Acerto (%) | **Fusão Acerto (%)** | **Fusão AUC (%)** |
| :--- | :--- | :---: | :---: | :---: | :---: | :---: |
| **Difusão** | **Stable Diffusion 2.1 (sd2.1)** | 250 | 99.2% | 100.0% | **100.0%** | **98.8%** |
| **Difusão** | **PixArt-$\alpha$** | 250 | 98.4% | 99.6% | **99.6%** | **96.2%** |
| **Edição T2I** | **WhichFaceIsReal** | 250 | 97.2% | 99.2% | **99.2%** | **95.1%** |
| **GAN** | **StyleGAN3** | 250 | 95.6% | 100.0% | **100.0%** | **94.1%** |
| **GAN** | **StyleGAN2** | 250 | 96.4% | 100.0% | **100.0%** | **93.4%** |
| **Face Swap** | **UniFace (High-Res)** | 250 | 92.4% | 99.6% | **98.8%** | **92.6%** |
| **Edição T2I** | **MidJourney** | 248 | 82.7% | 100.0% | **98.0%** | **92.2%** |
| **Edição T2I** | **e4e (encoder4editing)** | 250 | 96.8% | 95.6% | **98.0%** | **91.5%** |
| **Edição T2I** | **StyleCLIP** | 250 | 85.2% | 94.8% | **94.8%** | **90.1%** |
| **GAN** | **StyleGAN-XL** | 250 | 78.0% | 100.0% | **99.2%** | **88.8%** |
| **Face Swap** | **FaceSwap (Tradicional)** | 250 | 94.0% | 95.6% | **96.4%** | **88.7%** |
| **Face Swap** | **BlendFace** | 250 | 77.2% | 96.0% | **93.6%** | **87.7%** |
| **Reencenação** | **Celeb-DF Reenactment** | 500 | 74.6% | 93.8% | **89.0%** | **86.7%** |
| **Face Swap** | **MobileSwap** | 250 | 75.2% | 96.4% | **89.6%** | **86.4%** |
| **Face Swap** | **DeepFaceLab (DFL)** | 250 | 76.0% | 88.4% | **88.0%** | **84.7%** |
| **GAN** | **VQGAN** | 250 | 90.0% | 66.0% | **89.2%** | **83.1%** |
| **Difusão** | **SiT (Interpolant Transformers)**| 250 | 74.4% | 84.8% | **85.2%** | **83.1%** |
| **Difusão** | **DiT (Diffusion Transformers)** | 250 | 70.0% | 75.2% | **78.4%** | **80.6%** |
| **GAN** | **StarGAN** | 200 | 61.0% | 84.0% | **76.0%** | **79.8%** |
| **Difusão** | **RDDM** | 250 | 57.6% | 56.8% | **62.8%** | **74.5%** |
| **Difusão** | **CollabDiff** | 250 | 40.4% | 36.0% | **42.8%** | **64.1%** |
| **Difusão** | **DDIM** | 250 | 37.2% | 23.6% | **37.6%** | **58.8%** |
| **GAN** | **StarGAN v2** | 199 | 37.7% | 19.6% | **33.2%** | **57.4%** |
| **Avatar Comercial**| **HeyGen** | 250 | 30.4% | 33.2% | **33.2%** | **56.6%** |

---

## 4. Principais Conclusões Forenses

1. **Onde o Modelo Quase Não Erra (Acerto > 95%):**
   - **Difusão Moderna e T2I:** Stable Diffusion 2.1 (**100%**), PixArt (**99.6%**), WhichFaceIsReal (**99.2%**), MidJourney (**98.0%**) e e4e (**98.0%**).
   - **StyleGAN Clássico:** StyleGAN2 (**100%**) e StyleGAN3 (**100%**). Os filtros SRM de ruído de alta frequência expõem de forma inequívoca o padrão de grade característico da síntese por deconvolução/upsampling de GANs.
   - **Troca Facial em Alta Resolução:** UniFace (**98.8%**) e FaceSwap (**96.4%**). A descontinuidade espectral na borda do rosto clonado é capturada com precisão.

2. **Complementaridade Estratégica entre CLIP e DINO:**
   - No **VQGAN**, o DINO isolado teve apenas **66.0%** de acerto, enquanto o CLIP atingiu **90.0%**. A fusão recuperou **89.2%**.
   - No **MobileSwap**, o CLIP teve **75.2%**, mas o DINO alcançou **96.4%**, elevando a fusão para **89.6%**.
   - No **MidJourney**, o DINO gabaritou (**100%**), compensando a pontuação de 82.7% do CLIP para atingir **98.0%** de acerto combinado.

3. **As Técnicas Mais Desafiadoras e Evasivas:**
   - **HeyGen (Avatar Comercial - 33.2% de acerto):** Como os avatares comerciais passam por pipelines proprietários com interpolação temporal de vídeo, antialiasing estocástico e re-amostragem severa, a assinatura espectral SRM é atenuada, tornando-o o método mais evasivo de todo o DF-40.
   - **StarGAN v2 (33.2% de acerto) e DDIM (37.6% de acerto):** Representam métodos de baixa resolução ou técnicas com desvios latentes suaves onde o sinal de manipulação espacial é muito sutil.
