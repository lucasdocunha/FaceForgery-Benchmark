# Tabela 6: Revocação por Técnica no DF-40 (Top-3 Modelos e Melhor Ensemble MFFI)

> Avaliação oficial em **6.147 imagens falsas** (24 técnicas) e **4.999 imagens reais** (subset do DF-40, mesmo da Tabela 6 do artigo).
> Valores de **Revocação (%)** (média $\pm$ desvio-padrão de 3 sementes: 42, 123 e 2024), com limiar ótimo determinado na validação do MFFI.

| Paradigma | Técnica | N | DINOv3 (RGB+Freq)<br>*(Top-1 Individual)* | Xception (SRM-only)<br>*(Top-2 Individual)* | Xception (RGB+SRM)<br>*(Top-3 Individual)* | **Ensemble Top-3**<br>*(Melhor Validação MFFI)* |
| :--- | :--- | :---: | :---: | :---: | :---: | :---: |
| Difusão | SD 2.1 | 250 | 99.2% | 99.9% | 99.6% | **100.0%** |
| Difusão | PixArt-alpha | 250 | 99.1% | 98.0% | **99.9%** | **99.9%** |
| Difusão | SiT | 250 | 90.5% | 86.9% | 90.9% | **93.9%** |
| Difusão | DiT | 250 | 92.3% | 85.3% | 90.3% | **92.8%** |
| Difusão | RDDM | 250 | 98.9% | 91.7% | 96.5% | **100.0%** |
| Difusão | CollabDiff | 250 | **72.5%** | 71.2% | 66.8% | 71.2% |
| Difusão | DDIM | 250 | 57.3% | **68.8%** | 58.4% | 58.8% |
| Síntese GAN | StyleGAN3 | 250 | 99.3% | 98.7% | **100.0%** | **100.0%** |
| Síntese GAN | StyleGAN2 | 250 | 97.3% | 95.2% | **99.1%** | 98.9% |
| Síntese GAN | StyleGAN-XL | 250 | 99.1% | 95.6% | 93.9% | **99.2%** |
| Síntese GAN | VQGAN | 250 | **96.3%** | 71.2% | 71.1% | 94.7% |
| Síntese GAN | StarGAN | 200 | **74.3%** | 37.5% | 50.5% | 61.2% |
| Síntese GAN | StarGAN v2 | 199 | 39.4% | **42.2%** | 24.6% | 26.0% |
| Troca Facial | UniFace | 250 | 94.8% | 93.5% | 94.3% | **97.1%** |
| Troca Facial | FaceSwap | 250 | 94.8% | 89.3% | 92.8% | **96.7%** |
| Troca Facial | BlendFace | 250 | 98.1% | 94.5% | 94.1% | **99.1%** |
| Troca Facial | MobileSwap | 250 | 97.3% | 90.1% | 91.9% | **98.0%** |
| Troca Facial | DeepFaceLab | 250 | 94.4% | 89.3% | 83.3% | **96.4%** |
| Edição / T2I | WhichFaceIsReal | 250 | 60.7% | 90.9% | **93.1%** | 86.8% |
| Edição / T2I | MidJourney | 248 | 97.0% | 99.6% | 95.8% | **99.7%** |
| Edição / T2I | e4e | 250 | 92.5% | **97.9%** | 96.3% | 97.3% |
| Edição / T2I | StyleCLIP | 250 | 64.4% | **89.2%** | 86.3% | 83.6% |
| Misto | Celeb-DF (misto) | 500 | 92.3% | 94.7% | 95.3% | **96.7%** |
| Avatar | HeyGen | 250 | 89.7% | **96.8%** | 83.5% | 95.2% |
| **TOTAL GERAL (Revocação)** | **24 técnicas** | **6.147** | 87,9% | 87,3% | 86,5% | **90,3%** |
| *Especificidade (em Reais)* | *Reais* | *4.999* | 91,2% | 88,4% | 89,1% | 40,4% |

> **Destaque Metodológico:** O Ensemble Top-3 calibrado na validação do MFFI atinge **90,3%** de revocação global (+2,4 p.p. sobre o melhor modelo individual), resgatando expressivamente técnicas onde o DINOv3 falhava isoladamente, como WhichFaceIsReal (de 60,7% para 86,8%) e StyleCLIP (de 64,4% para 83,6%).
