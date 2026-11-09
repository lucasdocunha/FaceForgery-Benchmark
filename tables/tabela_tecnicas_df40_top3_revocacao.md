# Tabela: Desempenho por Técnica no DF-40 (Top-3 em Revocação)

> **Protocolo 2 Oficial:** Avaliação exaustiva em **650.549 imagens falsas** distribuídas em 24 técnicas de manipulação facial.
> Valores reportados: Revocação (%) com limiar ótimo calibrado na validação do MFFI (Média $\pm$ Desvio-Padrão em 3 sementes: 42, 123 e 2024).

| Paradigma | Técnica | N (Amostras) | DINOv3 (RGB+Freq) | Xception (SRM-only) | Xception (RGB+SRM) | DINOv3 (RGB+SRM)* |
| :--- | :--- | :---: | :---: | :---: | :---: | :---: |
| Difusão | SD 2.1 | 53,474 | 98.6 ± 1.2% | **99.6 ± 0.1%** | 99.5 ± 0.1% | 99.8 ± 0.1% |
| Difusão | PixArt-alpha | 31,885 | 98.5 ± 0.8% | 98.7 ± 0.7% | **99.9 ± 0.0%** | 99.6 ± 0.1% |
| Difusão | SiT | 31,885 | **93.1 ± 0.9%** | 90.4 ± 3.0% | 92.9 ± 1.0% | 85.5 ± 2.4% |
| Difusão | DiT | 31,885 | **92.4 ± 1.1%** | 87.8 ± 3.4% | 90.5 ± 1.2% | 75.7 ± 5.0% |
| Difusão | RDDM | 17,738 | **99.0 ± 1.7%** | 94.3 ± 3.1% | 97.0 ± 1.4% | 44.2 ± 17.7% |
| Difusão | CollabDiff | 1,000 | **76.4 ± 3.7%** | 74.8 ± 1.6% | 67.8 ± 7.2% | 37.0 ± 1.6% |
| Difusão | DDIM | 31,885 | 61.5 ± 7.8% | **75.8 ± 2.0%** | 57.7 ± 0.3% | 21.2 ± 7.8% |
| Síntese GAN | StyleGAN3 | 31,885 | 98.9 ± 0.3% | 99.0 ± 0.6% | **99.9 ± 0.1%** | 100.0 ± 0.0% |
| Síntese GAN | StyleGAN2 | 31,885 | 97.7 ± 0.7% | 94.3 ± 1.3% | **99.4 ± 0.2%** | 100.0 ± 0.0% |
| Síntese GAN | StyleGAN-XL | 31,885 | **98.8 ± 0.5%** | 96.3 ± 1.8% | 94.3 ± 1.7% | 98.9 ± 0.9% |
| Síntese GAN | VQGAN | 31,885 | **95.5 ± 0.5%** | 70.4 ± 4.0% | 67.5 ± 2.3% | 65.0 ± 4.6% |
| Síntese GAN | StarGAN | 1,984 | **74.2 ± 6.4%** | 42.7 ± 4.1% | 53.8 ± 5.1% | 71.1 ± 4.0% |
| Síntese GAN | StarGAN v2 | 2,000 | 38.4 ± 5.0% | **46.1 ± 4.9%** | 23.0 ± 2.7% | 16.4 ± 7.6% |
| Troca Facial | UniFace | 24,346 | 94.0 ± 0.8% | **96.0 ± 0.9%** | 95.7 ± 2.0% | 97.3 ± 1.2% |
| Troca Facial | FaceSwap | 24,634 | **95.1 ± 0.7%** | 91.1 ± 2.5% | 94.7 ± 1.5% | 90.0 ± 0.2% |
| Troca Facial | BlendFace | 24,790 | **96.0 ± 0.6%** | 92.1 ± 2.0% | 92.5 ± 2.3% | 94.6 ± 0.3% |
| Troca Facial | MobileSwap | 47,522 | **97.1 ± 0.8%** | 93.1 ± 1.4% | 94.5 ± 1.3% | 92.8 ± 1.5% |
| Troca Facial | DeepFaceLab | 3,122 | **95.1 ± 1.8%** | 93.0 ± 1.6% | 83.5 ± 4.8% | 84.7 ± 3.9% |
| Edição / T2I | WhichFaceIsReal | 1,001 | 60.9 ± 7.6% | **92.5 ± 1.7%** | 91.4 ± 2.5% | 99.0 ± 0.3% |
| Edição / T2I | MidJourney | 635 | 97.4 ± 2.1% | **99.7 ± 0.1%** | 95.2 ± 1.3% | 98.3 ± 1.6% |
| Edição / T2I | e4e | 60,123 | 93.7 ± 1.9% | **98.7 ± 0.2%** | 97.4 ± 0.6% | 94.0 ± 0.7% |
| Edição / T2I | StyleCLIP | 2,824 | 67.2 ± 6.2% | **90.3 ± 0.6%** | 87.8 ± 2.6% | 92.9 ± 2.3% |
| Misto | Celeb-DF (misto) | 128,671 | 92.6 ± 0.6% | **94.7 ± 1.9%** | 94.4 ± 1.9% | 90.6 ± 1.2% |
| Avatar | HeyGen | 1,605 | 93.2 ± 3.7% | **96.5 ± 0.6%** | 85.2 ± 4.8% | 30.5 ± 14.2% |
| **TOTAL GERAL** | **24 técnicas** | **650.549** | **93,4 ± 0,3%** | **92,6 ± 1,4%** | **92,1 ± 0,8%** | 86,6 ± 0,1% |

\* *DINOv3 (RGB+SRM) incluído como referência direta à Tabela 6 original do artigo.*
