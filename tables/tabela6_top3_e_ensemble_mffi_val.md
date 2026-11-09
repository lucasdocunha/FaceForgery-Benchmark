# Tabela 6: Revocação (%) por Técnica no Protocolo 2 do DF-40 (650.549 Imagens)

> **Protocolo 2 Oficial:** Avaliação exaustiva em **650.549 imagens falsas** distribuídas em 24 técnicas de manipulação facial.
> Modelos individuais e **Ensemble Top-3** (fusão por soft voting / média de probabilidades dos 3 melhores modelos: DINOv3 RGB+Freq, Xception SRM-only e Xception RGB+SRM) com limiar ótimo calibrado na validação do MFFI.
> Médias e desvios-padrão reportados ao longo de 3 sementes (42, 123 e 2024). Em negrito, o maior valor por técnica.

| Paradigma | Técnica | N (Amostras) | DINOv3 (RGB+Freq) | Xception (SRM-only) | Xception (RGB+SRM) | Ensemble Top-3 (Val. MFFI) |
| :--- | :--- | :---: | :---: | :---: | :---: | :---: |
| Difusão | SD 2.1 | 53.474 | 98,6% | 99,6% | 99,5% | **100,0%** |
| Difusão | PixArt-alpha | 31.885 | 98,5% | 98,7% | **99,9%** | **99,9%** |
| Difusão | SiT | 31.885 | 93,1% | 90,4% | 92,9% | **93,9%** |
| Difusão | DiT | 31.885 | 92,4% | 87,8% | 90,5% | **92,8%** |
| Difusão | RDDM | 17.738 | 99,0% | 94,3% | 97,0% | **100,0%** |
| Difusão | CollabDiff | 1.000 | **76,4%** | 74,8% | 67,8% | 71,2% |
| Difusão | DDIM | 31.885 | 61,5% | **75,8%** | 57,7% | 58,8% |
| Síntese GAN | StyleGAN3 | 31.885 | 98,9% | 99,0% | 99,9% | **100,0%** |
| Síntese GAN | StyleGAN2 | 31.885 | 97,7% | 94,3% | **99,4%** | 98,9% |
| Síntese GAN | StyleGAN-XL | 31.885 | 98,8% | 96,3% | 94,3% | **99,2%** |
| Síntese GAN | VQGAN | 31.885 | **95,5%** | 70,4% | 67,5% | 94,7% |
| Síntese GAN | StarGAN | 1.984 | **74,2%** | 42,7% | 53,8% | 61,2% |
| Síntese GAN | StarGAN v2 | 2.000 | 38,4% | **46,1%** | 23,0% | 26,0% |
| Troca Facial | UniFace | 24.346 | 94,0% | 96,0% | 95,7% | **97,1%** |
| Troca Facial | FaceSwap | 24.634 | 95,1% | 91,1% | 94,7% | **96,7%** |
| Troca Facial | BlendFace | 24.790 | 96,0% | 92,1% | 92,5% | **99,1%** |
| Troca Facial | MobileSwap | 47.522 | 97,1% | 93,1% | 94,5% | **98,0%** |
| Troca Facial | DeepFaceLab | 3.122 | 95,1% | 93,0% | 83,5% | **96,4%** |
| Edição / T2I | WhichFaceIsReal | 1.001 | 60,9% | **92,5%** | 91,4% | 86,8% |
| Edição / T2I | MidJourney | 635 | 97,4% | **99,7%** | 95,2% | **99,7%** |
| Edição / T2I | e4e | 60.123 | 93,7% | **98,7%** | 97,4% | 97,3% |
| Edição / T2I | StyleCLIP | 2.824 | 67,2% | **90,3%** | 87,8% | 83,6% |
| Misto | Celeb-DF (misto) | 128.671 | 92,6% | 94,7% | 94,4% | **96,7%** |
| Avatar | HeyGen | 1.605 | 93,2% | **96,5%** | 85,2% | 95,2% |
| **TOTAL GERAL** | **24 técnicas** | **650.549** | 93,4% | 92,6% | 92,1% | **95,1%** |
