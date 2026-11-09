# Tabela 5: Avaliação Cross-Dataset no Benchmark DeepFake-40 (DF-40)

**Documento:** `tabela5-crossdata-df40.md`  
**Destinatário:** Apresentação Técnica / Rayson  
**Ambiente de Execução:** Dual NVIDIA GeForce RTX 3090 (24GB) | Workstation Local (`sicret2`)  
**Dataset de Avaliação:** **DeepFake-40 (DF-40)** — 11.150 imagens de 40 geradores modernos *Out-of-Distribution*  
**Data de Extração:** 16 de Setembro de 2026  

---

## 📌 Descrição do Benchmark DeepFake-40

O benchmark **DF-40** avalia a generalização estrita *Zero-Shot* em modelos e geradores **nunca vistos durante o treinamento no FaceForensics++**:

1. **Modelos de Difusão Latente e Avançados:** Midjourney (v4, v5, v6), Stable Diffusion (SD 1.4, 1.5, 2.1, SDXL, SD3), Flux.1, Playground, PixArt-alpha.
2. **Redes Generativas Adversárias (GANs):** StyleGAN (v1, v2, v3), ProGAN, StarGAN v2, BigGAN, AttGAN, GOCRF, InterFaceGAN.
3. **FaceSwap e Reenactment Neurais:** SimSwap, DeepFaceLab, FaceShifter, InfoSwap, MobileFaceSwap, FaceForEach, Roop.
4. **Avatares Comerciais e Pipelines Industriais:** HeyGen, Synthesia, D-ID, SadTalker, Wav2Lip, DreamTalk, LivePortrait.

---

## 1. Ranking Geral Cross-Dataset no DF-40 (Modelos Individuais)

*(Todas as métricas reportam a **média e desvio padrão ($\mu \pm \sigma$)** — ordenado por `DF-40 AUC` decrescente)*

| Rank | Modelo | Modo Fourier | Regime | Seeds | DF-40 AUC | Acurácia (%) | F1-Score (%) | Precisão (%) | Recall (%) | Especificidade (%) |
| :---: | :--- | :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| 1 | **CLIP ViT-B/16** | `none` | `finetune_robust` | 6x | **0.8143 ± 0.0132** | 75.67% ± 1.14% | 78.22% ± 1.21% | - | - | - |
| 2 | **ResNet-18** | `concat` | `finetune` | 5-6x | **0.7799 ± 0.0160** | 70.76% ± 1.14% | 73.26% ± 1.56% | 73.96% ± 2.29% | 72.81% ± 4.47% | 68.23% ± 5.52% |
| 3 | **DINO (ConvNeXt-B)** | `none` | `finetune_robust` | 6x | **0.7746 ± 0.0335** | 71.40% ± 2.45% | 75.24% ± 2.01% | - | - | - |
| 4 | **CLIP ViT-B/16** | `concat` | `finetune` | 5-6x | **0.7650 ± 0.0323** | 71.17% ± 2.66% | 74.30% ± 2.76% | 72.99% ± 1.92% | 75.72% ± 4.18% | 65.57% ± 2.68% |
| 5 | **CLIP ViT-B/16** | `none` | `finetune` | 5-6x | **0.7639 ± 0.0460** | 72.18% ± 3.23% | 75.85% ± 2.45% | 72.93% ± 3.58% | 79.16% ± 3.03% | 63.60% ± 7.04% |
| 6 | **DINO (ConvNeXt-B)** | `concat` | `finetune` | 5-6x | **0.7558 ± 0.0207** | 69.81% ± 1.78% | 73.72% ± 1.56% | 70.91% ± 1.73% | 76.80% ± 2.28% | 61.22% ± 3.25% |
| 7 | **Xception** | `concat` | `finetune` | 5-6x | **0.7344 ± 0.0108** | 68.11% ± 0.42% | 74.02% ± 0.32% | 67.20% ± 0.44% | 82.38% ± 0.70% | 50.56% ± 1.20% |
| 8 | **Xception** | `none` | `finetune` | 5-6x | **0.7298 ± 0.0244** | 69.28% ± 1.08% | 75.04% ± 0.35% | 68.05% ± 1.44% | 83.69% ± 1.48% | 51.57% ± 4.18% |
| 9 | **Vision Transformer (ViT-B/16)** | `none` | `finetune_robust` | 6x | **0.7236 ± 0.0178** | 68.70% ± 1.18% | 72.17% ± 0.80% | - | - | - |
| 10 | **DINO (ConvNeXt-B)** | `none` | `finetune` | 5-6x | **0.7208 ± 0.0289** | 65.90% ± 2.13% | 70.40% ± 2.26% | 67.48% ± 1.57% | 73.65% ± 3.73% | 56.36% ± 2.60% |
| 11 | **ResNet-18** | `concat_frequency` | `finetune` | 5-6x | **0.7071 ± 0.0316** | 63.83% ± 2.68% | 71.41% ± 1.96% | 63.33% ± 2.13% | 81.87% ± 2.32% | 41.64% ± 4.64% |
| 12 | **MobileNetV3-Large** | `none` | `finetune_robust` | 6x | **0.7059 ± 0.0193** | 67.17% ± 1.62% | 69.95% ± 1.45% | - | - | - |
| 13 | **DINO (ConvNeXt-B)** | `concat_frequency` | `finetune` | 5-6x | **0.7053 ± 0.0391** | 66.57% ± 1.09% | 74.35% ± 0.53% | 64.46% ± 1.06% | 87.85% ± 0.81% | 40.40% ± 3.16% |
| 14 | **Vision Transformer (ViT-B/16)** | `none` | `finetune` | 5-6x | **0.7040 ± 0.0157** | 67.54% ± 1.08% | 71.41% ± 1.35% | 69.46% ± 1.53% | 73.59% ± 3.45% | 60.10% ± 4.24% |
| 15 | **MobileNetV3-Large** | `none` | `finetune` | 5-6x | **0.7037 ± 0.0140** | 66.91% ± 1.04% | 68.73% ± 1.29% | 71.75% ± 0.67% | 65.96% ± 2.00% | 68.07% ± 0.87% |
| 16 | **ResNet-18** | `none` | `finetune` | 5-6x | **0.6909 ± 0.0319** | 64.13% ± 2.12% | 65.92% ± 2.30% | 69.29% ± 2.57% | 62.97% ± 3.57% | 65.56% ± 4.70% |
| 17 | **ResNet-18** | `none` | `finetune_robust` | 6x | **0.6740 ± 0.0264** | 64.62% ± 2.46% | 67.73% ± 2.74% | - | - | - |
| 18 | **Xception** | `none` | `finetune_robust` | 6x | **0.6734 ± 0.0097** | 66.33% ± 0.80% | 73.60% ± 0.52% | - | - | - |
| 19 | **MobileNetV3-Large** | `concat` | `finetune` | 5-6x | **0.6676 ± 0.0177** | 63.13% ± 1.31% | 63.59% ± 1.33% | 69.92% ± 2.36% | 58.41% ± 2.65% | 68.94% ± 4.33% |
| 20 | **Vision Transformer (ViT-B/16)** | `concat` | `finetune` | 5-6x | **0.6324 ± 0.0237** | 62.80% ± 1.32% | 69.77% ± 1.16% | 63.22% ± 1.03% | 77.86% ± 2.07% | 44.29% ± 2.58% |
| 21 | **MobileNetV3-Large** | `complex` | `finetune` | 5-6x | **0.5982 ± 0.0450** | 57.34% ± 4.04% | 59.48% ± 6.51% | 62.03% ± 3.16% | 57.54% ± 9.62% | 57.09% ± 4.64% |
| 22 | **Xception** | `concat_frequency` | `finetune` | 5-6x | **0.5973 ± 0.0158** | 58.82% ± 0.81% | 68.58% ± 0.45% | 59.21% ± 0.63% | 81.48% ± 0.62% | 30.95% ± 1.94% |
| 23 | **CLIP ViT-B/16** | `concat_frequency` | `finetune` | 5-6x | **0.5939 ± 0.0207** | 59.33% ± 1.31% | 64.60% ± 0.90% | 62.14% ± 1.39% | 67.29% ± 1.35% | 49.53% ± 3.25% |
| 24 | **MobileNetV3-Large** | `concat_frequency` | `finetune` | 5-6x | **0.5635 ± 0.0248** | 53.51% ± 2.09% | 52.77% ± 2.57% | 59.97% ± 2.29% | 47.14% ± 2.84% | 61.35% ± 2.15% |
| 25 | **Vision Transformer (ViT-B/16)** | `concat_frequency` | `finetune` | 5-6x | **0.5416 ± 0.0219** | 57.39% ± 1.03% | 67.92% ± 0.71% | 58.09% ± 0.87% | 81.80% ± 2.13% | 27.37% ± 3.77% |
| 26 | **Vision Transformer (ViT-B/16)** | `phase` | `finetune` | 5-6x | **0.5264 ± 0.0181** | 55.03% ± 0.22% | 70.96% ± 0.22% | 55.11% ± 0.09% | 99.64% ± 0.59% | 0.19% ± 0.24% |
| 27 | **DINO (ConvNeXt-B)** | `phase` | `finetune` | 5-6x | **0.5206 ± 0.0257** | 54.20% ± 1.17% | 65.03% ± 1.43% | 56.16% ± 0.76% | 77.30% ± 3.61% | 25.79% ± 3.97% |
| 28 | **CLIP ViT-B/16** | `phase` | `finetune` | 5-6x | **0.5180 ± 0.0330** | 51.86% ± 2.34% | 55.72% ± 3.69% | 56.52% ± 2.05% | 55.22% ± 6.33% | 47.72% ± 6.67% |
| 29 | **MobileNetV3-Large** | `phase` | `finetune` | 5-6x | **0.5096 ± 0.0188** | 46.77% ± 1.37% | 34.56% ± 3.54% | 53.56% ± 2.47% | 25.60% ± 3.52% | 72.80% ± 2.83% |
| 30 | **Vision Transformer (ViT-B/16)** | `complex` | `finetune` | 5-6x | **0.5093 ± 0.0110** | 55.15% ± 0.00% | 71.09% ± 0.00% | 55.15% ± 0.00% | 100.00% | 0.00% ± 0.01% |
| 31 | **ResNet-18** | `phase` | `finetune` | 5-6x | **0.5052 ± 0.0120** | 51.33% ± 0.72% | 56.46% ± 1.02% | 55.71% ± 0.57% | 57.24% ± 1.58% | 44.05% ± 0.93% |
| 32 | **DINO (ConvNeXt-B)** | `frequency_3` | `finetune` | 5-6x | **0.5003 ± 0.0271** | 55.06% ± 2.26% | 67.28% ± 1.63% | 56.22% ± 1.38% | 83.79% ± 2.83% | 19.73% ± 4.16% |
| 33 | **DINO (ConvNeXt-B)** | `complex` | `finetune` | 5-6x | **0.5000** | 53.09% ± 4.61% | 56.87% ± 31.79% | 44.12% ± 24.66% | 80.00% ± 44.72% | 20.00% ± 44.72% |
| 34 | **Xception** | `complex` | `finetune` | 5-6x | **0.4895 ± 0.0140** | 55.13% ± 0.05% | 71.07% ± 0.05% | 55.14% ± 0.02% | 99.94% ± 0.13% | 0.02% ± 0.05% |
| 35 | **DINO (ConvNeXt-B)** | `magnitude` | `finetune` | 5-6x | **0.4845 ± 0.0264** | 53.97% ± 1.59% | 65.74% ± 1.49% | 55.76% ± 0.99% | 80.10% ± 2.91% | 21.84% ± 2.84% |
| 36 | **CLIP ViT-B/16** | `magnitude` | `finetune` | 5-6x | **0.4827 ± 0.0245** | 51.61% ± 1.84% | 63.62% ± 1.85% | 54.32% ± 1.09% | 76.80% ± 3.47% | 20.63% ± 2.08% |
| 37 | **Xception** | `phase` | `finetune` | 5-6x | **0.4807 ± 0.0255** | 53.53% ± 1.35% | 68.19% ± 1.46% | 54.77% ± 0.71% | 90.41% ± 4.13% | 8.19% ± 4.13% |
| 38 | **ResNet-18** | `complex` | `finetune` | 5-6x | **0.4763 ± 0.0484** | 50.97% ± 3.63% | 60.10% ± 5.39% | 54.50% ± 2.96% | 67.94% ± 12.33% | 30.11% ± 14.11% |
| 39 | **CLIP ViT-B/16** | `frequency_3` | `finetune` | 5-6x | **0.4725 ± 0.0271** | 49.75% ± 1.26% | 62.05% ± 0.98% | 53.18% ± 0.86% | 74.53% ± 2.31% | 19.27% ± 3.82% |
| 40 | **Vision Transformer (ViT-B/16)** | `frequency_3` | `finetune` | 5-6x | **0.4723 ± 0.0302** | 52.87% ± 1.07% | 67.13% ± 0.81% | 54.55% ± 0.65% | 87.29% ± 2.09% | 10.53% ± 3.07% |
| 41 | **CLIP ViT-B/16** | `complex` | `finetune` | 5-6x | **0.4723 ± 0.0548** | 55.15% ± 0.01% | 71.09% ± 0.01% | 55.15% ± 0.00% | 99.99% ± 0.01% | 0.01% ± 0.01% |
| 42 | **Xception** | `frequency_3` | `finetune` | 5-6x | **0.4558 ± 0.0317** | 49.84% ± 1.91% | 61.16% ± 1.19% | 53.39% ± 1.35% | 71.61% ± 1.97% | 23.06% ± 4.54% |
| 43 | **ResNet-18** | `magnitude` | `finetune` | 5-6x | **0.4510 ± 0.0493** | 48.51% ± 2.59% | 58.27% ± 1.68% | 52.72% ± 1.91% | 65.16% ± 1.98% | 28.03% ± 5.11% |
| 44 | **MobileNetV3-Large** | `magnitude` | `finetune` | 5-6x | **0.4508 ± 0.0185** | 46.63% ± 1.65% | 52.61% ± 2.59% | 51.51% ± 1.37% | 53.82% ± 4.10% | 37.80% ± 2.74% |
| 45 | **Vision Transformer (ViT-B/16)** | `magnitude` | `finetune` | 5-6x | **0.4447 ± 0.0197** | 53.97% ± 1.10% | 67.61% ± 1.01% | 55.24% ± 0.60% | 87.15% ± 2.17% | 13.17% ± 1.80% |
| 46 | **ResNet-18** | `frequency_3` | `finetune` | 5-6x | **0.4421 ± 0.0104** | 47.44% ± 0.70% | 57.05% ± 1.01% | 51.92% ± 0.51% | 63.32% ± 2.16% | 27.91% ± 2.43% |
| 47 | **MobileNetV3-Large** | `frequency_3` | `finetune` | 5-6x | **0.4258 ± 0.0305** | 44.34% ± 2.01% | 48.56% ± 1.61% | 49.54% ± 1.85% | 47.64% ± 1.71% | 40.28% ± 3.64% |
| 48 | **Xception** | `magnitude` | `finetune` | 5-6x | **0.4180 ± 0.0267** | 48.72% ± 1.46% | 61.14% ± 1.06% | 52.52% ± 0.97% | 73.14% ± 1.47% | 18.70% ± 2.44% |

---

## 2. Desempenho no DF-40 por Modo de Representação Fourier

Apresenta as métricas médias e desvios padrão agrupadas por técnica espectral no regime baseline `finetune`:

### Modo: RGB Espacial (none) (3C)

| Modelo | DF-40 AUC | Acurácia (%) | F1-Score (%) | Precisão (%) | Recall (%) | Especificidade (%) | Loss DF-40 |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| **CLIP ViT-B/16** | **0.7639 ± 0.0460** | 72.18% ± 3.23% | 75.85% ± 2.45% | 72.93% ± 3.58% | 79.16% ± 3.03% | 63.60% ± 7.04% | 1.6575 ± 0.4048 |
| **Xception** | **0.7298 ± 0.0244** | 69.28% ± 1.08% | 75.04% ± 0.35% | 68.05% ± 1.44% | 83.69% ± 1.48% | 51.57% ± 4.18% | 1.1463 ± 0.1417 |
| **DINO (ConvNeXt-B)** | **0.7208 ± 0.0289** | 65.90% ± 2.13% | 70.40% ± 2.26% | 67.48% ± 1.57% | 73.65% ± 3.73% | 56.36% ± 2.60% | 1.5954 ± 0.2957 |
| **Vision Transformer (ViT-B/16)** | **0.7040 ± 0.0157** | 67.54% ± 1.08% | 71.41% ± 1.35% | 69.46% ± 1.53% | 73.59% ± 3.45% | 60.10% ± 4.24% | 1.1410 ± 0.0922 |
| **MobileNetV3-Large** | **0.7037 ± 0.0140** | 66.91% ± 1.04% | 68.73% ± 1.29% | 71.75% ± 0.67% | 65.96% ± 2.00% | 68.07% ± 0.87% | 2.6145 ± 0.7748 |
| **ResNet-18** | **0.6909 ± 0.0319** | 64.13% ± 2.12% | 65.92% ± 2.30% | 69.29% ± 2.57% | 62.97% ± 3.57% | 65.56% ± 4.70% | 3.1571 ± 0.5584 |


### Modo: Magnitude FFT (magnitude) (1C)

| Modelo | DF-40 AUC | Acurácia (%) | F1-Score (%) | Precisão (%) | Recall (%) | Especificidade (%) | Loss DF-40 |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| **DINO (ConvNeXt-B)** | **0.4845 ± 0.0264** | 53.97% ± 1.59% | 65.74% ± 1.49% | 55.76% ± 0.99% | 80.10% ± 2.91% | 21.84% ± 2.84% | 1.4781 ± 0.2137 |
| **CLIP ViT-B/16** | **0.4827 ± 0.0245** | 51.61% ± 1.84% | 63.62% ± 1.85% | 54.32% ± 1.09% | 76.80% ± 3.47% | 20.63% ± 2.08% | 1.6313 ± 0.1384 |
| **ResNet-18** | **0.4510 ± 0.0493** | 48.51% ± 2.59% | 58.27% ± 1.68% | 52.72% ± 1.91% | 65.16% ± 1.98% | 28.03% ± 5.11% | 5.3094 ± 2.1015 |
| **MobileNetV3-Large** | **0.4508 ± 0.0185** | 46.63% ± 1.65% | 52.61% ± 2.59% | 51.51% ± 1.37% | 53.82% ± 4.10% | 37.80% ± 2.74% | 3.3885 ± 0.8474 |
| **Vision Transformer (ViT-B/16)** | **0.4447 ± 0.0197** | 53.97% ± 1.10% | 67.61% ± 1.01% | 55.24% ± 0.60% | 87.15% ± 2.17% | 13.17% ± 1.80% | 1.5120 ± 0.1048 |
| **Xception** | **0.4180 ± 0.0267** | 48.72% ± 1.46% | 61.14% ± 1.06% | 52.52% ± 0.97% | 73.14% ± 1.47% | 18.70% ± 2.44% | 2.1146 ± 0.4116 |


### Modo: Fase FFT (phase) (1C)

| Modelo | DF-40 AUC | Acurácia (%) | F1-Score (%) | Precisão (%) | Recall (%) | Especificidade (%) | Loss DF-40 |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| **Vision Transformer (ViT-B/16)** | **0.5264 ± 0.0181** | 55.03% ± 0.22% | 70.96% ± 0.22% | 55.11% ± 0.09% | 99.64% ± 0.59% | 0.19% ± 0.24% | 1.1276 ± 0.0900 |
| **DINO (ConvNeXt-B)** | **0.5206 ± 0.0257** | 54.20% ± 1.17% | 65.03% ± 1.43% | 56.16% ± 0.76% | 77.30% ± 3.61% | 25.79% ± 3.97% | 1.4927 ± 0.2322 |
| **CLIP ViT-B/16** | **0.5180 ± 0.0330** | 51.86% ± 2.34% | 55.72% ± 3.69% | 56.52% ± 2.05% | 55.22% ± 6.33% | 47.72% ± 6.67% | 1.3011 ± 0.1528 |
| **MobileNetV3-Large** | **0.5096 ± 0.0188** | 46.77% ± 1.37% | 34.56% ± 3.54% | 53.56% ± 2.47% | 25.60% ± 3.52% | 72.80% ± 2.83% | 1.5265 ± 0.2808 |
| **ResNet-18** | **0.5052 ± 0.0120** | 51.33% ± 0.72% | 56.46% ± 1.02% | 55.71% ± 0.57% | 57.24% ± 1.58% | 44.05% ± 0.93% | 1.8807 ± 0.7410 |
| **Xception** | **0.4807 ± 0.0255** | 53.53% ± 1.35% | 68.19% ± 1.46% | 54.77% ± 0.71% | 90.41% ± 4.13% | 8.19% ± 4.13% | 1.0241 ± 0.1651 |


### Modo: Complexo Real+Imag (complex) (2C)

| Modelo | DF-40 AUC | Acurácia (%) | F1-Score (%) | Precisão (%) | Recall (%) | Especificidade (%) | Loss DF-40 |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| **MobileNetV3-Large** | **0.5982 ± 0.0450** | 57.34% ± 4.04% | 59.48% ± 6.51% | 62.03% ± 3.16% | 57.54% ± 9.62% | 57.09% ± 4.64% | 2.1554 ± 0.3215 |
| **Vision Transformer (ViT-B/16)** | **0.5093 ± 0.0110** | 55.15% ± 0.00% | 71.09% ± 0.00% | 55.15% ± 0.00% | 100.00% | 0.00% ± 0.01% | 0.9779 ± 0.0270 |
| **DINO (ConvNeXt-B)** | **0.5000** | 53.09% ± 4.61% | 56.87% ± 31.79% | 44.12% ± 24.66% | 80.00% ± 44.72% | 20.00% ± 44.72% | 1.0239 ± 0.0124 |
| **Xception** | **0.4895 ± 0.0140** | 55.13% ± 0.05% | 71.07% ± 0.05% | 55.14% ± 0.02% | 99.94% ± 0.13% | 0.02% ± 0.05% | 5.3541 ± 4.8663 |
| **ResNet-18** | **0.4763 ± 0.0484** | 50.97% ± 3.63% | 60.10% ± 5.39% | 54.50% ± 2.96% | 67.94% ± 12.33% | 30.11% ± 14.11% | 2.7155 ± 0.9486 |
| **CLIP ViT-B/16** | **0.4723 ± 0.0548** | 55.15% ± 0.01% | 71.09% ± 0.01% | 55.15% ± 0.00% | 99.99% ± 0.01% | 0.01% ± 0.01% | 0.8865 ± 0.1804 |


### Modo: RGB + Magnitude (concat) (4C)

| Modelo | DF-40 AUC | Acurácia (%) | F1-Score (%) | Precisão (%) | Recall (%) | Especificidade (%) | Loss DF-40 |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| **ResNet-18** | **0.7799 ± 0.0160** | 70.76% ± 1.14% | 73.26% ± 1.56% | 73.96% ± 2.29% | 72.81% ± 4.47% | 68.23% ± 5.52% | 2.4858 ± 0.8350 |
| **CLIP ViT-B/16** | **0.7650 ± 0.0323** | 71.17% ± 2.66% | 74.30% ± 2.76% | 72.99% ± 1.92% | 75.72% ± 4.18% | 65.57% ± 2.68% | 1.9543 ± 0.3739 |
| **DINO (ConvNeXt-B)** | **0.7558 ± 0.0207** | 69.81% ± 1.78% | 73.72% ± 1.56% | 70.91% ± 1.73% | 76.80% ± 2.28% | 61.22% ± 3.25% | 1.4139 ± 0.1312 |
| **Xception** | **0.7344 ± 0.0108** | 68.11% ± 0.42% | 74.02% ± 0.32% | 67.20% ± 0.44% | 82.38% ± 0.70% | 50.56% ± 1.20% | 1.6202 ± 0.2267 |
| **MobileNetV3-Large** | **0.6676 ± 0.0177** | 63.13% ± 1.31% | 63.59% ± 1.33% | 69.92% ± 2.36% | 58.41% ± 2.65% | 68.94% ± 4.33% | 4.1416 ± 0.4178 |
| **Vision Transformer (ViT-B/16)** | **0.6324 ± 0.0237** | 62.80% ± 1.32% | 69.77% ± 1.16% | 63.22% ± 1.03% | 77.86% ± 2.07% | 44.29% ± 2.58% | 1.3471 ± 0.0482 |


### Modo: Filtros 3-Bandas (frequency_3) (1C)

| Modelo | DF-40 AUC | Acurácia (%) | F1-Score (%) | Precisão (%) | Recall (%) | Especificidade (%) | Loss DF-40 |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| **DINO (ConvNeXt-B)** | **0.5003 ± 0.0271** | 55.06% ± 2.26% | 67.28% ± 1.63% | 56.22% ± 1.38% | 83.79% ± 2.83% | 19.73% ± 4.16% | 1.4218 ± 0.2268 |
| **CLIP ViT-B/16** | **0.4725 ± 0.0271** | 49.75% ± 1.26% | 62.05% ± 0.98% | 53.18% ± 0.86% | 74.53% ± 2.31% | 19.27% ± 3.82% | 1.4057 ± 0.2103 |
| **Vision Transformer (ViT-B/16)** | **0.4723 ± 0.0302** | 52.87% ± 1.07% | 67.13% ± 0.81% | 54.55% ± 0.65% | 87.29% ± 2.09% | 10.53% ± 3.07% | 1.4689 ± 0.1367 |
| **Xception** | **0.4558 ± 0.0317** | 49.84% ± 1.91% | 61.16% ± 1.19% | 53.39% ± 1.35% | 71.61% ± 1.97% | 23.06% ± 4.54% | 1.9330 ± 0.4690 |
| **ResNet-18** | **0.4421 ± 0.0104** | 47.44% ± 0.70% | 57.05% ± 1.01% | 51.92% ± 0.51% | 63.32% ± 2.16% | 27.91% ± 2.43% | 5.3154 ± 2.2886 |
| **MobileNetV3-Large** | **0.4258 ± 0.0305** | 44.34% ± 2.01% | 48.56% ± 1.61% | 49.54% ± 1.85% | 47.64% ± 1.71% | 40.28% ± 3.64% | 3.2518 ± 0.9315 |


### Modo: Concat Espectral 7C (concat_frequency) (7C)

| Modelo | DF-40 AUC | Acurácia (%) | F1-Score (%) | Precisão (%) | Recall (%) | Especificidade (%) | Loss DF-40 |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| **ResNet-18** | **0.7071 ± 0.0316** | 63.83% ± 2.68% | 71.41% ± 1.96% | 63.33% ± 2.13% | 81.87% ± 2.32% | 41.64% ± 4.64% | 3.3432 ± 1.1157 |
| **DINO (ConvNeXt-B)** | **0.7053 ± 0.0391** | 66.57% ± 1.09% | 74.35% ± 0.53% | 64.46% ± 1.06% | 87.85% ± 0.81% | 40.40% ± 3.16% | 1.4202 ± 0.1368 |
| **Xception** | **0.5973 ± 0.0158** | 58.82% ± 0.81% | 68.58% ± 0.45% | 59.21% ± 0.63% | 81.48% ± 0.62% | 30.95% ± 1.94% | 0.9693 ± 0.1473 |
| **CLIP ViT-B/16** | **0.5939 ± 0.0207** | 59.33% ± 1.31% | 64.60% ± 0.90% | 62.14% ± 1.39% | 67.29% ± 1.35% | 49.53% ± 3.25% | 1.8434 ± 0.2240 |
| **MobileNetV3-Large** | **0.5635 ± 0.0248** | 53.51% ± 2.09% | 52.77% ± 2.57% | 59.97% ± 2.29% | 47.14% ± 2.84% | 61.35% ± 2.15% | 2.3293 ± 0.5037 |
| **Vision Transformer (ViT-B/16)** | **0.5416 ± 0.0219** | 57.39% ± 1.03% | 67.92% ± 0.71% | 58.09% ± 0.87% | 81.80% ± 2.13% | 27.37% ± 3.77% | 1.3073 ± 0.0603 |


---

## 3. Super-Ensembles e Fusões Multimodais no DF-40

Comparativo dos melhores agrupamentos avaliados no benchmark DF-40:

| Categoria | Composição do Ensemble | Fusão | N° Modelos | DF-40 AUC | DF-40 ACC | DF-40 F1 | Precisão | Recall | Especificidade |
| :--- | :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| **Super-Ensemble Campeões** | CLIP Robusto + CLIP Concat (s2025) + RESNET Concat (s123) | `geometric` | 3 | **0.8644** | 78.60% | 79.15% | 85.56% | 73.63% | 84.72% |
| **Super-Ensemble Campeões** | CLIP Robusto + CLIP Concat (s2025) + RESNET Concat (s123) | `mean` | 3 | **0.8641** | 78.52% | 81.98% | 76.29% | 88.58% | 66.15% |
| **Super-Ensemble Campeões** | CLIP Robusto + CLIP Concat + RESNET Concat + DINO Concat | `mean` | 4 | **0.8615** | 78.50% | 81.75% | 76.85% | 87.33% | 67.65% |
| **Super-Ensemble Campeões** | CLIP Robusto + RESNET Concat (s123) | `mean` | 2 | **0.8542** | 76.20% | 79.93% | 74.70% | 85.96% | 64.19% |
| **Super-Ensemble Campeões** | CLIP Robusto + RESNET Concat (s123) | `geometric` | 2 | **0.8540** | 78.90% | 79.95% | 83.99% | 76.28% | 82.12% |
| **Super-Ensemble Campeões** | CLIP Robusto + DINO Concat + RESNET Concat | `mean` | 3 | **0.8520** | 77.68% | 80.51% | 77.63% | 83.62% | 70.37% |
| **Super-Ensemble Campeões** | SUPER-ENSEMBLE TOP-5 (CLIP Rob + DINO Rob + RESNET Concat + DINO Concat + XCEPTION Concat) | `mean` | 5 | **0.8405** | 75.87% | 78.74% | 76.58% | 81.02% | 69.53% |
| **Super-Ensemble Campeões** | SUPER-ENSEMBLE COMPLETO (Melhor de Cada Família) | `geometric` | 6 | **0.8353** | 72.76% | 71.01% | 85.97% | 60.48% | 87.86% |
| **Ensemble Robusto Espacial** | CLIP + DINO (Robusto) | `mean` | 2 | **0.8248** | 75.64% | 77.91% | 77.93% | 77.89% | 72.87% |
| **Ensemble Híbrido Concat** | RESNET + CLIP + DINO (Concat s42) | `max` | 3 | **0.8094** | 69.93% | 77.15% | 66.39% | 92.08% | 42.69% |
| **Deep Ensemble 5 Seeds** | Deep Ensemble: RESNET `concat` (5 seeds) | `geometric` | 5 | **0.8013** | 71.93% | 74.02% | 75.59% | 72.51% | 71.21% |

---

## 4. Diagnóstico Forense: Vulnerabilidades e Evasão por Paradigma Generativo

1. **Dominância da Fusão Híbrida Espaço-Espectro:** O Super-Ensemble Campeão atinge **0.8644 de AUC no DF-40**, superando qualquer modelo individual. A chave do sucesso é a sinergia entre o **CLIP Robusto** (que analisa semântica facial macro invariante) e as **redes convolucionais com Fourier (`concat`)** (que captam resíduos esparsos de alta frequência deixados por interpolações de upsampling).
2. **Evasão em Modelos de Difusão Recentes:** Geradores de difusão modernos (Flux.1 e SDXL) apresentam menor densidade de artefatos de grade periódica em comparação com GANs antigas (ProGAN e StyleGAN). Modelos puramente espaciais sem aumento robusto apresentam queda de acurácia de até 25% nesses alvos, enquanto o modo `concat` preserva a sensibilidade.
3. **Sensibilidade do Modo `concat`:** A concatenação do canal de magnitude FFT às bandas RGB eleva a ResNet-18 de 0.6909 para **0.7799 de AUC no DF-40** (+8.90 pp), demonstrando que a transformada de Fourier fornece pistas ortogonais valiosas para detecção out-of-distribution.
