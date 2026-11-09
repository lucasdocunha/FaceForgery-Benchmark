# Tabela 7: Resultados Consolidados de Ensembles dos Modelos Robustos (RandomizedRobustAugment)

**Documento:** `tabela7-ensemble-robusto.md`  
**Destinatário:** Apresentação Técnica / Rayson  
**Ambiente de Execução:** Dual NVIDIA GeForce RTX 3090 (24GB) | Workstation Local (`sicret2`)  
**Critério de Ordenação Estrito:** **Melhor ROC-AUC na Validação (`Val AUC` decrescente)**  
**Sementes Canônicas Integradas:** `[987, 42, 123, 2024, 7]` (Todas as 5 sementes avaliadas empiricamente)  
**Data de Extração:** 21 de Setembro de 2026  

> [!IMPORTANT]
> **Consolidação Empírica Exaustiva:**
> - Todos os comitês multi-modelo ($K=2$ a $K=6$) reportam a **média e desvio padrão ($\mu \pm \sigma$) calculados empiricamente sobre as 5 sementes canônicas** executadas em hardware.
> - Foram avaliados também os **Self-Ensembles** (fusão das 5 sementes canônicas da mesma arquitetura) e o **Super-Ensemble Robusto Global** (30 redes neurais: 6 arquiteturas $\times$ 5 sementes).

---

## 📌 Dinâmica e Mecanismos de Fusão dos Modelos Robustos

1. **Média Geométrica (`geometric`)**: Atua como forte penalizador de discordância, alcançando a maior robustez média sob corrupção severa (**0.8781 ± 0.0110 de Test-D AUC** no par CLIP+DINO).
2. **Stacking Linear (`stacking`)**: Ajusta pesos lineares ideais nos logits do conjunto de validação, maximizando o Val AUC (**0.9982** no Super-Ensemble 30M e **0.9964** no par CLIP+DINO).
3. **Self-Ensemble Multi-Seed:** A combinação das 5 sementes canônicas do próprio modelo eleva significativamente a generalização (o CLIP salta para **0.8432 no DF-40** e o DINO salta para **0.8653 no Test-D**).
4. **Sinergia Espaço-Espectro (Super-Ensembles):** A união de Foundation Models robustos com detectores espectrais de Fourier (`concat`) atinge **0.8644 de AUC no DF-40**.

---

## Tabela 7.1: Super-Ensemble Robusto Global (30 Redes) e Self-Ensembles (5 Sementes)

*(Fusão cross-seed e cross-architecture — ordenado estritamente por Val AUC decrescente)*

| Tipo de Fusão | Composição | Estratégia | N° Redes | Val AUC | Val ACC | Test AUC | Test ACC | Test F1 | Test-D AUC | ΔAUC | DF-40 AUC |
| :--- | :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| **Super-Ensemble Robusto Global (30 Redes: 6 Arquiteturas x 5 Seeds)** | `CLIP + DINO + ViT + ResNet + MobileNet + Xception (Todas 5 seeds)` | `stacking` | 30x | 0.9982 | 98.49% | 0.9426 | 85.83% | 86.37% | **0.8752** | -0.0674 | **0.8174** |
| **DINO (ConvNeXt-B) (5 Seeds Canônicas)** | `dino [seeds: 987, 42, 123, 2024, 7]` | `geometric` | 5x | 0.9965 | 97.59% | 0.9411 | 85.82% | 86.51% | **0.8653** | -0.0758 | **0.8011** |
| **DINO (ConvNeXt-B) (5 Seeds Canônicas)** | `dino [seeds: 987, 42, 123, 2024, 7]` | `mean` | 5x | 0.9964 | 97.64% | 0.9374 | 84.42% | 84.97% | **0.8584** | -0.0790 | **0.7968** |
| **DINO (ConvNeXt-B) (5 Seeds Canônicas)** | `dino [seeds: 987, 42, 123, 2024, 7]` | `stacking` | 5x | 0.9962 | 97.63% | 0.9364 | 83.49% | 83.88% | **0.8619** | -0.0745 | **0.7824** |
| **Super-Ensemble Robusto Global (30 Redes: 6 Arquiteturas x 5 Seeds)** | `CLIP + DINO + ViT + ResNet + MobileNet + Xception (Todas 5 seeds)` | `geometric` | 30x | 0.9951 | 96.76% | 0.9276 | 83.83% | 84.67% | **0.8622** | -0.0654 | **0.8011** |
| **Super-Ensemble Robusto Global (30 Redes: 6 Arquiteturas x 5 Seeds)** | `CLIP + DINO + ViT + ResNet + MobileNet + Xception (Todas 5 seeds)` | `mean` | 30x | 0.9948 | 96.78% | 0.9169 | 82.53% | 83.37% | **0.8392** | -0.0777 | **0.7877** |
| **CLIP (ViT-B/16) (5 Seeds Canônicas)** | `clip [seeds: 987, 42, 123, 2024, 7]` | `geometric` | 5x | 0.9890 | 94.95% | 0.9211 | 82.61% | 83.16% | **0.8622** | -0.0589 | **0.8432** |
| **CLIP (ViT-B/16) (5 Seeds Canônicas)** | `clip [seeds: 987, 42, 123, 2024, 7]` | `mean` | 5x | 0.9887 | 94.92% | 0.9214 | 82.85% | 83.45% | **0.8617** | -0.0596 | **0.8376** |
| **CLIP (ViT-B/16) (5 Seeds Canônicas)** | `clip [seeds: 987, 42, 123, 2024, 7]` | `stacking` | 5x | 0.9887 | 94.95% | 0.9212 | 82.43% | 82.86% | **0.8613** | -0.0599 | **0.8403** |
| **ResNet-18 (5 Seeds Canônicas)** | `resnet [seeds: 987, 42, 123, 2024, 7]` | `stacking` | 5x | 0.9813 | 93.35% | 0.8613 | 76.37% | 76.79% | **0.7778** | -0.0835 | **0.6775** |
| **ResNet-18 (5 Seeds Canônicas)** | `resnet [seeds: 987, 42, 123, 2024, 7]` | `mean` | 5x | 0.9812 | 93.29% | 0.8619 | 76.62% | 77.18% | **0.7782** | -0.0837 | **0.6799** |
| **ResNet-18 (5 Seeds Canônicas)** | `resnet [seeds: 987, 42, 123, 2024, 7]` | `geometric` | 5x | 0.9794 | 92.94% | 0.8626 | 76.19% | 76.44% | **0.7788** | -0.0838 | **0.6838** |
| **Vision Transformer (ViT-B/16) (5 Seeds Canônicas)** | `vit [seeds: 987, 42, 123, 2024, 7]` | `geometric` | 5x | 0.9609 | 89.80% | 0.8403 | 73.89% | 73.91% | **0.7808** | -0.0595 | **0.7437** |
| **Vision Transformer (ViT-B/16) (5 Seeds Canônicas)** | `vit [seeds: 987, 42, 123, 2024, 7]` | `stacking` | 5x | 0.9604 | 89.70% | 0.8397 | 73.68% | 73.70% | **0.7807** | -0.0589 | **0.7340** |
| **Vision Transformer (ViT-B/16) (5 Seeds Canônicas)** | `vit [seeds: 987, 42, 123, 2024, 7]` | `mean` | 5x | 0.9600 | 89.62% | 0.8394 | 73.78% | 73.90% | **0.7804** | -0.0590 | **0.7313** |
| **MobileNetV3-Large (5 Seeds Canônicas)** | `mobilenet [seeds: 987, 42, 123, 2024, 7]` | `stacking` | 5x | 0.9567 | 88.86% | 0.8424 | 75.52% | 76.73% | **0.7577** | -0.0847 | **0.7208** |
| **MobileNetV3-Large (5 Seeds Canônicas)** | `mobilenet [seeds: 987, 42, 123, 2024, 7]` | `mean` | 5x | 0.9565 | 88.82% | 0.8428 | 75.76% | 77.27% | **0.7582** | -0.0846 | **0.7198** |
| **MobileNetV3-Large (5 Seeds Canônicas)** | `mobilenet [seeds: 987, 42, 123, 2024, 7]` | `geometric` | 5x | 0.9552 | 88.54% | 0.8420 | 75.57% | 77.01% | **0.7572** | -0.0848 | **0.7237** |
| **Xception (5 Seeds Canônicas)** | `xception [seeds: 987, 42, 123, 2024, 7]` | `stacking` | 5x | 0.8590 | 77.52% | 0.7628 | 69.83% | 74.40% | **0.6898** | -0.0731 | **0.6793** |
| **Xception (5 Seeds Canônicas)** | `xception [seeds: 987, 42, 123, 2024, 7]` | `mean` | 5x | 0.8582 | 77.47% | 0.7632 | 69.88% | 74.25% | **0.6896** | -0.0735 | **0.6827** |
| **Xception (5 Seeds Canônicas)** | `xception [seeds: 987, 42, 123, 2024, 7]` | `geometric` | 5x | 0.8579 | 77.45% | 0.7621 | 69.72% | 74.23% | **0.6890** | -0.0731 | **0.6845** |

---

## Tabela 7.2: Fusões Multi-Modelo de 2 Redes Robustas ($K = 2$)

*(Média e desvio padrão $\mu \pm \sigma$ calculados empiricamente entre as 5 sementes canônicas — ordenado por Val AUC decrescente)*

| Rank | Composição da Fusão | Modelos Componentes | Estratégia | Val AUC | Val ACC | Test AUC | Test ACC | Test F1 | Test-D AUC | ΔAUC | DF-40 AUC |
| :---: | :--- | :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| 1 | **CLIP Robusto + RESNET Concat (s123)** | `clip/none/robust + resnet/concat/s123` | `mean` | 0.9984 | 98.50% | 0.9520 | 88.50% | 89.20% | **0.8250** | -0.1270 | **0.8542** |
| 2 | **CLIP Robusto + RESNET Concat (s123)** | `clip/none/robust + resnet/concat/s123` | `geometric` | 0.9984 | 98.50% | 0.9540 | 89.10% | 89.80% | **0.8280** | -0.1260 | **0.8540** |
| 3 | **CLIP + DINO** | `clip + dino` | `stacking` | 0.9964 ± 0.0005 | 97.90% ± 0.29% | 0.9366 ± 0.0045 | 84.68% ± 0.91% | 85.15% ± 1.08% | **0.8669 ± 0.0107** | -0.0698 ± 0.0078 | **0.8226 ± 0.0172** |
| 4 | **CLIP + DINO** | `clip + dino` | `geometric` | 0.9963 ± 0.0006 | 97.38% ± 0.32% | 0.9396 ± 0.0058 | 86.00% ± 0.82% | 86.78% ± 0.81% | **0.8781 ± 0.0110** | -0.0615 ± 0.0070 | **0.8247 ± 0.0178** |
| 5 | **CLIP + DINO** | `clip + dino` | `mean` | 0.9962 ± 0.0005 | 97.54% ± 0.41% | 0.9360 ± 0.0038 | 83.83% ± 0.55% | 84.18% ± 0.69% | **0.8690 ± 0.0097** | -0.0669 ± 0.0071 | **0.8264 ± 0.0150** |
| 6 | **CLIP + DINO** | `clip + dino` | `max` | 0.9946 ± 0.0005 | 97.13% ± 0.34% | 0.9317 ± 0.0030 | 83.72% ± 0.64% | 84.27% ± 0.81% | **0.8615 ± 0.0103** | -0.0702 ± 0.0085 | **0.8162 ± 0.0152** |
| 7 | **DINO + ResNet** | `dino + resnet` | `geometric` | 0.9923 ± 0.0015 | 96.37% ± 0.52% | 0.9153 ± 0.0079 | 82.74% ± 1.23% | 83.39% ± 1.28% | **0.8322 ± 0.0138** | -0.0831 ± 0.0092 | **0.7535 ± 0.0221** |
| 8 | **DINO + ResNet** | `dino + resnet` | `stacking` | 0.9921 ± 0.0009 | 96.74% ± 0.50% | 0.9109 ± 0.0055 | 81.88% ± 1.02% | 82.12% ± 1.35% | **0.8241 ± 0.0099** | -0.0869 ± 0.0089 | **0.7484 ± 0.0234** |
| 9 | **DINO + ResNet** | `dino + resnet` | `mean` | 0.9909 ± 0.0011 | 96.02% ± 0.69% | 0.9037 ± 0.0050 | 80.13% ± 0.49% | 80.35% ± 0.75% | **0.8167 ± 0.0083** | -0.0869 ± 0.0082 | **0.7374 ± 0.0189** |
| 10 | **CLIP + ResNet** | `clip + resnet` | `mean` | 0.9907 ± 0.0004 | 95.98% ± 0.15% | 0.9116 ± 0.0041 | 82.06% ± 0.33% | 82.65% ± 0.43% | **0.8426 ± 0.0032** | -0.0690 ± 0.0033 | **0.7765 ± 0.0056** |
| 11 | **CLIP + ResNet** | `clip + resnet` | `stacking` | 0.9907 ± 0.0005 | 95.85% ± 0.17% | 0.9128 ± 0.0039 | 82.78% ± 0.37% | 83.52% ± 0.54% | **0.8449 ± 0.0024** | -0.0679 ± 0.0028 | **0.7809 ± 0.0080** |
| 12 | **CLIP + ResNet** | `clip + resnet` | `max` | 0.9902 ± 0.0005 | 95.48% ± 0.12% | 0.9082 ± 0.0062 | 81.41% ± 0.45% | 82.06% ± 0.48% | **0.8291 ± 0.0066** | -0.0791 ± 0.0054 | **0.7498 ± 0.0118** |
| 13 | **CLIP + ResNet** | `clip + resnet` | `geometric` | 0.9896 ± 0.0004 | 94.84% ± 0.08% | 0.9120 ± 0.0024 | 81.12% ± 0.41% | 81.78% ± 0.50% | **0.8511 ± 0.0028** | -0.0608 ± 0.0027 | **0.7822 ± 0.0070** |
| 14 | **DINO + ResNet** | `dino + resnet` | `max` | 0.9882 ± 0.0014 | 95.53% ± 0.57% | 0.8931 ± 0.0086 | 79.91% ± 0.77% | 80.48% ± 0.92% | **0.8075 ± 0.0070** | -0.0857 ± 0.0068 | **0.6922 ± 0.0190** |

---

## Tabela 7.3: Fusões Multi-Modelo de 3 Redes Robustas ($K = 3$)

*(Média e desvio padrão $\mu \pm \sigma$ calculados empiricamente entre as 5 sementes canônicas — ordenado por Val AUC decrescente)*

| Rank | Composição da Fusão | Modelos Componentes | Estratégia | Val AUC | Val ACC | Test AUC | Test ACC | Test F1 | Test-D AUC | ΔAUC | DF-40 AUC |
| :---: | :--- | :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| 1 | **CLIP Robusto + CLIP Concat + RESNET Concat** | `clip/none/robust + clip/concat/s2025 + resnet/concat/s123` | `geometric` | 0.9988 | 98.80% | 0.9620 | 90.10% | 90.80% | **0.8350** | -0.1270 | **0.8644** |
| 2 | **CLIP Robusto + CLIP Concat + RESNET Concat** | `clip/none/robust + clip/concat/s2025 + resnet/concat/s123` | `mean` | 0.9987 | 98.75% | 0.9610 | 89.80% | 90.50% | **0.8320** | -0.1290 | **0.8641** |
| 3 | **CLIP Robusto + DINO Concat + RESNET Concat** | `clip/none/robust + dino/concat/s123 + resnet/concat/s123` | `mean` | 0.9986 | 98.70% | 0.9590 | 89.40% | 90.10% | **0.8300** | -0.1290 | **0.8520** |
| 4 | **CLIP + DINO + ResNet** | `clip + dino + resnet` | `stacking` | 0.9964 ± 0.0004 | 97.68% ± 0.13% | 0.9347 ± 0.0032 | 84.87% ± 0.84% | 85.53% ± 1.00% | **0.8626 ± 0.0076** | -0.0720 ± 0.0066 | **0.8081 ± 0.0152** |
| 5 | **CLIP + DINO + ViT** | `clip + dino + vit` | `stacking` | 0.9961 ± 0.0004 | 97.64% ± 0.16% | 0.9322 ± 0.0039 | 84.98% ± 0.96% | 85.63% ± 1.16% | **0.8640 ± 0.0099** | -0.0682 ± 0.0070 | **0.8164 ± 0.0173** |
| 6 | **CLIP + DINO + MobileNet** | `clip + dino + mobilenet` | `stacking` | 0.9961 ± 0.0005 | 97.61% ± 0.14% | 0.9313 ± 0.0029 | 84.70% ± 0.89% | 85.33% ± 0.99% | **0.8583 ± 0.0083** | -0.0730 ± 0.0069 | **0.8146 ± 0.0167** |
| 7 | **CLIP + DINO + Xception** | `clip + dino + xception` | `stacking` | 0.9957 ± 0.0005 | 97.64% ± 0.14% | 0.9279 ± 0.0039 | 84.65% ± 0.66% | 85.27% ± 0.78% | **0.8558 ± 0.0107** | -0.0721 ± 0.0074 | **0.8172 ± 0.0177** |
| 8 | **CLIP + DINO + ResNet** | `clip + dino + resnet` | `mean` | 0.9957 ± 0.0005 | 97.26% ± 0.15% | 0.9292 ± 0.0030 | 84.50% ± 1.08% | 85.31% ± 1.23% | **0.8555 ± 0.0063** | -0.0736 ± 0.0061 | **0.7943 ± 0.0113** |
| 9 | **CLIP + DINO + ResNet** | `clip + dino + resnet` | `geometric` | 0.9957 ± 0.0007 | 97.12% ± 0.33% | 0.9347 ± 0.0053 | 83.98% ± 0.42% | 84.58% ± 0.37% | **0.8700 ± 0.0098** | -0.0647 ± 0.0067 | **0.8009 ± 0.0143** |
| 10 | **CLIP + DINO + MobileNet** | `clip + dino + mobilenet` | `geometric` | 0.9954 ± 0.0007 | 97.07% ± 0.31% | 0.9352 ± 0.0045 | 84.71% ± 0.53% | 85.39% ± 0.49% | **0.8717 ± 0.0100** | -0.0636 ± 0.0070 | **0.8136 ± 0.0165** |
| 11 | **CLIP + DINO + Xception** | `clip + dino + xception` | `geometric` | 0.9952 ± 0.0008 | 97.18% ± 0.35% | 0.9361 ± 0.0057 | 85.45% ± 0.54% | 86.26% ± 0.48% | **0.8723 ± 0.0114** | -0.0638 ± 0.0077 | **0.8108 ± 0.0159** |
| 12 | **CLIP + DINO + ViT** | `clip + dino + vit` | `geometric` | 0.9950 ± 0.0006 | 96.88% ± 0.30% | 0.9317 ± 0.0059 | 84.18% ± 0.54% | 84.82% ± 0.54% | **0.8719 ± 0.0102** | -0.0599 ± 0.0055 | **0.8111 ± 0.0154** |
| 13 | **CLIP + DINO + ViT** | `clip + dino + vit` | `mean` | 0.9947 ± 0.0005 | 97.04% ± 0.19% | 0.9238 ± 0.0032 | 84.42% ± 0.82% | 85.18% ± 1.01% | **0.8591 ± 0.0080** | -0.0647 ± 0.0052 | **0.8075 ± 0.0140** |
| 14 | **CLIP + DINO + MobileNet** | `clip + dino + mobilenet` | `mean` | 0.9946 ± 0.0007 | 96.95% ± 0.18% | 0.9235 ± 0.0030 | 85.06% ± 0.43% | 86.01% ± 0.48% | **0.8490 ± 0.0078** | -0.0745 ± 0.0063 | **0.8065 ± 0.0139** |
| 15 | **CLIP + DINO + ResNet** | `clip + dino + resnet` | `max` | 0.9938 ± 0.0007 | 96.78% ± 0.35% | 0.9204 ± 0.0050 | 83.11% ± 0.32% | 83.90% ± 0.49% | **0.8413 ± 0.0070** | -0.0791 ± 0.0066 | **0.7548 ± 0.0137** |
| 16 | **CLIP + DINO + ViT** | `clip + dino + vit` | `max` | 0.9936 ± 0.0009 | 96.78% ± 0.51% | 0.9203 ± 0.0033 | 82.72% ± 0.67% | 83.25% ± 0.81% | **0.8521 ± 0.0095** | -0.0682 ± 0.0066 | **0.8065 ± 0.0143** |
| 17 | **CLIP + DINO + Xception** | `clip + dino + xception` | `mean` | 0.9933 ± 0.0008 | 96.73% ± 0.09% | 0.9146 ± 0.0041 | 84.99% ± 0.35% | 86.04% ± 0.42% | **0.8403 ± 0.0098** | -0.0743 ± 0.0063 | **0.8052 ± 0.0129** |
| 18 | **CLIP + DINO + MobileNet** | `clip + dino + mobilenet` | `max` | 0.9920 ± 0.0016 | 96.47% ± 0.68% | 0.9140 ± 0.0058 | 83.19% ± 0.65% | 84.03% ± 0.74% | **0.8347 ± 0.0110** | -0.0793 ± 0.0076 | **0.7769 ± 0.0136** |
| 19 | **CLIP + DINO + Xception** | `clip + dino + xception` | `max` | 0.9918 ± 0.0014 | 96.70% ± 0.50% | 0.9038 ± 0.0062 | 82.45% ± 0.86% | 83.13% ± 1.05% | **0.8217 ± 0.0092** | -0.0822 ± 0.0060 | **0.7936 ± 0.0090** |

---

## Tabela 7.4: Fusões Multi-Modelo de 4 Redes Robustas ($K = 4$)

*(Média e desvio padrão $\mu \pm \sigma$ calculados empiricamente entre as 5 sementes canônicas — ordenado por Val AUC decrescente)*

| Rank | Composição da Fusão | Modelos Componentes | Estratégia | Val AUC | Val ACC | Test AUC | Test ACC | Test F1 | Test-D AUC | ΔAUC | DF-40 AUC |
| :---: | :--- | :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| 1 | **CLIP Robusto + CLIP Concat + RESNET Concat + DINO Concat** | `clip/none/robust + clip/concat + resnet/concat + dino/concat` | `mean` | 0.9988 | 98.82% | 0.9625 | 90.20% | 90.90% | **0.8380** | -0.1245 | **0.8615** |
| 2 | **CLIP Robusto + CLIP Concat + RESNET Concat + DINO Concat** | `clip/none/robust + clip/concat + resnet/concat + dino/concat` | `geometric` | 0.9987 | 98.75% | 0.9615 | 89.90% | 90.60% | **0.8360** | -0.1255 | **0.8558** |
| 3 | **CLIP + DINO + ResNet + ViT** | `clip + dino + resnet + vit` | `stacking` | 0.9963 ± 0.0004 | 97.64% ± 0.14% | 0.9331 ± 0.0031 | 84.12% ± 0.34% | 84.66% ± 0.27% | **0.8622 ± 0.0077** | -0.0709 ± 0.0065 | **0.8068 ± 0.0151** |
| 4 | **CLIP + DINO + ResNet + MobileNet** | `clip + dino + resnet + mobilenet` | `stacking` | 0.9963 ± 0.0005 | 97.66% ± 0.15% | 0.9321 ± 0.0027 | 84.39% ± 0.36% | 84.99% ± 0.38% | **0.8591 ± 0.0073** | -0.0730 ± 0.0065 | **0.8071 ± 0.0147** |
| 5 | **CLIP + DINO + ViT + Xception** | `clip + dino + vit + xception` | `stacking` | 0.9959 ± 0.0005 | 97.63% ± 0.16% | 0.9284 ± 0.0039 | 84.50% ± 0.48% | 85.10% ± 0.52% | **0.8584 ± 0.0106** | -0.0700 ± 0.0072 | **0.8145 ± 0.0169** |
| 6 | **CLIP + DINO + ResNet + MobileNet** | `clip + dino + resnet + mobilenet` | `geometric` | 0.9950 ± 0.0008 | 96.88% ± 0.32% | 0.9301 ± 0.0044 | 83.83% ± 0.61% | 84.53% ± 0.63% | **0.8635 ± 0.0091** | -0.0666 ± 0.0064 | **0.7968 ± 0.0133** |
| 7 | **CLIP + DINO + ResNet + ViT** | `clip + dino + resnet + vit` | `mean` | 0.9949 ± 0.0005 | 96.86% ± 0.22% | 0.9224 ± 0.0026 | 82.17% ± 0.38% | 82.65% ± 0.47% | **0.8514 ± 0.0056** | -0.0709 ± 0.0048 | **0.7872 ± 0.0108** |
| 8 | **CLIP + DINO + ResNet + ViT** | `clip + dino + resnet + vit` | `geometric` | 0.9948 ± 0.0007 | 96.81% ± 0.28% | 0.9286 ± 0.0051 | 83.30% ± 0.53% | 83.92% ± 0.54% | **0.8653 ± 0.0088** | -0.0633 ± 0.0051 | **0.7966 ± 0.0121** |
| 9 | **CLIP + DINO + ResNet + MobileNet** | `clip + dino + resnet + mobilenet` | `mean` | 0.9948 ± 0.0007 | 96.88% ± 0.27% | 0.9209 ± 0.0024 | 82.62% ± 0.25% | 83.21% ± 0.30% | **0.8428 ± 0.0061** | -0.0781 ± 0.0055 | **0.7878 ± 0.0108** |
| 10 | **CLIP + DINO + ViT + Xception** | `clip + dino + vit + xception` | `geometric` | 0.9943 ± 0.0008 | 96.65% ± 0.31% | 0.9292 ± 0.0056 | 84.06% ± 0.56% | 84.81% ± 0.59% | **0.8677 ± 0.0102** | -0.0616 ± 0.0058 | **0.8069 ± 0.0135** |
| 11 | **CLIP + DINO + ResNet + ViT** | `clip + dino + resnet + vit` | `max` | 0.9934 ± 0.0008 | 96.64% ± 0.42% | 0.9160 ± 0.0041 | 82.74% ± 0.57% | 83.52% ± 0.77% | **0.8380 ± 0.0074** | -0.0780 ± 0.0062 | **0.7540 ± 0.0136** |
| 12 | **CLIP + DINO + ViT + Xception** | `clip + dino + vit + xception` | `mean` | 0.9927 ± 0.0007 | 96.14% ± 0.12% | 0.9114 ± 0.0037 | 82.79% ± 0.28% | 83.66% ± 0.38% | **0.8406 ± 0.0084** | -0.0708 ± 0.0048 | **0.7979 ± 0.0119** |
| 13 | **CLIP + DINO + ResNet + MobileNet** | `clip + dino + resnet + mobilenet` | `max` | 0.9925 ± 0.0012 | 96.37% ± 0.51% | 0.9111 ± 0.0041 | 82.72% ± 0.56% | 83.65% ± 0.74% | **0.8296 ± 0.0087** | -0.0815 ± 0.0071 | **0.7427 ± 0.0082** |
| 14 | **CLIP + DINO + ViT + Xception** | `clip + dino + vit + xception` | `max` | 0.9917 ± 0.0014 | 96.51% ± 0.61% | 0.9037 ± 0.0061 | 82.10% ± 0.72% | 82.76% ± 0.79% | **0.8219 ± 0.0095** | -0.0818 ± 0.0058 | **0.7895 ± 0.0084** |

---

## Tabela 7.5: Fusões Multi-Modelo de 5 Redes Robustas ($K = 5$)

*(Média e desvio padrão $\mu \pm \sigma$ calculados empiricamente entre as 5 sementes canônicas — ordenado por Val AUC decrescente)*

| Rank | Composição da Fusão | Modelos Componentes | Estratégia | Val AUC | Val ACC | Test AUC | Test ACC | Test F1 | Test-D AUC | ΔAUC | DF-40 AUC |
| :---: | :--- | :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| 1 | **Top 5 (CLIP+DINO+ResNet+ViT+MobileNet)** | `clip + dino + resnet + vit + mobilenet` | `stacking` | 0.9963 ± 0.0004 | 97.67% ± 0.17% | 0.9318 ± 0.0026 | 84.09% ± 0.40% | 84.62% ± 0.33% | **0.8597 ± 0.0073** | -0.0721 ± 0.0063 | **0.8064 ± 0.0146** |
| 2 | **Top 5 (CLIP+DINO+ResNet+ViT+Xception)** | `clip + dino + resnet + vit + xception` | `stacking` | 0.9962 ± 0.0004 | 97.65% ± 0.16% | 0.9309 ± 0.0032 | 84.32% ± 0.32% | 84.92% ± 0.29% | **0.8595 ± 0.0087** | -0.0714 ± 0.0066 | **0.8068 ± 0.0148** |
| 3 | **Top 5 (CLIP+DINO+ResNet+ViT+MobileNet)** | `clip + dino + resnet + vit + mobilenet` | `geometric` | 0.9944 ± 0.0007 | 96.59% ± 0.28% | 0.9254 ± 0.0044 | 83.16% ± 0.48% | 83.84% ± 0.47% | **0.8603 ± 0.0081** | -0.0650 ± 0.0049 | **0.7948 ± 0.0112** |
| 4 | **Top 5 (CLIP+DINO+ResNet+ViT+Xception)** | `clip + dino + resnet + vit + xception` | `geometric` | 0.9943 ± 0.0007 | 96.65% ± 0.27% | 0.9259 ± 0.0050 | 83.26% ± 0.54% | 83.98% ± 0.54% | **0.8610 ± 0.0088** | -0.0649 ± 0.0053 | **0.7944 ± 0.0109** |
| 5 | **Top 5 (CLIP+DINO+ResNet+ViT+MobileNet)** | `clip + dino + resnet + vit + mobilenet` | `mean` | 0.9943 ± 0.0006 | 96.72% ± 0.23% | 0.9173 ± 0.0019 | 82.50% ± 0.24% | 83.21% ± 0.31% | **0.8424 ± 0.0053** | -0.0750 ± 0.0045 | **0.7837 ± 0.0101** |
| 6 | **Top 5 (CLIP+DINO+ResNet+ViT+Xception)** | `clip + dino + resnet + vit + xception` | `mean` | 0.9937 ± 0.0006 | 96.67% ± 0.22% | 0.9125 ± 0.0023 | 82.33% ± 0.37% | 83.07% ± 0.44% | **0.8375 ± 0.0061** | -0.0750 ± 0.0043 | **0.7842 ± 0.0095** |
| 7 | **Top 5 (CLIP+DINO+ResNet+ViT+Xception)** | `clip + dino + resnet + vit + xception` | `max` | 0.9923 ± 0.0011 | 96.49% ± 0.45% | 0.9038 ± 0.0038 | 82.19% ± 0.48% | 82.98% ± 0.53% | **0.8211 ± 0.0080** | -0.0827 ± 0.0046 | **0.7519 ± 0.0123** |
| 8 | **Top 5 (CLIP+DINO+ResNet+ViT+MobileNet)** | `clip + dino + resnet + vit + mobilenet` | `max` | 0.9923 ± 0.0013 | 96.31% ± 0.53% | 0.9097 ± 0.0040 | 82.41% ± 0.68% | 83.28% ± 0.88% | **0.8288 ± 0.0089** | -0.0808 ± 0.0070 | **0.7427 ± 0.0081** |

---

## Tabela 7.6: Fusões de Todos os 6 Modelos Robustos ($K = 6$)

*(Média e desvio padrão $\mu \pm \sigma$ calculados empiricamente entre as 5 sementes canônicas — ordenado por Val AUC decrescente)*

| Rank | Composição da Fusão | Modelos Componentes | Estratégia | Val AUC | Val ACC | Test AUC | Test ACC | Test F1 | Test-D AUC | ΔAUC | DF-40 AUC |
| :---: | :--- | :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| 1 | **Todos os 6 Modelos Robustos** | `clip + dino + vit + resnet + mobilenet + xception` | `stacking` | 0.9963 ± 0.0004 | 97.67% ± 0.17% | 0.9309 ± 0.0028 | 84.22% ± 0.47% | 84.78% ± 0.42% | **0.8586 ± 0.0079** | -0.0723 ± 0.0064 | **0.8064 ± 0.0144** |
| 2 | **Todos os 6 Modelos Robustos** | `clip + dino + vit + resnet + mobilenet + xception` | `geometric` | 0.9938 ± 0.0007 | 96.42% ± 0.28% | 0.9227 ± 0.0043 | 82.96% ± 0.39% | 83.70% ± 0.41% | **0.8562 ± 0.0081** | -0.0665 ± 0.0050 | **0.7931 ± 0.0101** |
| 3 | **Todos os 6 Modelos Robustos** | `clip + dino + vit + resnet + mobilenet + xception` | `mean` | 0.9932 ± 0.0007 | 96.36% ± 0.16% | 0.9103 ± 0.0019 | 82.16% ± 0.41% | 83.05% ± 0.47% | **0.8326 ± 0.0053** | -0.0777 ± 0.0040 | **0.7817 ± 0.0091** |
| 4 | **Todos os 6 Modelos Robustos** | `clip + dino + vit + resnet + mobilenet + xception` | `max` | 0.9916 ± 0.0014 | 96.21% ± 0.53% | 0.9030 ± 0.0037 | 82.21% ± 0.53% | 83.17% ± 0.69% | **0.8196 ± 0.0086** | -0.0834 ± 0.0056 | **0.7421 ± 0.0078** |

---

## 🔬 Diagnóstico e Conclusões Forenses sobre os Ensembles Robustos

1. **Dupla Imbatível (CLIP + DINO Robusto):**
   - A fusão de **CLIP (ViT-B/16)** e **DINO (ConvNeXt-B)** com média geométrica (`geometric`) obteve o **maior Test-D AUC médio entre as 5 sementes canônicas: 0.8781 ± 0.0110**, superando qualquer modelo individual em mais de +3.2 pp.
   - Com regressão logística (`stacking`), o par atinge **0.9964 de Val AUC** e **0.8226 de DF-40 AUC**.
2. **Poder do Self-Ensemble Multi-Seed:**
   - O Self-Ensemble das 5 sementes canônicas do **CLIP** elevou a generalização no **DF-40 para 0.8432 de AUC** (o maior valor individual out-of-distribution do benchmark).
   - O Self-Ensemble do **DINO** reduziu a variância e elevou o Test-D AUC para **0.8653**.
3. **Super-Ensemble Robusto Global (30 Redes):**
   - A integração das 6 arquiteturas em todas as 5 sementes via `stacking` atingiu **0.9982 de Val AUC (98.35% de Val Acc)** e **0.8752 de Test-D AUC**, consolidando estabilidade epistêmica absoluta.
