# Tabela 9: Resultados Consolidados de Ensembles SRM (Matriz Canônica de 5 Sementes)

**Documento:** `tabela9-ensemble-srm-5seeds.md`  
**Ambiente:** Dual NVIDIA GeForce RTX 3090 | Workstation Local (`sicret2`)  
**Sementes Avaliadas:** 5 Sementes Canônicas (`42`, `123`, `2024`, `7`, `2025`)  
**Total de Modelos Treinados:** 30 modelos (6 arquiteturas × 5 seeds)  
**Estratégias de Fusão:** `geometric` (Média Geométrica), `stacking` (Regressão Logística em Validação), `mean` (Média Aritmética), `max` (Max Pooling)  

---

## 1. 🌟 Mega-Ensembles Globais (Fusão de Metas e dos 30 Modelos)

| Composição / Comitê | Estratégia | N° Modelos | Test Limpo (AUC) | Test Acc | Test-D Difícil (AUC) | Test-D Acc | ΔAUC | DF-40 (AUC) | Celeb-DF Vídeo (AUC) |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| **DINO + CLIP (Vision-Language & Self-Supervised)** | `geometric` | 10x | 94.17% | 85.32% | **88.60%** | 78.01% | -5.57% | 84.15% | **75.06%** |
| **Transformers Committee (CLIP + DINO + ViT)** | `stacking` | 15x | 94.20% | 85.03% | **88.34%** | 77.07% | -5.86% | 83.57% | **76.76%** |
| **DINO + CLIP (Vision-Language & Self-Supervised)** | `stacking` | 10x | 94.22% | 84.52% | **88.33%** | 76.51% | -5.90% | 83.57% | **76.68%** |
| **DINO + CLIP (Vision-Language & Self-Supervised)** | `mean` | 10x | 93.90% | 83.84% | **88.16%** | 76.58% | -5.74% | 83.92% | **74.50%** |
| **DINO + CLIP + Xception (Multi-Spectral Champion)** | `stacking` | 15x | 93.96% | 84.86% | **87.86%** | 76.81% | -6.10% | 83.38% | **76.79%** |
| **Mega-Ensemble (All 6 Architectures x 5 Seeds)** | `stacking` | 30x | 94.00% | 85.04% | **87.71%** | 76.90% | -6.30% | 82.96% | **77.11%** |
| **DINO + CLIP + Xception (Multi-Spectral Champion)** | `geometric` | 15x | 93.65% | 84.94% | **87.70%** | 77.73% | -5.94% | 83.07% | **75.34%** |
| **Transformers Committee (CLIP + DINO + ViT)** | `geometric` | 15x | 92.91% | 83.33% | **87.21%** | 76.78% | -5.70% | 83.03% | **73.22%** |
| **Bagged 5-Seeds: DINO (ConvNeXt-B)** | `mean_seeds` | 5x | 93.67% | 84.69% | **87.19%** | 76.66% | -6.48% | 82.28% | **77.24%** |
| **Transformers Committee (CLIP + DINO + ViT)** | `mean` | 15x | 92.60% | 83.10% | **86.79%** | 76.22% | -5.81% | 82.66% | **72.68%** |

---

## 2. 🥇 Top 15 Ensembles Canônicos (Média das 5 Sementes ± Desvio Padrão)

Ordenado por **Test-D AUC** (Robustez sob corrupções não vistas):

| Rank | Composição da Fusão | Estratégia | K | Test Limpo (AUC) | Test-D Difícil (AUC) | ΔAUC | DF-40 Cross-Data (AUC) | Celeb-DF Vídeo (AUC) |
| :---: | :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| 🥇 | **CLIP ViT-B/16 + DINO (ConvNeXt-B)** | `geometric` | 2x | 93.07 ± 0.61% | **87.24 ± 0.49%** | -5.82% | 83.53 ± 1.74% | **74.11 ± 3.14%** |
| 🥈 | **CLIP ViT-B/16 + DINO (ConvNeXt-B)** | `stacking` | 2x | 92.99 ± 0.42% | **86.78 ± 0.49%** | -6.21% | 82.60 ± 2.02% | **75.02 ± 4.04%** |
| 🥉 | **CLIP ViT-B/16 + DINO (ConvNeXt-B)** | `mean` | 2x | 92.80 ± 0.40% | **86.71 ± 0.50%** | -6.09% | 82.80 ± 1.82% | **73.52 ± 3.39%** |
| 4 | **CLIP ViT-B/16 + DINO (ConvNeXt-B) + Xception** | `geometric` | 3x | 92.72 ± 0.48% | **86.64 ± 0.45%** | -6.08% | 82.83 ± 1.52% | **74.47 ± 2.95%** |
| 5 | **CLIP ViT-B/16 + DINO (ConvNeXt-B) + MobileNetV3** | `geometric` | 3x | 92.64 ± 0.45% | **86.55 ± 0.41%** | -6.09% | 83.08 ± 1.35% | **75.98 ± 3.16%** |
| 6 | **CLIP ViT-B/16 + DINO (ConvNeXt-B) + ViT-B/16** | `stacking` | 3x | 92.68 ± 0.42% | **86.51 ± 0.30%** | -6.17% | 82.41 ± 1.70% | **74.99 ± 3.88%** |
| 7 | **CLIP ViT-B/16 + DINO (ConvNeXt-B) + ResNet-18** | `geometric` | 3x | 92.64 ± 0.46% | **86.29 ± 0.22%** | -6.35% | 81.66 ± 1.73% | **73.84 ± 2.83%** |
| 8 | **CLIP ViT-B/16 + DINO (ConvNeXt-B) + ViT-B/16** | `geometric` | 3x | 92.03 ± 0.49% | **86.29 ± 0.42%** | -5.74% | 83.02 ± 1.02% | **72.57 ± 1.58%** |
| 9 | **CLIP ViT-B/16 + DINO (ConvNeXt-B) + ResNet-18** | `stacking` | 3x | 92.69 ± 0.41% | **86.03 ± 0.37%** | -6.66% | 81.15 ± 1.71% | **74.79 ± 3.77%** |
| 10 | **CLIP ViT-B/16 + DINO (ConvNeXt-B) + ViT-B/16 + ResNet-18** | `stacking` | 4x | 92.63 ± 0.41% | **86.02 ± 0.35%** | -6.61% | 81.17 ± 1.56% | **74.86 ± 3.68%** |
| 11 | **CLIP ViT-B/16 + DINO (ConvNeXt-B) + ViT-B/16 + MobileNetV3** | `stacking` | 4x | 92.58 ± 0.37% | **86.02 ± 0.51%** | -6.56% | 82.09 ± 1.47% | **75.92 ± 3.60%** |
| 12 | **CLIP ViT-B/16 + DINO (ConvNeXt-B) + MobileNetV3** | `stacking` | 3x | 92.62 ± 0.36% | **85.99 ± 0.56%** | -6.63% | 82.10 ± 1.57% | **76.05 ± 3.64%** |
| 13 | **CLIP ViT-B/16 + DINO (ConvNeXt-B) + ViT-B/16 + Xception** | `stacking` | 4x | 92.36 ± 0.48% | **85.96 ± 0.52%** | -6.40% | 82.17 ± 1.51% | **75.18 ± 3.86%** |
| 14 | **CLIP ViT-B/16 + DINO (ConvNeXt-B) + Xception** | `stacking` | 3x | 92.37 ± 0.51% | **85.90 ± 0.60%** | -6.47% | 82.17 ± 1.63% | **75.23 ± 3.93%** |
| 15 | **CLIP ViT-B/16 + DINO (ConvNeXt-B) + ViT-B/16 + Xception** | `geometric` | 4x | 91.82 ± 0.44% | **85.89 ± 0.39%** | -5.93% | 82.75 ± 0.96% | **72.87 ± 1.67%** |

---

## 3. 🎯 Comparativo do Comitê de Todos os 6 Modelos (K=6) nas 5 Sementes

| Estratégia | Test Limpo (AUC) | Test-D Difícil (AUC) | ΔAUC | DF-40 (AUC) | Celeb-DF Vídeo (AUC) |
| :---: | :---: | :---: | :---: | :---: | :---: |
| `stacking` | 92.47 ± 0.38% | **85.67 ± 0.51%** | -6.80% | 81.21 ± 1.44% | **75.53 ± 3.43%** |
| `geometric` | 91.36 ± 0.26% | **84.81 ± 0.26%** | -6.55% | 81.52 ± 0.83% | **74.00 ± 1.92%** |
| `mean` | 90.32 ± 0.22% | **82.83 ± 0.35%** | -7.48% | 79.76 ± 0.60% | **73.66 ± 2.57%** |
| `max` | 89.24 ± 0.32% | **81.12 ± 0.47%** | -8.12% | 75.02 ± 0.87% | **68.33 ± 3.55%** |