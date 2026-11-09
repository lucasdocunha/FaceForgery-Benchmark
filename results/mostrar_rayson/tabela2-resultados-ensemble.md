# Tabela 2: Resultados Consolidados de Ensembles e Fusões Multimodais (Baseline Finetune)

**Documento:** `tabela2-resultados-ensemble.md`  
**Destinatário:** Apresentação Técnica / Rayson  
**Ambiente de Execução:** Dual NVIDIA GeForce RTX 3090 (24GB) | Workstation Local (`sicret2`)  
**Critério de Ordenação Estrito:** **Melhor ROC-AUC na Validação (`Val AUC` decrescente)**  
**Regime dos Modelos Integrados:** **Exclusivamente Baseline `finetune`** *(sem modelos robustos)*  
**Data de Extração:** 16 de Setembro de 2026  

> [!NOTE]
> **Aviso Metodológico:** Conforme solicitado, esta tabela contempla **exclusivamente ensembles constituídos por modelos do regime padrão (`finetune`)**, sem nenhuma inclusão de modelos do regime `finetune_robust`. Todos os agrupamentos são particionados por quantidade de modelos integrados ($K = 2, 3, 4, 5, 6$) e ranqueados rigorosamente pelo **`Val AUC` decrescente**.

---

## 📌 Metodologia de Fusão e Estratégias Investigadas

As fusões combinam as predições probabilísticas $p_m(x) \in [0, 1]$ dos modelos baseline através de:
1. **Média Simples (`mean`)**: $P(y=1|x) = \frac{1}{K} \sum_{m=1}^{K} p_m(x)$ (Soft Voting uniforme).
2. **Média Ponderada (`weighted`)**: $P(y=1|x) = \sum_{m=1}^{K} w_m \cdot p_m(x)$, onde $w_m \propto \text{AUC}_{\text{val}, m}$.
3. **Média Geométrica (`geometric`)**: $P(y=1|x) = \left( \prod_{m=1}^{K} p_m(x) \right)^{1/K}$ (Penaliza modelos discordantes com alta incerteza).
4. **Stacking / Regressão Logística (`stacking`)**: Meta-classificador linear $\sigma(W^T \mathbf{p} + b)$ calibrado exclusivamente na validação.
5. **Pesos Ótimos (`optimal_weights`)**: Otimização SLSQP minimizando a perda de Brier/Cross-Entropy no conjunto de validação.
6. **Voto Majoritário (`majority`)**: Hard voting $P(y=1|x) = \mathbb{I}(\sum \mathbb{I}(p_m \ge 0.5) > K/2)$.

---
## Tabela 2.2: Fusões de 2 Modelos Baseline ($K = 2$)

*(Total de 30 estratégias avaliadas — ordenado estritamente por Val AUC decrescente)*

| Rank | Composição / Nome da Fusão | Modelos Componentes | Estratégia | Domínio | Val AUC | Val Acc | Test AUC | Test Acc | Test F1 | Test-D AUC | ΔAUC | DF-40 AUC | Celeb-DF Vídeo |
| :---: | :--- | :--- | :--- | :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| 1 | **CLIP + DINO (Baseline Espacial)** | `clip/none + dino/none` | `geometric` | `Espacial RGB` | **0.9968** | 95.87% | 0.9495 | 82.12% | 81.78% | **0.7341** | -0.2154 | **0.7552** | 0.3500 |
| 2 | **CLIP + DINO (Baseline Espacial)** | `clip/none + dino/none` | `stacking` | `Espacial RGB` | **0.9967** | 97.46% | 0.9477 | 85.48% | 85.87% | **0.7307** | -0.2169 | **0.7491** | 0.3500 |
| 3 | **CLIP + DINO (Baseline Espacial)** | `clip/none + dino/none` | `weighted` | `Espacial RGB` | **0.9966** | 97.67% | 0.9477 | 85.82% | 86.16% | **0.7345** | -0.2131 | **0.7542** | 0.3500 |
| 4 | **CLIP + DINO (Baseline Espacial)** | `clip/none + dino/none` | `mean` | `Espacial RGB` | **0.9965** | 97.96% | 0.9473 | 86.99% | 87.51% | **0.7350** | -0.2122 | **0.7566** | 0.3500 |
| 5 | **DINO + RESNET (Baseline Espacial)** | `dino/none + resnet/none` | `stacking` | `Espacial RGB` | **0.9949** | 97.47% | 0.9376 | 85.50% | 85.91% | **0.7058** | -0.2318 | **0.7231** | 0.3500 |
| 6 | **DINO + RESNET (Baseline Espacial)** | `dino/none + resnet/none` | `weighted` | `Espacial RGB` | **0.9946** | 96.67% | 0.9363 | 82.81% | 82.75% | **0.7031** | -0.2332 | **0.7228** | 0.3500 |
| 7 | **DINO + RESNET (Baseline Espacial)** | `dino/none + resnet/none` | `geometric` | `Espacial RGB` | **0.9945** | 94.59% | 0.9376 | 78.17% | 76.86% | **0.6942** | -0.2434 | **0.7179** | 0.3500 |
| 8 | **DINO + RESNET (Baseline Espacial)** | `dino/none + resnet/none` | `mean` | `Espacial RGB` | **0.9943** | 96.25% | 0.9356 | 82.26% | 82.32% | **0.7028** | -0.2328 | **0.7228** | 0.3500 |
| 9 | **CLIP + DINO (Concat RGB+Mag)** | `clip/concat + dino/concat` | `stacking` | `Híbrido RGB+Magnitude` | **0.9930** | 96.51% | 0.9343 | 84.05% | 84.41% | **0.7083** | -0.2260 | **0.7599** | 0.4200 |
| 10 | **CLIP + DINO (Concat RGB+Mag)** | `clip/concat + dino/concat` | `geometric` | `Híbrido RGB+Magnitude` | **0.9926** | 92.78% | 0.9344 | 79.05% | 78.00% | **0.7093** | -0.2251 | **0.7718** | 0.4200 |
| 11 | **CLIP + DINO (Concat RGB+Mag)** | `clip/concat + dino/concat` | `weighted` | `Híbrido RGB+Magnitude` | **0.9924** | 95.68% | 0.9336 | 82.50% | 82.39% | **0.7146** | -0.2189 | **0.7685** | 0.4200 |
| 12 | **DINO + RESNET (Baseline Espacial)** | `dino/none + resnet/none` | `max` | `Espacial RGB` | **0.9916** | 96.67% | 0.9274 | 86.30% | 87.36% | **0.7036** | -0.2239 | **0.7289** | 0.3500 |
| 13 | **RESNET + DINO (Concat RGB+Mag)** | `resnet/concat + dino/concat` | `geometric` | `Híbrido RGB+Magnitude` | **0.9915** | 94.98% | 0.9238 | 78.80% | 77.90% | **0.6699** | -0.2539 | **0.7702** | 0.4200 |
| 14 | **CLIP + DINO (Concat RGB+Mag)** | `clip/concat + dino/concat` | `mean` | `Híbrido RGB+Magnitude` | **0.9911** | 94.51% | 0.9322 | 83.06% | 83.19% | **0.7173** | -0.2148 | **0.7698** | 0.4200 |
| 15 | **RESNET + DINO (Concat RGB+Mag)** | `resnet/concat + dino/concat` | `stacking` | `Híbrido RGB+Magnitude` | **0.9911** | 96.45% | 0.9244 | 83.75% | 84.08% | **0.6751** | -0.2493 | **0.7611** | 0.4200 |
| 16 | **CLIP + RESNET (Baseline Espacial)** | `clip/none + resnet/none` | `mean` | `Espacial RGB` | **0.9906** | 95.85% | 0.9341 | 84.99% | 85.73% | **0.7285** | -0.2056 | **0.7652** | 0.3500 |
| 17 | **CLIP + RESNET (Baseline Espacial)** | `clip/none + resnet/none` | `weighted` | `Espacial RGB` | **0.9906** | 95.51% | 0.9339 | 83.86% | 84.48% | **0.7281** | -0.2058 | **0.7647** | 0.3500 |
| 18 | **RESNET + DINO (Concat RGB+Mag)** | `resnet/concat + dino/concat` | `weighted` | `Híbrido RGB+Magnitude` | **0.9904** | 96.29% | 0.9211 | 82.46% | 82.54% | **0.6731** | -0.2480 | **0.7645** | 0.4200 |
| 19 | **CLIP + RESNET (Baseline Espacial)** | `clip/none + resnet/none` | `stacking` | `Espacial RGB` | **0.9903** | 94.67% | 0.9321 | 81.37% | 81.69% | **0.7253** | -0.2069 | **0.7605** | 0.3500 |
| 20 | **RESNET + DINO (Concat RGB+Mag)** | `resnet/concat + dino/concat` | `mean` | `Híbrido RGB+Magnitude` | **0.9899** | 95.31% | 0.9196 | 82.45% | 83.01% | **0.6727** | -0.2470 | **0.7652** | 0.4200 |
| 21 | **CLIP + DINO (Baseline Espacial)** | `clip/none + dino/none` | `max` | `Espacial RGB` | **0.9897** | 93.09% | 0.9389 | 87.65% | 89.30% | **0.7410** | -0.1979 | **0.7451** | 0.3500 |
| 22 | **CLIP + RESNET (Baseline Espacial)** | `clip/none + resnet/none` | `geometric` | `Espacial RGB` | **0.9896** | 93.45% | 0.9332 | 77.83% | 76.73% | **0.7166** | -0.2166 | **0.7561** | 0.3500 |
| 23 | **CLIP + RESNET (Baseline Espacial)** | `clip/none + resnet/none` | `max` | `Espacial RGB` | **0.9895** | 91.78% | 0.9304 | 85.90% | 87.84% | **0.7228** | -0.2076 | **0.7595** | 0.3500 |
| 24 | **CLIP + DINO (Concat RGB+Mag)** | `clip/concat + dino/concat` | `max` | `Híbrido RGB+Magnitude` | **0.9815** | 93.77% | 0.9231 | 86.30% | 87.97% | **0.7336** | -0.1895 | **0.7798** | 0.4200 |
| 25 | **RESNET + CLIP (Concat RGB+Mag)** | `resnet/concat + clip/concat` | `mean` | `Híbrido RGB+Magnitude` | **0.9810** | 93.51% | 0.9168 | 83.79% | 85.02% | **0.7179** | -0.1989 | **0.8011** | 0.4200 |
| 26 | **RESNET + CLIP (Concat RGB+Mag)** | `resnet/concat + clip/concat` | `weighted` | `Híbrido RGB+Magnitude` | **0.9810** | 93.25% | 0.9163 | 83.11% | 84.35% | **0.7166** | -0.1997 | **0.8012** | 0.4200 |
| 27 | **RESNET + CLIP (Concat RGB+Mag)** | `resnet/concat + clip/concat` | `stacking` | `Híbrido RGB+Magnitude` | **0.9806** | 92.62% | 0.9128 | 80.49% | 81.47% | **0.7103** | -0.2026 | **0.7989** | 0.4200 |
| 28 | **RESNET + CLIP (Concat RGB+Mag)** | `resnet/concat + clip/concat` | `geometric` | `Híbrido RGB+Magnitude` | **0.9797** | 91.13% | 0.9125 | 77.16% | 76.32% | **0.7081** | -0.2044 | **0.7994** | 0.4200 |
| 29 | **RESNET + DINO (Concat RGB+Mag)** | `resnet/concat + dino/concat` | `max` | `Híbrido RGB+Magnitude` | **0.9773** | 92.30% | 0.9045 | 84.09% | 86.03% | **0.6644** | -0.2401 | **0.7790** | 0.4200 |
| 30 | **RESNET + CLIP (Concat RGB+Mag)** | `resnet/concat + clip/concat` | `max` | `Híbrido RGB+Magnitude` | **0.9772** | 89.57% | 0.9131 | 83.40% | 86.02% | **0.7037** | -0.2094 | **0.8091** | 0.4200 |

---

## Tabela 2.3: Fusões de 3 Modelos Baseline ($K = 3$)

*(Total de 21 estratégias avaliadas — ordenado estritamente por Val AUC decrescente)*

| Rank | Composição / Nome da Fusão | Modelos Componentes | Estratégia | Domínio | Val AUC | Val Acc | Test AUC | Test Acc | Test F1 | Test-D AUC | ΔAUC | DF-40 AUC | Celeb-DF Vídeo |
| :---: | :--- | :--- | :--- | :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| 1 | **Seleção Ótima no Val (3 modelos)** | `clip/none + dino/none + resnet/none` | `stacking` | `Espacial RGB` | **0.9987** | 98.62% | 0.9590 | 88.60% | 89.31% | **0.7570** | -0.2020 | **0.8519** | 0.7012 |
| 2 | **Top-3 Absoluto (Maiores Notas)** | `clip/none + dino/none + clip/concat` | `stacking` | `Híbrido RGB+Magnitude` | **0.9986** | 98.59% | 0.9597 | 88.82% | 89.50% | **0.7661** | -0.1936 | **0.8519** | 0.7012 |
| 3 | **Top-3 Diverso (ViT + ConvNeXt + CNN Fourier)** | `clip/none + dino/none + resnet/concat_frequency` | `stacking` | `Híbrido Espaço+Espectro` | **0.9986** | 98.66% | 0.9605 | 89.28% | 89.99% | **0.7705** | -0.1901 | **0.8654** | 0.7512 |
| 4 | **Top-3 Absoluto (Maiores Notas)** | `clip/none + dino/none + clip/concat` | `optimal_weights` | `Híbrido RGB+Magnitude` | **0.9985** | 97.86% | 0.9581 | 85.97% | 86.38% | **0.7521** | -0.2060 | **0.8519** | 0.7012 |
| 5 | **Top-3 Diverso (ViT + ConvNeXt + CNN Fourier)** | `clip/none + dino/none + resnet/concat_frequency` | `optimal_weights` | `Híbrido Espaço+Espectro` | **0.9985** | 97.86% | 0.9581 | 85.97% | 86.38% | **0.7521** | -0.2060 | **0.8654** | 0.7512 |
| 6 | **Seleção Ótima no Val (3 modelos)** | `clip/none + dino/none + resnet/none` | `optimal_weights` | `Espacial RGB` | **0.9985** | 97.86% | 0.9584 | 85.93% | 86.34% | **0.7511** | -0.2073 | **0.8519** | 0.7012 |
| 7 | **Seleção Ótima no Val (3 modelos)** | `clip/none + dino/none + resnet/none` | `mean` | `Espacial RGB` | **0.9983** | 98.07% | 0.9551 | 86.39% | 86.92% | **0.7483** | -0.2067 | **0.8519** | 0.7012 |
| 8 | **Seleção Ótima no Val (3 modelos)** | `clip/none + dino/none + resnet/none` | `weighted` | `Espacial RGB` | **0.9983** | 98.09% | 0.9551 | 86.39% | 86.91% | **0.7483** | -0.2068 | **0.8519** | 0.7012 |
| 9 | **Seleção Ótima no Val (3 modelos)** | `clip/none + dino/none + resnet/none` | `geometric` | `Espacial RGB` | **0.9983** | 96.78% | 0.9553 | 81.68% | 81.31% | **0.7498** | -0.2055 | **0.8519** | 0.7012 |
| 10 | **Top-3 Diverso (ViT + ConvNeXt + CNN Fourier)** | `clip/none + dino/none + resnet/concat_frequency` | `geometric` | `Híbrido Espaço+Espectro` | **0.9982** | 96.62% | 0.9601 | 84.11% | 84.17% | **0.7740** | -0.1861 | **0.8654** | 0.7512 |
| 11 | **Top-3 Absoluto (Maiores Notas)** | `clip/none + dino/none + clip/concat` | `geometric` | `Híbrido RGB+Magnitude` | **0.9981** | 97.07% | 0.9586 | 85.21% | 85.45% | **0.7667** | -0.1919 | **0.8519** | 0.7012 |
| 12 | **Top-3 Diverso (ViT + ConvNeXt + CNN Fourier)** | `clip/none + dino/none + resnet/concat_frequency` | `weighted` | `Híbrido Espaço+Espectro` | **0.9977** | 97.95% | 0.9578 | 89.07% | 89.80% | **0.7759** | -0.1819 | **0.8654** | 0.7512 |
| 13 | **Top-3 Diverso (ViT + ConvNeXt + CNN Fourier)** | `clip/none + dino/none + resnet/concat_frequency` | `mean` | `Híbrido Espaço+Espectro` | **0.9975** | 97.84% | 0.9570 | 89.03% | 89.78% | **0.7757** | -0.1813 | **0.8654** | 0.7512 |
| 14 | **Top-3 Absoluto (Maiores Notas)** | `clip/none + dino/none + clip/concat` | `weighted` | `Híbrido RGB+Magnitude` | **0.9972** | 97.36% | 0.9565 | 88.66% | 89.42% | **0.7707** | -0.1858 | **0.8519** | 0.7012 |
| 15 | **Top-3 Absoluto (Maiores Notas)** | `clip/none + dino/none + clip/concat` | `mean` | `Híbrido RGB+Magnitude` | **0.9971** | 97.30% | 0.9562 | 88.64% | 89.40% | **0.7708** | -0.1855 | **0.8519** | 0.7012 |
| 16 | **Seleção Ótima no Val (3 modelos)** | `clip/none + dino/none + resnet/none` | `max` | `Espacial RGB` | **0.9962** | 94.90% | 0.9487 | 88.54% | 90.11% | **0.7408** | -0.2079 | **0.8519** | 0.7012 |
| 17 | **Top-3 Absoluto (Maiores Notas)** | `clip/none + dino/none + clip/concat` | `max` | `Híbrido RGB+Magnitude` | **0.9911** | 92.33% | 0.9493 | 88.04% | 89.84% | **0.7632** | -0.1861 | **0.8519** | 0.7012 |
| 18 | **Top-3 Diverso (ViT + ConvNeXt + CNN Fourier)** | `clip/none + dino/none + resnet/concat_frequency` | `max` | `Híbrido Espaço+Espectro` | **0.9900** | 88.56% | 0.9442 | 84.37% | 87.41% | **0.7568** | -0.1874 | **0.8654** | 0.7512 |
| 19 | **Seleção Ótima no Val (3 modelos)** | `clip/none + dino/none + resnet/none` | `majority` | `Espacial RGB` | **0.9782** | 97.71% | 0.8720 | 85.87% | 86.39% | **0.6736** | -0.1984 | **0.8519** | 0.7012 |
| 20 | **Top-3 Diverso (ViT + ConvNeXt + CNN Fourier)** | `clip/none + dino/none + resnet/concat_frequency` | `majority` | `Híbrido Espaço+Espectro` | **0.9695** | 97.10% | 0.8898 | 88.12% | 88.92% | **0.6866** | -0.2032 | **0.8654** | 0.7512 |
| 21 | **Top-3 Absoluto (Maiores Notas)** | `clip/none + dino/none + clip/concat` | `majority` | `Híbrido RGB+Magnitude` | **0.9633** | 96.53% | 0.8890 | 88.05% | 88.87% | **0.6976** | -0.1914 | **0.8519** | 0.7012 |

---

## Tabela 2.4: Fusões de 4 Modelos Baseline ($K = 4$)

*(Total de 7 estratégias avaliadas — ordenado estritamente por Val AUC decrescente)*

| Rank | Composição / Nome da Fusão | Modelos Componentes | Estratégia | Domínio | Val AUC | Val Acc | Test AUC | Test Acc | Test F1 | Test-D AUC | ΔAUC | DF-40 AUC | Celeb-DF Vídeo |
| :---: | :--- | :--- | :--- | :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| 1 | **Top-4 Híbrido (ViT, ConvNeXt, FFTs)** | `clip/none + dino/none + clip/concat + resnet/concat_frequency` | `stacking` | `Híbrido Espaço+Espectro` | **0.9987** | 98.71% | 0.9614 | 89.49% | 90.20% | **0.7727** | -0.1886 | **0.8654** | 0.7512 |
| 2 | **Top-4 Híbrido (ViT, ConvNeXt, FFTs)** | `clip/none + dino/none + clip/concat + resnet/concat_frequency` | `optimal_weights` | `Híbrido Espaço+Espectro` | **0.9985** | 97.86% | 0.9581 | 85.96% | 86.38% | **0.7520** | -0.2061 | **0.8654** | 0.7512 |
| 3 | **Top-4 Híbrido (ViT, ConvNeXt, FFTs)** | `clip/none + dino/none + clip/concat + resnet/concat_frequency` | `geometric` | `Híbrido Espaço+Espectro` | **0.9979** | 96.48% | 0.9593 | 83.79% | 83.79% | **0.7735** | -0.1859 | **0.8654** | 0.7512 |
| 4 | **Top-4 Híbrido (ViT, ConvNeXt, FFTs)** | `clip/none + dino/none + clip/concat + resnet/concat_frequency` | `weighted` | `Híbrido Espaço+Espectro` | **0.9972** | 97.53% | 0.9582 | 89.45% | 90.23% | **0.7799** | -0.1783 | **0.8654** | 0.7512 |
| 5 | **Top-4 Híbrido (ViT, ConvNeXt, FFTs)** | `clip/none + dino/none + clip/concat + resnet/concat_frequency` | `mean` | `Híbrido Espaço+Espectro` | **0.9970** | 97.45% | 0.9578 | 89.45% | 90.24% | **0.7800** | -0.1778 | **0.8654** | 0.7512 |
| 6 | **Top-4 Híbrido (ViT, ConvNeXt, FFTs)** | `clip/none + dino/none + clip/concat + resnet/concat_frequency` | `max` | `Híbrido Espaço+Espectro` | **0.9882** | 86.13% | 0.9456 | 82.91% | 86.51% | **0.7498** | -0.1959 | **0.8654** | 0.7512 |
| 7 | **Top-4 Híbrido (ViT, ConvNeXt, FFTs)** | `clip/none + dino/none + clip/concat + resnet/concat_frequency` | `majority` | `Híbrido Espaço+Espectro` | **0.9478** | 95.59% | 0.8957 | 89.48% | 90.65% | **0.6770** | -0.2188 | **0.8654** | 0.7512 |

---

## Tabela 2.5: Fusões de 5 Modelos Baseline ($K = 5$)

*(Total de 12 estratégias avaliadas — ordenado estritamente por Val AUC decrescente)*

| Rank | Composição / Nome da Fusão | Modelos Componentes | Estratégia | Domínio | Val AUC | Val Acc | Test AUC | Test Acc | Test F1 | Test-D AUC | ΔAUC | DF-40 AUC | Celeb-DF Vídeo |
| :---: | :--- | :--- | :--- | :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| 1 | **Top-5 Completo** | `clip/none + dino/none + clip/concat + vit/concat + resnet/concat_frequency` | `stacking` | `Híbrido Espaço+Espectro` | **0.9987** | 98.69% | 0.9614 | 89.46% | 90.17% | **0.7734** | -0.1880 | **0.8654** | 0.7512 |
| 2 | **Top-5 Completo** | `clip/none + dino/none + clip/concat + vit/concat + resnet/concat_frequency` | `optimal_weights` | `Híbrido Espaço+Espectro` | **0.9985** | 97.86% | 0.9581 | 85.96% | 86.38% | **0.7520** | -0.2061 | **0.8654** | 0.7512 |
| 3 | **Top-5 Completo** | `clip/none + dino/none + clip/concat + vit/concat + resnet/concat_frequency` | `geometric` | `Híbrido Espaço+Espectro` | **0.9973** | 95.20% | 0.9551 | 80.89% | 80.29% | **0.7717** | -0.1834 | **0.8654** | 0.7512 |
| 4 | **Top-5 Completo** | `clip/none + dino/none + clip/concat + vit/concat + resnet/concat_frequency` | `weighted` | `Híbrido Espaço+Espectro` | **0.9966** | 97.30% | 0.9552 | 88.23% | 88.97% | **0.7796** | -0.1756 | **0.8654** | 0.7512 |
| 5 | **Top-5 Completo** | `clip/none + dino/none + clip/concat + vit/concat + resnet/concat_frequency` | `mean` | `Híbrido Espaço+Espectro` | **0.9964** | 97.19% | 0.9544 | 88.10% | 88.84% | **0.7794** | -0.1750 | **0.8654** | 0.7512 |
| 6 | **Top-5 Completo** | `clip/none + dino/none + clip/concat + vit/concat + resnet/concat_frequency` | `max` | `Híbrido Espaço+Espectro` | **0.9881** | 84.68% | 0.9448 | 81.67% | 85.71% | **0.7502** | -0.1946 | **0.8654** | 0.7512 |
| 7 | **Deep Ensemble: CLIP Padrão (5 Seeds)** | `clip/none/{42, 123, 2024, 7, 2025}` | `geometric` | `Espacial RGB (Multi-Seed)` | **0.9865** | 94.18% | 0.9345 | 85.13% | 85.92% | **0.7654** | -0.1692 | **0.7943** | 0.3150 |
| 8 | **Deep Ensemble: CLIP Padrão (5 Seeds)** | `clip/none/{42, 123, 2024, 7, 2025}` | `stacking` | `Espacial RGB (Multi-Seed)` | **0.9862** | 94.26% | 0.9343 | 85.19% | 85.97% | **0.7651** | -0.1691 | **0.7840** | 0.3150 |
| 9 | **Deep Ensemble: CLIP Padrão (5 Seeds)** | `clip/none/{42, 123, 2024, 7, 2025}` | `weighted` | `Espacial RGB (Multi-Seed)` | **0.9861** | 94.30% | 0.9342 | 85.07% | 85.83% | **0.7654** | -0.1687 | **0.7840** | 0.3150 |
| 10 | **Deep Ensemble: CLIP Padrão (5 Seeds)** | `clip/none/{42, 123, 2024, 7, 2025}` | `mean` | `Espacial RGB (Multi-Seed)` | **0.9861** | 94.30% | 0.9342 | 85.07% | 85.83% | **0.7654** | -0.1687 | **0.7840** | 0.3150 |
| 11 | **Deep Ensemble: CLIP Padrão (5 Seeds)** | `clip/none/{42, 123, 2024, 7, 2025}` | `max` | `Espacial RGB (Multi-Seed)` | **0.9837** | 93.35% | 0.9303 | 85.59% | 86.85% | **0.7473** | -0.1830 | **0.7840** | 0.3150 |
| 12 | **Top-5 Completo** | `clip/none + dino/none + clip/concat + vit/concat + resnet/concat_frequency` | `majority` | `Híbrido Espaço+Espectro` | **0.9638** | 96.49% | 0.8821 | 87.26% | 88.03% | **0.6958** | -0.1863 | **0.8654** | 0.7512 |

---

## Tabela 2.6: Fusões de 6 Modelos Baseline ($K = 6$)

*(Total de 10 estratégias avaliadas — ordenado estritamente por Val AUC decrescente)*

| Rank | Composição / Nome da Fusão | Modelos Componentes | Estratégia | Domínio | Val AUC | Val Acc | Test AUC | Test Acc | Test F1 | Test-D AUC | ΔAUC | DF-40 AUC | Celeb-DF Vídeo |
| :---: | :--- | :--- | :--- | :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| 1 | **TODOS OS 6 PADRÃO (Baseline Espacial)** | `clip/none + dino/none + vit/none + resnet/none + mobilenet/none + xception/none` | `stacking` | `Espacial RGB (6 Famílias)` | **0.9970** | 97.91% | 0.9451 | 86.77% | 87.42% | **0.7322** | -0.2129 | **0.7575** | 0.3600 |
| 2 | **TODOS OS 6 PADRÃO (Baseline Espacial)** | `clip/none + dino/none + vit/none + resnet/none + mobilenet/none + xception/none` | `geometric` | `Espacial RGB (6 Famílias)` | **0.9951** | 89.44% | 0.9407 | 70.49% | 65.64% | **0.7260** | -0.2148 | **0.7601** | 0.3600 |
| 3 | **TODOS OS 6 PADRÃO (Baseline Espacial)** | `clip/none + dino/none + vit/none + resnet/none + mobilenet/none + xception/none` | `weighted` | `Espacial RGB (6 Famílias)` | **0.9945** | 96.33% | 0.9325 | 82.74% | 82.96% | **0.7245** | -0.2079 | **0.7612** | 0.3600 |
| 4 | **TODOS OS 6 PADRÃO (Baseline Espacial)** | `clip/none + dino/none + vit/none + resnet/none + mobilenet/none + xception/none` | `mean` | `Espacial RGB (6 Famílias)` | **0.9942** | 96.20% | 0.9312 | 82.53% | 82.75% | **0.7236** | -0.2076 | **0.7615** | 0.3600 |
| 5 | **TODOS OS 6 CONCAT (RGB + Magnitude)** | `clip/concat + dino/concat + vit/concat + resnet/concat + mobilenet/concat + xception/concat` | `stacking` | `Híbrido RGB+Magnitude (6 Famílias)` | **0.9942** | 96.89% | 0.9361 | 85.36% | 85.99% | **0.7149** | -0.2212 | **0.7687** | 0.4500 |
| 6 | **TODOS OS 6 PADRÃO (Baseline Espacial)** | `clip/none + dino/none + vit/none + resnet/none + mobilenet/none + xception/none` | `max` | `Espacial RGB (6 Famílias)` | **0.9908** | 84.80% | 0.9245 | 79.98% | 84.51% | **0.7214** | -0.2031 | **0.7516** | 0.3600 |
| 7 | **TODOS OS 6 CONCAT (RGB + Magnitude)** | `clip/concat + dino/concat + vit/concat + resnet/concat + mobilenet/concat + xception/concat` | `weighted` | `Híbrido RGB+Magnitude (6 Famílias)` | **0.9889** | 95.00% | 0.9192 | 82.78% | 83.42% | **0.7079** | -0.2113 | **0.7788** | 0.4500 |
| 8 | **TODOS OS 6 CONCAT (RGB + Magnitude)** | `clip/concat + dino/concat + vit/concat + resnet/concat + mobilenet/concat + xception/concat` | `geometric` | `Híbrido RGB+Magnitude (6 Famílias)` | **0.9887** | 86.93% | 0.9186 | 70.47% | 65.69% | **0.6961** | -0.2225 | **0.7759** | 0.4500 |
| 9 | **TODOS OS 6 CONCAT (RGB + Magnitude)** | `clip/concat + dino/concat + vit/concat + resnet/concat + mobilenet/concat + xception/concat` | `mean` | `Híbrido RGB+Magnitude (6 Famílias)` | **0.9878** | 94.61% | 0.9164 | 82.49% | 83.21% | **0.7063** | -0.2101 | **0.7795** | 0.4500 |
| 10 | **TODOS OS 6 CONCAT (RGB + Magnitude)** | `clip/concat + dino/concat + vit/concat + resnet/concat + mobilenet/concat + xception/concat` | `max` | `Híbrido RGB+Magnitude (6 Famílias)` | **0.9478** | 77.60% | 0.8690 | 73.20% | 80.55% | **0.6712** | -0.1978 | **0.7560** | 0.4500 |

---

## 📌 Síntese e Comparativo Entre Níveis de Fusão Baseline

| Ordem ($K$) | Melhor Composição no Validação | Estratégia Campeã | Val AUC | Test AUC (FF++) | Test-D AUC (Corrompido) | ΔAUC | DF-40 AUC |
| :---: | :--- | :---: | :---: | :---: | :---: | :---: | :---: |
| **K = 2** | CLIP + DINO (Baseline Espacial) | `geometric` | **0.9968** | 0.9495 | **0.7341** | -0.2154 | 0.7552 |
| **K = 3** | Seleção Ótima no Val (3 modelos) | `stacking` | **0.9987** | 0.9590 | **0.7570** | -0.2020 | 0.8519 |
| **K = 4** | Top-4 Híbrido (ViT, ConvNeXt, FFTs) | `stacking` | **0.9987** | 0.9614 | **0.7727** | -0.1886 | 0.8654 |
| **K = 5** | Top-5 Completo | `stacking` | **0.9987** | 0.9614 | **0.7734** | -0.1880 | 0.8654 |
| **K = 6** | TODOS OS 6 PADRÃO (Baseline Espacial) | `stacking` | **0.9970** | 0.9451 | **0.7322** | -0.2129 | 0.7575 |

> [!TIP]
> **Destaques Forenses dos Ensembles Baseline:**
> 1. **Fusões Híbridas Superam Espaciais Puras:** Em $K = 3$ e $K = 4$, a integração de canais espectrais (`resnet/concat_frequency`) eleva o desempenho em dados corrompidos (`test_d`) de 0.7570 para **0.7727 de AUC**, e no benchmark cross-dataset DF-40 de 0.8519 para **0.8654 de AUC**.
> 2. **Stacking e Média Geométrica:** Em quase todos os patamares, `stacking` e `geometric` lideram o ranking de `Val AUC`. A média geométrica é particularmente eficaz ao punir predições onde um dos modelos apresenta alta incerteza.
> 3. **Escalabilidade com $K$:** O aumento de $K=2$ para $K=4$ e $K=5$ produz saltos de mais de **+4 pp em Test-D AUC** frente a qualquer modelo baseline isolado.
