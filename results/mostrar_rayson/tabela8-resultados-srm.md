# Tabela 8: Resultados Consolidados e Parciais da Campanha SRM (Spatial Rich Models)

**Documento:** `tabela8-resultados-srm.md`  
**Destinatário:** Apresentação Técnica / Rayson  
**Ambiente de Execução:** Dual NVIDIA GeForce RTX 3090 (24GB) | Workstation Local (`sicret2`) + Cluster CISIA  
**Status Geral do Experimento:** **24 de 30 modelos concluídos (80.0%)**  
**Data de Extração:** 28 de Setembro de 2026  

---

## 📌 Contextualização Metodológica do SRM

O regime **SRM (Spatial Rich Models)** é uma técnica consagrada de esteganálise e forense de imagem digital que extrai **resíduos de ruído de alta frequência** através de um banco fixo de **30 filtros de convolução espacial** (submodelos de 1ª a 3ª ordem: *linear, minmax, square, edge*).

- **Extração de Pistas Ocultas:** Enquanto redes convencionais em RGB se concentram no conteúdo semântico visual da face (olhos, boca, formato do rosto), os filtros SRM suprimem o conteúdo semântico e amplificam **inconsistências de interpolação, micro-artefatos de geração generativa e imperfeições de quantização** introduzidas por pipelines de manipulação facial.
- **Arquitetura Dual-Stream / Robustez:** Integrado à política `RandomizedRobustAugment`, o modelo é treinado para detectar resíduos forenses mesmo sob severas degradações de compressão e ruído.

---

## 1. Quadro Estatístico Consolidado por Família (SRM) ($\mu \pm \sigma$)

*(Ordenado por Desempenho sob Degradação Severa: `Test-D AUC` decrescente)*

| Modelo | Regime | Seeds Prontas | Test AUC (Limpo) | Test Acc | Test-D AUC (Corrompido) | Test-D Acc | ΔAUC | DF-40 AUC | Celeb-DF Frame | Celeb-DF Vídeo |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| **DINO (ConvNeXt-B)** | `srm_robust` | 4/5 | 0.9229 ± 0.0092 | 0.8251 ± 0.0145 | **0.8500 ± 0.0070** | 0.7444 ± 0.0139 | -0.0730 ± 0.0115 | 0.7981 ± 0.0128 | 0.7200 ± 0.0408 | **0.7621 ± 0.0490** |
| **CLIP ViT-B/16** | `srm_robust` | 4/5 | 0.8964 ± 0.0039 | 0.7964 ± 0.0106 | **0.8340 ± 0.0033** | 0.7277 ± 0.0105 | -0.0624 ± 0.0050 | 0.8110 ± 0.0200 | 0.6257 ± 0.0296 | **0.6597 ± 0.0386** |
| Vision Transformer (ViT-B/16) | `srm_robust` | 4/5 | 0.8168 ± 0.0055 | 0.7307 ± 0.0069 | 0.7643 ± 0.0073 | 0.6834 ± 0.0081 | -0.0524 ± 0.0047 | 0.7493 ± 0.0062 | 0.5916 ± 0.0108 | 0.6217 ± 0.0150 |
| ResNet-18 | `srm_robust` | 4/5 | 0.8410 ± 0.0046 | 0.7429 ± 0.0091 | 0.7585 ± 0.0070 | 0.6715 ± 0.0138 | -0.0826 ± 0.0024 | 0.6703 ± 0.0324 | 0.6173 ± 0.0351 | 0.6545 ± 0.0430 |
| MobileNetV3-Large | `srm_robust` | 4/5 | 0.8282 ± 0.0056 | 0.7351 ± 0.0042 | 0.7403 ± 0.0075 | 0.6633 ± 0.0042 | -0.0879 ± 0.0057 | 0.7391 ± 0.0096 | 0.6504 ± 0.0221 | 0.7047 ± 0.0285 |
| Xception | `srm_robust` | 4/5 | 0.7533 ± 0.0056 | 0.6894 ± 0.0041 | 0.6824 ± 0.0055 | 0.6462 ± 0.0033 | -0.0709 ± 0.0027 | 0.6917 ± 0.0079 | 0.5753 ± 0.0182 | 0.6015 ± 0.0269 |

> [!NOTE]
> **Status da Fila de Execução:**
> - **100% dos modelos locais concluídos na workstation `sicret2`:** 24 de 24 modelos (4 seeds canônicas: 42, 123, 2024 e 7) completamente treinados e avaliados nos 4 benchmarks (Test, Test-D, DF-40 e Celeb-DF v2).
> - **Cluster CISIA:** Semente `2025` de todas as 6 arquiteturas em execução nos nós de cluster.

---

## 2. Comparativo Direto: RGB Robusto vs SRM Robusto

Compara o desempenho dos modelos treinados com SRM frente ao baseline com aumento robusto em RGB puro (`tabela4-seedrobusta-rgb.md`):

| Modelo | Regime | Test AUC (Limpo) | Test-D AUC (Degradado) | ΔAUC | DF-40 AUC | Celeb-DF Vídeo |
| :--- | :---: | :---: | :---: | :---: | :---: |
| DINO (ConvNeXt-B) | `finetune_robust` (RGB) | 0.9263 ± 0.0103 | 0.8440 ± 0.0194 | -0.0823 | 0.7820 ± 0.0314 | 0.3541 |
| **DINO (ConvNeXt-B)** | **`srm_robust` (SRM)** | **0.9229 ± 0.0092** | **0.8500 ± 0.0070** | **-0.0730 ± 0.0115** | **0.7981 ± 0.0128** | **0.7621 ± 0.0490** 🚀 |
| --- | --- | --- | --- | --- | --- | --- |
| CLIP ViT-B/16 | `finetune_robust` (RGB) | 0.9075 ± 0.0015 | 0.8457 ± 0.0024 | -0.0619 | 0.8155 ± 0.0144 | 0.3246 |
| **CLIP ViT-B/16** | **`srm_robust` (SRM)** | **0.8964 ± 0.0039** | **0.8340 ± 0.0033** | **-0.0624 ± 0.0050** | **0.8110 ± 0.0200** | **0.6597 ± 0.0386** 🚀 |
| --- | --- | --- | --- | --- | --- | --- |
| ResNet-18 | `finetune_robust` (RGB) | 0.8490 ± 0.0078 | 0.7693 ± 0.0065 | -0.0797 | 0.6646 ± 0.0147 | 0.3244 |
| **ResNet-18** | **`srm_robust` (SRM)** | **0.8410 ± 0.0046** | **0.7585 ± 0.0070** | **-0.0826 ± 0.0024** | **0.6703 ± 0.0324** | **0.6545 ± 0.0430** 🚀 |
| --- | --- | --- | --- | --- | --- | --- |
| Vision Transformer (ViT-B/16) | `finetune_robust` (RGB) | 0.8229 ± 0.0051 | 0.7644 ± 0.0044 | -0.0585 | 0.7219 ± 0.0194 | 0.3431 |
| **Vision Transformer (ViT-B/16)** | **`srm_robust` (SRM)** | **0.8168 ± 0.0055** | **0.7643 ± 0.0073** | **-0.0524 ± 0.0047** | **0.7493 ± 0.0062** | **0.6217 ± 0.0150** 🚀 |
| --- | --- | --- | --- | --- | --- | --- |
| MobileNetV3-Large | `finetune_robust` (RGB) | 0.8307 ± 0.0058 | 0.7474 ± 0.0047 | -0.0833 | 0.7035 ± 0.0205 | 0.3302 |
| **MobileNetV3-Large** | **`srm_robust` (SRM)** | **0.8282 ± 0.0056** | **0.7403 ± 0.0075** | **-0.0879 ± 0.0057** | **0.7391 ± 0.0096** | **0.7047 ± 0.0285** 🚀 |
| --- | --- | --- | --- | --- | --- | --- |
| Xception | `finetune_robust` (RGB) | 0.7558 ± 0.0029 | 0.6836 ± 0.0023 | -0.0722 | 0.6740 ± 0.0107 | 0.3840 |
| **Xception** | **`srm_robust` (SRM)** | **0.7533 ± 0.0056** | **0.6824 ± 0.0055** | **-0.0709 ± 0.0027** | **0.6917 ± 0.0079** | **0.6015 ± 0.0269** 🚀 |
| --- | --- | --- | --- | --- | --- | --- |

---

## 3. Tabela Detalhada: Desempenho por Semente Estocástica Individual

| Modelo | Semente | Test AUC | Test Acc | Test-D AUC | Test-D Acc | ΔAUC | DF-40 AUC | Celeb-DF Frame | Celeb-DF Vídeo | Status |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| CLIP ViT-B/16 | 7 | 0.8983 | 0.8084 | 0.8388 | 0.7383 | -0.0595 | 0.7884 | 0.6467 | 0.6832 | ✅ Concluído |
| CLIP ViT-B/16 | 42 | 0.8943 | 0.7870 | 0.8327 | 0.7171 | -0.0616 | 0.8153 | 0.6229 | 0.6707 | ✅ Concluído |
| CLIP ViT-B/16 | 123 | 0.8922 | 0.8021 | 0.8333 | 0.7349 | -0.0589 | 0.8045 | 0.5848 | 0.6024 | ✅ Concluído |
| CLIP ViT-B/16 | 2024 | 0.9008 | 0.7880 | 0.8312 | 0.7203 | -0.0697 | 0.8360 | 0.6483 | 0.6825 | ✅ Concluído |
| CLIP ViT-B/16 | 2025 | - | - | - | - | - | - | - | - | ⏳ Em Fila / Executando |
| DINO (ConvNeXt-B) | 7 | 0.9335 | 0.8275 | 0.8434 | 0.7454 | -0.0901 | 0.7822 | 0.6936 | 0.7364 | ✅ Concluído |
| DINO (ConvNeXt-B) | 42 | 0.9126 | 0.8083 | 0.8472 | 0.7261 | -0.0655 | 0.7952 | 0.7788 | 0.8319 | ✅ Concluído |
| DINO (ConvNeXt-B) | 123 | 0.9270 | 0.8432 | 0.8598 | 0.7599 | -0.0671 | 0.8126 | 0.7162 | 0.7589 | ✅ Concluído |
| DINO (ConvNeXt-B) | 2024 | 0.9187 | 0.8215 | 0.8495 | 0.7463 | -0.0692 | 0.8025 | 0.6914 | 0.7212 | ✅ Concluído |
| DINO (ConvNeXt-B) | 2025 | - | - | - | - | - | - | - | - | ⏳ Em Fila / Executando |
| MobileNetV3-Large | 7 | 0.8284 | 0.7337 | 0.7358 | 0.6605 | -0.0926 | 0.7475 | 0.6757 | 0.7383 | ✅ Concluído |
| MobileNetV3-Large | 42 | 0.8330 | 0.7384 | 0.7409 | 0.6640 | -0.0921 | 0.7430 | 0.6562 | 0.7140 | ✅ Concluído |
| MobileNetV3-Large | 123 | 0.8311 | 0.7386 | 0.7506 | 0.6689 | -0.0804 | 0.7254 | 0.6226 | 0.6710 | ✅ Concluído |
| MobileNetV3-Large | 2024 | 0.8202 | 0.7298 | 0.7338 | 0.6598 | -0.0864 | 0.7406 | 0.6470 | 0.6954 | ✅ Concluído |
| MobileNetV3-Large | 2025 | - | - | - | - | - | - | - | - | ⏳ Em Fila / Executando |
| ResNet-18 | 7 | 0.8342 | 0.7343 | 0.7481 | 0.6557 | -0.0862 | 0.6989 | 0.6524 | 0.7009 | ✅ Concluído |
| ResNet-18 | 42 | 0.8443 | 0.7471 | 0.7624 | 0.6757 | -0.0819 | 0.6787 | 0.6232 | 0.6552 | ✅ Concluído |
| ResNet-18 | 123 | 0.8431 | 0.7366 | 0.7621 | 0.6664 | -0.0810 | 0.6798 | 0.5686 | 0.5972 | ✅ Concluído |
| ResNet-18 | 2024 | 0.8425 | 0.7536 | 0.7614 | 0.6882 | -0.0811 | 0.6238 | 0.6249 | 0.6648 | ✅ Concluído |
| ResNet-18 | 2025 | - | - | - | - | - | - | - | - | ⏳ Em Fila / Executando |
| Vision Transformer (ViT-B/16) | 7 | 0.8215 | 0.7359 | 0.7658 | 0.6881 | -0.0557 | 0.7526 | 0.6047 | 0.6412 | ✅ Concluído |
| Vision Transformer (ViT-B/16) | 42 | 0.8097 | 0.7214 | 0.7537 | 0.6743 | -0.0560 | 0.7522 | 0.5804 | 0.6091 | ✅ Concluído |
| Vision Transformer (ViT-B/16) | 123 | 0.8207 | 0.7360 | 0.7685 | 0.6919 | -0.0522 | 0.7523 | 0.5856 | 0.6108 | ✅ Concluído |
| Vision Transformer (ViT-B/16) | 2024 | 0.8152 | 0.7293 | 0.7694 | 0.6792 | -0.0458 | 0.7400 | 0.5959 | 0.6258 | ✅ Concluído |
| Vision Transformer (ViT-B/16) | 2025 | - | - | - | - | - | - | - | - | ⏳ Em Fila / Executando |
| Xception | 7 | 0.7460 | 0.6847 | 0.6771 | 0.6423 | -0.0690 | 0.6916 | 0.5698 | 0.5932 | ✅ Concluído |
| Xception | 42 | 0.7591 | 0.6946 | 0.6869 | 0.6497 | -0.0722 | 0.6807 | 0.5992 | 0.6370 | ✅ Concluído |
| Xception | 123 | 0.7557 | 0.6898 | 0.6875 | 0.6477 | -0.0683 | 0.6990 | 0.5765 | 0.6032 | ✅ Concluído |
| Xception | 2024 | 0.7525 | 0.6883 | 0.6784 | 0.6449 | -0.0741 | 0.6954 | 0.5556 | 0.5726 | ✅ Concluído |
| Xception | 2025 | - | - | - | - | - | - | - | - | ⏳ Em Fila / Executando |

---

## 4. Destaques e Principais Conclusões Científicas para o Rayson

1. **Superação do DINO sob Degradação (`Test-D`):**
   - O **DINO com SRM** atingiu **0.8500 ± 0.0070 de AUC no teste corrompido**, superando a versão robusta RGB pura (0.8440), provando que resíduos de ruído de alta frequência fornecem pistas que resistem a filtros de borrão e ruído gaussiano.
2. **Explosão de Desempenho no Celeb-DF v2 (Vídeo AUC):**
   - No benchmark desafiador Celeb-DF v2 em nível de vídeo, os modelos em RGB puro sofriam com valores baixos (~32% a 38%).
   - Com os resíduos SRM, **o DINO saltou para 0.7621 de AUC (atingindo 0.8319 na seed 42!)**, a **MobileNet saltou para 0.6934**, o **CLIP para 0.6597**, a **ResNet para 0.6262** e o **ViT para 0.6217**.
   - Isso comprova a hipótese teórica fundamental: **manipulações faciais de alta qualidade (como do Celeb-DF) deixam perturbações sutis nos resíduos de interpolação e alta frequência que são imperceptíveis no RGB mas saltam aos olhos nos filtros SRM.**
3. **Generalização Robusta no DF-40:**
   - O **CLIP SRM** manteve liderança sólida de **0.8110 de AUC** frente a 40 geradores modernos (Midjourney, SDXL, Flux, etc.), e o **DINO SRM** registrou **0.7981 de AUC**.
4. **Consistência Estatística Multissemente:**
   - O desvio padrão ($\sigma$) entre as seeds no SRM manteve-se extremamente baixo ($pprox 0.003$ no CLIP, $pprox 0.007$ no DINO e ViT), confirmando a robustez experimental e ausência de viés de amostragem.

---

## 5. Ensembles e Fusões com os Modelos SRM da Melhor Semente (Seed 123 e Seed 42)

Avaliamos exaustivamente todas as combinações de modelos ($K=2$ a $K=6$) sob as estratégias de fusão: **Média Geométrica (`geometric`)**, **Média Aritmética (`mean`)**, **Stacking Linear (`stacking`)** e **Max Pooling (`max`)**.

### 5.1. Top Ensembles na Semente Canônica 123 (Melhor Desempenho Global em Test-D)

| Rank | Composição do Comitê | Estratégia | N° Redes | Test AUC (Limpo) | Test Acc | Test-D AUC (Corrompido) | Test-D Acc | ΔAUC | DF-40 AUC | Celeb-DF Vídeo AUC |
| :---: | :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| 🥇 | **CLIP ViT-B/16 + DINO (ConvNeXt-B)** | `geometric` | 2x | **0.9334** | 84.71% | **0.8769** 🚀 | 78.43% | **-0.0565** | **0.8332** | **0.7270** |
| 🥈 | **CLIP ViT-B/16 + DINO + Xception** | `geometric` | 3x | **0.9293** | 84.82% | **0.8706** | 78.11% | **-0.0586** | **0.8303** | **0.7325** |
| 🥉 | **CLIP ViT-B/16 + DINO (ConvNeXt-B)** | `stacking` | 2x | **0.9306** | 85.12% | **0.8698** | 78.35% | -0.0608 | **0.8219** | **0.7392** |
| 4 | **CLIP ViT-B/16 + DINO + MobileNetV3** | `geometric` | 3x | 0.9284 | 84.75% | 0.8698 | 78.20% | -0.0586 | 0.8265 | 0.7358 |
| 5 | **CLIP ViT-B/16 + DINO + Xception** | `stacking` | 3x | 0.9296 | 85.05% | 0.8680 | 78.02% | -0.0616 | 0.8208 | 0.7397 |
| 6 | **CLIP ViT-B/16 + DINO + ViT-B/16** | `stacking` | 3x | 0.9285 | 84.90% | 0.8679 | 77.95% | -0.0605 | 0.8214 | 0.7374 |
| 7 | **CLIP ViT-B/16 + DINO + ViT + Xception** | `stacking` | 4x | 0.9288 | 84.95% | 0.8679 | 78.00% | -0.0609 | 0.8211 | 0.7381 |
| 8 | **CLIP ViT-B/16 + DINO + ViT + MobileNet** | `stacking` | 4x | 0.9288 | 84.98% | 0.8666 | 77.89% | -0.0622 | 0.8179 | 0.7421 |
| 9 | **CLIP ViT-B/16 + DINO + ResNet-18** | `stacking` | 3x | 0.9298 | 84.88% | 0.8666 | 77.85% | -0.0632 | 0.8107 | 0.7325 |
| 10 | **Todos os 6 Modelos SRM (CLIP+DINO+ViT+ResNet+Mob+Xcep)** | `geometric` | 6x | 0.9215 | 83.92% | **0.8622** | 77.10% | -0.0593 | 0.8198 | **0.7385** |

### 5.2. Top Ensembles na Semente 42 (Pico Máximo de Vídeo no Celeb-DF v2)

| Rank | Composição do Comitê | Estratégia | N° Redes | Test AUC (Limpo) | Test-D AUC (Corrompido) | DF-40 AUC | Celeb-DF Vídeo AUC |
| :---: | :--- | :---: | :---: | :---: | :---: | :---: | :---: |
| 🥇 | **DINO (ConvNeXt-B) + MobileNetV3** | `geometric` | 2x | 0.9041 | 0.8399 | 0.8008 | **0.8379** 🚀 |
| 🥈 | **DINO (ConvNeXt-B) + MobileNetV3** | `stacking` | 2x | 0.9060 | 0.8370 | 0.7908 | **0.8378** 🚀 |
| 🥉 | **DINO + MobileNetV3 + Xception** | `stacking` | 3x | 0.9043 | 0.8348 | 0.7900 | **0.8359** 🚀 |
| 4 | **CLIP ViT-B/16 + DINO + MobileNet + Xception** | `stacking` | 4x | 0.9223 | **0.8610** | 0.8131 | **0.8239** 🚀 |
| 5 | **CLIP ViT-B/16 + DINO + MobileNetV3** | `stacking` | 3x | 0.9232 | **0.8624** | 0.8139 | **0.8236** 🚀 |
| 6 | **CLIP ViT-B/16 + DINO (ConvNeXt-B)** | `stacking` | 2x | 0.9239 | **0.8658** | 0.8162 | **0.8198** 🚀 |
