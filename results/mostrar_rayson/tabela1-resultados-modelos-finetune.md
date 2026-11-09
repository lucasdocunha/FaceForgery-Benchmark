# Tabela 1: Resultados Consolidados de Modelos e Frequências (Fine-Tuning Padrão)

**Documento:** `tabela1-resultados-modelos-finetune.md`  
**Destinatário:** Apresentação Técnica / Rayson  
**Ambiente de Execução:** Dual NVIDIA GeForce RTX 3090 (24GB) | Workstation Local (`sicret2`)  
**Regime de Treino Avaliado:** `finetune` (Baseline com aumento padrão, 5 a 6 sementes por modelo)  
**Data de Extração:** 16 de Setembro de 2026  

> [!NOTE]
> **Aviso Metodológico:** Esta tabela contém **exclusivamente** os resultados do regime padrão (`finetune`) para todas as 7 representações espectrais de Fourier em todos os 6 modelos (42 pares avaliados), sem ensembles. Os modelos submetidos ao regime com perturbações estocásticas pesadas (`finetune_robust` / `RandomizedRobustAugment`) foram consolidados e detalhados na [**Tabela 4 (`tabela4-seedrobusta-rgb.md`)**](file:///home/lucas.ocunha/tcc/results/mostrar_rayson/tabela4-seedrobusta-rgb.md).

---

## 📌 Descrição dos Benchmarks Avaliados

Todas as métricas reportam a **média e desvio padrão ($\mu \pm \sigma$)** entre as sementes estatísticas avaliadas (5x a 6x seeds):

1. **FaceForensics++ Teste Limpo (`test`)**: Imagens não-vistas com distribuição idêntica ao treino (FF++ c23).
2. **FaceForensics++ Teste Corrompido (`test_d`)**: Avaliação sob 14 degradações severas não-vistas (compressão JPEG agressiva, blur Gaussiano, ruído impulsivo, desfoque de movimento, etc.).
3. **Taxa de Degradação ($\Delta\text{AUC}$)**: $\text{AUC}_{\text{test\_d}} - \text{AUC}_{\text{test}}$. Quanto mais próximo de zero, menor é a perda de eficácia sob corrupção.
4. **DeepFake-40 (`DF-40`)**: Generalização *Zero-Shot Out-of-Distribution* em 11.150 imagens de 40 geradores modernos (Difusão, GANs, FaceSwap e Comerciais).
5. **Celeb-DF v2**: Generalização *Zero-Shot* em vídeos de alta resolução em nível de frame e nível de vídeo agregado.

---

## 1. Quadro Geral Consolidado: Modelos Baseline Finetune (42 Combinações)

*(Ordenado estritamente por Desempenho sob Corrupção: `Test-D AUC` decrescente)*

| Rank | Modelo | Modo Fourier | Regime | Seeds | Test AUC (FF++) | Test-D AUC (Corrompido) | ΔAUC | DF-40 AUC | Celeb-DF Frame | Celeb-DF Vídeo |
| :---: | :--- | :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| 1 | **CLIP ViT-B/16** | `none` | `finetune` | 6x | 0.9173 ± 0.0064 | **0.7651 ± 0.0408** | -0.1522 ± 0.0456 | 0.7639 ± 0.0460 | 0.3324 ± 0.0310 | 0.2835 ± 0.0345 |
| 2 | **DINO (ConvNeXt-B)** | `none` | `finetune` | 6x | 0.9339 ± 0.0073 | **0.7370 ± 0.0555** | -0.1969 ± 0.0599 | 0.7208 ± 0.0289 | 0.3355 ± 0.0600 | 0.2863 ± 0.0749 |
| 3 | **CLIP ViT-B/16** | `concat` | `finetune` | 5x | 0.8941 ± 0.0042 | **0.7260 ± 0.0119** | -0.1681 ± 0.0147 | 0.7650 ± 0.0323 | 0.3948 ± 0.0195 | 0.3648 ± 0.0254 |
| 4 | **Vision Transformer (ViT-B/16)** | `concat` | `finetune` | 5x | 0.8265 ± 0.0035 | **0.7123 ± 0.0056** | -0.1142 ± 0.0036 | 0.6324 ± 0.0237 | 0.4519 ± 0.0146 | 0.4126 ± 0.0210 |
| 5 | **Vision Transformer (ViT-B/16)** | `none` | `finetune` | 6x | 0.8149 ± 0.0085 | **0.7036 ± 0.0284** | -0.1112 ± 0.0271 | 0.7040 ± 0.0157 | 0.3521 ± 0.0370 | 0.3062 ± 0.0425 |
| 6 | **ResNet-18** | `concat_frequency` | `finetune` | 5x | 0.8337 ± 0.0072 | **0.6866 ± 0.0103** | -0.1472 ± 0.0048 | 0.7071 ± 0.0316 | 0.4344 ± 0.0231 | 0.3860 ± 0.0186 |
| 7 | **DINO (ConvNeXt-B)** | `concat` | `finetune` | 5x | 0.9166 ± 0.0070 | **0.6834 ± 0.0102** | -0.2332 ± 0.0089 | 0.7558 ± 0.0207 | 0.3550 ± 0.0179 | 0.3059 ± 0.0213 |
| 8 | **MobileNetV3-Large** | `none` | `finetune` | 6x | 0.8454 ± 0.0045 | **0.6832 ± 0.0327** | -0.1622 ± 0.0367 | 0.7037 ± 0.0140 | 0.3804 ± 0.0115 | 0.3297 ± 0.0179 |
| 9 | **DINO (ConvNeXt-B)** | `concat_frequency` | `finetune` | 5x | 0.8603 ± 0.0082 | **0.6820 ± 0.0101** | -0.1783 ± 0.0039 | 0.7053 ± 0.0391 | 0.4123 ± 0.0282 | 0.3664 ± 0.0192 |
| 10 | **ResNet-18** | `none` | `finetune` | 6x | 0.8765 ± 0.0220 | **0.6804 ± 0.0391** | -0.1961 ± 0.0582 | 0.6909 ± 0.0319 | 0.3352 ± 0.0278 | 0.2804 ± 0.0369 |
| 11 | **Vision Transformer (ViT-B/16)** | `concat_frequency` | `finetune` | 5x | 0.7915 ± 0.0072 | **0.6756 ± 0.0065** | -0.1159 ± 0.0073 | 0.5416 ± 0.0219 | 0.4893 ± 0.0033 | 0.4523 ± 0.0141 |
| 12 | **ResNet-18** | `concat` | `finetune` | 5x | 0.8717 ± 0.0114 | **0.6662 ± 0.0228** | -0.2055 ± 0.0170 | 0.7799 ± 0.0160 | 0.3853 ± 0.0310 | 0.3524 ± 0.0265 |
| 13 | **CLIP ViT-B/16** | `concat_frequency` | `finetune` | 5x | 0.7789 ± 0.0072 | **0.6505 ± 0.0060** | -0.1284 ± 0.0106 | 0.5939 ± 0.0207 | 0.4690 ± 0.0185 | 0.4301 ± 0.0283 |
| 14 | **Xception** | `none` | `finetune` | 6x | 0.7708 ± 0.0063 | **0.6388 ± 0.0221** | -0.1321 ± 0.0275 | 0.7298 ± 0.0244 | 0.3979 ± 0.0144 | 0.3445 ± 0.0226 |
| 15 | **MobileNetV3-Large** | `concat` | `finetune` | 5x | 0.8049 ± 0.0066 | **0.6296 ± 0.0025** | -0.1752 ± 0.0071 | 0.6676 ± 0.0177 | 0.4511 ± 0.0144 | 0.4294 ± 0.0179 |
| 16 | **Xception** | `concat` | `finetune` | 5x | 0.7329 ± 0.0039 | **0.6119 ± 0.0026** | -0.1210 ± 0.0032 | 0.7344 ± 0.0108 | 0.4459 ± 0.0112 | 0.4191 ± 0.0191 |
| 17 | **MobileNetV3-Large** | `concat_frequency` | `finetune` | 5x | 0.7084 ± 0.0017 | **0.6032 ± 0.0030** | -0.1052 ± 0.0020 | 0.5635 ± 0.0248 | 0.4530 ± 0.0062 | 0.4147 ± 0.0220 |
| 18 | **Xception** | `concat_frequency` | `finetune` | 5x | 0.6592 ± 0.0060 | **0.5746 ± 0.0054** | -0.0846 ± 0.0042 | 0.5973 ± 0.0158 | 0.4772 ± 0.0121 | 0.4578 ± 0.0206 |
| 19 | **Vision Transformer (ViT-B/16)** | `complex` | `finetune` | 5x | 0.5701 ± 0.0087 | **0.5541 ± 0.0057** | -0.0160 ± 0.0056 | 0.5093 ± 0.0110 | 0.4998 ± 0.0128 | 0.4982 ± 0.0187 |
| 20 | **MobileNetV3-Large** | `frequency_3` | `finetune` | 5x | 0.7127 ± 0.0071 | **0.5452 ± 0.0089** | -0.1676 ± 0.0027 | 0.4258 ± 0.0305 | 0.4473 ± 0.0171 | 0.4305 ± 0.0195 |
| 21 | **ResNet-18** | `complex` | `finetune` | 5x | 0.6992 ± 0.0348 | **0.5436 ± 0.0095** | -0.1556 ± 0.0265 | 0.4763 ± 0.0484 | 0.4330 ± 0.0433 | 0.4113 ± 0.0603 |
| 22 | **MobileNetV3-Large** | `magnitude` | `finetune` | 5x | 0.7177 ± 0.0056 | **0.5420 ± 0.0028** | -0.1757 ± 0.0047 | 0.4508 ± 0.0185 | 0.4602 ± 0.0063 | 0.4565 ± 0.0140 |
| 23 | **ResNet-18** | `magnitude` | `finetune` | 5x | 0.7503 ± 0.0092 | **0.5409 ± 0.0132** | -0.2094 ± 0.0049 | 0.4510 ± 0.0493 | 0.4737 ± 0.0174 | 0.4725 ± 0.0298 |
| 24 | **DINO (ConvNeXt-B)** | `frequency_3` | `finetune` | 5x | 0.7631 ± 0.0166 | **0.5371 ± 0.0063** | -0.2260 ± 0.0139 | 0.5003 ± 0.0271 | 0.4655 ± 0.0231 | 0.4602 ± 0.0299 |
| 25 | **ResNet-18** | `frequency_3` | `finetune` | 5x | 0.7500 ± 0.0058 | **0.5365 ± 0.0052** | -0.2135 ± 0.0064 | 0.4421 ± 0.0104 | 0.4784 ± 0.0132 | 0.4750 ± 0.0178 |
| 26 | **DINO (ConvNeXt-B)** | `magnitude` | `finetune` | 5x | 0.7614 ± 0.0031 | **0.5365 ± 0.0040** | -0.2249 ± 0.0040 | 0.4845 ± 0.0264 | 0.4890 ± 0.0105 | 0.4850 ± 0.0226 |
| 27 | **Vision Transformer (ViT-B/16)** | `magnitude` | `finetune` | 5x | 0.7240 ± 0.0091 | **0.5347 ± 0.0090** | -0.1893 ± 0.0088 | 0.4447 ± 0.0197 | 0.4694 ± 0.0179 | 0.4494 ± 0.0264 |
| 28 | **MobileNetV3-Large** | `complex` | `finetune` | 5x | 0.6872 ± 0.0041 | **0.5337 ± 0.0031** | -0.1534 ± 0.0023 | 0.5982 ± 0.0450 | 0.4649 ± 0.0074 | 0.4408 ± 0.0137 |
| 29 | **DINO (ConvNeXt-B)** | `phase` | `finetune` | 5x | 0.7128 ± 0.0074 | **0.5332 ± 0.0090** | -0.1795 ± 0.0069 | 0.5206 ± 0.0257 | 0.4232 ± 0.0135 | 0.3723 ± 0.0169 |
| 30 | **Xception** | `frequency_3` | `finetune` | 5x | 0.6994 ± 0.0132 | **0.5330 ± 0.0109** | -0.1664 ± 0.0037 | 0.4558 ± 0.0317 | 0.4636 ± 0.0072 | 0.4408 ± 0.0084 |
| 31 | **Xception** | `magnitude` | `finetune` | 5x | 0.7035 ± 0.0075 | **0.5328 ± 0.0080** | -0.1707 ± 0.0035 | 0.4180 ± 0.0267 | 0.4856 ± 0.0134 | 0.4760 ± 0.0210 |
| 32 | **CLIP ViT-B/16** | `frequency_3` | `finetune` | 5x | 0.7316 ± 0.0060 | **0.5328 ± 0.0070** | -0.1988 ± 0.0095 | 0.4725 ± 0.0271 | 0.4537 ± 0.0092 | 0.4256 ± 0.0160 |
| 33 | **CLIP ViT-B/16** | `magnitude` | `finetune` | 5x | 0.7336 ± 0.0014 | **0.5300 ± 0.0041** | -0.2036 ± 0.0045 | 0.4827 ± 0.0245 | 0.4846 ± 0.0117 | 0.4822 ± 0.0159 |
| 34 | **Vision Transformer (ViT-B/16)** | `frequency_3` | `finetune` | 5x | 0.7240 ± 0.0072 | **0.5281 ± 0.0070** | -0.1959 ± 0.0021 | 0.4723 ± 0.0302 | 0.4587 ± 0.0150 | 0.4466 ± 0.0201 |
| 35 | **MobileNetV3-Large** | `phase` | `finetune` | 5x | 0.6287 ± 0.0047 | **0.5234 ± 0.0035** | -0.1052 ± 0.0022 | 0.5096 ± 0.0188 | 0.4685 ± 0.0045 | 0.4351 ± 0.0093 |
| 36 | **ResNet-18** | `phase` | `finetune` | 5x | 0.6467 ± 0.0072 | **0.5211 ± 0.0024** | -0.1256 ± 0.0056 | 0.5052 ± 0.0120 | 0.4281 ± 0.0036 | 0.3547 ± 0.0105 |
| 37 | **CLIP ViT-B/16** | `complex` | `finetune` | 5x | 0.5271 ± 0.0378 | **0.5197 ± 0.0252** | -0.0074 ± 0.0134 | 0.4723 ± 0.0548 | 0.5293 ± 0.0537 | 0.5299 ± 0.0596 |
| 38 | **Xception** | `complex` | `finetune` | 5x | 0.5622 ± 0.0045 | **0.5174 ± 0.0042** | -0.0448 ± 0.0048 | 0.4895 ± 0.0140 | 0.4631 ± 0.0107 | 0.4372 ± 0.0148 |
| 39 | **Xception** | `phase` | `finetune` | 5x | 0.5889 ± 0.0063 | **0.5111 ± 0.0050** | -0.0778 ± 0.0033 | 0.4807 ± 0.0255 | 0.4618 ± 0.0093 | 0.4325 ± 0.0130 |
| 40 | **CLIP ViT-B/16** | `phase` | `finetune` | 5x | 0.6265 ± 0.0100 | **0.4994 ± 0.0022** | -0.1272 ± 0.0093 | 0.5180 ± 0.0330 | 0.4586 ± 0.0113 | 0.4422 ± 0.0202 |
| 41 | **DINO (ConvNeXt-B)** | `complex` | `finetune` | 5x | 0.4995 ± 0.0013 | **0.4977 ± 0.0052** | -0.0018 ± 0.0039 | 0.5000 | 0.5025 ± 0.0054 | 0.5112 ± 0.0252 |
| 42 | **Vision Transformer (ViT-B/16)** | `phase` | `finetune` | 5x | 0.6020 ± 0.0051 | **0.4955 ± 0.0034** | -0.1064 ± 0.0023 | 0.5264 ± 0.0181 | 0.4734 ± 0.0129 | 0.4646 ± 0.0179 |

---

## 2. Desempenho Detalhado por Modo de Frequência (Baseline `finetune`)

Permite comparar diretamente a eficácia das 6 arquiteturas para cada uma das formulações espectrais de Fourier:

### Modo: RGB Espacial (none)

| Modelo | Canais | Test AUC (FF++) | Test Acc | Test F1 | Test-D AUC (Corrompido) | ΔAUC | DF-40 AUC | Celeb-DF Frame |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| **CLIP ViT-B/16** | 3C | 0.9173 ± 0.0064 | 83.33% ± 1.01% | 84.12% ± 1.22% | **0.7651 ± 0.0408** | -0.1522 ± 0.0456 | 0.7639 ± 0.0460 | 0.3324 ± 0.0310 |
| **DINO (ConvNeXt-B)** | 3C | 0.9339 ± 0.0073 | 87.12% ± 1.47% | 87.98% ± 1.53% | **0.7370 ± 0.0555** | -0.1969 ± 0.0599 | 0.7208 ± 0.0289 | 0.3355 ± 0.0600 |
| **Vision Transformer (ViT-B/16)** | 3C | 0.8149 ± 0.0085 | 73.11% ± 0.98% | 74.21% ± 1.32% | **0.7036 ± 0.0284** | -0.1112 ± 0.0271 | 0.7040 ± 0.0157 | 0.3521 ± 0.0370 |
| **MobileNetV3-Large** | 3C | 0.8454 ± 0.0045 | 74.15% ± 0.36% | 73.74% ± 0.43% | **0.6832 ± 0.0327** | -0.1622 ± 0.0367 | 0.7037 ± 0.0140 | 0.3804 ± 0.0115 |
| **ResNet-18** | 3C | 0.8765 ± 0.0220 | 77.04% ± 2.39% | 77.29% ± 2.73% | **0.6804 ± 0.0391** | -0.1961 ± 0.0582 | 0.6909 ± 0.0319 | 0.3352 ± 0.0278 |
| **Xception** | 3C | 0.7708 ± 0.0063 | 70.43% ± 0.45% | 74.53% ± 0.34% | **0.6388 ± 0.0221** | -0.1321 ± 0.0275 | 0.7298 ± 0.0244 | 0.3979 ± 0.0144 |


### Modo: Magnitude FFT (magnitude)

| Modelo | Canais | Test AUC (FF++) | Test Acc | Test F1 | Test-D AUC (Corrompido) | ΔAUC | DF-40 AUC | Celeb-DF Frame |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| **MobileNetV3-Large** | 1C | 0.7177 ± 0.0056 | 65.92% ± 0.42% | 67.18% ± 0.78% | **0.5420 ± 0.0028** | -0.1757 ± 0.0047 | 0.4508 ± 0.0185 | 0.4602 ± 0.0063 |
| **ResNet-18** | 1C | 0.7503 ± 0.0092 | 68.55% ± 0.72% | 73.15% ± 0.90% | **0.5409 ± 0.0132** | -0.2094 ± 0.0049 | 0.4510 ± 0.0493 | 0.4737 ± 0.0174 |
| **DINO (ConvNeXt-B)** | 1C | 0.7614 ± 0.0031 | 67.78% ± 0.93% | 74.62% ± 0.34% | **0.5365 ± 0.0040** | -0.2249 ± 0.0040 | 0.4845 ± 0.0264 | 0.4890 ± 0.0105 |
| **Vision Transformer (ViT-B/16)** | 1C | 0.7240 ± 0.0091 | 63.55% ± 0.65% | 73.58% ± 0.22% | **0.5347 ± 0.0090** | -0.1893 ± 0.0088 | 0.4447 ± 0.0197 | 0.4694 ± 0.0179 |
| **Xception** | 1C | 0.7035 ± 0.0075 | 65.14% ± 0.61% | 71.26% ± 0.59% | **0.5328 ± 0.0080** | -0.1707 ± 0.0035 | 0.4180 ± 0.0267 | 0.4856 ± 0.0134 |
| **CLIP ViT-B/16** | 1C | 0.7336 ± 0.0014 | 67.32% ± 0.27% | 72.99% ± 0.25% | **0.5300 ± 0.0041** | -0.2036 ± 0.0045 | 0.4827 ± 0.0245 | 0.4846 ± 0.0117 |


### Modo: Fase FFT (phase)

| Modelo | Canais | Test AUC (FF++) | Test Acc | Test F1 | Test-D AUC (Corrompido) | ΔAUC | DF-40 AUC | Celeb-DF Frame |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| **DINO (ConvNeXt-B)** | 1C | 0.7128 ± 0.0074 | 65.19% ± 0.50% | 72.66% ± 0.72% | **0.5332 ± 0.0090** | -0.1795 ± 0.0069 | 0.5206 ± 0.0257 | 0.4232 ± 0.0135 |
| **MobileNetV3-Large** | 1C | 0.6287 ± 0.0047 | 55.45% ± 0.76% | 48.12% ± 2.40% | **0.5234 ± 0.0035** | -0.1052 ± 0.0022 | 0.5096 ± 0.0188 | 0.4685 ± 0.0045 |
| **ResNet-18** | 1C | 0.6467 ± 0.0072 | 60.70% ± 0.61% | 64.77% ± 1.01% | **0.5211 ± 0.0024** | -0.1256 ± 0.0056 | 0.5052 ± 0.0120 | 0.4281 ± 0.0036 |
| **Xception** | 1C | 0.5889 ± 0.0063 | 58.08% ± 0.34% | 71.45% ± 0.78% | **0.5111 ± 0.0050** | -0.0778 ± 0.0033 | 0.4807 ± 0.0255 | 0.4618 ± 0.0093 |
| **CLIP ViT-B/16** | 1C | 0.6265 ± 0.0100 | 58.79% ± 1.08% | 61.19% ± 3.05% | **0.4994 ± 0.0022** | -0.1272 ± 0.0093 | 0.5180 ± 0.0330 | 0.4586 ± 0.0113 |
| **Vision Transformer (ViT-B/16)** | 1C | 0.6020 ± 0.0051 | 57.40% ± 0.08% | 72.86% ± 0.06% | **0.4955 ± 0.0034** | -0.1064 ± 0.0023 | 0.5264 ± 0.0181 | 0.4734 ± 0.0129 |


### Modo: Complexo Real+Imag (complex)

| Modelo | Canais | Test AUC (FF++) | Test Acc | Test F1 | Test-D AUC (Corrompido) | ΔAUC | DF-40 AUC | Celeb-DF Frame |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| **Vision Transformer (ViT-B/16)** | 2C | 0.5701 ± 0.0087 | 57.35% ± 0.00% | 72.89% ± 0.00% | **0.5541 ± 0.0057** | -0.0160 ± 0.0056 | 0.5093 ± 0.0110 | 0.4998 ± 0.0128 |
| **ResNet-18** | 2C | 0.6992 ± 0.0348 | 64.15% ± 2.36% | 70.02% ± 3.65% | **0.5436 ± 0.0095** | -0.1556 ± 0.0265 | 0.4763 ± 0.0484 | 0.4330 ± 0.0433 |
| **MobileNetV3-Large** | 2C | 0.6872 ± 0.0041 | 61.65% ± 0.41% | 58.81% ± 1.19% | **0.5337 ± 0.0031** | -0.1534 ± 0.0023 | 0.5982 ± 0.0450 | 0.4649 ± 0.0074 |
| **CLIP ViT-B/16** | 2C | 0.5271 ± 0.0378 | 57.35% ± 0.00% | 72.89% ± 0.00% | **0.5197 ± 0.0252** | -0.0074 ± 0.0134 | 0.4723 ± 0.0548 | 0.5293 ± 0.0537 |
| **Xception** | 2C | 0.5622 ± 0.0045 | 57.35% ± 0.01% | 72.89% ± 0.01% | **0.5174 ± 0.0042** | -0.0448 ± 0.0048 | 0.4895 ± 0.0140 | 0.4631 ± 0.0107 |
| **DINO (ConvNeXt-B)** | 2C | 0.4995 ± 0.0013 | 57.35% | 72.89% | **0.4977 ± 0.0052** | -0.0018 ± 0.0039 | 0.5000 | 0.5025 ± 0.0054 |


### Modo: RGB + Magnitude (concat)

| Modelo | Canais | Test AUC (FF++) | Test Acc | Test F1 | Test-D AUC (Corrompido) | ΔAUC | DF-40 AUC | Celeb-DF Frame |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| **CLIP ViT-B/16** | 4C | 0.8941 ± 0.0042 | 81.51% ± 0.71% | 82.46% ± 0.88% | **0.7260 ± 0.0119** | -0.1681 ± 0.0147 | 0.7650 ± 0.0323 | 0.3948 ± 0.0195 |
| **Vision Transformer (ViT-B/16)** | 4C | 0.8265 ± 0.0035 | 75.69% ± 0.49% | 77.41% ± 0.75% | **0.7123 ± 0.0056** | -0.1142 ± 0.0036 | 0.6324 ± 0.0237 | 0.4519 ± 0.0146 |
| **DINO (ConvNeXt-B)** | 4C | 0.9166 ± 0.0070 | 85.12% ± 1.39% | 85.97% ± 1.47% | **0.6834 ± 0.0102** | -0.2332 ± 0.0089 | 0.7558 ± 0.0207 | 0.3550 ± 0.0179 |
| **ResNet-18** | 4C | 0.8717 ± 0.0114 | 76.57% ± 1.90% | 76.63% ± 2.43% | **0.6662 ± 0.0228** | -0.2055 ± 0.0170 | 0.7799 ± 0.0160 | 0.3853 ± 0.0310 |
| **MobileNetV3-Large** | 4C | 0.8049 ± 0.0066 | 71.84% ± 0.68% | 71.67% ± 0.96% | **0.6296 ± 0.0025** | -0.1752 ± 0.0071 | 0.6676 ± 0.0177 | 0.4511 ± 0.0144 |
| **Xception** | 4C | 0.7329 ± 0.0039 | 67.20% ± 0.31% | 72.82% ± 0.32% | **0.6119 ± 0.0026** | -0.1210 ± 0.0032 | 0.7344 ± 0.0108 | 0.4459 ± 0.0112 |


### Modo: Filtros 3-Bandas (frequency_3)

| Modelo | Canais | Test AUC (FF++) | Test Acc | Test F1 | Test-D AUC (Corrompido) | ΔAUC | DF-40 AUC | Celeb-DF Frame |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| **MobileNetV3-Large** | 1C | 0.7127 ± 0.0071 | 65.10% ± 0.42% | 65.62% ± 0.64% | **0.5452 ± 0.0089** | -0.1676 ± 0.0027 | 0.4258 ± 0.0305 | 0.4473 ± 0.0171 |
| **DINO (ConvNeXt-B)** | 1C | 0.7631 ± 0.0166 | 67.59% ± 1.29% | 74.85% ± 1.11% | **0.5371 ± 0.0063** | -0.2260 ± 0.0139 | 0.5003 ± 0.0271 | 0.4655 ± 0.0231 |
| **ResNet-18** | 1C | 0.7500 ± 0.0058 | 68.55% ± 0.37% | 73.34% ± 0.75% | **0.5365 ± 0.0052** | -0.2135 ± 0.0064 | 0.4421 ± 0.0104 | 0.4784 ± 0.0132 |
| **Xception** | 1C | 0.6994 ± 0.0132 | 64.85% ± 1.03% | 70.83% ± 1.17% | **0.5330 ± 0.0109** | -0.1664 ± 0.0037 | 0.4558 ± 0.0317 | 0.4636 ± 0.0072 |
| **CLIP ViT-B/16** | 1C | 0.7316 ± 0.0060 | 67.13% ± 0.35% | 72.86% ± 0.83% | **0.5328 ± 0.0070** | -0.1988 ± 0.0095 | 0.4725 ± 0.0271 | 0.4537 ± 0.0092 |
| **Vision Transformer (ViT-B/16)** | 1C | 0.7240 ± 0.0072 | 63.10% ± 0.87% | 73.90% ± 0.36% | **0.5281 ± 0.0070** | -0.1959 ± 0.0021 | 0.4723 ± 0.0302 | 0.4587 ± 0.0150 |


### Modo: Concat Espectral 7C (concat_frequency)

| Modelo | Canais | Test AUC (FF++) | Test Acc | Test F1 | Test-D AUC (Corrompido) | ΔAUC | DF-40 AUC | Celeb-DF Frame |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| **ResNet-18** | 7C | 0.8337 ± 0.0072 | 74.69% ± 0.85% | 76.08% ± 1.03% | **0.6866 ± 0.0103** | -0.1472 ± 0.0048 | 0.7071 ± 0.0316 | 0.4344 ± 0.0231 |
| **DINO (ConvNeXt-B)** | 7C | 0.8603 ± 0.0082 | 78.25% ± 1.23% | 79.81% ± 1.58% | **0.6820 ± 0.0101** | -0.1783 ± 0.0039 | 0.7053 ± 0.0391 | 0.4123 ± 0.0282 |
| **Vision Transformer (ViT-B/16)** | 7C | 0.7915 ± 0.0072 | 71.66% ± 0.52% | 76.69% ± 0.57% | **0.6756 ± 0.0065** | -0.1159 ± 0.0073 | 0.5416 ± 0.0219 | 0.4893 ± 0.0033 |
| **CLIP ViT-B/16** | 7C | 0.7789 ± 0.0072 | 70.57% ± 0.69% | 72.70% ± 0.72% | **0.6505 ± 0.0060** | -0.1284 ± 0.0106 | 0.5939 ± 0.0207 | 0.4690 ± 0.0185 |
| **MobileNetV3-Large** | 7C | 0.7084 ± 0.0017 | 63.94% ± 0.38% | 63.31% ± 1.20% | **0.6032 ± 0.0030** | -0.1052 ± 0.0020 | 0.5635 ± 0.0248 | 0.4530 ± 0.0062 |
| **Xception** | 7C | 0.6592 ± 0.0060 | 61.01% ± 0.37% | 70.22% ± 0.39% | **0.5746 ± 0.0054** | -0.0846 ± 0.0042 | 0.5973 ± 0.0158 | 0.4772 ± 0.0121 |


---

## 3. Síntese: Melhor Representação Espectral por Arquitetura

| Arquitetura | Melhor Modo FF++ Limpo | Test AUC | Melhor Modo sob Corrupção (`test_d`) | Test-D AUC | Melhor Modo Cross-Dataset (DF-40) | DF-40 AUC |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: |
| **CLIP ViT-B/16** | `none` | 0.9173 | `none` | **0.7651** | `concat` | 0.7650 |
| **DINO (ConvNeXt-B)** | `none` | 0.9339 | `none` | **0.7370** | `concat` | 0.7558 |
| **Vision Transformer (ViT-B/16)** | `concat` | 0.8265 | `concat` | **0.7123** | `none` | 0.7040 |
| **ResNet-18** | `none` | 0.8765 | `concat_frequency` | **0.6866** | `concat` | 0.7799 |
| **MobileNetV3-Large** | `none` | 0.8454 | `none` | **0.6832** | `none` | 0.7037 |
| **Xception** | `none` | 0.7708 | `none` | **0.6388** | `concat` | 0.7344 |

> [!TIP]
> **Principais Observações:**
> 1. **Modelos Convolucionais Clássicos e Fourier:** O modo `concat` (RGB + magnitude FFT) confere um ganho substancial em generalização cross-dataset para ResNet (+8.90 pp no DF-40) e Xception (+0.46 pp), e o modo `concat_frequency` melhora a robustez contra degradações severas na ResNet-18 (`test_d` sobe de 0.6804 para 0.6866).
> 2. **Vision Transformers Foundation (CLIP e DINO):** Mantêm performance dominante no espaço RGB puro (`none`), mas sofrem forte queda de robustez em dados corrompidos quando não utilizam aumento estocástico pesado (degradação $\Delta\text{AUC}$ de -15 pp a -20 pp).
> 3. **Para os Resultados dos Modelos Robustos:** Consulte a [**Tabela 4 (`tabela4-seedrobusta-rgb.md`)**](file:///home/lucas.ocunha/tcc/results/mostrar_rayson/tabela4-seedrobusta-rgb.md) para ver como a pipeline `RandomizedRobustAugment` recupera até +11 pp de Test-D AUC.
