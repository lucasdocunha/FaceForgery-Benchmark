# Tabela 3: Resultados Consolidados dos Modelos Mixture of Experts (MoE)

**Documento:** `tabela3-resultados-moe.md`  
**Destinatário:** Apresentação Técnica / Rayson  
**Ambiente de Execução:** Dual NVIDIA GeForce RTX 3090 (24GB) | Workstation Local (`sicret2`)  
**Dataset Base:** FaceForensics++ (181.947 amostras no teste) + DF-40 + Degradações `test_d`  

---

## 📌 Descrição das Técnicas e Formulações MoE

A arquitetura Mixture of Experts (MoE) foi projetada para explorar especialização dinâmica entre peritos através de uma rede de roteamento convolucional leve (`MoERouter`):

- **MoE Standard**: Especialistas homogêneos operando no mesmo espaço RGB espacial puro ($C_{in}=3$). A especialização decorre da divergência estocástica de pesos e trajetórias de gradiente separadas.
- **Frequency MoE**: Especialistas heterogêneos dedicados a sub-espaços espectrais decompostos da Transformada de Fourier 2D (RGB, Magnitude, Fase, Filtros Passa-Alta e Passa-Baixa).
- **Mecanismo de Roteamento Dinâmico**: O roteador processa a entrada via $\text{AdaptiveAvgPool2d}((4, 4))$ seguido de MLP linear e função Softmax escalada por temperatura $\tau=1.0$, ativando os $k$ peritos mais relevantes via seleção esparsa Top-$k$ ($k=3$).

---

## 1. Comparativo Direto no Conjunto de Teste (181.947 Amostras)

| Modelo MoE | Técnica / Domínio | Val AUC | Acurácia (%) | ROC-AUC | F1-Score (%) | Precisão (%) | Recall (%) | Especificidade (%) | Loss Teste | Limiar (Threshold) |
| :--- | :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| **MoE Standard** | `RGB Espacial (3 ch)` | 0.9316 | **77.73%** | 0.8773 | **77.77%** | 90.98% | **67.90%** | 90.95% | 1.1193 | 0.5000 |
| **Frequency MoE** | `RGB + FFT Decomposto (7 ch)` | 0.9496 | **78.68%** | 0.8737 | **79.69%** | 87.84% | **72.92%** | 86.42% | 1.7828 | 0.5000 |
| *Δ (Frequency - Standard)* | *Diferencial Espectral* | *+0.0180* | **+0.95 pp** | *-0.0036* | **+1.92 pp** | *-3.14 pp* | **+5.02 pp (+5.777 acertos)** | *-4.53 pp* | *+0.6635* | *0.0000* |

> [!TIP]
> **Destaque Forense:** O **Frequency MoE** atinge **72,92% de Recall** contra **67,90%** do MoE Standard. Isso representa **+5.777 manipulações faciais detectadas com sucesso**, reduzindo substancialmente a taxa de falsos negativos em ataques de forjamento.

---

## 2. Diagnóstico de Erros Forenses e Matriz de Confusão (181.947 Imagens)

| Modelo MoE | Verdadeiros Positivos (TP) | Falsos Positivos (FP) | Falsos Negativos (FN) | Verdadeiros Negativos (TN) | Taxa Falsos Negativos (FNR %) | Taxa Falsos Positivos (FPR %) |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: |
| **MoE Standard** | **70,315** | 9,387 | **33,238** | 69,007 | **32.10%** | 11.97% |
| **Frequency MoE** | **76,092** | 14,076 | **27,461** | 64,318 | **26.52%** | 17.95% |
| *Diferencial Forense* | *+5.777 deepfakes detectados* | *+4.689 alarmes falsos* | *-5.777 falsos negativos evitados* | *-4.689* | *-5.58 pp (Menos evasões)* | *+5.98 pp* |

---

## 3. Arquitetura e Especialização por Quantidade de Peritos

Detalha o papel de cada especialista na versão original de 4 peritos (v1) e na versão expandida de 7 peritos (v2):

### Variante A: MoE de 4 Especialistas (v1 — Validação Preliminar)

| ID do Especialista | Técnica Standard (RGB) | Técnica Frequency (FFT) | Canais Entrada | Função Forense Primária |
| :---: | :--- | :--- | :--- | :--- |
| `Expert 0` | Backbone Convolucional A | **Espacial RGB Puro** | `3 canais` | Semântica facial, olhos, boca e textura macro |
| `Expert 1` | Backbone Convolucional B | **Magnitude Log-FFT** | `1 canal` | Análise de frequências globais e padrão de grade GAN |
| `Expert 2` | Backbone Convolucional C | **Fase Espectral FFT** | `1 canal` | Descontinuidades de fase angular e bordas de fusão |
| `Expert 3` | Backbone Convolucional D | **Magnitude Passa-Alta** | `1 canal` | Artefatos de alta frequência e ruído residual de interpolação |

### Variante B: MoE Expandido de 7 Especialistas (v2 — Arquitetura Completa)

| ID do Especialista | Nome do Perito | Canais de Entrada | Domínio Físico | Especialização Forense |
| :---: | :--- | :--- | :--- | :--- |
| `Expert 0` | **RGB Spatial Expert** | `3 canais` | `Espaço Espacial RGB` | Estrutura morfológica, simetria facial e iluminação |
| `Expert 1` | **Log-Magnitude Expert** | `1 canal` | `Espectro de Fourier Global` | Periodicidade de upsampling, artefatos de transposição GAN |
| `Expert 2` | **Spectral Phase Expert** | `1 canal` | `Fase Angular Fourier` | Deslocamento de fase em contornos de costura (boundary artifacts) |
| `Expert 3` | **High-Pass Filter Expert** | `1 canal` | `Altas Frequências` | Supressão de conteúdo semântico; foca em ruído de compressão e micro-bordas |
| `Expert 4` | **Low-Pass Filter Expert** | `1 canal` | `Baixas Frequências` | Gradientes suaves de cor, borrões de blending e inconsistências de tom de pele |
| `Expert 5` | **Spatial-Spectral Hybrid** | `4 canais (RGB + HP)` | `Híbrido Espaço + Alta Frequência` | Correlação direta entre pixels espaciais e descontinuidades locais |
| `Expert 6` | **Complex Fourier Expert** | `2 canais (Re + Im)` | `Representação Complexa Integral` | Preserva magnitude e fase simultâneas sem perda de quadratura |

---

## 4. Custos Computacionais, Parâmetros e Eficiência Física

Métricas medidas na GPU NVIDIA RTX 3090 com resolução $224 \times 224$ pixels e batch size = 32:

| Arquitetura MoE | Qtd Especialistas | Parâmetros Totais | Backbone Base | Roteador | GFLOPs ($224^2$) | Latência Lote (32) | Latência Imagem | Throughput (FPS) | Tamanho Checkpoint |
| :--- | :--- | :--- | :--- | :--- | :---: | :---: | :---: | :---: | :---: |
| **MoE Standard** | `7 Peritos (Top-3 Ativos)` | **29.435.749** | `MobileNetV3` | `Conv+MLP Top-3` | 3.01 GFLOPs | 35.2 ms | 1.10 ms | **905.2 FPS** | ~112 MB |
| **Frequency MoE** | `7 Peritos (Top-3 Ativos)` | **29.438.693** | `MobileNetV3` | `Conv+MLP Top-3` | 2.99 GFLOPs | 36.1 ms | 1.13 ms | **885.4 FPS** | ~112 MB |

---

## 5. Resiliência a Degradações Não-Vistas (`test_d`) e Validação

| Modelo MoE | Domínio | Validação (Val AUC) | Val Acurácia | Teste Limpo (Test AUC) | Teste Corrompido (Test-D AUC) | ΔAUC (Degradação) | Status de Retenção |
| :--- | :--- | :---: | :---: | :---: | :---: | :---: | :---: |
| **Frequency MoE** | `RGB + FFT (7 canais)` | **0.9496** | 88.00% | 0.8737 | **0.6650** | -0.2087 | Convergência Rápida (Pico Época 10) |
| **MoE Standard** | `RGB Espacial (3 canais)` | **0.9316** | 84.78% | 0.8773 | **0.6796** | -0.1977 | Convergência Lenta (20 Épocas completas) |