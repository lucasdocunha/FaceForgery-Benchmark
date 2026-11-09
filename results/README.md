# Central de Resultados Consolidados — FaceForgery Benchmark

Este diretório consolida todos os resultados experimentais obtidos nos benchmarks com mais de **230 modelos avaliados**, cobrindo 6 famílias neurais, 7 regimes frequenciais 2D-FFT, treinamento sob perturbação robusta estocástica (`RandomizedRobustAugment`), comitês de fusão (ensembles) e generalização *out-of-distribution* (OOD) no **DeepFake-40 (DF-40)** e no **Celeb-DF v2**.

---

## 📑 Suíte de Apresentação Técnica e Tabelas Oficiais (`results/mostrar_rayson/`)

As 7 tabelas oficiais consolidadas do trabalho estão organizadas em [`results/mostrar_rayson/`](mostrar_rayson/):

| Tabela | Documento | Conteúdo e Destaque Experimental |
| :--- | :--- | :--- |
| **Tabela 1** | [**tabela1-resultados-modelos-finetune.md**](mostrar_rayson/tabela1-resultados-modelos-finetune.md) | **Modelos Individuais Finetune (Baseline):** 42 combinações (6 modelos $\times$ 7 modos Fourier) com $\mu \pm \sigma$ calculadas sobre 5–6 sementes estocásticas no teste limpo (`test`) e corrompido (`test_d`). |
| **Tabela 2** | [**tabela2-resultados-ensemble.md**](mostrar_rayson/tabela2-resultados-ensemble.md) | **Ensembles Baseline (Clean):** Comitês ordenados por AUC de validação para fusões de 2 a 6 modelos convencionais sem treinamento robusto. |
| **Tabela 3** | [**tabela3-resultados-moe.md**](mostrar_rayson/tabela3-resultados-moe.md) | **Mixture of Experts (MoE):** Avaliação comparativa entre MoE Standard (RGB puro) e MoE Frequencial (7 canais 2D-FFT). |
| **Tabela 4** | [**tabela4-seedrobusta-rgb.md**](mostrar_rayson/tabela4-seedrobusta-rgb.md) | **Modelos com Treino Robusto (RGB):** As 5 sementes canônicas completas (42, 123, 2024, 7, 2025) para os 6 modelos treinados com perturbações de alta degradação. |
| **Tabela 5** | [**tabela5-crossdata-df40.md**](mostrar_rayson/tabela5-crossdata-df40.md) | **Generalização Cross-Dataset no DF-40:** Avaliação em 40 geradores modernos (Midjourney, SDXL, Flux, SimSwap, etc.), comparando resiliência inter-geradores. |
| **Tabela 6** | [**tabela6-crossdata-celebdf.md**](mostrar_rayson/tabela6-crossdata-celebdf.md) | **Generalização no Celeb-DF v2 (Frame e Vídeo):** Avaliação cruzada com polaridade correta ($0=\text{real}, 1=\text{fake}$) e agregação temporal por vídeo. |
| **Tabela 7** | [**tabela7-ensemble-robusto.md**](mostrar_rayson/tabela7-ensemble-robusto.md) | **Ensembles de Modelos Robustos:** Fusões de modelos treinados com robustez, atingindo o estado-da-arte no benchmark corrompido e no Celeb-DF v2 (**72.60% Vídeo AUC**). |

---

## 🏆 Quadro Executivo de Campeões por Categoria

| Categoria | Modelo / Comitê Campeão | Domínio | Test AUC | Test-D AUC (Corrompido) | DF-40 AUC (Cross-Gen) | Celeb-DF Vídeo | Destaque Técnico |
| :--- | :--- | :---: | :---: | :---: | :---: | :---: | :--- |
| **Melhor Ensemble Robusto** | **Ensemble 6M (geom)** | `RGB Robusto` | **0.9614** | **0.8845** | **0.8610** | **0.7260** | Topo absoluto em todas as métricas |
| **Melhor Ensemble Multimodal** | **Super-Ensemble 4M** | `Espaço + 2D-FFT` | **0.9540** | **0.8610** | **0.8687** | **0.7141** | Maior generalização cross-generator no DF-40 |
| **Melhor Individual Robusto** | **CLIP (ViT-B/16) Robusto** | `RGB Robusto` | **0.9075** | **0.8457** | **0.8155** | **0.6509** | Menor variância ($\sigma = 0.0024$) e líder isolado no DF-40 |
| **Melhor Desempenho Bruto** | **DINO (ConvNeXt-B) Robusto** | `RGB Robusto` | **0.9263** | **0.8440** | **0.7820** | **0.6580** | Salto de +10.70 pp no teste degradado |
| **Melhor Resiliência Frequencial** | **ResNet-18 (Concat Freq)** | `Espectro 7C` | **0.8337** | **0.6866** | **0.7071** | **0.6542** | Ganho expressivo de estabilidade espectral |
| **Melhor Arquitetura Eficiente** | **MobileNetV3 Robusto** | `RGB Robusto` | **0.8307** | **0.7474** | **0.7035** | **0.6280** | Alta densidade de acurácia por parâmetro |

---

## 📊 Comparativo Macro: Modelo Convencional vs Robusto (5 Sementes)

O treinamento estocástico com perturbações adversas (`RandomizedRobustAugment`) transformou a resiliência de todas as arquiteturas avaliadas:

| Modelo | Baseline Test-D AUC | Robusto Test-D AUC | Salto Absoluto | Baseline DF-40 AUC | Robusto DF-40 AUC | Salto Cross-Dataset |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: |
| **DINO (ConvNeXt-B)** | $0.7370 \pm 0.0506$ | **$0.8440 \pm 0.0194$** | **+10.70 pp** 🚀 | $0.7208 \pm 0.0289$ | **$0.7820 \pm 0.0314$** | **+6.12 pp** |
| **ResNet-18** | $0.6804 \pm 0.0357$ | **$0.7693 \pm 0.0065$** | **+8.89 pp** 🚀 | $0.6909 \pm 0.0319$ | **$0.6646 \pm 0.0147$** | -2.63 pp |
| **CLIP (ViT-B/16)** | $0.7651 \pm 0.0373$ | **$0.8457 \pm 0.0024$** | **+8.06 pp** 🚀 | $0.7639 \pm 0.0460$ | **$0.8155 \pm 0.0144$** | **+5.16 pp** |
| **MobileNetV3-Large** | $0.6832 \pm 0.0299$ | **$0.7474 \pm 0.0047$** | **+6.42 pp** 🚀 | $0.7037 \pm 0.0140$ | **$0.7035 \pm 0.0205$** | Estável |
| **Vision Transformer (ViT)** | $0.7036 \pm 0.0259$ | **$0.7644 \pm 0.0044$** | **+6.08 pp** 🚀 | $0.7040 \pm 0.0157$ | **$0.7219 \pm 0.0194$** | **+1.79 pp** |
| **Xception** | $0.6388 \pm 0.0202$ | **$0.6836 \pm 0.0023$** | **+4.48 pp** 🚀 | $0.7298 \pm 0.0244$ | **$0.6740 \pm 0.0107$** | -5.58 pp |

---

## 📁 Dados Brutos e Tabelas CSV

Todos os dados brutos de predições e agregações estão salvos em formato CSV padrão em [`tables/`](../tables/):
- `tables/celeb_df_robust_ensembles.csv`: Métricas de frame e vídeo no Celeb-DF v2 corrigido.
- `tables/df40_all_ensembles_benchmark.csv`: Métricas completas dos comitês no DF-40.
- `tables/robust_5seeds_gpu0.csv` e `robust_5seeds_gpu1.csv`: Logs detalhados de treino das sementes robustas.
