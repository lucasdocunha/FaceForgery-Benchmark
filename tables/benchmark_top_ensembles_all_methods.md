# Benchmark Exaustivo de Métodos de Fusão: Top-2 até Top-5

**Executado em hardware:** Workstation Local `sicret2`  
**Modelos Avaliados (Campeões de Validação):**
- `CLIP Robusto (none/finetune_robust/s987)`
- `ResNet Concat (concat/finetune/s123)`
- `CLIP Concat (concat/finetune/s2025)`
- `DINO Concat (concat/finetune/s123)`
- `DINO Robusto (none/finetune_robust/s987)`
- `Xception Concat (concat/finetune/s42)`

---

## 🏆 Quadro de Campeões por Benchmark (Qual método ganhou?)

| Benchmark de Avaliação | Comitê Vencedor | Estratégia Campeã | AUC (%) | Destaque Técnico |
| :--- | :--- | :---: | :---: | :--- |
| **DF-40 (40 Novos Geradores)** | **Top-3 Campeão Geral** | **`weighted`** | **86,53%** 🟡 | Ponderação por Val AUC superou média simples e geométrica |
| **Test Difícil (Test-D 14 Corrupções)** | **Top-5 Campeões** | **`optimal_weights`** | **83,92%** 🟡 | Ajuste numérico de pesos minimizou o impacto do ruído |
| **Celeb-DF v2 (Vídeo AUC H.264)** | **Top-4 Híbrido** | **`stacking`** | **74,79%** 🟡 | Meta-learner descorrelacionou as representações temporais |
| **Teste Limpo (FF++ c23)** | **Top-5 Campeões** | **`optimal_weights`** | **94,60%** 🟡 | Combinação convexa ótima atingiu o pico in-distribution |
| **Validação (FF++ Val)** | **Top-5 Campeões** | **`optimal_weights`** | **99,79%** 🟡 | Perda BCE minimizada diretamente no split de validação |

---

## 📊 Tabela Completa Consolidada: Todos os Comitês e Estratégias

| Comitê | K | Estratégia de Fusão | Val AUC (%) | Test AUC (%) | Test Difícil AUC (%) | DF-40 AUC (%) | Celeb-DF Vídeo AUC (%) |
| :--- | :---: | :--- | :---: | :---: | :---: | :---: | :---: |
| **Top-2 Híbrido** | 2 | `mean` | 99,06 | 92,40 | 78,79 | 85,42 | 70,29 |
| **Top-2 Híbrido** | 2 | `geometric` | 98,91 | 91,94 | 78,58 | 85,40 | 70,25 |
| **Top-2 Híbrido** | 2 | `harmonic` | 98,52 | 90,09 | 77,11 | 84,59 | 70,16 |
| **Top-2 Híbrido** | 2 | `max` | 98,42 | 91,93 | 73,81 | 83,40 | 64,38 |
| **Top-2 Híbrido** | 2 | `weighted` | 99,11 | 92,59 | 79,84 | 85,76 | 70,28 |
| **Top-2 Híbrido** | 2 | `optimal_weights` | 99,09 | 92,60 | 82,22 | 85,85 | 69,48 |
| **Top-2 Híbrido** | 2 | `stacking` | 99,13 | 92,67 | 81,44 | 86,00 | 69,89 |
| **Top-2 Híbrido** | 2 | `majority` | 97,57 | 88,62 | 75,43 | 82,05 | 60,37 |
| **Top-3 Campeão Geral** | 3 | `mean` | 99,02 | 93,26 | 78,29 | 86,41 | 70,96 |
| **Top-3 Campeão Geral** | 3 | `geometric` | 99,02 | 93,03 | 78,52 | 86,44 | 70,78 |
| **Top-3 Campeão Geral** | 3 | `harmonic` | 98,55 | 90,47 | 76,74 | 85,00 | 70,52 |
| **Top-3 Campeão Geral** | 3 | `max` | 97,63 | 92,64 | 72,79 | 83,37 | 65,82 |
| **Top-3 Campeão Geral** | 3 | `weighted` | 99,06 | 93,32 | 79,04 | **86,53** 🟡 | 70,95 |
| **Top-3 Campeão Geral** | 3 | `optimal_weights` | 99,11 | 92,92 | 82,05 | 86,15 | 69,85 |
| **Top-3 Campeão Geral** | 3 | `stacking` | 99,17 | 93,23 | 80,82 | 86,52 | 70,70 |
| **Top-3 Campeão Geral** | 3 | `majority` | 98,14 | 91,27 | 76,29 | 84,17 | 62,74 |
| **Top-4 Híbrido** | 4 | `mean` | 99,48 | 94,14 | 76,76 | 86,15 | 74,40 |
| **Top-4 Híbrido** | 4 | `geometric` | 99,50 | 94,16 | 77,02 | 85,58 | 74,63 |
| **Top-4 Híbrido** | 4 | `harmonic` | 99,08 | 91,59 | 75,68 | 83,51 | 74,32 |
| **Top-4 Híbrido** | 4 | `max` | 97,71 | 92,89 | 72,83 | 83,37 | 65,83 |
| **Top-4 Híbrido** | 4 | `weighted` | 99,51 | 94,19 | 76,95 | 86,12 | 74,45 |
| **Top-4 Híbrido** | 4 | `optimal_weights` | 99,59 | 93,84 | 78,23 | 84,25 | 74,67 |
| **Top-4 Híbrido** | 4 | `stacking` | 99,62 | 94,21 | 78,72 | 85,70 | **74,79** 🟡 |
| **Top-4 Híbrido** | 4 | `majority` | 99,15 | 92,75 | 75,19 | 84,59 | 68,53 |
| **Top-5 Campeões** | 5 | `mean` | 99,58 | 93,06 | 77,98 | 84,05 | 73,69 |
| **Top-5 Campeões** | 5 | `geometric` | 99,58 | 93,71 | 78,74 | 83,36 | 74,50 |
| **Top-5 Campeões** | 5 | `harmonic` | 98,61 | 90,41 | 76,29 | 81,19 | 74,46 |
| **Top-5 Campeões** | 5 | `max` | 97,36 | 89,27 | 71,37 | 82,33 | 64,23 |
| **Top-5 Campeões** | 5 | `weighted` | 99,66 | 93,58 | 78,96 | 84,14 | 73,87 |
| **Top-5 Campeões** | 5 | `optimal_weights` | **99,79** 🟡 | **94,60** 🟡 | **83,92** 🟡 | 82,08 | 72,93 |
| **Top-5 Campeões** | 5 | `stacking` | 99,78 | 94,35 | 83,60 | 83,92 | 73,45 |
| **Top-5 Campeões** | 5 | `majority` | 99,34 | 91,76 | 76,70 | 82,50 | 69,08 |
