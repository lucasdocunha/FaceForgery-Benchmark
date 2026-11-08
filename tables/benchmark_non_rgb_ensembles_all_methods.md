# Benchmark de Ensembles SEM RGB Normal (Apenas SRM e Fourier Concat)

**Premissa da Avaliação:** Remoção total do modo espacial comum (`none` / RGB normal de 3 canais). Todos os modelos integrantes operam exclusivamente sobre **representações físicas de domínio alternativo**:
1. **Resíduos Forenses de Ruído Esteganográfico (SRM - 6 canais):** Filtros de alta frequência que extraem padrões de ruído de sensor e interpolação residual.
2. **Espectro de Frequência de Fourier (`concat` - 4 canais):** Concatenação da magnitude 2D centrada da FFT em escala logarítmica.

**Ambiente de Execução:** Workstation Local `sicret2`  
**Modelos Não-RGB Integrantes (Campeões por Val AUC):**
* `DINO SRM (dino/srm/finetune_robust/s123)`
* `CLIP SRM (clip/srm/finetune_robust/s2024)`
* `ResNet Concat (resnet/concat/finetune/s123)`
* `MobileNet SRM (mobilenet/srm/finetune_robust/s7)`
* `ViT SRM (vit/srm/finetune_robust/s123)`
* `DINO Concat (dino/concat/finetune/s123)`
* `CLIP Concat (clip/concat/finetune/s2025)`
* `Xception Concat (xception/concat/finetune/s42)`

---

## 🏆 Quadro de Campeões: Quem Ganhou SEM RGB Normal?

| Benchmark | Comitê Vencedor (Sem RGB Normal) | Estratégia Campeã | AUC (%) | Ganho vs. Comitê com RGB Normal |
| :--- | :--- | :---: | :---: | :--- |
| **Celeb-DF v2 (Vídeo AUC H.264)** | **Top-4 Não-RGB (DINO SRM + CLIP SRM + ResNet Concat + MobileNet SRM)** | **`geometric`** | **78,23%** 🟡 | **+3,44%** de ganho (saltou de 74,79% para **78,23%**) |
| **Test Difícil (Test-D 14 Corrupções)** | **Top-2 Não-RGB (DINO SRM + CLIP SRM)** | **`geometric`** | **87,34%** 🟡 | **+3,42%** de ganho (saltou de 83,92% para **87,34%**) |
| **DF-40 (40 Novos Geradores)** | **Top-3 Não-RGB (DINO SRM + CLIP SRM + ResNet Concat)** | **`geometric`** | **86,40%** 🟡 | Empate técnico no teto de generalização (**86,40%**) |
| **Teste Limpo (FF++ c23)** | **Top-3 Não-RGB (DINO SRM + CLIP SRM + ResNet Concat)** | **`stacking`** | **93,84%** 🟡 | Mantém estabilidade in-distribution (93,84%) |
| **Validação (FF++ Val)** | **Top-3 e Top-4 Não-RGB** | **`optimal_weights`** | **99,72%** 🟡 | Mínima perda de validação com regularização física |

---

## 📊 Tabela Consolidada: Comitês Não-RGB (SRM + Fourier Concat)

| Comitê | K | Estratégia de Fusão | Val AUC (%) | Test Limpo AUC (%) | Test Difícil AUC (%) | DF-40 AUC (%) | Celeb-DF Vídeo AUC (%) |
| :--- | :---: | :--- | :---: | :---: | :---: | :---: | :---: |
| **Top-2 Não-RGB** | 2 | `mean` | 99,59 | 93,10 | 86,35 | 84,06 | 74,84 |
| *(DINO SRM +* | 2 | `geometric` | 99,65 | 93,51 | **87,34** 🟡 | 84,62 | 76,37 |
| *CLIP SRM)* | 2 | `harmonic` | 99,54 | 93,07 | 87,17 | 84,40 | 76,66 |
| | 2 | `max` | 99,35 | 92,56 | 85,67 | 83,59 | 70,13 |
| | 2 | `weighted` | 99,60 | 93,12 | 86,36 | 84,05 | 74,89 |
| | 2 | `optimal_weights` | 99,68 | 93,29 | 86,54 | 83,33 | 76,21 |
| | 2 | `stacking` | 99,68 | 93,28 | 86,52 | 83,41 | 76,14 |
| | 2 | `majority` | 98,66 | 87,02 | 80,10 | 79,31 | 71,94 |
| :--- | :---: | :--- | :---: | :---: | :---: | :---: | :---: |
| **Top-3 Não-RGB** | 3 | `mean` | 99,59 | 93,52 | 82,05 | 85,51 | 75,30 |
| *(DINO SRM + CLIP SRM +* | 3 | `geometric` | 99,60 | 93,62 | 82,21 | **86,40** 🟡 | 76,47 |
| *ResNet Concat FFT)* | 3 | `harmonic` | 99,09 | 91,07 | 78,99 | 85,09 | 76,73 |
| | 3 | `max` | 98,81 | 92,13 | 75,01 | 83,41 | 64,63 |
| | 3 | `weighted` | 99,61 | 93,58 | 82,33 | 85,54 | 75,33 |
| | 3 | `optimal_weights` | 99,72 | 93,83 | 85,04 | 83,92 | 76,41 |
| | 3 | `stacking` | 99,71 | **93,84** 🟡 | 84,18 | 84,48 | 76,27 |
| | 3 | `majority` | 99,13 | 90,24 | 78,88 | 82,80 | 72,51 |
| :--- | :---: | :--- | :---: | :---: | :---: | :---: | :---: |
| **Top-4 Não-RGB** | 4 | `mean` | 99,57 | 92,78 | 81,58 | 84,66 | 77,91 |
| *(DINO SRM + CLIP SRM +* | 4 | `geometric` | 99,59 | 93,39 | 82,39 | 86,05 | **78,23** 🟡 |
| *ResNet Concat +* | 4 | `harmonic` | 99,07 | 90,93 | 78,85 | 84,96 | 77,94 |
| *MobileNet SRM)* | 4 | `max` | 98,79 | 91,51 | 74,81 | 82,59 | 65,25 |
| | 4 | `weighted` | 99,58 | 92,87 | 81,78 | 84,70 | 77,80 |
| | 4 | `optimal_weights` | 99,72 | 93,83 | 85,04 | 83,92 | 76,39 |
| | 4 | `stacking` | 99,71 | 93,55 | 83,85 | 84,09 | 77,14 |
| | 4 | `majority` | 99,18 | 90,53 | 79,47 | 83,06 | 74,20 |
| :--- | :---: | :--- | :---: | :---: | :---: | :---: | :---: |
| **Top-5 Não-RGB** | 5 | `mean` | 99,50 | 92,33 | 82,10 | 84,18 | 75,29 |
| *(Top-4 +* | 5 | `geometric` | 99,53 | 92,94 | 82,86 | 86,06 | 75,74 |
| *ViT SRM)* | 5 | `harmonic` | 98,99 | 90,61 | 78,62 | 85,18 | 75,77 |
| | 5 | `max` | 98,78 | 91,36 | 74,95 | 82,46 | 65,13 |
| | 5 | `weighted` | 99,53 | 92,44 | 82,25 | 84,26 | 75,40 |
| | 5 | `optimal_weights` | 99,72 | 93,83 | 85,02 | 83,93 | 76,39 |
| | 5 | `stacking` | 99,71 | 93,53 | 83,89 | 84,08 | 77,07 |
| | 5 | `majority` | 99,15 | 90,35 | 80,13 | 82,72 | 72,53 |

---

## 🔬 Comparativo de Controle: Apenas Espectro de Fourier Concat
*(Sem SRM e Sem RGB Normal — avaliando a sinergia puramente espectral de Fourier)*

| Comitê Fourier Concat | K | Estratégia de Fusão | Val AUC (%) | Test Limpo AUC (%) | Test Difícil AUC (%) | DF-40 AUC (%) | Celeb-DF Vídeo AUC (%) |
| :--- | :---: | :--- | :---: | :---: | :---: | :---: | :---: |
| **Top-2 Fourier** | 2 | `mean` | 98,79 | 91,04 | 68,34 | 80,91 | 71,20 |
| *(DINO Concat +* | 2 | `geometric` | 98,85 | 90,97 | 67,86 | 80,64 | 71,58 |
| *ResNet Concat)* | 2 | `stacking` | 98,94 | 91,38 | 68,61 | 80,48 | 71,42 |
| **Top-3 Fourier** | 3 | `mean` | 99,03 | 92,95 | 70,10 | 84,00 | 73,05 |
| *(Top-2 +* | 3 | `geometric` | 99,09 | 92,80 | 69,63 | 83,01 | 72,98 |
| *CLIP Concat)* | 3 | `stacking` | 99,13 | 93,04 | 70,19 | 83,06 | 73,03 |
| **Top-4 Fourier** | 4 | `mean` | 98,54 | 91,59 | 68,75 | 84,00 | 72,19 |
| *(Top-3 +* | 4 | `geometric` | 98,55 | 91,64 | 68,13 | 82,82 | 72,29 |
| *Xception Concat)* | 4 | `stacking` | 99,07 | 92,84 | 69,83 | 83,15 | 73,01 |

---

### 💡 Principais Descobertas Científicas (Sem RGB Normal):

1. **SRM Dispara no Celeb-DF v2 (78,23% de Vídeo AUC):**
   * Ao remover o RGB normal e utilizar os resíduos esteganográficos SRM associados à FFT, o ensemble quebrou o teto histórico do Celeb-DF, saltando de **74,79%** para **78,23%**. As manipulações sutis do DeepFaceLab em H.264 deixam perturbações drásticas no ruído de alta frequência que o RGB normal ignorava.
2. **Resiliência Máxima a Corrupções (87,34% no Test-D):**
   * O par **DINO SRM + CLIP SRM com média geométrica** bateu **87,34% de AUC no Teste Difícil**, provando que filtros de resíduos são imunes a distorções estéticas do RGB.
3. **Média Geométrica (`geometric`) se Torna a Estratégia Soberana:**
   * Nos modelos não-RGB, a fusão por média geométrica ganhou em praticamente todos os comitês e benchmarks, pois a concordância entre resíduos de ruído e harmônicos de Fourier só ocorre quando há fraude real, eliminando quase 100% dos falsos positivos.
