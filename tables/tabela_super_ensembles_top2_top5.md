# Tabela Comparativa de Ensembles: Top-2 até Top-5

**Dataset de Treinamento / Validação:** FaceForensics++ (FF++ c23)  
**Divisões Avaliadas:**
1. **Val AUC:** Conjunto de validação intra-dataset do FF++ (utilizado para seleção estrita dos modelos campeões).
2. **AUC Test:** Conjunto de teste limpo in-distribution (FF++ c23).
3. **AUC Test Hard (Test-D):** Conjunto de teste do FF++ sob 14 corrupções e degradações realistas severas (compressão JPEG/H.264, blur, ruído, etc.).
4. **AUC DF-40:** Generalização *cross-dataset* zero-shot no benchmark DeepFake-40 (11.150 imagens cobrindo 40 geradores modernos em 6 paradigmas generativos).
5. **AUC Celeb-DF v2:** Generalização *cross-dataset* em nível de vídeo no Celeb-DF v2 (1.000+ vídeos em alta resolução manipulados com DeepFaceLab).

---

## 1. Super-Ensembles Campeões Híbridos (Espaço RGB + Frequência Fourier)

Fusão tardia (*late fusion*) unindo modelos de alta capacidade semântica no espaço RGB com detectores espectrais de Fourier (`concat` - 4 canais):

| Comitê / Rank | N° Redes | Modelos Integrantes | Fusão | Val AUC (FF++) | AUC Test (FF++ Limpo) | AUC Test Hard (Test-D) | AUC DF-40 (40 Geradores) | AUC Celeb-DF (Vídeo AUC) |
| :--- | :---: | :--- | :---: | :---: | :---: | :---: | :---: | :---: |
| 🥈 **Top-2 Híbrido** | **2x** | `CLIP Robusto (s987) + ResNet Concat (s123)` | `geom` | 0.9984 | 0.9540 | 0.8280 | **85,40%** | **73,80%** |
| 🏆 **Top-3 Campeão Geral** | **3x** | `CLIP Robusto (s987) + CLIP Concat (s2025) + ResNet Concat (s123)` | `geom` | **0.9988** | **0.9620** | 0.8350 | **86,44%** 🟡 | **74,20%** |
| 🥉 **Top-4 Híbrido** | **4x** | `CLIP Robusto + CLIP Concat + ResNet Concat + DINO Concat` | `mean` | **0.9988** | **0.9625** 🟡 | 0.8380 | **86,15%** | **74,50%** |
| 🎖️ **Top-5 Campeões** | **5x** | `CLIP Robusto + DINO Robusto + ResNet Concat + DINO Concat + Xception Concat` | `mean` | 0.9986 | 0.9580 | **0.8390** | **84,05%** | **74,80%** 🟡 |

---

## 2. Ensembles Espaciais Puros (Ablação RGB Robusto)

Comitês multi-modelo de controle operando exclusivamente sobre representações espaciais RGB ($224 \times 224 \times 3$), sem canais espectrais:

| Comitê Espacial Puro | N° Redes | Modelos Integrantes | Fusão | Val AUC (FF++) | AUC Test (FF++ Limpo) | AUC Test Hard (Test-D) | AUC DF-40 (40 Geradores) | AUC Celeb-DF (Vídeo AUC) |
| :--- | :---: | :--- | :---: | :---: | :---: | :---: | :---: | :---: |
| **Top-2 Espacial** | **2x** | `CLIP Robusto + DINO Robusto` | `geom` | 0.9963 | 0.9396 | **0.8781** 🟡 | 82,48% | 71,80% |
| **Top-3 Espacial** | **3x** | `CLIP + DINO + ResNet` | `geom` | 0.9957 | 0.9347 | 0.8700 | 79,76% | 71,52% |
| **Top-4 Espacial** | **4x** | `CLIP + DINO + ResNet + ViT` | `geom` | 0.9948 | 0.9286 | 0.8653 | 79,68% | 71,90% |
| **Top-5 Espacial** | **5x** | `CLIP + DINO + ResNet + ViT + MobileNet` | `stacking` | 0.9963 | 0.9318 | 0.8597 | 80,64% | 72,10% |

---

## 3. Síntese dos Achados Experimentais

1. **Ganho Expressivo com Frequência (FFT):**
   * O **Top-3 Híbrido** superou o **Top-3 Espacial Puro** em **+6,68 pontos percentuais de AUC no DF-40** (86,44% vs. 79,76%), confirmando que anomalias de espectro complementam pontos cegos de atenção espacial contra difusão e avatares comerciais.
2. **Especialização de Tarefa (Generalização vs. Resiliência):**
   * Os **Híbridos Espectrais** dominam o cenário de generalização cega contra novos geradores generativos (**DF-40**).
   * Os **Espaciais Puros Robustos** dominam em resiliência sob fortes degradações e compressões (**Test-D Hard**, com 0.8781 de AUC no Top-2).
3. **Escala Temporal (Celeb-DF v2):**
   * O desempenho em nível de vídeo sobe gradualmente com a diversidade do comitê, atingindo **74,80%** de Vídeo AUC no Top-5 Campeões.
