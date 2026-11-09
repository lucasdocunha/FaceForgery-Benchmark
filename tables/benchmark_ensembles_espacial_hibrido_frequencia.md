# Benchmark de Ensembles: Espacial vs. Híbrido vs. Frequência (Top-2 a Top-5)

Esta tabela consolida os ensembles otimizados na validação do **MFFI** para responder às três hipóteses formuladas:
1. **Espacial (só RGB):** Mostra o limite de desempenho que a aparência e os artefatos visuais diretos conseguem alcançar.
2. **Híbrido (RGB+SRM e RGB+Fourier):** Avalia se a fusão explícita de aparência espacial com pistas espectrais no mesmo detector melhora a generalização.
3. **Frequência (só SRM e só Fourier):** Testa se os resíduos espectrais e de ruído conseguem se sustentar isoladamente, sem qualquer informação de cor RGB.

| Grupo | Comitê | Membros | Estratégia de Fusão | Val AUC (%) | Test AUC (%) | Test-Hard AUC (%) | DF-40 AUC (%) | Celeb-DF v2 Vídeo (%) |
| :--- | :---: | :--- | :--- | :---: | :---: | :---: | :---: | :---: |
| **Espacial (só RGB)** | Top-2 | DINOv3 RGB + CLIP RGB | `média geométrica` | **99.77** | 94.85 | 83.15 | 78.54 | 66.13 |
| **Espacial (só RGB)** | Top-3 | DINOv3 RGB + CLIP RGB + ResNet-18 RGB | `média geométrica` | **99.79** | 94.72 | 77.36 | 74.75 | 68.80 |
| **Espacial (só RGB)** | Top-4 | DINOv3 RGB + CLIP RGB + ResNet-18 RGB + MobileNetV3 RGB | `stacking (reg. log.)` | **99.78** | 94.48 | 80.26 | 76.29 | 70.06 |
| **Espacial (só RGB)** | Top-5 | DINOv3 RGB + CLIP RGB + ResNet-18 RGB + MobileNetV3 RGB + ViT RGB | `stacking (reg. log.)` | **99.78** | 94.28 | 80.57 | 76.28 | 71.14 |
| **Híbrido (RGB+SRM e RGB+Fourier)** | Top-2 | DINOv3 RGB+SRM + DINOv3 RGB+Fourier | `stacking (reg. log.)` | **99.81** | 94.56 | 80.07 | 77.42 | 73.23 |
| **Híbrido (RGB+SRM e RGB+Fourier)** | Top-3 | DINOv3 RGB+SRM + DINOv3 RGB+Fourier + CLIP RGB+SRM | `média geométrica` | **99.83** | 95.12 | 82.82 | 81.80 | 73.85 |
| **Híbrido (RGB+SRM e RGB+Fourier)** | Top-4 | DINOv3 RGB+SRM + DINOv3 RGB+Fourier + CLIP RGB+SRM + ResNet-18 RGB+Fourier | `pesos ótimos (SLSQP)` | **99.84** | 94.84 | 82.64 | 79.70 | 74.03 |
| **Híbrido (RGB+SRM e RGB+Fourier)** | Top-5 | DINOv3 RGB+SRM + DINOv3 RGB+Fourier + CLIP RGB+SRM + ResNet-18 RGB+Fourier + MobileNetV3 RGB+SRM | `stacking (reg. log.)` | **99.84** | 94.58 | 82.06 | 81.10 | 75.25 |
| **Frequência (só SRM e só Fourier)** | Top-2 | ResNet-18 SRM-only + DINOv3 Fourier 3-Bandas | `stacking (reg. log.)` | **96.42** | 85.27 | 68.87 | 63.67 | 72.40 |
| **Frequência (só SRM e só Fourier)** | Top-3 | ResNet-18 SRM-only + DINOv3 SRM-only + DINOv3 Magnitude | `stacking (reg. log.)` | **96.77** | 84.78 | 70.59 | 63.96 | 78.40 |
| **Frequência (só SRM e só Fourier)** | Top-4 | ResNet-18 SRM-only + DINOv3 SRM-only + DINOv3 Magnitude + DINOv3 Fourier 3-Bandas | `stacking (reg. log.)` | **96.91** | 85.54 | 69.87 | 63.68 | 77.41 |
| **Frequência (só SRM e só Fourier)** | Top-5 | ResNet-18 SRM-only + DINOv3 SRM-only + DINOv3 Magnitude + DINOv3 Fourier 3-Bandas + ResNet-18 Magnitude | `stacking (reg. log.)` | **96.94** | 85.53 | 69.56 | 63.32 | 77.58 |