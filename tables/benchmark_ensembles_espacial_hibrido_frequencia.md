# Benchmark de Ensembles: Espacial Robusto vs. Espacial Convencional vs. Híbrido vs. Frequência (Top-2 a Top-5)

Esta tabela consolida os ensembles otimizados na partição de validação do **MFFI** para responder às hipóteses centrais da pesquisa:
1. **Espacial Robusto (RGB Robusto):** Avalia o teto da representação espacial quando submetida a data augmentation robusta (anti-degradação).
2. **Espacial Convencional (RGB Convencional):** Mede o comportamento dos modelos puramente espaciais sem aumento de robustez sob degradações do mundo real.
3. **Híbrido (RGB+SRM e RGB+Fourier):** Avalia se a fusão explícita de aparência espacial com pistas espectrais no mesmo detector melhora a generalização.
4. **Frequência (só SRM e só Fourier):** Testa se os resíduos espectrais e de ruído conseguem se sustentar isoladamente, sem qualquer informação de cor RGB.

| Grupo | Comitê | Membros | Estratégia de Fusão | Val AUC (%) | Test AUC (%) | Test-Hard AUC (%) | DF-40 AUC (%) | Celeb-DF v2 Vídeo (%) |
| :--- | :---: | :--- | :--- | :---: | :---: | :---: | :---: | :---: |
| **Espacial Robusto (RGB Robusto)** | Top-2 | DINOv3 RGB + CLIP RGB | `média geométrica` | **99.73** | 94.72 | 88.98 | 81.16 | 67.31 |
| **Espacial Robusto (RGB Robusto)** | Top-3 | DINOv3 RGB + CLIP RGB + MobileNetV3 RGB | `pesos ótimos (SLSQP)` | **99.71** | 94.20 | 87.30 | 80.62 | 65.59 |
| **Espacial Robusto (RGB Robusto)** | Top-4 | DINOv3 RGB + CLIP RGB + MobileNetV3 RGB + Xception RGB | `pesos ótimos (SLSQP)` | **99.71** | 94.20 | 87.30 | 80.62 | 65.58 |
| **Espacial Robusto (RGB Robusto)** | Top-5 | DINOv3 RGB + CLIP RGB + ViT RGB + MobileNetV3 RGB + Xception RGB | `pesos ótimos (SLSQP)` | **99.71** | 94.20 | 87.31 | 80.61 | 65.58 |
| **Espacial Convencional (RGB Convencional)** | Top-2 | DINOv3 RGB + CLIP RGB | `média geométrica` | **99.77** | 94.87 | 83.40 | 78.71 | 65.41 |
| **Espacial Convencional (RGB Convencional)** | Top-3 | DINOv3 RGB + CLIP RGB + ResNet-18 RGB | `stacking (reg. log.)` | **99.79** | 94.56 | 80.89 | 77.09 | 70.58 |
| **Espacial Convencional (RGB Convencional)** | Top-4 | DINOv3 RGB + CLIP RGB + ResNet-18 RGB + MobileNetV3 RGB | `stacking (reg. log.)` | **99.80** | 94.55 | 80.57 | 77.22 | 70.76 |
| **Espacial Convencional (RGB Convencional)** | Top-5 | DINOv3 RGB + CLIP RGB + ResNet-18 RGB + MobileNetV3 RGB + ViT RGB | `stacking (reg. log.)` | **99.79** | 94.44 | 80.83 | 76.97 | 71.18 |
| **Híbrido (RGB+SRM e RGB+Fourier)** | Top-2 | DINOv3 RGB+SRM + DINOv3 RGB+Fourier | `stacking (reg. log.)` | **99.81** | 94.56 | 80.07 | 77.42 | 73.23 |
| **Híbrido (RGB+SRM e RGB+Fourier)** | Top-3 | DINOv3 RGB+SRM + DINOv3 RGB+Fourier + CLIP RGB+SRM | `média geométrica` | **99.83** | 95.12 | 82.82 | 81.80 | 73.85 |
| **Híbrido (RGB+SRM e RGB+Fourier)** | Top-4 | DINOv3 RGB+SRM + DINOv3 RGB+Fourier + CLIP RGB+SRM + ResNet-18 RGB+Fourier | `pesos ótimos (SLSQP)` | **99.84** | 94.84 | 82.64 | 79.70 | 74.03 |
| **Híbrido (RGB+SRM e RGB+Fourier)** | Top-5 | DINOv3 RGB+SRM + DINOv3 RGB+Fourier + CLIP RGB+SRM + ResNet-18 RGB+Fourier + MobileNetV3 RGB+SRM | `stacking (reg. log.)` | **99.84** | 94.58 | 82.06 | 81.10 | 75.25 |
| **Frequência (só SRM e só Fourier)** | Top-2 | ResNet-18 SRM-only + DINOv3 Fourier 3-Bandas | `stacking (reg. log.)` | **96.42** | 85.27 | 68.87 | 63.67 | 72.40 |
| **Frequência (só SRM e só Fourier)** | Top-3 | ResNet-18 SRM-only + DINOv3 SRM-only + DINOv3 Magnitude | `stacking (reg. log.)` | **96.77** | 84.78 | 70.59 | 63.96 | 78.40 |
| **Frequência (só SRM e só Fourier)** | Top-4 | ResNet-18 SRM-only + DINOv3 SRM-only + DINOv3 Magnitude + DINOv3 Fourier 3-Bandas | `stacking (reg. log.)` | **96.91** | 85.54 | 69.87 | 63.68 | 77.41 |
| **Frequência (só SRM e só Fourier)** | Top-5 | ResNet-18 SRM-only + DINOv3 SRM-only + DINOv3 Magnitude + DINOv3 Fourier 3-Bandas + ResNet-18 Magnitude | `stacking (reg. log.)` | **96.94** | 85.53 | 69.56 | 63.32 | 77.58 |