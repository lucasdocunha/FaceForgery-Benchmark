# Comparativo Sistemático de Ensembles (Top-2 ao Top-4 em 4 Categorias)

> Avaliação abrangente comparando comitês otimizados na validação do MFFI.
> Inclui AUC (%) nos 4 benchmarks de teste e a **Revocação no Protocolo 2 do DF-40 (650.549 imagens falsas)** com limiar ótimo da validação.

| Categoria | Comitê | Membros Integrantes | Fusão | Val AUC (%) | Test AUC (%) | Test-Hard AUC (%) | Celeb-DF Vídeo (%) | DF-40 AUC (%) | DF-40 P2 Revoc. (%) |
| :--- | :---: | :--- | :--- | :---: | :---: | :---: | :---: | :---: | :---: |
| Espacial Robusto | Top-2 | DINOv3 RGB + CLIP RGB | `média geométrica` | 99,7% | 94,7% | **89,0%** | 67,3% | 81,2% | **88,2%** |
| Espacial Robusto | Top-3 | DINOv3 RGB + CLIP RGB + MobileNetV3 RGB | `pesos ótimos (SLSQP)` | 99,7% | 94,2% | 87,3% | 65,6% | 80,6% | 81,8% |
| Espacial Robusto | Top-4 | DINOv3 RGB + CLIP RGB + MobileNetV3 RGB + Xception RGB | `pesos ótimos (SLSQP)` | 99,7% | 94,2% | 87,3% | 65,6% | 80,6% | 81,8% |
| Espacial Convencional | Top-2 | DINOv3 RGB + CLIP RGB | `média geométrica` | 99,8% | 94,9% | 83,4% | 65,4% | 78,7% | 77,7% |
| Espacial Convencional | Top-3 | DINOv3 RGB + CLIP RGB + ResNet-18 RGB | `stacking (reg. log.)` | **99,8%** | 94,6% | 80,9% | 70,6% | 77,1% | 72,3% |
| Espacial Convencional | Top-4 | DINOv3 RGB + CLIP RGB + ResNet-18 RGB + MobileNetV3 RGB | `stacking (reg. log.)` | **99,8%** | 94,6% | 80,6% | 70,8% | 77,2% | 74,6% |
| Híbrido Multidomínio | Top-2 | DINOv3 RGB+SRM + DINOv3 RGB+Fourier | `stacking (reg. log.)` | **99,8%** | 94,6% | 80,1% | 73,2% | 77,4% | 80,5% |
| Híbrido Multidomínio | Top-3 | DINOv3 RGB+SRM + DINOv3 RGB+Fourier + CLIP RGB+SRM | `média geométrica` | **99,8%** | **95,1%** | 82,8% | 73,8% | **81,8%** | 84,5% |
| Híbrido Multidomínio | Top-4 | DINOv3 RGB+SRM + DINOv3 RGB+Fourier + CLIP RGB+SRM + ResNet-18 RGB+Fourier | `pesos ótimos (SLSQP)` | **99,8%** | 94,8% | 82,6% | 74,0% | 79,7% | 84,2% |
| Não-RGB (SRM e Frequência) | Top-2 | ResNet-18 SRM-only + DINOv3 Fourier 3-Bandas | `stacking (reg. log.)` | 96,4% | 85,3% | 68,9% | 72,4% | 63,7% | 85,1% |
| Não-RGB (SRM e Frequência) | Top-3 | ResNet-18 SRM-only + DINOv3 SRM-only + DINOv3 Magnitude | `stacking (reg. log.)` | 96,8% | 84,8% | 70,6% | **78,4%** | 64,0% | 82,2% |
| Não-RGB (SRM e Frequência) | Top-4 | ResNet-18 SRM-only + DINOv3 SRM-only + DINOv3 Magnitude + DINOv3 Fourier 3-Bandas | `stacking (reg. log.)` | 96,9% | 85,5% | 69,9% | 77,4% | 63,7% | 83,6% |
