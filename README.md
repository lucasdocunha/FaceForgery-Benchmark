# FaceForgery Benchmark — Robustness, Generalization & Explainability

[![Python 3.10+](https://img.shields.io/badge/python-3.10%20%7C%203.11%20%7C%203.12-blue.svg)](https://www.python.org/)
[![PyTorch 2.6+](https://img.shields.io/badge/PyTorch-2.6+-ee4c2c.svg)](https://pytorch.org/)
[![Tests Passing](https://img.shields.io/badge/tests-209%20passed-brightgreen.svg)]()
[![Hardware](https://img.shields.io/badge/GPU-Dual%20RTX%203090%2024GB-76b900.svg)]()
[![License](https://img.shields.io/badge/license-MIT-green.svg)](LICENSE)

Framework unificado de pesquisa para detecção forense de manipulações faciais (*Deepfakes*), avaliando robustez sob degradações realistas, generalização cruzada entre geradores (*out-of-distribution*) e explicabilidade axiomática de decisões neurais.

Desenvolvido no âmbito do Programa de Pós-Graduação em Informática (**PPGIA**) da Pontifícia Universidade Católica do Paraná (**PUCPR**).

---

## 📌 Escopo e Contribuições Científicas

1. **Benchmark de Arquiteturas:** Avaliação de 6 famílias neurais representativas:
   - **CNNs:** ResNet-18, MobileNetV3-Large, Xception.
   - **Vision Transformers:** ViT-B/16 (ImageNet-21k), CLIP ViT-B/16 (OpenAI multimodal).
   - **Auto-Supervisionado:** DINOv3 (ConvNeXt-Tiny).
2. **Representações Frequenciais (2D-FFT):** Análise comparativa entre domínio espacial (RGB puro) e 6 variantes de transformada de Fourier (`magnitude`, `phase`, `complex`, `frequency_3`, `concat`, `concat_frequency`).
3. **Treinamento Robusto Estocástico (`RandomizedRobustAugment`):** Pipeline de aumentos severos (compressão H.264/JPEG, desfoques gaussianos/movimento, ruídos isotrópicos) avaliado em 5 sementes canônicas (42, 123, 2024, 7, 2025).
4. **Generalização Cross-Dataset:**
   - **DeepFake-40 (DF-40):** 40 métodos geradores contemporâneos (Midjourney, SDXL, Flux, SimSwap, etc.).
   - **Celeb-DF v2:** Avaliação temporal agregada por vídeo com **72.60% de Vídeo AUC** no comitê robusto.
5. **Explicabilidade e Interpretabilidade (XAI):** Módulo com 12 métodos (Grad-CAM, Integrated Gradients, Occlusion, etc.) e análise decomposta em anéis de frequência Fourier.
6. **Rigor Estatístico e Proveniência:** Intervalos de confiança de 95% via Cluster-Bootstrap pareado e hashes SHA-256 de auditoria.

---

## 📑 Navegação Rápida do Repositório

### 1. Resultados Oficiais e Tabelas Canônicas ([`results/mostrar_rayson/`](results/mostrar_rayson/))
- [**Tabela 1: Modelos Individuais Finetune**](results/mostrar_rayson/tabela1-resultados-modelos-finetune.md) — 42 combinações (6 modelos $\times$ 7 modos Fourier) com $\mu \pm \sigma$.
- [**Tabela 2: Ensembles Baseline (Clean)**](results/mostrar_rayson/tabela2-resultados-ensemble.md) — Fusões de 2 a 6 modelos sem treino robusto.
- [**Tabela 3: Mixture of Experts (MoE)**](results/mostrar_rayson/tabela3-resultados-moe.md) — MoE Standard vs MoE Frequencial 7C.
- [**Tabela 4: Modelos Robustos RGB (5 Sementes)**](results/mostrar_rayson/tabela4-seedrobusta-rgb.md) — Todas as 5 sementes canônicas concluídas.
- [**Tabela 5: Cross-Dataset no DF-40**](results/mostrar_rayson/tabela5-crossdata-df40.md) — Ranking de 40 geradores modernos.
- [**Tabela 6: Cross-Dataset no Celeb-DF v2**](results/mostrar_rayson/tabela6-crossdata-celebdf.md) — Frame e Vídeo AUC com polaridade corrigida.
- [**Tabela 7: Ensembles de Modelos Robustos**](results/mostrar_rayson/tabela7-ensemble-robusto.md) — Comitês campeões atingindo **72.60% Vídeo AUC** no Celeb-DF.

### 2. Pacote de Publicação e Defesa da Banca
- [**Artigo Principal (IEEE / SIBGRAPI 6 páginas)**](paper/six-page/): Código LaTeX em [`paper/six-page/main.tex`](paper/six-page/main.tex) e PDF compilado em [`paper/six-page/main.pdf`](paper/six-page/main.pdf).
- [**Artigo Estendido com XAI (10 páginas)**](paper/explicability/): Código LaTeX em [`paper/explicability/main.tex`](paper/explicability/main.tex) e PDF compilado em [`paper/explicability/main.pdf`](paper/explicability/main.pdf).
- [**Apresentação da Banca**](presentation/): Slides em HTML interativo ([`presentation/apresentacao.html`](presentation/apresentacao.html)), PPTX e PDF.
- [**Roteiro Narrativo da Apresentação**](presentation/roteiro.md): Roteiro de fala minuto a minuto detalhando o que explicar em cada slide para a banca.

### 3. Documentação Técnica Aprofundada ([`docs/`](docs/))
- [**Análise Técnica dos Modelos**](docs/analise_tecnica_modelos_deepfake.md): Relatório técnico dissecando as 6 arquiteturas.
- [**Auditoria do Dataset DF-40**](docs/df40-audit.md): Protocolos e escopo dos métodos generativos do DF-40.
- [**Guia de Explicabilidade (XAI)**](docs/explicability.md): Manual dos métodos de atribuição e renderização.
- [**Guia do Piloto XAI**](docs/pilot-xai.md): Instruções para geração de coortes determinísticas.
- [**Manual de Robustez**](docs/robustness.md): Guia de execução de pipelines de robustez e calibração.
- [**Auditoria de Proveniência**](research/robustness/AUDIT.md): Diagnóstico e resolução dos bugs históricos de avaliação.

---

## 🚀 Como Executar

### Ambiente e Dependências
```bash
conda activate cae
pip install -r requirements.txt
```

### Execução da Suíte de Testes
```bash
pytest tests/
```
*Garante conformidade de todos os 209 testes unitários e de integração.*

### Treinamento de Modelos
```bash
# Treino padrão
python train.py --config configs/resnet.yaml --fourier none --regime finetune --seed 42

# Treino com robustez estocástica
python train.py --config configs/resnet.yaml --fourier none --regime finetune_robust --robust --seed 42
```

### CLI de Pesquisa Unificada
```bash
# Visualizar comandos de auditoria, calibração e avaliação
python research_cli.py --help

# Avaliação com calibração estrita
python research_cli.py evaluate-legacy --help
```

---

## 🏛️ Estrutura do Repositório

```text
├── configs/                     # Arquivos YAML de configuração de modelos e treinos
├── data/                        # Manifestos de datasets (raw_min, df40, celeb_df)
├── docs/                        # Manuais e documentação técnica aprofundada
├── paper/                       # Manuscritos LaTeX (versões 6 e 10 páginas)
├── presentation/                # Apresentação para a banca (HTML, PPTX, PDF, roteiro)
├── research/                    # Racionais científicos e relatórios de auditoria
├── research_cli.py              # CLI de pesquisa, auditoria e proveniência
├── results/                     # Resultados consolidados e tabelas oficiais (mostrar_rayson/)
├── scripts/                     # Scripts auxiliares de treino, ensemble e avaliação
├── src/                         # Código-fonte principal
│   ├── data/                   # DataLoaders, augmentations e transformada 2D-FFT
│   ├── explicability/          # Pipeline de explicabilidade e atribuição (XAI)
│   ├── models/                 # Construtores de modelos, MoE e regimes
│   ├── pipelines/              # Loops de treino, avaliação e checkpoints
│   └── robustness/             # Engine de auditoria, proveniência e estatística pareada
├── tables/                      # Dados brutos das métricas em formato CSV
├── tests/                       # Suíte completa de 209 testes automatizados
└── train.py                     # Script principal de treinamento
```


## Execução no cluster CISIA

Para SLURM/H100, use [o guia CISIA](docs/cisia-hpc.md) e os entrypoints
`scripts/slurm_*.sh`. O guia cobre ambiente CUDA 12.8, caches/checkpoints locais,
publicação versionada, retomada, recuperação de falhas e validações pendentes no
cluster. Não execute treinos ou preparadores de modelos diretamente no Shell Access.
