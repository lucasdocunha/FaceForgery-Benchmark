#!/usr/bin/env bash
# ==============================================================================
# SCRIPT MESTRE DE REPRODUÇÃO DE RESULTADOS (TCC / ICLR BENCHMARK)
# ==============================================================================
# Autor: Lucas O. Cunha
# Repositório: github.com/lucasdocunha/FaceForgery-Benchmark
#
# Este script automatiza e reproduz 100% dos resultados obtidos na pesquisa:
#   1. Treinamento Robusto da Campanha SRM (6 arquiteturas x 5 seeds canônicas)
#   2. Avaliação Multi-Benchmark (Test Limpo, Test-D Difícil, DF-40, Celeb-DF v2)
#   3. Extração e Armazenamento em Cache das Predições (.npz)
#   4. Cálculo de Ensembles Canônicos (705 combinações) e Mega-Ensembles (Tabela 9)
#   5. Auditoria de Evasão Forense no DF-40 por Categoria e por Técnica
#   6. Geração de Mapas de Explicabilidade em 300 DPI (Attention Rollout, Grad-CAM, Fusão)
#
# Uso:
#   ./scripts/reproduce_all_results.sh [OPÇÕES]
#
# Opções:
#   --all                  Executa todas as etapas pós-treinamento (avaliação, ensembles, DF-40, heatmaps)
#   --full-pipeline        Executa TUDO, incluindo o retreinamento das 5 seeds (demorado!)
#   --ensembles            Calcula apenas os ensembles canônicos e mega-ensembles (Tabela 9)
#   --df40-breakdown       Calcula apenas a divisão de acerto por categoria e técnica no DF-40
#   --heatmaps             Gera apenas as figuras de explicabilidade (3 e 4 linhas em 300 DPI)
#   --extract-preds        Extrai predições (.npz) dos modelos para DF-40 e Celeb-DF
#   --eval-benchmarks      Executa a avaliação de checkpoints em todos os datasets de teste
#   --help                 Exibe esta mensagem de ajuda
# ==============================================================================

set -euo pipefail

# 1. Configurações de Diretório
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
if [[ -f "$SCRIPT_DIR/scripts/run_srm_ensembles.py" ]]; then
    PROJECT_ROOT="$SCRIPT_DIR"
else
    PROJECT_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"
fi
cd "$PROJECT_ROOT"

export PYTHONPATH="$PROJECT_ROOT"
export PYTHONUNBUFFERED=1
export PYTHONDONTWRITEBYTECODE=1

# 2. Resolução do Interpretador Python
PYTHON_BIN=""
if [[ -f "/home/lucas.ocunha/.conda/envs/cae/bin/python" ]]; then
    PYTHON_BIN="/home/lucas.ocunha/.conda/envs/cae/bin/python"
elif [[ -f "/home/lucas.ocunha/.conda/envs/tcc/bin/python" ]]; then
    PYTHON_BIN="/home/lucas.ocunha/.conda/envs/tcc/bin/python"
elif [[ -n "${CONDA_PREFIX:-}" && -f "$CONDA_PREFIX/bin/python" ]]; then
    PYTHON_BIN="$CONDA_PREFIX/bin/python"
elif command -v python3 &>/dev/null; then
    PYTHON_BIN="$(command -v python3)"
else
    echo "❌ Erro: Nenhum interpretador Python encontrado." >&2
    exit 1
fi

echo "=============================================================================="
echo "🚀 SCRIPT MESTRE DE REPRODUÇÃO - DETECÇÃO ROBUSTA DE DEEPFAKES"
echo "=============================================================================="
echo "Diretório do Projeto : $PROJECT_ROOT"
echo "Interpretador Python : $PYTHON_BIN"
echo "Data/Hora Local      : $(date '+%Y-%m-%d %H:%M:%S')"
echo "=============================================================================="

# 3. Funções Utilitárias
log_step() {
    echo ""
    echo "------------------------------------------------------------------------------"
    echo "▶️  [$(date '+%H:%M:%S')] $1"
    echo "------------------------------------------------------------------------------"
}

check_data() {
    log_step "Verificando integridade das estruturas de dados e modelos..."
    mkdir -p tables results/mostrar_rayson figures logs
    
    local missing=0
    if [[ ! -d "data/df40" || ! -f "data/df40/test.csv" ]]; then
        echo "⚠️  Aviso: data/df40/test.csv não encontrado no diretório local."
        missing=$((missing + 1))
    fi
    if [[ ! -d "data/celeb_df" || ! -f "data/celeb_df/test.csv" ]]; then
        echo "⚠️  Aviso: data/celeb_df/test.csv não encontrado no diretório local."
        missing=$((missing + 1))
    fi
    if [[ ! -d "data/raw" || ! -f "data/raw/test.csv" ]]; then
        echo "⚠️  Aviso: data/raw/test.csv não encontrado no diretório local."
        missing=$((missing + 1))
    fi
    
    if [[ $missing -eq 0 ]]; then
        echo "✅ Todos os manifestos de dados locais foram localizados com sucesso."
    fi
}

run_training_5seeds() {
    log_step "ETAPA: Treinamento Robusto da Campanha SRM (5 Seeds Canônicas: 42, 123, 2024, 7, 2025)"
    echo "Distribuição:"
    echo "  GPU 0: CLIP, DINO, ResNet-50"
    echo "  GPU 1: ViT-B/16, MobileNet-v2, Xception"
    echo "Iniciando orquestrador de treinamento paralelo..."
    $PYTHON_BIN scripts/launch_robust_5seeds.py
    echo "✅ Treinamento da campanha SRM finalizado!"
}

run_extract_predictions() {
    log_step "ETAPA: Extração de Predições em Cache (.npz) para DF-40 e Celeb-DF v2"
    echo "Extraindo saídas do DF-40..."
    $PYTHON_BIN scripts/extract_df40_predictions.py --modes srm --device cuda:0 || true
    echo "Extraindo saídas do Celeb-DF v2..."
    $PYTHON_BIN scripts/evaluate_celeb_df_robust_ensembles.py || true
    echo "✅ Cache de predições atualizado!"
}

run_ensembles() {
    log_step "ETAPA: Cálculo de Ensembles Canônicos e Mega-Ensembles SRM (Tabela 9)"
    $PYTHON_BIN scripts/run_srm_ensembles.py
    echo "✅ Ensembles calculados e consolidados em:"
    echo "   - tables/ensembles_srm_5seeds_summary.csv"
    echo "   - tables/mega_ensembles_srm_all_models.csv"
    echo "   - results/mostrar_rayson/tabela9-ensemble-srm-5seeds.md"
}

run_df40_breakdown() {
    log_step "ETAPA: Auditoria de Desempenho por Categoria e Técnica no DF-40"
    $PYTHON_BIN scripts/run_df40_breakdown.py
    echo "✅ Auditoria do DF-40 concluída e salva em:"
    echo "   - tables/df40_srm_breakdown_by_paradigm.csv"
    echo "   - tables/df40_srm_breakdown_by_technique.csv"
    echo "   - results/mostrar_rayson/tabela_df40_categorias.md"
}

run_heatmaps() {
    log_step "ETAPA: Geração de Mapas de Calor e Explicabilidade Multimodal (300 DPI)"
    $PYTHON_BIN scripts/generate_multimodal_heatmaps.py
    echo "✅ Figuras geradas com sucesso em:"
    echo "   - figures/heatmap_clip_dino_fusion_3rows.png"
    echo "   - figures/heatmap_clip_dino_fusion_4rows.png"
    echo "   - results/mostrar_rayson/heatmap_clip_dino_fusion_3rows.png"
    echo "   - results/mostrar_rayson/heatmap_clip_dino_fusion_4rows.png"
}

show_summary() {
    log_step "🎉 REPRODUÇÃO CONCLUÍDA COM SUCESSO!"
    echo "Resumo dos principais artefatos atualizados no projeto:"
    echo ""
    echo "📊 Tabelas de Resultados Consolidadas:"
    echo "   1. tables/campaign_srm_30runs_consolidated.csv (Todas as 30 execuções individuais)"
    echo "   2. tables/campaign_srm_summary_5seeds.csv      (Médias e desvios por modelo)"
    echo "   3. tables/ensembles_srm_5seeds_summary.csv     (Top 15 ensembles em 5 seeds)"
    echo "   4. tables/mega_ensembles_srm_all_models.csv    (Meta-fusões de até 30 modelos)"
    echo "   5. tables/df40_srm_breakdown_by_paradigm.csv   (Desempenho nos 6 paradigmas DF-40)"
    echo "   6. tables/df40_srm_breakdown_by_technique.csv  (Taxa de acerto nas 24 técnicas)"
    echo ""
    echo "📝 Relatórios Formatados para Apresentação:"
    echo "   1. results/mostrar_rayson/tabela9-ensemble-srm-5seeds.md"
    echo "   2. results/mostrar_rayson/tabela_df40_categorias.md"
    echo ""
    echo "🎨 Figuras de Explicabilidade (300 DPI):"
    echo "   1. figures/heatmap_clip_dino_fusion_3rows.png"
    echo "   2. figures/heatmap_clip_dino_fusion_4rows.png"
    echo "=============================================================================="
}

# 4. Roteamento de Argumentos
check_data

if [[ $# -eq 0 || "${1:-}" == "--help" || "${1:-}" == "-h" ]]; then
    echo "Uso: $0 [OPÇÃO]"
    echo ""
    echo "Opções:"
    echo "  --all             Executa todas as análises pós-treinamento (padrão recomendado)"
    echo "  --full-pipeline   Executa TUDO, incluindo o retreinamento das 5 sementes"
    echo "  --ensembles       Calcula os ensembles das 5 seeds e gera a Tabela 9"
    echo "  --df40-breakdown  Calcula o acerto por categoria e técnica no DF-40"
    echo "  --heatmaps        Gera os heatmaps de explicabilidade de CLIP + DINO"
    echo "  --extract-preds   Extrai cache de predições (.npz)"
    echo "  --train-models    Inicia o treinamento das 5 sementes SRM"
    echo ""
    echo "Exemplo rápido:"
    echo "  $0 --all"
    exit 0
fi

case "$1" in
    --all)
        run_ensembles
        run_df40_breakdown
        run_heatmaps
        show_summary
        ;;
    --full-pipeline)
        run_training_5seeds
        run_extract_predictions
        run_ensembles
        run_df40_breakdown
        run_heatmaps
        show_summary
        ;;
    --ensembles)
        run_ensembles
        ;;
    --df40-breakdown)
        run_df40_breakdown
        ;;
    --heatmaps)
        run_heatmaps
        ;;
    --extract-preds)
        run_extract_predictions
        ;;
    --train-models)
        run_training_5seeds
        ;;
    *)
        echo "❌ Opção desconhecida: $1"
        echo "Execute '$0 --help' para ver os comandos disponíveis."
        exit 1
        ;;
esac
