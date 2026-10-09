#!/bin/bash -l
#SBATCH --job-name=TCC-15-df40-g1
#SBATCH --partition=dev
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=28G
#SBATCH --output=logs/TCC-15-df40_all_fakes_gpu1_%j.out
#SBATCH --error=logs/TCC-15-df40_all_fakes_gpu1_%j.err

set -euo pipefail
cd /home/lucas.ocunha/tcc

echo "=============================================================================="
echo "🚀 INICIANDO INFERÊNCIA EXAUSTIVA DF-40 NO SLURM (WORKER GPU 1 - Transformers)"
echo "Node: $(hostname) | Data: $(date)"
echo "CUDA_VISIBLE_DEVICES: ${CUDA_VISIBLE_DEVICES:-none}"
echo "=============================================================================="

PYTHON="/home/lucas.ocunha/tcc/.venv/bin/python"

MODELS=(
    "clip_srm clip srm /media/ssd2/lucas.ocunha/models-tcc/clip/srm/finetune_robust/seed_42/weights/best.pth 128"
    "xception_srm xception srm /media/ssd2/lucas.ocunha/models-tcc/xception/srm/finetune_robust/seed_42/weights/best.pth 128"
    "vit_srm vit srm /media/ssd2/lucas.ocunha/models-tcc/vit/srm/finetune_robust/seed_42/weights/best.pth 128"
    "clip_rgb_robust clip none /media/ssd2/lucas.ocunha/models-tcc/clip/none/finetune_robust/seed_42/weights/best.pth 128"
    "dino_rgb_robust dino none /media/ssd2/lucas.ocunha/models-tcc/dino/none/finetune_robust/seed_42/weights/best.pth 128"
    "dino_srm_only dino srm_only /media/ssd2/lucas.ocunha/models-tcc/dino/srm_only/finetune_robust/seed_42/weights/best.pth 128"
    "dino_concat_freq dino concat_frequency /media/ssd2/lucas.ocunha/models-tcc/dino/concat_frequency/finetune/seed_42/weights/best.pth 128"
    "dino_concat dino concat /media/ssd2/lucas.ocunha/models-tcc/dino/concat/finetune/seed_42/weights/best.pth 128"
)

mkdir -p tables/df40_all_fakes_models logs

for entry in "${MODELS[@]}"; do
    read -r name fam mode ckpt bs <<< "$entry"
    out_file="tables/df40_all_fakes_models/df40_all_fakes_${name}.csv"
    if [ -f "$out_file" ]; then
        echo "⏭️  Modelo $name já avaliado ($out_file). Pulando..."
        continue
    fi

    echo "------------------------------------------------------------------------------"
    echo "▶️  Executando inferência: $name (Família: $fam, Modo: $mode, BS: $bs)"
    echo "------------------------------------------------------------------------------"

    "$PYTHON" -u scripts/evaluate_df40_all_fakes_generic.py \
        --model-name "$name" \
        --family "$fam" \
        --mode "$mode" \
        --checkpoint "$ckpt" \
        --batch-size "$bs" \
        --num-workers 6 \
        --output-dir "tables/df40_all_fakes_models"

    echo "✅ Concluído: $name"
done

echo "=============================================================================="
echo "🎉 FINALIZADO WORKER GPU 1 DF-40 EXAUSTIVO!"
echo "=============================================================================="
