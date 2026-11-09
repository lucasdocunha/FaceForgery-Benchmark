#!/bin/bash -l
#SBATCH --job-name=TCC-15-df40-g0
#SBATCH --partition=dev
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=28G
#SBATCH --output=logs/TCC-15-df40_all_fakes_gpu0_%j.out
#SBATCH --error=logs/TCC-15-df40_all_fakes_gpu0_%j.err

set -euo pipefail
cd /home/lucas.ocunha/tcc

echo "=============================================================================="
echo "🚀 INICIANDO INFERÊNCIA EXAUSTIVA DF-40 NO SLURM (WORKER GPU 0 - CNNs)"
echo "Node: $(hostname) | Data: $(date)"
echo "CUDA_VISIBLE_DEVICES: ${CUDA_VISIBLE_DEVICES:-none}"
echo "=============================================================================="

PYTHON="/home/lucas.ocunha/tcc/.venv/bin/python"

MODELS=(
    "resnet_srm resnet srm /media/ssd2/lucas.ocunha/models-tcc/resnet/srm/finetune_robust/seed_42/weights/best.pth 256"
    "mobilenet_srm mobilenet srm /media/ssd2/lucas.ocunha/models-tcc/mobilenet/srm/finetune_robust/seed_42/weights/best.pth 256"
    "resnet_rgb_robust resnet none /media/ssd2/lucas.ocunha/models-tcc/resnet/none/finetune_robust/seed_42/weights/best.pth 256"
    "mobilenet_rgb_robust mobilenet none /media/ssd2/lucas.ocunha/models-tcc/mobilenet/none/finetune_robust/seed_42/weights/best.pth 256"
    "xception_srm_only xception srm_only /media/ssd2/lucas.ocunha/models-tcc/xception/srm_only/finetune_robust/seed_42/weights/best.pth 128"
    "resnet_srm_only resnet srm_only /media/ssd2/lucas.ocunha/models-tcc/resnet/srm_only/finetune_robust/seed_42/weights/best.pth 256"
    "resnet_concat_freq resnet concat_frequency /media/ssd2/lucas.ocunha/models-tcc/resnet/concat_frequency/finetune/seed_42/weights/best.pth 128"
    "resnet_concat resnet concat /media/ssd2/lucas.ocunha/models-tcc/resnet/concat/finetune/seed_42/weights/best.pth 128"
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
echo "🎉 FINALIZADO WORKER GPU 0 DF-40 EXAUSTIVO!"
echo "=============================================================================="
