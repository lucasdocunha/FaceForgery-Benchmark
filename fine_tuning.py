"""
Fine-tuning dos pipelines principais a partir dos pesos ORIGINAIS pretreinados de cada
arquitetura (ImageNet para ResNet/MobileNet/Xception/ViT, CLIP da OpenAI, DINOv3/v2),
em vez de treinar do zero como o main.py faz.

Salva tudo em models_finetuned/ (separado de models/) para não sobrescrever os
resultados do treino original.

Ajuste FT_EPOCHS, RAW_MIN, RUN_* e BATCH_SIZE abaixo.

Xception/ViT/CLIP pretreinados só suportam entrada RGB (fourier="none") — os pesos
ImageNet/CLIP são de 3 canais e não têm adaptação para os demais modos de Fourier.
ResNet/MobileNet pretreinados rodam em todos os modos (a adaptação do primeiro conv
para outros números de canais já existe em src/models/{resnet,mobilenet}.py).
"""

from __future__ import annotations
try:
    from dotenv import load_dotenv
except ModuleNotFoundError:
    def load_dotenv() -> bool:
        return False

import logging
import os

# Raiz onde estão as imagens do dataset (trainset/valset/testset) neste servidor.
# Troque aqui ao mudar de máquina — tem prioridade menor que a variável de ambiente
# TCC_DATASET_ROOT, caso ela já esteja definida no ambiente.
DATASET_ROOT = "/datasets/Images/MFFI/"
os.environ.setdefault("TCC_DATASET_ROOT", DATASET_ROOT)

from src.data import ALL_FOURIER_MODES
from src.pipelines.mobilenet import run_mobilenet
from src.pipelines.resnet import run_resnet
from src.pipelines.vit import run_vit
from src.pipelines.xcpetion import run_xception
from src.pipelines.clip import run_clip
from src.pipelines.dino import run_dino

logger = logging.getLogger(__name__)

# Configurações globais de fine-tuning (curto e conservador: poucas épocas, LR baixo)
FT_EPOCHS = 10
RAW_MIN = False
BATCH_SIZE = 64
NUM_WORKERS = 4
MULTI_GPU = True
SEED = 26
UNFREEZE_LAST_N = 2
EARLY_STOP_PATIENCE = 3

# Pasta de saída separada da usada pelo main.py — não sobrescreve os pesos/resultados
# do treino original em models/.
OUTPUT_ROOT = "models_finetuned"

# Flags de ativação dos modelos
RUN_XCEPTION = True
RUN_RESNET = True
RUN_MOBILENET = True
RUN_VIT = True
RUN_CLIP = True
RUN_DINO = True

# Configurações do Multiprocessamento por GPU
MULTIPROCESS = True  # Se True, treina em paralelo usando processos separados para cada GPU
GPUS = None  # Lista de GPUs físicas a usar (ex: [0, 1]). Se None, auto-detecta todas as disponíveis.


def main() -> None:
    load_dotenv()
    logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(name)s: %(message)s")

    tasks = []

    # 1. Xception (pretreinado real via timm) — só RGB
    if RUN_XCEPTION:
        tasks.append({
            "fn": run_xception,
            "name": "Xception_finetune_rgb",
            "kwargs": {
                "fourier": "none",
                "epochs": FT_EPOCHS,
                "raw_min": RAW_MIN,
                "data_limit": float("inf"),
                "output_root": OUTPUT_ROOT,
                "batch_size": BATCH_SIZE,
                "image_size": 224,
                "num_workers": NUM_WORKERS,
                "pretrained": True,
                "allow_pretrained": True,
                "unfreeze_last_n": UNFREEZE_LAST_N,
                "learning_rate_head": 1e-4,
                "learning_rate_backbone": 1e-5,
                "early_stop_patience": EARLY_STOP_PATIENCE,
                "multi_gpu": MULTI_GPU,
                "seed": SEED,
            }
        })

    # 2. ResNet (pretreinado real ImageNet via torchvision) — todos os modos de Fourier
    if RUN_RESNET:
        for mode in ALL_FOURIER_MODES:
            tasks.append({
                "fn": run_resnet,
                "name": f"ResNet_finetune_{mode}",
                "kwargs": {
                    "epochs": FT_EPOCHS,
                    "raw_min": RAW_MIN,
                    "architecture": "resnet18",
                    "image_size": 224,
                    "batch_size": BATCH_SIZE,
                    "num_workers": NUM_WORKERS,
                    "pretrained": True,
                    "allow_pretrained": True,
                    "train_backbone": True,
                    "train_layer3": False,
                    "learning_rate_head": 1e-4,
                    "learning_rate_backbone": 1e-5,
                    "use_weighted_sampler": True,
                    "use_class_weights": False,
                    "threshold_strategy": "accuracy",
                    "fourier": mode,
                    "data_limit": float("inf"),
                    "early_stop_patience": EARLY_STOP_PATIENCE,
                    "output_root": OUTPUT_ROOT,
                    "multi_gpu": MULTI_GPU,
                    "seed": SEED,
                }
            })

    # 3. MobileNet (pretreinado real ImageNet via torchvision) — todos os modos de Fourier
    if RUN_MOBILENET:
        for mode in ALL_FOURIER_MODES:
            tasks.append({
                "fn": run_mobilenet,
                "name": f"MobileNet_finetune_{mode}",
                "kwargs": {
                    "epochs": FT_EPOCHS,
                    "raw_min": RAW_MIN,
                    "variant": "large",
                    "input_mode": mode,
                    "image_size": 224,
                    "batch_size": BATCH_SIZE,
                    "num_workers": NUM_WORKERS,
                    "pretrained": True,
                    "allow_pretrained": True,
                    "use_weighted_sampler": True,
                    "use_class_weights": False,
                    "last_n_blocks": 2,
                    "warmup_epochs": 0,
                    "learning_rate_classifier": 1e-4,
                    "learning_rate_backbone": 3e-6,
                    "threshold_metric": "accuracy",
                    "data_limit": None,
                    "early_stop_patience": EARLY_STOP_PATIENCE,
                    "output_root": OUTPUT_ROOT,
                    "multi_gpu": MULTI_GPU,
                    "seed": SEED,
                }
            })

    # 4. ViT (pretreinado real ImageNet via timm, vit_base_patch16_224) — só RGB
    if RUN_VIT:
        tasks.append({
            "fn": run_vit,
            "name": "ViT_finetune_rgb",
            "kwargs": {
                "fourier": "none",
                "epochs": FT_EPOCHS,
                "raw_min": RAW_MIN,
                "batch_size": BATCH_SIZE,
                "num_workers": NUM_WORKERS,
                "image_size": 224,
                "pretrained": True,
                "unfreeze_last_n": UNFREEZE_LAST_N,
                "learning_rate_classifier": 1e-4,
                "learning_rate_backbone": 1e-5,
                "threshold_metric": "f1",
                "early_stop_patience": EARLY_STOP_PATIENCE,
                "output_root": OUTPUT_ROOT,
                "multi_gpu": MULTI_GPU,
                "seed": SEED,
            }
        })

    # 5. CLIP (pretreinado real OpenAI via Hugging Face) — só RGB
    if RUN_CLIP:
        tasks.append({
            "fn": run_clip,
            "name": "CLIP_finetune_rgb",
            "kwargs": {
                "epochs": FT_EPOCHS,
                "raw_min": RAW_MIN,
                "batch_size": BATCH_SIZE,
                "image_size": 224,
                "num_workers": NUM_WORKERS,
                "pretrained": True,
                "train_backbone": True,
                "last_n_layers": UNFREEZE_LAST_N,
                "learning_rate_head": 5e-5,
                "learning_rate_backbone": 1e-5,
                "early_stop_patience": EARLY_STOP_PATIENCE,
                "output_root": OUTPUT_ROOT,
                "multi_gpu": MULTI_GPU,
                "seed": SEED,
            }
        })

    # 6. DINO (já é pretreinado por padrão) — destrava as últimas UNFREEZE_LAST_N camadas
    if RUN_DINO:
        tasks.append({
            "fn": run_dino,
            "name": "DINO_finetune_rgb",
            "kwargs": {
                "dino_version": "v3",
                "model_size": "base",
                "epochs": FT_EPOCHS,
                "raw_min": RAW_MIN,
                "image_size": 224,
                "batch_size": BATCH_SIZE,
                "num_workers": NUM_WORKERS,
                "freeze_backbone": True,
                "unfreeze_last_n": UNFREEZE_LAST_N,
                "learning_rate_classifier": 1e-4,
                "learning_rate_backbone": 1e-5,
                "early_stop_patience": EARLY_STOP_PATIENCE,
                "output_root": OUTPUT_ROOT,
                "multi_gpu": MULTI_GPU,
                "seed": SEED,
            }
        })

    if not tasks:
        logger.info("Nenhuma tarefa de fine-tuning selecionada para execução.")
        return

    # Executa as tarefas
    if MULTIPROCESS:
        logger.info(f"Iniciando {len(tasks)} tarefas de fine-tuning via MULTIPROCESSAMENTO em paralelo...")
        from src.utils.multiprocess import run_tasks_on_gpus
        run_tasks_on_gpus(tasks, gpus=GPUS)
    else:
        logger.info(f"Iniciando {len(tasks)} tarefas de fine-tuning SEQUENCIALMENTE...")
        for task in tasks:
            fn = task["fn"]
            name = task["name"]
            args = task.get("args", ())
            kwargs = task.get("kwargs", {})
            logger.info("========================================")
            logger.info(f"Iniciando tarefa sequencial: {name}")
            logger.info("========================================")
            fn(*args, **kwargs)


if __name__ == "__main__":
    import multiprocessing as mp
    # Configura o método de inicialização 'spawn' como padrão seguro para PyTorch + CUDA
    mp.set_start_method("spawn", force=True)
    main()
