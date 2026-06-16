"""
Treina os pipelines principais × `ALL_FOURIER_MODES`.

Ajuste EPOCHS, RAW_MIN, RUN_VIT e BATCH_SIZE abaixo.

CLIP local: só RGB; roda uma vez ao final se RUN_VIT for True.
"""

from __future__ import annotations
try:
    from dotenv import load_dotenv
except ModuleNotFoundError:
    def load_dotenv() -> bool:
        return False

import logging

from src.data import ALL_FOURIER_MODES
from src.pipelines.mobilenet import run_mobilenet
from src.pipelines.resnet import run_resnet
from src.pipelines.vit import run_vit
from src.pipelines.xcpetion import run_xception
from src.pipelines.clip import run_clip
from src.pipelines.dino import run_dino

logger = logging.getLogger(__name__)

# Configurações globais de treinamento
EPOCHS = 30
RAW_MIN = False
BATCH_SIZE = 64
NUM_WORKERS = 1
MULTI_GPU = False

# Flags de ativação dos modelos
RUN_XCEPTION = True
RUN_RESNET = False
RUN_MOBILENET = False
RUN_VIT = True
RUN_CLIP = False
RUN_DINO = False

# Configurações do Multiprocessamento por GPU
MULTIPROCESS = False  # Se True, treina em paralelo usando processos separados para cada GPU
GPUS = [0]  # Lista de GPUs físicas a usar (ex: [0, 1]). Se None, auto-detecta todas as disponíveis.


def main() -> None:
    load_dotenv()
    logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(name)s: %(message)s")

    tasks = []

    # 1. Xception
    if RUN_XCEPTION:
        for mode in ALL_FOURIER_MODES:
            tasks.append({
                "fn": run_xception,
                "name": f"Xception_{mode}",
                "kwargs": {
                    "fourier": mode,
                    "epochs": EPOCHS,
                    "raw_min": RAW_MIN,
                    "data_limit": float("inf"),
                    "batch_size": BATCH_SIZE,
                    "image_size": 224,
                    "num_workers": NUM_WORKERS,
                    "pretrained": False,
                    "multi_gpu": MULTI_GPU,
                }
            })

    # 2. ResNet
    if RUN_RESNET:
        for mode in ALL_FOURIER_MODES:
            tasks.append({
                "fn": run_resnet,
                "name": f"ResNet_{mode}",
                "kwargs": {
                    "epochs": EPOCHS,
                    "raw_min": RAW_MIN,
                    "architecture": "resnet18",
                    "image_size": 224,
                    "batch_size": BATCH_SIZE,
                    "num_workers": NUM_WORKERS,
                    "pretrained": False,
                    "use_weighted_sampler": True,
                    "use_class_weights": False,
                    "train_layer3": True,
                    "threshold_strategy": "accuracy",
                    "fourier": mode,
                    "data_limit": float("inf"),
                    "multi_gpu": MULTI_GPU,
                }
            })

    # 3. MobileNet
    if RUN_MOBILENET:
        for mode in ALL_FOURIER_MODES:
            tasks.append({
                "fn": run_mobilenet,
                "name": f"MobileNet_{mode}",
                "kwargs": {
                    "epochs": EPOCHS,
                    "raw_min": RAW_MIN,
                    "variant": "large",
                    "input_mode": mode,
                    "image_size": 224,
                    "batch_size": BATCH_SIZE,
                    "num_workers": NUM_WORKERS,
                    "pretrained": False,
                    "use_weighted_sampler": True,
                    "use_class_weights": False,
                    "last_n_blocks": 4,
                    "learning_rate_backbone": 3e-5,
                    "threshold_metric": "accuracy",
                    "data_limit": None,
                    "multi_gpu": MULTI_GPU,
                }
            })

    # 4. ViT
    if RUN_VIT:
        for mode in ALL_FOURIER_MODES:
            tasks.append({
                "fn": run_vit,
                "name": f"ViT_{mode}",
                "kwargs": {
                    "fourier": mode,
                    "epochs": EPOCHS,
                    "raw_min": RAW_MIN,
                    "batch_size": BATCH_SIZE,
                    "num_workers": NUM_WORKERS,
                    "image_size": 224,
                    "patch_size": 16,
                    "hidden_size": 128,
                    "num_hidden_layers": 3,
                    "num_attention_heads": 4,
                    "dropout": 0.25,
                    "threshold_metric": "f1",
                    "mixup_alpha": 0.2,
                    "multi_gpu": MULTI_GPU,
                }
            })

    # 5. CLIP
    if RUN_CLIP:
        tasks.append({
            "fn": run_clip,
            "name": "CLIP_RGB",
            "kwargs": {
                "epochs": EPOCHS,
                "raw_min": RAW_MIN,
                "batch_size": BATCH_SIZE,
                "image_size": 224,
                "num_workers": NUM_WORKERS,
                "multi_gpu": MULTI_GPU,
            }
        })

    # 6. DINO
    if RUN_DINO:
        tasks.append({
            "fn": run_dino,
            "name": "DINO_RGB",
            "kwargs": {
                "dino_version": "v3",
                "model_size": "base",
                "epochs": EPOCHS,
                "raw_min": RAW_MIN,
                "image_size": 224,
                "batch_size": BATCH_SIZE,
                "num_workers": NUM_WORKERS,
                "multi_gpu": MULTI_GPU,
            }
        })

    if not tasks:
        logger.info("Nenhuma tarefa selecionada para execução.")
        return

    # Executa as tarefas
    if MULTIPROCESS:
        logger.info(f"Iniciando {len(tasks)} tarefas via MULTIPROCESSAMENTO em paralelo...")
        from src.utils.multiprocess import run_tasks_on_gpus
        run_tasks_on_gpus(tasks, gpus=GPUS)
    else:
        logger.info(f"Iniciando {len(tasks)} tarefas SEQUENCIALMENTE...")
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
