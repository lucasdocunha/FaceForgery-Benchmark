from __future__ import annotations
import argparse
from pathlib import Path
from src.data.data import ALL_FOURIER_MODES
from src.pipelines.config import load_config
from src.utils.multiprocess import run_tasks_on_gpus
from train import train_from_config

FAMILIES=("resnet","xception","mobilenet","vit","clip","dino")
def build_tasks(regime, only=None):
    families=only or FAMILIES; tasks=[]
    for family in families:
        path=Path("configs")/f"{family}.yaml"; config=load_config(path)
        for mode in ALL_FOURIER_MODES:
            for seed in config.seeds:
                tasks.append({"fn":train_from_config,"name":f"{family}/{mode}/{regime}/seed_{seed}","kwargs":{"config_path":str(path),"fourier":mode,"regime":regime,"seed":seed}})
    return tasks
def main(argv=None):
    p=argparse.ArgumentParser(); p.add_argument("--regime",required=True,choices=("scratch","finetune")); p.add_argument("--only"); p.add_argument("--gpus"); p.add_argument("--workers-per-gpu",type=int,default=1); p.add_argument("--dry-run",action="store_true"); a=p.parse_args(argv)
    tasks=build_tasks(a.regime,a.only.split(",") if a.only else None)
    if a.dry_run:
        print("\n".join(t["name"] for t in tasks)); return
    run_tasks_on_gpus(tasks,gpus=[int(x) for x in a.gpus.split(",")] if a.gpus else None,workers_per_gpu=a.workers_per_gpu)
if __name__ == "__main__": main()
