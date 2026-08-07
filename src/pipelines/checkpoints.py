from __future__ import annotations
from dataclasses import dataclass
from pathlib import Path
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torchvision import transforms
from src.data.data import ImageDataset
from src.data.paths import phase1_split_root
from src.models.registry import MODEL_REGISTRY
from src.pipelines.config import TrainingConfig
from src.pipelines.evaluation import evaluate_classifier

SUPPORTED_FAMILIES=frozenset(MODEL_REGISTRY)
@dataclass(frozen=True)
class TrainedRun:
    model_family:str; fourier_mode:str; regime:str; seed:int; run_dir:Path; weights_path:Path; metadata:dict; threshold:float=.5
@dataclass(frozen=True)
class SplitSpec:
    name:str; csv_path:Path; images_dir:Path

def _metadata(path):
    files=list((path/"results").glob("metrics_*.csv"))
    return pd.read_csv(files[0]).iloc[0].to_dict() if files else {}
def discover_trained_runs(root, only_model_family=None):
    root=Path(root); runs=[]
    for weights in sorted(root.glob("*/*/*/seed_*/weights/best.pth")):
        rel=weights.relative_to(root).parts; family,mode,regime,seed_part=rel[:4]
        if family not in SUPPORTED_FAMILIES or (only_model_family and family!=only_model_family): continue
        run_dir=weights.parent.parent; meta=_metadata(run_dir)
        runs.append(TrainedRun(family,mode,regime,int(seed_part.removeprefix("seed_")),run_dir,weights,meta,float(meta.get("threshold",.5))))
    return runs
def config_from_run(run):
    allowed=TrainingConfig.__dataclass_fields__; values={k:v for k,v in run.metadata.items() if k in allowed and not pd.isna(v)}
    values.update(model_family=run.model_family,fourier_mode=run.fourier_mode,regime=run.regime,seed=run.seed)
    if "seeds" in values and not isinstance(values["seeds"],(list,tuple)): values.pop("seeds")
    return TrainingConfig(**values)
def build_model_from_run(run): return MODEL_REGISTRY[run.model_family].build(config_from_run(run))
def load_model_from_run(run, device="cpu"):
    model=build_model_from_run(run); state=torch.load(run.weights_path,map_location=device,weights_only=True)
    if all(str(k).startswith("module.") for k in state): state={str(k).removeprefix("module."):v for k,v in state.items()}
    model.load_state_dict(state); return model.to(device).eval()
def build_split_specs(data_dir,splits,test_d_csv=None,test_d_images_dir=None):
    specs=[]
    for split in splits:
        if split in {"val","test"}: specs.append(SplitSpec(split,Path(data_dir)/f"{split}.csv",phase1_split_root(split)))
        elif split=="test_d" and test_d_csv and test_d_images_dir: specs.append(SplitSpec(split,Path(test_d_csv),Path(test_d_images_dir)))
        elif split=="test_d": raise ValueError("test_d requires CSV and images directory")
        else: raise ValueError(f"Unknown split: {split}")
    return specs
def evaluate_trained_runs(models_root,data_dir,splits=("val","test","test_d"),test_d_csv=None,test_d_images_dir=None,output_csv=None,batch_size=32,num_workers=0,device=None,only_model_family=None,limit_per_split=None):
    device=torch.device(device or ("cuda" if torch.cuda.is_available() else "cpu")); rows=[]
    for run in discover_trained_runs(models_root,only_model_family):
        model=load_model_from_run(run,device); config=config_from_run(run)
        for spec in build_split_specs(data_dir,splits,test_d_csv,test_d_images_dir):
            tf=transforms.Compose([transforms.Resize((config.image_size,config.image_size)),transforms.ToTensor(),transforms.Normalize([.485,.456,.406],[.229,.224,.225])])
            ds=ImageDataset(spec.csv_path,spec.images_dir,tf,np.inf if limit_per_split is None else limit_per_split,run.fourier_mode,(config.image_size,config.image_size))
            metrics=evaluate_classifier(model,DataLoader(ds,batch_size=batch_size,num_workers=num_workers),nn.CrossEntropyLoss(),device,run.threshold,use_amp=device.type=="cuda")
            result_dir=run.run_dir/"results"; result_dir.mkdir(exist_ok=True); np.savez_compressed(result_dir/f"outputs_{spec.name}.npz",probs=metrics["probs"],y_true=metrics["y_true"],y_pred=metrics["y_pred"],ids=metrics["ids"])
            row={"model_family":run.model_family,"fourier_mode":run.fourier_mode,"regime":run.regime,"seed":run.seed,"split":spec.name,**{k:metrics[k] for k in ("loss","acc","precision","recall","f1","auc","specificity","tp","fp","fn","tn")}}
            pd.DataFrame([row]).to_csv(result_dir/f"metrics_{spec.name}.csv",index=False); rows.append(row)
    frame=pd.DataFrame(rows); out=Path(output_csv or Path(models_root)/"all_metrics_by_split.csv"); out.parent.mkdir(parents=True,exist_ok=True); frame.to_csv(out,index=False); return frame
