from __future__ import annotations
import argparse
from pathlib import Path
import torch
from PIL import Image
from torchvision import transforms
from torchvision.utils import save_image
from src.data.paths import output_root
from src.pipelines.checkpoints import TrainedRun,load_model_from_run
from src.plots.heatmap import generate,overlay
def main(argv=None):
 p=argparse.ArgumentParser();p.add_argument("--checkpoint",required=True);p.add_argument("--image",required=True);p.add_argument("--method",default="auto");p.add_argument("--family");p.add_argument("--grid",action="store_true");p.add_argument("--output");a=p.parse_args(argv);w=Path(a.checkpoint);parts=w.parts;seed_i=next(i for i,x in enumerate(parts) if x.startswith("seed_"));family=a.family or parts[seed_i-3];run=TrainedRun(family,parts[seed_i-2],parts[seed_i-1],int(parts[seed_i][5:]),w.parent.parent,w,{},.5);model=load_model_from_run(run);size=299 if family=="xception" else 224;x=transforms.Compose([transforms.Resize((size,size)),transforms.ToTensor(),transforms.Normalize([.485,.456,.406],[.229,.224,.225])])(Image.open(a.image).convert("RGB")).unsqueeze(0);result=overlay(x,generate(model,family,x,a.method));out=Path(a.output or output_root()/"heatmaps"/f"{family}_{w.parent.parent.name}.png");out.parent.mkdir(parents=True,exist_ok=True);save_image(result,out);print(out)
if __name__=="__main__":main()
