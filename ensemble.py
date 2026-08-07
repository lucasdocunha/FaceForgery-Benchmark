from __future__ import annotations
import argparse,itertools
from pathlib import Path
import numpy as np,pandas as pd
from src.data.paths import models_root
from src.pipelines.checkpoints import discover_trained_runs
from src.pipelines.ensemble_strategies import STRATEGIES
from src.pipelines.evaluation import safe_auc

def load_candidates(root,pool):
    items=[]
    for run in discover_trained_runs(root):
        outputs={s:run.run_dir/"results"/f"outputs_{s}.npz" for s in ("val","test","test_d")}
        if all(p.exists() for p in outputs.values()): items.append((run,outputs))
    if pool=="best-mode":
        best={}
        for item in items:
            run,paths=item; auc=safe_auc(np.load(paths["val"])["y_true"],np.load(paths["val"])["probs"]); key=(run.model_family,run.regime)
            if key not in best or auc>best[key][0]: best[key]=(auc,item)
        items=[x[1] for x in best.values()]
    return items
def search_subset(predictions,y,strategy,exhaustive=True):
    n=len(predictions); candidates=[]
    if exhaustive:
        candidates=(combo for size in range(2,n+1) for combo in itertools.combinations(range(n),size))
    else:
        selected=[]
        while len(selected)<n:
            choice=max((i for i in range(n) if i not in selected),key=lambda i:safe_auc(y,STRATEGIES[strategy]([predictions[j] for j in selected+[i]])))
            selected.append(choice); candidates.append(tuple(selected))
    return max(candidates,key=lambda c:safe_auc(y,STRATEGIES[strategy]([predictions[i] for i in c])))
def run(root,pool="best-mode",strategy="mean",output_dir="."):
    candidates=load_candidates(root,pool)
    if not candidates: raise ValueError("No candidates with val/test/test_d predictions")
    val=[np.load(paths["val"])["probs"] for _,paths in candidates]; y=np.load(candidates[0][1]["val"])["y_true"]
    chosen=search_subset(val,y,"mean",pool=="best-mode") if strategy=="search" else tuple(range(len(candidates))); combine="mean" if strategy=="search" else strategy
    val_arrays=[np.load(candidates[i][1]["val"]) for i in chosen]
    weights=[safe_auc(a["y_true"],a["probs"]) for a in val_arrays]
    out=Path(output_dir); out.mkdir(parents=True,exist_ok=True); rows=[]
    for split in ("val","test","test_d"):
        arrays=[np.load(candidates[i][1][split]) for i in chosen]; labels=arrays[0]["y_true"]
        if any(not np.array_equal(arrays[0]["ids"],a["ids"]) for a in arrays[1:]): raise ValueError("Candidate prediction IDs are not aligned")
        if combine=="stacking": probs=STRATEGIES[combine]([a["probs"] for a in val_arrays],val_arrays[0]["y_true"],[a["probs"] for a in arrays])
        elif combine=="weighted": probs=STRATEGIES[combine]([a["probs"] for a in arrays],weights=weights)
        else: probs=STRATEGIES[combine]([a["probs"] for a in arrays])
        pd.DataFrame({"y_true":labels,"prob":probs}).to_csv(out/f"ensemble_predictions_{split}.csv",index=False); rows.append({"split":split,"strategy":combine,"pool":pool,"members":len(chosen),"auc":safe_auc(labels,probs)})
    pd.DataFrame(rows).to_csv(out/"ensemble_report.csv",index=False); return pd.DataFrame(rows)
def main(argv=None):
 p=argparse.ArgumentParser();p.add_argument("--strategy",default="search",choices=tuple(STRATEGIES)+("search",));p.add_argument("--pool",default="best-mode",choices=("best-mode","all"));p.add_argument("--models-root",default=None);p.add_argument("--output-dir",default=".");a=p.parse_args(argv);print(run(a.models_root or models_root(),a.pool,a.strategy,a.output_dir).to_string(index=False))
if __name__=="__main__":main()
