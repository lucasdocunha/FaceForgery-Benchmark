from __future__ import annotations
import argparse
from pathlib import Path
import pandas as pd
from src.data.paths import models_root,output_root
from src.pipelines.checkpoints import discover_trained_runs
METRICS=("auc","acc","f1","precision","recall","specificity")
def make_tables(root=None,out=None):
 rows=[]
 for run in discover_trained_runs(root or models_root()):
  for path in (run.run_dir/"results").glob("metrics_*.csv"):
   frame=pd.read_csv(path);split=path.stem.removeprefix("metrics_")
   for _,row in frame.iterrows():rows.append({"model_family":run.model_family,"fourier_mode":run.fourier_mode,"regime":run.regime,"split":split,"seed":run.seed,**{m:row.get(m) for m in METRICS}})
 group=["model_family","fourier_mode","regime","split"]
 if rows:
  raw=pd.DataFrame(rows);agg=raw.groupby(group)[list(METRICS)].agg(["mean","std"]).reset_index();agg.columns=["_".join(c).rstrip("_") for c in agg.columns]
 else:agg=pd.DataFrame(columns=group+[f"{metric}_{stat}" for metric in METRICS for stat in ("mean","std")])
 out=Path(out or output_root()/"tables");out.mkdir(parents=True,exist_ok=True);agg.to_csv(out/"results_full.csv",index=False);(out/"results_full.md").write_text(agg.to_markdown(index=False),encoding="utf-8");(out/"results_paper.tex").write_text(agg.to_latex(index=False,float_format="%.3f"),encoding="utf-8");return agg
def main(argv=None):
 p=argparse.ArgumentParser();p.add_argument("--models-root");p.add_argument("--output-dir");a=p.parse_args(argv);print(make_tables(a.models_root,a.output_dir).to_string(index=False))
if __name__=="__main__":main()
