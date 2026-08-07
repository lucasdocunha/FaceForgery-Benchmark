import argparse
from src.data.paths import data_root, models_root
from src.pipelines.checkpoints import evaluate_trained_runs
def main(argv=None):
 p=argparse.ArgumentParser();p.add_argument("--models-root");p.add_argument("--data-dir");p.add_argument("--splits",default="val,test");p.add_argument("--test-d-csv");p.add_argument("--test-d-images-dir");p.add_argument("--only-model-family");p.add_argument("--batch-size",type=int,default=32);p.add_argument("--num-workers",type=int,default=0);p.add_argument("--limit-per-split",type=int);a=p.parse_args(argv)
 frame=evaluate_trained_runs(a.models_root or models_root(),a.data_dir or data_root()/"raw",tuple(a.splits.split(",")),a.test_d_csv,a.test_d_images_dir,batch_size=a.batch_size,num_workers=a.num_workers,only_model_family=a.only_model_family,limit_per_split=a.limit_per_split);print(f"Saved {len(frame)} rows")
if __name__=="__main__":main()
