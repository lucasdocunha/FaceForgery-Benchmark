import pandas as pd
from make_tables import make_tables
def test_tables_aggregate_three_seeds(tmp_path):
    root=tmp_path/"models"
    for seed,auc in ((42,.7),(123,.8),(2024,.9)):
        run=root/"resnet"/"none"/"scratch"/f"seed_{seed}";(run/"weights").mkdir(parents=True);(run/"weights"/"best.pth").touch();(run/"results").mkdir();pd.DataFrame([{"auc":auc,"acc":auc,"f1":auc,"precision":auc,"recall":auc,"specificity":auc}]).to_csv(run/"results"/"metrics_test.csv",index=False)
    result=make_tables(root,tmp_path/"tables")
    assert abs(result.iloc[0]["auc_mean"]-.8)<1e-9
    assert (tmp_path/"tables"/"results_paper.tex").exists()
