import pandas as pd

from make_tables import make_tables


def test_tables_aggregate_three_seeds_and_emit_booktabs_per_split(tmp_path):
    root = tmp_path / "models"
    for seed, auc in ((42, .7), (123, .8), (2024, .9)):
        run = root / "resnet" / "none" / "scratch" / f"seed_{seed}"
        (run / "weights").mkdir(parents=True)
        (run / "weights" / "best.pth").touch()
        (run / "results").mkdir()
        for split in ("val", "test", "test_d"):
            pd.DataFrame([{metric: auc for metric in ("auc", "acc", "f1", "precision", "recall", "specificity", "loss")}]).to_csv(
                run / "results" / f"metrics_{split}.csv", index=False,
            )
    result = make_tables(root, tmp_path / "tables")
    assert len(result) == 3
    assert abs(result.iloc[0]["auc_mean"] - .8) < 1e-9
    latex = (tmp_path / "tables" / "results_paper.tex").read_text(encoding="utf-8")
    assert latex.count("\\toprule") == 3 and "tab:results_test_d" in latex and "$\\pm$" in latex
    assert (tmp_path / "tables" / "results_full.md").exists()
