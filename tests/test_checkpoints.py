from pathlib import Path
from src.pipelines.checkpoints import discover_trained_runs
def test_discovers_all_six_families_with_regime_and_seed(tmp_path):
    for family in ("resnet","xception","mobilenet","vit","clip","dino"):
        path=tmp_path/family/"none"/"scratch"/"seed_42"/"weights";path.mkdir(parents=True);(path/"best.pth").touch()
    runs=discover_trained_runs(tmp_path)
    assert {r.model_family for r in runs}=={"resnet","xception","mobilenet","vit","clip","dino"}
    assert all(r.seed==42 and r.regime=="scratch" for r in runs)
