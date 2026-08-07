from pathlib import Path
import shutil

from ensemble import run as run_ensemble
from generate_heatmaps import generate_from_paths
from make_tables import make_tables
from run_matrix import build_tasks
from src.pipelines.checkpoints import evaluate_trained_runs
from train import train_from_config


def test_end_to_end_smoke_workflow(tiny_phase1_dataset, tmp_path, monkeypatch):
    repo = Path(__file__).resolve().parents[1]
    models = tmp_path / "models"
    outputs = tmp_path / "outputs"
    monkeypatch.setenv("TCC_DATA_ROOT", str(repo / "data"))
    monkeypatch.setenv("TCC_MODELS_ROOT", str(models))
    monkeypatch.setenv("TCC_OUTPUT_ROOT", str(outputs))
    configs = tmp_path / "configs"
    configs.mkdir()
    (configs / "base.yaml").write_text(
        "epochs: 1\nbatch_size: 2\nnum_workers: 0\nimage_size: 32\nraw_min: true\n"
        "early_stop_patience: 1\nscheduler_patience: 1\nseeds: [42, 123, 2024]\n"
        "multi_gpu: false\naugment: false\n",
        encoding="utf-8",
    )
    config_path = configs / "resnet.yaml"
    config_path.write_text("model_family: resnet\narchitecture: resnet18\n", encoding="utf-8")

    train_from_config(config_path, fourier="none", regime="scratch", seed=42,
                      epochs=1, data_limit=4, raw_min=True, multi_gpu=False)
    run_dir = models / "resnet" / "none" / "scratch" / "seed_42"
    checkpoint = run_dir / "weights" / "best.pth"
    assert checkpoint.exists()

    summary = evaluate_trained_runs(
        models, repo / "data" / "raw_min", splits=("val", "test"),
        batch_size=2, num_workers=0, limit_per_split=4,
    )
    assert len(summary) == 2 and (models / "all_metrics_by_split.csv").exists()
    shutil.copy2(run_dir / "results" / "outputs_test.npz",
                 run_dir / "results" / "outputs_test_d.npz")
    report = run_ensemble(models, "best-mode", "search", outputs / "ensemble", max_workers=1)
    assert len(report) == 3
    tables = make_tables(models, outputs / "tables")
    assert not tables.empty

    image = next((tiny_phase1_dataset / "trainset").glob("*.jpg"))
    heatmap = generate_from_paths(checkpoint, [image, image], grid_mode=True,
                                  output=outputs / "heatmaps" / "grid.png")
    assert heatmap.exists() and heatmap.stat().st_size > 0


def test_matrix_expands_six_families_seven_modes_three_seeds():
    tasks = build_tasks("scratch")
    assert len(tasks) == 126
