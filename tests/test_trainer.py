import torch
from torch.utils.data import DataLoader, TensorDataset

from src.models.registry import ModelSpec
from src.pipelines.config import TrainingConfig
from src.pipelines.training import Trainer


class TinyClassifier(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.classifier = torch.nn.Linear(3, 2)
    def forward(self, x):
        return self.classifier(x.mean((2, 3)))


def _noop(*_args):
    return None


def test_trainer_saves_all_artifacts_and_stops_early(tmp_path):
    x = torch.randn(8, 3, 8, 8)
    y = torch.tensor([0, 1] * 4)
    loader = DataLoader(TensorDataset(x, y, torch.arange(8)), batch_size=4)
    config = TrainingConfig(
        epochs=5, early_stop_patience=1, scheduler_patience=1, batch_size=4,
        num_workers=0, multi_gpu=False, lr_head=0, lr_backbone=0,
    )
    spec = ModelSpec(lambda _config: TinyClassifier(), _noop, _noop)
    trainer = Trainer(TinyClassifier(), loader, loader, loader, config, tmp_path, spec, device="cpu")
    result = trainer.fit()
    assert result["y_true"].shape == (8,)
    assert trainer.stopped_early and trainer.epochs_completed == 2
    for path in (
        "weights/best.pth", "weights/final.pth", "results/metrics_val.csv",
        "results/metrics_test.csv", "results/outputs_val.npz", "results/outputs_test.npz",
        "results/predictions_val.csv", "results/predictions_test.csv",
        "plots/confusion_matrix.png", "plots/roc_auc.png",
    ):
        assert (tmp_path / path).exists()
