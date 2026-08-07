import torch
from torch.utils.data import DataLoader, TensorDataset
from src.pipelines.config import TrainingConfig
from src.pipelines.training import Trainer

class TinyClassifier(torch.nn.Module):
    def __init__(self):
        super().__init__(); self.classifier=torch.nn.Linear(3,2)
    def forward(self,x): return self.classifier(x.mean((2,3)))

def test_trainer_saves_best_final_metrics_and_plots(tmp_path):
    x=torch.randn(8,3,8,8);y=torch.tensor([0,1]*4);ids=torch.arange(8)
    loader=DataLoader(TensorDataset(x,y,ids),batch_size=4)
    config=TrainingConfig(epochs=2,early_stop_patience=1,batch_size=4,num_workers=0,multi_gpu=False)
    result=Trainer(TinyClassifier(),loader,loader,loader,config,tmp_path,device=torch.device("cpu")).fit()
    assert result["y_true"].shape==(8,)
    for path in ("weights/best.pth","weights/final.pth","results/metrics_test.csv","plots/confusion_matrix.png","plots/roc_auc.png"):
        assert (tmp_path/path).exists()
