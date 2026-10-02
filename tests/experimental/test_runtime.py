from copy import deepcopy

import pytest
import torch
from torch.utils.data import DataLoader, TensorDataset

from src.experimental.runtime import fit_model


def execute(root, initial, batch_size, accumulation, epochs, resume=False):
    model = torch.nn.Sequential(torch.nn.Linear(2, 4), torch.nn.Tanh(), torch.nn.Linear(4, 1))
    model.load_state_dict(initial)
    x = torch.arange(10, dtype=torch.float32).reshape(5, 2) / 10
    y = x.sum(1, keepdim=True)
    loader = DataLoader(TensorDataset(x, y), batch_size=batch_size)
    optimizer = torch.optim.SGD(model.parameters(), lr=.05)

    def step(m, batch, state):
        a, b = batch
        loss = (m(a) - b).square().mean()
        return loss, {}, len(a)

    def validation(m):
        return {"loss": float((m(x) - y).square().mean())}

    result = fit_model(model, loader, optimizer, loss_step=step, validate=validation,
                       run_dir=root, config={"seed": 42}, epochs=epochs,
                       grad_accum_steps=accumulation, max_grad_norm=None, resume=resume)
    return model.state_dict(), result


def test_accumulation_matches_full_batch_with_partial_window(tmp_path):
    torch.manual_seed(7)
    initial = torch.nn.Sequential(torch.nn.Linear(2, 4), torch.nn.Tanh(), torch.nn.Linear(4, 1)).state_dict()
    a, _ = execute(tmp_path / "whole", initial, 5, 1, 1)
    b, _ = execute(tmp_path / "micro", initial, 2, 4, 1)
    for key in a:
        torch.testing.assert_close(a[key], b[key], atol=1e-7, rtol=1e-6)


def test_epoch_resume_matches_continuous_and_rejects_overwrite(tmp_path):
    torch.manual_seed(11)
    initial = deepcopy(torch.nn.Sequential(torch.nn.Linear(2, 4), torch.nn.Tanh(), torch.nn.Linear(4, 1)).state_dict())
    a, full = execute(tmp_path / "whole", initial, 2, 2, 4)
    execute(tmp_path / "resume", initial, 2, 2, 1)
    b, resumed = execute(tmp_path / "resume", initial, 2, 2, 4, resume=True)
    for key in a:
        torch.testing.assert_close(a[key], b[key], atol=0, rtol=0)
    assert full["history"] == resumed["history"]
    with pytest.raises(FileExistsError):
        execute(tmp_path / "resume", initial, 2, 2, 4)
