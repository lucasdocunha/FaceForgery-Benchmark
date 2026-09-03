import pytest
import torch
import torch.nn as nn

from src.models.registry import MODEL_REGISTRY, get_model_spec
from src.pipelines.config import TrainingConfig


@pytest.mark.parametrize("expert_family", ["mobilenet", "resnet"])
@pytest.mark.parametrize("routing_strategy", ["dense", "top_k"])
def test_moe_frequency_forward(expert_family, routing_strategy):
    config = TrainingConfig(
        model_family="moe_frequency",
        fourier_mode="concat_frequency",
        expert_family=expert_family,
        variant="small" if expert_family == "mobilenet" else "resnet18",
        routing_strategy=routing_strategy,
        top_k=2,
        multi_gpu=False,
    )
    spec = get_model_spec("moe_frequency")
    model = spec.build(config)
    model.eval()

    batch_size = 4
    x = torch.randn(batch_size, 6, 64, 64)
    with torch.no_grad():
        out = model(x)

    assert out.shape == (batch_size, 2)
    assert model.last_routing_weights is not None
    assert model.last_routing_weights.shape == (batch_size, 4)
    # Check that weights sum to 1 per sample
    torch.testing.assert_close(
        model.last_routing_weights.sum(dim=-1),
        torch.ones(batch_size),
        rtol=1e-4,
        atol=1e-4,
    )


@pytest.mark.parametrize("expert_family", ["mobilenet", "resnet"])
@pytest.mark.parametrize("num_experts", [2, 4])
@pytest.mark.parametrize("routing_strategy", ["dense", "top_k"])
def test_moe_standard_forward(expert_family, num_experts, routing_strategy):
    config = TrainingConfig(
        model_family="moe_standard",
        fourier_mode="none",
        num_experts=num_experts,
        expert_family=expert_family,
        variant="small" if expert_family == "mobilenet" else "resnet18",
        routing_strategy=routing_strategy,
        top_k=2,
        multi_gpu=False,
    )
    spec = get_model_spec("moe_standard")
    model = spec.build(config)
    model.eval()

    batch_size = 4
    x = torch.randn(batch_size, 3, 64, 64)
    with torch.no_grad():
        out = model(x)

    assert out.shape == (batch_size, 2)
    assert model.last_routing_weights is not None
    assert model.last_routing_weights.shape == (batch_size, num_experts)
    torch.testing.assert_close(
        model.last_routing_weights.sum(dim=-1),
        torch.ones(batch_size),
        rtol=1e-4,
        atol=1e-4,
    )


def test_moe_backward_pass():
    for family, mode in [("moe_frequency", "concat_frequency"), ("moe_standard", "none")]:
        config = TrainingConfig(
            model_family=family,
            fourier_mode=mode,
            expert_family="mobilenet",
            variant="small",
            multi_gpu=False,
        )
        model = get_model_spec(family).build(config)
        model.train()

        in_ch = 6 if mode == "concat_frequency" else 3
        x = torch.randn(2, in_ch, 32, 32, requires_grad=True)
        target = torch.tensor([0, 1], dtype=torch.long)
        out = model(x)
        loss = nn.CrossEntropyLoss()(out, target)
        loss.backward()

        # Check that router parameters received gradients
        for p in model.router.parameters():
            if p.requires_grad:
                assert p.grad is not None and not torch.isnan(p.grad).any()

        # Check that experts received gradients
        for expert in model.experts:
            has_grad = any(p.grad is not None for p in expert.parameters() if p.requires_grad)
            assert has_grad


def test_moe_freeze_and_unfreeze():
    for family, mode in [("moe_frequency", "concat_frequency"), ("moe_standard", "none")]:
        config = TrainingConfig(
            model_family=family,
            fourier_mode=mode,
            expert_family="mobilenet",
            variant="small",
            unfreeze_last_n=1,
            multi_gpu=False,
        )
        spec = get_model_spec(family)
        model = spec.build(config)
        total_params = sum(p.numel() for p in model.parameters())

        # Test freeze_backbone
        spec.freeze_backbone(model)
        trainable_frozen = sum(p.numel() for p in model.parameters() if p.requires_grad)
        assert 0 < trainable_frozen < total_params

        # Router and heads should be trainable
        assert all(p.requires_grad for p in model.router.parameters())

        # Test unfreeze_for_finetune
        spec.unfreeze_for_finetune(model, config.unfreeze_last_n)
        trainable_unfrozen = sum(p.numel() for p in model.parameters() if p.requires_grad)
        assert trainable_frozen < trainable_unfrozen < total_params


def test_moe_parameter_groups():
    config = TrainingConfig(
        model_family="moe_frequency",
        fourier_mode="concat_frequency",
        lr_head=1e-3,
        lr_backbone=1e-4,
    )
    spec = get_model_spec("moe_frequency")
    model = spec.build(config)
    groups = spec.parameter_groups(model, config)
    assert len(groups) == 2
    names = {g["name"] for g in groups}
    assert names == {"head", "backbone"}


def test_moe_training_end_to_end(tiny_phase1_dataset, tmp_path, monkeypatch):
    from train import train_from_config
    models_dir = tmp_path / "models"
    monkeypatch.setenv("TCC_MODELS_ROOT", str(models_dir))

    # Test Frequency MoE
    config_freq = tmp_path / "moe_frequency.yaml"
    config_freq.write_text(
        "model_family: moe_frequency\nfourier_mode: concat_frequency\nvariant: small\n"
        "expert_family: mobilenet\nrouting_strategy: dense\nepochs: 1\nbatch_size: 2\n"
        "num_workers: 0\nimage_size: 32\nraw_min: true\nmulti_gpu: false\naugment: false\n",
        encoding="utf-8",
    )
    test_res_freq = train_from_config(config_freq, epochs=1, data_limit=4, raw_min=True, multi_gpu=False)
    assert test_res_freq["acc"] >= 0.0
    checkpoint_freq = models_dir / "moe_frequency" / "concat_frequency" / "scratch" / "seed_42" / "weights" / "best.pth"
    assert checkpoint_freq.exists()

    # Test Standard MoE
    config_std = tmp_path / "moe_standard.yaml"
    config_std.write_text(
        "model_family: moe_standard\nfourier_mode: none\nnum_experts: 2\nvariant: small\n"
        "expert_family: mobilenet\nrouting_strategy: dense\nepochs: 1\nbatch_size: 2\n"
        "num_workers: 0\nimage_size: 32\nraw_min: true\nmulti_gpu: false\naugment: false\n",
        encoding="utf-8",
    )
    test_res_std = train_from_config(config_std, epochs=1, data_limit=4, raw_min=True, multi_gpu=False)
    assert test_res_std["acc"] >= 0.0
    checkpoint_std = models_dir / "moe_standard" / "none" / "scratch" / "seed_42" / "weights" / "best.pth"
    assert checkpoint_std.exists()

