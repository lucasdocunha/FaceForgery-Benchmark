import copy

import pytest
import torch
import torch.nn.functional as F

from src.experimental.reconstruction import (
    AutoencoderConfig, BetaSchedule, CompositeReconstructionLoss, FaceAutoencoder,
    LatentDetector, MeanReconstructionEnsemble, ReconstructionAnomaly,
    ResidualDetector, ResidualFusionBlock, analytical_kl, structural_similarity,
)


@pytest.fixture(autouse=True)
def limited_threads():
    previous = torch.get_num_threads()
    torch.set_num_threads(1)
    yield
    torch.set_num_threads(previous)


def ae(kind="cae"):
    return FaceAutoencoder({"kind": kind, "image_size": 32, "width": 4, "latent_dim": 8})


@pytest.mark.parametrize("kind", ["cae", "vae", "gated"])
def test_reconstruction_shape_gradients_and_roundtrip(kind):
    model = ae(kind)
    x = torch.rand(2, 3, 32, 32)
    details = model.reconstruct(x)
    assert details["reconstruction"].shape == x.shape
    assert torch.all((details["reconstruction"] >= 0) & (details["reconstruction"] <= 1))
    details["reconstruction"].square().mean().backward()
    assert model.to_mu.weight.grad is not None
    clone = ae(kind)
    clone.load_state_dict(model.state_dict(), strict=True)
    model.eval()
    clone.eval()
    assert torch.equal(model(x), clone(x))


def test_vae_sampling_and_analytic_kl():
    model = ae("vae")
    x = torch.rand(2, 3, 32, 32)
    assert not torch.equal(model.reconstruct(x)["latent"], model.reconstruct(x)["latent"])
    model.eval()
    assert torch.equal(model.reconstruct(x)["latent"], model.reconstruct(x)["mu"])
    assert torch.equal(analytical_kl(torch.zeros(2, 3), torch.zeros(2, 3)), torch.zeros(2))
    assert torch.allclose(analytical_kl(torch.ones(2, 3), torch.zeros(2, 3)), torch.full((2,), 1.5))
    normal = torch.distributions.Normal(torch.randn(2, 4), torch.rand(2, 4) + 0.3)
    expected = torch.distributions.kl_divergence(normal, torch.distributions.Normal(0., 1.)).sum(1)
    assert torch.allclose(analytical_kl(normal.loc, 2 * normal.scale.log()), expected, atol=1e-6)


def test_gated_decoder_only_receives_deep_skips_and_can_disable_them():
    model = ae("gated")
    shapes = []
    handles = [gate.register_forward_pre_hook(lambda module, args: shapes.append(args[0].shape[-2:])) for gate in model.gates.values()]
    x = torch.rand(2, 3, 32, 32)
    model(x)
    assert shapes == [torch.Size([4, 4])]
    for handle in handles:
        handle.remove()
    plain = ae("cae")
    plain.load_state_dict({k: v for k, v in model.state_dict().items() if not k.startswith("gates.")})
    model.skip_enabled = False
    assert torch.equal(model(x), plain(x))
    with pytest.raises(ValueError, match="Skips"):
        AutoencoderConfig(kind="gated", skip_divisor=2)


@pytest.mark.parametrize("gradient,channels", [(False, 9), (True, 10)])
def test_fusion_preserves_components_and_zero_gradient(gradient, channels):
    x, y = torch.rand(2, 3, 32, 32), torch.rand(2, 3, 32, 32)
    fusion = ResidualFusionBlock(gradient)
    output = fusion(x, y)
    assert output.shape == (2, channels, 32, 32)
    assert torch.equal(output[:, :3], x)
    assert torch.equal(output[:, 3:6], y)
    assert torch.equal(output[:, 6:9], (x-y).abs())
    if gradient:
        assert torch.equal(fusion(x, x)[:, 9], torch.zeros_like(x[:, 0]))


def test_composite_loss_masking_and_optional_assets(tmp_path):
    prediction = torch.rand(3, 3, 32, 32, requires_grad=True)
    target = torch.rand_like(prediction)
    mask = torch.tensor([True, False, True])
    loss = CompositeReconstructionLoss(lambda_ssim=0.1)
    parts = loss(prediction, target, real_mask=mask)
    parts["loss"].backward()
    assert prediction.grad[1].count_nonzero() == 0
    assert prediction.grad[0].abs().sum() > 0
    assert torch.isfinite(parts["loss"])
    assert torch.allclose(structural_similarity(target, target), torch.ones(3), atol=1e-6)
    empty = loss(prediction, target, real_mask=torch.zeros(3, dtype=torch.bool))
    assert empty["loss"].item() == 0
    with pytest.raises(FileNotFoundError, match="complete pretrained"):
        CompositeReconstructionLoss(lambda_lpips=0.1, lpips_state_path=tmp_path / "missing.pth")


def test_beta_schedules_use_successful_optimizer_steps():
    linear = BetaSchedule(maximum=0.2, warmup_steps=4)
    assert [linear(s) for s in (0, 2, 4, 9)] == [0., 0.1, 0.2, 0.2]
    cyclic = BetaSchedule(kind="cyclic", maximum=1., cycle_steps=8, ramp_fraction=0.5)
    assert [cyclic(s) for s in (0, 2, 4, 7, 8, 10)] == [0., 0.5, 1., 1., 0., 0.5]
    saved_config, saved_step = copy.deepcopy(cyclic.__dict__), 11
    resumed = BetaSchedule(**saved_config)
    assert [resumed(s) for s in range(saved_step, 24)] == [cyclic(s) for s in range(saved_step, 24)]
    with pytest.raises(ValueError):
        BetaSchedule(cycle_steps=0)


@pytest.mark.parametrize("frozen", [True, False])
def test_detector_freeze_and_real_only_autoencoder_updates(frozen):
    model = ResidualDetector(ae(), freeze_ae=frozen, width=4)
    model.train()
    assert model.autoencoder.training is not frozen
    x = torch.rand(2, 3, 32, 32)
    details = model.forward_details(x)
    F.cross_entropy(details["logits"], torch.tensor([0, 1])).backward()
    assert all(p.grad is None for p in model.autoencoder.parameters())
    if not frozen:
        loss = CompositeReconstructionLoss(lambda_ssim=0)
        loss(details["reconstruction"], x, real_mask=torch.tensor([True, False]))["loss"].backward()
        assert model.autoencoder.to_mu.weight.grad.abs().sum() > 0
    groups = model.parameter_groups(1e-5, 1e-4, 1e-3)
    actual = [id(p) for group in groups for p in group["params"]]
    expected = [id(p) for p in model.parameters() if p.requires_grad]
    assert len(actual) == len(set(actual)) and set(actual) == set(expected)


def test_latent_anomaly_and_fixed_mean_scoring():
    latent = LatentDetector(ae("vae"))
    spatial = ResidualDetector(ae(), width=4)
    x = torch.rand(2, 3, 32, 32)
    latent.eval()
    spatial.eval()
    assert latent.forward_details(x)["features"].shape == (2, 17)
    combined = MeanReconstructionEnsemble(spatial, latent)(x).softmax(1)
    assert torch.allclose(combined, (spatial(x).softmax(1) + latent(x).softmax(1)) / 2)
    anomaly = ReconstructionAnomaly(ae()).eval()
    raw = anomaly.forward_details(x)["anomaly_score"]
    assert torch.allclose(anomaly(x).softmax(1)[:, 1], raw, atol=1e-6)
    with pytest.raises(ValueError, match="requires a VAE"):
        LatentDetector(ae())


@pytest.mark.parametrize("kind", ["residual", "latent"])
@pytest.mark.parametrize("mode", ["frozen", "recon_finetune", "end_to_end"])
def test_explicit_autoencoder_gradient_modes(kind, mode):
    autoencoder = ae("vae" if kind == "latent" else "cae")
    detector = ResidualDetector(autoencoder, ae_mode=mode, width=4) if kind == "residual" else LatentDetector(autoencoder, ae_mode=mode)
    details = detector.forward_details(torch.rand(2, 3, 32, 32))
    F.cross_entropy(details["logits"], torch.tensor([0, 1])).backward(retain_graph=True)
    ae_grad = sum(float(p.grad.abs().sum()) for p in autoencoder.parameters() if p.grad is not None)
    assert (ae_grad > 0) == (mode == "end_to_end")
    if mode != "frozen":
        detector.zero_grad(set_to_none=True)
        details["reconstruction"].retain_grad()
        CompositeReconstructionLoss(lambda_ssim=0)(details["reconstruction"], torch.zeros(2, 3, 32, 32), real_mask=torch.tensor([True, False]))["loss"].backward()
        assert details["reconstruction"].grad[0].abs().sum() > 0
        assert details["reconstruction"].grad[1].count_nonzero() == 0
        assert any(p.grad is not None and p.grad.abs().sum() > 0 for p in autoencoder.parameters())
    groups = {g["name"]: g["lr"] for g in detector.parameter_groups(1e-5, 1e-4, 1e-3)}
    assert groups["head"] == 1e-3
    assert ("autoencoder" in groups) == (mode != "frozen")
    if mode != "frozen":
        assert groups["autoencoder"] == 1e-5


def test_input_ablations_preserve_backbone_initialization_and_isolate_signals():
    models, tensors = {}, {}
    x = torch.rand(2, 3, 32, 32)
    for mode in ("x_only", "residual_only", "full"):
        torch.manual_seed(12)
        model = ResidualDetector(ae(), input_mode=mode, gradient=True, width=4).eval()
        models[mode] = model
        handle = model.backbone.register_forward_pre_hook(lambda module, args, key=mode: tensors.update({key: args[0].detach()}))
        model(x)
        handle.remove()
    for mode in ("x_only", "residual_only"):
        assert tensors[mode].shape == tensors["full"].shape
        assert tensors[mode][:, 3:9].count_nonzero() == 0
        for key, value in models[mode].state_dict().items():
            assert torch.equal(value, models["full"].state_dict()[key])
    assert tensors["x_only"][:, 9:].count_nonzero() == 0
    assert torch.equal(tensors["x_only"][:, :3], tensors["full"][:, :3])
    assert torch.equal(tensors["residual_only"][:, :3], tensors["full"][:, 6:9])
    assert torch.equal(tensors["residual_only"][:, 9:], tensors["full"][:, 9:])


def test_kl_reduction_logs_dimension_independent_nats():
    x = torch.rand(2, 3, 32, 32)
    mu, logvar = torch.ones(2, 8), torch.zeros(2, 8)
    common = dict(lambda_ssim=0, beta={"kind": "constant", "maximum": 0.1})
    summed = CompositeReconstructionLoss(**common, kl_reduction="sum")(x, x, mu=mu, logvar=logvar)
    mean = CompositeReconstructionLoss(**common, kl_reduction="mean_per_dim")(x, x, mu=mu, logvar=logvar)
    assert summed["kl_nats_per_sample"] == mean["kl_nats_per_sample"] == 4
    assert summed["kl_nats_per_dimension"] == mean["kl_nats_per_dimension"] == 0.5
    assert summed["kl"] == 4 and mean["kl"] == 0.5
    torch.testing.assert_close(summed["loss"], 8 * mean["loss"])


def test_resnet_zero_stem_learns_path_before_classification_reaches_autoencoder():
    model = ResidualDetector(ae(), ae_mode="end_to_end", backbone="resnet18", initialize=False)
    x, y = torch.rand(2, 3, 32, 32), torch.tensor([0, 1])
    F.cross_entropy(model(x), y).backward()
    assert model.autoencoder.to_mu.weight.grad.count_nonzero() == 0
    assert model.backbone.conv1.weight.grad[:, 3:].abs().sum() > 0
    torch.optim.SGD(model.parameters(), lr=0.001).step()
    model.zero_grad(set_to_none=True)
    F.cross_entropy(model(x), y).backward()
    assert model.autoencoder.to_mu.weight.grad.abs().sum() > 0
