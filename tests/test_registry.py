import pytest

from src.models.registry import _default_parameter_groups, get_model_spec
from src.pipelines.config import TrainingConfig

# Famílias cujo backbone contém submódulos chamados `fc1`/`fc2` (MLP de transformer,
# squeeze-excitation do MobileNetV3). São exatamente os casos que um teste de
# substring em "fc" classificaria como cabeça.
SMALL_CONFIGS = {
    "clip": dict(patch_size=8, image_size=32, hidden_size=32,
                 num_hidden_layers=2, num_attention_heads=4),
    "dino": dict(model_size="tiny"),
    "mobilenet": dict(variant="small"),
}


def _groups_by_name(family, **overrides):
    config = TrainingConfig(model_family=family, lr_head=1e-3, lr_backbone=1e-4,
                            **SMALL_CONFIGS[family], **overrides)
    model = get_model_spec(family).build(config)
    groups = {group["name"]: group for group in _default_parameter_groups(model, config)}
    return model, config, groups


@pytest.mark.parametrize("family", sorted(SMALL_CONFIGS))
def test_backbone_mlp_and_se_params_keep_the_backbone_lr(family):
    """Cada grupo deve receber o LR pretendido, e o backbone não pode vazar para o head.

    Vazar significa treinar pesos pré-treinados a 10x o LR planejado (lr_head vs
    lr_backbone), o que degrada silenciosamente justamente os números de fine-tuning.
    """
    model, config, groups = _groups_by_name(family)
    head_ids = {id(param) for param in groups["head"]["params"]}

    leaked = [
        name for name, param in model.named_parameters()
        if id(param) in head_ids and not name.startswith(("classifier.", "fc.", "head."))
    ]
    assert leaked == [], f"{family}: parâmetros de backbone no grupo head: {leaked[:5]}"

    assert groups["backbone"]["lr"] == config.lr_backbone
    assert groups["head"]["lr"] == config.lr_head


@pytest.mark.parametrize("family", sorted(SMALL_CONFIGS))
def test_real_classifier_head_still_reaches_the_head_lr(family):
    """Contraprova do teste acima: ancorar não pode ter esvaziado o grupo head."""
    model, _config, groups = _groups_by_name(family)
    head_ids = {id(param) for param in groups["head"]["params"]}

    expected = {
        name for name, param in model.named_parameters()
        if param.requires_grad and name.startswith(("classifier.", "fc.", "head."))
    }
    assert expected, f"{family}: modelo não expõe cabeça reconhecível"
    assert all(id(dict(model.named_parameters())[name]) in head_ids for name in expected)


# Xception scratch fica de fora: só a variante pretrained (timm) expõe
# unfreeze_last_n_layers, então no scratch o descongelamento parcial é no-op.
PARTIAL_THAW_FAMILIES = ("resnet", "mobilenet", "vit", "clip", "dino")


@pytest.mark.parametrize("family", PARTIAL_THAW_FAMILIES)
def test_unfreeze_for_finetune_actually_thaws_encoder_blocks(family):
    """Regressão: o caminho dos blocos do encoder tem que resolver de verdade.

    `backbone.encoder.layer` (ViT) e `backbone.vision_model.encoder.layers` (CLIP)
    deixaram de existir no transformers 5.x, então unfreeze_for_finetune levantava
    AttributeError e derrubava as 42 execuções de finetune de ViT/CLIP da matriz.
    Nada exercitava essa função, então a suíte passava. Aqui exigimos
    estritamente mais parâmetros treináveis que a cabeça sozinha.
    """
    config = TrainingConfig(
        model_family=family, fourier_mode="none", image_size=32, patch_size=8,
        hidden_size=32, num_hidden_layers=3, num_attention_heads=4,
        variant="small", model_size="tiny",
    )
    spec = get_model_spec(family)
    model = spec.build(config)
    total = sum(p.numel() for p in model.parameters())

    spec.freeze_backbone(model)
    head_only = sum(p.numel() for p in model.parameters() if p.requires_grad)
    spec.unfreeze_for_finetune(model, 2)
    partial = sum(p.numel() for p in model.parameters() if p.requires_grad)

    assert partial > head_only, f"{family}: nenhum bloco do backbone foi descongelado"
    assert partial < total, f"{family}: descongelou o backbone inteiro"


def test_frozen_backbone_yields_a_head_only_group():
    """Com o backbone congelado sobra só o grupo head, sem grupo de backbone vazio."""
    family = "mobilenet"
    config = TrainingConfig(model_family=family, **SMALL_CONFIGS[family])
    spec = get_model_spec(family)
    model = spec.build(config)
    spec.freeze_backbone(model)

    groups = {group["name"]: group for group in _default_parameter_groups(model, config)}
    assert set(groups) == {"head"}
