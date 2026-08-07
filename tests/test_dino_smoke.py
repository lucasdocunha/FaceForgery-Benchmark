from tests.model_smoke_helpers import assert_family_smoke
def test_dino_registry_modes_and_pretrained_policy(tiny_phase1_dataset):
    assert_family_smoke("dino", tiny_phase1_dataset)
