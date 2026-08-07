def test_phase1_split_root_supports_min_dataset_short_split_names(
    tiny_short_split_dataset,
):
    from src.data.paths import phase1_split_root

    assert phase1_split_root("train") == tiny_short_split_dataset / "train"
    assert phase1_split_root("val") == tiny_short_split_dataset / "val"
    assert phase1_split_root("test") == tiny_short_split_dataset / "test"


def test_models_and_output_roots_follow_environment(tmp_path, monkeypatch):
    from src.data.paths import models_root, output_root
    monkeypatch.setenv("TCC_MODELS_ROOT", str(tmp_path / "checkpoints"))
    monkeypatch.setenv("TCC_OUTPUT_ROOT", str(tmp_path / "artifacts"))
    assert models_root() == tmp_path / "checkpoints"
    assert output_root() == tmp_path / "artifacts"
