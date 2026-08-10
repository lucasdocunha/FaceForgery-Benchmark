import inspect

from src.pipelines.config import load_config


def test_yaml_can_disable_multi_gpu_through_train_from_config(tmp_path):
    """`multi_gpu: false` no YAML tem que sobreviver ao caminho do train.py.

    load_config trata todo override não-None como explícito, então um default
    True em train_from_config sobrescrevia o YAML silenciosamente.
    """
    from train import train_from_config

    assert inspect.signature(train_from_config).parameters["multi_gpu"].default is None
    (tmp_path / "base.yaml").write_text("multi_gpu: false\n")
    (tmp_path / "resnet.yaml").write_text("model_family: resnet\n")
    overrides = {"fourier_mode": None, "regime": None, "seed": None, "epochs": None,
                 "data_limit": None, "raw_min": None, "multi_gpu": None}
    assert load_config(tmp_path / "resnet.yaml", overrides).multi_gpu is False


def test_base_family_and_cli_merge(tmp_path):
    (tmp_path/"base.yaml").write_text("epochs: 5\nseeds: [1, 2]\n")
    (tmp_path/"resnet.yaml").write_text("model_family: resnet\nbatch_size: 8\n")
    config=load_config(tmp_path/"resnet.yaml",{"epochs":1,"fourier_mode":"complex"})
    assert (config.epochs,config.batch_size,config.seeds,config.in_channels)==(1,8,(1,2),2)
    assert config.data_split_dir == "raw"
