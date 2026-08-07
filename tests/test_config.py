from src.pipelines.config import load_config
def test_base_family_and_cli_merge(tmp_path):
    (tmp_path/"base.yaml").write_text("epochs: 5\nseeds: [1, 2]\n")
    (tmp_path/"resnet.yaml").write_text("model_family: resnet\nbatch_size: 8\n")
    config=load_config(tmp_path/"resnet.yaml",{"epochs":1,"fourier_mode":"complex"})
    assert (config.epochs,config.batch_size,config.seeds,config.in_channels)==(1,8,(1,2),2)
