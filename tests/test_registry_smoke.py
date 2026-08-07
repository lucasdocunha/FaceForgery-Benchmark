import pytest,torch
from src.models.registry import MODEL_REGISTRY
from src.pipelines.config import TrainingConfig
@pytest.mark.parametrize("family",("resnet","mobilenet","xception","vit","clip"))
@pytest.mark.parametrize("channels,mode",((1,"phase"),(2,"complex"),(4,"concat"),(6,"concat_frequency")))
def test_scratch_registry_supports_every_channel_count(family,channels,mode):
    config=TrainingConfig(model_family=family,fourier_mode=mode,image_size=32,patch_size=8,hidden_size=32,num_hidden_layers=1,num_attention_heads=4,variant="small")
    model=MODEL_REGISTRY[family].build(config).eval()
    with torch.no_grad(): assert model(torch.randn(1,channels,32,32)).shape==(1,2)
def test_pretrained_is_rejected_when_disabled():
    config=TrainingConfig(model_family="resnet",regime="finetune",allow_pretrained=False)
    with pytest.raises(ValueError): MODEL_REGISTRY["resnet"].build(config)
