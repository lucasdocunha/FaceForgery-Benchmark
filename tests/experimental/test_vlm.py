"""Hand-checked sequence scoring and real offline multimodal adapter training."""

from pathlib import Path
from types import SimpleNamespace

import pandas as pd
import pytest
import torch
from PIL import Image

from src.experimental.vlm.batches import answer_batch
from src.experimental.vlm.models import MAX_PARAMETERS, UnsupportedQuantizationError, adapt_model, load_base, load_processor, stored_parameters
from src.experimental.vlm.scoring import fake_probability, sequence_logprob, sft_loss_step


def test_sequence_scores_sum_all_tokens_at_preceding_positions():
    probabilities = torch.tensor([
        [[.1, .2, .7], [.6, .3, .1], [.2, .5, .3], [.3, .3, .4]],
        [[.5, .2, .3], [.1, .8, .1], [.2, .2, .6], [.3, .3, .4]],
    ])
    ids = torch.tensor([[0, 2, 0, 1], [0, 1, 2, 0]])
    mask = torch.tensor([[False, True, True, False], [False, False, True, False]])
    scores = sequence_logprob(probabilities.log(), ids, mask)
    assert torch.allclose(scores, torch.tensor([.7 * .6, .1]).log())
    assert torch.allclose(scores[:1], sequence_logprob(probabilities[:1].log(), ids[:1], mask[:1]))
    assert torch.allclose(fake_probability(torch.log(torch.tensor([.4])), torch.log(torch.tensor([.1]))), torch.tensor([.2]))
    with pytest.raises(ValueError, match="Padding"):
        sequence_logprob(probabilities.log(), ids, mask, torch.tensor([[1, 1, 0, 0], [1, 1, 1, 0]]))
    with pytest.raises(ValueError, match="noninitial"):
        sequence_logprob(probabilities.log(), ids, torch.ones_like(ids, dtype=torch.bool))


class SpanProcessor:
    """Deliberately unequal verbalizers with EOS also used as padding."""
    tokenizer = SimpleNamespace(padding_side="right", eos_token_id=2)

    def apply_chat_template(self, messages, *, tokenize, add_generation_prompt):
        if add_generation_prompt:
            return "prefix"
        return messages[-1]["content"][0]["text"]

    def __call__(self, *, text, images, padding, truncation, return_tensors):
        suffix = {"prefix": [], "Real": [7, 9, 2, 10], "Fake": [8, 2, 10]}
        rows = [[1, 3, 5] + suffix[item] for item in text]
        width = max(map(len, rows))
        ids = torch.tensor([row + [2] * (width - len(row)) for row in rows])
        mask = torch.tensor([[1] * len(row) + [0] * (width - len(row)) for row in rows])
        return {"input_ids": ids, "attention_mask": mask}


def test_answer_masks_exclude_headers_padding_and_eos_from_score():
    inputs, score_mask, ids = answer_batch(SpanProcessor(), [object(), object()], ["Real", "Fake"])
    assert ids == [[7, 9], [8]]
    assert inputs["labels"].tolist() == [[-100, -100, -100, 7, 9, 2, -100], [-100, -100, -100, 8, 2, -100, -100]]
    assert score_mask.sum(-1).tolist() == [2, 1]
    with pytest.raises(ValueError, match="truncation"):
        answer_batch(SpanProcessor(), [object()], ["Real"], max_length=4)
    processor = SpanProcessor()
    processor.tokenizer = SimpleNamespace(padding_side="left", eos_token_id=2)
    with pytest.raises(ValueError, match="right padding"):
        answer_batch(processor, [object()], ["Fake"])


@pytest.fixture
def tiny_vlm(tmp_path):
    transformers = pytest.importorskip("transformers")
    pytest.importorskip("peft")
    from tokenizers import Tokenizer
    from tokenizers.models import WordLevel
    from tokenizers.pre_tokenizers import WhitespaceSplit
    from transformers import Idefics3Config, Idefics3ForConditionalGeneration, Idefics3ImageProcessor, Idefics3Processor, LlamaConfig, PreTrainedTokenizerFast

    torch.set_num_threads(2)
    torch.manual_seed(42)
    vocab = {"[UNK]": 0, "<|im_start|>": 1, "<end_of_utterance>": 2, "[PAD]": 3,
             "<image>": 4, "<fake_token_around_image>": 5, "<global-img>": 6, "Real": 7,
             "Fake": 8, "User:": 9, "Assistant:": 10}
    tokenizer_object = Tokenizer(WordLevel(vocab, unk_token="[UNK]"))
    tokenizer_object.pre_tokenizer = WhitespaceSplit()
    tokenizer = PreTrainedTokenizerFast(tokenizer_object=tokenizer_object, unk_token="[UNK]", bos_token="<|im_start|>",
                                       eos_token="<end_of_utterance>", pad_token="[PAD]")
    template = "<|im_start|> {% for message in messages %}{{message['role'] | capitalize}}: {% for item in message['content'] %}{% if item['type']=='image' %}<image> {% else %}{{item['text']}} {% endif %}{% endfor %}<end_of_utterance> {% endfor %}{% if add_generation_prompt %}Assistant: {% endif %}"
    processor = Idefics3Processor(Idefics3ImageProcessor(do_image_splitting=False, size={"longest_edge": 32}, max_image_size={"longest_edge": 32}),
                                 tokenizer, image_seq_len=16, chat_template=template)
    text = LlamaConfig(vocab_size=64, hidden_size=32, intermediate_size=64, num_hidden_layers=1,
                       num_attention_heads=4, num_key_value_heads=2, max_position_embeddings=256, pad_token_id=3, bos_token_id=1, eos_token_id=2)
    architecture = Idefics3Config(text_config=text, vision_config={"hidden_size": 32, "intermediate_size": 64,
        "num_hidden_layers": 1, "num_attention_heads": 4, "image_size": 32, "patch_size": 4},
        image_token_id=4, scale_factor=2, pad_token_id=3)
    base = Idefics3ForConditionalGeneration(architecture)
    base_dir = tmp_path / "pretrained" / "tiny-vlm"
    base.save_pretrained(base_dir, safe_serialization=True)
    processor.save_pretrained(base_dir)
    config = {"model_id": "offline/tiny-vlm", "model_type": "idefics3", "pretrained_path": str(base_dir),
              "dtype": "float32", "attention": "eager", "quantization": "none", "max_length": 256,
              "gradient_checkpointing": True, "processor": {"max_image_edge": 32},
              "lora": {"rank": 2, "alpha": 4, "dropout": 0.0, "projections": ["q_proj", "v_proj"]}}
    return base, processor, config


def test_real_architecture_adapter_gradients_masked_loss_and_bundle_reload(tiny_vlm, tmp_path, monkeypatch):
    from src.experimental.vlm.artifacts import rebuild_bundle, save_bundle
    from src.experimental.vlm.inference import VLMPredictor

    base, processor, config = tiny_vlm
    model, targets = adapt_model(base, config)
    assert all(name.startswith("model.text_model.") for name in targets)
    images = [Image.new("RGB", (32, 32), color) for color in ((40, 90, 160), (190, 80, 20))]
    batch, _, _ = answer_batch(processor, images, ["Real", "Fake"], max_length=256)
    model.train()
    loss, _, weight = sft_loss_step(model, batch)
    assert torch.isfinite(loss) and weight == 2
    loss.backward()
    assert any(parameter.grad is not None and parameter.grad.abs().sum() > 0
               for name, parameter in model.named_parameters() if "lora_" in name)
    assert all(parameter.grad is None for name, parameter in model.named_parameters() if "lora_" not in name)
    torch.optim.AdamW([parameter for parameter in model.parameters() if parameter.requires_grad], lr=.01).step()
    model.eval()
    artifact = save_bundle(tmp_path / "run", model, processor, config)
    monkeypatch.setenv("TCC_PRETRAINED_ROOT", str(Path(config["pretrained_path"]).parent))
    rebuilt, rebuilt_processor, _ = rebuild_bundle(artifact.parent)
    image_root = tmp_path / "images"
    image_root.mkdir()
    for i, image in enumerate(images):
        image.save(image_root / f"{i}.png")
    frame = pd.DataFrame({"img_name": ["0.png", "1.png"], "label": [0, 1], "sample_id": ["a", "b"],
                          "group_id": ["a", "b"], "dataset": ["synthetic"] * 2, "split": ["val"] * 2})
    first = VLMPredictor(model, processor, config)(frame, image_root, batch_size=2)
    second = VLMPredictor(rebuilt, rebuilt_processor, config)(frame, image_root, batch_size=1)
    assert first.sample_id.tolist() == ["a", "b"]
    assert first.image_sha256.str.len().eq(64).all()
    assert torch.allclose(torch.tensor(first.p_fake), torch.tensor(second.p_fake), atol=1e-6, rtol=1e-5)
    assert first.p_fake.between(0, 1).all()
    (artifact.parent / "adapter" / "adapter_config.json").write_text("{}")
    with pytest.raises(ValueError, match="immutable identity"):
        rebuild_bundle(artifact.parent)


def test_nf4_cpu_selection_fails_explicitly_and_base_count_enforces_cap(tiny_vlm, tmp_path, monkeypatch):
    _, _, config = tiny_vlm
    with pytest.raises(UnsupportedQuantizationError, match="verified CUDA-device"):
        load_base({**config, "quantization": "nf4"}, "cpu")
    assert stored_parameters(config["pretrained_path"]) < MAX_PARAMETERS
    import safetensors
    class OversizedHeader:
        def __enter__(self): return self
        def __exit__(self, *args): pass
        def keys(self): return ["too_large"]
        def get_slice(self, key): return SimpleNamespace(get_shape=lambda: [MAX_PARAMETERS + 1])
    monkeypatch.setattr(safetensors, "safe_open", lambda *args, **kwargs: OversizedHeader())
    with pytest.raises(ValueError, match="exceeding"):
        stored_parameters(config["pretrained_path"])


def test_fit_reuses_runtime_and_exports_bound_calibration(tiny_vlm, tmp_path, monkeypatch):
    import json
    from src.experimental.vlm import fit, input_contract, load_predictor
    from src.robustness.manifests import load_manifest, save_manifest
    from src.robustness.provenance import digest_file

    _, _, model_config = tiny_vlm
    monkeypatch.setenv("TCC_PRETRAINED_ROOT", str(Path(model_config["pretrained_path"]).parent))
    data = {}
    for split in ("train", "val"):
        root = tmp_path / split
        root.mkdir()
        for index in range(2):
            Image.new("RGB", (32, 32), (40 + index * 90, 70, 160)).save(root / f"{index}.png")
        frame = pd.DataFrame({"img_name": ["0.png", "1.png"], "label": [0, 1],
            "sample_id": [f"{split}-a", f"{split}-b"], "group_id": [f"{split}-a", f"{split}-b"],
            "dataset": ["synthetic"] * 2, "split": [split] * 2})
        manifest = tmp_path / f"{split}.csv"
        save_manifest(frame, manifest, {})
        data[f"{split}_manifest"], data[f"{split}_root"] = str(manifest), str(root)
    run = fit({"task": "vlm", "scope": "synthetic-structural", "seed": 42, "output_dir": str(tmp_path / "fit"),
               "data": data, "model": model_config, "training": {"device": "cpu", "epochs": 1,
                   "batch_size": 1, "grad_accum_steps": 2, "lr": .01, "amp": False}})
    assert json.loads((run / "status.json").read_text())["state"] == "complete"
    calibration = json.loads((run / "calibration.json").read_text())
    document = json.loads((run / "bundle.json").read_text())
    assert calibration["model_sha256"] == digest_file(run / "bundle.json")
    assert calibration["selection_split"] == "val"
    assert calibration["input_contract"] == input_contract(document)
    frame, _ = load_manifest(data["val_manifest"])
    predicted = load_predictor(run)(frame, data["val_root"])
    assert len(predicted) == 2
    assert predicted.p_fake.notna().all()
