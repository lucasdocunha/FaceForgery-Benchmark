from __future__ import annotations

import tempfile
from pathlib import Path
import pytest
from src.models._pretrained import (
    is_valid_safetensors,
    is_valid_bin,
    get_candidate_pretrained_dirs,
    load_hf_backbone,
)


def test_is_valid_safetensors_on_bad_inputs():
    # 1. Arquivo não existente
    assert not is_valid_safetensors(Path("/caminho/inexistente/model.safetensors"))

    # 2. Arquivo vazio (0 bytes)
    with tempfile.NamedTemporaryFile() as tmp:
        assert not is_valid_safetensors(Path(tmp.name))

    # 3. Ponteiro Git-LFS falso ou arquivo corrompido
    with tempfile.NamedTemporaryFile(mode="w", delete=False) as tmp:
        tmp.write("version https://git-lfs.github.com/spec/v1\noid sha256:abcd\nsize 123456\n")
        tmp_path = Path(tmp.name)
    try:
        assert not is_valid_safetensors(tmp_path)
    finally:
        tmp_path.unlink(missing_ok=True)


def test_is_valid_bin():
    assert not is_valid_bin(Path("/caminho/inexistente.pth"))

    with tempfile.NamedTemporaryFile(delete=False) as tmp:
        tmp.write(b"0" * 1024)
        tmp_path = Path(tmp.name)
    try:
        # Menor que 10 MB deve falhar
        assert not is_valid_bin(tmp_path, min_size_mb=10.0)
        # Menor que 1 KB deve passar com min_size_mb=0.0001
        assert is_valid_bin(tmp_path, min_size_mb=0.0001)
    finally:
        tmp_path.unlink(missing_ok=True)


def test_get_candidate_pretrained_dirs():
    dirs = get_candidate_pretrained_dirs("vit")
    assert len(dirs) > 0
    assert all(isinstance(d, Path) for d in dirs)
    assert any("vit" in str(d) for d in dirs)


def test_load_hf_backbone_skips_corrupted_local_safetensors(monkeypatch, tmp_path):
    # Cria uma pasta local fingindo ser o diretório de pesos
    fake_local = tmp_path / "fake_vit"
    fake_local.mkdir()
    corrupt_sf = fake_local / "model.safetensors"
    # Escreve um arquivo falso de 50 bytes (corrompido/truncado)
    corrupt_sf.write_text("corrupted content not a safetensor")

    # Mocka get_candidate_pretrained_dirs para retornar a pasta com o arquivo corrompido
    monkeypatch.setattr(
        "src.models._pretrained.get_candidate_pretrained_dirs",
        lambda family: [fake_local],
    )

    # Cria uma classe Mock para simular o modelo do Hugging Face
    class MockModel:
        @classmethod
        def from_pretrained(cls, path_or_repo, use_safetensors=True, **kwargs):
            if path_or_repo == str(fake_local):
                raise RuntimeError("Should never try to load corrupted local safetensors!")
            # Retorna com sucesso quando cai no fallback
            return f"Loaded from {path_or_repo} (use_safetensors={use_safetensors})"

    res = load_hf_backbone(MockModel, "vit", "google/vit-base-patch16-224")
    assert "google/vit-base-patch16-224" in res
