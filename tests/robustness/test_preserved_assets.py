import json
from pathlib import Path
from src.robustness.provenance import digest_file

ROOT = Path(__file__).resolve().parents[2]


def test_primary_six_page_paper_and_references_are_preserved():
    record = json.loads((ROOT / "paper/six-page/build.json").read_text())
    assert record["pages"] == 6
    assert digest_file(ROOT / "paper.pdf") == record["pdf_sha256"]
    assert digest_file(ROOT / "paper/six-page/main.tex") == record["source_sha256"]
    for name, checksum in record["preserved_reference_sha256"].items():
        assert digest_file(ROOT / name) == checksum


def test_presentation_and_notebook_match_preserved_build():
    record = json.loads((ROOT / "presentation/build.json").read_text())
    assert record["idioma"] == "pt-BR" and record["slides"] == 15
    for name, checksum in record["sha256"].items():
        assert digest_file(ROOT / "presentation" / name) == checksum
