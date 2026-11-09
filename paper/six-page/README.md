# Primary manuscript: six-page IEEE edition

The repository's authoritative manuscript is [`../../paper.pdf`](../../paper.pdf), identical to [`main.pdf`](main.pdf). It contains **exactly six pages including references** and uses the original IEEE/SIBGRAPI conference layout: two columns, 10-point body text, US Letter paper, numeric citations, and the original author/affiliation style. No reduced margins or scaled body text are used to force the page count.

Edit [`main.tex`](main.tex) for this edition. The shared BibTeX database remains [`../explicability/references.bib`](../explicability/references.bib).

The ten-page version at [`../explicability/main.pdf`](../explicability/main.pdf) and its source remain unchanged as an expanded reference. Its operational appendices are intentionally excluded from the six-page edition; all three XAI protocols, their controls, the evidence boundaries, and the cited literature remain in the primary manuscript. The original baseline PDF remains [`../original-sibgrapi.pdf`](../original-sibgrapi.pdf).

## Build

From this directory, with IEEEtran and a standard TeX installation:

```bash
pdflatex -interaction=nonstopmode -halt-on-error main.tex
bibtex main
pdflatex -interaction=nonstopmode -halt-on-error main.tex
pdflatex -interaction=nonstopmode -halt-on-error main.tex
pdfinfo main.pdf
```

On Ubuntu, the required packages are `texlive-publishers`, `texlive-latex-extra`, `texlive-fonts-recommended`, and `poppler-utils`.

The read-only **Six-page IEEE manuscript** workflow performs the build and uploads page renders, extracted text, and logs. The separate **Package primary six-page paper** workflow verifies exactly six pages, standard IEEE class, original page dimensions, citation/layout diagnostics, source hashes, and preservation of both reference PDFs before updating the root manuscript. It writes only to `explicability` and never merges the PR. Its build-run and source-commit inputs must identify the same successful six-page build.

The old long-form packaging workflow is manual and updates only the reference directory. It cannot overwrite `paper.pdf`.

## Verification and evidence

[`build.json`](build.json) records the compiled edition's SHA-256 hashes, page dimensions, text bounds, fonts, page count, and preserved-reference hashes. The successful initial six-page build is [run 34695303971](https://github.com/Lucas-PG/FaceForgery-Benchmark/actions/runs/34695303971); its validated packaging is [run 34695579409](https://github.com/Lucas-PG/FaceForgery-Benchmark/actions/runs/34695579409).

This is a formatting and editorial revision, not a new experiment. Previously reported results retain their attribution, and unexecuted image-dependent findings remain hypotheses. Build checks do not imply experimental validation.

`scripts/prepare_six_page_manuscript.py` was the one-time conversion utility. It refuses to overwrite the committed source; subsequent edits belong directly in `main.tex`.
