# The paper, in the Journal of Open Aviation Science template

`main.tex` is the paper's source, with `reference.bib` and the figures in `figures/`.

Build (TeX Live 2026, pdflatex and biber):

```
cd analysis/joas
latexmk -pdf main.tex
```

The class is the journal's and is not part of this repository: copy the `extra/` folder of
https://github.com/open-aviation/joas-template into this directory first.

Every number in the paper is traced to the measurement that produced it. After an edit:

```
python3 scripts/paper_claims.py check analysis/joas/main.tex
```

fails if the edit introduced a number with no recorded source (`analysis/paper-v2/added.tsv`) or dropped one without a
recorded reason (`analysis/paper-v2/dropped.tsv`).
