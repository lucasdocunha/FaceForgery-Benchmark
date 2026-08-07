# Evaluation guide

Checkpoints must use `models/<family>/<mode>/<regime>/seed_<N>/weights/best.pth`.

```bash
python evaluate.py --splits val,test
python evaluate.py --splits val,test,test_d --test-d-csv <csv> --test-d-images-dir <dir>
```

Each seed directory receives `metrics_<split>.csv`, `outputs_<split>.npz`, and
`predictions_<split>.csv`. A combined `all_metrics_by_split.csv` is written at the models root.

Ensemble candidates aggregate predictions across seeds before selection. `--pool best-mode` chooses
one validation-best mode for every family×regime and performs parallel exhaustive subset search;
`--pool all` uses all family×mode×regime candidates and greedy search. Both select on validation and
apply the same fitted combination to Test and Test-Hard.

```bash
python ensemble.py --strategy search --pool best-mode
python ensemble.py --strategy weighted --pool all
python make_tables.py
```

Available direct strategies are `mean`, `weighted`, `majority`, `max`, `geometric`, and `stacking`.
The table command aggregates seeds and exports full CSV/Markdown plus one booktabs LaTeX table per
split.
