# Evaluation guide

Checkpoints use `models/<family>/<mode>/<regime>/seed_<N>/`. Evaluate validation and test with
`python evaluate.py`. Add Test-Hard with `--splits val,test,test_d --test-d-csv <csv>
--test-d-images-dir <dir>`.

Each seed receives `metrics_<split>.csv` and `outputs_<split>.npz`. Select ensemble members on
validation with `python ensemble.py --strategy search --pool best-mode`; `--pool all` uses greedy
search. Direct strategies are mean, weighted, majority, max, geometric, and stacking. Generate
seed-aggregated CSV, Markdown, and LaTeX with `python make_tables.py`.
