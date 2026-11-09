# DF40 local-subset audit and migration

The imported `scripts/prepare_df40_benchmark.py` is **not an implementation of every official DF40 evaluation protocol**. It samples a fixed list of locally available folders, silently skips missing folders, pools selected real-image sources, uses unsorted filesystem enumeration before sampling, and stores absolute paths. It also assigns every selected image under `cdf/frames` the fake label without inspecting an authoritative per-sample annotation. These are confirmed code properties, not proof that the team's actual local files were mislabeled.

Therefore, the imported DF40 tables should be labeled **results on the exact local subset and preprocessing used**, not automatically full official DF40 results or directly comparable to another paper's protocol. Retain the existing subset manifest, counts, source versions and exact predictions. Verify class/source identities for the `cdf/frames` data independently before interpreting that subgroup. Different dataset names or generation-method names do not prove independent source domains.

## Bring the existing reviewed subset into the audited pipeline

After reviewing the actual labels and selected population:

```bash
python -m src.robustness.df40 \
  --source data/df40/test.csv \
  --images-root /current/path/to/df40_extracted \
  --path-prefix /media/ssd2/lucas.ocunha/datasets/df40_extracted \
  --output data/canonical/df40-local-subset/test.csv \
  --convention fake-is-1 --acknowledge-reviewed-subset
```

`--path-prefix` is the old absolute prefix stored in the source CSV; `--images-root` is the current mounted root. Relative paths are also accepted. Escaping paths, missing or unreadable images, duplicate identities and invalid labels fail certification. The adapter preserves CSV order, verifies the images, hashes their bytes and explicitly records that official coverage has **not** been established. It does not copy, relabel by model performance, or regenerate the image content.

When trusted annotations exist, add `--group-column video_or_identity_id` and `--source-domain-column source_domain`. The converter does not infer these from generation-method labels. Without source-domain metadata, source-matched per-generator AUC is unavailable; without video/identity groups, frame dependence remains unresolved. The default image-level grouping is disclosed in the certificate.

Use the resulting canonical manifest with `research_cli.py evaluate-legacy` or `research_cli.py evaluate`, following [the main guide](robustness.md). Existing raw predictions can be imported only when their schema and **original manifest order** are known. A custom DF40 export that lacks stable IDs cannot be safely guessed into the legacy importer merely by sorting filenames.

## Official-protocol confirmation is a separate experiment

Choose the exact official DF40 split/source direction appropriate for the comparator, build a manifest from its authoritative labels and identities, verify source overlap, and declare frame/video aggregation. Use the general canonical-manifest path for that separately named protocol. Do not reuse the local-subset name, mix its source pools, or tune the held-out protocol by selecting its best target-test row. The official project and paper are listed in `research/robustness/RESEARCH.md`.
