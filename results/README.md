# results/

Generated outputs meant to be shown rather than run: evidence that the
system works and how well.

## `evaluation_report.json`

Written by `python -m src.evaluation.run` (Module F). **Committed to git**, so
the measured performance is visible without rerunning the pipeline.

Contains:

- `dataset` — size of the graph and test split the numbers were computed on
- `results.validation` / `results.test` — AUC, AUPR, accuracy, precision,
  recall, F1, and the decision threshold used
- `baseline_comparison` — our test figures against the baseline paper's
  published accuracy, precision, and recall
- `comparison_caveat` — why this is a reference point rather than a
  like-for-like reproduction

Headline test-set result: **AUC 0.9501, accuracy 0.8741, recall 0.9189.**
See `docs/model_selection.md` for how the architecture was chosen and why
accuracy sits below the paper's reported figure.

The decision threshold is tuned on the validation split and then applied
unchanged to test — tuning it on test would inflate every reported number.
