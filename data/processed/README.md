# data/processed/

Cleaned, structured tables produced by `src/preprocessing` (Module B) from
the files in `data/raw/`: `drugs`, `ddi_pairs`, `adverse_events`. This is the
schema-fixed dataset every downstream module reads from.

Not committed to git — regenerate by running the preprocessing pipeline
against `data/raw/`.
