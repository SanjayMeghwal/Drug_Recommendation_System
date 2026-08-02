# Module B — Preprocessing

**Purpose:** Turn raw data from `data/raw/` into clean, schema-fixed tables
in `data/processed/`: `drugs`, `ddi_pairs`, `adverse_events`. Normalizes drug
identifiers, removes duplicates, handles missing values.

**Depends on:** Module A.
**Feeds into:** Module C (Graph Construction), Module G (Recommendation).
**Type:** Software Engineering.
