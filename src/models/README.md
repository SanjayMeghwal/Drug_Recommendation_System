# Module D — DDI Prediction Model

**Purpose:** The core reproduction of the baseline paper. A GNN trained on
the graph from `data/graph/` to predict interaction likelihood between two
drugs. Exposes a `predict(drug_a, drug_b) -> probability` interface and
saves trained weights to `artifacts/trained_models/`.

**Depends on:** Module C.
**Feeds into:** Module E (Explainability), Module F (Evaluation), Module G
(Recommendation).
**Type:** Machine Learning — the only module that learns from data.
