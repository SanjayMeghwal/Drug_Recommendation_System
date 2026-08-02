# Module C — Graph Construction

**Purpose:** Build the drug-interaction graph from `data/processed/`: drugs
as nodes with feature vectors, known DDIs as edges. Produces the train/test
edge split and saves the graph object to `data/graph/`.

**Depends on:** Module B.
**Feeds into:** Module D (DDI Prediction Model).
**Type:** Software Engineering, with ML-influenced design decisions
(feature representation, edge split strategy).
