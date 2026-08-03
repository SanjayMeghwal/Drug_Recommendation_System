# data/graph/

The serialized drug-interaction graph produced by `src/graph_construction`
(Module C) from `data/processed/`. Saved as `ddi_graph.pt`, a PyTorch
Geometric `Data` object.

Not committed to git — regenerate with:

```
python src/graph_construction/run.py
```

## Contents

| Attribute | Shape | Meaning |
|---|---|---|
| `x` | [1704, 12] | Node features: standardized RDKit molecular descriptors |
| `feature_names` | 12 | Descriptor names, so explanations can reference them |
| `drug_ids` | 1704 | Node index → DrugBank ID |
| `edge_index` | [2, 228406] | Message-passing edges: **training positives only**, both directions |
| `train_pos_edges` / `train_neg_edges` | [2, 114203] each | Training supervision |
| `val_pos_edges` / `val_neg_edges` | [2, 38090] each | Validation supervision |
| `test_pos_edges` / `test_neg_edges` | [2, 38116] each | Test supervision |

## Design decisions

**Binary link prediction.** The source data labels 86 distinct interaction
types; these are collapsed to interacts (`type != 0`) vs. does not
(`type == 0`), which is what the baseline paper's metrics imply and what the
recommendation module needs (a 0–1 risk score).

**Negative sampling.** Only 1,055 explicit non-interactions exist against
~190k positives, far too few to train a balanced classifier. Negatives are
sampled from unobserved drug pairs (seeded, reproducible), excluding every
known positive across all splits so a held-out interaction is never
mislabelled as a negative.

**Leakage control.** Two safeguards, both regression-tested:

1. `edge_index` contains only training positives — validation and test edges
   never reach the message-passing graph.
2. A drug pair can appear multiple times in the source data under different
   interaction types, and those rows may fall in different splits. Since the
   binary collapse makes them the same edge, pairs are deduplicated to unique
   undirected edges and assigned to exactly one split, with priority
   `testing > validation > training`. Without this, 231 pairs would have been
   both trained on and tested against.
