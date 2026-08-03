# Model Selection

How the final architecture was chosen, and why the reported figures differ
from the baseline paper's.

**All selection decisions were made on the validation split.** The test split
was scored only once, after the architecture and training length were fixed.
Selecting on test would tune against the data used to report performance.

## 1. Encoder and decoder comparison

Starting point was the simplest configuration that met the requirement: a GCN
encoder with a parameter-free dot-product decoder. Four variants were then
compared under identical settings (200 epochs, patience 20, seed 42):

| Variant | Validation AUC | Best epoch | Epochs run |
|---|---|---|---|
| GCN + dot | 0.9081 | 119 | 139 |
| **GCN + mlp** | **0.9249** | 200 | 200 |
| GAT + dot | 0.7279 | 1 | 21 |
| GAT + mlp | 0.6915 | 17 | 37 |

**GCN + MLP decoder won.** Two observations:

- The learned MLP scorer beat the dot product by ~1.7 AUC points. The dot
  product forces the interaction score to be a plain inner product of two
  embeddings; the MLP can learn a more flexible combination.
- Both GAT variants performed far worse and stopped early, meaning validation
  AUC degraded almost immediately. With only 12 input features on a dense
  graph (13% of all possible drug pairs interact), attention weights appear to
  add parameters without adding usable signal. Attention was not pursued
  further — SHAP already satisfies the project's explainability requirement,
  so there was no reason to force an architecture that measurably hurt
  performance.

## 2. Training length

GCN + MLP recorded its best epoch at exactly the 200-epoch cap, meaning it was
still improving when training stopped. Re-running with an 800-epoch budget and
patience 50:

| Budget | Validation AUC |
|---|---|
| 200 epochs | 0.9249 |
| 800 epochs | 0.9491 |

Validation AUC plateaus around epochs 650–800, so 800 was adopted as the
final setting. Training takes roughly six minutes on CPU.

## 3. Final test-set results

Scored once, with the decision threshold (0.31) tuned on validation:

| Metric | Ours | Baseline paper | Difference |
|---|---|---|---|
| Accuracy | 0.8741 | 0.923 | −0.049 |
| Precision | 0.8433 | 0.891 | −0.048 |
| Recall | **0.9189** | 0.914 | **+0.005** |
| AUC | 0.9501 | not reported | — |
| AUPR | 0.9484 | not reported | — |

## 4. Why our accuracy is below the paper's

The comparison is a reference point, not a like-for-like reproduction. Three
concrete differences:

1. **Different dataset.** The paper uses DrugBank 5.1.13 plus FDA FAERS and
   also covers drug-food interactions. We use a DrugBank-derived DDI dataset
   and scope to drug-drug interactions only (see `data_acquisition.md`).
2. **Stricter splitting.** We remove 231 drug pairs that appeared in more than
   one split under different interaction types. Leaving them in would raise
   our scores while training on test data. Papers do not always document
   whether this deduplication was performed.
3. **Sampled negatives.** The source data contains only 1,055 explicit
   non-interactions against ~190k positives, so negatives are sampled from
   unobserved pairs. Some sampled "negatives" are likely real but
   undocumented interactions, which caps achievable precision.

Recall slightly exceeding the paper while precision sits lower is consistent
with point 3: the model flags interactions the label set does not record.

## Reproducing

```
python -m src.models.train      # ~6 minutes on CPU
python -m src.evaluation.run    # writes results/evaluation_report.json
```

Both are seeded (seed 42 in `config/config.yaml`).
