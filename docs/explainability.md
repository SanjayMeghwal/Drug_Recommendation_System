# Explainability

How Module E explains predictions, why the baseline paper's SHAP approach was
tried and rejected, and the evidence behind both decisions.

## Summary

The baseline paper explains predictions with SHAP and LIME. We implemented
SHAP first, measured it, and found it produces attributions far too small to
explain anything for this architecture. Explanations are therefore built from
the interaction network the model demonstrably relies on. The SHAP analysis is
retained here as a documented negative result, and descriptor importance is
reported globally instead, where it is measurable.

## 1. Why per-pair SHAP failed

SHAP was applied exactly as intended on Day 4: a function taking the 24
standardized descriptors of a drug pair (12 each), substituting them into the
node feature matrix, re-running the GNN, and returning the interaction
probability. `shap.KernelExplainer` then attributed the prediction across
those 24 named inputs.

The attributions were negligible:

| Drug pair | Predicted probability | Prediction with descriptors zeroed | Largest \|SHAP value\| |
|---|---|---|---|
| Warfarin + Ibuprofen | 0.965 | 0.965 | 0.003 |
| borderline pair | 0.501 | 0.511 | 0.005 |
| mid-confidence pair | 0.700 | 0.705 | 0.006 |
| high-confidence pair | 0.900 | 0.909 | 0.007 |

Replacing a pair's entire chemistry with the dataset average moved the
prediction by roughly 0.01. An explanation built on that would attribute a
0.96 prediction to effects of size 0.005 — technically correct and practically
meaningless.

**Cause: neighbourhood dilution.** The drug graph has 1,704 nodes and 228,406
message-passing edges, an average degree of 134. In a GCN, a node's embedding
is an aggregation over its neighbours, so after two layers a drug's own twelve
descriptors account for roughly 1/135 of its representation. Perturbing one
pair barely changes that pair's embedding.

This was verified not to be a saturation artefact: borderline predictions
near 0.5 showed the same negligible attributions as confident ones.

The cost also mattered. Each SHAP sample requires a full graph forward pass
(~96 ms), so SHAP's default 2,048 samples would take about 197 seconds per
explanation.

## 2. The descriptors still matter — globally

Dilution is a per-pair effect. Removing a descriptor from **all** drugs at once
measurably degrades the model (`python -m src.explainability.feature_importance`):

| Descriptor | Test AUC drop when removed |
|---|---|
| LogP (lipophilicity) | 0.0260 |
| NumAromaticRings | 0.0190 |
| TPSA (polar surface area) | 0.0166 |
| FractionCSP3 | 0.0069 |
| HeavyAtomCount | 0.0065 |
| RingCount | 0.0062 |
| NumHAcceptors | 0.0054 |
| NumHDonors | 0.0052 |
| NumHeteroatoms | 0.0025 |
| MolecularWeight | 0.0024 |
| NumRotatableBonds | 0.0022 |
| MolarRefractivity | 0.0016 |

Removing every descriptor drops test AUC from **0.9501 to 0.8527**, so
chemistry contributes about 0.097 AUC and network structure carries the rest.

LogP leading is chemically sensible: lipophilicity drives membrane permeability
and cytochrome-P450 metabolism, a common mechanism behind real interactions.

So the Day 4 choice of interpretable descriptors was not wasted — it supports a
meaningful *global* statement about what the model uses, just not a per-pair one.

## 3. What Module E actually reports

Since predictions are driven mainly by graph structure, explanations describe
that structure. Model output correlates with the number of interaction
partners two drugs share at **Spearman 0.784**, and the relationship is
monotonic:

| Shared interaction partners | Mean predicted probability |
|---|---|
| 0 | 0.034 |
| 1–25 | 0.151 |
| 26–75 | 0.500 |
| 76–150 | 0.922 |
| 151+ | 0.986 |

Each explanation therefore reports:

- the predicted probability and whether the pair is likely to interact
- how many interaction partners the two drugs share
- how often pairs sharing that many partners actually interact in the data
- how many known interactions each drug has individually
- the most-connected shared partners, by name

Example output:

> Warfarin and Ibuprofen are likely to interact (predicted probability 0.96).
> They share many known interaction partners (98); in the training data, drug
> pairs sharing this many partners interact 53% of the time. Warfarin has 316
> known interactions and Ibuprofen has 229. Shared partners include Phenytoin,
> Amiodarone, Carbamazepine, Fluvoxamine, Vemurafenib.

Two properties worth noting:

- **Faithful.** It describes the signal the model measurably responds to,
  rather than one that sounds plausible.
- **Fast.** Sub-millisecond after a one-off 0.6 s index build, versus ~10 s
  for a single reduced-sample SHAP explanation.

Neighbourhoods are built from `edge_index`, which contains **training
positives only**. An explanation can never cite a held-out interaction the
model was never shown — enforced by a test.

## 4. Honest limitations

- Shared partners explain *that* two drugs behave like an interacting pair in
  the network, not the pharmacological *mechanism*. It is association, not
  causation.
- Drugs with few recorded interactions get thin explanations, because there is
  little network evidence to cite.
- 446 of 1,704 drugs have no name in the source data. These are excluded from
  the listed shared partners, since "Unknown (DB01232)" explains nothing.
- Band interaction rates are estimated from a 20,000-pair sample of the
  training graph, so they carry sampling error.

## Reproducing

```
python -m src.explainability.feature_importance   # writes results/feature_importance.json
```

Per-pair explanations are available through `POST /explain` or
`src.explainability.explain.explain_interaction(drug_a, drug_b)`.
