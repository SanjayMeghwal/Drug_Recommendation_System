# Project Report

**Explainable Graph Neural Network Framework for Personalized Drug
Recommendation and Drug-Drug Interaction Prediction**

MCA final-year project. This document is the overview: what was reproduced,
what was added, what the results are, and what is genuinely wrong with it.

---

## 1. Problem

Two questions matter when a clinician adds a drug to a patient's regimen:

1. Will it interact with what the patient already takes?
2. Among the drugs that treat this condition, which is the best choice?

Both are usually answered from lookup tables and clinical memory rather than
learned from data. Drug interaction data is naturally a *network* — drug A
interacts with B, B with C — which makes it a graph learning problem rather
than a table classification one.

The project's constraint is that predictions must be **explainable**. An
unexplained "these drugs interact" is not actionable, and in a healthcare
context an unjustified recommendation is worse than none.

**Task types.** Graph learning is the foundation; DDI prediction is link
prediction on that graph; recommendation is personalized ranking; and
explainability is a requirement applied across both.

---

## 2. Baseline paper

> "Graph Neural Network-Based Framework for Predicting Drug-To-Drug and
> Drug-To-Food Interactions in Pharmacovigilance", IEEE, 2025.
> https://ieeexplore.ieee.org/document/11089819/

Reproduced: the DDI-prediction and explainability pipeline. Deliberately out
of scope: the drug-food interaction half, which is not part of this project's
problem statement.

**Reproduction is not the whole project.** The paper stops at predicting
whether two drugs interact. The problem statement also demands *personalized
recommendation*, which the paper does not address — that is this project's
contribution, described in section 5.

---

## 3. Data

No clinical indications were available, and DrugBank itself requires
individual registration that cannot be scripted. We use a DrugBank-derived
DDI dataset redistributed as CSVs, under DrugBank's **CC BY-NC 4.0** licence
— free for non-commercial academic use with attribution.

| | |
|---|---|
| Drugs | 1,704 (1,258 with names) |
| Interaction records | 191,870 across 86 interaction types |
| Unique undirected interactions | 190,409 |
| Node features | 12 RDKit molecular descriptors |
| Drug-target records | 1,215 drugs across 481 protein targets |

Full source, licence, and citation in `data_acquisition.md`.

---

## 4. Reproduction: DDI prediction

### Model

Encoder-decoder link prediction. Two GCN layers produce a 32-dimensional
embedding per drug; a small MLP scores each pair. 3,457 parameters, a 17 KB
checkpoint, about six minutes to train on CPU.

Architecture chosen on **validation only** (`model_selection.md`):

| Variant | Validation AUC |
|---|---|
| GCN + dot product | 0.9081 |
| **GCN + MLP** | **0.9249** |
| GAT + dot product | 0.7279 |
| GAT + MLP | 0.6915 |

GAT was dropped: it measurably hurt performance, and since SHAP was intended
to satisfy the explainability requirement there was no reason to force
attention. Extending training from 200 to 800 epochs — the MLP variant had
not converged at the cap — reached 0.9491.

### Results

Test set scored once, threshold tuned on validation:

| Metric | Ours | Baseline paper | Difference |
|---|---|---|---|
| AUC | **0.9501** | not reported | — |
| AUPR | **0.9484** | not reported | — |
| Accuracy | 0.8741 | 0.923 | −0.049 |
| Precision | 0.8433 | 0.891 | −0.048 |
| Recall | **0.9189** | 0.914 | **+0.005** |

**Why accuracy sits below the paper's.** Three concrete reasons, not
hand-waving:

1. **Different dataset.** The paper uses DrugBank 5.1.13 plus FDA FAERS and
   covers drug-food interactions too.
2. **Stricter splitting.** We remove 231 drug pairs that appeared in more
   than one split under different interaction types. Keeping them would raise
   our numbers while training on test data. Papers rarely state whether this
   deduplication was done.
3. **Sampled negatives.** Only 1,055 explicit non-interactions exist against
   190k positives, so negatives are drawn from unobserved pairs. Some are
   likely real but undocumented interactions, which caps achievable precision.

Recall exceeding the paper while precision lags is consistent with the third
point: the model flags interactions the labels do not record.

---

## 5. Contribution: personalized recommendation

The addition beyond the paper. Given a condition and the patient's current
medications, rank the candidate drugs by how safe they are to add.

Since the data has no indications, `config/conditions.yaml` maps nine
conditions to target genes **and the action** the therapeutic class exerts.
Candidate drug lists come from the data; only the condition-to-target link is
curated.

```
score = 1 − max(interaction risk against each current medication)
```

Worst-case rather than average: a drug dangerous with one medication is not
made safe by being harmless with four others. Candidates at or above 0.8 risk
are excluded and reported as warnings rather than silently dropped.

Design and limitations in `recommendation.md`.

---

## 6. Findings

Three results that were not anticipated at the design stage.

### 6.1 SHAP is uninformative for this model

The paper explains predictions with SHAP. Implemented as intended, it
produced attributions of roughly **0.005 on predictions spanning 0.5–0.96** —
too small to explain anything.

The cause is **neighbourhood dilution**: the graph averages degree 134, so
after two message-passing rounds a drug's own twelve descriptors are about
1/135 of its embedding. Perturbing one pair barely moves that pair's
prediction. Verified not to be saturation — borderline pairs near 0.5 behaved
identically.

The descriptors are not useless; they are simply not *per-pair* explanatory.
Removing all of them from all drugs costs **0.097 AUC** (0.9501 → 0.8527),
with LogP, aromatic ring count, and polar surface area leading. LogP topping
the list is chemically sensible given its role in P450 metabolism.

Explanations were therefore rebuilt on what the model measurably responds to:
shared interaction partners, correlated with its output at **Spearman 0.784**
and monotonic from 0.034 (no shared partners) to 0.986 (151+). Full evidence
in `explainability.md`.

### 6.2 Collapsing interaction types created hidden leakage

The source lists a drug pair once per interaction type, and those rows can
fall in different splits. Collapsing 86 types to binary makes them the same
edge, so **231 pairs were both trained on and tested against**. Fixed by
deduplicating to unique undirected edges with `testing > validation >
training` priority. Our metrics are lower for it, and honest.

### 6.3 Target alone is not enough to pick candidates

The same protein is targeted in opposite directions for different conditions.
Matching only on target gene recommended **ADRB2 agonists (bronchodilators)
for hypertension** — drugs that *raise* blood pressure. Required action is
now part of the mapping.

The first fix then had its own bug: `"antagonist"` contains `"agonist"` as a
substring, so beta-blockers still matched asthma's agonist requirement. The
regression test written for the first bug caught the second. Matching is now
word-level.

---

## 7. Engineering

| | |
|---|---|
| Tests | 163, all passing |
| Modules | 9, one responsibility each |
| API endpoints | 7, documented at `/docs` |
| Full recommendation request | ~6 ms |
| Reproducibility | Verified from a clean clone |

Clean-clone verification caught that the install instruction was **broken for
everyone except the development machine** — `--index-url` replaces PyPI, and
the PyTorch index does not host torch's dependencies. The project worked
perfectly where it was written while being uninstallable anywhere else.
Details in `reproducibility.md`.

---

## 8. Limitations

Stated plainly, because they matter more than the metrics.

- **One target per drug.** The dataset records a single target, not always
  the primary therapeutic mechanism. Asenapine, an antipsychotic, appears as
  a hypertension candidate because its recorded target is ADRB2.
- **The condition mapping is ours**, covering nine conditions with standard
  but coarse pharmacology, omitting many valid drug classes.
- **Relevance is uniform.** The system cannot say one candidate treats the
  condition better than another — only that it is safer alongside current
  medications.
- **Personalization stops at current medications.** Age, weight, renal
  function, comorbidities, and dosage are not modelled.
- **Explanations are associative.** Shared partners show two drugs behave
  like an interacting pair in the network; they do not establish a
  pharmacological mechanism.
- **Sampled negatives may include real interactions**, so precision is
  understated by an unknown amount.
- **Predictions are model output, not verified fact.** Test AUC 0.95 is
  strong and still wrong sometimes.

**This is an academic prototype and must not inform real prescribing.**

---

## 9. Possible extensions

- A real indication dataset (repoDB is free) to replace the curated mapping.
- Multi-class prediction over all 86 interaction types instead of binary.
- Severity weighting, so a dangerous interaction outranks a mild one.
- GNNExplainer for edge-level attribution alongside structural explanations.
- Richer patient context: age, renal function, comorbidities.

---

## 10. Reproducing

```
python -m src.ingestion.run           # download data
python -m src.preprocessing.run       # clean
python -m src.graph_construction.run  # build graph
python -m src.models.train            # optional, ~6 min; checkpoint is committed
python -m src.evaluation.run          # metrics
pytest                                # 163 tests
python -m streamlit run src/interface/app.py
```

Seeded throughout (seed 42). Verified to produce identical figures from a
clean clone.

## References

1. Graph Neural Network-Based Framework for Predicting Drug-To-Drug and
   Drug-To-Food Interactions in Pharmacovigilance. IEEE, 2025.
2. Wishart DS, et al. DrugBank. Licensed CC BY-NC 4.0.
3. Das B, Dagdogen HA, Kaya MO, Tuncel O, Akgul MS, Das R. GAINET.
   *Chemometrics and Intelligent Laboratory Systems*, vol. 259, 2025.
   (Source of the redistributed DDI CSVs.)
4. Lundberg SM, Lee S-I. A Unified Approach to Interpreting Model
   Predictions. *NeurIPS*, 2017. (SHAP.)
5. Kipf TN, Welling M. Semi-Supervised Classification with Graph
   Convolutional Networks. *ICLR*, 2017. (GCN.)
