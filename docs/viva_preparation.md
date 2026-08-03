# Viva Preparation

Questions an examiner is likely to ask, with the honest answer and where the
evidence lives. Every number here is reproducible from the repository.

---

## The project

**What did you build?**
A graph neural network that predicts drug-drug interactions, plus a layer that
uses those predictions to recommend drugs for a condition ranked by how safe
they are alongside a patient's current medications — with an explanation for
every result. 1,704 drugs, 190,409 known interactions, test AUC 0.9501.

**What is reproduction and what is yours?**
Reproduction: the DDI prediction and explainability pipeline from the IEEE
2025 baseline paper. Mine: the personalized recommendation layer, which the
paper does not address, and the finding that the paper's SHAP-based
explanation method does not work for this architecture — with the evidence
and a working replacement.

**Why a graph neural network rather than a normal classifier?**
Interaction data is a network. Whether A interacts with B depends on what
else A and B each interact with — the ablation shows exactly this: shared
interaction partners correlate with the model's output at Spearman 0.784,
while molecular descriptors contribute 0.097 AUC. A row-wise classifier
cannot see that neighbourhood structure at all.

---

## Model and results

**Why GCN and not GAT, when the problem statement mentions attention?**
I tested both. GAT scored 0.7279 validation AUC against GCN's 0.9081 — it
collapsed within a few epochs. With only 12 features on a dense graph,
attention added parameters without signal. The problem statement asks for
"attention mechanisms **or** XAI techniques", so the requirement is met by
explainability. Choosing an architecture that measurably performs worse, to
match a word in the brief, would be the wrong trade. `model_selection.md`.

**Your accuracy is 0.874 but the paper reports 0.923. Why is it lower?**
Three reasons, and I would not want it to match:

1. Different dataset — the paper uses DrugBank 5.1.13 plus FAERS and includes
   drug-food interactions.
2. Stricter splitting — I removed 231 pairs that appeared in multiple splits.
   Keeping them would raise my accuracy by training on test data.
3. Sampled negatives — only 1,055 explicit non-interactions exist, so
   negatives come from unobserved pairs, some of which are probably real
   interactions. That caps precision.

Recall is 0.9189, slightly *above* the paper's 0.914, which fits reason 3:
the model flags interactions the labels do not record.

**How do you know you aren't leaking test data?**
Two enforced safeguards. The message-passing graph contains training edges
only. And pairs are deduplicated to unique undirected edges assigned to one
split, priority `testing > validation > training` — without which 231 pairs
would sit in both training and test. Both are regression-tested, and a test
asserts no validation or test edge appears in `edge_index`.

**How was the decision threshold chosen?**
Tuned on validation to maximise F1, then applied unchanged to test. Tuning it
on test would inflate every reported figure.

---

## Explainability

**The paper uses SHAP. Why don't you?**
I implemented it first. It produced attributions around 0.005 on predictions
spanning 0.5 to 0.96 — mathematically correct, practically meaningless.

The cause is neighbourhood dilution: average degree is 134, so after two
message-passing rounds a drug's own 12 descriptors are roughly 1/135 of its
embedding. Perturbing one pair barely moves its prediction. I checked this
was not just saturation on confident pairs — borderline pairs at 0.5 behaved
identically.

Rather than ship explanations that look authoritative and say nothing, I
built them on what the model demonstrably uses: shared interaction partners,
Spearman 0.784 with model output. `explainability.md` has the full evidence.

**So the molecular descriptors were wasted?**
No — they are just not *per-pair* explanatory. Removing all of them from all
1,704 drugs drops test AUC from 0.9501 to 0.8527, so chemistry contributes
0.097 AUC. LogP leads, which is chemically sensible: lipophilicity drives
cytochrome-P450 metabolism, a common real interaction mechanism. Feature
importance is therefore reported globally, where it is measurable.

**Is your explanation causal?**
No, and I state that. It shows two drugs behave like an interacting pair in
the network. It does not establish a mechanism. Anything stronger would
overclaim.

---

## Recommendation

**How do you know which drugs treat a condition, without indication data?**
I map nine conditions to their drug targets *and the action* the therapeutic
class exerts. The drug lists come from the data; only the condition-to-target
link is curated, and it is documented as demonstration scope.

**Why does action matter?**
ADRB2 is the clearest case. Hypertension needs beta-**antagonists**;
asthma needs beta-**agonists** — same protein, opposite direction. My first
version matched on target alone and recommended bronchodilators for high
blood pressure, drugs that would *raise* it. That is a real pharmacological
error I caught and fixed, with a regression test.

**Why worst-case risk rather than average?**
A drug dangerous with one medication is not made safe by being harmless with
four others. Averaging would hide exactly the interaction that matters.

**Sometimes it recommends nothing. Is that broken?**
No — it is the correct answer. Type 2 diabetes alongside Warfarin excludes
all 12 candidates, because Warfarin has 316 known interactions and is one of
the most interaction-prone drugs in use. The interface says so explicitly
rather than showing an empty screen, and the response reports how many were
considered and why each was excluded.

---

## Weaknesses

**What is the biggest weakness?**
The dataset records one target per drug, and it is not always the primary
therapeutic mechanism. Asenapine, an antipsychotic, appears as a hypertension
candidate because its recorded target happens to be ADRB2. Without indication
data I cannot fix this — a real indication dataset such as repoDB is the
first extension I would make.

**Could this be used clinically?**
No, and the interface says so above the results. It is trained on public data
with no clinical validation, personalization stops at current medications, and
predictions are model output rather than verified fact.

**What would you do with more time?**
A real indication dataset; multi-class prediction across all 86 interaction
types instead of binary; severity weighting so a dangerous interaction
outranks a mild one; and richer patient context.

---

## Engineering

**How do I know it works?**
163 tests. Unit tests per module on synthetic data, 21 integration tests
checking that modules agree with each other — that a recommendation's risk
equals what the interaction endpoint reports, that explanations cite the
probability they explain — and interface tests driving the app headlessly.

**Does it run on another machine?**
Verified. I cloned from GitHub into an empty directory, built a fresh
environment, and ran everything from the README. Every figure matched
exactly. It also caught that the install instruction was broken for everyone
except my own machine: `--index-url` replaces PyPI, and the PyTorch index
does not host torch's dependencies. The project worked perfectly where it was
written while being uninstallable anywhere else. `reproducibility.md`.

**Do I need to retrain to run it?**
No. The 17 KB checkpoint is committed, so a clone can evaluate and run the app
immediately. Training is reproducible in about six minutes on CPU, seeded.

---

## Live demo order

1. `python -m streamlit run src/interface/app.py`
2. **Recommend tab** — hypertension, patient on Warfarin. Show the ranked
   results, then expand an explanation.
3. Switch to **type 2 diabetes** with Warfarin — show that recommending
   nothing is an explicit, reasoned answer.
4. **Interaction tab** — Warfarin + Ibuprofen scores 0.96. A genuine clinical
   interaction: both raise bleeding risk. Point at the 98 shared partners.
5. **Performance tab** — metrics against the paper, with the caveat visible.
6. If asked for the API: `python -m uvicorn src.api.main:app` and open
   `/docs`.

## Numbers worth memorising

| | |
|---|---|
| Drugs / interactions | 1,704 / 190,409 |
| Test AUC / AUPR | 0.9501 / 0.9484 |
| Accuracy / precision / recall | 0.8741 / 0.8433 / 0.9189 |
| Model size | 3,457 parameters, 17 KB |
| Chemistry's contribution | 0.097 AUC |
| Shared-partner correlation | Spearman 0.784 |
| Leaking pairs removed | 231 |
| Tests | 163 |
| Request latency | ~6 ms |
