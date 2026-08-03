# Personalized Recommendation (Module G)

This module is the project's addition beyond the baseline paper. The paper
predicts whether two drugs interact; this takes that prediction and answers a
patient-facing question: *given what I already take, which drug for my
condition is safest to add?*

## How it works

```
condition + current medications
        |
        v
config/conditions.yaml   ->  target genes + required action
        |
        v
data/processed/drug_targets.csv  ->  candidate drugs
        |
        v
Module D scores every (candidate, current medication) pair   [one batched pass]
        |
        v
score = 1 - worst interaction risk        -> ranked, filtered
        |
        v
Module E explains the interaction that drove each score
```

A full request takes about 6 ms.

## Choosing candidates

The dataset has no clinical indications, only each drug's protein target.
`config/conditions.yaml` therefore maps nine conditions to the target genes
their established drug classes act on. The drug lists are never written by
hand — only the condition-to-target link is curated, and candidates are read
from the data.

**Action direction is part of the mapping, not an afterthought.** The same
protein is targeted in opposite directions for different conditions:

| Condition | Target | Required action | Drug class |
|---|---|---|---|
| Hypertension | ADRB2 | antagonist | beta-blockers |
| Asthma / COPD | ADRB2 | **agonist** | bronchodilators |

Matching on the target alone recommended bronchodilators for high blood
pressure — drugs that would *raise* it. Both the mapping and the matcher were
corrected for this, and the behaviour is regression-tested.

Matching is also **word-level, not substring**: `"antagonist"` contains
`"agonist"`, so substring matching silently classified every beta-blocker as
a beta-agonist. This bug was caught by the ADRB2 regression test and is now
covered by its own parametrised test.

## Ranking

Every candidate acts on a target associated with the condition, so relevance
carries no ranking signal and safety does all the work:

```
score = 1 - max(interaction risk against each current medication)
```

**Worst-case, not average.** A drug that is dangerous with one medication and
harmless with four others is still dangerous. Averaging would hide exactly the
interaction that matters.

**Tie-break by interaction burden.** With no current medications every
candidate scores 1.0, so ties break toward drugs with fewer known interactions
overall — a lower chance of future conflicts. Deterministic, and no arbitrary
blend weights.

## Risk bands

Configured in `config/config.yaml`:

| Risk | Threshold | Handling |
|---|---|---|
| Low | < 0.5 | Recommended |
| Moderate | 0.5 – 0.8 | Recommended, flagged for caution |
| High | ≥ 0.8 | **Excluded** and reported as a warning |

Excluded drugs are reported rather than silently dropped. Knowing what was
ruled out, and why, is part of the answer — and an unrecognised medication is
reported too, because an interaction that could not be checked must not look
like an interaction that was checked and found safe.

## Worked example

Condition *hypertension*, patient taking **Warfarin** and **Ibuprofen**:
34 candidates considered, 5 recommended, 29 excluded.

The exclusions are dominated by Warfarin, which has 316 known interactions —
realistic behaviour, since warfarin is one of the most interaction-prone drugs
in clinical use. Testosterone being excluded for a warfarin interaction
matches documented pharmacology, where androgens potentiate its
anticoagulant effect.

## Limitations

These are real and worth stating plainly.

- **One target per drug.** The dataset records a single target per drug, and
  it is not always the primary therapeutic mechanism. Asenapine, an
  antipsychotic, appears as a hypertension candidate because its recorded
  target happens to be ADRB2. Without indication data this cannot be fixed
  here.
- **The condition mapping is curated by us**, covering nine conditions with
  standard but coarse pharmacology. It omits many valid drug classes.
- **Relevance is uniform.** The system cannot say one candidate treats the
  condition better than another, only that it is safer alongside the
  patient's current medications.
- **Personalization is limited to current medications.** Age, weight, renal
  function, comorbidities and dosage are not modelled.
- **Predicted interactions are not verified facts.** They come from a model
  with test AUC 0.95 — strong, but wrong sometimes.

This is an academic prototype. It must not be used to inform real
prescribing.

## Trying it

```
python main.py                                    # worked example
python -m uvicorn src.api.main:app --reload       # then POST /recommend
```

`GET /conditions` lists the supported conditions.
