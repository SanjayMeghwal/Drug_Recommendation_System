# Data Acquisition

## Source

We use a pre-processed, DrugBank-derived DDI dataset rather than a raw DrugBank
export, because raw DrugBank requires an individual academic-license
registration that can't be scripted. This dataset is redistributed as CSV
files in a public GitHub repository (Das et al., GAINET, 2025) and is derived
from DrugBank, whose data is released under a **Creative Commons
Attribution-NonCommercial 4.0 International License (CC BY-NC 4.0)** —
free for non-commercial academic use with attribution, which this project
satisfies.

**Repository:** https://github.com/ozkantuncel/GAINET
**Underlying data source:** DrugBank (https://go.drugbank.com), CC BY-NC 4.0

**Citation (required by the license terms):**
- DrugBank: Wishart DS, et al. DrugBank.
- Das B, Dagdogen HA, Kaya MO, Tuncel O, Akgul MS, Das R. "GAINET: Enhancing
  Drug–Drug Interaction Predictions Through Graph Neural Networks and
  Attention Mechanisms." Chemometrics and Intelligent Laboratory Systems,
  vol. 259, 2025.

## Files

Downloaded by `src/ingestion/run.py` into `data/raw/`:

| File | Contents | Rows |
|---|---|---|
| `drug_smiles.csv` | DrugBank ID, SMILES string, per drug | 1,704 |
| `file_drugs.csv` | Full drug attribute table: name, synonyms, targets, pathways, external IDs (KEGG, PubChem, ChEBI, ChEMBL, etc.) | 1,704 |
| `ddi_training.csv` | Labeled drug pairs (d1, d2, interaction type, SMILES) — training split | 153,455 |
| `ddi_validation.csv` | Same schema — validation split | 19,148 |
| `ddi_test.csv` | Same schema — test split | 19,267 |

The `ddi_*.csv` files share this schema: `d1`, `d2` (DrugBank IDs), `type`
(interaction class — `0` denotes no known interaction, used as the negative
class; other values are specific interaction event types), `split`,
`smiles1`, `smiles2`.

## Reproducing this step

```
python src/ingestion/run.py
```

Downloads are skipped if the files already exist in `data/raw/` (idempotent).
Raw files are not committed to git — see `data/raw/README.md`.
