# Explainable Graph Neural Network Framework for Personalized Drug Recommendation and Drug-Drug Interaction Prediction

MCA final-year project. A graph neural network that predicts drug-drug interactions,
and recommends drugs for a condition ranked by how safe they are alongside a patient's
current medications — with an explanation for every result.

> **Academic prototype — not for clinical use.** Predictions come from a model trained
> on public data and are not verified medical facts.

## Results

| | |
|---|---|
| Drugs / known interactions | 1,704 / 190,409 |
| Test AUC / AUPR | **0.9501** / 0.9484 |
| Accuracy / precision / recall | 0.8741 / 0.8433 / **0.9189** |
| Model | 3,457 parameters, 17 KB checkpoint |
| Tests | 163 passing |
| Recommendation latency | ~6 ms |

Recall slightly exceeds the baseline paper's 0.914; accuracy sits 4.9 points below its
0.923, for reasons set out in `docs/model_selection.md`.

## Reproduction and contribution

Baseline: *"Graph Neural Network-Based Framework for Predicting Drug-To-Drug and
Drug-To-Food Interactions in Pharmacovigilance"*, IEEE, 2025.

**Reproduced** — the DDI prediction and explainability pipeline. The drug-food half is
deliberately out of scope.

**Added** — the personalized recommendation layer, which the paper does not address,
plus the finding that the paper's SHAP-based explanation method does not work for this
architecture, with evidence and a working replacement. See `docs/project_report.md`.

## Project Structure

| Folder | Purpose |
|---|---|
| `src/` | Source code, one subfolder per architecture module (A–I) |
| `data/` | Raw and processed datasets, and the constructed interaction graph |
| `artifacts/` | Trained model checkpoint (committed, 17 KB) |
| `config/` | Paths, hyperparameters, safety thresholds, condition mapping |
| `docs/` | Project report, architecture, and design documentation |
| `results/` | Measured evaluation results (committed) |
| `tests/` | 163 tests |
| `notebooks/` | Exploratory analysis |

Each folder and `src/` module has its own README explaining what belongs there.

## Setup

Requires Python 3.10. On Windows, clone to a short path such as
`C:\projects\` — RDKit fails to load from deeply nested directories because
of the 260-character path limit.

```
python -m venv .venv

# Windows
.venv\Scripts\activate
# macOS/Linux
source .venv/bin/activate

# Install PyTorch first, from the CPU-only wheel index, so the install stays
# portable regardless of whether the machine has a GPU. Both indexes are
# needed: --index-url alone replaces PyPI, and PyTorch's index does not host
# torch's own dependencies.
pip install torch==2.13.0 --index-url https://download.pytorch.org/whl/cpu --extra-index-url https://pypi.org/simple

# Install everything else
pip install -r requirements.txt
```

Verify the install:

```
python -c "import torch, torch_geometric, networkx, pandas, numpy, shap, fastapi, uvicorn, streamlit, pytest; print('All imports OK')"
```

## Running the Pipeline

Run each stage from the repository root, in order. Every stage is a module
(`python -m ...`) so imports resolve regardless of the working directory.

```
python -m src.ingestion.run           # Module A: download the dataset into data/raw/
python -m src.preprocessing.run       # Module B: clean into data/processed/
python -m src.graph_construction.run  # Module C: build data/graph/ddi_graph.pt
python -m src.models.train            # Module D: train, save artifacts/trained_models/
```

Stages are idempotent — ingestion skips files already downloaded, and the
later stages simply overwrite their outputs.

Then run the tests and launch the app:

```
pytest
python -m uvicorn src.api.main:app --reload        # API on http://127.0.0.1:8000
python -m streamlit run src/interface/app.py       # UI on http://localhost:8501
```

The trained checkpoint is committed, so steps 1–3 followed by `pytest` and the
app work without retraining. `python -m src.models.train` takes about six
minutes on CPU if you want to reproduce training itself.

## Documentation

Start with **`docs/project_report.md`** — the overview of what was built, the results,
the findings, and the limitations.

| Document | Contents |
|---|---|
| `docs/project_report.md` | **The overview.** Problem, reproduction, contribution, results, limitations |
| `docs/architecture.md` | Nine modules, data flow, dependencies, leakage controls |
| `docs/model_selection.md` | How the architecture was chosen, and why our figures differ from the paper's |
| `docs/explainability.md` | Why SHAP was rejected for this model, with evidence, and what replaced it |
| `docs/recommendation.md` | The recommendation layer's design and its limitations |
| `docs/data_acquisition.md` | Dataset source, licence, and required citation |
| `docs/reproducibility.md` | Clean-clone verification, and the install bug it caught |
| `docs/viva_preparation.md` | Anticipated examiner questions with evidence-backed answers |

## Tech Stack

Python 3.10, PyTorch, PyTorch Geometric, RDKit, NetworkX, Pandas/NumPy, scikit-learn,
FastAPI, Streamlit, Pytest, Black, Ruff. All free and open-source — see
`requirements.txt` for pinned versions.

SHAP is installed and was used for the explainability analysis in
`docs/explainability.md`, which concluded it is uninformative for this architecture;
explanations are structural instead.

## Licence and attribution

Data derives from DrugBank under **CC BY-NC 4.0** — free for non-commercial academic
use with attribution. Required citations are in `docs/data_acquisition.md`.
