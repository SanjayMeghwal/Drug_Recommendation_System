# Explainable Graph Neural Network Framework for Personalized Drug Recommendation and Drug-Drug Interaction Prediction

MCA final-year project. A GNN-based system that predicts Drug-Drug Interactions (DDIs) and
provides personalized, explainable drug recommendations using XAI techniques (SHAP).

## Project Status

Design phase complete (architecture, technology stack, API design, folder structure).
Implementation has not started — see `docs/` for the finalized design.

## Reference Paper (Baseline)

"Graph Neural Network-Based Framework for Predicting Drug-To-Drug and Drug-To-Food
Interactions in Pharmacovigilance", IEEE, 2025. We reproduce the DDI + explainability
portion (DrugBank + FAERS) and add a personalized recommendation layer as our
academic improvement.

## Project Structure

| Folder | Purpose |
|---|---|
| `data/` | Raw and processed datasets, and the constructed drug interaction graph |
| `src/` | Source code, one subfolder per architecture module |
| `artifacts/` | Trained model files (generated, not hand-written) |
| `notebooks/` | Exploratory analysis and prototyping |
| `tests/` | Unit tests for the modules in `src/` |
| `config/` | Central configuration (paths, hyperparameters, thresholds) |
| `docs/` | Architecture diagrams, baseline paper notes, design documentation |
| `results/` | Generated evaluation reports and sample outputs |

See the README in each folder for details on what belongs there.

## Setup

```
python -m venv .venv
pip install -r requirements.txt
```

## Tech Stack

Python, PyTorch, PyTorch Geometric, NetworkX, Pandas/NumPy, SHAP, FastAPI, Streamlit,
Pytest, Black, Ruff. All free and open-source — see `requirements.txt`.
