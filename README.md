# Explainable Graph Neural Network Framework for Personalized Drug Recommendation and Drug-Drug Interaction Prediction

MCA final-year project. A GNN-based system that predicts Drug-Drug Interactions (DDIs) and
provides personalized, explainable drug recommendations using XAI techniques (SHAP).

## Project Status

Design phase complete (architecture, technology stack, API design, folder structure).
Environment verified and pinned. Implementation is starting — see `docs/` for the
finalized design.

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

Requires Python 3.10.

```
python -m venv .venv

# Windows
.venv\Scripts\activate
# macOS/Linux
source .venv/bin/activate

# Install PyTorch (CPU-only build first, for a portable install regardless of GPU)
pip install torch==2.13.0 --index-url https://download.pytorch.org/whl/cpu

# Install everything else
pip install -r requirements.txt
```

Verify the install:

```
python -c "import torch, torch_geometric, networkx, pandas, numpy, shap, fastapi, uvicorn, streamlit, pytest; print('All imports OK')"
```

## Tech Stack

Python, PyTorch, PyTorch Geometric, NetworkX, Pandas/NumPy, SHAP, FastAPI, Streamlit,
Pytest, Black, Ruff. All free and open-source — see `requirements.txt`.
