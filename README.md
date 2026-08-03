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

See `docs/data_acquisition.md` for the dataset source, license, and citation.

## Tech Stack

Python, PyTorch, PyTorch Geometric, NetworkX, Pandas/NumPy, SHAP, FastAPI, Streamlit,
Pytest, Black, Ruff. All free and open-source — see `requirements.txt`.
