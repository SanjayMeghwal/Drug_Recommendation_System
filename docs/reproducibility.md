# Reproducibility Verification

Record of a clean-clone check: cloning from GitHub into an empty directory,
building a fresh environment, and running the whole system following only the
README. Nothing was carried over from the development machine.

Verified on Windows 11, Python 3.10.11.

## Procedure and results

| Step | Result |
|---|---|
| `git clone` into an empty directory | 66 files, including the 17 KB trained checkpoint and both result reports |
| Fresh `python -m venv .venv` | Clean, with pip 23.0.1 (not the dev machine's 26.2) |
| Install PyTorch, then `requirements.txt` | **Failed at first — see below**, passes after the fix |
| Import check from the README | All imports OK |
| `python -m src.ingestion.run` | 5 files, ~61 MB downloaded and verified |
| `python -m src.preprocessing.run` | 1,704 drugs, 191,870 pairs, 1,215 drug-target rows |
| `python -m src.graph_construction.run` | 228,406 message-passing edges; 114,203 / 38,090 / 38,116 supervision |
| `python -m src.evaluation.run` (committed checkpoint, no retraining) | AUC 0.9501, accuracy 0.8741, recall 0.9189 |
| `pytest` | 163 passed |
| `python main.py` | Identical recommendations to the dev machine |
| `uvicorn src.api.main:app` | `/health` reports ready, `/docs` renders |
| `streamlit run src/interface/app.py` | Serves successfully |

Every figure matched the development machine exactly, which is what the seeds
in `config/config.yaml` are there to guarantee.

## The bug this caught

The README's install instruction was **broken for anyone but us**:

```
pip install torch==2.13.0 --index-url https://download.pytorch.org/whl/cpu
```

`--index-url` *replaces* PyPI rather than adding to it, and the PyTorch index
does not host torch's dependencies, so a fresh machine fails with:

```
ERROR: Could not find a version that satisfies the requirement typing-extensions>=4.10.0
```

It never surfaced during development because the development environment
already had `typing-extensions`, and a later attempt silently succeeded from
pip's local wheel cache. Re-running with `--no-cache-dir` reproduced the
failure reliably.

The fix keeps PyPI available alongside the PyTorch index:

```
pip install torch==2.13.0 \
    --index-url https://download.pytorch.org/whl/cpu \
    --extra-index-url https://pypi.org/simple
```

This is the value of the exercise: the instruction looked correct, and the
project worked perfectly on the machine that wrote it, while being impossible
to install anywhere else.

## Retraining is optional

The trained checkpoint (17 KB) is committed, so a clone can evaluate and run
the app immediately. Training remains fully reproducible:

```
python -m src.models.train      # ~6 minutes on CPU, seed 42
```

## Known environment constraint

RDKit fails to import on Windows when the repository path is very long:

```
ImportError: DLL load failed while importing rdchem: The filename or extension is too long
```

This is the Windows 260-character path limit, not a project fault — it
appeared only when testing inside a deeply nested temporary directory. Clone
to a short path such as `C:\projects\` and it does not arise.
