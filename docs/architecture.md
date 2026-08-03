# System Architecture

Nine modules, arranged so each owns one question. Nothing decides *what* to
recommend or *why* outside the module responsible for that decision.

## Two pipelines

**Offline** runs occasionally and ends with a trained, evaluated model.
**Online** runs per request and answers a patient's question.

```
OFFLINE (run once, or to reproduce training)

  A Ingestion ──▶ B Preprocessing ──▶ C Graph Construction ──▶ D Model
                                                                 │
                                                    ┌────────────┴───────────┐
                                                    ▼                        ▼
                                              F Evaluation            E Explainability
                                            (metrics report)          (wraps trained D)


ONLINE (per request, ~6 ms)

  I Interface ──▶ H Orchestration ──▶ G Recommendation ──┬──▶ D  (interaction scores)
        ▲                 │                              └──▶ B  (candidate drugs)
        │                 │
        │                 └──▶ E Explainability ──▶ D
        │                                                 
        └──────────── assembled response ─────────────────┘
```

## Modules

| # | Module | Owns | Type |
|---|---|---|---|
| A | `src/ingestion` | Fetching raw data | SE |
| B | `src/preprocessing` | Cleaning into fixed schemas | SE |
| C | `src/graph_construction` | Turning tables into a leakage-free graph | SE, ML-informed |
| D | `src/models` | Learning and scoring interactions | **ML** |
| E | `src/explainability` | Justifying a prediction | **ML / XAI** |
| F | `src/evaluation` | Measuring model quality | **ML** |
| G | `src/recommendation` | Ranking drugs by safety | SE over ML output |
| H | `src/api` | Coordination and HTTP surface | SE |
| I | `src/interface` | Presentation | SE |

Only D, E, and F are machine learning. Most of the code is ordinary software
engineering wrapped around them, but nearly all the risk sits in those three.

## Data flow, by shape

Each arrow is a real change in the data's shape:

```
raw CSVs                          A → data/raw/
  │ clean, normalise, dedupe
  ▼
drugs / ddi_pairs / drug_targets  B → data/processed/
  │ nodes + features + edges, split
  ▼
PyG graph object                  C → data/graph/
  │ message passing, training
  ▼
trained weights                   D → artifacts/trained_models/
  │
  ├─ metrics                      F → results/
  └─ predict(a, b) → probability

request: condition + medications  I
  │ resolve to identifiers
  ▼
condition_code, medication IDs    H
  │ targets → candidate drugs
  ▼
candidate list                    G
  │ batched scoring against each medication
  ▼
risk per (candidate, medication)  D
  │ worst-case aggregate, rank, filter
  ▼
ranked candidates                 G
  │ structural evidence
  ▼
explanations                      E
  │ merge, resolve names
  ▼
response → rendered screen        H → I
```

## Dependencies

```
A → B → C → D → E
        │       ▲
        │       │
        └─▶ G ◀─┘   (G needs both candidate data and interaction scores)
            │
D, E, G ──▶ H ──▶ I

D → F   (evaluation is off the request path)
```

**D is the critical path.** Everything in the intelligence layer waits on it.
That is why the build order fixed interface contracts first and stubbed every
module on day one, so E, F, G, and I could be developed against mocks while D
was still being trained.

## Design decisions

**No database.** A few thousand drugs fit comfortably in memory as CSVs read
at startup. A database server would add an operational dependency and buy
nothing at this size.

**No microservices.** One process, nine packages. Splitting them would
multiply deployment complexity for a system one person runs on a laptop.

**Configuration is data, not code.** `config/config.yaml` holds paths,
hyperparameters, and safety thresholds; `config/conditions.yaml` holds the
condition-to-target mapping. Nothing in `src/` hardcodes them, so the full
parameter set is readable in one place.

**Stage scripts run as modules** (`python -m src.<module>.run`), so imports
resolve regardless of working directory, and every stage can be re-run
independently.

**Services are cached singletons.** Scoring one drug pair requires a forward
pass over the whole graph, so the model, graph, and node embeddings load once.
This is what makes a 34-candidate recommendation take ~6 ms rather than
several seconds.

**The API and interface share one orchestration function.** The interface
calls it directly rather than over HTTP, so the demo needs a single command,
while the API remains independently testable. An integration test asserts
both paths produce identical rankings.

## Where the leakage controls live

Two safeguards in Module C, both regression-tested, because a link-prediction
model that has seen its test edges reports inflated metrics:

1. `edge_index` — the message-passing graph — contains **training positives
   only**.
2. Drug pairs are deduplicated to unique undirected edges and assigned to
   exactly one split, priority `testing > validation > training`. The source
   data lists a pair once per interaction type, and those rows can fall in
   different splits; collapsing 86 types to binary would otherwise have put
   231 pairs in both training and test.

Module E inherits this: it builds its neighbourhood index from `edge_index`,
so an explanation can never cite a held-out interaction.

## Testing

163 tests, structured to match the architecture:

| Suite | Tests | Focus |
|---|---|---|
| `test_ingestion` | 4 | Download integrity |
| `test_preprocessing` | 5 | Schema and cleaning, on synthetic data |
| `test_graph_construction` | 19 | Features, negative sampling, **leakage** |
| `test_models` | 16 | Architecture variants, training, save/reload |
| `test_predict` | 10 | Inference contract, symmetry, determinism |
| `test_evaluation` | 8 | Metric and threshold logic |
| `test_explainability` | 16 | Explanation content and evidence sourcing |
| `test_recommendation` | 29 | Candidate selection, scoring, risk bands |
| `test_api` | 24 | All seven endpoints and error handling |
| `test_interface` | 11 | User journeys, driven headlessly |
| `test_integration` | 21 | **Agreements between modules** |

Unit tests use small synthetic data so they neither depend on the 61 MB
dataset nor on a trained checkpoint. Tests that genuinely need the pipeline
skip themselves when the artifacts are absent, so a fresh clone can still run
the suite.
