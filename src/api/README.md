# Module H — API / Orchestration

The integration point. Validates input, calls Modules D, E, and G, and shapes
the reply. It holds no domain logic of its own — anything deciding *what* to
recommend or *why* lives in the module that owns that question.

Run it:

```
python -m uvicorn src.api.main:app --reload
```

Interactive documentation is then at `http://127.0.0.1:8000/docs`, generated
from the response models in `schemas.py`.

## Endpoints

| Method | Path | Purpose | Backed by |
|---|---|---|---|
| GET | `/health` | Whether the service can actually answer requests | — |
| GET | `/drugs/search` | Find drugs by partial name, for autocomplete | Module D index |
| GET | `/conditions` | Conditions the system can recommend for | Module G |
| POST | `/recommend` | Rank drugs for a condition by safety | Modules G + E |
| POST | `/ddi/check` | Predict whether two drugs interact | Module D |
| POST | `/explain` | Explain a drug pair's interaction score | Module E |
| GET | `/model/metrics` | The model's measured performance | Module F report |

## Conventions

**404 for unknown inputs, not 500.** An unrecognised drug or condition is a
client mistake, and the message names the offending value so the caller can
correct it. A 500 would suggest the server is broken.

**`/health` loads rather than assumes.** It attempts to construct the
prediction and recommendation services and reports what failed, so a fresh
clone that has not run the pipeline is told exactly which command to run.
A server that is running is not necessarily a server that can answer.

**Nothing is silently dropped.** `/recommend` returns candidates excluded for
high interaction risk under `warnings`, and medications it could not resolve
under `unrecognised_medications`. An interaction that could not be checked
must never look like one that was checked and found safe.

**An empty recommendation list is still an answer.** Every candidate can be
too risky — type 2 diabetes alongside Warfarin excludes all 12. The reply
carries `candidates_considered` and the full `warnings` list, so an empty
result is never ambiguous.

**`/model/metrics` is served from the committed report**, not recomputed, so
the figures shown always match what is under version control.
