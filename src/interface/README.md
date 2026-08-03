# Module I — Presentation / Demo Interface

The screen a user actually works with. Run it:

```
python -m streamlit run src/interface/app.py
```

Opens at `http://localhost:8501`.

## Three tabs

| Tab | Purpose |
|---|---|
| **Recommend a drug** | Pick a condition and current medications; get drugs ranked safest first, each with its explanation, plus the ones excluded for risk |
| **Check an interaction** | Score any two drugs directly and see why |
| **Model performance** | Test-set metrics against the baseline paper, with the caveat |

The two halves of the problem statement — interaction prediction and
personalized recommendation — get a tab each; the third is the evidence that
the reproduction works.

## Design decisions

**Dropdowns, not free text.** Only nine conditions are supported and drug
names must match exactly, so typing invites a failure the user cannot
diagnose. Selecting from the known set makes an invalid request impossible to
express. Streamlit's multiselect provides type-to-search, so picking from
1,258 named drugs stays fast.

**Calls the orchestration layer directly, not over HTTP.** The demo then runs
from a single command instead of requiring a separate API process. The API is
its own deliverable, covered by `tests/test_api.py` and browsable at `/docs`.

**Services are cached** with `@st.cache_resource`. Streamlit re-runs the whole
script on every widget change; without caching, each click would rebuild the
graph and reload the model.

**"No safe option" is stated outright.** Every candidate can be too risky —
type 2 diabetes alongside Warfarin excludes all twelve. An empty results area
would look like a broken app, so that case gets an explicit message and the
excluded list.

**Unrecognised medications are reported as an error**, never passed over. An
interaction that could not be checked must not be mistaken for one that was
checked and found safe.

**The prototype disclaimer sits above the tabs**, not inside one, so it cannot
be missed.

## Testing

`tests/test_interface.py` drives the app headlessly with Streamlit's
`AppTest`, which executes the script the way a browser session does. This
matters because Streamlit apps fail at runtime rather than import time — an
earlier version read a response key that no longer existed and would only
have crashed once a user clicked the button.
