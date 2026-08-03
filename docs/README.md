# docs/

Written material for the project. Start with `project_report.md`.

| Document | Contents |
|---|---|
| **`project_report.md`** | **The overview**: problem, what was reproduced, what was added, results, findings, limitations |
| `architecture.md` | The nine modules, data flow, dependencies, and where the leakage controls sit |
| `model_selection.md` | How the architecture was chosen, and why our figures differ from the paper's |
| `explainability.md` | Why SHAP was rejected for this model, with evidence, and what replaced it |
| `recommendation.md` | The recommendation layer's design and limitations |
| `data_acquisition.md` | Dataset source, licence, and required citation |
| `reproducibility.md` | Clean-clone verification, and the install bug it caught |
| `viva_preparation.md` | Anticipated examiner questions with evidence-backed answers |

Module-level notes live beside the code, in each `src/<module>/README.md`.

Measured results are in `results/` — `evaluation_report.json` and
`feature_importance.json`, both committed and regenerable.
