"""Orchestration layer (Module H).

Assembles the recommendation (Module G), explanation (Module E), and
prediction (Module D) components into one HTTP surface, and exposes the
evaluation results (Module F) for inspection.

The layer holds no domain logic of its own — it validates input, calls the
modules, and shapes the reply. Anything that decides *what* to recommend or
*why* belongs in the module that owns that question.
"""

import json

from fastapi import FastAPI, HTTPException, Query

from src.api.schemas import (
    ConditionsResponse,
    DDICheckRequest,
    DDICheckResponse,
    DrugSearchResponse,
    ExplanationResponse,
    HealthResponse,
    MetricsResponse,
    RecommendationRequest,
    RecommendationResponse,
)
from src.config import load_config, resolve_path
from src.evaluation.run import REPORT_NAME
from src.explainability.explain import explain_interaction
from src.models.predict import UnknownDrugError, get_service, predict
from src.recommendation.recommend import (
    UnknownConditionError,
    get_recommendation_service,
    recommend_drugs,
)

app = FastAPI(
    title="Explainable DDI & Drug Recommendation API",
    description=(
        "Predicts drug-drug interactions with a graph neural network and "
        "recommends drugs for a condition ranked by how safe they are "
        "alongside a patient's current medications. Academic prototype — "
        "not for clinical use."
    ),
    version="1.0.0",
)


def check_readiness() -> tuple[bool, list[str]]:
    """Report whether the trained model and processed data actually loaded.

    Loading is attempted rather than assumed, so /health distinguishes a
    running server from a usable one. A fresh clone that has not run the
    pipeline yet gets told exactly what is missing.
    """
    missing: list[str] = []

    try:
        get_service()
    except (FileNotFoundError, OSError) as error:
        missing.append(str(error))
        return False, missing

    try:
        get_recommendation_service()
    except (FileNotFoundError, OSError) as error:
        missing.append(str(error))

    return not missing, missing


def _explain_candidate(candidate_drug: str, riskiest_medication: str | None) -> str:
    """Explain a candidate against the medication that drove its score.

    Only the worst interaction is explained, because that is the one the
    score is based on — explaining every medication would bury it.
    """
    if riskiest_medication is None:
        return (
            "No current medications were provided, so no drug-drug "
            "interaction could be assessed for this recommendation."
        )

    try:
        return explain_interaction(candidate_drug, riskiest_medication).summary
    except UnknownDrugError as error:
        return str(error)


def orchestrate_recommendation(condition: str, current_medications: list) -> dict:
    """Core orchestration logic, shared by the API route and the Streamlit interface."""
    result = recommend_drugs(condition, current_medications)

    def with_explanation(candidate) -> dict:
        return {
            **candidate.to_dict(),
            "explanation": _explain_candidate(candidate.drug, candidate.riskiest_medication),
        }

    return {
        "condition": result.condition,
        "recommended": [with_explanation(candidate) for candidate in result.recommended],
        # Drugs ruled out for interaction risk. Reported rather than silently
        # dropped: knowing what was excluded, and why, is part of the answer.
        "warnings": [with_explanation(candidate) for candidate in result.excluded],
        "candidates_considered": result.candidates_considered,
        "unrecognised_medications": result.unrecognised_medications,
    }


@app.get("/health", response_model=HealthResponse)
def health_endpoint() -> dict:
    """Report whether the service can actually answer requests."""
    ready, missing = check_readiness()
    return {"status": "ready" if ready else "not ready", "missing": missing}


@app.get("/drugs/search", response_model=DrugSearchResponse)
def drug_search_endpoint(
    query: str = Query(description="Partial drug name", min_length=1),
    limit: int = Query(default=10, ge=1, le=50),
) -> dict:
    """Find known drugs by partial name, for autocomplete."""
    return {"query": query, "matches": get_service().search_drugs(query, limit=limit)}


@app.get("/conditions", response_model=ConditionsResponse)
def conditions_endpoint() -> dict:
    """List the conditions the system can recommend for."""
    return {"conditions": get_recommendation_service().available_conditions()}


@app.post("/recommend", response_model=RecommendationResponse)
def recommend_endpoint(request: RecommendationRequest) -> dict:
    """Rank drugs for a condition by safety against current medications."""
    try:
        return orchestrate_recommendation(request.condition, request.current_medications)
    except UnknownConditionError as error:
        raise HTTPException(status_code=404, detail=str(error)) from error


@app.post("/ddi/check", response_model=DDICheckResponse)
def ddi_check_endpoint(request: DDICheckRequest) -> dict:
    """Predict whether two drugs interact."""
    try:
        probability = predict(request.drug_a, request.drug_b)
    except UnknownDrugError as error:
        # An unrecognised drug is a client mistake, not a server fault, and
        # the caller needs to know which identifier failed.
        raise HTTPException(status_code=404, detail=str(error)) from error

    return {
        "drug_a": request.drug_a,
        "drug_b": request.drug_b,
        "interaction_probability": probability,
    }


@app.post("/explain", response_model=ExplanationResponse)
def explain_endpoint(request: DDICheckRequest) -> dict:
    """Explain why a drug pair received its interaction score."""
    try:
        explanation = explain_interaction(request.drug_a, request.drug_b)
    except UnknownDrugError as error:
        raise HTTPException(status_code=404, detail=str(error)) from error

    return {
        "drug_a": explanation.drug_a,
        "drug_b": explanation.drug_b,
        "interaction_probability": explanation.probability,
        "summary": explanation.summary,
        "shared_partners": explanation.shared_partners,
        "shared_partner_names": explanation.shared_partner_names,
        "drug_a_interactions": explanation.drug_a_interactions,
        "drug_b_interactions": explanation.drug_b_interactions,
        "observed_interaction_rate": explanation.observed_interaction_rate,
    }


@app.get("/model/metrics", response_model=MetricsResponse)
def metrics_endpoint() -> dict:
    """Return the trained model's measured performance.

    Served from the report written by Module F rather than recomputed, so
    the figures shown always match the ones under version control.
    """
    config = load_config()
    report_path = resolve_path(config["paths"]["results_dir"]) / REPORT_NAME

    if not report_path.exists():
        raise HTTPException(
            status_code=404,
            detail=(
                f"No evaluation report at {report_path}. "
                "Run `python -m src.evaluation.run` first."
            ),
        )

    with open(report_path, encoding="utf-8") as handle:
        return json.load(handle)
