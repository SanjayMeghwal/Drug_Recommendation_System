"""Orchestration layer (Module H). Assembles Module G and Module E into the
/recommend response, and exposes Module D directly via /ddi/check.
"""

from fastapi import FastAPI, HTTPException

from src.api.schemas import DDICheckRequest, RecommendationRequest
from src.explainability.explain import explain_interaction
from src.models.predict import UnknownDrugError, predict
from src.recommendation.recommend import recommend

app = FastAPI(title="Explainable DDI & Drug Recommendation API")


def _explain_candidate(candidate_drug: str, current_medications: list[str]) -> str:
    """Explain a candidate against the patient's existing medications.

    An explanation only means something relative to what the patient is
    already taking, so with no current medications there is no interaction
    to justify.
    """
    if not current_medications:
        return (
            "No current medications were provided, so no drug-drug "
            "interaction could be assessed for this recommendation."
        )

    summaries = []
    for medication in current_medications:
        try:
            summaries.append(explain_interaction(candidate_drug, medication).summary)
        except UnknownDrugError as error:
            summaries.append(str(error))

    return " ".join(summaries)


def orchestrate_recommendation(condition: str, current_medications: list) -> dict:
    """Core orchestration logic, shared by the API route and the Streamlit interface."""
    candidates = recommend(condition, current_medications)
    recommended = [
        {**candidate, "explanation": _explain_candidate(candidate["drug"], current_medications)}
        for candidate in candidates
    ]
    return {"recommended": recommended, "warnings": []}


@app.get("/health")
def health() -> dict:
    return {"status": "ready"}


@app.post("/recommend")
def recommend_endpoint(request: RecommendationRequest) -> dict:
    return orchestrate_recommendation(request.condition, request.current_medications)


@app.post("/ddi/check")
def ddi_check_endpoint(request: DDICheckRequest) -> dict:
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


@app.post("/explain")
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
