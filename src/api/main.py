"""Orchestration layer (Module H). Assembles Module G and Module E into the
/recommend response, and exposes Module D directly via /ddi/check.
"""

from fastapi import FastAPI, HTTPException

from src.api.schemas import DDICheckRequest, RecommendationRequest
from src.explainability.explain import explain_interaction
from src.models.predict import UnknownDrugError, predict
from src.recommendation.recommend import (
    UnknownConditionError,
    get_recommendation_service,
    recommend_drugs,
)

app = FastAPI(title="Explainable DDI & Drug Recommendation API")


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


@app.get("/health")
def health() -> dict:
    return {"status": "ready"}


@app.get("/conditions")
def conditions_endpoint() -> dict:
    """List the conditions the system can recommend for."""
    return {"conditions": get_recommendation_service().available_conditions()}


@app.post("/recommend")
def recommend_endpoint(request: RecommendationRequest) -> dict:
    try:
        return orchestrate_recommendation(request.condition, request.current_medications)
    except UnknownConditionError as error:
        raise HTTPException(status_code=404, detail=str(error)) from error


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
