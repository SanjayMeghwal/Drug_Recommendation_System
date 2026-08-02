"""Orchestration layer (Module H). Assembles Module G and Module E into the
/recommend response, and exposes Module D directly via /ddi/check.
"""

from fastapi import FastAPI

from src.api.schemas import DDICheckRequest, RecommendationRequest
from src.explainability.explain import explain
from src.models.predict import predict
from src.recommendation.recommend import recommend

app = FastAPI(title="Explainable DDI & Drug Recommendation API")


def orchestrate_recommendation(condition: str, current_medications: list) -> dict:
    """Core orchestration logic, shared by the API route and the Streamlit interface."""
    candidates = recommend(condition, current_medications)
    recommended = [
        {**candidate, "explanation": explain(candidate["drug"], candidate["score"])}
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
    probability = predict(request.drug_a, request.drug_b)
    return {
        "drug_a": request.drug_a,
        "drug_b": request.drug_b,
        "interaction_probability": probability,
    }
