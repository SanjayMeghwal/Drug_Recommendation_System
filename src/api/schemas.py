"""Request models for the API (Module H)."""

from pydantic import BaseModel


class RecommendationRequest(BaseModel):
    condition: str
    current_medications: list[str] = []


class DDICheckRequest(BaseModel):
    drug_a: str
    drug_b: str
