"""Request and response models for the API (Module H).

Response models exist so FastAPI's generated documentation at /docs shows the
exact shape of every reply, rather than an opaque object.
"""

from pydantic import BaseModel, Field


class RecommendationRequest(BaseModel):
    condition: str = Field(description="Condition to recommend for, e.g. 'hypertension'")
    current_medications: list[str] = Field(
        default=[],
        description="Drugs the patient already takes, by name or DrugBank ID",
    )


class DDICheckRequest(BaseModel):
    drug_a: str = Field(description="First drug, by name or DrugBank ID")
    drug_b: str = Field(description="Second drug, by name or DrugBank ID")


class HealthResponse(BaseModel):
    status: str = Field(description="'ready' when the model and data are loaded")
    missing: list[str] = Field(default=[], description="What could not be loaded, when not ready")


class DrugMatch(BaseModel):
    drug_id: str
    name: str


class DrugSearchResponse(BaseModel):
    query: str
    matches: list[DrugMatch]


class ConditionSummary(BaseModel):
    condition: str
    description: str
    targets: list[dict]


class ConditionsResponse(BaseModel):
    conditions: list[ConditionSummary]


class DDICheckResponse(BaseModel):
    drug_a: str
    drug_b: str
    interaction_probability: float = Field(description="Predicted probability, 0 to 1")


class ExplanationResponse(BaseModel):
    drug_a: str
    drug_b: str
    interaction_probability: float
    summary: str = Field(description="Human-readable account of the prediction")
    shared_partners: int = Field(description="Interaction partners the two drugs share")
    shared_partner_names: list[str]
    drug_a_interactions: int
    drug_b_interactions: int
    observed_interaction_rate: float = Field(
        description="How often pairs sharing this many partners interact in training data"
    )


class RecommendedDrug(BaseModel):
    drug: str
    drug_id: str
    target_gene: str
    score: float = Field(description="1 minus the worst interaction risk")
    max_interaction_risk: float
    riskiest_medication: str | None
    risk_level: str = Field(description="low, moderate, or high")
    known_interactions: int
    explanation: str


class RecommendationResponse(BaseModel):
    condition: str
    recommended: list[RecommendedDrug]
    warnings: list[RecommendedDrug] = Field(
        description="Candidates excluded for high interaction risk"
    )
    candidates_considered: int
    safe_candidates_found: int = Field(
        description=(
            "Acceptable candidates found. Can exceed len(recommended), which is "
            "capped, so the counts still add up against candidates_considered."
        )
    )
    unrecognised_medications: list[str] = Field(
        description="Medications that could not be resolved, so were not checked"
    )


class MetricsResponse(BaseModel):
    dataset: dict
    results: dict
    baseline_comparison: dict
    comparison_caveat: str
