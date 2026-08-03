"""Personalized recommendation (Module G) — the project's academic addition.

The baseline paper stops at predicting whether two drugs interact. This
module builds on that: given a condition and the drugs a patient already
takes, it ranks candidate drugs by how safe they are to add.

How candidates are chosen: the dataset records each drug's protein target but
no clinical indications, so `config/conditions.yaml` maps each condition to
the target genes its established drug classes act on, together with the
action those drugs exert. Both halves matter — ADRB2 antagonists treat
hypertension while ADRB2 agonists treat asthma, so matching on the target
alone would recommend bronchodilators for high blood pressure. Candidates are
then read from data/processed/drug_targets.csv. Only the condition-to-target
link is curated; the drug lists come from the data.

How candidates are ranked: every candidate acts on a target associated with
the condition, so relevance carries no ranking signal here and safety does
all the work. A candidate's score is 1 minus its worst predicted interaction
against the patient's current medications — worst rather than average,
because one dangerous combination is not offset by several harmless ones.
Ties break toward drugs with fewer known interactions overall, which is a
lower ongoing interaction burden.
"""

from dataclasses import asdict, dataclass, field
from functools import lru_cache

import pandas as pd
import yaml

from src.config import PROJECT_ROOT, load_config, resolve_path
from src.models.predict import UnknownDrugError, get_service

CONDITIONS_PATH = PROJECT_ROOT / "config" / "conditions.yaml"

RISK_LOW = "low"
RISK_MODERATE = "moderate"
RISK_HIGH = "high"


class UnknownConditionError(KeyError):
    """Raised when a condition is not in the curated condition mapping."""


@dataclass
class CandidateDrug:
    """One candidate drug, scored against the patient's current medications."""

    drug: str
    drug_id: str
    target_gene: str
    score: float
    max_interaction_risk: float
    riskiest_medication: str | None
    risk_level: str
    known_interactions: int

    def to_dict(self) -> dict:
        return asdict(self)


@dataclass
class RecommendationResult:
    """Recommendations for one patient context, plus what was ruled out."""

    condition: str
    recommended: list[CandidateDrug] = field(default_factory=list)
    excluded: list[CandidateDrug] = field(default_factory=list)
    candidates_considered: int = 0
    unrecognised_medications: list[str] = field(default_factory=list)

    def to_dict(self) -> dict:
        return {
            "condition": self.condition,
            "recommended": [drug.to_dict() for drug in self.recommended],
            "excluded": [drug.to_dict() for drug in self.excluded],
            "candidates_considered": self.candidates_considered,
            "unrecognised_medications": self.unrecognised_medications,
        }


class RecommendationService:
    """Ranks candidate drugs for a condition by interaction safety."""

    def __init__(self) -> None:
        config = load_config()
        settings = config.get("recommendation") or {}

        self.high_risk_threshold = settings.get("high_risk_threshold", 0.8)
        self.moderate_risk_threshold = settings.get("moderate_risk_threshold", 0.5)
        self.max_recommendations = settings.get("max_recommendations", 10)

        self.predictor = get_service()

        with open(CONDITIONS_PATH, encoding="utf-8") as handle:
            self.conditions = yaml.safe_load(handle)["conditions"]

        targets_path = resolve_path(config["paths"]["processed_data_dir"]) / "drug_targets.csv"
        if not targets_path.exists():
            raise FileNotFoundError(
                f"Missing {targets_path}. Run `python -m src.preprocessing.run` first."
            )
        self.drug_targets = pd.read_csv(targets_path)

        # How many known interactions each drug has. Used only to break ties
        # between candidates that are equally safe for this patient.
        graph = self.predictor.graph
        self._interaction_counts: dict[str, int] = dict.fromkeys(graph.drug_ids, 0)
        for node_index in graph.edge_index[0].tolist():
            self._interaction_counts[graph.drug_ids[node_index]] += 1

    def available_conditions(self) -> list[dict]:
        """List the conditions the system can recommend for."""
        return [
            {
                "condition": name,
                "description": details.get("description", ""),
                "targets": details.get("targets", []),
            }
            for name, details in sorted(self.conditions.items())
        ]

    def resolve_condition(self, condition: str) -> str:
        """Match a condition name case-insensitively."""
        candidate = str(condition).strip().lower()
        for name in self.conditions:
            if name.lower() == candidate:
                return name

        known = ", ".join(sorted(self.conditions))
        raise UnknownConditionError(f"{condition!r} is not a known condition. Known: {known}.")

    @staticmethod
    def action_matches(recorded_action: str, required_actions: list[str]) -> bool:
        """Test whether a drug's recorded action is one the condition needs.

        Matching is word-level, not substring. "antagonist" contains
        "agonist" as a substring, so substring matching would treat every
        beta-blocker as a beta-agonist and offer bronchodilators and
        beta-blockers interchangeably.

        Recorded actions may list several roles separated by semicolons
        ("antagonist;inhibitor") or qualify one ("partial agonist"), so the
        string is split into words and each required action must appear as a
        whole word.
        """
        words = set()
        for part in str(recorded_action).lower().replace(",", ";").split(";"):
            words.update(part.split())

        return any(action.lower() in words for action in required_actions)

    def candidates_for(self, condition: str) -> pd.DataFrame:
        """Return drugs acting on a condition's targets in the right direction.

        A drug qualifies only if it hits one of the condition's target genes
        AND its recorded action matches what that condition requires, so an
        ADRB2 agonist is never offered for hypertension.
        """
        resolved = self.resolve_condition(condition)
        targets = self.conditions[resolved].get("targets", [])

        required_actions = {entry["gene"]: entry.get("actions", []) for entry in targets}

        matching = self.drug_targets[self.drug_targets["target_gene"].isin(required_actions)]
        if matching.empty:
            return matching

        qualifies = matching.apply(
            lambda row: self.action_matches(row["action"], required_actions[row["target_gene"]]),
            axis=1,
        )
        return matching[qualifies]

    def _risk_level(self, risk: float) -> str:
        if risk >= self.high_risk_threshold:
            return RISK_HIGH
        if risk >= self.moderate_risk_threshold:
            return RISK_MODERATE
        return RISK_LOW

    def recommend(self, condition: str, current_medications: list[str]) -> RecommendationResult:
        """Rank drugs for a condition by safety against current medications."""
        resolved_condition = self.resolve_condition(condition)
        candidates = self.candidates_for(resolved_condition)

        medication_ids: list[str] = []
        unrecognised: list[str] = []
        for medication in current_medications:
            try:
                medication_ids.append(self.predictor.resolve(medication))
            except UnknownDrugError:
                # Report rather than silently ignore: a medication we cannot
                # resolve is an interaction we cannot check for.
                unrecognised.append(medication)

        # A drug the patient already takes is not something to recommend adding.
        already_taking = set(medication_ids)
        candidate_rows = [
            row
            for _, row in candidates.iterrows()
            if row["drug_id"] not in already_taking
            and row["drug_id"] in self.predictor.index_of_drug_id
        ]

        scored = self._score_candidates(candidate_rows, medication_ids)

        # Safest first; among equally safe drugs prefer the one carrying
        # fewer known interactions overall.
        scored.sort(key=lambda drug: (-drug.score, drug.known_interactions, drug.drug))

        recommended = [drug for drug in scored if drug.risk_level != RISK_HIGH]
        excluded = [drug for drug in scored if drug.risk_level == RISK_HIGH]

        return RecommendationResult(
            condition=resolved_condition,
            recommended=recommended[: self.max_recommendations],
            excluded=excluded,
            candidates_considered=len(scored),
            unrecognised_medications=unrecognised,
        )

    def _score_candidates(
        self, candidate_rows: list, medication_ids: list[str]
    ) -> list[CandidateDrug]:
        """Score every candidate against every current medication.

        All pairs go through a single batched forward pass — scoring them one
        at a time would make an interactive request noticeably slow.
        """
        if not candidate_rows:
            return []

        pairs = [
            (row["drug_id"], medication_id)
            for row in candidate_rows
            for medication_id in medication_ids
        ]
        probabilities = self.predictor.predict_batch(pairs) if pairs else []

        scored = []
        for position, row in enumerate(candidate_rows):
            drug_id = row["drug_id"]

            if medication_ids:
                start = position * len(medication_ids)
                risks = probabilities[start : start + len(medication_ids)]
                max_risk = max(risks)
                riskiest = self.predictor.name_for(medication_ids[risks.index(max_risk)])
            else:
                # With nothing to interact against, no candidate carries risk.
                max_risk, riskiest = 0.0, None

            scored.append(
                CandidateDrug(
                    drug=self.predictor.name_for(drug_id),
                    drug_id=drug_id,
                    target_gene=row["target_gene"],
                    score=round(1.0 - max_risk, 4),
                    max_interaction_risk=round(max_risk, 4),
                    riskiest_medication=riskiest,
                    risk_level=self._risk_level(max_risk),
                    known_interactions=self._interaction_counts.get(drug_id, 0),
                )
            )

        return scored


@lru_cache(maxsize=1)
def get_recommendation_service() -> RecommendationService:
    """Return the shared recommendation service, loading it on first use."""
    return RecommendationService()


def recommend_drugs(condition: str, current_medications: list[str]) -> RecommendationResult:
    """Rank drugs for a condition against the patient's current medications."""
    return get_recommendation_service().recommend(condition, current_medications)


def recommend(condition: str, current_medications: list[str]) -> list[dict]:
    """Return recommended drugs as plain dictionaries."""
    return [drug.to_dict() for drug in recommend_drugs(condition, current_medications).recommended]
