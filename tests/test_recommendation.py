"""Tests for Module G (Recommendation) — the project's academic addition.

Requires the built graph, trained checkpoint, and processed tables, so these
skip on a fresh clone that has not run the pipeline.
"""

import pytest

from src.config import load_config, resolve_path
from src.models.train import CHECKPOINT_NAME
from src.recommendation.recommend import (
    RISK_HIGH,
    RISK_LOW,
    RISK_MODERATE,
    RecommendationService,
    UnknownConditionError,
)


# These need no fixtures — action_matches is a pure function.
@pytest.mark.parametrize(
    "recorded,required,expected",
    [
        # REGRESSION: "antagonist" contains "agonist" as a substring, so
        # substring matching classified every beta-blocker as a beta-agonist.
        ("antagonist", ["agonist"], False),
        ("agonist", ["agonist"], True),
        ("partial agonist", ["agonist"], True),
        ("antagonist", ["antagonist"], True),
        ("antagonist;inhibitor", ["inhibitor"], True),
        ("antagonist;partial agonist", ["agonist"], True),
        ("positive modulator", ["positive"], True),
        ("inhibitor", ["agonist"], False),
        ("unknown", ["inhibitor"], False),
        ("inhibitor, blocker", ["blocker"], True),
    ],
)
def test_action_matches_uses_whole_words(recorded, required, expected):
    assert RecommendationService.action_matches(recorded, required) is expected


_config = load_config()
_graph_path = resolve_path(_config["paths"]["graph_dir"]) / "ddi_graph.pt"
_checkpoint_path = resolve_path(_config["paths"]["trained_models_dir"]) / CHECKPOINT_NAME
_targets_path = resolve_path(_config["paths"]["processed_data_dir"]) / "drug_targets.csv"

pytestmark = pytest.mark.skipif(
    not (_graph_path.exists() and _checkpoint_path.exists() and _targets_path.exists()),
    reason="Requires the built graph, trained checkpoint, and processed tables.",
)


@pytest.fixture(scope="module")
def service():
    from src.recommendation.recommend import get_recommendation_service

    return get_recommendation_service()


def test_available_conditions_are_listed(service):
    conditions = service.available_conditions()

    assert len(conditions) == 9
    names = {entry["condition"] for entry in conditions}
    assert "hypertension" in names
    assert all(entry["targets"] for entry in conditions)


def test_condition_matching_is_case_insensitive(service):
    assert service.resolve_condition("HYPERTENSION") == "hypertension"
    assert service.resolve_condition("  Hypertension  ") == "hypertension"


def test_unknown_condition_raises(service):
    with pytest.raises(UnknownConditionError, match="not a known condition"):
        service.resolve_condition("dragon pox")


def test_every_condition_yields_candidates(service):
    """A condition with no candidates would be dead weight in the demo."""
    for entry in service.available_conditions():
        candidates = service.candidates_for(entry["condition"])
        assert not candidates.empty, f"{entry['condition']} has no candidate drugs"


def test_candidate_actions_match_the_condition(service):
    """A drug qualifies only if its recorded action matches what the
    condition requires — this is what keeps agonists and antagonists apart.
    """
    for entry in service.available_conditions():
        name = service.resolve_condition(entry["condition"])
        required = {
            target["gene"]: [action.lower() for action in target["actions"]]
            for target in service.conditions[name]["targets"]
        }

        for _, row in service.candidates_for(name).iterrows():
            # Checked with the real matcher, not a substring test — a
            # substring assertion would still pass with the agonist bug.
            assert RecommendationService.action_matches(row["action"], required[row["target_gene"]])


def test_adrb2_direction_differs_by_condition(service):
    """REGRESSION: ADRB2 agonists treat asthma, ADRB2 antagonists treat
    hypertension. Matching on target alone recommended bronchodilators for
    high blood pressure, which would raise it rather than lower it.
    """
    hypertension = service.candidates_for("hypertension")
    asthma = service.candidates_for("asthma and COPD")

    hypertension_adrb2 = hypertension[hypertension["target_gene"] == "ADRB2"]
    asthma_adrb2 = asthma[asthma["target_gene"] == "ADRB2"]

    assert not hypertension_adrb2.empty and not asthma_adrb2.empty
    assert all(
        RecommendationService.action_matches(a, ["antagonist"])
        for a in hypertension_adrb2["action"]
    )
    assert all(RecommendationService.action_matches(a, ["agonist"]) for a in asthma_adrb2["action"])

    overlap = set(hypertension_adrb2["drug_id"]) & set(asthma_adrb2["drug_id"])
    assert not overlap, "a drug cannot be both an ADRB2 agonist and antagonist here"


def test_recommendation_returns_ranked_results(service):
    result = service.recommend("hypertension", ["Warfarin"])

    assert result.condition == "hypertension"
    assert result.recommended
    scores = [drug.score for drug in result.recommended]
    assert scores == sorted(scores, reverse=True)


def test_score_is_one_minus_worst_risk(service):
    result = service.recommend("hypertension", ["Warfarin", "Ibuprofen"])

    for drug in result.recommended + result.excluded:
        assert drug.score == pytest.approx(1.0 - drug.max_interaction_risk, abs=1e-4)


def test_high_risk_candidates_are_excluded_not_recommended(service):
    result = service.recommend("hypertension", ["Warfarin"])

    assert all(drug.risk_level != RISK_HIGH for drug in result.recommended)
    assert all(drug.risk_level == RISK_HIGH for drug in result.excluded)
    assert all(drug.max_interaction_risk >= service.high_risk_threshold for drug in result.excluded)


def test_risk_levels_follow_thresholds(service):
    result = service.recommend("hypertension", ["Warfarin"])

    for drug in result.recommended + result.excluded:
        if drug.max_interaction_risk >= service.high_risk_threshold:
            assert drug.risk_level == RISK_HIGH
        elif drug.max_interaction_risk >= service.moderate_risk_threshold:
            assert drug.risk_level == RISK_MODERATE
        else:
            assert drug.risk_level == RISK_LOW


def test_current_medications_are_not_recommended_back(service):
    """Warfarin is an anticoagulation candidate; a patient already taking it
    should not be told to add it.
    """
    result = service.recommend("anticoagulation", ["Warfarin"])
    assert all(drug.drug_id != "DB00682" for drug in result.recommended + result.excluded)


def test_no_medications_means_no_risk(service):
    result = service.recommend("hypertension", [])

    assert result.recommended
    for drug in result.recommended:
        assert drug.max_interaction_risk == 0.0
        assert drug.score == 1.0
        assert drug.riskiest_medication is None
        assert drug.risk_level == RISK_LOW


def test_ties_break_toward_fewer_known_interactions(service):
    """With no medications every candidate scores 1.0, so ordering must fall
    back to overall interaction burden.
    """
    result = service.recommend("hypertension", [])
    burdens = [drug.known_interactions for drug in result.recommended]
    assert burdens == sorted(burdens)


def test_riskiest_medication_is_reported(service):
    result = service.recommend("hypertension", ["Warfarin", "Ibuprofen"])

    for drug in result.recommended + result.excluded:
        assert drug.riskiest_medication in {"Warfarin", "Ibuprofen"}


def test_unrecognised_medications_are_reported_not_ignored(service):
    """Silently dropping an unknown medication would hide an interaction we
    could not check.
    """
    result = service.recommend("hypertension", ["Warfarin", "NotARealDrug"])

    assert result.unrecognised_medications == ["NotARealDrug"]
    assert result.recommended or result.excluded


def test_recommendation_count_respects_configured_maximum(service):
    result = service.recommend("hypertension", [])
    assert len(result.recommended) <= service.max_recommendations


def test_more_medications_cannot_lower_risk(service):
    """Risk is the worst interaction, so adding a medication can only raise
    or maintain it, never reduce it.
    """
    single = {
        d.drug_id: d.max_interaction_risk
        for d in service.recommend("hypertension", ["Warfarin"]).recommended
    }
    both = service.recommend("hypertension", ["Warfarin", "Ibuprofen"])

    for drug in both.recommended:
        if drug.drug_id in single:
            assert drug.max_interaction_risk >= single[drug.drug_id] - 1e-9


def test_results_are_deterministic(service):
    first = service.recommend("hypertension", ["Warfarin"])
    second = service.recommend("hypertension", ["Warfarin"])

    assert [d.drug_id for d in first.recommended] == [d.drug_id for d in second.recommended]
    assert [d.score for d in first.recommended] == [d.score for d in second.recommended]


def test_result_serialises_to_dict(service):
    payload = service.recommend("hypertension", ["Warfarin"]).to_dict()

    assert set(payload) == {
        "condition",
        "recommended",
        "excluded",
        "candidates_considered",
        "unrecognised_medications",
    }
    if payload["recommended"]:
        assert "risk_level" in payload["recommended"][0]
