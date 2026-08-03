"""End-to-end integration tests.

Unit tests check each module against its own contract. These check the
agreements *between* modules — that a risk score shown in a recommendation is
the same number the interaction endpoint reports, that the saved artifacts
describe the same dataset, and that realistic multi-drug cases behave.

A mismatch here means two modules each pass their own tests while disagreeing
with each other, which is exactly the class of bug unit tests miss.
"""

import pandas as pd
import pytest
import torch
from fastapi.testclient import TestClient

from src.api.main import app
from src.config import load_config, resolve_path
from src.explainability.explain import explain_interaction
from src.models.predict import get_service
from src.models.train import CHECKPOINT_NAME
from src.recommendation.recommend import get_recommendation_service

_config = load_config()
_graph_path = resolve_path(_config["paths"]["graph_dir"]) / "ddi_graph.pt"
_checkpoint_path = resolve_path(_config["paths"]["trained_models_dir"]) / CHECKPOINT_NAME
_processed_dir = resolve_path(_config["paths"]["processed_data_dir"])

pytestmark = pytest.mark.skipif(
    not (_graph_path.exists() and _checkpoint_path.exists()),
    reason="Requires the full pipeline to have been run.",
)


@pytest.fixture(scope="module")
def client():
    return TestClient(app)


@pytest.fixture(scope="module")
def predictor():
    return get_service()


@pytest.fixture(scope="module")
def recommender():
    return get_recommendation_service()


# --- Artifacts describe the same dataset -----------------------------------


def test_graph_matches_processed_drug_table(predictor):
    """The graph and the drug table are produced by different modules; a
    drift between them would silently misalign every drug ID."""
    drugs = pd.read_csv(_processed_dir / "drugs.csv")

    assert predictor.graph.num_nodes == len(drugs)
    assert set(predictor.graph.drug_ids) == set(drugs["drug_id"])


def test_checkpoint_matches_graph_features(predictor):
    """A checkpoint trained on a different feature count would load and then
    produce nonsense, so the dimensions must agree."""
    checkpoint = torch.load(_checkpoint_path, weights_only=False)
    assert checkpoint["in_channels"] == predictor.graph.x.shape[1]
    assert checkpoint["in_channels"] == len(predictor.graph.feature_names)


def test_drug_targets_reference_known_drugs(recommender, predictor):
    unknown = set(recommender.drug_targets["drug_id"]) - set(predictor.graph.drug_ids)
    assert not unknown, f"{len(unknown)} target rows reference drugs absent from the graph"


# --- Modules agree on the same number --------------------------------------


def test_recommendation_risk_matches_direct_interaction_check(client, recommender):
    """The risk shown against a recommendation must equal what /ddi/check
    reports for that same pair. If these diverge, the interface is showing a
    different number than the one the user can verify.
    """
    result = recommender.recommend("hypertension", ["Warfarin"])
    candidates = (result.recommended + result.excluded)[:5]
    assert candidates

    for candidate in candidates:
        response = client.post(
            "/ddi/check", json={"drug_a": candidate.drug_id, "drug_b": "Warfarin"}
        )
        assert response.json()["interaction_probability"] == pytest.approx(
            candidate.max_interaction_risk, abs=1e-4
        )


def test_explanation_probability_matches_recommendation_risk(recommender):
    """An explanation that cites a different probability than the score it
    explains would undermine the whole point of explaining it."""
    result = recommender.recommend("hypertension", ["Warfarin"])

    for candidate in (result.recommended + result.excluded)[:5]:
        explanation = explain_interaction(candidate.drug_id, "Warfarin")
        assert explanation.probability == pytest.approx(candidate.max_interaction_risk, abs=1e-4)


def test_api_and_direct_call_agree(client, recommender):
    """The Streamlit interface calls the orchestration function directly
    while the API goes over HTTP; both must produce the same ranking."""
    body = client.post(
        "/recommend",
        json={"condition": "hypertension", "current_medications": ["Warfarin"]},
    ).json()
    direct = recommender.recommend("hypertension", ["Warfarin"])

    assert [item["drug_id"] for item in body["recommended"]] == [
        candidate.drug_id for candidate in direct.recommended
    ]


# --- Realistic polypharmacy ------------------------------------------------


def test_polypharmacy_scenario(client):
    """A patient on five drugs — the case the recommender exists for."""
    medications = ["Warfarin", "Ibuprofen", "Metformin", "Losartan", "Phenytoin"]
    body = client.post(
        "/recommend",
        json={"condition": "depression", "current_medications": medications},
    ).json()

    assert body["candidates_considered"] > 0
    assert body["unrecognised_medications"] == []
    assert body["safe_candidates_found"] + len(body["warnings"]) == body["candidates_considered"]

    for item in body["recommended"] + body["warnings"]:
        assert item["riskiest_medication"] in medications


def test_risk_never_decreases_as_medications_are_added(recommender):
    """Risk is the worst interaction, so adding a medication can only raise
    it. A drop would mean the aggregation is wrong."""
    condition = "depression"
    one = {d.drug_id: d.max_interaction_risk for d in _all(recommender, condition, ["Warfarin"])}
    three = _all(recommender, condition, ["Warfarin", "Ibuprofen", "Phenytoin"])

    for candidate in three:
        if candidate.drug_id in one:
            assert candidate.max_interaction_risk >= one[candidate.drug_id] - 1e-9


def _all(recommender, condition, medications):
    result = recommender.recommend(condition, medications)
    return result.recommended + result.excluded


def test_patient_already_on_a_candidate_drug(recommender):
    """If the patient already takes a drug for this condition, it must not be
    recommended back to them."""
    candidates = recommender.candidates_for("depression")
    already_taking = candidates.iloc[0]["drug_id"]

    result = recommender.recommend("depression", [already_taking])
    returned = {c.drug_id for c in result.recommended + result.excluded}

    assert already_taking not in returned


# --- Every condition works end to end --------------------------------------


@pytest.mark.parametrize(
    "condition",
    [
        "hypertension",
        "pain and inflammation",
        "depression",
        "asthma and COPD",
        "cancer",
        "type 2 diabetes",
        "epilepsy",
        "anticoagulation",
        "anxiety and insomnia",
    ],
)
def test_every_condition_returns_a_usable_response(client, condition):
    response = client.post(
        "/recommend",
        json={"condition": condition, "current_medications": ["Warfarin"]},
    )

    assert response.status_code == 200
    body = response.json()
    assert body["condition"] == condition
    assert body["candidates_considered"] > 0

    # Every candidate must be accounted for as either acceptable or excluded.
    # Checked against safe_candidates_found rather than len(recommended),
    # which is capped — without that field these numbers would not add up and
    # a reader could not tell trimmed options from ruled-out ones.
    assert body["safe_candidates_found"] + len(body["warnings"]) == body["candidates_considered"]
    assert len(body["recommended"]) <= body["safe_candidates_found"]


# --- Determinism -----------------------------------------------------------


def test_full_stack_is_deterministic(client):
    """Same request, same answer — dropout must be off and nothing may
    depend on dictionary or set ordering."""
    payload = {"condition": "hypertension", "current_medications": ["Warfarin", "Ibuprofen"]}

    first = client.post("/recommend", json=payload).json()
    second = client.post("/recommend", json=payload).json()

    assert first == second


def test_medication_order_does_not_change_results(recommender):
    """Worst-case risk is order-independent, so listing medications in a
    different order must not change what is recommended."""
    forward = recommender.recommend("hypertension", ["Warfarin", "Ibuprofen"])
    reverse = recommender.recommend("hypertension", ["Ibuprofen", "Warfarin"])

    assert [c.drug_id for c in forward.recommended] == [c.drug_id for c in reverse.recommended]
    assert [c.score for c in forward.recommended] == [c.score for c in reverse.recommended]


# --- Reported metrics match the committed report ---------------------------


def test_served_metrics_match_the_committed_report(client):
    """The API serves the report from disk; if they disagree, the demo is
    showing figures that are not the ones under version control."""
    import json

    from src.evaluation.run import REPORT_NAME

    served = client.get("/model/metrics").json()
    with open(
        resolve_path(_config["paths"]["results_dir"]) / REPORT_NAME, encoding="utf-8"
    ) as handle:
        on_disk = json.load(handle)

    assert served == on_disk
