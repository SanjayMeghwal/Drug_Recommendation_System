"""Tests for Module H (API / orchestration).

Covers all seven designed endpoints, their error handling, and the response
contracts the Streamlit interface depends on.
"""

import pytest
from fastapi.testclient import TestClient

from src.api.main import app, check_readiness, orchestrate_recommendation
from src.config import load_config, resolve_path
from src.models.train import CHECKPOINT_NAME

_config = load_config()
_graph_path = resolve_path(_config["paths"]["graph_dir"]) / "ddi_graph.pt"
_checkpoint_path = resolve_path(_config["paths"]["trained_models_dir"]) / CHECKPOINT_NAME

pytestmark = pytest.mark.skipif(
    not (_graph_path.exists() and _checkpoint_path.exists()),
    reason="Requires the built graph and a trained checkpoint.",
)


@pytest.fixture(scope="module")
def client():
    return TestClient(app)


# --- /health ---------------------------------------------------------------


def test_health_reports_ready_when_artifacts_present(client):
    response = client.get("/health")

    assert response.status_code == 200
    body = response.json()
    assert body["status"] == "ready"
    assert body["missing"] == []


def test_check_readiness_returns_reasons_not_just_a_flag():
    """A bare boolean would not tell a user what to run to fix it."""
    ready, missing = check_readiness()
    assert ready is True
    assert isinstance(missing, list)


# --- /drugs/search ---------------------------------------------------------


def test_drug_search_finds_matches(client):
    response = client.get("/drugs/search", params={"query": "warf"})

    assert response.status_code == 200
    body = response.json()
    assert body["query"] == "warf"
    assert any(match["name"] == "Warfarin" for match in body["matches"])


def test_drug_search_is_case_insensitive(client):
    lower = client.get("/drugs/search", params={"query": "warfarin"}).json()
    upper = client.get("/drugs/search", params={"query": "WARFARIN"}).json()
    assert lower["matches"] == upper["matches"]


def test_drug_search_ranks_prefix_matches_first(client):
    """Typing "warf" should surface Warfarin, not a drug that merely
    contains those letters somewhere."""
    matches = client.get("/drugs/search", params={"query": "warf"}).json()["matches"]
    assert matches[0]["name"].lower().startswith("warf")


def test_drug_search_respects_limit(client):
    matches = client.get("/drugs/search", params={"query": "a", "limit": 3}).json()["matches"]
    assert len(matches) <= 3


def test_drug_search_rejects_empty_query(client):
    assert client.get("/drugs/search", params={"query": ""}).status_code == 422


def test_drug_search_returns_empty_for_no_match(client):
    matches = client.get("/drugs/search", params={"query": "zzzzznotadrug"}).json()["matches"]
    assert matches == []


# --- /conditions -----------------------------------------------------------


def test_conditions_lists_supported_conditions(client):
    response = client.get("/conditions")

    assert response.status_code == 200
    conditions = response.json()["conditions"]
    assert len(conditions) == 9
    assert "hypertension" in {entry["condition"] for entry in conditions}
    assert all(entry["targets"] for entry in conditions)


# --- /recommend ------------------------------------------------------------


def test_recommend_returns_full_contract(client):
    response = client.post(
        "/recommend",
        json={"condition": "hypertension", "current_medications": ["Warfarin"]},
    )

    assert response.status_code == 200
    body = response.json()
    assert body["condition"] == "hypertension"
    assert body["candidates_considered"] > 0
    assert body["recommended"]

    for item in body["recommended"]:
        assert 0.0 <= item["score"] <= 1.0
        assert item["risk_level"] in {"low", "moderate", "high"}
        assert item["explanation"]


def test_recommend_reports_excluded_drugs_as_warnings(client):
    """Warfarin interacts widely, so some candidates should be ruled out and
    surfaced rather than silently dropped."""
    body = client.post(
        "/recommend",
        json={"condition": "hypertension", "current_medications": ["Warfarin"]},
    ).json()

    assert body["warnings"]
    assert all(item["risk_level"] == "high" for item in body["warnings"])


def test_recommend_can_return_nothing_safe(client):
    """Every candidate may be too risky — type 2 diabetes alongside Warfarin
    excludes all 12. The reply must still explain itself: an empty list plus
    the candidate count and the warnings, never a bare empty list.
    """
    body = client.post(
        "/recommend",
        json={"condition": "type 2 diabetes", "current_medications": ["Warfarin"]},
    ).json()

    assert body["recommended"] == []
    assert body["candidates_considered"] > 0
    assert len(body["warnings"]) == body["candidates_considered"]


def test_recommend_works_without_current_medications(client):
    body = client.post(
        "/recommend", json={"condition": "hypertension", "current_medications": []}
    ).json()

    assert body["recommended"]
    assert all(item["max_interaction_risk"] == 0.0 for item in body["recommended"])


def test_recommend_reports_unrecognised_medications(client):
    body = client.post(
        "/recommend",
        json={"condition": "hypertension", "current_medications": ["NotARealDrug"]},
    ).json()

    assert body["unrecognised_medications"] == ["NotARealDrug"]


def test_recommend_rejects_unknown_condition(client):
    response = client.post(
        "/recommend", json={"condition": "dragon pox", "current_medications": []}
    )

    assert response.status_code == 404
    assert "dragon pox" in response.json()["detail"]


def test_recommend_rejects_malformed_body(client):
    assert client.post("/recommend", json={"current_medications": []}).status_code == 422


# --- /ddi/check ------------------------------------------------------------


def test_ddi_check_returns_probability(client):
    response = client.post("/ddi/check", json={"drug_a": "Warfarin", "drug_b": "Ibuprofen"})

    assert response.status_code == 200
    assert 0.0 <= response.json()["interaction_probability"] <= 1.0


def test_ddi_check_accepts_drugbank_ids(client):
    by_name = client.post("/ddi/check", json={"drug_a": "Warfarin", "drug_b": "Ibuprofen"}).json()
    by_id = client.post("/ddi/check", json={"drug_a": "DB00682", "drug_b": "DB01050"}).json()

    assert by_name["interaction_probability"] == pytest.approx(by_id["interaction_probability"])


def test_ddi_check_rejects_unknown_drug(client):
    response = client.post("/ddi/check", json={"drug_a": "NotARealDrug", "drug_b": "Warfarin"})

    assert response.status_code == 404
    assert "NotARealDrug" in response.json()["detail"]


# --- /explain --------------------------------------------------------------


def test_explain_returns_structural_evidence(client):
    response = client.post("/explain", json={"drug_a": "Warfarin", "drug_b": "Ibuprofen"})

    assert response.status_code == 200
    body = response.json()
    assert body["summary"]
    assert body["shared_partners"] > 0
    assert body["drug_a_interactions"] > 0
    assert 0.0 <= body["observed_interaction_rate"] <= 1.0


def test_explain_rejects_unknown_drug(client):
    response = client.post("/explain", json={"drug_a": "NotARealDrug", "drug_b": "Warfarin"})
    assert response.status_code == 404


# --- /model/metrics --------------------------------------------------------


def test_metrics_returns_evaluation_report(client):
    response = client.get("/model/metrics")

    assert response.status_code == 200
    body = response.json()
    assert body["dataset"]["drugs"] > 0
    assert 0.0 <= body["results"]["test"]["auc"] <= 1.0
    assert body["baseline_comparison"]
    # The caveat explains why our figures differ from the paper's; dropping
    # it would present the comparison as more direct than it is.
    assert body["comparison_caveat"]


# --- orchestration used directly by the interface --------------------------


def test_orchestrate_recommendation_callable_without_http():
    """The Streamlit interface calls this function directly rather than
    going over HTTP, so it must work standalone."""
    result = orchestrate_recommendation("type 2 diabetes", [])

    assert result["condition"] == "type 2 diabetes"
    assert isinstance(result["recommended"], list)
    assert isinstance(result["warnings"], list)


# --- documentation ---------------------------------------------------------


def test_openapi_documents_every_endpoint(client):
    paths = client.get("/openapi.json").json()["paths"]

    assert set(paths) == {
        "/health",
        "/drugs/search",
        "/conditions",
        "/recommend",
        "/ddi/check",
        "/explain",
        "/model/metrics",
    }
