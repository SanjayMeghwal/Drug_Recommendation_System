"""End-to-end test for the Day 1 walking skeleton: confirms the full
request/response chain (H -> G -> E, and H -> D) is wired correctly, even
though every module is currently a stub.
"""

from fastapi.testclient import TestClient

from src.api.main import app, orchestrate_recommendation

client = TestClient(app)


def test_health_endpoint():
    response = client.get("/health")
    assert response.status_code == 200
    assert response.json() == {"status": "ready"}


def test_recommend_endpoint_returns_expected_shape():
    response = client.post(
        "/recommend",
        json={"condition": "hypertension", "current_medications": ["Warfarin"]},
    )
    assert response.status_code == 200

    body = response.json()
    assert body["condition"] == "hypertension"
    assert "recommended" in body
    assert "warnings" in body
    assert len(body["recommended"]) > 0
    for item in body["recommended"]:
        assert "drug" in item
        assert "score" in item
        assert "risk_level" in item
        assert "explanation" in item


def test_recommend_endpoint_reports_unknown_condition():
    response = client.post(
        "/recommend", json={"condition": "dragon pox", "current_medications": []}
    )
    assert response.status_code == 404
    assert "dragon pox" in response.json()["detail"]


def test_conditions_endpoint_lists_supported_conditions():
    response = client.get("/conditions")
    assert response.status_code == 200

    conditions = response.json()["conditions"]
    assert conditions
    assert "hypertension" in {entry["condition"] for entry in conditions}


def test_ddi_check_endpoint_returns_expected_shape():
    response = client.post("/ddi/check", json={"drug_a": "Warfarin", "drug_b": "Ibuprofen"})
    assert response.status_code == 200

    body = response.json()
    assert body["drug_a"] == "Warfarin"
    assert body["drug_b"] == "Ibuprofen"
    assert 0.0 <= body["interaction_probability"] <= 1.0


def test_ddi_check_endpoint_reports_unknown_drug():
    """An unrecognised drug should return 404 rather than a 500, so the
    caller can tell the difference between a bad input and a broken server.
    """
    response = client.post("/ddi/check", json={"drug_a": "NotARealDrug", "drug_b": "Warfarin"})
    assert response.status_code == 404
    assert "NotARealDrug" in response.json()["detail"]


def test_orchestrate_recommendation_function_directly():
    result = orchestrate_recommendation("type 2 diabetes", [])
    assert isinstance(result["recommended"], list)
    assert isinstance(result["warnings"], list)
    assert result["condition"] == "type 2 diabetes"
