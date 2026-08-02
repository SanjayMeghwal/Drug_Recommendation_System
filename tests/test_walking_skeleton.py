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
        json={"condition": "hypertension", "current_medications": ["Aspirin"]},
    )
    assert response.status_code == 200

    body = response.json()
    assert "recommended" in body
    assert "warnings" in body
    assert len(body["recommended"]) > 0
    for item in body["recommended"]:
        assert "drug" in item
        assert "score" in item
        assert "explanation" in item


def test_ddi_check_endpoint_returns_expected_shape():
    response = client.post("/ddi/check", json={"drug_a": "Aspirin", "drug_b": "Warfarin"})
    assert response.status_code == 200

    body = response.json()
    assert body["drug_a"] == "Aspirin"
    assert body["drug_b"] == "Warfarin"
    assert isinstance(body["interaction_probability"], float)


def test_orchestrate_recommendation_function_directly():
    result = orchestrate_recommendation("diabetes", [])
    assert isinstance(result["recommended"], list)
    assert isinstance(result["warnings"], list)
