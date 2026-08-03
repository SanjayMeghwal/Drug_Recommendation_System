"""Tests for the prediction service (Module D inference).

These require the full pipeline to have been run (graph + trained
checkpoint), so they are skipped when those artifacts are absent — a fresh
clone can still run the rest of the suite.
"""

import pytest

from src.config import load_config, resolve_path
from src.models.predict import UnknownDrugError
from src.models.train import CHECKPOINT_NAME

_config = load_config()
_graph_path = resolve_path(_config["paths"]["graph_dir"]) / "ddi_graph.pt"
_checkpoint_path = resolve_path(_config["paths"]["trained_models_dir"]) / CHECKPOINT_NAME

pytestmark = pytest.mark.skipif(
    not (_graph_path.exists() and _checkpoint_path.exists()),
    reason="Requires the built graph and a trained checkpoint.",
)


@pytest.fixture(scope="module")
def service():
    from src.models.predict import get_service

    return get_service()


def test_predict_returns_probability(service):
    score = service.predict("Warfarin", "Ibuprofen")
    assert 0.0 <= score <= 1.0


def test_predict_accepts_drugbank_ids(service):
    by_name = service.predict("Warfarin", "Ibuprofen")
    by_id = service.predict("DB00682", "DB01050")
    assert by_name == pytest.approx(by_id)


def test_predict_is_case_insensitive_for_names(service):
    assert service.predict("warfarin", "IBUPROFEN") == pytest.approx(
        service.predict("Warfarin", "Ibuprofen")
    )


def test_predict_is_symmetric(service):
    """Drug interactions are undirected, so argument order must not matter."""
    forward = service.predict("Warfarin", "Ibuprofen")
    reverse = service.predict("Ibuprofen", "Warfarin")
    assert forward == pytest.approx(reverse)


def test_predict_is_deterministic(service):
    """Dropout must be disabled at inference, or scores would vary per call."""
    scores = {service.predict("Warfarin", "Ibuprofen") for _ in range(5)}
    assert len(scores) == 1


def test_unknown_drug_raises(service):
    with pytest.raises(UnknownDrugError, match="NotARealDrug"):
        service.predict("NotARealDrug", "Warfarin")


def test_is_known_drug(service):
    assert service.is_known_drug("Warfarin")
    assert service.is_known_drug("DB00682")
    assert not service.is_known_drug("NotARealDrug")


def test_predict_batch_matches_individual_calls(service):
    pairs = [("Warfarin", "Ibuprofen"), ("Metformin", "Losartan")]

    batched = service.predict_batch(pairs)
    individual = [service.predict(a, b) for a, b in pairs]

    assert batched == pytest.approx(individual)


def test_predict_batch_handles_empty_input(service):
    assert service.predict_batch([]) == []


def test_name_for_maps_id_back_to_name(service):
    assert service.name_for("DB00682") == "Warfarin"
