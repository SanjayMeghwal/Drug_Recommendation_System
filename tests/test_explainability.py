"""Tests for Module E (Explainability).

Requires the built graph and trained checkpoint, so these skip on a fresh
clone that has not run the pipeline yet.
"""

import pytest

from src.config import load_config, resolve_path
from src.explainability.explain import (
    SHARED_PARTNER_BANDS,
    Explanation,
    InteractionExplainer,
)
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
def explainer():
    from src.explainability.explain import get_explainer

    return get_explainer()


def test_band_label_covers_full_range():
    """Every non-negative shared-partner count must map to some band."""
    for count in [0, 1, 25, 26, 75, 76, 150, 151, 10_000]:
        assert InteractionExplainer._band_label(count) in {
            label for *_, label in SHARED_PARTNER_BANDS
        }


def test_band_labels_are_ordered_by_count():
    labels = [InteractionExplainer._band_label(c) for c in (0, 10, 50, 100, 500)]
    assert labels == ["no", "few", "a moderate number of", "many", "very many"]


def test_explanation_has_expected_shape(explainer):
    explanation = explainer.explain_interaction("Warfarin", "Ibuprofen")

    assert isinstance(explanation, Explanation)
    assert explanation.drug_a == "Warfarin"
    assert explanation.drug_b == "Ibuprofen"
    assert 0.0 <= explanation.probability <= 1.0
    assert explanation.shared_partners >= 0
    assert explanation.summary


def test_explanation_is_symmetric_in_probability(explainer):
    forward = explainer.explain_interaction("Warfarin", "Ibuprofen")
    reverse = explainer.explain_interaction("Ibuprofen", "Warfarin")

    assert forward.probability == pytest.approx(reverse.probability)
    assert forward.shared_partners == reverse.shared_partners


def test_summary_mentions_both_drugs_and_the_probability(explainer):
    explanation = explainer.explain_interaction("Warfarin", "Ibuprofen")

    assert "Warfarin" in explanation.summary
    assert "Ibuprofen" in explanation.summary
    assert f"{explanation.probability:.2f}" in explanation.summary


def test_summary_reports_shared_partner_count(explainer):
    explanation = explainer.explain_interaction("Warfarin", "Ibuprofen")
    assert str(explanation.shared_partners) in explanation.summary


def test_interacting_pair_reports_shared_partners(explainer):
    """Warfarin and Ibuprofen is a real clinical interaction; the model
    scores it highly, and the explanation should point at shared partners.
    """
    explanation = explainer.explain_interaction("Warfarin", "Ibuprofen")

    assert explanation.probability > 0.5
    assert explanation.shared_partners > 0
    assert explanation.shared_partner_names


def test_unrelated_pair_reports_no_shared_partners(explainer):
    """Titanium dioxide is an inert excipient with a single recorded
    interaction, so it should share nothing with Warfarin.
    """
    explanation = explainer.explain_interaction("Warfarin", "Titanium dioxide")

    assert explanation.shared_partners == 0
    assert explanation.probability < 0.5


def test_placeholder_names_are_not_listed(explainer):
    """446 drugs have no name in the source data; "Unknown (DB01232)"
    explains nothing and must not appear in a user-facing explanation.
    """
    for drug_a, drug_b in [("Warfarin", "Ibuprofen"), ("Metformin", "Losartan")]:
        explanation = explainer.explain_interaction(drug_a, drug_b)
        assert not any(name.startswith("Unknown (") for name in explanation.shared_partner_names)
        assert "Unknown (" not in explanation.summary


def test_shared_partners_listed_are_capped(explainer):
    from src.explainability.explain import MAX_LISTED_PARTNERS

    explanation = explainer.explain_interaction("Warfarin", "Ibuprofen")
    assert len(explanation.shared_partner_names) <= MAX_LISTED_PARTNERS


def test_explanation_uses_training_edges_only(explainer):
    """Neighbourhoods come from edge_index, which holds training positives
    only — an explanation must never cite a held-out interaction.
    """
    graph = explainer.service.graph
    training_pairs = {
        (min(s, t), max(s, t))
        for s, t in zip(graph.edge_index[0].tolist(), graph.edge_index[1].tolist())
    }

    for index, partners in enumerate(explainer._neighbors[:50]):
        for partner in partners:
            assert (min(index, partner), max(index, partner)) in training_pairs


def test_unknown_drug_raises(explainer):
    with pytest.raises(UnknownDrugError):
        explainer.explain_interaction("NotARealDrug", "Warfarin")


def test_explain_against_medications_returns_one_per_medication(explainer):
    explanations = explainer.explain_against_medications("Warfarin", ["Ibuprofen", "Metformin"])

    assert len(explanations) == 2
    assert explanations[0].drug_b == "Ibuprofen"
    assert explanations[1].drug_b == "Metformin"


def test_explain_against_no_medications_returns_empty(explainer):
    assert explainer.explain_against_medications("Warfarin", []) == []


def test_band_rates_are_probabilities(explainer):
    for label, rate in explainer._band_rates.items():
        assert 0.0 <= rate <= 1.0, f"band {label} has invalid rate {rate}"


def test_band_rates_increase_with_shared_partners(explainer):
    """More shared interaction partners should mean a higher observed
    interaction rate — this is the relationship the explanation relies on.
    """
    rates = [explainer._band_rates[label] for *_, label in SHARED_PARTNER_BANDS]
    assert rates[0] < rates[-1]
