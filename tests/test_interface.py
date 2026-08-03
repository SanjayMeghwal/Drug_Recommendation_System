"""Tests for Module I (the Streamlit interface).

Driven headlessly with Streamlit's AppTest, which executes the script exactly
as a browser session would. This matters because a Streamlit app fails at
runtime rather than import time — the previous version read a response key
that no longer existed and would have crashed only once a user clicked.
"""

import pytest
from streamlit.testing.v1 import AppTest

from src.config import load_config, resolve_path
from src.models.train import CHECKPOINT_NAME

APP_PATH = "src/interface/app.py"
TIMEOUT = 240

_config = load_config()
_graph_path = resolve_path(_config["paths"]["graph_dir"]) / "ddi_graph.pt"
_checkpoint_path = resolve_path(_config["paths"]["trained_models_dir"]) / CHECKPOINT_NAME

pytestmark = pytest.mark.skipif(
    not (_graph_path.exists() and _checkpoint_path.exists()),
    reason="Requires the built graph and a trained checkpoint.",
)


@pytest.fixture(scope="module")
def app():
    return AppTest.from_file(APP_PATH, default_timeout=TIMEOUT).run()


def test_app_runs_without_exceptions(app):
    assert not app.exception


def test_app_shows_the_prototype_disclaimer(app):
    """The tool is medical-adjacent, so the warning that it is not for
    clinical use must be present, not buried in a tab."""
    warnings = " ".join(element.value for element in app.warning)
    assert "not for clinical use" in warnings.lower()


def test_app_has_all_three_tabs(app):
    assert len(app.tabs) == 3


def test_condition_selector_is_a_dropdown_not_free_text(app):
    """Only nine conditions are valid, so free text would invite a failure
    the user cannot diagnose."""
    condition_widget = app.selectbox[0]
    assert len(condition_widget.options) == 9
    assert "hypertension" in condition_widget.options


def test_medication_input_offers_known_drugs_only(app):
    medications = app.multiselect[0]
    assert "Warfarin" in medications.options
    # Placeholder names explain nothing and must not be selectable.
    assert not any(option.startswith("Unknown (") for option in medications.options)


def test_recommendation_flow_produces_results(app):
    """The primary user journey: pick a condition and medication, click, and
    get recommendations with explanations."""
    app.selectbox[0].set_value("hypertension").run()
    app.multiselect[0].set_value(["Warfarin"]).run()
    app.button[0].click().run()

    assert not app.exception
    rendered = " ".join(element.value for element in app.markdown)
    assert "Worst interaction" in rendered


def test_no_safe_option_is_stated_explicitly():
    """Every candidate can be too risky — type 2 diabetes alongside Warfarin
    excludes all of them. An empty screen would look like a broken app, so
    the interface must say so outright.
    """
    app = AppTest.from_file(APP_PATH, default_timeout=TIMEOUT).run()
    app.selectbox[0].set_value("type 2 diabetes").run()
    app.multiselect[0].set_value(["Warfarin"]).run()
    app.button[0].click().run()

    assert not app.exception
    errors = " ".join(element.value for element in app.error)
    assert "No safe option found" in errors


def test_recommendation_works_with_no_medications():
    app = AppTest.from_file(APP_PATH, default_timeout=TIMEOUT).run()
    app.selectbox[0].set_value("hypertension").run()
    app.multiselect[0].set_value([]).run()
    app.button[0].click().run()

    assert not app.exception
    successes = " ".join(element.value for element in app.success)
    assert "ranked safest first" in successes


def test_interaction_check_flow():
    """Second tab: two drugs in, probability and explanation out."""
    app = AppTest.from_file(APP_PATH, default_timeout=TIMEOUT).run()
    app.selectbox[1].set_value("Warfarin").run()
    app.selectbox[2].set_value("Ibuprofen").run()
    app.button[1].click().run()

    assert not app.exception
    verdicts = " ".join(
        element.value for element in list(app.error) + list(app.warning) + list(app.success)
    )
    assert "probability" in verdicts.lower()


def test_interaction_check_rejects_identical_drugs():
    app = AppTest.from_file(APP_PATH, default_timeout=TIMEOUT).run()
    app.selectbox[1].set_value("Warfarin").run()
    app.selectbox[2].set_value("Warfarin").run()
    app.button[1].click().run()

    assert not app.exception
    warnings = " ".join(element.value for element in app.warning)
    assert "two different drugs" in warnings


def test_performance_tab_shows_metrics_and_caveat(app):
    """Reported numbers must travel with the caveat explaining why they
    differ from the baseline paper's."""
    metric_labels = {element.label for element in app.metric}
    assert {"AUC", "AUPR", "Accuracy", "F1"} <= metric_labels

    notes = " ".join(element.value for element in app.info)
    assert "baseline paper" in notes.lower()
