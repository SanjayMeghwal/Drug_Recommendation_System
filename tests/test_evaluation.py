"""Unit tests for Module F (Evaluation).

The metric and threshold logic is tested on synthetic scores, so correctness
does not depend on a trained checkpoint being present.
"""

import numpy as np

from src.evaluation.run import (
    BASELINE_PAPER_METRICS,
    build_report,
    compute_metrics,
    select_threshold,
)


def test_compute_metrics_on_perfect_predictions():
    scores = np.array([0.9, 0.8, 0.1, 0.2])
    labels = np.array([1, 1, 0, 0])

    metrics = compute_metrics(scores, labels, threshold=0.5)

    assert metrics["accuracy"] == 1.0
    assert metrics["precision"] == 1.0
    assert metrics["recall"] == 1.0
    assert metrics["auc"] == 1.0


def test_compute_metrics_on_inverted_predictions():
    """Scores that rank negatives above positives should yield AUC 0."""
    scores = np.array([0.1, 0.2, 0.9, 0.8])
    labels = np.array([1, 1, 0, 0])

    metrics = compute_metrics(scores, labels, threshold=0.5)

    assert metrics["auc"] == 0.0
    assert metrics["accuracy"] == 0.0


def test_compute_metrics_reports_all_expected_keys():
    scores = np.array([0.9, 0.4, 0.6, 0.1])
    labels = np.array([1, 0, 1, 0])

    metrics = compute_metrics(scores, labels, threshold=0.5)

    assert set(metrics) == {
        "auc",
        "aupr",
        "accuracy",
        "precision",
        "recall",
        "f1",
        "threshold",
        "num_examples",
    }
    assert metrics["num_examples"] == 4


def test_compute_metrics_respects_threshold():
    scores = np.array([0.6, 0.6, 0.4, 0.4])
    labels = np.array([1, 1, 0, 0])

    lenient = compute_metrics(scores, labels, threshold=0.5)
    strict = compute_metrics(scores, labels, threshold=0.7)

    assert lenient["recall"] == 1.0
    assert strict["recall"] == 0.0  # nothing clears the higher bar


def test_select_threshold_finds_separating_cutoff():
    scores = np.array([0.1, 0.2, 0.8, 0.9])
    labels = np.array([0, 0, 1, 1])

    threshold = select_threshold(scores, labels)

    predictions = (scores >= threshold).astype(int)
    assert np.array_equal(predictions, labels)


def test_select_threshold_returns_value_in_search_range():
    rng = np.random.default_rng(0)
    scores = rng.random(200)
    labels = (rng.random(200) > 0.5).astype(int)

    threshold = select_threshold(scores, labels)

    assert 0.05 <= threshold <= 0.95


class _StubGraph:
    """Minimal stand-in for the PyG graph used by build_report."""

    def __init__(self):
        self.num_nodes = 1704
        self.x = np.zeros((1704, 12))
        self.edge_index = np.zeros((2, 228406))
        self.test_pos_edges = np.zeros((2, 38116))
        self.test_neg_edges = np.zeros((2, 38116))


def test_build_report_compares_against_every_baseline_metric():
    results = {
        "threshold_selected_on": "validation",
        "validation": compute_metrics(np.array([0.9, 0.1]), np.array([1, 0]), 0.5),
        "test": compute_metrics(np.array([0.9, 0.1]), np.array([1, 0]), 0.5),
    }

    report = build_report(results, _StubGraph())

    assert set(report["baseline_comparison"]) == set(BASELINE_PAPER_METRICS)
    for name, values in report["baseline_comparison"].items():
        assert values["baseline_paper"] == BASELINE_PAPER_METRICS[name]
        expected = round(values["ours"] - values["baseline_paper"], 4)
        assert values["difference"] == expected


def test_build_report_records_dataset_shape():
    results = {
        "threshold_selected_on": "validation",
        "validation": compute_metrics(np.array([0.9, 0.1]), np.array([1, 0]), 0.5),
        "test": compute_metrics(np.array([0.9, 0.1]), np.array([1, 0]), 0.5),
    }

    report = build_report(results, _StubGraph())

    assert report["dataset"]["drugs"] == 1704
    assert report["dataset"]["node_features"] == 12
    assert report["comparison_caveat"]  # the caveat must not be silently dropped
