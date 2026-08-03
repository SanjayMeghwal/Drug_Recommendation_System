"""Unit tests for Module C (Graph Construction).

Uses small synthetic tables so the graph logic — especially the leakage
controls — is verified independently of the full 190k-pair dataset.
"""

import numpy as np
import pandas as pd
import pytest

from src.graph_construction.run import (
    DESCRIPTORS,
    assign_pairs_to_splits,
    build_graph,
    compute_node_features,
    sample_negative_edges,
    standardize_features,
)


@pytest.fixture
def synthetic_drugs():
    return pd.DataFrame(
        {
            "drug_id": ["DB001", "DB002", "DB003", "DB004", "DB005", "DB006"],
            "smiles": [
                "CCO",
                "CCC",
                "c1ccccc1",
                "CC(=O)O",
                "CCN",
                "CCCl",
            ],
            "name": ["A", "B", "C", "D", "E", "F"],
            "drug_groups": ["approved"] * 6,
        }
    )


@pytest.fixture
def synthetic_pairs():
    return pd.DataFrame(
        {
            "d1": ["DB001", "DB002", "DB003", "DB001"],
            "d2": ["DB002", "DB003", "DB004", "DB005"],
            "type": [5, 6, 7, 0],  # last row is an explicit non-interaction
            "split": ["training", "validation", "testing", "training"],
        }
    )


def test_compute_node_features_shape_matches_descriptor_count():
    features = compute_node_features(["CCO", "c1ccccc1"])
    assert features.shape == (2, len(DESCRIPTORS))


def test_compute_node_features_handles_unparseable_smiles():
    features = compute_node_features(["CCO", "not_a_valid_smiles"])
    assert features.shape == (2, len(DESCRIPTORS))
    assert np.all(features[1] == 0.0)  # invalid molecule becomes a zero vector


def test_compute_node_features_has_no_nan_or_inf():
    features = compute_node_features(["CCO", "CCC", "c1ccccc1"])
    assert not np.isnan(features).any()
    assert not np.isinf(features).any()


def test_standardize_features_produces_zero_mean_unit_variance():
    raw = np.array([[1.0, 100.0], [2.0, 200.0], [3.0, 300.0]])
    standardized = standardize_features(raw)
    assert np.allclose(standardized.mean(axis=0), 0.0)
    assert np.allclose(standardized.std(axis=0), 1.0)


def test_standardize_features_handles_constant_column():
    raw = np.array([[1.0, 5.0], [2.0, 5.0], [3.0, 5.0]])
    standardized = standardize_features(raw)
    assert np.all(standardized[:, 1] == 0.0)  # constant column, not NaN
    assert not np.isnan(standardized).any()


def test_sample_negative_edges_avoids_forbidden_pairs():
    forbidden = {(0, 1), (1, 2), (0, 2)}
    rng = np.random.default_rng(0)
    negatives = sample_negative_edges(5, num_nodes=10, forbidden_pairs=forbidden, rng=rng)

    assert negatives.shape == (2, 5)
    for source, target in zip(negatives[0], negatives[1]):
        assert (min(source, target), max(source, target)) not in forbidden


def test_sample_negative_edges_produces_no_self_loops():
    rng = np.random.default_rng(0)
    negatives = sample_negative_edges(20, num_nodes=8, forbidden_pairs=set(), rng=rng)
    assert not np.any(negatives[0] == negatives[1])


def test_sample_negative_edges_is_reproducible_with_same_seed():
    first = sample_negative_edges(10, 20, set(), np.random.default_rng(42))
    second = sample_negative_edges(10, 20, set(), np.random.default_rng(42))
    assert np.array_equal(first, second)


def test_build_graph_node_count_matches_drugs_table(synthetic_drugs, synthetic_pairs):
    data = build_graph(synthetic_drugs, synthetic_pairs)
    assert data.num_nodes == len(synthetic_drugs)
    assert data.x.shape == (len(synthetic_drugs), len(DESCRIPTORS))


def test_build_graph_excludes_non_interactions_from_positives(synthetic_drugs, synthetic_pairs):
    """The type == 0 row is an explicit non-interaction and must not become
    a positive supervision edge."""
    data = build_graph(synthetic_drugs, synthetic_pairs)
    total_positives = (
        data.train_pos_edges.shape[1] + data.val_pos_edges.shape[1] + data.test_pos_edges.shape[1]
    )
    assert total_positives == 3  # 4 rows, minus the type == 0 row


def test_build_graph_message_passing_uses_only_training_edges(synthetic_drugs, synthetic_pairs):
    data = build_graph(synthetic_drugs, synthetic_pairs)
    # 1 training positive, duplicated for both directions.
    assert data.train_pos_edges.shape[1] == 1
    assert data.edge_index.shape[1] == 2


def test_build_graph_has_no_validation_or_test_leakage(synthetic_drugs, synthetic_pairs):
    """The critical check: no validation or test edge may appear in the
    message-passing graph, or the model would see the links it is scored on.
    """
    data = build_graph(synthetic_drugs, synthetic_pairs)

    message_passing_pairs = {
        (min(s, t), max(s, t))
        for s, t in zip(data.edge_index[0].tolist(), data.edge_index[1].tolist())
    }

    for prefix in ("val", "test"):
        held_out = getattr(data, f"{prefix}_pos_edges")
        for source, target in zip(held_out[0].tolist(), held_out[1].tolist()):
            assert (min(source, target), max(source, target)) not in message_passing_pairs


def test_build_graph_negatives_match_positive_counts(synthetic_drugs, synthetic_pairs):
    data = build_graph(synthetic_drugs, synthetic_pairs)
    for prefix in ("train", "val", "test"):
        pos = getattr(data, f"{prefix}_pos_edges").shape[1]
        neg = getattr(data, f"{prefix}_neg_edges").shape[1]
        assert pos == neg


def test_build_graph_negatives_are_never_known_positives(synthetic_drugs, synthetic_pairs):
    """A held-out test interaction must never be sampled as a negative."""
    data = build_graph(synthetic_drugs, synthetic_pairs)

    known_positives = set()
    for prefix in ("train", "val", "test"):
        edges = getattr(data, f"{prefix}_pos_edges")
        for source, target in zip(edges[0].tolist(), edges[1].tolist()):
            known_positives.add((min(source, target), max(source, target)))

    for prefix in ("train", "val", "test"):
        edges = getattr(data, f"{prefix}_neg_edges")
        for source, target in zip(edges[0].tolist(), edges[1].tolist()):
            assert (min(source, target), max(source, target)) not in known_positives


def test_assign_pairs_to_splits_deduplicates_undirected_pairs():
    """(A, B) and (B, A) are the same undirected interaction."""
    positives = pd.DataFrame(
        {
            "d1": ["DB001", "DB002"],
            "d2": ["DB002", "DB001"],
            "type": [5, 9],
            "split": ["training", "training"],
        }
    )
    grouped = assign_pairs_to_splits(positives, {"DB001": 0, "DB002": 1})
    assert grouped["training"] == [(0, 1)]


def test_assign_pairs_to_splits_prefers_held_out_split():
    """REGRESSION: a pair listed under different interaction types can fall
    in different splits. It must resolve to the most-held-out split, or the
    same edge ends up both trained on and tested against.
    """
    positives = pd.DataFrame(
        {
            "d1": ["DB001", "DB001", "DB002"],
            "d2": ["DB002", "DB002", "DB003"],
            "type": [8, 5, 9],
            "split": ["training", "testing", "training"],
        }
    )
    grouped = assign_pairs_to_splits(positives, {"DB001": 0, "DB002": 1, "DB003": 2})

    assert grouped["testing"] == [(0, 1)]
    assert (0, 1) not in grouped["training"]
    assert grouped["training"] == [(1, 2)]


def test_assign_pairs_to_splits_drops_self_interactions():
    positives = pd.DataFrame({"d1": ["DB001"], "d2": ["DB001"], "type": [5], "split": ["training"]})
    grouped = assign_pairs_to_splits(positives, {"DB001": 0})
    assert grouped["training"] == []


def test_build_graph_no_leakage_when_pair_spans_multiple_splits(synthetic_drugs):
    """REGRESSION (end-to-end): the duplicated pair must not reach the
    message-passing graph while also being scored as a test edge.
    """
    pairs = pd.DataFrame(
        {
            "d1": ["DB001", "DB001", "DB003"],
            "d2": ["DB002", "DB002", "DB004"],
            "type": [8, 5, 9],
            "split": ["training", "testing", "training"],
        }
    )
    data = build_graph(synthetic_drugs, pairs)

    message_passing_pairs = {
        (min(s, t), max(s, t))
        for s, t in zip(data.edge_index[0].tolist(), data.edge_index[1].tolist())
    }
    for source, target in zip(data.test_pos_edges[0].tolist(), data.test_pos_edges[1].tolist()):
        assert (min(source, target), max(source, target)) not in message_passing_pairs


def test_build_graph_is_reproducible(synthetic_drugs, synthetic_pairs):
    first = build_graph(synthetic_drugs, synthetic_pairs, seed=7)
    second = build_graph(synthetic_drugs, synthetic_pairs, seed=7)
    assert first.train_neg_edges.equal(second.train_neg_edges)
    assert first.test_neg_edges.equal(second.test_neg_edges)
