"""Unit tests for Module D (GNN architecture and training).

Uses a small synthetic graph so the architecture and training loop are
verified without depending on the full 190k-edge dataset or a trained
checkpoint.
"""

import pytest
import torch
from torch import nn
from torch_geometric.data import Data

from src.models.gnn import DDIPredictor, build_model
from src.models.train import evaluate, get_split_edges, set_seed, train, train_one_epoch

NUM_NODES = 12
NUM_FEATURES = 12


@pytest.fixture
def synthetic_graph():
    """A small graph with a learnable structure: nodes are split into two
    clusters that interact within, but not across, cluster boundaries.
    """
    torch.manual_seed(0)

    cluster_a = list(range(6))
    cluster_b = list(range(6, 12))

    positives = [(i, j) for i in cluster_a for j in cluster_a if i < j]
    negatives = [(i, j) for i in cluster_a for j in cluster_b]

    def to_tensor(pairs):
        return torch.tensor(pairs, dtype=torch.long).T

    train_pos = to_tensor(positives[:10])
    train_neg = to_tensor(negatives[:10])
    val_pos = to_tensor(positives[10:13])
    val_neg = to_tensor(negatives[10:13])
    test_pos = to_tensor(positives[13:15])
    test_neg = to_tensor(negatives[13:15])

    data = Data(x=torch.randn(NUM_NODES, NUM_FEATURES), num_nodes=NUM_NODES)
    data.edge_index = torch.cat([train_pos, train_pos.flip(0)], dim=1)
    data.train_pos_edges, data.train_neg_edges = train_pos, train_neg
    data.val_pos_edges, data.val_neg_edges = val_pos, val_neg
    data.test_pos_edges, data.test_neg_edges = test_pos, test_neg
    return data


@pytest.fixture
def training_config():
    return {
        "model": {
            "conv_type": "gcn",
            "hidden_channels": 16,
            "embedding_dim": 8,
            "dropout": 0.0,
            "decoder": "dot",
        },
        "training": {
            "epochs": 30,
            "learning_rate": 0.01,
            "weight_decay": 0.0005,
            "patience": 30,
            "seed": 42,
        },
    }


def test_model_rejects_unknown_conv_type():
    with pytest.raises(ValueError, match="conv_type"):
        DDIPredictor(in_channels=NUM_FEATURES, conv_type="transformer")


def test_model_rejects_unknown_decoder():
    with pytest.raises(ValueError, match="decoder"):
        DDIPredictor(in_channels=NUM_FEATURES, decoder="cosine")


@pytest.mark.parametrize("conv_type", ["gcn", "gat"])
def test_encode_produces_one_embedding_per_node(synthetic_graph, conv_type):
    model = DDIPredictor(
        in_channels=NUM_FEATURES, hidden_channels=16, embedding_dim=8, conv_type=conv_type
    )
    embeddings = model.encode(synthetic_graph.x, synthetic_graph.edge_index)
    assert embeddings.shape == (NUM_NODES, 8)


@pytest.mark.parametrize("decoder", ["dot", "mlp"])
def test_forward_returns_one_logit_per_pair(synthetic_graph, decoder):
    model = DDIPredictor(in_channels=NUM_FEATURES, embedding_dim=8, decoder=decoder)
    pairs = synthetic_graph.train_pos_edges
    logits = model(synthetic_graph.x, synthetic_graph.edge_index, pairs)
    assert logits.shape == (pairs.shape[1],)


def test_forward_accepts_modified_features(synthetic_graph):
    """Module E perturbs node features and re-runs the forward pass, so the
    model must score against whatever feature matrix it is handed.
    """
    model = DDIPredictor(in_channels=NUM_FEATURES, embedding_dim=8, dropout=0.0)
    model.eval()
    pairs = synthetic_graph.train_pos_edges

    original = model(synthetic_graph.x, synthetic_graph.edge_index, pairs)
    perturbed_x = synthetic_graph.x.clone()
    perturbed_x[pairs[0, 0]] += 5.0
    perturbed = model(perturbed_x, synthetic_graph.edge_index, pairs)

    assert not torch.allclose(original, perturbed)


def test_build_model_reads_config(training_config):
    model = build_model(in_channels=NUM_FEATURES, model_config=training_config["model"])
    assert model.conv_type == "gcn"
    assert model.decoder_type == "dot"
    assert model.pair_scorer is None  # dot decoder has no learned scorer


def test_get_split_edges_labels_positives_then_negatives(synthetic_graph):
    edges, labels = get_split_edges(synthetic_graph, "train")

    num_pos = synthetic_graph.train_pos_edges.shape[1]
    num_neg = synthetic_graph.train_neg_edges.shape[1]

    assert edges.shape == (2, num_pos + num_neg)
    assert labels.shape == (num_pos + num_neg,)
    assert torch.all(labels[:num_pos] == 1.0)
    assert torch.all(labels[num_pos:] == 0.0)


def test_training_reduces_loss(synthetic_graph, training_config):
    """The core Day 5 check: the training loop actually learns."""
    set_seed(42)
    model = build_model(NUM_FEATURES, training_config["model"])
    optimizer = torch.optim.Adam(model.parameters(), lr=0.01)
    criterion = nn.BCEWithLogitsLoss()
    edges, labels = get_split_edges(synthetic_graph, "train")

    first_loss = train_one_epoch(model, synthetic_graph, optimizer, criterion, edges, labels)
    for _ in range(29):
        last_loss = train_one_epoch(model, synthetic_graph, optimizer, criterion, edges, labels)

    assert last_loss < first_loss


def test_evaluate_returns_metrics_in_valid_range(synthetic_graph, training_config):
    model = build_model(NUM_FEATURES, training_config["model"])
    edges, labels = get_split_edges(synthetic_graph, "val")

    metrics = evaluate(model, synthetic_graph, edges, labels)

    assert set(metrics) == {"auc", "ap"}
    assert 0.0 <= metrics["auc"] <= 1.0
    assert 0.0 <= metrics["ap"] <= 1.0


def test_evaluate_does_not_change_weights(synthetic_graph, training_config):
    model = build_model(NUM_FEATURES, training_config["model"])
    before = [p.detach().clone() for p in model.parameters()]

    edges, labels = get_split_edges(synthetic_graph, "val")
    evaluate(model, synthetic_graph, edges, labels)

    for original, current in zip(before, model.parameters()):
        assert torch.equal(original, current)


def test_train_returns_best_checkpoint_and_history(synthetic_graph, training_config):
    model, history = train(synthetic_graph, training_config, verbose=False)

    assert isinstance(model, nn.Module)
    assert 0.0 <= history["best_val_auc"] <= 1.0
    assert history["best_epoch"] >= 1
    assert len(history["history"]) == history["epochs_run"]


def test_training_is_reproducible(synthetic_graph, training_config):
    _, first = train(synthetic_graph, training_config, verbose=False)
    _, second = train(synthetic_graph, training_config, verbose=False)
    assert first["best_val_auc"] == second["best_val_auc"]


def test_early_stopping_triggers(synthetic_graph, training_config):
    training_config["training"]["epochs"] = 500
    training_config["training"]["patience"] = 3

    _, history = train(synthetic_graph, training_config, verbose=False)

    assert history["epochs_run"] < 500


def test_model_survives_save_and_reload(synthetic_graph, training_config, tmp_path):
    """Day 5 requirement: the trained model must persist to disk and give
    identical predictions after being reloaded.
    """
    model, _ = train(synthetic_graph, training_config, verbose=False)
    model.eval()
    pairs = synthetic_graph.test_pos_edges

    with torch.no_grad():
        before = model(synthetic_graph.x, synthetic_graph.edge_index, pairs)

    checkpoint_path = tmp_path / "checkpoint.pt"
    torch.save(
        {"state_dict": model.state_dict(), "model_config": training_config["model"]},
        checkpoint_path,
    )

    checkpoint = torch.load(checkpoint_path, weights_only=False)
    reloaded = build_model(NUM_FEATURES, checkpoint["model_config"])
    reloaded.load_state_dict(checkpoint["state_dict"])
    reloaded.eval()

    with torch.no_grad():
        after = reloaded(synthetic_graph.x, synthetic_graph.edge_index, pairs)

    assert torch.allclose(before, after)
