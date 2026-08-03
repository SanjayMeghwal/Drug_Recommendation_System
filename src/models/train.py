"""Training entry point for Module D.

Trains the GNN on the graph produced by Module C and saves the best
checkpoint (by validation AUC) to artifacts/trained_models/.

Only the training edges are ever used for message passing — Module C
guarantees validation and test edges are absent from `edge_index` — so the
validation score reported here reflects genuinely unseen interactions.
"""

import random

import numpy as np
import torch
from sklearn.metrics import average_precision_score, roc_auc_score
from torch import nn
from torch_geometric.data import Data

from src.config import load_config, resolve_path
from src.models.gnn import build_model

CHECKPOINT_NAME = "ddi_gnn.pt"


def set_seed(seed: int) -> None:
    """Seed every source of randomness so training runs are reproducible."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def load_graph() -> Data:
    """Load the graph built by Module C."""
    config = load_config()
    graph_path = resolve_path(config["paths"]["graph_dir"]) / "ddi_graph.pt"
    if not graph_path.exists():
        raise FileNotFoundError(
            f"Graph not found at {graph_path}. Run src/graph_construction/run.py first."
        )
    return torch.load(graph_path, weights_only=False)


def get_split_edges(data: Data, split: str) -> tuple[torch.Tensor, torch.Tensor]:
    """Return (edge_pairs, labels) for a split, positives followed by negatives."""
    positives = getattr(data, f"{split}_pos_edges")
    negatives = getattr(data, f"{split}_neg_edges")

    edge_pairs = torch.cat([positives, negatives], dim=1)
    labels = torch.cat(
        [torch.ones(positives.shape[1]), torch.zeros(negatives.shape[1])],
        dim=0,
    )
    return edge_pairs, labels


def train_one_epoch(
    model: nn.Module,
    data: Data,
    optimizer: torch.optim.Optimizer,
    criterion: nn.Module,
    edge_pairs: torch.Tensor,
    labels: torch.Tensor,
) -> float:
    """Run a single full-batch training step and return the loss."""
    model.train()
    optimizer.zero_grad()

    logits = model(data.x, data.edge_index, edge_pairs)
    loss = criterion(logits, labels)

    loss.backward()
    optimizer.step()
    return float(loss.item())


@torch.no_grad()
def evaluate(
    model: nn.Module, data: Data, edge_pairs: torch.Tensor, labels: torch.Tensor
) -> dict[str, float]:
    """Compute ranking metrics for the given edges without updating weights."""
    model.eval()

    logits = model(data.x, data.edge_index, edge_pairs)
    scores = torch.sigmoid(logits).cpu().numpy()
    targets = labels.cpu().numpy()

    return {
        "auc": float(roc_auc_score(targets, scores)),
        "ap": float(average_precision_score(targets, scores)),
    }


def train(data: Data, config: dict, verbose: bool = True) -> tuple[nn.Module, dict]:
    """Train the model, keeping the checkpoint with the best validation AUC.

    Returns the trained model (restored to its best state) and a history
    dictionary describing the run.
    """
    training_config = config["training"]
    set_seed(training_config.get("seed", 42))

    model = build_model(in_channels=data.x.shape[1], model_config=config["model"])
    optimizer = torch.optim.Adam(
        model.parameters(),
        lr=training_config.get("learning_rate", 0.01),
        weight_decay=training_config.get("weight_decay", 5e-4),
    )
    criterion = nn.BCEWithLogitsLoss()

    train_edges, train_labels = get_split_edges(data, "train")
    val_edges, val_labels = get_split_edges(data, "val")

    best_val_auc = 0.0
    best_state = None
    best_epoch = 0
    epochs_without_improvement = 0
    patience = training_config.get("patience", 20)
    history: list[dict] = []

    for epoch in range(1, training_config.get("epochs", 200) + 1):
        loss = train_one_epoch(model, data, optimizer, criterion, train_edges, train_labels)
        val_metrics = evaluate(model, data, val_edges, val_labels)
        history.append({"epoch": epoch, "loss": loss, **val_metrics})

        if val_metrics["auc"] > best_val_auc:
            best_val_auc = val_metrics["auc"]
            best_state = {k: v.detach().clone() for k, v in model.state_dict().items()}
            best_epoch = epoch
            epochs_without_improvement = 0
        else:
            epochs_without_improvement += 1

        if verbose and (epoch == 1 or epoch % 10 == 0):
            print(
                f"  epoch {epoch:3d} | loss {loss:.4f} "
                f"| val AUC {val_metrics['auc']:.4f} | val AP {val_metrics['ap']:.4f}"
            )

        if epochs_without_improvement >= patience:
            if verbose:
                print(f"  early stopping at epoch {epoch} (no improvement for {patience} epochs)")
            break

    if best_state is not None:
        model.load_state_dict(best_state)

    return model, {
        "best_val_auc": best_val_auc,
        "best_epoch": best_epoch,
        "epochs_run": len(history),
        "history": history,
    }


def run() -> None:
    config = load_config()

    print("Loading graph...")
    data = load_graph()
    print(f"  {data.num_nodes} drugs, {data.x.shape[1]} features")
    print(f"  {data.edge_index.shape[1]} message-passing edges")

    print(
        f"Training {config['model']['conv_type'].upper()} encoder "
        f"with {config['model']['decoder']} decoder..."
    )
    model, history = train(data, config)

    print(f"Best validation AUC {history['best_val_auc']:.4f} at epoch {history['best_epoch']}")

    output_dir = resolve_path(config["paths"]["trained_models_dir"])
    output_dir.mkdir(parents=True, exist_ok=True)
    checkpoint_path = output_dir / CHECKPOINT_NAME

    torch.save(
        {
            "state_dict": model.state_dict(),
            "model_config": config["model"],
            "in_channels": data.x.shape[1],
            "best_val_auc": history["best_val_auc"],
            "best_epoch": history["best_epoch"],
        },
        checkpoint_path,
    )
    print(f"Saved checkpoint to {checkpoint_path}")


if __name__ == "__main__":
    run()
