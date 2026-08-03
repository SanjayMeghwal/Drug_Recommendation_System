"""Entry point for Module F (Evaluation).

Scores the trained model on the held-out test edges and writes a report to
results/, including a comparison against the baseline paper's published
numbers.

Methodology note: metrics like accuracy and F1 need a decision threshold.
That threshold is chosen on the VALIDATION split and then applied unchanged
to the test split. Choosing it on test would tune against the very data used
to report performance and inflate every figure.
"""

import json

import numpy as np
import torch
from sklearn.metrics import (
    accuracy_score,
    average_precision_score,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)
from torch_geometric.data import Data

from src.config import load_config, resolve_path
from src.models.gnn import DDIPredictor, build_model
from src.models.train import CHECKPOINT_NAME, get_split_edges, load_graph

# Reported in the baseline paper: "Graph Neural Network-Based Framework for
# Predicting Drug-To-Drug and Drug-To-Food Interactions in Pharmacovigilance"
# (IEEE, 2025). Recall is stated for the paper's own dataset and pipeline, so
# these are a reference point rather than a like-for-like target.
BASELINE_PAPER_METRICS = {
    "accuracy": 0.923,
    "precision": 0.891,
    "recall": 0.914,
}

REPORT_NAME = "evaluation_report.json"


def load_trained_model(in_channels: int) -> DDIPredictor:
    """Rebuild the trained model from its checkpoint.

    The checkpoint carries its own model_config, so evaluation reflects the
    architecture actually trained rather than whatever config.yaml holds now.
    """
    config = load_config()
    checkpoint_path = resolve_path(config["paths"]["trained_models_dir"]) / CHECKPOINT_NAME
    if not checkpoint_path.exists():
        raise FileNotFoundError(
            f"No checkpoint at {checkpoint_path}. Run `python -m src.models.train` first."
        )

    checkpoint = torch.load(checkpoint_path, weights_only=False)
    model = build_model(
        in_channels=checkpoint.get("in_channels", in_channels),
        model_config=checkpoint["model_config"],
    )
    model.load_state_dict(checkpoint["state_dict"])
    model.eval()
    return model


@torch.no_grad()
def score_split(model: DDIPredictor, data: Data, split: str) -> tuple[np.ndarray, np.ndarray]:
    """Return (predicted probabilities, true labels) for one split."""
    edge_pairs, labels = get_split_edges(data, split)
    logits = model(data.x, data.edge_index, edge_pairs)
    return torch.sigmoid(logits).cpu().numpy(), labels.cpu().numpy()


def select_threshold(scores: np.ndarray, labels: np.ndarray) -> float:
    """Pick the probability cut-off that maximises F1 on the given split.

    Called with validation data only; the resulting threshold is then frozen
    for test-set reporting.
    """
    candidates = np.linspace(0.05, 0.95, 91)
    best_threshold, best_f1 = 0.5, -1.0

    for threshold in candidates:
        f1 = f1_score(labels, (scores >= threshold).astype(int), zero_division=0)
        if f1 > best_f1:
            best_threshold, best_f1 = float(threshold), float(f1)

    return best_threshold


def compute_metrics(scores: np.ndarray, labels: np.ndarray, threshold: float) -> dict[str, float]:
    """Compute threshold-free and threshold-dependent metrics together."""
    predictions = (scores >= threshold).astype(int)

    return {
        "auc": float(roc_auc_score(labels, scores)),
        "aupr": float(average_precision_score(labels, scores)),
        "accuracy": float(accuracy_score(labels, predictions)),
        "precision": float(precision_score(labels, predictions, zero_division=0)),
        "recall": float(recall_score(labels, predictions, zero_division=0)),
        "f1": float(f1_score(labels, predictions, zero_division=0)),
        "threshold": float(threshold),
        "num_examples": len(labels),
    }


def evaluate_model(model: DDIPredictor, data: Data) -> dict:
    """Tune the threshold on validation, then report validation and test."""
    val_scores, val_labels = score_split(model, data, "val")
    threshold = select_threshold(val_scores, val_labels)

    test_scores, test_labels = score_split(model, data, "test")

    return {
        "threshold_selected_on": "validation",
        "validation": compute_metrics(val_scores, val_labels, threshold),
        "test": compute_metrics(test_scores, test_labels, threshold),
    }


def build_report(results: dict, data: Data) -> dict:
    """Assemble the saved report, including the baseline comparison."""
    test_metrics = results["test"]

    comparison = {
        name: {
            "ours": round(test_metrics[name], 4),
            "baseline_paper": paper_value,
            "difference": round(test_metrics[name] - paper_value, 4),
        }
        for name, paper_value in BASELINE_PAPER_METRICS.items()
    }

    return {
        "dataset": {
            "drugs": int(data.num_nodes),
            "node_features": int(data.x.shape[1]),
            "message_passing_edges": int(data.edge_index.shape[1]),
            "test_positive_edges": int(data.test_pos_edges.shape[1]),
            "test_negative_edges": int(data.test_neg_edges.shape[1]),
        },
        "results": results,
        "baseline_comparison": comparison,
        "comparison_caveat": (
            "The baseline paper reports on its own DrugBank 5.1.13 + FAERS pipeline, "
            "including drug-food interactions, so these figures are a reference point "
            "rather than a like-for-like reproduction. Our split also removes 231 drug "
            "pairs that appeared in more than one split under different interaction "
            "types; leaving them in would raise our scores but train on test data."
        ),
    }


def print_report(report: dict) -> None:
    test = report["results"]["test"]
    print("\nTest set results")
    print(f"  examples:  {test['num_examples']} (threshold {test['threshold']:.2f})")
    for name in ("auc", "aupr", "accuracy", "precision", "recall", "f1"):
        print(f"  {name:10s} {test[name]:.4f}")

    print("\nAgainst the baseline paper")
    print(f"  {'metric':12s} {'ours':>8s} {'paper':>8s} {'diff':>8s}")
    for name, values in report["baseline_comparison"].items():
        print(
            f"  {name:12s} {values['ours']:>8.4f} "
            f"{values['baseline_paper']:>8.4f} {values['difference']:>+8.4f}"
        )


def run() -> None:
    config = load_config()

    print("Loading graph and trained model...")
    data = load_graph()
    model = load_trained_model(in_channels=data.x.shape[1])

    print("Evaluating...")
    results = evaluate_model(model, data)
    report = build_report(results, data)
    print_report(report)

    results_dir = resolve_path(config["paths"]["results_dir"])
    results_dir.mkdir(parents=True, exist_ok=True)
    report_path = results_dir / REPORT_NAME

    with open(report_path, "w", encoding="utf-8") as handle:
        json.dump(report, handle, indent=2)

    print(f"\nSaved report to {report_path}")


if __name__ == "__main__":
    run()
