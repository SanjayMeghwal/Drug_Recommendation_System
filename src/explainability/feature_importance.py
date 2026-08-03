"""Global feature importance for the trained model (Module E, report side).

Per-pair SHAP is uninformative for this architecture (see the module
docstring in explain.py). Importance is therefore measured globally, by
ablation: zero one descriptor across all 1,704 drugs and observe how much
test AUC falls. That answers "which molecular properties does the model rely
on overall?", which per-pair attribution could not.

This is analysis for the project report, not part of the request path.
"""

import json

import torch
from sklearn.metrics import roc_auc_score

from src.config import load_config, resolve_path
from src.evaluation.run import load_trained_model
from src.models.train import get_split_edges, load_graph

REPORT_NAME = "feature_importance.json"


def _auc_with_features(model, data, features: torch.Tensor, edges, labels) -> float:
    with torch.no_grad():
        logits = model(features, data.edge_index, edges)
    return float(roc_auc_score(labels.numpy(), torch.sigmoid(logits).numpy()))


def compute_feature_importance(model, data) -> dict:
    """Rank descriptors by the test AUC lost when each is removed."""
    edges, labels = get_split_edges(data, "test")
    baseline_auc = _auc_with_features(model, data, data.x, edges, labels)

    per_feature = []
    for column, name in enumerate(data.feature_names):
        ablated = data.x.clone()
        ablated[:, column] = 0.0
        auc = _auc_with_features(model, data, ablated, edges, labels)
        per_feature.append(
            {
                "feature": name,
                "auc_without": round(auc, 4),
                "auc_drop": round(baseline_auc - auc, 5),
            }
        )

    per_feature.sort(key=lambda item: -item["auc_drop"])

    # Removing every descriptor at once shows how much the model depends on
    # chemistry versus pure network structure.
    all_removed = _auc_with_features(model, data, torch.zeros_like(data.x), edges, labels)

    return {
        "baseline_auc": round(baseline_auc, 4),
        "auc_with_all_features_removed": round(all_removed, 4),
        "total_feature_contribution": round(baseline_auc - all_removed, 4),
        "per_feature": per_feature,
    }


def run() -> None:
    config = load_config()

    print("Loading graph and trained model...")
    data = load_graph()
    model = load_trained_model(data.x.shape[1])

    print("Measuring feature importance by ablation...")
    report = compute_feature_importance(model, data)

    print(f"\nBaseline test AUC: {report['baseline_auc']:.4f}")
    print(
        f"With all descriptors removed: {report['auc_with_all_features_removed']:.4f} "
        f"(chemistry contributes {report['total_feature_contribution']:.4f} AUC)"
    )
    print("\nPer-descriptor AUC drop when removed:")
    for item in report["per_feature"]:
        print(f"  {item['feature']:20s} {item['auc_drop']:+.5f}")

    results_dir = resolve_path(config["paths"]["results_dir"])
    results_dir.mkdir(parents=True, exist_ok=True)
    report_path = results_dir / REPORT_NAME

    with open(report_path, "w", encoding="utf-8") as handle:
        json.dump(report, handle, indent=2)

    print(f"\nSaved to {report_path}")


if __name__ == "__main__":
    run()
