"""DDI prediction interface (Module D).

Wraps the trained GNN behind a simple `predict(drug_a, drug_b) -> probability`
call for the rest of the system.

The model, graph, and drug table are loaded once and cached, because scoring
a drug pair requires a forward pass over the whole graph — reloading per call
would make the recommendation module (which scores many candidate pairs)
unusably slow.
"""

from functools import lru_cache
from pathlib import Path

import pandas as pd
import torch

from src.config import load_config, resolve_path
from src.models.gnn import DDIPredictor, build_model
from src.models.train import CHECKPOINT_NAME


class UnknownDrugError(KeyError):
    """Raised when a drug identifier is not present in the trained graph."""


class DDIPredictionService:
    """Scores drug pairs with the trained GNN.

    Drugs may be referenced either by DrugBank ID (e.g. "DB00682") or by
    exact name (e.g. "Warfarin"). Name matching is case-insensitive but not
    fuzzy — unrecognised identifiers raise UnknownDrugError so callers can
    report the problem instead of silently scoring the wrong drug.
    """

    def __init__(
        self,
        checkpoint_path: Path | None = None,
        graph_path: Path | None = None,
        drugs_path: Path | None = None,
    ) -> None:
        config = load_config()

        checkpoint_path = checkpoint_path or (
            resolve_path(config["paths"]["trained_models_dir"]) / CHECKPOINT_NAME
        )
        graph_path = graph_path or (resolve_path(config["paths"]["graph_dir"]) / "ddi_graph.pt")
        drugs_path = drugs_path or (
            resolve_path(config["paths"]["processed_data_dir"]) / "drugs.csv"
        )

        for path, hint in (
            (graph_path, "python -m src.graph_construction.run"),
            (checkpoint_path, "python -m src.models.train"),
            (drugs_path, "python -m src.preprocessing.run"),
        ):
            if not path.exists():
                raise FileNotFoundError(f"Required file missing: {path}. Run `{hint}` first.")

        self.graph = torch.load(graph_path, weights_only=False)

        checkpoint = torch.load(checkpoint_path, weights_only=False)
        self.model: DDIPredictor = build_model(
            in_channels=checkpoint["in_channels"], model_config=checkpoint["model_config"]
        )
        self.model.load_state_dict(checkpoint["state_dict"])
        self.model.eval()

        self.index_of_drug_id = {
            drug_id: index for index, drug_id in enumerate(self.graph.drug_ids)
        }

        drugs = pd.read_csv(drugs_path)
        self.drug_id_of_name = {
            str(name).strip().lower(): drug_id
            for name, drug_id in zip(drugs["name"], drugs["drug_id"])
            if not str(name).startswith("Unknown (")
        }
        self.name_of_drug_id = dict(zip(drugs["drug_id"], drugs["name"]))

        # Embeddings depend only on the fixed graph and trained weights, so
        # they are computed once rather than per prediction.
        with torch.no_grad():
            self._embeddings = self.model.encode(self.graph.x, self.graph.edge_index)

    def resolve(self, identifier: str) -> str:
        """Map a DrugBank ID or exact drug name to its DrugBank ID."""
        candidate = str(identifier).strip()

        if candidate in self.index_of_drug_id:
            return candidate

        resolved = self.drug_id_of_name.get(candidate.lower())
        if resolved is not None:
            return resolved

        raise UnknownDrugError(
            f"{identifier!r} is not a known drug. Provide a DrugBank ID "
            f"(e.g. 'DB00682') or an exact drug name (e.g. 'Warfarin')."
        )

    def is_known_drug(self, identifier: str) -> bool:
        try:
            self.resolve(identifier)
        except UnknownDrugError:
            return False
        return True

    def name_for(self, drug_id: str) -> str:
        return self.name_of_drug_id.get(drug_id, drug_id)

    def predict(self, drug_a: str, drug_b: str) -> float:
        """Return the probability that two drugs interact, in [0, 1]."""
        return self.predict_batch([(drug_a, drug_b)])[0]

    def predict_batch(self, pairs: list[tuple[str, str]]) -> list[float]:
        """Score many drug pairs in a single forward pass.

        Module G scores every candidate drug against every current
        medication, so batching matters for interactive response times.
        """
        if not pairs:
            return []

        sources = []
        targets = []
        for drug_a, drug_b in pairs:
            sources.append(self.index_of_drug_id[self.resolve(drug_a)])
            targets.append(self.index_of_drug_id[self.resolve(drug_b)])

        edge_pairs = torch.tensor([sources, targets], dtype=torch.long)

        with torch.no_grad():
            logits = self.model.decode(self._embeddings, edge_pairs)
            probabilities = torch.sigmoid(logits)

        return [float(value) for value in probabilities]


@lru_cache(maxsize=1)
def get_service() -> DDIPredictionService:
    """Return the shared prediction service, loading it on first use."""
    return DDIPredictionService()


def predict(drug_a: str, drug_b: str) -> float:
    """Return the probability that drug_a and drug_b interact.

    Accepts DrugBank IDs or exact drug names. Raises UnknownDrugError for
    unrecognised drugs.
    """
    return get_service().predict(drug_a, drug_b)
