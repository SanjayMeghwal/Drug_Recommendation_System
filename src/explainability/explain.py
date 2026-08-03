"""Explanation interface (Module E).

Explains why the model scored a drug pair as it did, in terms of the
interaction network the model actually learned from.

Why structural rather than SHAP over molecular descriptors: SHAP was tried
first, following the baseline paper, and produced attributions of roughly
0.005 on predictions ranging 0.5-0.96 — far too small to explain anything.
The cause is neighbourhood dilution. The drug graph has an average degree of
134, so after two rounds of message passing a drug's own twelve descriptors
contribute on the order of 1/135 of its embedding. Perturbing one pair's
features therefore barely moves that pair's prediction, even though the
descriptors matter in aggregate (zeroing them for all 1,704 drugs costs
about 10 AUC points).

What the model does respond to is graph structure: the number of interaction
partners two drugs share correlates with its output at Spearman 0.78. That
is what this module reports. See docs/explainability.md for the full
evidence, including the global feature-importance analysis.
"""

from dataclasses import dataclass, field
from functools import lru_cache

from src.models.predict import get_service

# Shared-partner bands used to give a prediction context. Boundaries are
# inclusive lower / exclusive upper, with None meaning unbounded.
SHARED_PARTNER_BANDS = [
    (0, 1, "no"),
    (1, 26, "few"),
    (26, 76, "a moderate number of"),
    (76, 151, "many"),
    (151, None, "very many"),
]

MAX_LISTED_PARTNERS = 5


@dataclass
class Explanation:
    """A structural account of one drug-pair prediction."""

    drug_a: str
    drug_b: str
    probability: float
    shared_partners: int
    shared_partner_names: list[str]
    drug_a_interactions: int
    drug_b_interactions: int
    observed_interaction_rate: float
    summary: str
    evidence: dict = field(default_factory=dict)


class InteractionExplainer:
    """Explains predictions using the interaction network the model saw.

    Neighbourhoods are built from `edge_index`, which Module C populates with
    training positives only. Explanations therefore cite evidence the model
    was actually trained on, never a held-out interaction it never saw.
    """

    def __init__(self) -> None:
        self.service = get_service()
        graph = self.service.graph

        self._neighbors: list[set[int]] = [set() for _ in range(graph.num_nodes)]
        sources = graph.edge_index[0].tolist()
        targets = graph.edge_index[1].tolist()
        for source, target in zip(sources, targets):
            self._neighbors[source].add(target)

        self._band_rates = self._measure_band_rates()

    def _measure_band_rates(self) -> dict[str, float]:
        """Measure, per shared-partner band, how often pairs are known to
        interact — so an explanation can say "pairs like this interact X% of
        the time" from observed data rather than a guessed figure.
        """
        import random

        graph = self.service.graph
        num_nodes = graph.num_nodes
        known = {
            (min(s, t), max(s, t))
            for s, t in zip(graph.edge_index[0].tolist(), graph.edge_index[1].tolist())
        }

        rng = random.Random(42)
        counts: dict[str, list[int]] = {label: [] for *_, label in SHARED_PARTNER_BANDS}

        for _ in range(20000):
            a, b = rng.randrange(num_nodes), rng.randrange(num_nodes)
            if a == b:
                continue
            shared = len(self._neighbors[a] & self._neighbors[b])
            label = self._band_label(shared)
            counts[label].append(1 if (min(a, b), max(a, b)) in known else 0)

        return {
            label: (sum(values) / len(values)) if values else 0.0
            for label, values in counts.items()
        }

    def _named_partners(self, ranked_indices: list[int]) -> list[str]:
        """List the top shared partners, skipping drugs whose name is a
        placeholder — 446 of the 1,704 drugs have no name in the source data,
        and "Unknown (DB01232)" tells a reader nothing.
        """
        service = self.service
        names = []

        for index in ranked_indices:
            name = service.name_for(service.graph.drug_ids[index])
            if name.startswith("Unknown ("):
                continue
            names.append(name)
            if len(names) == MAX_LISTED_PARTNERS:
                break

        return names

    @staticmethod
    def _band_label(shared: int) -> str:
        for lower, upper, label in SHARED_PARTNER_BANDS:
            if shared >= lower and (upper is None or shared < upper):
                return label
        return SHARED_PARTNER_BANDS[-1][2]

    def explain_interaction(self, drug_a: str, drug_b: str) -> Explanation:
        """Explain the predicted interaction between two drugs."""
        service = self.service
        id_a, id_b = service.resolve(drug_a), service.resolve(drug_b)
        index_a = service.index_of_drug_id[id_a]
        index_b = service.index_of_drug_id[id_b]

        probability = service.predict(id_a, id_b)

        shared_indices = self._neighbors[index_a] & self._neighbors[index_b]
        # Most-connected shared partners first: these carry the most weight
        # in the aggregation the model performs.
        ranked = sorted(shared_indices, key=lambda index: -len(self._neighbors[index]))
        shared_names = self._named_partners(ranked)

        band = self._band_label(len(shared_indices))
        rate = self._band_rates.get(band, 0.0)

        name_a = service.name_for(id_a)
        name_b = service.name_for(id_b)

        return Explanation(
            drug_a=name_a,
            drug_b=name_b,
            probability=probability,
            shared_partners=len(shared_indices),
            shared_partner_names=shared_names,
            drug_a_interactions=len(self._neighbors[index_a]),
            drug_b_interactions=len(self._neighbors[index_b]),
            observed_interaction_rate=rate,
            summary=self._build_summary(
                name_a,
                name_b,
                probability,
                len(shared_indices),
                band,
                rate,
                shared_names,
                len(self._neighbors[index_a]),
                len(self._neighbors[index_b]),
            ),
            evidence={
                "shared_partner_band": band,
                "top_shared_partners": shared_names,
                "drug_a_id": id_a,
                "drug_b_id": id_b,
            },
        )

    @staticmethod
    def _build_summary(
        name_a: str,
        name_b: str,
        probability: float,
        shared: int,
        band: str,
        rate: float,
        shared_names: list[str],
        degree_a: int,
        degree_b: int,
    ) -> str:
        verdict = (
            "likely to interact"
            if probability >= 0.5
            else "unlikely to interact based on the known interaction network"
        )

        parts = [
            f"{name_a} and {name_b} are {verdict} (predicted probability {probability:.2f}).",
            (
                f"They share {band} known interaction partners ({shared}); "
                f"in the training data, drug pairs sharing this many partners "
                f"interact {rate:.0%} of the time."
            ),
            f"{name_a} has {degree_a} known interactions and {name_b} has {degree_b}.",
        ]

        if shared_names:
            listed = ", ".join(shared_names)
            parts.append(f"Shared partners include {listed}.")

        return " ".join(parts)

    def explain_against_medications(
        self, candidate: str, current_medications: list[str]
    ) -> list[Explanation]:
        """Explain a candidate drug against each of a patient's current
        medications — the form Module G and the API need.
        """
        return [
            self.explain_interaction(candidate, medication) for medication in current_medications
        ]


@lru_cache(maxsize=1)
def get_explainer() -> InteractionExplainer:
    """Return the shared explainer, building its neighbourhood index once."""
    return InteractionExplainer()


def explain_interaction(drug_a: str, drug_b: str) -> Explanation:
    """Explain the predicted interaction between two drugs."""
    return get_explainer().explain_interaction(drug_a, drug_b)


def explain(drug_a: str, drug_b: str) -> str:
    """Return just the human-readable summary for a drug pair."""
    return explain_interaction(drug_a, drug_b).summary
