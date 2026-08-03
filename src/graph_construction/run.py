"""Entry point for Module C (Graph Construction).

Builds the drug-interaction graph from data/processed/ and saves it to
data/graph/ as a PyTorch Geometric Data object.

Design notes:
  * Nodes are drugs; features are interpretable RDKit molecular descriptors
    (chosen so Module E's SHAP explanations are human-readable).
  * The task is binary link prediction: does an interaction exist between
    two drugs? The raw `type` column encodes 86 specific interaction
    classes, which we collapse to interacts (type != 0) / does not (type == 0).
  * LEAKAGE CONTROL: the message-passing graph (`edge_index`) contains ONLY
    training positive edges. Validation and test edges are held out entirely,
    otherwise the model would observe the very links it is asked to predict.
    Because a drug pair can appear several times in the source data under
    different interaction types — and those rows may fall in different
    splits — pairs are collapsed to unique undirected edges and assigned to
    exactly one split before any of this is built (see assign_pairs_to_splits).
"""

from pathlib import Path

import numpy as np
import pandas as pd
import torch
from rdkit import Chem, RDLogger
from rdkit.Chem import Descriptors
from torch_geometric.data import Data

RDLogger.DisableLog("rdApp.*")

PROCESSED_DIR = Path(__file__).resolve().parents[2] / "data" / "processed"
GRAPH_DIR = Path(__file__).resolve().parents[2] / "data" / "graph"

RANDOM_SEED = 42

# When one drug pair appears in several splits (the same pair can be listed
# under multiple interaction types), the most-held-out split wins. Keeping a
# pair out of training is what preserves an honest evaluation.
SPLIT_PRIORITY = {"testing": 0, "validation": 1, "training": 2}

# Interpretable molecular descriptors used as node features. Names are kept
# alongside the functions so Module E can report them in explanations.
DESCRIPTORS = [
    ("MolecularWeight", Descriptors.MolWt),
    ("LogP", Descriptors.MolLogP),
    ("TPSA", Descriptors.TPSA),
    ("NumHDonors", Descriptors.NumHDonors),
    ("NumHAcceptors", Descriptors.NumHAcceptors),
    ("NumRotatableBonds", Descriptors.NumRotatableBonds),
    ("NumAromaticRings", Descriptors.NumAromaticRings),
    ("HeavyAtomCount", Descriptors.HeavyAtomCount),
    ("FractionCSP3", Descriptors.FractionCSP3),
    ("RingCount", Descriptors.RingCount),
    ("NumHeteroatoms", Descriptors.NumHeteroatoms),
    ("MolarRefractivity", Descriptors.MolMR),
]

FEATURE_NAMES = [name for name, _ in DESCRIPTORS]


def compute_node_features(smiles_list: list[str]) -> np.ndarray:
    """Compute RDKit descriptors for each SMILES string.

    Unparseable SMILES yield a zero vector rather than dropping the drug,
    since the drug is still a valid node in the interaction graph.
    """
    features = []
    for smiles in smiles_list:
        molecule = Chem.MolFromSmiles(smiles)
        if molecule is None:
            features.append([0.0] * len(DESCRIPTORS))
            continue
        features.append([float(fn(molecule)) for _, fn in DESCRIPTORS])

    array = np.array(features, dtype=np.float64)
    return np.nan_to_num(array, nan=0.0, posinf=0.0, neginf=0.0)


def standardize_features(features: np.ndarray) -> np.ndarray:
    """Z-score each feature column. Descriptor scales differ by orders of
    magnitude (molecular weight in the hundreds vs. ring counts in single
    digits), which would otherwise destabilize GNN training.
    """
    mean = features.mean(axis=0)
    std = features.std(axis=0)
    std[std == 0] = 1.0  # constant columns become all-zero rather than NaN
    return (features - mean) / std


def sample_negative_edges(
    num_samples: int,
    num_nodes: int,
    forbidden_pairs: set[tuple[int, int]],
    rng: np.random.Generator,
) -> np.ndarray:
    """Randomly sample drug pairs that are NOT known to interact.

    Only 1,055 explicit non-interactions exist in the source data — far too
    few to balance ~190k positives — so negatives are drawn from unobserved
    pairs, the standard approach for link prediction.

    `forbidden_pairs` must contain every known positive pair across all
    splits (as ordered (min, max) tuples), so a held-out test interaction is
    never accidentally sampled as a negative.
    """
    sampled: set[tuple[int, int]] = set()

    while len(sampled) < num_samples:
        remaining = num_samples - len(sampled)
        sources = rng.integers(0, num_nodes, size=remaining * 2)
        targets = rng.integers(0, num_nodes, size=remaining * 2)

        for source, target in zip(sources, targets):
            if source == target:
                continue
            pair = (int(min(source, target)), int(max(source, target)))
            if pair in forbidden_pairs or pair in sampled:
                continue
            sampled.add(pair)
            if len(sampled) == num_samples:
                break

    return np.array(sorted(sampled), dtype=np.int64).T


def assign_pairs_to_splits(
    positives: pd.DataFrame, index_of: dict[str, int]
) -> dict[str, list[tuple[int, int]]]:
    """Collapse positive interactions to unique undirected node-index pairs,
    assigning each pair to exactly one split.

    The source data lists a drug pair once per interaction type, and those
    rows can land in different splits. Collapsing 86 interaction types into
    a binary "interacts" label therefore risks the same pair appearing in
    both training and test. Each pair is resolved to its highest-priority
    split (testing > validation > training) so held-out edges stay held out.
    """
    split_of_pair: dict[tuple[int, int], str] = {}

    for drug_a, drug_b, split in zip(positives["d1"], positives["d2"], positives["split"]):
        source, target = index_of[drug_a], index_of[drug_b]
        if source == target:
            continue  # a drug interacting with itself is not a meaningful edge
        pair = (min(source, target), max(source, target))

        current = split_of_pair.get(pair)
        if current is None or SPLIT_PRIORITY[split] < SPLIT_PRIORITY[current]:
            split_of_pair[pair] = split

    grouped: dict[str, list[tuple[int, int]]] = {
        "training": [],
        "validation": [],
        "testing": [],
    }
    for pair, split in split_of_pair.items():
        grouped[split].append(pair)

    return {split: sorted(pairs) for split, pairs in grouped.items()}


def build_graph(drugs_df: pd.DataFrame, pairs_df: pd.DataFrame, seed: int = RANDOM_SEED) -> Data:
    """Assemble the PyG graph: node features, a leakage-free message-passing
    edge index, and per-split positive/negative supervision edges.
    """
    rng = np.random.default_rng(seed)

    drug_ids = drugs_df["drug_id"].tolist()
    index_of = {drug_id: index for index, drug_id in enumerate(drug_ids)}
    num_nodes = len(drug_ids)

    features = standardize_features(compute_node_features(drugs_df["smiles"].tolist()))
    x = torch.tensor(features, dtype=torch.float)

    positives = pairs_df[pairs_df["type"] != 0]
    pairs_by_split = assign_pairs_to_splits(positives, index_of)

    # Every known positive pair, across all splits, is off-limits when
    # sampling negatives.
    forbidden_pairs = {pair for pairs in pairs_by_split.values() for pair in pairs}

    data = Data(x=x, num_nodes=num_nodes)
    data.drug_ids = drug_ids
    data.feature_names = FEATURE_NAMES

    for split in ("training", "validation", "testing"):
        split_pairs = pairs_by_split[split]
        pos_edges = np.array(split_pairs, dtype=np.int64).T.reshape(2, -1)
        neg_edges = sample_negative_edges(
            num_samples=pos_edges.shape[1],
            num_nodes=num_nodes,
            forbidden_pairs=forbidden_pairs,
            rng=rng,
        )

        prefix = {"training": "train", "validation": "val", "testing": "test"}[split]
        setattr(data, f"{prefix}_pos_edges", torch.tensor(pos_edges, dtype=torch.long))
        setattr(data, f"{prefix}_neg_edges", torch.tensor(neg_edges, dtype=torch.long))

    # Message passing sees training positives only, in both directions
    # (drug interactions are symmetric).
    train_pos = data.train_pos_edges
    data.edge_index = torch.cat([train_pos, train_pos.flip(0)], dim=1)

    return data


def run() -> None:
    GRAPH_DIR.mkdir(parents=True, exist_ok=True)

    print("Loading processed tables...")
    drugs_df = pd.read_csv(PROCESSED_DIR / "drugs.csv")
    pairs_df = pd.read_csv(PROCESSED_DIR / "ddi_pairs.csv")

    print(f"Building graph from {len(drugs_df)} drugs and {len(pairs_df)} pairs...")
    data = build_graph(drugs_df, pairs_df)

    print(f"  nodes:            {data.num_nodes}")
    print(f"  node features:    {data.x.shape[1]} ({', '.join(data.feature_names)})")
    print(
        f"  message-passing edges: {data.edge_index.shape[1]} (training positives, both directions)"
    )
    for prefix in ("train", "val", "test"):
        pos = getattr(data, f"{prefix}_pos_edges").shape[1]
        neg = getattr(data, f"{prefix}_neg_edges").shape[1]
        print(f"  {prefix:5s} supervision: {pos} positive / {neg} negative")

    output_path = GRAPH_DIR / "ddi_graph.pt"
    torch.save(data, output_path)
    print(f"Saved to {output_path}")


if __name__ == "__main__":
    run()
