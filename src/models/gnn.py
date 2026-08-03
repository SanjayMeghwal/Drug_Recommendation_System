"""GNN architecture for binary drug-drug interaction prediction (Module D).

Encoder-decoder design, the standard formulation for link prediction:

  encode:  node features + graph structure  ->  one embedding per drug
  decode:  two drug embeddings              ->  interaction score

`forward` deliberately accepts the node feature matrix as an argument rather
than reading it from a stored graph. Module E (explainability) relies on
this: SHAP perturbs the twelve molecular descriptors of a drug pair, re-runs
the forward pass, and measures the effect on the prediction. That is what
lets explanations cite named chemical properties instead of opaque
embedding dimensions.
"""

import torch
import torch.nn.functional as F
from torch import nn
from torch_geometric.nn import GATConv, GCNConv


class DDIPredictor(nn.Module):
    """Two-layer graph convolution encoder with a pairwise link decoder.

    Args:
        in_channels: number of node features (12 molecular descriptors).
        hidden_channels: width of the first convolution layer.
        embedding_dim: dimensionality of the final drug embedding.
        conv_type: "gcn" (default) or "gat" for attention-based convolution.
        attention_heads: number of GAT heads; ignored when conv_type is "gcn".
        dropout: dropout probability applied between the two conv layers.
        decoder: "dot" for a parameter-free dot product, or "mlp" for a
            small learned scorer over the element-wise product.
    """

    def __init__(
        self,
        in_channels: int,
        hidden_channels: int = 64,
        embedding_dim: int = 32,
        conv_type: str = "gcn",
        attention_heads: int = 4,
        dropout: float = 0.2,
        decoder: str = "dot",
    ) -> None:
        super().__init__()

        if conv_type not in {"gcn", "gat"}:
            raise ValueError(f"conv_type must be 'gcn' or 'gat', got {conv_type!r}")
        if decoder not in {"dot", "mlp"}:
            raise ValueError(f"decoder must be 'dot' or 'mlp', got {decoder!r}")

        self.conv_type = conv_type
        self.decoder_type = decoder
        self.dropout = dropout

        if conv_type == "gcn":
            self.conv1 = GCNConv(in_channels, hidden_channels)
            self.conv2 = GCNConv(hidden_channels, embedding_dim)
        else:
            # Heads are concatenated in the first layer and averaged in the
            # second, so the output width stays embedding_dim either way.
            self.conv1 = GATConv(in_channels, hidden_channels, heads=attention_heads)
            self.conv2 = GATConv(
                hidden_channels * attention_heads,
                embedding_dim,
                heads=attention_heads,
                concat=False,
            )

        self.pair_scorer = (
            nn.Sequential(
                nn.Linear(embedding_dim, embedding_dim // 2),
                nn.ReLU(),
                nn.Linear(embedding_dim // 2, 1),
            )
            if decoder == "mlp"
            else None
        )

    def encode(self, x: torch.Tensor, edge_index: torch.Tensor) -> torch.Tensor:
        """Produce one embedding per drug from features and graph structure."""
        hidden = F.relu(self.conv1(x, edge_index))
        hidden = F.dropout(hidden, p=self.dropout, training=self.training)
        return self.conv2(hidden, edge_index)

    def decode(self, embeddings: torch.Tensor, edge_pairs: torch.Tensor) -> torch.Tensor:
        """Score each drug pair in `edge_pairs` ([2, num_pairs]) as a logit."""
        source = embeddings[edge_pairs[0]]
        target = embeddings[edge_pairs[1]]
        combined = source * target

        if self.pair_scorer is not None:
            return self.pair_scorer(combined).squeeze(-1)
        return combined.sum(dim=-1)

    def forward(
        self, x: torch.Tensor, edge_index: torch.Tensor, edge_pairs: torch.Tensor
    ) -> torch.Tensor:
        """Return interaction logits for the given drug pairs."""
        return self.decode(self.encode(x, edge_index), edge_pairs)


def build_model(in_channels: int, model_config: dict) -> DDIPredictor:
    """Construct a DDIPredictor from the `model` section of config.yaml."""
    return DDIPredictor(
        in_channels=in_channels,
        hidden_channels=model_config.get("hidden_channels", 64),
        embedding_dim=model_config.get("embedding_dim", 32),
        conv_type=model_config.get("conv_type", "gcn"),
        attention_heads=model_config.get("attention_heads", 4),
        dropout=model_config.get("dropout", 0.2),
        decoder=model_config.get("decoder", "dot"),
    )
