"""
Graph Neural Network (GNN) implementations for the QuantumAgent knowledge graph.

Architectures:
  - GraphSAGE: inductive node embeddings
  - GAT: graph attention networks
  - MPNN: message-passing for molecular graphs
  - HeteroGNN: heterogeneous node/edge types (paper, algorithm, hardware, molecule)
"""
from __future__ import annotations

import math
from typing import Dict, List, Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F


# --------------------------------------------------------------------------- #
#  GraphSAGE                                                                   #
# --------------------------------------------------------------------------- #

class SAGEConv(nn.Module):
    def __init__(self, in_dim: int, out_dim: int, aggregator: str = "mean"):
        super().__init__()
        self.aggregator = aggregator
        self.W = nn.Linear(in_dim * 2, out_dim, bias=False)
        self.bias = nn.Parameter(torch.zeros(out_dim))
        self.norm = nn.LayerNorm(out_dim)

    def forward(self, h: torch.Tensor, adj: torch.Tensor) -> torch.Tensor:
        if self.aggregator == "mean":
            deg = adj.sum(-1, keepdim=True).clamp(min=1)
            agg = adj @ h / deg
        elif self.aggregator == "max":
            inf_mask = (adj == 0).float() * -1e9
            agg = (h.unsqueeze(0) + inf_mask.unsqueeze(-1)).max(1).values
        else:
            agg = adj @ h / adj.sum(-1, keepdim=True).clamp(min=1)

        out = self.W(torch.cat([h, agg], dim=-1)) + self.bias
        return F.relu(self.norm(out))


class GraphSAGE(nn.Module):
    def __init__(self, in_dim: int, hidden: int, out_dim: int, n_layers: int = 3, dropout: float = 0.2):
        super().__init__()
        dims = [in_dim] + [hidden] * (n_layers - 1) + [out_dim]
        self.convs = nn.ModuleList([SAGEConv(dims[i], dims[i+1]) for i in range(n_layers)])
        self.dropout = nn.Dropout(dropout)

    def forward(self, h: torch.Tensor, adj: torch.Tensor) -> torch.Tensor:
        for i, conv in enumerate(self.convs):
            h = conv(h, adj)
            if i < len(self.convs) - 1:
                h = self.dropout(h)
        return h


# --------------------------------------------------------------------------- #
#  Graph Attention Network (GAT)                                               #
# --------------------------------------------------------------------------- #

class GATLayer(nn.Module):
    def __init__(self, in_dim: int, out_dim: int, heads: int = 4, dropout: float = 0.1):
        super().__init__()
        self.heads = heads
        self.d = out_dim // heads
        self.W = nn.Linear(in_dim, out_dim, bias=False)
        self.a_src = nn.Parameter(torch.randn(heads, self.d))
        self.a_dst = nn.Parameter(torch.randn(heads, self.d))
        self.dropout = nn.Dropout(dropout)
        self.leaky = nn.LeakyReLU(0.2)
        self.norm = nn.LayerNorm(out_dim)

    def forward(self, h: torch.Tensor, adj: torch.Tensor) -> torch.Tensor:
        N = h.size(0)
        h = self.W(h).view(N, self.heads, self.d)          # (N, H, d)

        e_src = (h * self.a_src).sum(-1)                   # (N, H)
        e_dst = (h * self.a_dst).sum(-1)

        attn = self.leaky(e_src.unsqueeze(1) + e_dst.unsqueeze(0))  # (N, N, H)
        mask = (adj == 0).unsqueeze(-1)
        attn = attn.masked_fill(mask, float("-inf"))
        attn = F.softmax(attn, dim=1)
        attn = self.dropout(attn)

        out = (attn.unsqueeze(-1) * h.unsqueeze(0)).sum(1)  # (N, H, d)
        out = out.view(N, -1)
        return F.elu(self.norm(out))


class GAT(nn.Module):
    def __init__(self, in_dim: int, hidden: int, out_dim: int, n_layers: int = 3,
                 heads: int = 4, dropout: float = 0.2):
        super().__init__()
        dims = [in_dim] + [hidden] * (n_layers - 1) + [out_dim]
        self.layers = nn.ModuleList([
            GATLayer(dims[i], dims[i+1], heads if i < n_layers - 1 else 1, dropout)
            for i in range(n_layers)
        ])
        self.dropout = nn.Dropout(dropout)

    def forward(self, h: torch.Tensor, adj: torch.Tensor) -> torch.Tensor:
        for i, layer in enumerate(self.layers):
            h = layer(h, adj)
            if i < len(self.layers) - 1:
                h = self.dropout(h)
        return h


# --------------------------------------------------------------------------- #
#  MPNN — Message Passing Neural Network (for molecules)                       #
# --------------------------------------------------------------------------- #

class MPNNLayer(nn.Module):
    def __init__(self, node_dim: int, edge_dim: int, hidden: int):
        super().__init__()
        self.message_net = nn.Sequential(
            nn.Linear(node_dim * 2 + edge_dim, hidden),
            nn.ReLU(),
            nn.Linear(hidden, hidden),
        )
        self.update_gru = nn.GRUCell(hidden, node_dim)
        self.norm = nn.LayerNorm(node_dim)

    def forward(self, h: torch.Tensor, edge_index: torch.Tensor, edge_attr: torch.Tensor) -> torch.Tensor:
        src, dst = edge_index[0], edge_index[1]
        msg = self.message_net(torch.cat([h[src], h[dst], edge_attr], dim=-1))

        agg = torch.zeros(h.size(0), msg.size(-1), device=h.device, dtype=h.dtype)
        agg.index_add_(0, dst, msg)

        h_new = self.update_gru(agg, h)
        return self.norm(h_new)


class MPNN(nn.Module):
    def __init__(self, node_dim: int, edge_dim: int, hidden: int, n_steps: int = 3,
                 readout_dim: int = 1):
        super().__init__()
        self.steps = nn.ModuleList([MPNNLayer(node_dim, edge_dim, hidden) for _ in range(n_steps)])
        self.readout = nn.Sequential(
            nn.Linear(node_dim, hidden),
            nn.ReLU(),
            nn.Linear(hidden, readout_dim),
        )

    def forward(self, h: torch.Tensor, edge_index: torch.Tensor, edge_attr: torch.Tensor,
                batch: Optional[torch.Tensor] = None) -> Tuple[torch.Tensor, torch.Tensor]:
        for step in self.steps:
            h = step(h, edge_index, edge_attr)

        if batch is None:
            graph_repr = h.mean(0, keepdim=True)
        else:
            n_graphs = batch.max().item() + 1
            graph_repr = torch.zeros(n_graphs, h.size(-1), device=h.device)
            graph_repr.index_add_(0, batch, h)
            counts = torch.bincount(batch, minlength=n_graphs).float().unsqueeze(-1)
            graph_repr = graph_repr / counts.clamp(min=1)

        return h, self.readout(graph_repr)


# --------------------------------------------------------------------------- #
#  HeteroGNN — handles multiple node/edge types                                #
# --------------------------------------------------------------------------- #

class HeteroGNNLayer(nn.Module):
    """
    Single HeteroGNN layer.
    node_types: list of type names (e.g. 'paper', 'algorithm', 'hardware')
    edge_types: list of (src_type, relation, dst_type) triples
    """

    def __init__(
        self,
        node_dims: Dict[str, int],
        hidden: int,
        edge_types: List[Tuple[str, str, str]],
    ):
        super().__init__()
        self.edge_types = edge_types

        self.W_src = nn.ModuleDict({
            f"{s}_{r}_{d}": nn.Linear(node_dims[s], hidden, bias=False)
            for s, r, d in edge_types
        })
        self.W_dst = nn.ModuleDict({
            f"{s}_{r}_{d}": nn.Linear(node_dims.get(d, hidden), hidden, bias=False)
            for s, r, d in edge_types
        })
        self.W_out = nn.ModuleDict({
            nt: nn.Linear(hidden, hidden) for nt in node_dims
        })
        self.norms = nn.ModuleDict({nt: nn.LayerNorm(hidden) for nt in node_dims})

    def forward(
        self,
        h_dict: Dict[str, torch.Tensor],
        edge_index_dict: Dict[Tuple, torch.Tensor],
    ) -> Dict[str, torch.Tensor]:
        agg = {nt: torch.zeros(h.size(0), next(iter(self.W_out.values())).out_features, device=h.device)
               for nt, h in h_dict.items()}
        counts = {nt: torch.zeros(h.size(0), 1, device=h.device) for nt, h in h_dict.items()}

        for (s, r, d), eidx in edge_index_dict.items():
            key = f"{s}_{r}_{d}"
            if key not in self.W_src:
                continue
            src_feats = self.W_src[key](h_dict[s][eidx[0]])
            dst_feats = self.W_dst[key](h_dict[d][eidx[1]])
            msg = F.relu(src_feats + dst_feats)
            agg[d].index_add_(0, eidx[1], msg)
            counts[d].index_add_(0, eidx[1], torch.ones(len(eidx[1]), 1, device=eidx.device))

        out = {}
        for nt in h_dict:
            mean_agg = agg[nt] / counts[nt].clamp(min=1)
            out[nt] = F.relu(self.norms[nt](self.W_out[nt](mean_agg)))
        return out


# --------------------------------------------------------------------------- #
#  Link Prediction Head                                                         #
# --------------------------------------------------------------------------- #

class LinkPredictor(nn.Module):
    def __init__(self, embed_dim: int, hidden: int = 128):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(embed_dim * 2, hidden),
            nn.ReLU(),
            nn.LayerNorm(hidden),
            nn.Linear(hidden, 64),
            nn.ReLU(),
            nn.Linear(64, 1),
            nn.Sigmoid(),
        )

    def forward(self, h_src: torch.Tensor, h_dst: torch.Tensor) -> torch.Tensor:
        return self.net(torch.cat([h_src, h_dst], dim=-1))


# --------------------------------------------------------------------------- #
#  Contrastive / Self-supervised pre-training                                   #
# --------------------------------------------------------------------------- #

class GraphContrastiveLoss(nn.Module):
    """
    InfoNCE-style contrastive loss for self-supervised GNN pre-training.
    Positive pairs: augmented views of the same graph.
    Negative pairs: different graphs in the batch.
    """

    def __init__(self, temperature: float = 0.1):
        super().__init__()
        self.tau = temperature

    def forward(self, z1: torch.Tensor, z2: torch.Tensor) -> torch.Tensor:
        z1 = F.normalize(z1, dim=-1)
        z2 = F.normalize(z2, dim=-1)
        N = z1.size(0)

        logits = torch.mm(z1, z2.T) / self.tau
        labels = torch.arange(N, device=z1.device)
        return (F.cross_entropy(logits, labels) + F.cross_entropy(logits.T, labels)) / 2
