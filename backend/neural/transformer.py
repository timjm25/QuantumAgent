"""
Transformer architectures for scientific reasoning and literature synthesis.

Models:
  - ScientificTransformer: domain-adapted transformer for quantum research text
  - CrossAttentionReasoner: multi-document attention for synthesis
  - TemporalTransformer: for time-series experimental data
  - MixtureOfExperts: conditional computation for multi-domain queries
"""
from __future__ import annotations

import math
from typing import Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F


class ScaledDotProductAttention(nn.Module):
    def __init__(self, dropout: float = 0.1):
        super().__init__()
        self.dropout = nn.Dropout(dropout)

    def forward(
        self,
        Q: torch.Tensor,
        K: torch.Tensor,
        V: torch.Tensor,
        mask: Optional[torch.Tensor] = None,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        d = Q.size(-1)
        scores = torch.matmul(Q, K.transpose(-2, -1)) / math.sqrt(d)
        if mask is not None:
            scores = scores.masked_fill(mask == 0, float("-inf"))
        weights = F.softmax(scores, dim=-1)
        weights = self.dropout(weights)
        return torch.matmul(weights, V), weights


class MultiHeadAttention(nn.Module):
    def __init__(self, d_model: int, n_heads: int, dropout: float = 0.1):
        super().__init__()
        assert d_model % n_heads == 0
        self.d_model = d_model
        self.n_heads = n_heads
        self.d_k = d_model // n_heads

        self.W_Q = nn.Linear(d_model, d_model)
        self.W_K = nn.Linear(d_model, d_model)
        self.W_V = nn.Linear(d_model, d_model)
        self.W_O = nn.Linear(d_model, d_model)
        self.attn = ScaledDotProductAttention(dropout)
        self.dropout = nn.Dropout(dropout)

    def forward(
        self,
        query: torch.Tensor,
        key: torch.Tensor,
        value: torch.Tensor,
        mask: Optional[torch.Tensor] = None,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        B, T, _ = query.shape

        Q = self.W_Q(query).view(B, T, self.n_heads, self.d_k).transpose(1, 2)
        K = self.W_K(key).view(B, -1, self.n_heads, self.d_k).transpose(1, 2)
        V = self.W_V(value).view(B, -1, self.n_heads, self.d_k).transpose(1, 2)

        x, weights = self.attn(Q, K, V, mask)
        x = x.transpose(1, 2).contiguous().view(B, T, self.d_model)
        return self.W_O(x), weights.mean(1)


class PositionwiseFFN(nn.Module):
    def __init__(self, d_model: int, d_ff: int, dropout: float = 0.1):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(d_model, d_ff),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(d_ff, d_model),
            nn.Dropout(dropout),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


class TransformerEncoderLayer(nn.Module):
    def __init__(self, d_model: int, n_heads: int, d_ff: int, dropout: float = 0.1):
        super().__init__()
        self.attn = MultiHeadAttention(d_model, n_heads, dropout)
        self.ffn = PositionwiseFFN(d_model, d_ff, dropout)
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor, mask: Optional[torch.Tensor] = None) -> torch.Tensor:
        attn_out, _ = self.attn(x, x, x, mask)
        x = self.norm1(x + self.dropout(attn_out))
        x = self.norm2(x + self.ffn(x))
        return x


class SinusoidalPositionalEncoding(nn.Module):
    def __init__(self, d_model: int, max_len: int = 2048, dropout: float = 0.1):
        super().__init__()
        self.dropout = nn.Dropout(dropout)
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len).unsqueeze(1).float()
        div = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000) / d_model))
        pe[:, 0::2] = torch.sin(position * div)
        pe[:, 1::2] = torch.cos(position * div)
        self.register_buffer("pe", pe.unsqueeze(0))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.dropout(x + self.pe[:, :x.size(1)])


class ScientificTransformer(nn.Module):
    """
    Full transformer encoder adapted for scientific document embeddings.
    Outputs: (1) token-level representations, (2) [CLS] pooled document embedding.
    """

    def __init__(
        self,
        vocab_size: int = 32000,
        d_model: int = 512,
        n_heads: int = 8,
        n_layers: int = 6,
        d_ff: int = 2048,
        max_len: int = 2048,
        dropout: float = 0.1,
        n_classes: int = 0,
    ):
        super().__init__()
        self.embed = nn.Embedding(vocab_size, d_model, padding_idx=0)
        self.pe = SinusoidalPositionalEncoding(d_model, max_len, dropout)
        self.layers = nn.ModuleList([
            TransformerEncoderLayer(d_model, n_heads, d_ff, dropout) for _ in range(n_layers)
        ])
        self.norm = nn.LayerNorm(d_model)
        self.classifier = nn.Linear(d_model, n_classes) if n_classes > 0 else None
        self.d_model = d_model

    def forward(
        self,
        token_ids: torch.Tensor,
        mask: Optional[torch.Tensor] = None,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        x = self.pe(self.embed(token_ids) * math.sqrt(self.d_model))
        for layer in self.layers:
            x = layer(x, mask)
        x = self.norm(x)
        cls = x[:, 0]  # [CLS] token
        logits = self.classifier(cls) if self.classifier else None
        return x, cls if logits is None else logits


class CrossAttentionReasoner(nn.Module):
    """
    Attends over a set of document embeddings (retrieved from vector store)
    to synthesise a unified research context representation.
    """

    def __init__(self, d_model: int = 512, n_heads: int = 8, n_layers: int = 3, dropout: float = 0.1):
        super().__init__()
        self.self_attn_layers = nn.ModuleList([
            TransformerEncoderLayer(d_model, n_heads, d_model * 4, dropout) for _ in range(n_layers)
        ])
        self.cross_attn = MultiHeadAttention(d_model, n_heads, dropout)
        self.norm_q = nn.LayerNorm(d_model)
        self.norm_kv = nn.LayerNorm(d_model)
        self.out_proj = nn.Sequential(
            nn.LayerNorm(d_model),
            nn.Linear(d_model, d_model),
            nn.GELU(),
            nn.Dropout(dropout),
        )

    def forward(
        self,
        query_emb: torch.Tensor,
        context_embs: torch.Tensor,
        context_mask: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        for layer in self.self_attn_layers:
            context_embs = layer(context_embs)

        q = self.norm_q(query_emb.unsqueeze(1))
        kv = self.norm_kv(context_embs)
        attended, _ = self.cross_attn(q, kv, kv, context_mask)
        return self.out_proj(attended.squeeze(1))


class TemporalTransformer(nn.Module):
    """
    Transformer for experimental time-series data (e.g., VQE convergence,
    calibration curves, hardware performance over time).
    """

    def __init__(self, input_dim: int, d_model: int = 128, n_heads: int = 4,
                 n_layers: int = 3, output_dim: int = 1, max_len: int = 512):
        super().__init__()
        self.input_proj = nn.Linear(input_dim, d_model)
        self.pe = SinusoidalPositionalEncoding(d_model, max_len)
        self.layers = nn.ModuleList([
            TransformerEncoderLayer(d_model, n_heads, d_model * 4) for _ in range(n_layers)
        ])
        self.norm = nn.LayerNorm(d_model)
        self.head = nn.Sequential(
            nn.Linear(d_model, d_model // 2),
            nn.GELU(),
            nn.Linear(d_model // 2, output_dim),
        )

    def forward(self, x: torch.Tensor, mask: Optional[torch.Tensor] = None) -> torch.Tensor:
        x = self.pe(self.input_proj(x))
        for layer in self.layers:
            x = layer(x, mask)
        x = self.norm(x)
        return self.head(x[:, -1])  # predict from final token


# --------------------------------------------------------------------------- #
#  Mixture-of-Experts                                                           #
# --------------------------------------------------------------------------- #

class Expert(nn.Module):
    def __init__(self, d_model: int, d_ff: int, dropout: float = 0.1):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(d_model, d_ff),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(d_ff, d_model),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


class SparseTopKRouter(nn.Module):
    """Top-K gating router for mixture-of-experts."""

    def __init__(self, d_model: int, n_experts: int, top_k: int = 2):
        super().__init__()
        self.gate = nn.Linear(d_model, n_experts, bias=False)
        self.top_k = top_k

    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        logits = self.gate(x)
        topk_vals, topk_idx = torch.topk(logits, self.top_k, dim=-1)
        weights = F.softmax(topk_vals, dim=-1)
        return weights, topk_idx


class MixtureOfExperts(nn.Module):
    """
    Sparse MoE transformer layer: each token is processed by top-K experts.
    Used for multi-domain queries (chemistry, algorithms, hardware).
    """

    def __init__(self, d_model: int, n_experts: int = 8, top_k: int = 2, d_ff: int = 2048):
        super().__init__()
        self.router = SparseTopKRouter(d_model, n_experts, top_k)
        self.experts = nn.ModuleList([Expert(d_model, d_ff) for _ in range(n_experts)])
        self.norm = nn.LayerNorm(d_model)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        B, T, D = x.shape
        x_flat = x.view(-1, D)
        weights, indices = self.router(x_flat)

        out = torch.zeros_like(x_flat)
        for k in range(weights.size(-1)):
            expert_idx = indices[:, k]
            w = weights[:, k:k+1]
            for expert_id in range(len(self.experts)):
                mask = (expert_idx == expert_id)
                if mask.any():
                    out[mask] += w[mask] * self.experts[expert_id](x_flat[mask])

        return self.norm(x + out.view(B, T, D))
