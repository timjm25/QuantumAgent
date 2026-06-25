"""
Scientific embedding layer for the QuantumAgent knowledge graph.

Provides dense vector representations for:
  - Scientific papers (title + abstract → 768-dim embedding)
  - Quantum algorithms (name + description)
  - Molecules (SMILES → fingerprint embedding)
  - Hardware backends
  - Research entities
"""
from __future__ import annotations

import hashlib
import math
from typing import Dict, List, Optional

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F


class BagOfWordsEmbedder(nn.Module):
    """Fast BoW with learned word weights — used for scientific keywords."""

    def __init__(self, vocab_size: int = 50000, embed_dim: int = 256):
        super().__init__()
        self.embed = nn.EmbeddingBag(vocab_size, embed_dim, mode="mean", padding_idx=0)
        self.proj = nn.Sequential(
            nn.LayerNorm(embed_dim),
            nn.Linear(embed_dim, embed_dim),
            nn.GELU(),
        )

    def forward(self, token_ids: torch.Tensor, offsets: Optional[torch.Tensor] = None) -> torch.Tensor:
        x = self.embed(token_ids, offsets)
        return F.normalize(self.proj(x), dim=-1)


class SentenceEmbedder(nn.Module):
    """
    Lightweight sentence encoder using mean-pooled transformer features.
    Drop-in replacement for sentence-transformers when offline.
    """

    def __init__(self, vocab_size: int = 32000, d_model: int = 256, n_heads: int = 4,
                 n_layers: int = 3, max_len: int = 512, embed_dim: int = 256):
        super().__init__()
        self.embed = nn.Embedding(vocab_size, d_model, padding_idx=0)
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model, nhead=n_heads, dim_feedforward=d_model * 4,
            dropout=0.1, batch_first=True, norm_first=True,
        )
        self.encoder = nn.TransformerEncoder(encoder_layer, num_layers=n_layers)
        self.norm = nn.LayerNorm(d_model)
        self.proj = nn.Linear(d_model, embed_dim)

    def forward(self, token_ids: torch.Tensor, attention_mask: Optional[torch.Tensor] = None) -> torch.Tensor:
        x = self.embed(token_ids) * math.sqrt(self.embed.embedding_dim)
        key_padding_mask = (attention_mask == 0) if attention_mask is not None else None
        x = self.encoder(x, src_key_padding_mask=key_padding_mask)
        x = self.norm(x)

        if attention_mask is not None:
            mask = attention_mask.unsqueeze(-1).float()
            pooled = (x * mask).sum(1) / mask.sum(1).clamp(min=1)
        else:
            pooled = x.mean(1)

        return F.normalize(self.proj(pooled), dim=-1)


class MolecularEmbedder(nn.Module):
    """
    SMILES-to-embedding via character-level CNN + pooling.
    Can be replaced with a pre-trained ChemBERTa or MolBERT in production.
    """

    SMILES_CHARSET = list("CNOSPFClBrI#=-()[]1234567890@+. ")
    CHARSET_SIZE = len(SMILES_CHARSET) + 2  # +2 for UNK and PAD

    def __init__(self, embed_dim: int = 256, conv_channels: int = 128, max_smiles: int = 256):
        super().__init__()
        self.char_embed = nn.Embedding(self.CHARSET_SIZE, conv_channels, padding_idx=0)
        self.convs = nn.ModuleList([
            nn.Conv1d(conv_channels, conv_channels, k, padding=k // 2) for k in [3, 5, 7]
        ])
        self.pool = nn.AdaptiveMaxPool1d(1)
        self.proj = nn.Sequential(
            nn.Linear(conv_channels * 3, embed_dim),
            nn.GELU(),
            nn.LayerNorm(embed_dim),
        )
        self.char_to_idx = {c: i + 1 for i, c in enumerate(self.SMILES_CHARSET)}

    def smiles_to_ids(self, smiles: str, max_len: int = 256) -> torch.Tensor:
        ids = [self.char_to_idx.get(c, len(self.SMILES_CHARSET) + 1) for c in smiles[:max_len]]
        ids += [0] * (max_len - len(ids))
        return torch.tensor(ids, dtype=torch.long)

    def forward(self, smiles_ids: torch.Tensor) -> torch.Tensor:
        x = self.char_embed(smiles_ids).transpose(1, 2)
        feats = [F.relu(conv(x)) for conv in self.convs]
        pooled = [self.pool(f).squeeze(-1) for f in feats]
        return F.normalize(self.proj(torch.cat(pooled, dim=-1)), dim=-1)


class EntityEmbedder(nn.Module):
    """
    Unified embedder for all knowledge-graph entity types.
    Each type has a dedicated embedding sub-module.
    """

    NODE_TYPES = ["paper", "algorithm", "experiment", "hardware", "molecule", "researcher", "institution", "dataset", "result"]

    def __init__(self, embed_dim: int = 256, vocab_size: int = 32000):
        super().__init__()
        self.embed_dim = embed_dim
        self.type_bias = nn.Embedding(len(self.NODE_TYPES), embed_dim)
        self.text_encoder = SentenceEmbedder(vocab_size, d_model=256, embed_dim=embed_dim)
        self.mol_encoder = MolecularEmbedder(embed_dim)
        self.type_to_idx = {t: i for i, t in enumerate(self.NODE_TYPES)}

    def embed_text(self, token_ids: torch.Tensor, node_type: str) -> torch.Tensor:
        text_emb = self.text_encoder(token_ids)
        type_idx = torch.tensor(self.type_to_idx.get(node_type, 0), device=token_ids.device)
        type_emb = F.normalize(self.type_bias(type_idx).unsqueeze(0), dim=-1)
        return F.normalize(text_emb + 0.1 * type_emb, dim=-1)

    def embed_molecule(self, smiles: str) -> torch.Tensor:
        ids = self.mol_encoder.smiles_to_ids(smiles).unsqueeze(0)
        return self.mol_encoder(ids)

    def random_embed(self, n: int, node_type: str) -> torch.Tensor:
        """Fallback: random unit-sphere embedding (used when tokenizer unavailable)."""
        emb = torch.randn(n, self.embed_dim)
        return F.normalize(emb, dim=-1)


class VectorStore:
    """
    In-memory approximate nearest-neighbour vector store.
    Production replacement: Qdrant or Milvus.
    """

    def __init__(self, dim: int = 256):
        self.dim = dim
        self._vectors: Dict[str, np.ndarray] = {}
        self._metadata: Dict[str, dict] = {}

    def add(self, key: str, vector: np.ndarray, metadata: Optional[dict] = None) -> None:
        v = np.array(vector, dtype=np.float32)
        v /= max(np.linalg.norm(v), 1e-8)
        self._vectors[key] = v
        self._metadata[key] = metadata or {}

    def search(self, query: np.ndarray, top_k: int = 10) -> List[dict]:
        if not self._vectors:
            return []
        q = np.array(query, dtype=np.float32)
        q /= max(np.linalg.norm(q), 1e-8)

        scores = {k: float(np.dot(q, v)) for k, v in self._vectors.items()}
        top = sorted(scores.items(), key=lambda x: x[1], reverse=True)[:top_k]
        return [{"key": k, "score": round(s, 4), **self._metadata.get(k, {})} for k, s in top]

    def __len__(self) -> int:
        return len(self._vectors)

    def embed_and_add(self, key: str, text: str, metadata: Optional[dict] = None) -> None:
        h = hashlib.md5(text.encode()).digest()
        vec = np.frombuffer(h, dtype=np.float32)
        vec = np.tile(vec, self.dim // len(vec) + 1)[:self.dim]
        self.add(key, vec, metadata)


global_vector_store = VectorStore()
