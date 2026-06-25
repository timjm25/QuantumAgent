"""Tests for neural network components."""
import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import torch
import numpy as np
import pytest

from backend.neural.gnn import GraphSAGE, GAT, MPNN, LinkPredictor, GraphContrastiveLoss
from backend.neural.transformer import ScientificTransformer, CrossAttentionReasoner, MixtureOfExperts
from backend.neural.bayesian import BayesianLinear, BayesianMLP, MCDropoutNet, DeepEnsemble, expected_calibration_error
from backend.neural.embeddings import SentenceEmbedder, MolecularEmbedder, VectorStore


class TestGNN:
    def test_graphsage_forward(self):
        model = GraphSAGE(16, 32, 8, n_layers=2)
        h = torch.randn(5, 16)
        adj = torch.eye(5) + torch.rand(5, 5) * 0.3
        adj = (adj > 0.5).float()
        out = model(h, adj)
        assert out.shape == (5, 8)

    def test_gat_forward(self):
        model = GAT(16, 32, 8, n_layers=2, heads=2)
        h = torch.randn(4, 16)
        adj = torch.ones(4, 4)
        out = model(h, adj)
        assert out.shape[0] == 4

    def test_mpnn_forward(self):
        model = MPNN(node_dim=8, edge_dim=4, hidden=16, n_steps=2, readout_dim=2)
        h = torch.randn(6, 8)
        edge_index = torch.tensor([[0, 1, 2, 3], [1, 2, 3, 4]])
        edge_attr = torch.randn(4, 4)
        node_emb, graph_repr = model(h, edge_index, edge_attr)
        assert node_emb.shape == (6, 8)
        assert graph_repr.shape == (1, 2)

    def test_link_predictor(self):
        pred = LinkPredictor(embed_dim=32)
        h_src = torch.randn(5, 32)
        h_dst = torch.randn(5, 32)
        scores = pred(h_src, h_dst)
        assert scores.shape == (5, 1)
        assert (scores >= 0).all() and (scores <= 1).all()

    def test_contrastive_loss(self):
        loss_fn = GraphContrastiveLoss(temperature=0.1)
        z1 = torch.randn(4, 16)
        z2 = torch.randn(4, 16)
        loss = loss_fn(z1, z2)
        assert loss.item() > 0
        assert not torch.isnan(loss)


class TestTransformer:
    def test_scientific_transformer_forward(self):
        model = ScientificTransformer(vocab_size=100, d_model=64, n_heads=4, n_layers=2, d_ff=256)
        tokens = torch.randint(0, 100, (2, 16))
        out, cls = model(tokens)
        assert out.shape == (2, 16, 64)
        assert cls.shape == (2, 64)

    def test_cross_attention_reasoner(self):
        model = CrossAttentionReasoner(d_model=64, n_heads=4, n_layers=2)
        query = torch.randn(3, 64)
        context = torch.randn(3, 8, 64)
        out = model(query, context)
        assert out.shape == (3, 64)

    def test_mixture_of_experts(self):
        model = MixtureOfExperts(d_model=64, n_experts=4, top_k=2, d_ff=128)
        x = torch.randn(2, 6, 64)
        out = model(x)
        assert out.shape == (2, 6, 64)


class TestBayesian:
    def test_bayesian_linear_kl(self):
        layer = BayesianLinear(8, 4)
        x = torch.randn(3, 8)
        out = layer(x)
        assert out.shape == (3, 4)
        kl = layer.kl_divergence()
        assert kl.item() > 0

    def test_bayesian_mlp_uncertainty(self):
        model = BayesianMLP(4, [16, 16], 2)
        x = torch.randn(5, 4)
        mean, std = model.predict_with_uncertainty(x, n_samples=10)
        assert mean.shape == (5, 2)
        assert std.shape == (5, 2)
        assert (std >= 0).all()

    def test_mc_dropout_uncertainty(self):
        model = MCDropoutNet(4, 16, 2)
        x = torch.randn(3, 4)
        mean, std = model.predict_with_uncertainty(x, n_samples=20)
        assert mean.shape == (3, 2)

    def test_deep_ensemble(self):
        model = DeepEnsemble(4, 16, 1, n_members=3)
        x = torch.randn(5, 4)
        mean, std = model(x)
        assert mean.shape == (5, 1)
        assert (std >= 0).all()

    def test_ece_computation(self):
        confs = np.random.uniform(0, 1, 100)
        accs = (np.random.rand(100) < confs).astype(float)
        ece = expected_calibration_error(confs, accs)
        assert 0 <= ece <= 1


class TestEmbeddings:
    def test_sentence_embedder(self):
        model = SentenceEmbedder(vocab_size=100, d_model=64, embed_dim=32)
        tokens = torch.randint(0, 100, (2, 16))
        emb = model(tokens)
        assert emb.shape == (2, 32)
        norms = emb.norm(dim=-1)
        assert torch.allclose(norms, torch.ones(2), atol=1e-5)

    def test_molecular_embedder(self):
        model = MolecularEmbedder(embed_dim=64)
        smiles = "CC(=O)Oc1ccccc1C(=O)O"  # aspirin
        ids = model.smiles_to_ids(smiles).unsqueeze(0)
        emb = model(ids)
        assert emb.shape == (1, 64)

    def test_vector_store_search(self):
        store = VectorStore(dim=8)
        v1 = np.array([1, 0, 0, 0, 0, 0, 0, 0], dtype=np.float32)
        v2 = np.array([0, 1, 0, 0, 0, 0, 0, 0], dtype=np.float32)
        v3 = np.array([1, 0.1, 0, 0, 0, 0, 0, 0], dtype=np.float32)
        store.add("a", v1, {"label": "paper A"})
        store.add("b", v2, {"label": "paper B"})
        store.add("c", v3, {"label": "paper C"})
        results = store.search(v1, top_k=2)
        assert len(results) == 2
        assert results[0]["key"] in ("a", "c")


if __name__ == "__main__":
    pytest.main([__file__, "-v", "--tb=short"])
