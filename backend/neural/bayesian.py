"""
Bayesian neural network components for uncertainty-aware predictions.

Includes:
  - MC Dropout inference
  - Deep Ensembles
  - Bayesian Linear layers (variational inference)
  - Calibration metrics (ECE, reliability diagrams)
"""
from __future__ import annotations

import math
from typing import List, Optional, Tuple

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.distributions import Normal


class BayesianLinear(nn.Module):
    """
    Variational Bayesian linear layer using local reparameterisation trick.
    Maintains a Normal distribution over weights and biases.
    """

    def __init__(self, in_features: int, out_features: int, prior_var: float = 1.0):
        super().__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.prior_var = prior_var

        self.weight_mu = nn.Parameter(torch.randn(out_features, in_features) * 0.1)
        self.weight_rho = nn.Parameter(torch.full((out_features, in_features), -3.0))
        self.bias_mu = nn.Parameter(torch.zeros(out_features))
        self.bias_rho = nn.Parameter(torch.full((out_features,), -3.0))

    @property
    def weight_sigma(self) -> torch.Tensor:
        return F.softplus(self.weight_rho) + 1e-5

    @property
    def bias_sigma(self) -> torch.Tensor:
        return F.softplus(self.bias_rho) + 1e-5

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if self.training:
            eps_w = torch.randn_like(self.weight_mu)
            eps_b = torch.randn_like(self.bias_mu)
            weight = self.weight_mu + self.weight_sigma * eps_w
            bias = self.bias_mu + self.bias_sigma * eps_b
        else:
            weight = self.weight_mu
            bias = self.bias_mu
        return F.linear(x, weight, bias)

    def kl_divergence(self) -> torch.Tensor:
        prior = Normal(0, math.sqrt(self.prior_var))
        q_w = Normal(self.weight_mu, self.weight_sigma)
        q_b = Normal(self.bias_mu, self.bias_sigma)
        return (
            torch.distributions.kl_divergence(q_w, prior).sum()
            + torch.distributions.kl_divergence(q_b, prior).sum()
        )


class BayesianMLP(nn.Module):
    """Bayesian MLP with variational inference for property prediction."""

    def __init__(
        self,
        in_dim: int,
        hidden_dims: List[int],
        out_dim: int,
        prior_var: float = 1.0,
    ):
        super().__init__()
        dims = [in_dim] + hidden_dims
        self.layers = nn.ModuleList([
            BayesianLinear(dims[i], dims[i+1], prior_var) for i in range(len(dims)-1)
        ])
        self.out = BayesianLinear(dims[-1], out_dim, prior_var)
        self.activations = nn.ModuleList([nn.ReLU() for _ in range(len(dims)-1)])

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        for layer, act in zip(self.layers, self.activations):
            x = act(layer(x))
        return self.out(x)

    def kl_loss(self) -> torch.Tensor:
        kl = sum(layer.kl_divergence() for layer in self.layers)
        kl += self.out.kl_divergence()
        return kl

    def predict_with_uncertainty(
        self, x: torch.Tensor, n_samples: int = 50
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        self.train()
        samples = torch.stack([self.forward(x) for _ in range(n_samples)], dim=0)
        mean = samples.mean(0)
        std = samples.std(0)
        self.eval()
        return mean, std


class MCDropoutNet(nn.Module):
    """
    Standard MLP with MC Dropout for predictive uncertainty.
    At inference time, keep dropout active and sample K forward passes.
    """

    def __init__(self, in_dim: int, hidden: int, out_dim: int, n_layers: int = 3, p: float = 0.1):
        super().__init__()
        layers: List[nn.Module] = [nn.Linear(in_dim, hidden), nn.ReLU(), nn.Dropout(p)]
        for _ in range(n_layers - 1):
            layers += [nn.Linear(hidden, hidden), nn.ReLU(), nn.Dropout(p)]
        layers.append(nn.Linear(hidden, out_dim))
        self.net = nn.Sequential(*layers)
        self.p = p

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)

    def predict_with_uncertainty(
        self, x: torch.Tensor, n_samples: int = 100
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        self.train()
        samples = torch.stack([self(x) for _ in range(n_samples)], dim=0)
        self.eval()
        return samples.mean(0), samples.std(0)

    def epistemic_uncertainty(self, x: torch.Tensor, n_samples: int = 100) -> torch.Tensor:
        _, std = self.predict_with_uncertainty(x, n_samples)
        return std

    def aleatoric_uncertainty_approx(self, x: torch.Tensor) -> torch.Tensor:
        self.eval()
        return torch.abs(self(x)) * 0.05


class DeepEnsemble(nn.Module):
    """
    Ensemble of deterministic NNs for uncertainty estimation.
    Better calibrated than MC Dropout for out-of-distribution detection.
    """

    def __init__(self, in_dim: int, hidden: int, out_dim: int, n_members: int = 5):
        super().__init__()
        self.members = nn.ModuleList([
            nn.Sequential(
                nn.Linear(in_dim, hidden), nn.ReLU(),
                nn.Linear(hidden, hidden), nn.ReLU(),
                nn.Linear(hidden, out_dim * 2),  # predict mean and log-variance
            )
            for _ in range(n_members)
        ])

    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        means, vars_ = [], []
        for member in self.members:
            out = member(x)
            half = out.size(-1) // 2
            mu, log_var = out[..., :half], out[..., half:]
            means.append(mu)
            vars_.append(torch.exp(log_var.clamp(-6, 6)))

        mean_stack = torch.stack(means, 0)
        var_stack = torch.stack(vars_, 0)

        ensemble_mean = mean_stack.mean(0)
        ensemble_var = (var_stack + mean_stack ** 2).mean(0) - ensemble_mean ** 2
        return ensemble_mean, ensemble_var.sqrt()

    def nll_loss(self, x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
        total_loss = torch.tensor(0.0)
        for member in self.members:
            out = member(x)
            half = out.size(-1) // 2
            mu, log_var = out[..., :half], out[..., half:].clamp(-6, 6)
            var = torch.exp(log_var)
            total_loss = total_loss + (log_var + (y - mu) ** 2 / var).mean()
        return total_loss / len(self.members)


# --------------------------------------------------------------------------- #
#  Calibration metrics                                                          #
# --------------------------------------------------------------------------- #

def expected_calibration_error(
    confidences: np.ndarray,
    accuracies: np.ndarray,
    n_bins: int = 15,
) -> float:
    """
    ECE: |sum_b (|B_b|/n) * |acc(b) - conf(b)|
    Lower is better (0 = perfectly calibrated).
    """
    bin_edges = np.linspace(0, 1, n_bins + 1)
    ece = 0.0
    n = len(confidences)
    for lo, hi in zip(bin_edges[:-1], bin_edges[1:]):
        mask = (confidences >= lo) & (confidences < hi)
        if mask.sum() == 0:
            continue
        acc = accuracies[mask].mean()
        conf = confidences[mask].mean()
        ece += (mask.sum() / n) * abs(acc - conf)
    return float(ece)


def reliability_diagram_data(
    confidences: np.ndarray, accuracies: np.ndarray, n_bins: int = 10
) -> dict:
    """Returns data for plotting a reliability diagram."""
    bin_edges = np.linspace(0, 1, n_bins + 1)
    diagram = {"bin_midpoints": [], "avg_confidence": [], "avg_accuracy": [], "counts": []}
    for lo, hi in zip(bin_edges[:-1], bin_edges[1:]):
        mask = (confidences >= lo) & (confidences < hi)
        diagram["bin_midpoints"].append(round((lo + hi) / 2, 3))
        diagram["avg_confidence"].append(float(confidences[mask].mean()) if mask.any() else 0.0)
        diagram["avg_accuracy"].append(float(accuracies[mask].mean()) if mask.any() else 0.0)
        diagram["counts"].append(int(mask.sum()))
    return diagram
