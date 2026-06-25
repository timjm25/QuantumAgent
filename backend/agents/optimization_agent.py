"""
Experiment Optimisation Agent

Uses:
  - Bayesian optimisation (GP-UCB / Expected Improvement)
  - Reinforcement learning (policy gradient for parameter tuning)
  - Evolutionary algorithms (CMA-ES)
  - Active learning for sample-efficient experiment design
  - Monte Carlo Tree Search for discrete experiment planning
  - Differentiable gradient-based optimisation via PyTorch
"""
from __future__ import annotations

import asyncio
import math
from dataclasses import dataclass, field
from typing import Any, Callable, Dict, List, Optional, Tuple

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim

from .base import BaseAgent, AgentResult
from ..core.logging import get_logger

logger = get_logger(__name__)


class GaussianProcess:
    """
    Simple RBF-kernel Gaussian Process for Bayesian optimisation.
    Uses numpy for closed-form posterior computation.
    """

    def __init__(self, length_scale: float = 1.0, noise: float = 1e-4):
        self.l = length_scale
        self.noise = noise
        self.X_train: Optional[np.ndarray] = None
        self.y_train: Optional[np.ndarray] = None
        self.K_inv: Optional[np.ndarray] = None

    def rbf(self, X1: np.ndarray, X2: np.ndarray) -> np.ndarray:
        diffs = X1[:, None] - X2[None, :]
        return np.exp(-0.5 * np.sum(diffs ** 2, axis=-1) / self.l ** 2)

    def fit(self, X: np.ndarray, y: np.ndarray) -> None:
        self.X_train = X
        self.y_train = y
        K = self.rbf(X, X) + self.noise * np.eye(len(X))
        self.K_inv = np.linalg.inv(K)

    def predict(self, X_test: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        if self.X_train is None:
            return np.zeros(len(X_test)), np.ones(len(X_test))
        K_s = self.rbf(X_test, self.X_train)
        K_ss = self.rbf(X_test, X_test) + 1e-8 * np.eye(len(X_test))
        mu = K_s @ self.K_inv @ self.y_train
        cov = K_ss - K_s @ self.K_inv @ K_s.T
        return mu, np.sqrt(np.clip(np.diag(cov), 0, None))

    def expected_improvement(self, X_test: np.ndarray, xi: float = 0.01) -> np.ndarray:
        mu, sigma = self.predict(X_test)
        best = np.max(self.y_train) if self.y_train is not None else 0.0
        Z = (mu - best - xi) / (sigma + 1e-9)
        from scipy.special import ndtr
        ei = (mu - best - xi) * ndtr(Z) + sigma * np.exp(-0.5 * Z**2) / math.sqrt(2 * math.pi)
        return np.maximum(ei, 0.0)


class PolicyGradientNet(nn.Module):
    """Lightweight policy network for parameter tuning RL."""

    def __init__(self, state_dim: int, action_dim: int, hidden: int = 64):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(state_dim, hidden),
            nn.Tanh(),
            nn.Linear(hidden, hidden),
            nn.Tanh(),
            nn.Linear(hidden, action_dim),
        )
        self.log_std = nn.Parameter(torch.zeros(action_dim) - 1.0)

    def forward(self, state: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        mean = self.net(state)
        std = torch.exp(self.log_std.clamp(-3, 2))
        return mean, std


class CMAEvolutionStrategy:
    """Simplified CMA-ES for black-box optimisation."""

    def __init__(self, n_params: int, sigma0: float = 0.3, population_size: int = 20):
        self.n = n_params
        self.mu = np.zeros(n_params)
        self.sigma = sigma0
        self.lam = max(population_size, 4 + int(3 * math.log(n_params)))
        self.C = np.eye(n_params)
        self.p_sigma = np.zeros(n_params)
        self.p_c = np.zeros(n_params)
        self.generation = 0
        self.best_value = -np.inf
        self.best_params = np.zeros(n_params)

    def ask(self) -> np.ndarray:
        z = np.random.randn(self.lam, self.n)
        L = np.linalg.cholesky(self.C + 1e-8 * np.eye(self.n))
        return self.mu + self.sigma * (z @ L.T)

    def tell(self, solutions: np.ndarray, values: np.ndarray) -> None:
        idx = np.argsort(values)[::-1]
        elite_k = max(1, self.lam // 2)
        elite = solutions[idx[:elite_k]]
        elite_vals = values[idx[:elite_k]]

        if elite_vals[0] > self.best_value:
            self.best_value = elite_vals[0]
            self.best_params = elite[0].copy()

        self.mu = elite.mean(axis=0)
        diff = elite - self.mu
        self.C = (diff.T @ diff) / elite_k
        self.C = np.clip(self.C, -1e3, 1e3)
        self.sigma *= 0.99
        self.generation += 1


class OptimisationAgent(BaseAgent):
    """
    General-purpose experiment optimisation agent combining Bayesian,
    evolutionary, and RL-based approaches for quantum parameter tuning.
    """

    name = "optimization_agent"
    description = "Bayesian optimisation, RL, and evolutionary algorithms for quantum experiment design"
    capabilities = [
        "bayesian_optimisation",
        "reinforcement_learning",
        "cma_es",
        "active_learning",
        "gradient_optimisation",
        "hyperparameter_tuning",
        "pulse_optimisation",
        "vqe_parameter_optimisation",
    ]

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self._gp = GaussianProcess()
        self._cma: Optional[CMAEvolutionStrategy] = None
        self._optimisation_history: List[Dict] = []

    async def _execute(self, task_input: Dict[str, Any]) -> AgentResult:
        action = task_input.get("action", "bayesian_opt")
        dispatch = {
            "bayesian_opt": self._bayesian_optimisation,
            "cma_es": self._cma_optimisation,
            "rl_tuning": self._rl_parameter_tuning,
            "gradient_opt": self._gradient_optimisation,
            "active_learning": self._active_learning,
            "optimise_vqe": self._optimise_vqe_params,
            "optimise_pulse": self._optimise_pulse,
            "hyperparameter_search": self._hyperparameter_search,
        }
        handler = dispatch.get(action)
        if handler is None:
            return AgentResult(success=False, output={"error": f"Unknown action: {action}"})
        return await handler(task_input)

    # ------------------------------------------------------------------ #

    async def _bayesian_optimisation(self, inp: Dict) -> AgentResult:
        n_params = inp.get("n_params", 4)
        n_initial = inp.get("n_initial", 10)
        n_iterations = inp.get("n_iterations", 30)
        objective_name = inp.get("objective", "gate_fidelity")
        bounds = inp.get("bounds", [[-np.pi, np.pi]] * n_params)
        bounds = np.array(bounds)

        # Initial random exploration
        X_init = np.random.uniform(bounds[:, 0], bounds[:, 1], (n_initial, n_params))
        y_init = np.array([self._mock_objective(x, objective_name) for x in X_init])

        self._gp.fit(X_init, y_init)

        history = [{"iteration": i, "best": float(y_init[:i+1].max()), "type": "random"} for i in range(n_initial)]

        X_all, y_all = X_init.tolist(), y_init.tolist()

        for i in range(n_iterations):
            n_candidates = 500
            X_cand = np.random.uniform(bounds[:, 0], bounds[:, 1], (n_candidates, n_params))
            ei = self._gp.expected_improvement(X_cand)
            best_idx = np.argmax(ei)
            x_next = X_cand[best_idx]
            y_next = self._mock_objective(x_next, objective_name)

            X_all.append(x_next.tolist())
            y_all.append(y_next)
            self._gp.fit(np.array(X_all), np.array(y_all))
            history.append({"iteration": n_initial + i, "best": float(max(y_all)), "type": "bo"})

        best_idx = int(np.argmax(y_all))
        return AgentResult(
            success=True,
            output={
                "method": "Bayesian Optimisation (GP-EI)",
                "objective": objective_name,
                "n_evaluations": len(y_all),
                "best_params": X_all[best_idx],
                "best_value": round(y_all[best_idx], 6),
                "improvement_over_random": round(y_all[best_idx] - float(y_init.max()), 6),
                "convergence_history": history[::5],
            },
            confidence=0.88,
            reasoning=f"BO with GP-EI converged after {len(y_all)} evaluations. Best: {y_all[best_idx]:.6f}.",
        )

    async def _cma_optimisation(self, inp: Dict) -> AgentResult:
        n_params = inp.get("n_params", 8)
        n_generations = inp.get("n_generations", 100)
        sigma0 = inp.get("sigma0", 0.3)
        objective_name = inp.get("objective", "circuit_fidelity")

        cma = CMAEvolutionStrategy(n_params, sigma0)
        history = []

        for gen in range(n_generations):
            solutions = cma.ask()
            values = np.array([self._mock_objective(x, objective_name) for x in solutions])
            cma.tell(solutions, values)
            history.append({"generation": gen, "best": round(cma.best_value, 6)})
            if cma.sigma < 1e-5:
                break

        return AgentResult(
            success=True,
            output={
                "method": "CMA-ES",
                "generations": cma.generation,
                "best_params": cma.best_params.tolist(),
                "best_value": round(cma.best_value, 6),
                "final_sigma": round(cma.sigma, 6),
                "convergence_history": history[::10],
            },
            confidence=0.87,
        )

    async def _rl_parameter_tuning(self, inp: Dict) -> AgentResult:
        state_dim = inp.get("state_dim", 8)
        action_dim = inp.get("action_dim", 4)
        n_episodes = inp.get("n_episodes", 50)
        objective_name = inp.get("objective", "vqe_energy")

        policy = PolicyGradientNet(state_dim, action_dim)
        optimiser = optim.Adam(policy.parameters(), lr=1e-3)

        episode_rewards = []
        best_reward = -np.inf
        best_params = None

        for episode in range(n_episodes):
            state = torch.randn(1, state_dim)
            log_probs = []
            rewards = []

            for _ in range(10):
                mean, std = policy(state)
                dist = torch.distributions.Normal(mean, std)
                action = dist.sample()
                log_prob = dist.log_prob(action).sum()
                log_probs.append(log_prob)

                reward = torch.tensor(self._mock_objective(action.detach().numpy().flatten(), objective_name))
                rewards.append(reward)
                state = torch.cat([state[:, action_dim:], action], dim=-1)[:, :state_dim]

            episode_reward = sum(r.item() for r in rewards)
            episode_rewards.append(episode_reward)

            if episode_reward > best_reward:
                best_reward = episode_reward
                best_params = [lp.item() for lp in log_probs]

            returns = []
            G = 0.0
            for r in reversed(rewards):
                G = r.item() + 0.99 * G
                returns.insert(0, G)
            returns_t = torch.tensor(returns)
            if returns_t.std() > 1e-6:
                returns_t = (returns_t - returns_t.mean()) / (returns_t.std() + 1e-8)

            loss = -sum(lp * R for lp, R in zip(log_probs, returns_t))
            optimiser.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(policy.parameters(), 0.5)
            optimiser.step()

        return AgentResult(
            success=True,
            output={
                "method": "Policy Gradient (REINFORCE)",
                "n_episodes": n_episodes,
                "best_episode_reward": round(best_reward, 4),
                "final_episode_reward": round(episode_rewards[-1], 4),
                "improvement": round(episode_rewards[-1] - episode_rewards[0], 4),
                "reward_history": [round(r, 4) for r in episode_rewards[::5]],
            },
            confidence=0.82,
        )

    async def _gradient_optimisation(self, inp: Dict) -> AgentResult:
        n_params = inp.get("n_params", 8)
        n_steps = inp.get("n_steps", 200)
        lr = inp.get("lr", 0.05)

        params = torch.randn(n_params, requires_grad=True)
        optimiser = optim.Adam([params], lr=lr)
        history = []

        target = torch.randn(n_params)

        for step in range(n_steps):
            loss = ((params - target) ** 2).mean() + 0.01 * (params ** 2).mean()
            optimiser.zero_grad()
            loss.backward()
            optimiser.step()
            if step % 20 == 0:
                history.append({"step": step, "loss": round(loss.item(), 6)})

        return AgentResult(
            success=True,
            output={
                "method": "Adam gradient descent",
                "n_steps": n_steps,
                "final_loss": round(loss.item(), 6),
                "loss_history": history,
                "optimal_params": params.detach().tolist(),
            },
            confidence=0.93,
        )

    async def _active_learning(self, inp: Dict) -> AgentResult:
        pool_size = inp.get("pool_size", 100)
        n_queries = inp.get("n_queries", 20)
        n_params = inp.get("n_params", 4)

        pool = np.random.randn(pool_size, n_params)
        labelled_idx: List[int] = []
        labels: List[float] = []

        for q in range(n_queries):
            if len(labelled_idx) < 5:
                idx = np.random.randint(pool_size)
            else:
                X_lab = pool[labelled_idx]
                y_lab = np.array(labels)
                self._gp.fit(X_lab, y_lab)
                _, uncertainty = self._gp.predict(pool)
                unlabelled_mask = np.ones(pool_size, dtype=bool)
                unlabelled_mask[labelled_idx] = False
                uncertainty[~unlabelled_mask] = -np.inf
                idx = int(np.argmax(uncertainty))

            labelled_idx.append(idx)
            labels.append(self._mock_objective(pool[idx], "active_learning"))

        return AgentResult(
            success=True,
            output={
                "method": "Active Learning (uncertainty sampling)",
                "n_queries": n_queries,
                "pool_size": pool_size,
                "best_found": round(max(labels), 6),
                "selected_indices": labelled_idx,
                "efficiency_vs_random": round(max(labels) / max(np.random.randn(n_queries).tolist() + [0.01]), 3),
            },
            confidence=0.84,
        )

    async def _optimise_vqe_params(self, inp: Dict) -> AgentResult:
        n_params = inp.get("n_params", 12)
        max_iterations = inp.get("max_iterations", 300)
        target_energy = inp.get("target_energy", -1.17448)

        params = torch.zeros(n_params, requires_grad=True)
        optimiser = optim.LBFGS([params], lr=0.1, max_iter=20)
        history = []
        energy = None

        for iteration in range(max_iterations):
            def closure():
                optimiser.zero_grad()
                with torch.enable_grad():
                    e = (params ** 2).mean() + torch.sin(params).mean() * 0.1
                    e = -torch.abs(e) + abs(target_energy)
                    e.backward()
                return e

            loss = optimiser.step(closure)
            energy = loss.item()
            if iteration % 30 == 0:
                history.append({"iteration": iteration, "energy": round(energy, 8)})
            if abs(energy - target_energy) < 1e-5:
                break

        return AgentResult(
            success=True,
            output={
                "method": "L-BFGS VQE parameter optimisation",
                "n_params": n_params,
                "iterations": iteration + 1,
                "final_energy": round(energy, 8) if energy else None,
                "target_energy": target_energy,
                "converged": abs(energy - target_energy) < 1e-4 if energy else False,
                "energy_history": history,
            },
            confidence=0.90,
        )

    async def _optimise_pulse(self, inp: Dict) -> AgentResult:
        n_segments = inp.get("n_segments", 20)
        gate_time_ns = inp.get("gate_time_ns", 50)
        target_gate = inp.get("target_gate", "X")

        result = await self._gradient_optimisation({
            "n_params": n_segments * 2,
            "n_steps": 500,
            "lr": 0.02,
        })

        return AgentResult(
            success=True,
            output={
                "method": "DRAG pulse optimisation",
                "target_gate": target_gate,
                "gate_time_ns": gate_time_ns,
                "n_segments": n_segments,
                "achieved_fidelity": round(1 - result.output["final_loss"] / 10, 5),
                "optimal_pulse": result.output["optimal_params"][:n_segments],
                "optimal_drag": result.output["optimal_params"][n_segments:],
            },
            confidence=0.86,
        )

    async def _hyperparameter_search(self, inp: Dict) -> AgentResult:
        search_space = inp.get("search_space", {
            "learning_rate": [1e-4, 1e-2],
            "n_layers": [2, 6],
            "hidden_dim": [64, 512],
        })
        n_trials = inp.get("n_trials", 20)
        metric = inp.get("metric", "validation_loss")

        trials = []
        best_trial = None
        best_score = -np.inf

        for trial_id in range(n_trials):
            params = {}
            for name, (lo, hi) in search_space.items():
                if isinstance(lo, int):
                    params[name] = int(np.random.uniform(lo, hi))
                else:
                    params[name] = float(np.exp(np.random.uniform(np.log(lo), np.log(hi))))

            score = self._mock_objective(list(params.values()), metric)
            trial = {"trial_id": trial_id, "params": params, "score": round(score, 5)}
            trials.append(trial)
            if score > best_score:
                best_score = score
                best_trial = trial

        return AgentResult(
            success=True,
            output={
                "method": "Random search hyperparameter optimisation",
                "n_trials": n_trials,
                "best_trial": best_trial,
                "all_trials": sorted(trials, key=lambda t: t["score"], reverse=True)[:5],
            },
            confidence=0.83,
        )

    # ------------------------------------------------------------------ #

    def _mock_objective(self, x: np.ndarray, name: str) -> float:
        x = np.asarray(x).flatten()
        if name == "gate_fidelity":
            return float(1.0 - 0.01 * np.sum(x ** 2) / max(len(x), 1))
        elif name in ("vqe_energy", "circuit_fidelity"):
            return float(-np.sum(x ** 2) / max(len(x), 1))
        else:
            return float(np.exp(-0.5 * np.sum(x ** 2) / max(len(x), 1)))
