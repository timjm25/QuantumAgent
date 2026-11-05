"""
Variational Quantum Eigensolver running on the local state-vector simulator.

The ansatz is a hardware-efficient "RY + CX ladder" (Qiskit RealAmplitudes
style); the classical optimiser is SciPy (COBYLA by default, or L-BFGS-B /
Nelder-Mead / Powell). Energies are exact expectation values (no shot noise)
unless ``shots`` is given, in which case each Pauli term is estimated from
sampled measurements in its own basis.
"""
from __future__ import annotations

import time
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Sequence

import numpy as np
from scipy.optimize import minimize

from .pauli import EXACT_DIAG_MAX_QUBITS, PauliHamiltonian
from .statevector import Circuit, StatevectorSimulator

HARTREE_TO_KCAL = 627.509474
CHEMICAL_ACCURACY_HA = 1.6e-3  # 1 kcal/mol

SUPPORTED_OPTIMISERS = ("COBYLA", "L-BFGS-B", "Nelder-Mead", "Powell", "SLSQP")


def ry_ansatz(n_qubits: int, layers: int, params: Sequence[float], initial_bits: Sequence[int] = ()) -> Circuit:
    """RY rotations on every qubit, then a linear CX ladder, repeated ``layers`` times,
    followed by a final RY layer. Number of parameters = n_qubits * (layers + 1)."""
    expected = n_qubits * (layers + 1)
    if len(params) != expected:
        raise ValueError(f"ansatz needs {expected} parameters, got {len(params)}")
    c = Circuit(n_qubits)
    for q in initial_bits:
        c.add("x", [q])
    k = 0
    for _ in range(layers):
        for q in range(n_qubits):
            c.add("ry", [q], [params[k]])
            k += 1
        for q in range(n_qubits - 1):
            c.add("cx", [q, q + 1])
    for q in range(n_qubits):
        c.add("ry", [q], [params[k]])
        k += 1
    return c


@dataclass
class VQEResult:
    energy: float                     # electronic energy (Ha)
    total_energy: float               # + nuclear repulsion
    params: List[float]
    n_evaluations: int
    iterations: int
    converged: bool
    optimiser: str
    layers: int
    exact_energy: Optional[float] = None
    exact_total_energy: Optional[float] = None
    error_ha: Optional[float] = None
    history: List[float] = field(default_factory=list)
    runtime_s: float = 0.0
    shots: Optional[int] = None
    message: str = ""

    @property
    def within_chemical_accuracy(self) -> Optional[bool]:
        return None if self.error_ha is None else self.error_ha < CHEMICAL_ACCURACY_HA

    def to_dict(self) -> Dict:
        return {
            "energy": round(self.energy, 10),
            "total_energy": round(self.total_energy, 10),
            "exact_energy": None if self.exact_energy is None else round(self.exact_energy, 10),
            "exact_total_energy": None if self.exact_total_energy is None else round(self.exact_total_energy, 10),
            "error_ha": None if self.error_ha is None else round(self.error_ha, 10),
            "error_kcal": None if self.error_ha is None else round(self.error_ha * HARTREE_TO_KCAL, 6),
            "within_chemical_accuracy": self.within_chemical_accuracy,
            "params": [round(p, 8) for p in self.params],
            "n_params": len(self.params),
            "n_evaluations": self.n_evaluations,
            "iterations": self.iterations,
            "converged": self.converged,
            "optimiser": self.optimiser,
            "ansatz": f"ry_cx_linear(layers={self.layers})",
            "history": [round(e, 8) for e in self.history[-50:]],
            "runtime_s": round(self.runtime_s, 3),
            "shots": self.shots,
            "message": self.message,
            "engine": "numpy_statevector",
        }


def _sampled_expectation(ham: PauliHamiltonian, psi: np.ndarray, shots: int, rng: np.random.Generator) -> float:
    """Estimate <H> with ``shots`` measurements per non-identity Pauli term."""
    sim = StatevectorSimulator()
    n = ham.n_qubits
    total = 0.0
    for label, coeff in ham.terms:
        if set(label) == {"I"}:
            total += coeff
            continue
        rot = Circuit(n)
        for q, ch in enumerate(reversed(label)):
            if ch == "X":
                rot.add("h", [q])
            elif ch == "Y":
                rot.add("sdg", [q]).add("h", [q])
        phi = sim.statevector(rot, initial_state=psi)
        probs = np.abs(phi) ** 2
        probs /= probs.sum()
        outcomes = rng.choice(len(probs), size=shots, p=probs)
        mask = sum(1 << q for q, ch in enumerate(reversed(label)) if ch != "I")
        parity = np.array([bin(int(o) & mask).count("1") & 1 for o in outcomes])
        total += coeff * float(np.mean(1 - 2 * parity))
    return total


def run_vqe(
    ham: PauliHamiltonian,
    layers: int = 2,
    optimiser: str = "COBYLA",
    max_iterations: int = 500,
    initial_params: Optional[Sequence[float]] = None,
    initial_bits: Sequence[int] = (),
    seed: int = 7,
    restarts: int = 2,
    shots: Optional[int] = None,
    tol: float = 1e-8,
) -> VQEResult:
    """Minimise <H> over the RY ansatz. Best of ``restarts`` random starts is returned."""
    if optimiser not in SUPPORTED_OPTIMISERS:
        raise ValueError(f"optimiser must be one of {SUPPORTED_OPTIMISERS}")
    if layers < 1:
        raise ValueError("layers must be >= 1")
    sim = StatevectorSimulator()
    rng = np.random.default_rng(seed)
    n = ham.n_qubits
    n_params = n * (layers + 1)
    t0 = time.perf_counter()

    best = None
    total_evals = 0
    starts = [np.asarray(initial_params, float)] if initial_params is not None else []
    while len(starts) < max(1, restarts):
        starts.append(rng.uniform(-np.pi, np.pi, n_params) * (0.1 if not starts else 1.0))

    for x0 in starts:
        history: List[float] = []

        def cost(theta: np.ndarray) -> float:
            psi = sim.statevector(ry_ansatz(n, layers, theta, initial_bits))
            e = _sampled_expectation(ham, psi, shots, rng) if shots else ham.expectation(psi)
            history.append(e)
            return e

        opts = {"maxiter": max_iterations}
        res = minimize(cost, x0, method=optimiser, tol=tol, options=opts)
        total_evals += len(history)
        if best is None or res.fun < best[0].fun:
            best = (res, list(history))

    res, history = best
    exact = None
    if n <= EXACT_DIAG_MAX_QUBITS:
        exact = ham.exact_ground_energy()
    energy = float(res.fun)
    if shots:
        # report the noiseless energy of the optimal parameters as well
        energy = ham.expectation(sim.statevector(ry_ansatz(n, layers, res.x, initial_bits)))
    error = None if exact is None else abs(energy - exact)
    return VQEResult(
        energy=energy,
        total_energy=energy + ham.nuclear_repulsion,
        params=[float(p) for p in res.x],
        n_evaluations=total_evals,
        iterations=int(getattr(res, "nit", len(history)) or len(history)),
        converged=bool(res.success) or (error is not None and error < CHEMICAL_ACCURACY_HA),
        optimiser=optimiser,
        layers=layers,
        exact_energy=exact,
        exact_total_energy=None if exact is None else exact + ham.nuclear_repulsion,
        error_ha=error,
        history=history,
        runtime_s=time.perf_counter() - t0,
        shots=shots,
        message=str(getattr(res, "message", "")),
    )
