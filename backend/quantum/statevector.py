"""
Local state-vector simulator (pure NumPy, no external services).

This is the default execution engine for every circuit the platform runs.
It is exact (no noise) and practical up to roughly 20-24 qubits on a laptop.

Bit ordering follows the Qiskit convention: qubit 0 is the *rightmost*
character of a measured bitstring, and basis index ``k`` has bit ``q`` set
when qubit ``q`` is |1>.
"""
from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Dict, Iterable, List, Optional, Sequence

import numpy as np

MAX_QUBITS = 24


class SimulationError(ValueError):
    """Raised when a circuit cannot be simulated (unknown gate, bad qubit, too large)."""


@dataclass
class Operation:
    name: str
    qubits: List[int]
    params: List[float] = field(default_factory=list)


@dataclass
class Circuit:
    n_qubits: int
    ops: List[Operation] = field(default_factory=list)
    measured: List[int] = field(default_factory=list)  # qubits that are measured (order = classical bit order)

    def add(self, name: str, qubits: Sequence[int], params: Sequence[float] = ()) -> "Circuit":
        self.ops.append(Operation(name.lower(), list(qubits), [float(p) for p in params]))
        return self

    def measure_all(self) -> "Circuit":
        self.measured = list(range(self.n_qubits))
        return self


# --------------------------------------------------------------------------- #
#  Gate matrices                                                               #
# --------------------------------------------------------------------------- #

_I = np.eye(2, dtype=complex)
_X = np.array([[0, 1], [1, 0]], dtype=complex)
_Y = np.array([[0, -1j], [1j, 0]], dtype=complex)
_Z = np.array([[1, 0], [0, -1]], dtype=complex)
_H = np.array([[1, 1], [1, -1]], dtype=complex) / math.sqrt(2)
_S = np.array([[1, 0], [0, 1j]], dtype=complex)
_T = np.array([[1, 0], [0, np.exp(1j * math.pi / 4)]], dtype=complex)
_SX = 0.5 * np.array([[1 + 1j, 1 - 1j], [1 - 1j, 1 + 1j]], dtype=complex)


def _rx(t: float) -> np.ndarray:
    c, s = math.cos(t / 2), math.sin(t / 2)
    return np.array([[c, -1j * s], [-1j * s, c]], dtype=complex)


def _ry(t: float) -> np.ndarray:
    c, s = math.cos(t / 2), math.sin(t / 2)
    return np.array([[c, -s], [s, c]], dtype=complex)


def _rz(t: float) -> np.ndarray:
    return np.array([[np.exp(-1j * t / 2), 0], [0, np.exp(1j * t / 2)]], dtype=complex)


def _phase(l: float) -> np.ndarray:
    return np.array([[1, 0], [0, np.exp(1j * l)]], dtype=complex)


def _u3(theta: float, phi: float, lam: float) -> np.ndarray:
    c, s = math.cos(theta / 2), math.sin(theta / 2)
    return np.array(
        [[c, -np.exp(1j * lam) * s], [np.exp(1j * phi) * s, np.exp(1j * (phi + lam)) * c]],
        dtype=complex,
    )


def single_qubit_matrix(name: str, params: Sequence[float]) -> np.ndarray:
    p = list(params)
    fixed = {
        "id": _I, "i": _I, "x": _X, "y": _Y, "z": _Z, "h": _H,
        "s": _S, "sdg": _S.conj().T, "t": _T, "tdg": _T.conj().T,
        "sx": _SX, "sxdg": _SX.conj().T,
    }
    if name in fixed:
        return fixed[name]
    if name in ("rx", "ry", "rz", "p", "u1", "phase"):
        if len(p) != 1:
            raise SimulationError(f"Gate '{name}' needs 1 parameter, got {len(p)}")
        return {"rx": _rx, "ry": _ry, "rz": _rz}.get(name, _phase)(p[0])
    if name == "u2":
        if len(p) != 2:
            raise SimulationError("Gate 'u2' needs 2 parameters")
        return _u3(math.pi / 2, p[0], p[1])
    if name in ("u3", "u"):
        if len(p) != 3:
            raise SimulationError(f"Gate '{name}' needs 3 parameters")
        return _u3(*p)
    raise SimulationError(f"Unsupported single-qubit gate '{name}'")


# controlled versions of single-qubit gates: name -> (base gate name)
_CONTROLLED = {
    "cx": "x", "cnot": "x", "cy": "y", "cz": "z", "ch": "h",
    "crx": "rx", "cry": "ry", "crz": "rz", "cp": "p", "cu1": "u1", "cphase": "p", "cu3": "u3",
}

SUPPORTED_GATES = sorted(
    {"id", "i", "x", "y", "z", "h", "s", "sdg", "t", "tdg", "sx", "sxdg",
     "rx", "ry", "rz", "p", "u1", "u2", "u3", "u", "phase",
     "swap", "rzz", "rxx", "ryy", "ccx", "toffoli"} | set(_CONTROLLED)
)


# --------------------------------------------------------------------------- #
#  Simulator                                                                   #
# --------------------------------------------------------------------------- #

class StatevectorSimulator:
    """Exact state-vector simulation of a :class:`Circuit`."""

    def __init__(self, seed: Optional[int] = None) -> None:
        self.rng = np.random.default_rng(seed)

    # -- core ----------------------------------------------------------------
    @staticmethod
    def zero_state(n_qubits: int) -> np.ndarray:
        if n_qubits < 1:
            raise SimulationError("Circuit must have at least one qubit")
        if n_qubits > MAX_QUBITS:
            raise SimulationError(
                f"{n_qubits} qubits exceeds the local simulator limit of {MAX_QUBITS}"
            )
        psi = np.zeros(2 ** n_qubits, dtype=complex)
        psi[0] = 1.0
        return psi

    @staticmethod
    def _apply_matrix(psi: np.ndarray, n: int, matrix: np.ndarray, qubits: Sequence[int]) -> np.ndarray:
        """Apply a k-qubit unitary. ``qubits[0]`` is the most-significant index of ``matrix``."""
        k = len(qubits)
        tensor = psi.reshape([2] * n)
        axes = [n - 1 - q for q in qubits]  # axis of qubit q in the reshaped tensor
        gate = matrix.reshape([2] * (2 * k))
        tensor = np.tensordot(gate, tensor, axes=(list(range(k, 2 * k)), axes))
        # tensordot puts the gate output axes first; move them back into place
        tensor = np.moveaxis(tensor, list(range(k)), axes)
        return tensor.reshape(-1)

    def apply(self, psi: np.ndarray, n: int, op: Operation) -> np.ndarray:
        name, qs, ps = op.name, op.qubits, op.params
        for q in qs:
            if not 0 <= q < n:
                raise SimulationError(f"Gate '{name}' acts on qubit {q}, circuit has {n} qubits")
        if len(set(qs)) != len(qs):
            raise SimulationError(f"Gate '{name}' repeats a qubit: {qs}")

        if name in ("barrier",):
            return psi
        if name in _CONTROLLED:
            if len(qs) != 2:
                raise SimulationError(f"Gate '{name}' needs 2 qubits (control, target)")
            base = single_qubit_matrix(_CONTROLLED[name], ps)
            m = np.eye(4, dtype=complex)
            m[2:, 2:] = base
            return self._apply_matrix(psi, n, m, [qs[0], qs[1]])
        if name == "swap":
            m = np.eye(4, dtype=complex)[[0, 2, 1, 3]]
            return self._apply_matrix(psi, n, m, qs)
        if name in ("rzz", "rxx", "ryy"):
            if len(ps) != 1 or len(qs) != 2:
                raise SimulationError(f"Gate '{name}' needs 2 qubits and 1 parameter")
            pauli = {"rzz": _Z, "rxx": _X, "ryy": _Y}[name]
            pp = np.kron(pauli, pauli)
            m = math.cos(ps[0] / 2) * np.eye(4) - 1j * math.sin(ps[0] / 2) * pp
            return self._apply_matrix(psi, n, m, qs)
        if name in ("ccx", "toffoli"):
            if len(qs) != 3:
                raise SimulationError("Gate 'ccx' needs 3 qubits")
            m = np.eye(8, dtype=complex)
            m[[6, 7]] = m[[7, 6]]
            return self._apply_matrix(psi, n, m, qs)
        if len(qs) != 1:
            raise SimulationError(f"Gate '{name}' acts on 1 qubit, got {len(qs)}")
        return self._apply_matrix(psi, n, single_qubit_matrix(name, ps), qs)

    def statevector(self, circuit: Circuit, initial_state: Optional[np.ndarray] = None) -> np.ndarray:
        n = circuit.n_qubits
        psi = self.zero_state(n) if initial_state is None else np.asarray(initial_state, dtype=complex).copy()
        for op in circuit.ops:
            psi = self.apply(psi, n, op)
        return psi

    # -- measurement ---------------------------------------------------------
    @staticmethod
    def probabilities(psi: np.ndarray) -> np.ndarray:
        p = np.abs(psi) ** 2
        return p / p.sum()

    @staticmethod
    def marginal_probabilities(probs: np.ndarray, n: int, measured: Sequence[int]) -> Dict[str, float]:
        """Probability of each outcome on the measured qubits (Qiskit bit order)."""
        measured = list(measured) or list(range(n))
        out: Dict[str, float] = {}
        idx = np.nonzero(probs > 1e-15)[0]
        for k in idx:
            bits = "".join("1" if (k >> q) & 1 else "0" for q in reversed(measured))
            out[bits] = out.get(bits, 0.0) + float(probs[k])
        return out

    def sample_counts(self, probs: Dict[str, float], shots: int) -> Dict[str, int]:
        if shots < 1:
            raise SimulationError("shots must be >= 1")
        keys = sorted(probs)
        p = np.array([probs[k] for k in keys])
        draws = self.rng.multinomial(shots, p / p.sum())
        return {k: int(c) for k, c in zip(keys, draws) if c > 0}

    def run(self, circuit: Circuit, shots: int = 1024) -> Dict:
        psi = self.statevector(circuit)
        probs = self.probabilities(psi)
        marg = self.marginal_probabilities(probs, circuit.n_qubits, circuit.measured)
        counts = self.sample_counts(marg, shots)
        return {
            "counts": counts,
            "probabilities": {k: round(v, 10) for k, v in sorted(marg.items())},
            "shots": shots,
            "n_qubits": circuit.n_qubits,
            "measured_qubits": circuit.measured or list(range(circuit.n_qubits)),
            "engine": "numpy_statevector",
        }


def bell_circuit() -> Circuit:
    """Two-qubit Bell-state circuit used when a job is submitted without a circuit."""
    return Circuit(2).add("h", [0]).add("cx", [0, 1]).measure_all()


def circuit_from_gates(n_qubits: int, gates: Iterable, measured: Optional[Sequence[int]] = None) -> Circuit:
    """Build a Circuit from objects with ``name``, ``qubits`` and ``params`` attributes
    (e.g. ``backend.agents.circuit_designer.QuantumGate``)."""
    c = Circuit(n_qubits)
    for g in gates:
        qubits = list(getattr(g, "control_qubits", []) or []) + list(g.qubits)
        c.add(g.name, qubits, list(getattr(g, "params", []) or []))
    c.measured = list(measured) if measured else list(range(n_qubits))
    return c
