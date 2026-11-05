"""
Qubit (Pauli-sum) Hamiltonians.

A Hamiltonian is a list of ``(label, coefficient)`` terms. Labels use the
Qiskit convention: a dense string such as ``"XZI"`` where the *rightmost*
character acts on qubit 0. Sparse labels such as ``"X2 Z1"`` (or ``"X2Z1"``)
are also accepted and converted.

Bring your own Hamiltonian by exporting the qubit operator from PySCF /
OpenFermion / Qiskit Nature to the CSV schema in
``data/templates/hamiltonian_template.csv``.
"""
from __future__ import annotations

import re
from dataclasses import dataclass, field
from typing import Dict, Iterable, List, Optional, Sequence, Tuple

import numpy as np

EXACT_DIAG_MAX_QUBITS = 14

_SPARSE_TOKEN = re.compile(r"([IXYZixyz])\s*_?(\d+)")


class HamiltonianError(ValueError):
    pass


def normalise_label(label: str, n_qubits: Optional[int] = None) -> str:
    """Return a dense Qiskit-order Pauli label for ``label``."""
    raw = str(label).strip()
    if not raw:
        raise HamiltonianError("empty Pauli label")
    compact = raw.replace(" ", "")
    if re.fullmatch(r"[IXYZixyz]+", compact) and not any(ch.isdigit() for ch in raw):
        dense = compact.upper()
        if n_qubits is not None:
            if len(dense) > n_qubits:
                raise HamiltonianError(f"label '{raw}' longer than n_qubits={n_qubits}")
            dense = dense.rjust(n_qubits, "I")
        return dense
    tokens = _SPARSE_TOKEN.findall(raw)
    if not tokens or _SPARSE_TOKEN.sub("", raw).strip(" *"):
        raise HamiltonianError(
            f"cannot read Pauli label '{raw}': use a dense string like 'XZI' or sparse form like 'X2 Z0'"
        )
    ops: Dict[int, str] = {}
    for p, q in tokens:
        q = int(q)
        if q in ops and ops[q] != "I":
            raise HamiltonianError(f"label '{raw}' acts twice on qubit {q}")
        ops[q] = p.upper()
    width = max(ops) + 1
    if n_qubits is not None:
        if width > n_qubits:
            raise HamiltonianError(f"label '{raw}' acts on qubit {width - 1} but n_qubits={n_qubits}")
        width = n_qubits
    return "".join(ops.get(q, "I") for q in reversed(range(width)))


@dataclass
class PauliHamiltonian:
    terms: List[Tuple[str, float]]
    n_qubits: int
    name: str = "custom"
    basis: str = ""
    n_electrons: Optional[int] = None
    nuclear_repulsion: float = 0.0  # added to energies when reporting *total* energy
    source: str = "user_import"
    metadata: Dict[str, str] = field(default_factory=dict)

    # -- construction -------------------------------------------------------
    @classmethod
    def from_terms(cls, terms: Iterable[Tuple[str, float]], n_qubits: Optional[int] = None, **kw) -> "PauliHamiltonian":
        terms = list(terms)
        if not terms:
            raise HamiltonianError("Hamiltonian has no terms")
        if n_qubits is None:
            n_qubits = max(len(normalise_label(lbl)) for lbl, _ in terms)
        combined: Dict[str, float] = {}
        for lbl, c in terms:
            dense = normalise_label(lbl, n_qubits)
            try:
                coeff = complex(c)
            except (TypeError, ValueError):
                raise HamiltonianError(f"coefficient for '{lbl}' is not a number: {c!r}")
            if abs(coeff.imag) > 1e-10:
                raise HamiltonianError(
                    f"coefficient for '{lbl}' is complex ({c}); a physical Hamiltonian needs real Pauli coefficients"
                )
            combined[dense] = combined.get(dense, 0.0) + coeff.real
        return cls(terms=list(combined.items()), n_qubits=n_qubits, **kw)

    # -- linear algebra -------------------------------------------------------
    @staticmethod
    def _masks(label: str) -> Tuple[int, int, int]:
        x_mask = z_mask = 0
        n_y = 0
        for q, ch in enumerate(reversed(label)):
            if ch in "XY":
                x_mask |= 1 << q
            if ch in "ZY":
                z_mask |= 1 << q
            n_y += ch == "Y"
        return x_mask, z_mask, n_y

    @staticmethod
    def _parity(values: np.ndarray) -> np.ndarray:
        v = values.copy()
        out = np.zeros_like(v)
        while np.any(v):
            out ^= v & 1
            v >>= 1
        return out

    def expectation(self, psi: np.ndarray) -> float:
        """<psi|H|psi> (electronic part; add ``nuclear_repulsion`` for the total)."""
        dim = 2 ** self.n_qubits
        if psi.shape[0] != dim:
            raise HamiltonianError(f"state has dimension {psi.shape[0]}, expected {dim}")
        idx = np.arange(dim)
        total = 0.0
        for label, coeff in self.terms:
            xm, zm, ny = self._masks(label)
            sign = 1 - 2 * self._parity(idx & zm)
            phase = (1j) ** ny
            # P|b> = i^ny (-1)^{pc(b&zm)} |b ^ xm>
            val = np.vdot(psi[idx ^ xm], phase * sign * psi)
            total += coeff * val.real
        return float(total)

    def to_matrix(self) -> np.ndarray:
        if self.n_qubits > EXACT_DIAG_MAX_QUBITS:
            raise HamiltonianError(
                f"dense matrix for {self.n_qubits} qubits is too large (limit {EXACT_DIAG_MAX_QUBITS})"
            )
        dim = 2 ** self.n_qubits
        idx = np.arange(dim)
        mat = np.zeros((dim, dim), dtype=complex)
        for label, coeff in self.terms:
            xm, zm, ny = self._masks(label)
            sign = 1 - 2 * self._parity(idx & zm)
            mat[idx ^ xm, idx] += coeff * (1j) ** ny * sign
        return mat

    def exact_ground_energy(self) -> float:
        """Exact (full-diagonalisation) lowest eigenvalue, electronic part."""
        return float(np.linalg.eigvalsh(self.to_matrix())[0])

    def summary(self) -> Dict:
        return {
            "name": self.name,
            "basis": self.basis,
            "n_qubits": self.n_qubits,
            "n_terms": len(self.terms),
            "n_electrons": self.n_electrons,
            "nuclear_repulsion": self.nuclear_repulsion,
            "source": self.source,
            "largest_terms": [
                f"{c:+.6f} {lbl}" for lbl, c in sorted(self.terms, key=lambda t: -abs(t[1]))[:8]
            ],
        }


# --------------------------------------------------------------------------- #
#  Built-in reference Hamiltonians                                             #
# --------------------------------------------------------------------------- #

def h2_sto3g() -> PauliHamiltonian:
    """H2 / STO-3G at 0.735 Angstrom, parity mapping with two-qubit reduction.

    Exact ground electronic energy -1.857275 Ha; with nuclear repulsion
    0.719969 Ha the total is -1.137306 Ha (the FCI value in this basis).
    These are the coefficients published in the Qiskit VQE tutorial.
    """
    return PauliHamiltonian.from_terms(
        [
            ("II", -1.052373245772859),
            ("IZ", 0.39793742484318045),
            ("ZI", -0.39793742484318045),
            ("ZZ", -0.01128010425623538),
            ("XX", 0.18093119978423156),
        ],
        n_qubits=2,
        name="H2",
        basis="sto-3g",
        n_electrons=2,
        nuclear_repulsion=0.7199689944489797,
        source="builtin_reference",
        metadata={"geometry": "H 0 0 0; H 0 0 0.735", "mapping": "parity, 2-qubit reduced"},
    )


BUILTIN_HAMILTONIANS = {("H2", "sto-3g"): h2_sto3g}


def builtin_hamiltonian(molecule: str, basis: str = "sto-3g") -> Optional[PauliHamiltonian]:
    factory = BUILTIN_HAMILTONIANS.get((molecule, basis.lower()))
    return factory() if factory else None
