"""Local quantum execution layer: state-vector simulator, QASM reader,
Pauli Hamiltonians, VQE, qubit-characterisation fits and optional cloud adapters."""
from .statevector import Circuit, StatevectorSimulator, SimulationError, bell_circuit, circuit_from_gates
from .qasm import QasmParseError, parse_qasm
from .pauli import PauliHamiltonian, HamiltonianError, builtin_hamiltonian, h2_sto3g
from .vqe import run_vqe, VQEResult

__all__ = [
    "Circuit", "StatevectorSimulator", "SimulationError", "bell_circuit", "circuit_from_gates",
    "QasmParseError", "parse_qasm", "PauliHamiltonian", "HamiltonianError", "builtin_hamiltonian",
    "h2_sto3g", "run_vqe", "VQEResult",
]
