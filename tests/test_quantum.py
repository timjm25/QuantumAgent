"""Local simulator, QASM reader, Pauli Hamiltonians and VQE."""
import math

import numpy as np
import pytest

from backend.quantum.pauli import HamiltonianError, PauliHamiltonian, h2_sto3g, normalise_label
from backend.quantum.qasm import QasmParseError, parse_qasm
from backend.quantum.statevector import Circuit, SimulationError, StatevectorSimulator, bell_circuit
from backend.quantum.vqe import CHEMICAL_ACCURACY_HA, run_vqe


def test_bell_state_probabilities():
    out = StatevectorSimulator(seed=1).run(bell_circuit(), shots=4000)
    assert out["probabilities"] == {"00": 0.5, "11": 0.5}
    assert sum(out["counts"].values()) == 4000
    assert set(out["counts"]) <= {"00", "11"}


def test_qiskit_bit_order():
    # X on qubit 0 of 3 -> bitstring '001' (qubit 0 is rightmost)
    out = StatevectorSimulator(seed=0).run(Circuit(3).add("x", [0]).measure_all(), shots=10)
    assert out["counts"] == {"001": 10}


def test_matches_qiskit_statevector_on_random_circuit():
    qiskit = pytest.importorskip("qiskit")
    from qiskit.quantum_info import Statevector

    rng = np.random.default_rng(3)
    ours = Circuit(3)
    qc = qiskit.QuantumCircuit(3)
    for _ in range(25):
        g = rng.choice(["h", "rx", "ry", "rz", "cx", "cz", "s", "t", "swap", "u3", "crz"])
        a, b = (int(x) for x in rng.choice(3, size=2, replace=False))
        if g in ("rx", "ry", "rz"):
            th = float(rng.uniform(-3, 3)); ours.add(g, [a], [th]); getattr(qc, g)(th, a)
        elif g == "u3":
            p = [float(x) for x in rng.uniform(-3, 3, 3)]; ours.add("u3", [a], p); qc.u(*p, a)
        elif g == "crz":
            th = float(rng.uniform(-3, 3)); ours.add("crz", [a, b], [th]); qc.crz(th, a, b)
        elif g in ("cx", "cz", "swap"):
            ours.add(g, [a, b]); getattr(qc, g)(a, b)
        else:
            ours.add(g, [a]); getattr(qc, g)(a)
    psi = StatevectorSimulator().statevector(ours)
    ref = Statevector(qc).data
    assert abs(abs(np.vdot(ref, psi)) - 1) < 1e-9


def test_simulator_errors():
    sim = StatevectorSimulator()
    with pytest.raises(SimulationError):
        sim.statevector(Circuit(2).add("cx", [0, 5]))
    with pytest.raises(SimulationError):
        sim.statevector(Circuit(2).add("foo", [0]))
    with pytest.raises(SimulationError):
        sim.statevector(Circuit(1).add("rx", [0]))  # missing parameter


def test_parse_qasm2_and_qasm3():
    q2 = 'OPENQASM 2.0;\ninclude "qelib1.inc";\nqreg q[2];\ncreg c[2];\nh q[0]; // comment\ncx q[0],q[1];\nmeasure q -> c;'
    q3 = "OPENQASM 3.0;\nqubit[2] q;\nbit[2] c;\nrx(pi/2) q[0];\nrz(-0.5*pi) q[1];\nc[0] = measure q[0];\nc[1] = measure q[1];"
    c2 = parse_qasm(q2)
    assert c2.n_qubits == 2 and [o.name for o in c2.ops] == ["h", "cx"] and c2.measured == [0, 1]
    c3 = parse_qasm(q3)
    assert math.isclose(c3.ops[0].params[0], math.pi / 2)
    assert math.isclose(c3.ops[1].params[0], -math.pi / 2)


def test_parse_qasm_broadcast_and_errors():
    c = parse_qasm("OPENQASM 2.0; qreg q[3]; h q;")
    assert len(c.ops) == 3
    with pytest.raises(QasmParseError, match="unsupported gate"):
        parse_qasm("OPENQASM 2.0; qreg q[1]; magic q[0];")
    with pytest.raises(QasmParseError, match="out of range"):
        parse_qasm("OPENQASM 2.0; qreg q[1]; x q[3];")
    with pytest.raises(QasmParseError, match="parameter"):
        parse_qasm("OPENQASM 2.0; qreg q[1]; rx(__import__('os')) q[0];")
    with pytest.raises(QasmParseError):
        parse_qasm("")


def test_normalise_label_dense_and_sparse():
    assert normalise_label("xz") == "XZ"
    assert normalise_label("X2 Z0") == "XIZ"
    assert normalise_label("Z0", n_qubits=3) == "IIZ"
    with pytest.raises(HamiltonianError):
        normalise_label("Q1")


def test_expectation_matches_dense_matrix():
    rng = np.random.default_rng(5)
    ham = PauliHamiltonian.from_terms([("XYZ", 0.3), ("ZZI", -0.7), ("IYY", 0.2), ("III", 1.1)])
    psi = rng.normal(size=8) + 1j * rng.normal(size=8)
    psi /= np.linalg.norm(psi)
    dense = np.real(np.vdot(psi, ham.to_matrix() @ psi))
    assert abs(ham.expectation(psi) - dense) < 1e-12
    assert np.allclose(ham.to_matrix(), ham.to_matrix().conj().T)


def test_complex_coefficient_rejected():
    with pytest.raises(HamiltonianError, match="complex"):
        PauliHamiltonian.from_terms([("Z", 1 + 1j)])


def test_h2_exact_energy_is_fci():
    ham = h2_sto3g()
    assert abs(ham.exact_ground_energy() + ham.nuclear_repulsion - (-1.137306)) < 1e-5


def test_vqe_reaches_chemical_accuracy_on_h2():
    r = run_vqe(h2_sto3g(), layers=1)
    assert r.error_ha < CHEMICAL_ACCURACY_HA
    assert r.within_chemical_accuracy
    assert abs(r.total_energy - (-1.137306)) < 1e-4
    assert r.history and r.n_evaluations > 0


def test_vqe_with_shot_noise_is_close():
    r = run_vqe(h2_sto3g(), layers=1, shots=4000, max_iterations=150, restarts=1)
    assert r.error_ha < 0.02


def test_vqe_bad_optimiser():
    with pytest.raises(ValueError):
        run_vqe(h2_sto3g(), optimiser="ADAM")
