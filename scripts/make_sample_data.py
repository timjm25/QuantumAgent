#!/usr/bin/env python3
"""Regenerate the demo datasets in data/sample_data/.

Everything written here is either synthetic (curves from the simulated
transmon in backend/quantum/characterisation.py) or public bibliographic
metadata. Nothing is real lab or customer data.

    python3 scripts/make_sample_data.py
"""
from __future__ import annotations

import csv
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from backend.quantum.characterisation import SimulatedTransmon  # noqa: E402

OUT = ROOT / "data" / "sample_data"

PAPERS = [
    ("A variational eigenvalue solver on a photonic quantum processor",
     "Peruzzo A; McClean J; Shadbolt P; Yung M-H; Zhou X-Q; Love PJ; Aspuru-Guzik A; O'Brien JL", 2014,
     "First experimental demonstration of the variational quantum eigensolver (VQE) on a photonic chip.",
     "10.1038/ncomms5213", "Nature Communications", "VQE; variational; quantum chemistry; photonic"),
    ("A Quantum Approximate Optimization Algorithm", "Farhi E; Goldstone J; Gutmann S", 2014,
     "Introduces QAOA, a variational algorithm for combinatorial optimisation such as MaxCut.",
     "", "arXiv:1411.4028", "QAOA; optimisation; MaxCut; variational"),
    ("The theory of variational hybrid quantum-classical algorithms",
     "McClean JR; Romero J; Babbush R; Aspuru-Guzik A", 2016,
     "Theoretical framework for VQE-style hybrid algorithms, including UCC ansatz and optimisation.",
     "10.1088/1367-2630/18/2/023023", "New Journal of Physics", "VQE; variational; ansatz; UCCSD"),
    ("Hardware-efficient variational quantum eigensolver for small molecules and quantum magnets",
     "Kandala A; Mezzacapo A; Temme K; Takita M; Brink M; Chow JM; Gambetta JM", 2017,
     "Hardware-efficient ansatz VQE for H2, LiH and BeH2 on a superconducting processor.",
     "10.1038/nature23879", "Nature", "VQE; hardware-efficient ansatz; superconducting; quantum chemistry"),
    ("Error mitigation for short-depth quantum circuits", "Temme K; Bravyi S; Gambetta JM", 2017,
     "Proposes zero-noise extrapolation and probabilistic error cancellation for NISQ circuits.",
     "10.1103/PhysRevLett.119.180509", "Physical Review Letters", "error mitigation; zero-noise extrapolation; NISQ"),
    ("Quantum Computing in the NISQ era and beyond", "Preskill J", 2018,
     "Perspective on noisy intermediate-scale quantum devices and their near-term uses.",
     "10.22331/q-2018-08-06-79", "Quantum", "NISQ; quantum advantage; error correction"),
    ("Barren plateaus in quantum neural network training landscapes",
     "McClean JR; Boixo S; Smelyanskiy VN; Babbush R; Neven H", 2018,
     "Shows gradients of random parameterised circuits vanish exponentially with qubit count.",
     "10.1038/s41467-018-07090-4", "Nature Communications", "barren plateau; variational; quantum machine learning"),
    ("Surface codes: Towards practical large-scale quantum computation",
     "Fowler AG; Mariantoni M; Martinis JM; Cleland AN", 2012,
     "Tutorial review of the surface code for fault-tolerant quantum computation.",
     "10.1103/PhysRevA.86.032324", "Physical Review A", "surface code; error correction; fault tolerant"),
    ("Error mitigation extends the computational reach of a noisy quantum processor",
     "Kandala A; Temme K; Corcoles AD; Mezzacapo A; Chow JM; Gambetta JM", 2019,
     "Experimental zero-noise extrapolation improving VQE accuracy on superconducting hardware.",
     "10.1038/s41586-019-1040-7", "Nature", "error mitigation; zero-noise extrapolation; VQE"),
    ("Supervised learning with quantum-enhanced feature spaces",
     "Havlicek V; Corcoles AD; Temme K; Harrow AW; Kandala A; Chow JM; Gambetta JM", 2019,
     "Quantum kernel and variational classifiers demonstrated on a superconducting processor.",
     "10.1038/s41586-019-0980-2", "Nature", "quantum machine learning; quantum kernel; classifier"),
    ("An adaptive variational algorithm for exact molecular simulations on a quantum computer",
     "Grimsley HR; Economou SE; Barnes E; Mayhall NJ", 2019,
     "ADAPT-VQE grows the ansatz operator by operator for compact, accurate molecular circuits.",
     "10.1038/s41467-019-10988-2", "Nature Communications", "ADAPT-VQE; VQE; ansatz; quantum chemistry"),
    ("Quantum Chemistry in the Age of Quantum Computing",
     "Cao Y; Romero J; Olson JP; Degroote M; Johnson PD; Kieferova M; Kivlichan ID; Menke T; Peropadre B; "
     "Sawaya NPD; Sim S; Veis L; Aspuru-Guzik A", 2019,
     "Review of quantum algorithms for electronic structure, from phase estimation to VQE.",
     "10.1021/acs.chemrev.8b00803", "Chemical Reviews", "quantum chemistry; review; VQE; phase estimation"),
    ("Suppressing quantum errors by scaling a surface code logical qubit", "Google Quantum AI", 2023,
     "Distance-5 surface code logical qubit outperforming distance-3 on a superconducting processor.",
     "10.1038/s41586-022-05434-1", "Nature", "surface code; error correction; logical qubit; superconducting"),
]


def write_measurements() -> Path:
    path = OUT / "qubit_measurements_demo.csv"
    with path.open("w", newline="") as fh:
        w = csv.writer(fh)
        w.writerow(["qubit_id", "experiment", "x", "y", "device", "measured_at"])
        for qid in range(3):
            dev = SimulatedTransmon(qid)
            for exp, n in (("T1", 25), ("T2", 25), ("T2_echo", 20), ("RB", 15)):
                xs, ys = dev.curve(exp, n_points=n)
                for x, y in zip(xs, ys):
                    w.writerow([qid, exp, round(float(x), 3), round(float(y), 5),
                                "DEMO_simulated_transmon", "2026-01-15T09:00:00Z"])
    return path


def write_papers() -> Path:
    path = OUT / "papers_demo.csv"
    with path.open("w", newline="") as fh:
        w = csv.writer(fh)
        w.writerow(["title", "authors", "year", "abstract", "doi", "url", "venue", "keywords", "citations"])
        for title, authors, year, summary, doi, venue, kw in PAPERS:
            url = f"https://doi.org/{doi}" if doi else ("https://arxiv.org/abs/1411.4028" if "arXiv" in venue else "")
            w.writerow([title, authors, year, f"Summary (not the published abstract): {summary}",
                        doi, url, venue, kw, ""])
    return path


def write_calibration() -> Path:
    path = OUT / "device_calibration_demo.csv"
    with path.open("w", newline="") as fh:
        w = csv.writer(fh)
        w.writerow(["device", "qubit_id", "T1_us", "T2_us", "readout_error", "cx_error",
                    "gate_error_1q", "frequency_ghz", "provider", "connectivity"])
        for qid in range(5):
            d = SimulatedTransmon(qid)
            ro = 1 - sum(d.readout) / 2
            cx = round(10 * d.gate_error(**d.opt), 5) if qid < 4 else ""
            w.writerow(["DEMO_simulated_5q", qid, round(d.t1_us, 1), round(d.t2_us, 1), round(ro, 4), cx,
                        round(d.gate_error(**d.opt), 6), round(4.8 + 0.07 * qid, 3), "DEMO (synthetic)", "linear"])
    return path


if __name__ == "__main__":
    OUT.mkdir(parents=True, exist_ok=True)
    for fn in (write_measurements, write_papers, write_calibration):
        print("wrote", fn().relative_to(ROOT))
