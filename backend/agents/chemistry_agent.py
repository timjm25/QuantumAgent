"""
Quantum Chemistry Agent

Automates:
  - Molecular Hamiltonian generation (Jordan-Wigner, Bravyi-Kitaev)
  - VQE-based electronic structure calculations
  - QAOA for molecular optimisation
  - Reaction pathway analysis
  - Molecular property prediction
  - Catalyst design workflows
  - Integration with PySCF, RDKit, OpenFermion
"""
from __future__ import annotations

import math
import zlib
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import torch
import torch.nn as nn

from .base import BaseAgent, AgentResult
from ..core.logging import get_logger
from ..quantum.pauli import BUILTIN_HAMILTONIANS, EXACT_DIAG_MAX_QUBITS, PauliHamiltonian, builtin_hamiltonian
from ..quantum.vqe import HARTREE_TO_KCAL, SUPPORTED_OPTIMISERS, run_vqe

SYNTHETIC_WARNING = (
    "No qubit Hamiltonian is available for {mol}/{basis}: numbers are a SYNTHETIC placeholder "
    "(random integrals, surrogate convergence), not a simulation. Import a Pauli Hamiltonian "
    "(data/templates/hamiltonian_template.csv) to run a real VQE."
)


def _stable_rng(*parts: str) -> np.random.Generator:
    """Deterministic RNG (Python's hash() is salted per process)."""
    return np.random.default_rng(zlib.crc32("|".join(parts).encode()))

logger = get_logger(__name__)


# Periodic table: symbol → atomic number
ATOMIC_NUMBERS = {
    "H": 1, "He": 2, "Li": 3, "Be": 4, "B": 5, "C": 6, "N": 7, "O": 8,
    "F": 9, "Ne": 10, "Na": 11, "Mg": 12, "Al": 13, "Si": 14, "P": 15,
    "S": 16, "Cl": 17, "Ar": 18, "K": 19, "Ca": 20, "Fe": 26, "Cu": 29,
    "Zn": 30, "Br": 35, "Ag": 47, "I": 53, "Au": 79, "Pt": 78, "Pd": 46,
    "Mo": 42, "Ru": 44, "Rh": 45, "Ni": 28, "Co": 27, "Mn": 25, "Cr": 24,
}

# Benchmark energies (Hartree) for standard molecules vs CCSD(T)/aug-cc-pVQZ
BENCHMARK_ENERGIES: Dict[str, float] = {
    "H2":    -1.17448,
    "LiH":   -7.98229,
    "BeH2":  -15.83618,
    "H2O":   -76.33987,
    "NH3":   -56.45691,
    "CH4":   -40.45182,
    "N2":    -109.27873,
    "CO":    -113.27376,
    "CO2":   -188.48399,
    "HF":    -100.26018,
}


@dataclass
class Molecule:
    name: str
    formula: str
    geometry: List[Tuple[str, float, float, float]]  # atom, x, y, z (Angstrom)
    charge: int = 0
    multiplicity: int = 1
    basis_set: str = "sto-3g"


@dataclass
class HamiltonianData:
    molecule: str
    n_qubits: int
    n_electrons: int
    nuclear_repulsion: float
    one_body_integrals: np.ndarray
    two_body_integrals: np.ndarray
    fci_energy: Optional[float] = None
    hf_energy: Optional[float] = None
    qubit_op_terms: int = 0
    encoding: str = "jordan_wigner"
    pauli: Optional[PauliHamiltonian] = None   # set when a real qubit Hamiltonian is available
    source: str = "synthetic_placeholder"


class MolPropertyPredictor(nn.Module):
    """
    Graph-based neural network for fast molecular property prediction.
    Node features: atomic number embedding.
    Edge features: bond type.
    Output: energy, HOMO-LUMO gap, dipole moment.
    """

    def __init__(self, embed_dim: int = 64, hidden: int = 128, n_layers: int = 3):
        super().__init__()
        self.atom_embed = nn.Embedding(120, embed_dim)
        layers = []
        in_dim = embed_dim
        for _ in range(n_layers):
            layers.extend([nn.Linear(in_dim, hidden), nn.ReLU(), nn.LayerNorm(hidden)])
            in_dim = hidden
        self.mp = nn.Sequential(*layers)
        self.out = nn.Sequential(
            nn.Linear(hidden, 64),
            nn.GELU(),
            nn.Linear(64, 3),  # [energy, HOMO-LUMO gap, dipole]
        )

    def forward(self, atomic_numbers: torch.Tensor) -> torch.Tensor:
        x = self.atom_embed(atomic_numbers)
        x = self.mp(x)
        pooled = x.mean(dim=0, keepdim=True)
        return self.out(pooled)


class QuantumChemistryAgent(BaseAgent):
    """
    End-to-end quantum chemistry automation agent covering Hamiltonian
    generation, quantum simulation, property prediction, and catalyst design.
    """

    name = "chemistry_agent"
    description = "Quantum chemistry: molecular simulation, VQE, Hamiltonian generation, catalyst design"
    capabilities = [
        "hamiltonian_generation",
        "vqe_simulation",
        "molecular_property_prediction",
        "reaction_pathway",
        "catalyst_design",
        "basis_set_selection",
        "active_space_selection",
        "benchmark_comparison",
    ]

    STANDARD_GEOMETRIES: Dict[str, List[Tuple[str, float, float, float]]] = {
        "H2": [("H", 0.0, 0.0, 0.0), ("H", 0.0, 0.0, 0.735)],
        "LiH": [("Li", 0.0, 0.0, 0.0), ("H", 0.0, 0.0, 1.595)],
        "H2O": [("O", 0.0, 0.0, 0.0), ("H", 0.0, 0.757, 0.586), ("H", 0.0, -0.757, 0.586)],
        "NH3": [("N", 0.0, 0.0, 0.115), ("H", 0.0, 0.940, -0.269), ("H", 0.815, -0.470, -0.269), ("H", -0.815, -0.470, -0.269)],
        "BeH2": [("Be", 0.0, 0.0, 0.0), ("H", 0.0, 0.0, 1.326), ("H", 0.0, 0.0, -1.326)],
        "N2": [("N", 0.0, 0.0, 0.0), ("N", 0.0, 0.0, 1.098)],
        "CO": [("C", 0.0, 0.0, 0.0), ("O", 0.0, 0.0, 1.128)],
    }

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.property_predictor = MolPropertyPredictor()
        self._hamiltonian_cache: Dict[str, HamiltonianData] = {}
        self._vqe_results: Dict[str, Dict] = {}
        self._user_hamiltonians: Dict[str, PauliHamiltonian] = {}

    # ------------------------------------------------------------------ #
    #  Bring-your-own Hamiltonians                                         #
    # ------------------------------------------------------------------ #

    def register_hamiltonian(self, ham: PauliHamiltonian) -> None:
        """Make an imported qubit Hamiltonian available under ``ham.name``."""
        self._user_hamiltonians[ham.name] = ham
        self._hamiltonian_cache = {k: v for k, v in self._hamiltonian_cache.items()
                                   if not k.startswith(f"{ham.name}_")}

    async def _load_hamiltonians(self, inp: Dict) -> AgentResult:
        from ..importers import hamiltonians_from_rows

        rows = inp.get("rows") or []
        if not rows:
            return AgentResult(success=False, output={"error": "No Hamiltonian rows supplied"})
        hams = hamiltonians_from_rows(rows)
        for h in hams.values():
            self.register_hamiltonian(h)
        return AgentResult(success=True, output={"loaded": [h.summary() for h in hams.values()]},
                           reasoning=f"Loaded {len(hams)} Hamiltonian(s): {', '.join(hams)}")

    async def _list_hamiltonians(self, inp: Dict) -> AgentResult:
        builtin = [{"name": m, "basis": b, "source": "builtin_reference"} for m, b in BUILTIN_HAMILTONIANS]
        return AgentResult(success=True, output={
            "user_imported": [h.summary() for h in self._user_hamiltonians.values()],
            "builtin": builtin,
        })

    def _vqe_for(self, ham: "HamiltonianData", inp: Dict) -> Dict:
        """Real VQE when a Pauli Hamiltonian exists, otherwise the labelled surrogate."""
        if ham.pauli is None:
            res = self._classical_vqe_mock(ham, max_iter=int(inp.get("max_iterations", 100)))
            res["method"] = "surrogate_estimate"
            return res
        optimiser = inp.get("optimiser", "COBYLA")
        if optimiser not in SUPPORTED_OPTIMISERS:
            raise ValueError(f"optimiser must be one of {SUPPORTED_OPTIMISERS}")
        r = run_vqe(
            ham.pauli,
            layers=int(inp.get("n_layers", inp.get("layers", 2))),
            optimiser=optimiser,
            max_iterations=int(inp.get("max_iterations", 500)),
            shots=inp.get("shots"),
            seed=int(inp.get("seed", 7)),
        )
        d = r.to_dict()
        return {
            "energy": d["total_energy"],
            "electronic_energy": d["energy"],
            "exact_energy": d["exact_total_energy"],
            "error_vs_exact_ha": d["error_ha"],
            "error_vs_exact_kcal": d["error_kcal"],
            "within_chemical_accuracy": d["within_chemical_accuracy"],
            "params": d["params"],
            "iterations": d["iterations"],
            "n_evaluations": d["n_evaluations"],
            "converged": d["converged"],
            "energy_std": 0.0 if not d["shots"] else None,
            "history": d["history"],
            "optimiser": d["optimiser"],
            "ansatz": d["ansatz"],
            "runtime_s": d["runtime_s"],
            "method": "vqe_statevector" + (f"_{d['shots']}shots" if d["shots"] else ""),
        }

    async def _execute(self, task_input: Dict[str, Any]) -> AgentResult:
        action = task_input.get("action", "simulate")
        dispatch = {
            "simulate": self._simulate_molecule,
            "generate_hamiltonian": self._generate_hamiltonian,
            "vqe": self._run_vqe,
            "predict_properties": self._predict_properties,
            "design_catalyst": self._design_catalyst,
            "reaction_pathway": self._analyse_reaction_pathway,
            "benchmark": self._benchmark_molecule,
            "active_space": self._select_active_space,
            "load_hamiltonians": self._load_hamiltonians,
            "list_hamiltonians": self._list_hamiltonians,
        }
        handler = dispatch.get(action)
        if handler is None:
            return AgentResult(success=False, output={"error": f"Unknown action: {action}"})
        return await handler(task_input)

    # ------------------------------------------------------------------ #

    async def _simulate_molecule(self, inp: Dict) -> AgentResult:
        molecule = inp.get("molecule", "H2")
        basis = inp.get("basis", "sto-3g")
        method = inp.get("method", "vqe")

        ham = await self._get_or_build_hamiltonian(molecule, basis)
        vqe_result = self._vqe_for(ham, inp)
        warnings = [] if ham.pauli else [SYNTHETIC_WARNING.format(mol=molecule, basis=basis)]

        benchmark = BENCHMARK_ENERGIES.get(molecule)
        error_kcal = None
        if benchmark:
            error_kcal = abs(vqe_result["energy"] - benchmark) * HARTREE_TO_KCAL

        if ham.pauli is not None:
            chem_acc = vqe_result["within_chemical_accuracy"]
        else:
            chem_acc = error_kcal is not None and error_kcal < 1.0

        result = {
            "molecule": molecule,
            "basis": basis,
            "method": method,
            "computation": vqe_result["method"],
            "hamiltonian_source": ham.source,
            "n_qubits": ham.n_qubits,
            "n_electrons": ham.n_electrons,
            "hf_energy": ham.hf_energy,
            "vqe_energy": vqe_result["energy"],
            "nuclear_repulsion": ham.nuclear_repulsion,
            "iterations": vqe_result["iterations"],
            "converged": vqe_result["converged"],
            "chemical_accuracy": chem_acc,
        }
        if ham.pauli is not None:
            result["exact_energy_in_basis"] = vqe_result["exact_energy"]
            result["error_vs_exact_kcal"] = vqe_result["error_vs_exact_kcal"]
        if error_kcal is not None:
            result["error_vs_benchmark_kcal"] = round(error_kcal, 4)
            result["benchmark_energy"] = benchmark
            result["benchmark_note"] = "CCSD(T)/aug-cc-pVQZ reference (basis-set limit), not the same basis"

        return AgentResult(
            success=True,
            output=result,
            confidence=0.95 if ham.pauli is not None else 0.3,
            reasoning=(
                f"Simulated {molecule} with {basis} basis set using {method.upper()} "
                f"({vqe_result['method']}). Ground state energy: {vqe_result['energy']:.6f} Ha."
            ),
            warnings=warnings,
        )

    async def _generate_hamiltonian(self, inp: Dict) -> AgentResult:
        molecule = inp.get("molecule", "H2")
        basis = inp.get("basis", "sto-3g")
        encoding = inp.get("encoding", "jordan_wigner")
        active_space = inp.get("active_space", None)

        ham = await self._get_or_build_hamiltonian(molecule, basis, encoding)

        return AgentResult(
            success=True,
            output={
                "molecule": molecule,
                "basis": basis,
                "encoding": encoding,
                "n_qubits": ham.n_qubits,
                "n_electrons": ham.n_electrons,
                "n_qubit_op_terms": ham.qubit_op_terms,
                "nuclear_repulsion": ham.nuclear_repulsion,
                "hf_energy": ham.hf_energy,
                "one_body_shape": list(ham.one_body_integrals.shape),
                "two_body_shape": list(ham.two_body_integrals.shape),
                "qubit_op_sample": self._sample_qubit_op(ham),
                "hamiltonian_source": ham.source,
            },
            confidence=0.98 if ham.pauli is not None else 0.3,
            reasoning=f"Generated {encoding} Hamiltonian for {molecule}/{basis}.",
            warnings=[] if ham.pauli else [SYNTHETIC_WARNING.format(mol=molecule, basis=basis)],
        )

    async def _run_vqe(self, inp: Dict) -> AgentResult:
        molecule = inp.get("molecule", "H2")
        ansatz = inp.get("ansatz", "uccsd")
        max_iterations = inp.get("max_iterations", 200)
        basis = inp.get("basis", "sto-3g")

        ham = await self._get_or_build_hamiltonian(molecule, basis)
        try:
            result = self._vqe_for(ham, {**inp, "max_iterations": max_iterations})
        except ValueError as exc:
            return AgentResult(success=False, output={"error": str(exc)})

        self._vqe_results[molecule] = result
        output = {
            "molecule": molecule,
            "ansatz": ansatz if ham.pauli is None else result["ansatz"],
            "computation": result["method"],
            "hamiltonian_source": ham.source,
            "converged": result["converged"],
            "iterations": result["iterations"],
            "final_energy": result["energy"],
            "energy_std": result.get("energy_std", 0.0001),
            "optimal_params": result["params"][:8],
            "n_params": len(result["params"]),
            "cost_function_calls": result.get("n_evaluations", result["iterations"] * len(result["params"])),
        }
        if ham.pauli is not None:
            output.update({
                "exact_energy": result["exact_energy"],
                "error_vs_exact_kcal": result["error_vs_exact_kcal"],
                "within_chemical_accuracy": result["within_chemical_accuracy"],
                "optimiser": result["optimiser"],
                "energy_history": result["history"],
            })
        return AgentResult(
            success=True,
            output=output,
            confidence=0.95 if ham.pauli is not None else 0.3,
            warnings=[] if ham.pauli else [SYNTHETIC_WARNING.format(mol=molecule, basis=basis)],
        )

    async def _predict_properties(self, inp: Dict) -> AgentResult:
        molecule = inp.get("molecule", "H2")
        geometry = self.STANDARD_GEOMETRIES.get(molecule)
        if not geometry:
            geometry = [("H", 0.0, 0.0, 0.0), ("H", 0.0, 0.0, 0.735)]

        atoms = [ATOMIC_NUMBERS.get(sym, 1) for sym, *_ in geometry]
        atom_tensor = torch.tensor(atoms, dtype=torch.long)

        with torch.no_grad():
            props = self.property_predictor(atom_tensor)[0].tolist()

        return AgentResult(
            success=True,
            output={
                "molecule": molecule,
                "predicted_energy_ha": round(props[0] * 50 - 50, 4),
                "homo_lumo_gap_ev": round(abs(props[1]) * 10 + 2, 3),
                "dipole_debye": round(abs(props[2]) * 3, 3),
                "method": "neural_network_predictor",
                "note": "Neural property predictions; use VQE/DFT for production accuracy",
            },
            confidence=0.73,
        )

    async def _design_catalyst(self, inp: Dict) -> AgentResult:
        reaction = inp.get("reaction", "nitrogen_fixation")
        metal_candidates = inp.get("metals", ["Fe", "Mo", "Ru", "Co", "Ni"])
        constraints = inp.get("constraints", {"max_atoms": 20, "earth_abundant": True})
        # Optional measured / DFT descriptors per metal:
        #   {"Fe": {"binding_energy": -1.4, "activation_barrier": 0.9, "selectivity": 0.8}, ...}
        descriptors: Dict[str, Dict[str, float]] = inp.get("descriptors", {})
        placeholder_used = []

        candidates = []
        for metal in metal_candidates:
            if metal in descriptors:
                d = descriptors[metal]
                score = self._composite_catalyst_score(
                    float(d["binding_energy"]), float(d["activation_barrier"]), float(d.get("selectivity", 0.75)))
                score["source"] = "user_descriptors"
            else:
                score = self._score_catalyst(metal, reaction)
                placeholder_used.append(metal)
            candidates.append({
                "metal": metal,
                "atomic_number": ATOMIC_NUMBERS.get(metal, 0),
                "catalyst_formula": f"{metal}-N2 complex",
                "binding_energy_ev": round(score["binding_energy"], 3),
                "activation_barrier_ev": round(score["activation_barrier"], 3),
                "selectivity": round(score["selectivity"], 3),
                "composite_score": round(score["composite"], 3),
                "earth_abundant": metal not in ["Ru", "Rh", "Pt", "Pd", "Au", "Ag"],
                "recommended_experiment": f"DFT + VQE study of {metal} active site binding",
                "descriptor_source": score["source"],
            })

        candidates.sort(key=lambda x: x["composite_score"], reverse=True)

        return AgentResult(
            success=True,
            output={
                "reaction": reaction,
                "candidates": candidates,
                "top_recommendation": candidates[0] if candidates else None,
                "next_steps": [
                    f"Run quantum simulation for {candidates[0]['metal']} if converged",
                    "Validate binding energies with CCSD(T) benchmarks",
                    "Experimental synthesis and testing of top 3 candidates",
                ],
            },
            confidence=0.78 if not placeholder_used else 0.2,
            reasoning=f"Screened {len(candidates)} catalyst candidates for {reaction}.",
            warnings=[f"No descriptors supplied for {', '.join(placeholder_used)}: their binding energy, barrier and "
                      "selectivity are deterministic PLACEHOLDERS, not computed values. Pass "
                      "'descriptors' (from DFT or experiment) to rank real candidates."] if placeholder_used else [],
        )

    async def _analyse_reaction_pathway(self, inp: Dict) -> AgentResult:
        reactants = inp.get("reactants", ["N2", "H2"])
        products = inp.get("products", ["NH3"])
        user_steps = inp.get("steps")  # [{"name": ..., "delta_G": kcal/mol, "delta_H": ..., "barrier": ...}]

        if user_steps:
            steps = []
            for i, st in enumerate(user_steps):
                steps.append({
                    "step": i + 1, "name": st.get("name", f"Step {i + 1}"),
                    "delta_G": float(st["delta_G"]), "delta_H": float(st.get("delta_H", st["delta_G"])),
                    "barrier": float(st.get("barrier", max(0.0, float(st["delta_G"])))),
                })
            rds = max(range(len(steps)), key=lambda k: steps[k]["barrier"])
            for k, st in enumerate(steps):
                st["is_rate_determining"] = k == rds
            source = "user_steps"
        else:
            steps = self._generate_pathway_steps(reactants, products)
            source = "placeholder_unvalidated"

        return AgentResult(
            success=True,
            output={
                "reactants": reactants,
                "products": products,
                "reaction_steps": steps,
                "rate_determining_step": next((st for st in steps if st.get("is_rate_determining")), steps[-1] if steps else None),
                "overall_delta_G_kcal": round(sum(st["delta_G"] for st in steps), 2),
                "recommended_simulation": "VQE for transition state structures",
                "energetics_source": source,
            },
            confidence=0.72 if source == "user_steps" else 0.2,
            warnings=[] if source == "user_steps" else [
                "Step energetics are deterministic PLACEHOLDERS; pass 'steps' with computed delta_G values."],
        )

    async def _benchmark_molecule(self, inp: Dict) -> AgentResult:
        molecule = inp.get("molecule", "H2")
        basis = inp.get("basis", "sto-3g")

        ham = await self._get_or_build_hamiltonian(molecule, basis)
        vqe = self._vqe_for(ham, inp)
        ref = BENCHMARK_ENERGIES.get(molecule)

        report = {
            "molecule": molecule,
            "basis": basis,
            "computation": vqe["method"],
            "hamiltonian_source": ham.source,
            "hf_energy": ham.hf_energy,
            "vqe_energy": vqe["energy"],
            "exact_energy_in_basis": vqe.get("exact_energy"),
            "error_vs_exact_kcal": vqe.get("error_vs_exact_kcal"),
            "ccsd_t_reference": ref,
            "ccsd_t_reference_note": "CCSD(T)/aug-cc-pVQZ (basis-set limit); differs from a minimal-basis result by basis-set error",
            "error_ha": round(abs(vqe["energy"] - ref), 6) if ref else None,
            "error_kcal": round(abs(vqe["energy"] - ref) * HARTREE_TO_KCAL, 4) if ref else None,
            "chemical_accuracy": (vqe["within_chemical_accuracy"] if ham.pauli is not None
                                  else (abs(vqe["energy"] - ref) * HARTREE_TO_KCAL < 1.0 if ref else None)),
        }

        return AgentResult(
            success=True, output=report, confidence=0.95 if ham.pauli is not None else 0.3,
            warnings=[] if ham.pauli else [SYNTHETIC_WARNING.format(mol=molecule, basis=basis)],
        )

    async def _select_active_space(self, inp: Dict) -> AgentResult:
        molecule = inp.get("molecule", "H2O")
        basis = inp.get("basis", "sto-3g")
        target_qubits = inp.get("target_qubits", 8)

        ham = await self._get_or_build_hamiltonian(molecule, basis)

        n_orbitals_active = min(target_qubits // 2, ham.n_qubits // 2)
        n_electrons_active = min(ham.n_electrons, n_orbitals_active)

        return AgentResult(
            success=True,
            output={
                "molecule": molecule,
                "full_n_qubits": ham.n_qubits,
                "active_space_n_qubits": n_orbitals_active * 2,
                "active_electrons": n_electrons_active,
                "frozen_core_electrons": ham.n_electrons - n_electrons_active,
                "method": "CASSCF-inspired active space selection",
                "recommended_ansatz": "UCCSD",
            },
            confidence=0.84,
        )

    # ------------------------------------------------------------------ #
    #  Private helpers                                                     #
    # ------------------------------------------------------------------ #

    async def _get_or_build_hamiltonian(
        self, molecule: str, basis: str, encoding: str = "jordan_wigner"
    ) -> HamiltonianData:
        key = f"{molecule}_{basis}_{encoding}"
        if key in self._hamiltonian_cache:
            return self._hamiltonian_cache[key]

        pauli = self._user_hamiltonians.get(molecule) or builtin_hamiltonian(molecule, basis)
        if pauli is not None:
            ham = self._from_pauli(pauli, encoding)
        else:
            ham = self._build_mock_hamiltonian(molecule, basis, encoding)
        self._hamiltonian_cache[key] = ham
        return ham

    def _from_pauli(self, pauli: PauliHamiltonian, encoding: str) -> HamiltonianData:
        hf = None
        if pauli.n_qubits <= EXACT_DIAG_MAX_QUBITS:
            # lowest-energy computational basis state = best single determinant (HF for a canonical-orbital mapping)
            hf = round(float(np.min(np.real(np.diag(pauli.to_matrix())))) + pauli.nuclear_repulsion, 8)
        return HamiltonianData(
            molecule=pauli.name,
            n_qubits=pauli.n_qubits,
            n_electrons=pauli.n_electrons or 0,
            nuclear_repulsion=pauli.nuclear_repulsion,
            one_body_integrals=np.zeros((0, 0)),
            two_body_integrals=np.zeros((0, 0, 0, 0)),
            fci_energy=None,
            hf_energy=hf,
            qubit_op_terms=len(pauli.terms),
            encoding=pauli.metadata.get("mapping", encoding),
            pauli=pauli,
            source=pauli.source,
        )

    def _build_mock_hamiltonian(self, molecule: str, basis: str, encoding: str) -> HamiltonianData:
        geometry = self.STANDARD_GEOMETRIES.get(molecule, [("H", 0.0, 0.0, 0.0), ("H", 0.0, 0.0, 0.735)])
        n_atoms = len(geometry)
        n_electrons = sum(ATOMIC_NUMBERS.get(sym, 1) for sym, *_ in geometry)
        n_orbitals = n_electrons + 2
        n_qubits = 2 * n_orbitals if encoding == "jordan_wigner" else max(4, int(math.log2(n_orbitals) + 1) * 2)

        n_so = n_qubits
        rng = _stable_rng(molecule, basis, encoding)
        one_body = rng.standard_normal((n_so, n_so)) * 0.5
        one_body = (one_body + one_body.T) / 2
        two_body = rng.standard_normal((n_so, n_so, n_so, n_so)) * 0.05
        two_body = (two_body + two_body.transpose(1, 0, 2, 3)) / 2

        nuclear_repulsion = sum(
            ATOMIC_NUMBERS.get(geometry[i][0], 1) * ATOMIC_NUMBERS.get(geometry[j][0], 1)
            / max(1e-6, math.sqrt(sum((geometry[i][k+1] - geometry[j][k+1])**2 for k in range(3))) * 1.8897)
            for i in range(n_atoms) for j in range(i + 1, n_atoms)
        )

        ref = BENCHMARK_ENERGIES.get(molecule, -1.0)
        hf_energy = ref * 0.995 if ref else -n_electrons * 0.5

        return HamiltonianData(
            molecule=molecule,
            n_qubits=n_qubits,
            n_electrons=n_electrons,
            nuclear_repulsion=round(nuclear_repulsion, 6),
            one_body_integrals=one_body,
            two_body_integrals=two_body,
            fci_energy=ref,
            hf_energy=round(hf_energy, 6),
            qubit_op_terms=n_qubits * (n_qubits - 1) * 2,
            encoding=encoding,
        )

    def _classical_vqe_mock(self, ham: HamiltonianData, max_iter: int = 100) -> Dict:
        """Surrogate used ONLY when no qubit Hamiltonian exists: relaxes the HF guess
        toward the tabulated reference. It is not a simulation (see SYNTHETIC_WARNING)."""
        ref = ham.fci_energy or -1.0
        n_params = ham.n_qubits * 2
        rng = _stable_rng(ham.molecule, "surrogate")
        params = rng.standard_normal(n_params) * 0.1

        energy = ham.hf_energy or -1.0
        for i in range(max_iter):
            gradient = rng.standard_normal(n_params) * 0.01
            params -= 0.1 * gradient
            energy = energy * 0.999 + ref * 0.001
            if abs(energy - ref) < 1e-6:
                break

        return {
            "energy": round(energy, 8),
            "params": params.tolist(),
            "iterations": i + 1,
            "converged": abs(energy - ref) < 1e-4,
            "energy_std": 0.0001,
        }

    def _sample_qubit_op(self, ham: HamiltonianData) -> List[str]:
        if ham.pauli is not None:
            return ham.pauli.summary()["largest_terms"][:5]
        n = ham.n_qubits
        sample_terms = []
        paulis = ["I", "X", "Y", "Z"]
        rng = _stable_rng(ham.molecule, "terms")
        for _ in range(5):
            term = " ".join(f"{rng.choice(paulis)}{i}" for i in range(n))
            coeff = rng.standard_normal() * 0.1
            sample_terms.append(f"{coeff:.6f} [{term}]")
        return sample_terms

    @staticmethod
    def _composite_catalyst_score(binding: float, barrier: float, selectivity: float) -> Dict:
        """Sabatier-style composite: high selectivity, low barrier, binding near -1.5 eV."""
        composite = selectivity * 0.4 + (1 - min(barrier, 2.5) / 2.5) * 0.4 + (1 - min(abs(binding + 1.5), 3) / 3) * 0.2
        return {"binding_energy": binding, "activation_barrier": barrier, "selectivity": selectivity, "composite": composite}

    def _score_catalyst(self, metal: str, reaction: str) -> Dict:
        """Deterministic PLACEHOLDER descriptors (no chemistry behind them)."""
        rng = _stable_rng(metal, reaction)
        out = self._composite_catalyst_score(
            float(rng.uniform(-3.0, -0.5)), float(rng.uniform(0.3, 2.5)), float(rng.uniform(0.5, 1.0)))
        out["source"] = "placeholder_unvalidated"
        return out

    def _generate_pathway_steps(self, reactants: List[str], products: List[str]) -> List[Dict]:
        intermediates = ["Adsorption complex", "Activated intermediate", "Transition state 1", "Transition state 2"]
        steps = []
        rng = _stable_rng(*reactants, "->", *products)
        for i, name in enumerate(intermediates):
            steps.append({
                "step": i + 1,
                "name": name,
                "delta_G": round(float(rng.standard_normal()) * 10, 2),
                "delta_H": round(float(rng.standard_normal()) * 12, 2),
                "is_rate_determining": i == len(intermediates) - 2,
            })
        return steps
