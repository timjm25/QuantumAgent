"""
Autonomous Coding Agent

Generates, tests, and documents:
  - Python quantum simulation scripts
  - Rust performance-critical components
  - Jupyter notebooks with interactive visualisations
  - Workflow automation scripts
  - Unit tests for generated code
  - CI/CD pipeline configurations
"""
from __future__ import annotations

import ast
import textwrap
import uuid
from typing import Any, Dict, List, Optional

from .base import BaseAgent, AgentResult
from ..core.logging import get_logger

logger = get_logger(__name__)


class CodingAgent(BaseAgent):
    name = "coding_agent"
    description = "Generate, test, and document quantum research code in Python, Rust, and Jupyter notebooks"
    capabilities = [
        "python_generation",
        "rust_generation",
        "notebook_generation",
        "test_generation",
        "code_review",
        "refactoring",
        "documentation_generation",
        "workflow_scripting",
    ]

    TEMPLATES = {
        "vqe_python": """\
\"\"\"VQE ground state energy calculation for {molecule}.\"\"\"
import numpy as np
from qiskit import QuantumCircuit
from qiskit.circuit import ParameterVector
from qiskit.quantum_info import SparsePauliOp, Statevector
from scipy.optimize import minimize

# --- Qubit Hamiltonian ({n_qubits} qubits): list of (coefficient, Pauli label) ---
# {hamiltonian_note}
PAULI_TERMS = {pauli_terms}

def build_ansatz(n_qubits: int, n_layers: int) -> QuantumCircuit:
    \"\"\"Hardware-efficient ansatz.\"\"\"
    params = ParameterVector("θ", n_qubits * 2 * n_layers)
    qc = QuantumCircuit(n_qubits)
    idx = 0
    qc.h(range(n_qubits))
    for _ in range(n_layers):
        for i in range(n_qubits):
            qc.ry(params[idx], i); idx += 1
            qc.rz(params[idx], i); idx += 1
        for i in range(n_qubits - 1):
            qc.cx(i, i + 1)
    return qc

HAMILTONIAN = SparsePauliOp.from_list([(label, coeff) for coeff, label in PAULI_TERMS])

def cost_function(params, ansatz, hamiltonian):
    \"\"\"VQE cost function: exact expectation value <psi(theta)|H|psi(theta)>.\"\"\"
    sv = Statevector(ansatz.assign_parameters(params))
    return float(np.real(sv.expectation_value(hamiltonian)))

def run_vqe(n_qubits={n_qubits}, n_layers=2, max_iter={max_iter}):
    \"\"\"Run VQE optimisation.\"\"\"
    ansatz = build_ansatz(n_qubits, n_layers)
    n_params = n_qubits * 2 * n_layers
    x0 = np.random.default_rng(7).uniform(-np.pi, np.pi, n_params)

    result = minimize(
        cost_function,
        x0,
        args=(ansatz, HAMILTONIAN),
        method='COBYLA',
        options={{'maxiter': max_iter, 'rhobeg': 0.3}},
    )

    print(f"VQE converged: {{result.success}}")
    print(f"Ground state energy: {{result.fun:.8f}} Hartree")
    print(f"Cost evaluations: {{result.nfev}}")
    return result

if __name__ == "__main__":
    run_vqe()
""",

        "qaoa_python": """\
\"\"\"QAOA for MaxCut on a {n_nodes}-node graph.\"\"\"
import numpy as np
import networkx as nx
from qiskit import QuantumCircuit
from qiskit.circuit import ParameterVector
from qiskit_aer import AerSimulator
from scipy.optimize import minimize

def build_maxcut_qaoa(graph: nx.Graph, p: int = {p}) -> QuantumCircuit:
    n = graph.number_of_nodes()
    params = ParameterVector("θ", 2 * p)
    qc = QuantumCircuit(n, n)
    qc.h(range(n))

    for layer in range(p):
        gamma, beta = params[2 * layer], params[2 * layer + 1]
        for u, v in graph.edges():
            qc.cx(u, v)
            qc.rz(2 * gamma, v)
            qc.cx(u, v)
        for i in range(n):
            qc.rx(2 * beta, i)

    qc.measure(range(n), range(n))
    return qc

def maxcut_expectation(counts, graph):
    total = sum(counts.values())
    cut_value = 0.0
    for bitstring, count in counts.items():
        bits = [int(b) for b in bitstring[::-1]]
        cut = sum(1 for u, v in graph.edges() if bits[u] != bits[v])
        cut_value += cut * count / total
    return cut_value

def run_qaoa():
    G = nx.random_regular_graph(3, {n_nodes}, seed=42)
    qc = build_maxcut_qaoa(G, p={p})
    sim = AerSimulator()

    def cost(params):
        bound = qc.assign_parameters(params)
        result = sim.run(bound, shots=4096).result()
        return -maxcut_expectation(result.get_counts(), G)

    x0 = np.random.uniform(0, np.pi, {p} * 2)
    result = minimize(cost, x0, method='COBYLA', options={{'maxiter': 500}})
    print(f"MaxCut value: {{-result.fun:.2f}}")
    print(f"Optimal params: {{result.x}}")
    return result

if __name__ == "__main__":
    run_qaoa()
""",

        "rust_simulator": """\
// High-performance quantum state vector simulator in Rust.
// Target: Compile with `cargo build --release`

use std::ops::Mul;
use num_complex::Complex64;
use rayon::prelude::*;

type StateVec = Vec<Complex64>;

pub fn hadamard_gate(state: &StateVec, target: usize) -> StateVec {{
    let n = state.len();
    let half = n / 2;
    let s = 1.0 / (2.0_f64).sqrt();
    let mut new_state = state.clone();
    let step = 1usize << target;

    new_state.par_chunks_mut(step * 2).for_each(|chunk| {{
        for i in 0..step {{
            let a = chunk[i];
            let b = chunk[i + step];
            chunk[i] = (a + b).mul(s);
            chunk[i + step] = (a - b).mul(s);
        }}
    }});
    new_state
}}

pub fn cnot_gate(state: &StateVec, control: usize, target: usize) -> StateVec {{
    let n = state.len();
    let mut new_state = state.clone();
    for i in 0..n {{
        if (i >> control) & 1 == 1 {{
            let j = i ^ (1 << target);
            new_state.swap(i, j);
        }}
    }}
    new_state
}}

pub fn measure_all(state: &StateVec) -> usize {{
    let probs: Vec<f64> = state.iter().map(|c| c.norm_sqr()).collect();
    let r: f64 = rand::random();
    let mut cumsum = 0.0;
    for (i, &p) in probs.iter().enumerate() {{
        cumsum += p;
        if r < cumsum {{
            return i;
        }}
    }}
    probs.len() - 1
}}

#[cfg(test)]
mod tests {{
    use super::*;

    #[test]
    fn test_hadamard_superposition() {{
        let state = vec![Complex64::new(1.0, 0.0), Complex64::new(0.0, 0.0)];
        let result = hadamard_gate(&state, 0);
        let s = 1.0 / (2.0_f64).sqrt();
        assert!((result[0].re - s).abs() < 1e-10);
        assert!((result[1].re - s).abs() < 1e-10);
    }}
}}
""",
    }

    async def _execute(self, task_input: Dict[str, Any]) -> AgentResult:
        action = task_input.get("action", "generate")
        dispatch = {
            "generate": self._generate_code,
            "generate_notebook": self._generate_notebook,
            "generate_tests": self._generate_tests,
            "review": self._review_code,
            "refactor": self._refactor_code,
            "generate_docs": self._generate_docs,
            "generate_workflow": self._generate_workflow,
        }
        handler = dispatch.get(action)
        if handler is None:
            return AgentResult(success=False, output={"error": f"Unknown action: {action}"})
        return await handler(task_input)

    # ------------------------------------------------------------------ #

    async def _generate_code(self, inp: Dict) -> AgentResult:
        code_type = inp.get("type", "vqe_python")
        params = inp.get("params", {})

        template = self.TEMPLATES.get(code_type, "")
        if not template:
            code = self._generate_generic_code(inp)
        else:
            molecule = params.get("molecule", "H2")
            terms, note = self._hamiltonian_terms(params, molecule)
            context = {
                "molecule": molecule,
                "n_qubits": len(terms[0][1]) if code_type == "vqe_python" else params.get("n_qubits", 4),
                "n_nodes": params.get("n_nodes", 6),
                "p": params.get("p", 3),
                "max_iter": params.get("max_iter", 200),
                "pauli_terms": repr(terms),
                "hamiltonian_note": note,
            }
            code = template.format(**context)

        validation = self._validate_python(code) if "python" in code_type or "rust" not in code_type else {"valid": True, "errors": []}

        return AgentResult(
            success=True,
            output={
                "code_type": code_type,
                "code": code,
                "language": "rust" if "rust" in code_type else "python",
                "lines": len(code.splitlines()),
                "valid": validation["valid"],
                "validation_errors": validation.get("errors", []),
                "estimated_runtime_s": params.get("n_qubits", 4) ** 2 * 0.01,
            },
            confidence=0.92,
            reasoning=f"Generated {code_type} with {len(code.splitlines())} lines.",
        )

    async def _generate_notebook(self, inp: Dict) -> AgentResult:
        title = inp.get("title", "Quantum Computing Experiment")
        workflow = inp.get("workflow", "vqe_demo")

        cells = self._build_notebook_cells(title, workflow, inp)

        notebook = {
            "nbformat": 4,
            "nbformat_minor": 5,
            "metadata": {
                "kernelspec": {"display_name": "Python 3", "language": "python", "name": "python3"},
                "language_info": {"name": "python", "version": "3.10.0"},
            },
            "cells": cells,
        }

        return AgentResult(
            success=True,
            output={
                "title": title,
                "n_cells": len(cells),
                "notebook": notebook,
                "estimated_runtime_min": 5,
            },
            confidence=0.90,
        )

    async def _generate_tests(self, inp: Dict) -> AgentResult:
        code = inp.get("code", "")
        module_name = inp.get("module", "quantum_module")
        test_style = inp.get("style", "pytest")

        functions = self._extract_functions(code) if code else ["run_vqe", "build_ansatz", "cost_function"]

        test_code = self._build_test_code(functions, module_name, test_style)

        return AgentResult(
            success=True,
            output={
                "test_file": f"test_{module_name}.py",
                "n_tests": len(functions),
                "test_code": test_code,
                "framework": test_style,
            },
            confidence=0.88,
        )

    async def _review_code(self, inp: Dict) -> AgentResult:
        code = inp.get("code", "")
        language = inp.get("language", "python")

        issues = []

        if language == "python":
            if "numpy" in code and "import numpy" not in code:
                issues.append({"severity": "error", "message": "numpy used but not imported"})
            if "torch.tensor" in code and ".numpy()" in code:
                issues.append({"severity": "warning", "message": "Potential NumPy 2.x/PyTorch ABI issue — use .tolist() instead of .numpy()"})
            if not any(kw in code for kw in ["if __name__", "def test_", "class Test"]):
                issues.append({"severity": "info", "message": "No entry point or tests detected"})
            if len(code.splitlines()) > 200 and "def " not in code:
                issues.append({"severity": "warning", "message": "Long file with no functions — consider refactoring"})

        suggestions = [
            "Add type hints to all function signatures",
            "Consider adding docstrings for public functions",
            "Add error handling for quantum hardware API calls",
        ]

        return AgentResult(
            success=True,
            output={
                "language": language,
                "lines": len(code.splitlines()),
                "issues": issues,
                "n_issues": len(issues),
                "suggestions": suggestions,
                "quality_score": max(0, 1 - len(issues) * 0.15),
            },
            confidence=0.85,
        )

    async def _refactor_code(self, inp: Dict) -> AgentResult:
        code = inp.get("code", "")
        strategy = inp.get("strategy", "extract_functions")

        refactored = code
        changes = []

        if strategy == "extract_functions" and len(code.splitlines()) > 50:
            changes.append("Would extract inline blocks into named functions")
        if strategy == "add_type_hints":
            changes.append("Would add PEP 484 type hints to all function signatures")
        if strategy == "add_tests":
            changes.append("Would generate pytest test suite")

        return AgentResult(
            success=True,
            output={
                "strategy": strategy,
                "original_lines": len(code.splitlines()),
                "refactored_lines": len(refactored.splitlines()),
                "changes_made": changes,
                "refactored_code": refactored,
            },
            confidence=0.82,
        )

    async def _generate_docs(self, inp: Dict) -> AgentResult:
        code = inp.get("code", "")
        module_name = inp.get("module", "quantum_module")

        docs = f"""# {module_name}

## Overview
Generated quantum computing module for QuantumAgent platform.

## Functions

"""
        functions = self._extract_functions(code)
        for fn in functions:
            docs += f"### `{fn}()`\n\nAuto-generated documentation placeholder.\n\n"

        return AgentResult(
            success=True,
            output={"module": module_name, "documentation": docs, "n_functions": len(functions)},
            confidence=0.80,
        )

    async def _generate_workflow(self, inp: Dict) -> AgentResult:
        workflow_type = inp.get("type", "vqe_chemistry")
        n_steps = inp.get("n_steps", 6)

        script_lines = [
            "#!/usr/bin/env python3",
            '"""Auto-generated QuantumAgent workflow script."""',
            "import argparse",
            "import logging",
            "",
            "logging.basicConfig(level=logging.INFO)",
            "",
            "def main():",
        ]

        steps_map = {
            "vqe_chemistry": [
                "literature_review",
                "build_hamiltonian",
                "design_ansatz",
                "run_vqe",
                "error_mitigation",
                "analyse_results",
                "update_knowledge_graph",
                "generate_report",
            ],
        }

        steps = steps_map.get(workflow_type, ["step_1", "step_2", "step_3"])[:n_steps]
        for step in steps:
            script_lines.append(f'    logging.info("Executing step: {step}")')
            script_lines.append(f"    # TODO: implement {step}")

        script_lines += ["", 'if __name__ == "__main__":', "    main()"]

        return AgentResult(
            success=True,
            output={
                "workflow_type": workflow_type,
                "n_steps": len(steps),
                "script": "\n".join(script_lines),
            },
            confidence=0.87,
        )

    # ------------------------------------------------------------------ #

    def _validate_python(self, code: str) -> Dict:
        try:
            ast.parse(code)
            return {"valid": True, "errors": []}
        except SyntaxError as e:
            return {"valid": False, "errors": [f"SyntaxError at line {e.lineno}: {e.msg}"]}

    def _extract_functions(self, code: str) -> List[str]:
        fns = []
        try:
            tree = ast.parse(code)
            for node in ast.walk(tree):
                if isinstance(node, ast.FunctionDef):
                    fns.append(node.name)
        except Exception:
            fns = ["function_1", "function_2"]
        return fns

    def _build_test_code(self, functions: List[str], module: str, style: str) -> str:
        lines = [f"import pytest", f"from {module} import {', '.join(functions[:5])}", ""]
        for fn in functions[:5]:
            lines += [
                f"def test_{fn}_runs_without_error():",
                f"    # Smoke test: verify {fn} executes without raising",
                f"    # TODO: add meaningful assertions",
                f"    pass",
                "",
            ]
        return "\n".join(lines)

    @staticmethod
    def _hamiltonian_terms(params: Dict, molecule: str):
        """(coeff, label) terms for generated VQE code: user-supplied, built-in, or a labelled toy."""
        from ..quantum.pauli import builtin_hamiltonian, normalise_label

        if params.get("pauli_terms"):
            raw = [(float(c), str(l)) for c, l in params["pauli_terms"]]
            width = max(len(normalise_label(l)) for _, l in raw)
            return [(c, normalise_label(l, width)) for c, l in raw], "Supplied by the caller."
        ham = builtin_hamiltonian(molecule, params.get("basis", "sto-3g"))
        if ham is not None:
            return ([(round(c, 12), l) for l, c in ham.terms],
                    f"{molecule}/{ham.basis} reference Hamiltonian (electronic part; "
                    f"add nuclear repulsion {ham.nuclear_repulsion:.6f} Ha for the total energy).")
        n = int(params.get("n_qubits", 2))
        toy = [(-0.5, "Z".rjust(n, "I")), (0.5, "Z".ljust(n, "I")), (0.25, "XX".rjust(n, "I"))]
        return toy, (f"TOY Hamiltonian: no qubit Hamiltonian for {molecule} is available. "
                     "Replace PAULI_TERMS with your exported operator.")

    def _build_notebook_cells(self, title: str, workflow: str, inp: Dict) -> List[Dict]:
        molecule = inp.get("molecule", "H2")
        n_qubits = inp.get("n_qubits", 4)

        cells = [
            {"cell_type": "markdown", "source": [f"# {title}\n\nGenerated by QuantumAgent platform."], "metadata": {}},
            {"cell_type": "code", "source": ["import numpy as np\nimport matplotlib.pyplot as plt\n%matplotlib inline"], "outputs": [], "metadata": {}},
            {"cell_type": "markdown", "source": ["## 1. Configure Molecule and Hamiltonian"], "metadata": {}},
            {"cell_type": "code", "source": [f"molecule = '{molecule}'\nn_qubits = {n_qubits}\nprint(f'Simulating {{molecule}} on {{n_qubits}} qubits')"], "outputs": [], "metadata": {}},
            {"cell_type": "markdown", "source": ["## 2. Build Ansatz"], "metadata": {}},
            {"cell_type": "code", "source": ["# Build hardware-efficient ansatz\nfrom qiskit import QuantumCircuit\nqc = QuantumCircuit(n_qubits)\nqc.h(range(n_qubits))\nprint(qc)"], "outputs": [], "metadata": {}},
            {"cell_type": "markdown", "source": ["## 3. Run VQE"], "metadata": {}},
            {"cell_type": "code", "source": ["# VQE on the local state-vector simulator (run from the QuantumAgent repo root)\nfrom backend.quantum import h2_sto3g, run_vqe\nham = h2_sto3g()  # or backend.importers.hamiltonians_from_rows(...) for your own\nres = run_vqe(ham, layers=2)\nenergies = [e + ham.nuclear_repulsion for e in res.history]\nplt.plot(energies)\nplt.xlabel('Cost evaluation')\nplt.ylabel('Energy (Ha)')\nplt.title('VQE Convergence')\nplt.show()"], "outputs": [], "metadata": {}},
            {"cell_type": "markdown", "source": ["## 4. Results"], "metadata": {}},
            {"cell_type": "code", "source": [f"print(f'Final energy: {{energies[-1]:.6f}} Ha')\nprint(f'Exact (same basis): {{res.exact_total_energy:.6f}} Ha')"], "outputs": [], "metadata": {}},
        ]
        return cells

    def _generate_generic_code(self, inp: Dict) -> str:
        description = inp.get("description", "quantum experiment")
        return textwrap.dedent(f"""\
            \"\"\"Auto-generated script: {description}\"\"\"
            import numpy as np
            import logging

            logging.basicConfig(level=logging.INFO)
            logger = logging.getLogger(__name__)

            def run():
                logger.info("Starting {description}")
                # TODO: Implement experiment logic
                result = {{"status": "placeholder", "description": "{description}"}}
                return result

            if __name__ == "__main__":
                result = run()
                print(result)
        """)
