"""Tests for all 12 QuantumAgent specialized agents."""
import asyncio
import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import pytest
from backend.agents.pi_agent import PrincipalInvestigatorAgent
from backend.agents.literature_agent import LiteratureAgent
from backend.agents.circuit_designer import CircuitDesignerAgent
from backend.agents.chemistry_agent import QuantumChemistryAgent
from backend.agents.lab_automation import LabAutomationAgent
from backend.agents.hardware_agent import HardwareIntegrationAgent
from backend.agents.optimization_agent import OptimisationAgent
from backend.agents.analysis_agent import DataAnalysisAgent
from backend.agents.explainability_agent import ExplainabilityAgent
from backend.agents.knowledge_graph_agent import KnowledgeGraphAgent
from backend.agents.coding_agent import CodingAgent
from backend.agents.publication_agent import PublicationAgent


def run(coro):
    return asyncio.get_event_loop().run_until_complete(coro)


# --------------------------------------------------------------------------- #
#  PI Agent                                                                    #
# --------------------------------------------------------------------------- #

class TestPIAgent:
    def setup_method(self):
        self.agent = PrincipalInvestigatorAgent()

    def test_research_plan(self):
        r = run(self.agent.run({"action": "generate_research_plan", "domain": "quantum_computing"}))
        assert r.success
        assert "phases" in r.output
        assert len(r.output["phases"]) > 0

    def test_rank_hypotheses(self):
        hyps = [{"hypothesis_id": "H01", "statement": "Test hypothesis"}]
        r = run(self.agent.run({"action": "rank_hypotheses", "hypotheses": hyps}))
        assert r.success
        assert "ranked_hypotheses" in r.output

    def test_milestone_plan(self):
        r = run(self.agent.run({"action": "create_milestone_plan", "duration_weeks": 52}))
        assert r.success
        assert "milestones" in r.output

    def test_funding_outline(self):
        r = run(self.agent.run({"action": "generate_funding_outline", "agency": "NSF"}))
        assert r.success
        assert "sections" in r.output

    def test_status_dict(self):
        s = self.agent.status_dict()
        assert s["name"] == "pi_agent"
        assert "capabilities" in s


# --------------------------------------------------------------------------- #
#  Literature Agent                                                            #
# --------------------------------------------------------------------------- #

class TestLiteratureAgent:
    def setup_method(self):
        self.agent = LiteratureAgent()

    def test_search_returns_papers(self):
        r = run(self.agent.run({"action": "search", "query": "VQE quantum chemistry", "sources": ["arxiv"], "max_results": 5}))
        assert r.success
        assert "papers" in r.output
        assert r.output["total_found"] > 0

    def test_detect_trends(self):
        run(self.agent.run({"action": "search", "query": "quantum error correction", "max_results": 5}))
        r = run(self.agent.run({"action": "detect_trends"}))
        assert r.success

    def test_find_gaps(self):
        r = run(self.agent.run({"action": "find_gaps", "domain": "quantum_computing"}))
        assert r.success
        assert "research_gaps" in r.output

    def test_recommend_experiments(self):
        r = run(self.agent.run({"action": "recommend_experiments", "gaps": ["benchmark gap"]}))
        assert r.success
        assert "recommendations" in r.output


# --------------------------------------------------------------------------- #
#  Circuit Designer Agent                                                      #
# --------------------------------------------------------------------------- #

class TestCircuitDesigner:
    def setup_method(self):
        self.agent = CircuitDesignerAgent()

    def test_vqe_circuit(self):
        r = run(self.agent.run({"action": "design_vqe", "n_qubits": 4, "n_layers": 2}))
        assert r.success
        assert r.output["gate_count"] > 0
        assert r.output["depth"] > 0
        assert "qasm3" in r.output
        assert "pennylane" in r.output

    def test_qaoa_circuit(self):
        r = run(self.agent.run({"action": "design_qaoa", "n_qubits": 4, "p": 2}))
        assert r.success
        assert r.output["n_parameters"] == 4

    def test_qec_surface_code(self):
        r = run(self.agent.run({"action": "design_qec", "code": "surface_code", "distance": 3}))
        assert r.success
        assert r.output["n_physical_qubits"] > 0

    def test_circuit_optimise(self):
        r1 = run(self.agent.run({"action": "design_vqe", "n_qubits": 4}))
        circuit_id = r1.output["circuit_id"]
        r2 = run(self.agent.run({"action": "optimise", "circuit_id": circuit_id}))
        assert r2.success

    def test_circuit_analyse(self):
        r1 = run(self.agent.run({"action": "design_vqe", "n_qubits": 4}))
        circuit_id = r1.output["circuit_id"]
        r2 = run(self.agent.run({"action": "analyse", "circuit_id": circuit_id}))
        assert r2.success
        assert "estimated_fidelity" in r2.output

    def test_fidelity_between_0_and_1(self):
        r = run(self.agent.run({"action": "design_vqe", "n_qubits": 4}))
        cid = r.output["circuit_id"]
        r2 = run(self.agent.run({"action": "analyse", "circuit_id": cid}))
        f = r2.output["estimated_fidelity"]
        assert 0 <= f <= 1


# --------------------------------------------------------------------------- #
#  Chemistry Agent                                                             #
# --------------------------------------------------------------------------- #

class TestChemistryAgent:
    def setup_method(self):
        self.agent = QuantumChemistryAgent()

    def test_simulate_h2(self):
        r = run(self.agent.run({"action": "simulate", "molecule": "H2", "basis": "sto-3g"}))
        assert r.success
        assert "vqe_energy" in r.output
        assert r.output["n_qubits"] > 0

    def test_generate_hamiltonian(self):
        r = run(self.agent.run({"action": "generate_hamiltonian", "molecule": "LiH"}))
        assert r.success
        assert r.output["n_qubits"] > 0
        assert r.output["n_electrons"] > 0

    def test_design_catalyst(self):
        r = run(self.agent.run({"action": "design_catalyst", "reaction": "nitrogen_fixation"}))
        assert r.success
        assert "candidates" in r.output
        assert len(r.output["candidates"]) > 0

    def test_benchmark(self):
        r = run(self.agent.run({"action": "benchmark", "molecule": "H2"}))
        assert r.success
        assert r.output["ccsd_t_reference"] is not None


# --------------------------------------------------------------------------- #
#  Hardware Agent                                                              #
# --------------------------------------------------------------------------- #

class TestHardwareAgent:
    def setup_method(self):
        self.agent = HardwareIntegrationAgent()

    def test_list_backends(self):
        r = run(self.agent.run({"action": "list_backends"}))
        assert r.success
        assert r.output["total"] > 0

    def test_submit_and_get_results(self):
        r_submit = run(self.agent.run({"action": "submit_job", "backend_id": "aer_simulator", "shots": 1024}))
        assert r_submit.success
        job_id = r_submit.output["job_id"]
        assert r_submit.output["status"] == "completed"
        assert r_submit.output["counts"] is not None

    def test_select_backend(self):
        r = run(self.agent.run({"action": "select_backend", "n_qubits": 5, "priority": "fidelity"}))
        assert r.success
        assert r.output["recommended_backend"] is not None

    def test_compare_backends(self):
        r = run(self.agent.run({"action": "compare_backends"}))
        assert r.success
        assert len(r.output["comparison"]) > 0


# --------------------------------------------------------------------------- #
#  Optimisation Agent                                                          #
# --------------------------------------------------------------------------- #

class TestOptimisationAgent:
    def setup_method(self):
        self.agent = OptimisationAgent()

    def test_bayesian_opt(self):
        r = run(self.agent.run({"action": "bayesian_opt", "n_params": 2, "n_initial": 5, "n_iterations": 10}))
        assert r.success
        assert "best_value" in r.output

    def test_cma_es(self):
        r = run(self.agent.run({"action": "cma_es", "n_params": 3, "n_generations": 10}))
        assert r.success
        assert "best_value" in r.output

    def test_gradient_opt(self):
        r = run(self.agent.run({"action": "gradient_opt", "n_params": 4, "n_steps": 50}))
        assert r.success
        assert r.output["final_loss"] >= 0


# --------------------------------------------------------------------------- #
#  Analysis Agent                                                              #
# --------------------------------------------------------------------------- #

class TestAnalysisAgent:
    def setup_method(self):
        self.agent = DataAnalysisAgent()

    def test_statistical_analysis(self):
        import numpy as np
        data = np.random.randn(100).tolist()
        r = run(self.agent.run({"action": "analyse", "data": data}))
        assert r.success
        assert "mean" in r.output
        assert "std" in r.output

    def test_state_tomography(self):
        r = run(self.agent.run({"action": "state_tomography", "n_qubits": 2}))
        assert r.success
        assert 0 <= r.output["state_purity"] <= 1

    def test_error_analysis(self):
        r = run(self.agent.run({"action": "error_analysis", "expected_state": "0000"}))
        assert r.success
        assert 0 <= r.output["success_probability"] <= 1

    def test_benchmark_compare(self):
        r = run(self.agent.run({"action": "benchmark_compare", "methods": ["VQE", "HF"]}))
        assert r.success
        assert "benchmark_results" in r.output


# --------------------------------------------------------------------------- #
#  Explainability Agent                                                        #
# --------------------------------------------------------------------------- #

class TestExplainabilityAgent:
    def setup_method(self):
        self.agent = ExplainabilityAgent()

    def test_explain(self):
        r = run(self.agent.run({"action": "explain", "recommendation": {"circuit_type": "VQE"}, "type": "circuit_design"}))
        assert r.success
        assert "rationale" in r.output
        assert "supporting_publications" in r.output
        assert "alternatives_considered" in r.output

    def test_risk_assessment(self):
        r = run(self.agent.run({"action": "assess_risks", "n_qubits": 25, "circuit_depth": 60}))
        assert r.success
        assert len(r.output["risks"]) > 0


# --------------------------------------------------------------------------- #
#  Knowledge Graph Agent                                                       #
# --------------------------------------------------------------------------- #

class TestKnowledgeGraphAgent:
    def setup_method(self):
        self.agent = KnowledgeGraphAgent()

    def test_graph_stats(self):
        r = run(self.agent.run({"action": "stats"}))
        assert r.success
        assert r.output["total_nodes"] > 0

    def test_add_and_query_node(self):
        run(self.agent.run({"action": "add_node", "node_type": "algorithm", "label": "Test Algorithm"}))
        r = run(self.agent.run({"action": "query", "query": "Test Algorithm"}))
        assert r.success
        labels = [res["label"] for res in r.output["results"]]
        assert "Test Algorithm" in labels

    def test_generate_hypotheses(self):
        r = run(self.agent.run({"action": "generate_hypotheses", "domain": "quantum_computing", "n": 3}))
        assert r.success
        assert len(r.output["hypotheses"]) > 0

    def test_ingest_paper(self):
        r = run(self.agent.run({
            "action": "ingest_paper",
            "title": "Test Quantum Paper",
            "abstract": "We study VQE on H2 using IBM quantum hardware.",
            "authors": ["Author A"],
        }))
        assert r.success
        assert "paper_id" in r.output


# --------------------------------------------------------------------------- #
#  Coding Agent                                                                #
# --------------------------------------------------------------------------- #

class TestCodingAgent:
    def setup_method(self):
        self.agent = CodingAgent()

    def test_vqe_code_generation(self):
        r = run(self.agent.run({"action": "generate", "type": "vqe_python", "params": {"molecule": "H2", "n_qubits": 4}}))
        assert r.success
        assert "def " in r.output["code"]
        assert r.output["lines"] > 10

    def test_notebook_generation(self):
        r = run(self.agent.run({"action": "generate_notebook", "title": "Test Notebook"}))
        assert r.success
        assert r.output["n_cells"] > 0
        assert "cells" in r.output["notebook"]

    def test_code_review(self):
        code = "import numpy as np\ndef test(): return np.zeros(3)"
        r = run(self.agent.run({"action": "review", "code": code}))
        assert r.success
        assert "quality_score" in r.output

    def test_test_generation(self):
        code = "def run_vqe(): pass\ndef build_ansatz(): pass"
        r = run(self.agent.run({"action": "generate_tests", "code": code, "module": "test_mod"}))
        assert r.success
        assert "test_code" in r.output


# --------------------------------------------------------------------------- #
#  Publication Agent                                                           #
# --------------------------------------------------------------------------- #

class TestPublicationAgent:
    def setup_method(self):
        self.agent = PublicationAgent()

    def test_abstract_generation(self):
        r = run(self.agent.run({"action": "abstract", "title": "VQE for Quantum Chemistry"}))
        assert r.success
        assert len(r.output["abstract"]) > 100

    def test_paper_draft(self):
        r = run(self.agent.run({"action": "draft_paper", "title": "Test Paper"}))
        assert r.success
        assert "latex_source" in r.output
        assert r.output["word_count"] > 100

    def test_bibliography(self):
        r = run(self.agent.run({"action": "bibliography", "topics": ["vqe"]}))
        assert r.success
        assert r.output["n_references"] > 0

    def test_grant_proposal(self):
        r = run(self.agent.run({"action": "grant_proposal", "agency": "NSF"}))
        assert r.success
        assert "sections" in r.output


if __name__ == "__main__":
    import pytest
    pytest.main([__file__, "-v", "--tb=short"])
