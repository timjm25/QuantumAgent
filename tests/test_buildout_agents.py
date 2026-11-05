"""Filled-in agent logic: real execution, imported data, honest fallbacks."""
import asyncio
import os

import pytest

from backend.agents.analysis_agent import DataAnalysisAgent
from backend.agents.chemistry_agent import QuantumChemistryAgent
from backend.agents.circuit_designer import CircuitDesignerAgent
from backend.agents.explainability_agent import ExplainabilityAgent
from backend.agents.hardware_agent import HardwareIntegrationAgent
from backend.agents.knowledge_graph_agent import KnowledgeGraphAgent
from backend.agents.lab_automation import LabAutomationAgent
from backend.agents.literature_agent import LiteratureAgent
from backend.agents.pi_agent import PrincipalInvestigatorAgent
from backend.data_loader import SAMPLE_DIR
from backend.importers import load_rows, template_path
from backend.orchestrator.workflow import WorkflowEngine

GHZ = "OPENQASM 2.0;\ninclude \"qelib1.inc\";\nqreg q[3];\ncreg c[3];\nh q[0];\ncx q[0],q[1];\ncx q[1],q[2];\nmeasure q -> c;"


def run(coro):
    return asyncio.run(coro)


# -- hardware ---------------------------------------------------------------

def test_submit_qasm_runs_real_circuit():
    a = HardwareIntegrationAgent(seed=4)
    r = run(a.run({"action": "submit_job", "qasm": GHZ, "shots": 2000}))
    assert r.success and r.output["engine"] == "numpy_statevector"
    assert set(r.output["counts"]) == {"000", "111"}
    res = run(a.run({"action": "get_results", "job_id": r.output["job_id"]}))
    assert res.output["most_probable_state"] in ("000", "111")


def test_no_circuit_runs_labelled_bell_demo():
    r = run(HardwareIntegrationAgent().run({"action": "submit_job", "shots": 100}))
    assert r.success and set(r.output["counts"]) <= {"00", "11"}
    assert any("Bell" in w for w in r.warnings)


def test_bad_qasm_is_reported():
    r = run(HardwareIntegrationAgent().run({"action": "submit_job", "qasm": "OPENQASM 2.0; qreg q[1]; zap q[0];"}))
    assert not r.success and "unsupported gate" in r.output["error"]


def test_cloud_backend_disabled_by_default(monkeypatch):
    monkeypatch.delenv("QUANTUMAGENT_ENABLE_CLOUD", raising=False)
    r = run(HardwareIntegrationAgent().run({"action": "submit_job", "backend_id": "ibm_nairobi", "qasm": GHZ}))
    assert not r.success and "disabled" in r.output["error"] and "emulate" in r.output["hint"]


def test_cloud_enabled_without_token_asks_for_env_var(monkeypatch):
    monkeypatch.setenv("QUANTUMAGENT_ENABLE_CLOUD", "true")
    monkeypatch.delenv("IONQ_API_KEY", raising=False)
    r = run(HardwareIntegrationAgent().run({"action": "submit_job", "backend_id": "ionq_aria", "qasm": GHZ}))
    assert not r.success and "IONQ_API_KEY" in r.output["error"]
    st = run(HardwareIntegrationAgent().run({"action": "cloud_status"})).output
    assert st["cloud_enabled"] and not st["providers"]["ionq"]["credential_set"]


def test_emulated_backend_adds_noise():
    r = run(HardwareIntegrationAgent(seed=2).run(
        {"action": "submit_job", "backend_id": "rigetti_ankaa", "qasm": GHZ, "shots": 5000, "emulate": True}))
    assert r.success and "noise" in r.output["engine"]
    assert len(r.output["counts"]) > 2  # readout/depolarising errors populate other bitstrings


def test_register_device_from_calibration_and_rank():
    a = HardwareIntegrationAgent()
    rows = load_rows("device_calibration", template_path("device_calibration"))
    reg = run(a.run({"action": "register_device", "rows": rows}))
    dev = reg.output["registered"][0]
    assert dev["n_qubits"] == 3 and dev["type"] == "user_device"
    sel = run(a.run({"action": "select_backend", "n_qubits": 3, "exclude_simulators": True}))
    assert dev["backend_id"] in [b["backend_id"] for b in sel.output["ranking"]]
    assert all(b["type"] not in ("aer_simulator", "local_simulator") for b in sel.output["ranking"])
    job = run(a.run({"action": "submit_job", "backend_id": dev["backend_id"], "qasm": GHZ, "shots": 500}))
    assert job.success


# -- circuit designer ---------------------------------------------------------

def test_designed_circuit_can_be_simulated():
    a = CircuitDesignerAgent()
    cid = run(a.run({"action": "design_qaoa", "n_qubits": 4, "p": 1})).output["circuit_id"]
    r = run(a.run({"action": "simulate", "circuit_id": cid, "shots": 256}))
    assert r.success and sum(r.output["counts"].values()) == 256


# -- chemistry ---------------------------------------------------------------

def test_imported_hamiltonian_drives_real_vqe():
    a = QuantumChemistryAgent()
    rows = load_rows("hamiltonian", template_path("hamiltonian"))
    assert run(a.run({"action": "load_hamiltonians", "rows": rows})).success
    r = run(a.run({"action": "vqe", "molecule": "SAMPLE_H2"}))
    assert r.output["hamiltonian_source"] == "user_import"
    assert r.output["within_chemical_accuracy"]
    assert abs(r.output["final_energy"] - (-1.137306)) < 1e-4
    listed = run(a.run({"action": "list_hamiltonians"})).output
    assert listed["user_imported"][0]["name"] == "SAMPLE_H2"


def test_missing_hamiltonian_is_clearly_synthetic_and_deterministic():
    a1, a2 = QuantumChemistryAgent(), QuantumChemistryAgent()
    r1 = run(a1.run({"action": "simulate", "molecule": "LiH"}))
    r2 = run(a2.run({"action": "simulate", "molecule": "LiH"}))
    assert r1.output["hamiltonian_source"] == "synthetic_placeholder"
    assert r1.output["computation"] == "surrogate_estimate"
    assert r1.warnings and "SYNTHETIC" in r1.warnings[0]
    assert r1.output["vqe_energy"] == r2.output["vqe_energy"]


def test_catalyst_uses_user_descriptors():
    r = run(QuantumChemistryAgent().run({
        "action": "design_catalyst", "metals": ["Fe", "Ru"],
        "descriptors": {"Fe": {"binding_energy": -1.5, "activation_barrier": 0.2, "selectivity": 0.95},
                        "Ru": {"binding_energy": -2.9, "activation_barrier": 2.0, "selectivity": 0.5}}}))
    assert r.output["top_recommendation"]["metal"] == "Fe"
    assert not r.warnings


# -- lab automation ------------------------------------------------------------

def test_characterise_uses_imported_measurements():
    a = LabAutomationAgent()
    a.load_measurements(load_rows("qubit_measurements", SAMPLE_DIR / "qubit_measurements_demo.csv"))
    r = run(a.run({"action": "characterise", "qubit_id": 1, "measurements": ["T1", "T2", "RB"]}))
    assert r.output["data_source"] == "measured_data"
    assert 100 < r.output["T1_us"] < 220
    assert 0.99 < r.output["gate_fidelity"] < 1


def test_characterise_without_data_is_labelled_and_report_is_derived():
    a = LabAutomationAgent()
    r = run(a.run({"action": "characterise", "qubit_id": 0}))
    assert r.output["data_source"] == "simulated_device" and r.warnings
    rep = run(a.run({"action": "generate_report"}))
    assert rep.output["key_metrics"]["T1_us"] == r.output["T1_us"]
    assert rep.output["generated_at"].startswith("20")


def test_report_requires_results():
    assert not run(LabAutomationAgent().run({"action": "generate_report"})).success


def test_closed_loop_and_calibration_improve():
    a = LabAutomationAgent()
    cl = run(a.run({"action": "closed_loop", "qubit_id": 0, "max_iterations": 80}))
    assert cl.output["final_value"] > cl.output["initial_value"]
    cal = run(a.run({"action": "calibrate_qubit", "qubit_id": 0, "target_fidelity": 0.999}))
    assert cal.output["achieved_fidelity"] <= cal.output["coherence_limit"] + 1e-9


# -- literature ----------------------------------------------------------------

def test_local_library_search_offline():
    a = LiteratureAgent()
    a.load_library(load_rows("papers", SAMPLE_DIR / "papers_demo.csv"))
    r = run(a.run({"action": "search", "query": "surface code", "max_results": 3}))
    assert r.output["papers"][0]["source"] == "local"
    assert "surface code" in r.output["papers"][0]["title"].lower()
    assert any("Offline" in w for w in r.warnings)


def test_no_fabricated_papers_when_offline():
    r = run(LiteratureAgent().run({"action": "search", "query": "VQE", "sources": ["arxiv"]}))
    assert r.output["used_demo_library"]
    assert all(not p["authors"][0].startswith("Author") for p in r.output["papers"])


# -- PI ------------------------------------------------------------------------

def test_rank_hypotheses_respects_user_scores_and_weights():
    hyps = [{"statement": "a", "novelty": 0.9, "feasibility": 0.1, "impact": 0.1},
            {"statement": "b", "novelty": 0.1, "feasibility": 0.9, "impact": 0.9}]
    pi = PrincipalInvestigatorAgent()
    r = run(pi.run({"action": "rank_hypotheses", "hypotheses": hyps}))
    assert r.output["ranked_hypotheses"][0]["statement"] == "b"
    r2 = run(pi.run({"action": "rank_hypotheses", "hypotheses": hyps,
                     "weights": {"novelty": 1, "feasibility": 0, "impact": 0}}))
    assert r2.output["ranked_hypotheses"][0]["statement"] == "a"
    r3 = run(pi.run({"action": "rank_hypotheses", "hypotheses": hyps}))
    assert r3.output == r.output  # deterministic


def test_prioritise_experiments_prefers_cheap_valuable():
    exps = [{"name": "costly", "scientific_value": 0.5, "cost_usd": 10000, "estimated_hours": 10},
            {"name": "cheap", "scientific_value": 0.9, "cost_usd": 100, "estimated_hours": 1}]
    r = run(PrincipalInvestigatorAgent().run({"action": "prioritise_experiments", "experiments": exps}))
    assert r.output["prioritised_experiments"][0]["name"] == "cheap"


def test_progress_summary_reflects_plan():
    pi = PrincipalInvestigatorAgent()
    run(pi.run({"action": "generate_research_plan"}))
    s = run(pi.run({"action": "summarise_progress", "completed_milestones": ["Literature review"]})).output
    assert s["n_phases"] > 0 and s["completion_percentage"] == round(100 / s["n_phases"], 1)


# -- analysis / explainability / KG -------------------------------------------

def test_tomography_of_bell_circuit_has_high_fidelity():
    q = "OPENQASM 2.0; qreg q[2]; h q[0]; cx q[0],q[1];"
    r = run(DataAnalysisAgent().run({"action": "state_tomography", "qasm": q, "shots": 20000}))
    assert r.output["fidelity_vs_target"] > 0.98
    assert r.output["pauli_expectations"]["XX"] == pytest.approx(1, abs=0.03)


def test_tomography_from_supplied_counts():
    counts = {b: ({"0": 1000} if b == "Z" else {"0": 500, "1": 500}) for b in "XYZ"}
    r = run(DataAnalysisAgent().run({"action": "state_tomography", "basis_counts": counts, "target_state": [1, 0]}))
    assert r.output["data_source"] == "measured_counts"
    assert r.output["fidelity_vs_target"] == pytest.approx(1, abs=1e-6)


def test_error_analysis_with_real_counts():
    r = run(DataAnalysisAgent().run({"action": "error_analysis", "expected_state": "00",
                                     "results": {"00": 900, "01": 50, "10": 30, "11": 20}}))
    assert r.output["success_probability"] == 0.9
    assert r.output["error_breakdown"]["bit_flip"] == 0.08
    assert not r.warnings


def test_fidelity_computations():
    a = DataAnalysisAgent()
    same = run(a.run({"action": "fidelity", "type": "process", "unitary_a": [[1, 0], [0, 1]], "unitary_b": [[1, 0], [0, 1]]}))
    assert same.output["fidelity"] == 1.0
    st = run(a.run({"action": "fidelity", "type": "state", "state_a": [1, 0], "state_b": [0.70710678, 0.70710678]}))
    assert st.output["fidelity"] == pytest.approx(0.5, abs=1e-6)
    assert not run(a.run({"action": "fidelity", "type": "process"})).success


def test_audit_hash_is_real_and_stable():
    a = ExplainabilityAgent()
    payload = {"action": "audit_decision", "decision_id": "d1", "agent_id": "x",
               "decision": {"inputs": {"a": 1}, "outputs": {"b": 2}}}
    h1 = run(a.run(payload)).output["audit_hash"]
    h2 = run(a.run(payload)).output["audit_hash"]
    assert h1 == h2 and h1.startswith("sha256:") and "placeholder" not in h1


def test_kg_hypotheses_are_deterministic():
    r1 = run(KnowledgeGraphAgent().run({"action": "generate_hypotheses", "n": 3}))
    r2 = run(KnowledgeGraphAgent().run({"action": "generate_hypotheses", "n": 3}))
    strip = lambda hs: [{k: v for k, v in h.items() if k != "entities_involved"} for h in hs]  # noqa: E731
    assert strip(r1.output["hypotheses"]) == strip(r2.output["hypotheses"])  # node ids are random uuids


# -- workflow engine -----------------------------------------------------------

def test_workflow_with_failing_step_finishes_and_skips_dependents():
    eng = WorkflowEngine({"analysis_agent": DataAnalysisAgent(), "pi_agent": PrincipalInvestigatorAgent()})
    wf = eng.create_workflow(custom_tasks=[
        {"name": "bad", "agent_name": "analysis_agent", "input": {"action": "fidelity", "type": "process"}},
        {"name": "after", "agent_name": "pi_agent", "input": {"action": "summarise_progress"}, "deps": ["bad"]},
    ])
    res = asyncio.run(asyncio.wait_for(eng.run(wf.workflow_id), timeout=20))
    assert res["status"] == "completed_with_errors"
    assert wf.tasks["after"].status.value == "skipped"


# -- config ------------------------------------------------------------------

def test_llm_model_comes_from_env(monkeypatch):
    from backend.core.config import Settings

    monkeypatch.delenv("ANTHROPIC_MODEL", raising=False)
    monkeypatch.delenv("LLM_MODEL", raising=False)
    assert Settings().llm_model == "claude-sonnet-5-5"
    monkeypatch.setenv("ANTHROPIC_MODEL", "my-model")
    assert Settings().llm_model == "my-model"
