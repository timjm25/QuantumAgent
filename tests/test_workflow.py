"""Tests for the workflow orchestration engine."""
import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import asyncio
import pytest
from backend.orchestrator.workflow import WorkflowEngine, WorkflowStatus, TaskStatus, WORKFLOW_TEMPLATES
from backend.agents.pi_agent import PrincipalInvestigatorAgent
from backend.agents.analysis_agent import DataAnalysisAgent
from backend.agents.circuit_designer import CircuitDesignerAgent


def run(coro):
    return asyncio.run(coro)


MINIMAL_REGISTRY = {
    "pi_agent": PrincipalInvestigatorAgent(),
    "analysis_agent": DataAnalysisAgent(),
    "circuit_designer": CircuitDesignerAgent(),
}


class TestWorkflowEngine:
    def setup_method(self):
        self.engine = WorkflowEngine(MINIMAL_REGISTRY)

    def test_list_templates(self):
        templates = self.engine.list_templates()
        assert len(templates) > 0
        for t in templates:
            assert "template_id" in t
            assert "n_tasks" in t

    def test_create_workflow_from_template(self):
        wf = self.engine.create_workflow(template_name="literature_synthesis")
        assert wf.workflow_id
        assert len(wf.tasks) > 0

    def test_create_custom_workflow(self):
        tasks = [
            {"name": "step1", "agent_name": "pi_agent", "input": {"action": "summarise_progress"}},
            {"name": "step2", "agent_name": "analysis_agent", "input": {"action": "analyse"}, "deps": ["step1"]},
        ]
        wf = self.engine.create_workflow(custom_tasks=tasks, name="custom_test")
        assert len(wf.tasks) == 2

    def test_workflow_status_structure(self):
        wf = self.engine.create_workflow(
            custom_tasks=[{"name": "t1", "agent_name": "pi_agent", "input": {"action": "summarise_progress"}}]
        )
        status = self.engine.get_workflow_status(wf.workflow_id)
        assert "tasks" in status
        assert "progress_pct" in status

    def test_run_simple_workflow(self):
        wf = self.engine.create_workflow(
            custom_tasks=[
                {"name": "plan", "agent_name": "pi_agent", "input": {"action": "generate_research_plan"}},
            ]
        )
        result = run(self.engine.run(wf.workflow_id))
        assert result["status"] == "completed"

    def test_workflow_with_dependencies(self):
        wf = self.engine.create_workflow(
            custom_tasks=[
                {"name": "a", "agent_name": "pi_agent", "input": {"action": "summarise_progress"}},
                {"name": "b", "agent_name": "analysis_agent", "input": {"action": "analyse"}, "deps": ["a"]},
            ]
        )
        result = run(self.engine.run(wf.workflow_id))
        assert result["status"] == "completed"
        status = self.engine.get_workflow_status(wf.workflow_id)
        assert status["tasks"]["a"]["status"] == "completed"
        assert status["tasks"]["b"]["status"] == "completed"

    def test_workflow_not_found(self):
        result = self.engine.get_workflow_status("nonexistent_id")
        assert "error" in result

    def test_approve_task(self):
        wf = self.engine.create_workflow(
            custom_tasks=[{"name": "gated", "agent_name": "pi_agent", "input": {"action": "summarise_progress"}, "requires_approval": False}]
        )
        ok = self.engine.approve_task(wf.workflow_id, "gated")
        assert ok

    def test_all_templates_have_valid_structure(self):
        for name, tmpl in WORKFLOW_TEMPLATES.items():
            assert "name" in tmpl
            assert "tasks" in tmpl
            assert len(tmpl["tasks"]) > 0
            for task in tmpl["tasks"]:
                assert "agent_name" in task
                assert "input" in task


if __name__ == "__main__":
    pytest.main([__file__, "-v", "--tb=short"])
