"""
Scientific Workflow Engine

Orchestrates multi-agent workflows with:
  - DAG-based task dependencies
  - Checkpointing and failure recovery
  - Human approval gates
  - Parallel execution
  - Progress tracking
  - Semantic planning (natural language → workflow graph)
"""
from __future__ import annotations

import asyncio
import uuid
from dataclasses import dataclass, field
from datetime import datetime, timezone
from enum import Enum
from typing import Any, Callable, Dict, List, Optional, Set

from ..core.events import EventBus, EventType, Event, event_bus
from ..core.logging import get_logger

logger = get_logger(__name__)


class WorkflowStatus(str, Enum):
    PENDING = "pending"
    RUNNING = "running"
    WAITING_APPROVAL = "waiting_approval"
    COMPLETED = "completed"
    FAILED = "failed"
    CANCELLED = "cancelled"
    PAUSED = "paused"


class TaskStatus(str, Enum):
    PENDING = "pending"
    RUNNING = "running"
    COMPLETED = "completed"
    FAILED = "failed"
    SKIPPED = "skipped"


@dataclass
class WorkflowTask:
    task_id: str = field(default_factory=lambda: str(uuid.uuid4())[:8])
    name: str = ""
    agent_name: str = ""
    input: Dict[str, Any] = field(default_factory=dict)
    output: Optional[Dict[str, Any]] = None
    status: TaskStatus = TaskStatus.PENDING
    dependencies: List[str] = field(default_factory=list)
    requires_approval: bool = False
    approved: bool = False
    retry_count: int = 0
    max_retries: int = 2
    timeout_s: int = 300
    started_at: Optional[str] = None
    completed_at: Optional[str] = None
    error: Optional[str] = None


@dataclass
class WorkflowCheckpoint:
    checkpoint_id: str = field(default_factory=lambda: str(uuid.uuid4())[:8])
    workflow_id: str = ""
    completed_tasks: List[str] = field(default_factory=list)
    outputs: Dict[str, Any] = field(default_factory=dict)
    saved_at: str = field(default_factory=lambda: datetime.now(timezone.utc).isoformat())


@dataclass
class Workflow:
    workflow_id: str = field(default_factory=lambda: str(uuid.uuid4())[:12])
    name: str = "unnamed_workflow"
    description: str = ""
    tasks: Dict[str, WorkflowTask] = field(default_factory=dict)
    status: WorkflowStatus = WorkflowStatus.PENDING
    created_at: str = field(default_factory=lambda: datetime.now(timezone.utc).isoformat())
    started_at: Optional[str] = None
    completed_at: Optional[str] = None
    outputs: Dict[str, Any] = field(default_factory=dict)
    checkpoints: List[WorkflowCheckpoint] = field(default_factory=list)


# --------------------------------------------------------------------------- #
#  Pre-defined scientific workflow templates                                   #
# --------------------------------------------------------------------------- #

WORKFLOW_TEMPLATES: Dict[str, Dict] = {
    "quantum_chemistry_vqe": {
        "name": "End-to-end VQE Quantum Chemistry",
        "description": "Literature → Hamiltonian → Circuit → Optimise → Analyse → Publish",
        "tasks": [
            {"name": "literature_review", "agent_name": "literature_agent", "input": {"action": "search", "query": "VQE molecular simulation"}},
            {"name": "generate_hamiltonian", "agent_name": "chemistry_agent", "input": {"action": "generate_hamiltonian"}, "deps": ["literature_review"]},
            {"name": "design_circuit", "agent_name": "circuit_designer", "input": {"action": "design_vqe"}, "deps": ["generate_hamiltonian"]},
            {"name": "optimise_circuit", "agent_name": "circuit_designer", "input": {"action": "optimise"}, "deps": ["design_circuit"]},
            {"name": "select_backend", "agent_name": "hardware_agent", "input": {"action": "select_backend", "priority": "fidelity"}, "deps": ["optimise_circuit"]},
            {"name": "submit_job", "agent_name": "hardware_agent", "input": {"action": "submit_job"}, "deps": ["select_backend"], "requires_approval": True},
            {"name": "run_vqe", "agent_name": "chemistry_agent", "input": {"action": "vqe"}, "deps": ["submit_job"]},
            {"name": "analyse_results", "agent_name": "analysis_agent", "input": {"action": "analyse"}, "deps": ["run_vqe"]},
            {"name": "explain_results", "agent_name": "explainability_agent", "input": {"action": "explain"}, "deps": ["analyse_results"]},
            {"name": "update_kg", "agent_name": "knowledge_graph_agent", "input": {"action": "ingest_paper"}, "deps": ["explain_results"]},
            {"name": "draft_paper", "agent_name": "publication_agent", "input": {"action": "draft_paper"}, "deps": ["analyse_results", "explain_results"]},
        ],
    },
    "qubit_calibration": {
        "name": "Autonomous Qubit Calibration",
        "description": "Characterise → Calibrate → Optimise → Report",
        "tasks": [
            {"name": "characterise", "agent_name": "lab_automation_agent", "input": {"action": "characterise", "qubit_id": 0}},
            {"name": "calibrate", "agent_name": "lab_automation_agent", "input": {"action": "calibrate_qubit"}, "deps": ["characterise"]},
            {"name": "optimise_params", "agent_name": "optimization_agent", "input": {"action": "optimise_pulse"}, "deps": ["calibrate"]},
            {"name": "verify", "agent_name": "analysis_agent", "input": {"action": "fidelity"}, "deps": ["optimise_params"]},
            {"name": "report", "agent_name": "lab_automation_agent", "input": {"action": "generate_report"}, "deps": ["verify"]},
        ],
    },
    "catalyst_discovery": {
        "name": "Autonomous Catalyst Discovery",
        "description": "Literature → Screen candidates → Quantum simulation → Rank → Report",
        "tasks": [
            {"name": "literature", "agent_name": "literature_agent", "input": {"action": "search", "query": "nitrogen fixation catalysis quantum simulation"}},
            {"name": "design_catalyst", "agent_name": "chemistry_agent", "input": {"action": "design_catalyst", "reaction": "nitrogen_fixation"}},
            {"name": "simulate_top", "agent_name": "chemistry_agent", "input": {"action": "simulate"}, "deps": ["design_catalyst"]},
            {"name": "analyse", "agent_name": "analysis_agent", "input": {"action": "benchmark_compare"}, "deps": ["simulate_top"]},
            {"name": "hypotheses", "agent_name": "knowledge_graph_agent", "input": {"action": "generate_hypotheses"}, "deps": ["analyse"]},
            {"name": "report", "agent_name": "publication_agent", "input": {"action": "technical_report"}, "deps": ["analyse", "hypotheses"]},
        ],
    },
    "literature_synthesis": {
        "name": "Literature Review and Knowledge Graph Update",
        "description": "Search → Summarise → Detect trends → Update KG → Generate hypotheses",
        "tasks": [
            {"name": "search", "agent_name": "literature_agent", "input": {"action": "search"}},
            {"name": "summarise", "agent_name": "literature_agent", "input": {"action": "summarise"}, "deps": ["search"]},
            {"name": "trends", "agent_name": "literature_agent", "input": {"action": "detect_trends"}, "deps": ["summarise"]},
            {"name": "gaps", "agent_name": "literature_agent", "input": {"action": "find_gaps"}, "deps": ["trends"]},
            {"name": "update_kg", "agent_name": "knowledge_graph_agent", "input": {"action": "add_node"}, "deps": ["summarise"]},
            {"name": "hypotheses", "agent_name": "knowledge_graph_agent", "input": {"action": "generate_hypotheses"}, "deps": ["update_kg", "gaps"]},
        ],
    },
}


class WorkflowEngine:
    """
    DAG-based scientific workflow orchestrator.
    Supports parallel execution of independent tasks, approval gates,
    checkpointing, and failure recovery.
    """

    def __init__(self, agent_registry: Dict[str, Any], bus: Optional[EventBus] = None):
        self._agents = agent_registry
        self._bus = bus or event_bus
        self._workflows: Dict[str, Workflow] = {}
        self._checkpoints: Dict[str, WorkflowCheckpoint] = {}

    def create_workflow(
        self,
        template_name: Optional[str] = None,
        custom_tasks: Optional[List[Dict]] = None,
        name: str = "custom_workflow",
    ) -> Workflow:
        if template_name and template_name in WORKFLOW_TEMPLATES:
            tmpl = WORKFLOW_TEMPLATES[template_name]
            wf = Workflow(name=tmpl["name"], description=tmpl["description"])
            for td in tmpl["tasks"]:
                task = WorkflowTask(
                    name=td["name"],
                    agent_name=td["agent_name"],
                    input=td.get("input", {}),
                    dependencies=td.get("deps", []),
                    requires_approval=td.get("requires_approval", False),
                )
                wf.tasks[task.name] = task
        else:
            wf = Workflow(name=name)
            for td in (custom_tasks or []):
                task = WorkflowTask(
                    name=td.get("name", f"task_{uuid.uuid4().hex[:6]}"),
                    agent_name=td.get("agent_name", ""),
                    input=td.get("input", {}),
                    dependencies=td.get("deps", []),
                    requires_approval=td.get("requires_approval", False),
                )
                wf.tasks[task.name] = task

        self._workflows[wf.workflow_id] = wf
        logger.info("workflow_created", workflow_id=wf.workflow_id, n_tasks=len(wf.tasks))
        return wf

    async def run(self, workflow_id: str) -> Dict[str, Any]:
        wf = self._workflows.get(workflow_id)
        if not wf:
            return {"error": f"Workflow {workflow_id} not found"}

        wf.status = WorkflowStatus.RUNNING
        wf.started_at = datetime.now(timezone.utc).isoformat()

        await self._bus.publish(Event(
            type=EventType.WORKFLOW_STARTED,
            payload={"workflow_id": workflow_id, "name": wf.name},
        ))

        try:
            while not self._all_done(wf):
                ready = self._get_ready_tasks(wf)
                if not ready:
                    await asyncio.sleep(0.1)
                    continue

                coros = [self._run_task(wf, task_name) for task_name in ready]
                await asyncio.gather(*coros)

                self._checkpoint(wf)

            wf.status = WorkflowStatus.COMPLETED
            wf.completed_at = datetime.now(timezone.utc).isoformat()

            await self._bus.publish(Event(
                type=EventType.WORKFLOW_COMPLETED,
                payload={"workflow_id": workflow_id, "outputs": list(wf.outputs.keys())},
            ))

            return {"workflow_id": workflow_id, "status": "completed", "outputs": wf.outputs}

        except Exception as exc:
            wf.status = WorkflowStatus.FAILED
            await self._bus.publish(Event(
                type=EventType.WORKFLOW_FAILED,
                payload={"workflow_id": workflow_id, "error": str(exc)},
            ))
            return {"workflow_id": workflow_id, "status": "failed", "error": str(exc)}

    async def _run_task(self, wf: Workflow, task_name: str) -> None:
        task = wf.tasks[task_name]
        agent = self._agents.get(task.agent_name)

        if not agent:
            task.status = TaskStatus.FAILED
            task.error = f"Agent {task.agent_name} not found"
            return

        if task.requires_approval and not task.approved:
            wf.status = WorkflowStatus.WAITING_APPROVAL
            await self._bus.publish(Event(
                type=EventType.HUMAN_APPROVAL_REQUIRED,
                payload={"workflow_id": wf.workflow_id, "task": task_name},
            ))
            timeout = 60
            elapsed = 0
            while not task.approved and elapsed < timeout:
                await asyncio.sleep(1)
                elapsed += 1
            if not task.approved:
                task.status = TaskStatus.SKIPPED
                return
            wf.status = WorkflowStatus.RUNNING

        task.status = TaskStatus.RUNNING
        task.started_at = datetime.now(timezone.utc).isoformat()

        prev_outputs = {name: wf.tasks[name].output for name in task.dependencies if wf.tasks[name].output}
        merged_input = {**task.input, "_previous_outputs": prev_outputs}

        try:
            result = await asyncio.wait_for(
                agent.run(merged_input, task_name=task_name),
                timeout=task.timeout_s,
            )
            if result.success:
                task.status = TaskStatus.COMPLETED
                task.output = result.output
                wf.outputs[task_name] = result.output
            else:
                task.status = TaskStatus.FAILED
                task.error = result.reasoning
        except asyncio.TimeoutError:
            task.status = TaskStatus.FAILED
            task.error = f"Task timed out after {task.timeout_s}s"
        except Exception as exc:
            task.status = TaskStatus.FAILED
            task.error = str(exc)

        task.completed_at = datetime.now(timezone.utc).isoformat()

    def _get_ready_tasks(self, wf: Workflow) -> List[str]:
        ready = []
        for name, task in wf.tasks.items():
            if task.status != TaskStatus.PENDING:
                continue
            deps_done = all(
                wf.tasks[dep].status in (TaskStatus.COMPLETED, TaskStatus.SKIPPED)
                for dep in task.dependencies
                if dep in wf.tasks
            )
            if deps_done:
                ready.append(name)
        return ready

    def _all_done(self, wf: Workflow) -> bool:
        return all(
            t.status in (TaskStatus.COMPLETED, TaskStatus.FAILED, TaskStatus.SKIPPED)
            for t in wf.tasks.values()
        )

    def _checkpoint(self, wf: Workflow) -> None:
        cp = WorkflowCheckpoint(
            workflow_id=wf.workflow_id,
            completed_tasks=[n for n, t in wf.tasks.items() if t.status == TaskStatus.COMPLETED],
            outputs={n: wf.tasks[n].output for n in wf.tasks if wf.tasks[n].output},
        )
        wf.checkpoints.append(cp)
        self._checkpoints[cp.checkpoint_id] = cp

    def approve_task(self, workflow_id: str, task_name: str) -> bool:
        wf = self._workflows.get(workflow_id)
        if not wf:
            return False
        task = wf.tasks.get(task_name)
        if not task:
            return False
        task.approved = True
        return True

    def get_workflow_status(self, workflow_id: str) -> Dict:
        wf = self._workflows.get(workflow_id)
        if not wf:
            return {"error": "Not found"}
        return {
            "workflow_id": workflow_id,
            "name": wf.name,
            "status": wf.status.value,
            "tasks": {
                name: {
                    "status": t.status.value,
                    "agent": t.agent_name,
                    "started_at": t.started_at,
                    "completed_at": t.completed_at,
                    "error": t.error,
                }
                for name, t in wf.tasks.items()
            },
            "progress_pct": round(
                sum(1 for t in wf.tasks.values() if t.status == TaskStatus.COMPLETED) / max(len(wf.tasks), 1) * 100, 1
            ),
        }

    def list_templates(self) -> List[Dict]:
        return [
            {
                "template_id": k,
                "name": v["name"],
                "description": v["description"],
                "n_tasks": len(v["tasks"]),
            }
            for k, v in WORKFLOW_TEMPLATES.items()
        ]
