"""
QuantumAgent FastAPI Application

Provides REST + WebSocket APIs for:
  - Agent invocation
  - Workflow management
  - Quantum circuit operations
  - Experiment tracking
  - Authentication
"""
from __future__ import annotations

import asyncio
from contextlib import asynccontextmanager
from typing import Any, Dict, List, Optional

from fastapi import FastAPI, HTTPException, Depends, BackgroundTasks, File, UploadFile, WebSocket, WebSocketDisconnect
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse, JSONResponse
import uvicorn

from ..core.config import settings
from ..core.logging import configure_logging, get_logger
from ..core.events import event_bus
from ..agents.pi_agent import PrincipalInvestigatorAgent
from ..agents.literature_agent import LiteratureAgent
from ..agents.circuit_designer import CircuitDesignerAgent
from ..agents.chemistry_agent import QuantumChemistryAgent
from ..agents.lab_automation import LabAutomationAgent
from ..agents.hardware_agent import HardwareIntegrationAgent
from ..agents.optimization_agent import OptimisationAgent
from ..agents.analysis_agent import DataAnalysisAgent
from ..agents.explainability_agent import ExplainabilityAgent
from ..agents.knowledge_graph_agent import KnowledgeGraphAgent
from ..agents.coding_agent import CodingAgent
from ..agents.publication_agent import PublicationAgent
from ..orchestrator.workflow import WorkflowEngine, WORKFLOW_TEMPLATES
from ..importers import ImportValidationError, SCHEMAS, template_path
from ..data_loader import import_file, load_demo_data, schemas as data_schemas

configure_logging(settings.log_level)
logger = get_logger(__name__)


# --------------------------------------------------------------------------- #
#  Agent and workflow registry                                                 #
# --------------------------------------------------------------------------- #

AGENT_REGISTRY = {
    "pi_agent": PrincipalInvestigatorAgent(),
    "literature_agent": LiteratureAgent(),
    "circuit_designer": CircuitDesignerAgent(),
    "chemistry_agent": QuantumChemistryAgent(),
    "lab_automation_agent": LabAutomationAgent(),
    "hardware_agent": HardwareIntegrationAgent(),
    "optimization_agent": OptimisationAgent(),
    "analysis_agent": DataAnalysisAgent(),
    "explainability_agent": ExplainabilityAgent(),
    "knowledge_graph_agent": KnowledgeGraphAgent(),
    "coding_agent": CodingAgent(),
    "publication_agent": PublicationAgent(),
}

workflow_engine = WorkflowEngine(AGENT_REGISTRY)


@asynccontextmanager
async def lifespan(app: FastAPI):
    logger.info("quantumagent_starting", version=settings.app_version)
    dispatch_task = asyncio.create_task(event_bus.dispatch())
    yield
    dispatch_task.cancel()
    logger.info("quantumagent_shutdown")


app = FastAPI(
    title="QuantumAgent",
    description="Multi-Agent AI Platform for Quantum Research and Laboratory Automation",
    version=settings.app_version,
    lifespan=lifespan,
    docs_url="/docs",
    redoc_url="/redoc",
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=settings.cors_origins,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


# --------------------------------------------------------------------------- #
#  WebSocket connection manager                                                #
# --------------------------------------------------------------------------- #

class ConnectionManager:
    def __init__(self):
        self.active: List[WebSocket] = []

    async def connect(self, ws: WebSocket):
        await ws.accept()
        self.active.append(ws)

    def disconnect(self, ws: WebSocket):
        if ws in self.active:
            self.active.remove(ws)

    async def broadcast(self, message: dict):
        for ws in list(self.active):
            try:
                await ws.send_json(message)
            except Exception:
                self.disconnect(ws)


ws_manager = ConnectionManager()


# --------------------------------------------------------------------------- #
#  Health and info endpoints                                                   #
# --------------------------------------------------------------------------- #

@app.get("/", tags=["info"])
async def root():
    return {
        "platform": "QuantumAgent",
        "version": settings.app_version,
        "status": "running",
        "agents": list(AGENT_REGISTRY.keys()),
        "n_agents": len(AGENT_REGISTRY),
        "docs": "/docs",
    }


@app.get("/health", tags=["info"])
async def health():
    return {"status": "healthy", "agents_loaded": len(AGENT_REGISTRY)}


@app.get("/api/v1/agents", tags=["agents"])
async def list_agents():
    return {
        "agents": [agent.status_dict() for agent in AGENT_REGISTRY.values()],
        "total": len(AGENT_REGISTRY),
    }


@app.get("/api/v1/agents/{agent_id}", tags=["agents"])
async def get_agent(agent_id: str):
    agent = AGENT_REGISTRY.get(agent_id)
    if not agent:
        raise HTTPException(404, f"Agent {agent_id} not found")
    return agent.status_dict()


# --------------------------------------------------------------------------- #
#  Agent invocation                                                            #
# --------------------------------------------------------------------------- #

@app.post("/api/v1/agents/{agent_id}/run", tags=["agents"])
async def run_agent(agent_id: str, payload: Dict[str, Any]):
    agent = AGENT_REGISTRY.get(agent_id)
    if not agent:
        raise HTTPException(404, f"Agent {agent_id} not found")
    result = await agent.run(payload, task_name=payload.get("action", agent_id))
    await ws_manager.broadcast({"event": "agent_completed", "agent_id": agent_id, "success": result.success})
    return {
        "agent_id": agent_id,
        "success": result.success,
        "output": result.output,
        "confidence": result.confidence,
        "reasoning": result.reasoning,
        "duration_ms": result.duration_ms,
        "supporting_refs": result.supporting_refs,
        "warnings": result.warnings,
    }


# --------------------------------------------------------------------------- #
#  PI Agent (research planning)                                                #
# --------------------------------------------------------------------------- #

@app.post("/api/v1/research/plan", tags=["research"])
async def generate_research_plan(payload: Dict[str, Any]):
    result = await AGENT_REGISTRY["pi_agent"].run({
        "action": "generate_research_plan",
        **payload,
    })
    return result.output


@app.post("/api/v1/research/hypotheses/rank", tags=["research"])
async def rank_hypotheses(payload: Dict[str, Any]):
    result = await AGENT_REGISTRY["pi_agent"].run({"action": "rank_hypotheses", **payload})
    return result.output


# --------------------------------------------------------------------------- #
#  Literature endpoints                                                        #
# --------------------------------------------------------------------------- #

@app.post("/api/v1/literature/search", tags=["literature"])
async def search_literature(payload: Dict[str, Any]):
    result = await AGENT_REGISTRY["literature_agent"].run({"action": "search", **payload})
    return result.output


@app.post("/api/v1/literature/trends", tags=["literature"])
async def detect_trends(payload: Dict[str, Any]):
    result = await AGENT_REGISTRY["literature_agent"].run({"action": "detect_trends", **payload})
    return result.output


# --------------------------------------------------------------------------- #
#  Quantum circuit endpoints                                                   #
# --------------------------------------------------------------------------- #

@app.post("/api/v1/circuits/design", tags=["quantum"])
async def design_circuit(payload: Dict[str, Any]):
    agent = AGENT_REGISTRY["circuit_designer"]
    action = payload.pop("circuit_type", "design_vqe")
    result = await agent.run({"action": action, **payload})
    return result.output


@app.post("/api/v1/circuits/{circuit_id}/optimise", tags=["quantum"])
async def optimise_circuit(circuit_id: str, payload: Dict[str, Any]):
    result = await AGENT_REGISTRY["circuit_designer"].run({
        "action": "optimise", "circuit_id": circuit_id, **payload
    })
    return result.output


@app.post("/api/v1/circuits/{circuit_id}/transpile", tags=["quantum"])
async def transpile_circuit(circuit_id: str, payload: Dict[str, Any]):
    result = await AGENT_REGISTRY["circuit_designer"].run({
        "action": "transpile", "circuit_id": circuit_id, **payload
    })
    return result.output


# --------------------------------------------------------------------------- #
#  Quantum chemistry endpoints                                                 #
# --------------------------------------------------------------------------- #

@app.post("/api/v1/chemistry/simulate", tags=["chemistry"])
async def simulate_molecule(payload: Dict[str, Any]):
    result = await AGENT_REGISTRY["chemistry_agent"].run({"action": "simulate", **payload})
    return result.output


@app.post("/api/v1/chemistry/hamiltonian", tags=["chemistry"])
async def generate_hamiltonian(payload: Dict[str, Any]):
    result = await AGENT_REGISTRY["chemistry_agent"].run({"action": "generate_hamiltonian", **payload})
    return result.output


@app.post("/api/v1/chemistry/catalyst", tags=["chemistry"])
async def design_catalyst(payload: Dict[str, Any]):
    result = await AGENT_REGISTRY["chemistry_agent"].run({"action": "design_catalyst", **payload})
    return result.output


# --------------------------------------------------------------------------- #
#  Hardware endpoints                                                          #
# --------------------------------------------------------------------------- #

@app.get("/api/v1/hardware/backends", tags=["hardware"])
async def list_backends():
    result = await AGENT_REGISTRY["hardware_agent"].run({"action": "list_backends"})
    return result.output


@app.post("/api/v1/hardware/jobs", tags=["hardware"])
async def submit_job(payload: Dict[str, Any]):
    result = await AGENT_REGISTRY["hardware_agent"].run({"action": "submit_job", **payload})
    return result.output


@app.get("/api/v1/hardware/jobs/{job_id}", tags=["hardware"])
async def get_job(job_id: str):
    result = await AGENT_REGISTRY["hardware_agent"].run({"action": "monitor_job", "job_id": job_id})
    return result.output


@app.post("/api/v1/hardware/select", tags=["hardware"])
async def select_backend(payload: Dict[str, Any]):
    result = await AGENT_REGISTRY["hardware_agent"].run({"action": "select_backend", **payload})
    return result.output


@app.get("/api/v1/hardware/cloud-status", tags=["hardware"])
async def cloud_status():
    result = await AGENT_REGISTRY["hardware_agent"].run({"action": "cloud_status"})
    return result.output


# --------------------------------------------------------------------------- #
#  Bring-your-own data                                                         #
# --------------------------------------------------------------------------- #

MAX_UPLOAD_BYTES = 20 * 1024 * 1024


@app.get("/api/v1/data/schemas", tags=["data"])
async def list_data_schemas():
    """Column schema for every importable dataset."""
    return data_schemas()


@app.get("/api/v1/data/templates/{kind}", tags=["data"])
async def download_template(kind: str):
    if kind not in SCHEMAS:
        raise HTTPException(404, f"Unknown data kind '{kind}'. Choose one of {sorted(SCHEMAS)}")
    return FileResponse(template_path(kind), media_type="text/csv", filename=SCHEMAS[kind].template)


@app.post("/api/v1/data/import/{kind}", tags=["data"])
async def import_data(kind: str, file: UploadFile = File(...)):
    """Upload a CSV / TSV / JSON / XLSX file matching the schema for ``kind``."""
    if kind not in SCHEMAS:
        raise HTTPException(404, f"Unknown data kind '{kind}'. Choose one of {sorted(SCHEMAS)}")
    data = await file.read()
    if len(data) > MAX_UPLOAD_BYTES:
        raise HTTPException(413, "File too large (limit 20 MB)")
    try:
        result = await import_file(kind, data, AGENT_REGISTRY, filename=file.filename)
    except ImportValidationError as exc:
        return JSONResponse(status_code=422, content={"kind": kind, "errors": exc.errors})
    return result


@app.post("/api/v1/data/demo", tags=["data"])
async def load_demo():
    """Load the bundled demo datasets (synthetic curves + public paper metadata)."""
    return await load_demo_data(AGENT_REGISTRY)


# --------------------------------------------------------------------------- #
#  Workflow endpoints                                                          #
# --------------------------------------------------------------------------- #

@app.get("/api/v1/workflows/templates", tags=["workflows"])
async def list_workflow_templates():
    return {"templates": workflow_engine.list_templates()}


@app.post("/api/v1/workflows", tags=["workflows"])
async def create_workflow(payload: Dict[str, Any]):
    wf = workflow_engine.create_workflow(
        template_name=payload.get("template"),
        custom_tasks=payload.get("tasks"),
        name=payload.get("name", "custom_workflow"),
    )
    return {"workflow_id": wf.workflow_id, "name": wf.name, "n_tasks": len(wf.tasks)}


@app.post("/api/v1/workflows/{workflow_id}/run", tags=["workflows"])
async def run_workflow(workflow_id: str, background_tasks: BackgroundTasks):
    background_tasks.add_task(workflow_engine.run, workflow_id)
    return {"workflow_id": workflow_id, "status": "started", "message": "Workflow running in background"}


@app.get("/api/v1/workflows/{workflow_id}", tags=["workflows"])
async def get_workflow_status(workflow_id: str):
    return workflow_engine.get_workflow_status(workflow_id)


@app.post("/api/v1/workflows/{workflow_id}/approve/{task_name}", tags=["workflows"])
async def approve_task(workflow_id: str, task_name: str):
    ok = workflow_engine.approve_task(workflow_id, task_name)
    if not ok:
        raise HTTPException(404, "Workflow or task not found")
    return {"approved": True}


# --------------------------------------------------------------------------- #
#  Optimisation endpoints                                                      #
# --------------------------------------------------------------------------- #

@app.post("/api/v1/optimise/bayesian", tags=["optimisation"])
async def bayesian_optimisation(payload: Dict[str, Any]):
    result = await AGENT_REGISTRY["optimization_agent"].run({"action": "bayesian_opt", **payload})
    return result.output


@app.post("/api/v1/optimise/vqe", tags=["optimisation"])
async def optimise_vqe(payload: Dict[str, Any]):
    result = await AGENT_REGISTRY["optimization_agent"].run({"action": "optimise_vqe", **payload})
    return result.output


# --------------------------------------------------------------------------- #
#  Knowledge graph endpoints                                                   #
# --------------------------------------------------------------------------- #

@app.get("/api/v1/knowledge-graph/stats", tags=["knowledge-graph"])
async def kg_stats():
    result = await AGENT_REGISTRY["knowledge_graph_agent"].run({"action": "stats"})
    return result.output


@app.post("/api/v1/knowledge-graph/query", tags=["knowledge-graph"])
async def kg_query(payload: Dict[str, Any]):
    result = await AGENT_REGISTRY["knowledge_graph_agent"].run({"action": "query", **payload})
    return result.output


@app.post("/api/v1/knowledge-graph/hypotheses", tags=["knowledge-graph"])
async def generate_hypotheses(payload: Dict[str, Any]):
    result = await AGENT_REGISTRY["knowledge_graph_agent"].run({"action": "generate_hypotheses", **payload})
    return result.output


# --------------------------------------------------------------------------- #
#  Code generation endpoints                                                   #
# --------------------------------------------------------------------------- #

@app.post("/api/v1/code/generate", tags=["code"])
async def generate_code(payload: Dict[str, Any]):
    result = await AGENT_REGISTRY["coding_agent"].run({"action": "generate", **payload})
    return result.output


@app.post("/api/v1/code/notebook", tags=["code"])
async def generate_notebook(payload: Dict[str, Any]):
    result = await AGENT_REGISTRY["coding_agent"].run({"action": "generate_notebook", **payload})
    return result.output


# --------------------------------------------------------------------------- #
#  Publication endpoints                                                       #
# --------------------------------------------------------------------------- #

@app.post("/api/v1/publications/draft", tags=["publications"])
async def draft_paper(payload: Dict[str, Any]):
    result = await AGENT_REGISTRY["publication_agent"].run({"action": "draft_paper", **payload})
    return result.output


@app.post("/api/v1/publications/abstract", tags=["publications"])
async def generate_abstract(payload: Dict[str, Any]):
    result = await AGENT_REGISTRY["publication_agent"].run({"action": "abstract", **payload})
    return result.output


# --------------------------------------------------------------------------- #
#  WebSocket (real-time agent monitoring)                                      #
# --------------------------------------------------------------------------- #

@app.websocket("/ws/events")
async def websocket_events(websocket: WebSocket):
    await ws_manager.connect(websocket)
    try:
        while True:
            data = await websocket.receive_text()
            await websocket.send_json({"echo": data, "status": "connected"})
    except WebSocketDisconnect:
        ws_manager.disconnect(websocket)


if __name__ == "__main__":
    uvicorn.run(
        "backend.api.main:app",
        host=settings.host,
        port=settings.port,
        reload=settings.reload,
        workers=1,
    )
