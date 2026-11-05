"""
Route imported datasets to the agent that uses them.

Shared by the REST API (``POST /api/v1/data/import/{kind}``) and the CLI
(``python -m backend.cli import <kind> <file>``).
"""
from __future__ import annotations

from pathlib import Path
from typing import Any, Dict, List, Mapping

from .importers import REPO_ROOT, SCHEMAS, load_rows

SAMPLE_DIR = REPO_ROOT / "data" / "sample_data"

# kind -> (agent name in the registry, action, input key)
ROUTES = {
    "hamiltonian": ("chemistry_agent", "load_hamiltonians"),
    "qubit_measurements": ("lab_automation_agent", "load_measurements"),
    "device_calibration": ("hardware_agent", "register_device"),
    "papers": ("literature_agent", "load_library"),
    "hypotheses": ("pi_agent", "rank_hypotheses"),
    "experiments": ("pi_agent", "prioritise_experiments"),
}

DEMO_FILES = {
    "hamiltonian": "hamiltonians_demo.csv",
    "qubit_measurements": "qubit_measurements_demo.csv",
    "device_calibration": "device_calibration_demo.csv",
    "papers": "papers_demo.csv",
    "hypotheses": "hypotheses_demo.csv",
    "experiments": "experiments_demo.csv",
}


async def load_into_agents(kind: str, rows: List[Dict[str, Any]], registry: Mapping[str, Any]) -> Dict[str, Any]:
    if kind not in ROUTES:
        raise KeyError(f"unknown data kind '{kind}'; choose one of {sorted(ROUTES)}")
    agent_name, action = ROUTES[kind]
    agent = registry[agent_name]
    key = {"hypotheses": "hypotheses", "experiments": "experiments"}.get(kind, "rows")
    if kind == "hypotheses":
        agent._hypotheses = list(rows)  # becomes the default set for later rank_hypotheses calls
    result = await agent.run({"action": action, key: rows})
    return {
        "kind": kind,
        "rows": len(rows),
        "agent": agent_name,
        "action": action,
        "success": result.success,
        "output": result.output,
        "warnings": result.warnings,
    }


async def import_file(kind: str, source, registry: Mapping[str, Any], filename: str = None) -> Dict[str, Any]:
    rows = load_rows(kind, source, filename)
    return await load_into_agents(kind, rows, registry)


async def load_demo_data(registry: Mapping[str, Any], kinds: List[str] = None) -> Dict[str, Any]:
    """Load the bundled demo datasets (synthetic / public metadata) into the agents."""
    summary = {}
    for kind in kinds or list(DEMO_FILES):
        path = SAMPLE_DIR / DEMO_FILES[kind]
        if not Path(path).exists():
            summary[kind] = {"error": f"missing {path.name}; run scripts/make_sample_data.py"}
            continue
        res = await import_file(kind, path, registry)
        summary[kind] = {"rows": res["rows"], "success": res["success"], "file": f"data/sample_data/{path.name}"}
    return summary


def schemas() -> Dict[str, Any]:
    return {k: s.to_dict() for k, s in SCHEMAS.items()}
