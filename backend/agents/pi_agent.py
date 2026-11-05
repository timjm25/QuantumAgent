"""
Principal Investigator (PI) Agent

Responsibilities:
  - Research planning and project management
  - Hypothesis generation from literature and experimental results
  - Milestone planning and experimental roadmaps
  - Priority ranking of research directions
  - Funding proposal outline generation
  - Coordination of all other agents
"""
from __future__ import annotations

import json
from datetime import datetime, timedelta, timezone
from typing import Any, Dict, List, Optional

import numpy as np
import torch
import torch.nn as nn

from .base import BaseAgent, AgentResult
from ..core.config import settings
from ..core.logging import get_logger

logger = get_logger(__name__)


class HypothesisRanker(nn.Module):
    """
    Small transformer that scores scientific hypotheses by feasibility,
    novelty, and impact given a context embedding.
    """

    def __init__(self, embed_dim: int = 256, heads: int = 4, layers: int = 2):
        super().__init__()
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=embed_dim, nhead=heads, dim_feedforward=embed_dim * 4,
            dropout=0.1, batch_first=True
        )
        self.encoder = nn.TransformerEncoder(encoder_layer, num_layers=layers)
        self.score_head = nn.Sequential(
            nn.LayerNorm(embed_dim),
            nn.Linear(embed_dim, 64),
            nn.GELU(),
            nn.Linear(64, 3),  # [feasibility, novelty, impact]
            nn.Sigmoid(),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        encoded = self.encoder(x)
        return self.score_head(encoded.mean(dim=1))


class PrincipalInvestigatorAgent(BaseAgent):
    """
    Top-level research orchestration agent.  Acts as the 'PI' of the
    AI research team — sets direction, allocates sub-tasks, evaluates
    progress, and maintains the research roadmap.
    """

    name = "pi_agent"
    description = "Principal Investigator — research planning, hypothesis generation, team coordination"
    capabilities = [
        "research_planning",
        "hypothesis_generation",
        "milestone_planning",
        "priority_ranking",
        "funding_proposals",
        "agent_coordination",
    ]

    # Templates for research domains the PI agent understands
    DOMAIN_TEMPLATES = {
        "quantum_computing": {
            "key_challenges": [
                "qubit coherence and decoherence",
                "gate fidelity and error rates",
                "scalability beyond NISQ era",
                "quantum error correction overhead",
                "cross-platform algorithm portability",
            ],
            "methodologies": ["VQE", "QAOA", "QEC codes", "quantum ML", "quantum simulation"],
        },
        "quantum_chemistry": {
            "key_challenges": [
                "accurate electronic structure beyond DFT",
                "reaction pathway discovery",
                "catalyst design",
                "drug-target binding energies",
                "excited state dynamics",
            ],
            "methodologies": ["VQE", "UCCSD", "CASSCF", "DMRG", "full CI"],
        },
        "quantum_materials": {
            "key_challenges": [
                "topological phases of matter",
                "high-Tc superconductivity",
                "strongly correlated electrons",
                "quantum spin liquids",
            ],
            "methodologies": ["tensor networks", "DMFT", "QMC", "VQE for lattice models"],
        },
    }

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.ranker = HypothesisRanker()
        self._projects: Dict[str, Dict] = {}
        self._hypotheses: List[Dict] = []

    async def _execute(self, task_input: Dict[str, Any]) -> AgentResult:
        action = task_input.get("action", "generate_research_plan")

        dispatch = {
            "generate_research_plan": self._generate_research_plan,
            "rank_hypotheses": self._rank_hypotheses,
            "create_milestone_plan": self._create_milestone_plan,
            "generate_funding_outline": self._generate_funding_outline,
            "prioritise_experiments": self._prioritise_experiments,
            "summarise_progress": self._summarise_progress,
        }

        handler = dispatch.get(action)
        if handler is None:
            return AgentResult(
                success=False,
                output={"error": f"Unknown action: {action}"},
                reasoning=f"PI agent does not support action '{action}'",
            )

        return await handler(task_input)

    # ------------------------------------------------------------------ #

    async def _generate_research_plan(self, inp: Dict) -> AgentResult:
        domain = inp.get("domain", "quantum_computing")
        objective = inp.get("objective", "Improve qubit fidelity using machine learning")
        duration_weeks = inp.get("duration_weeks", 52)
        resources = inp.get("resources", {"phd_students": 2, "postdocs": 1, "compute_budget_usd": 50000})

        domain_info = self.DOMAIN_TEMPLATES.get(domain, self.DOMAIN_TEMPLATES["quantum_computing"])

        phases = self._build_phases(objective, duration_weeks, domain_info)
        hypotheses = self._generate_hypotheses(objective, domain_info)
        risks = self._assess_risks(domain, resources)

        project_id = f"proj_{len(self._projects):04d}"
        plan = {
            "project_id": project_id,
            "title": f"Research Plan: {objective}",
            "domain": domain,
            "objective": objective,
            "duration_weeks": duration_weeks,
            "hypotheses": hypotheses,
            "phases": phases,
            "resources": resources,
            "risks": risks,
            "kpis": self._define_kpis(domain),
            "created_at": datetime.now(timezone.utc).isoformat(),
        }

        self._projects[project_id] = plan
        self.remember("current_project", project_id)

        return AgentResult(
            success=True,
            output=plan,
            confidence=0.82,
            reasoning=(
                f"Generated {len(phases)}-phase research plan for '{objective}' in {domain}. "
                f"Identified {len(hypotheses)} testable hypotheses ranked by feasibility and impact."
            ),
            supporting_refs=[
                "Preskill, J. (2018). Quantum Computing in the NISQ Era and Beyond. Quantum, 2, 79.",
                "Cao et al. (2019). Quantum Chemistry in the Age of Quantum Computing. Chem. Rev.",
            ],
        )

    # Transparent keyword cues for hypotheses with no user-supplied scores.
    # (The HypothesisRanker network above is untrained, so it is not used for scoring.)
    SCORE_CUES = {
        "feasibility": {
            "+": ["simulator", "simulation", "small", "h2", "2-qubit", "two-qubit", "benchmark", "existing", "classical"],
            "-": ["fault-tolerant", "fault tolerant", "logical qubit", "100 qubits", "1000", "large-scale", "million"],
        },
        "novelty": {
            "+": ["first", "novel", "new", "hybrid", "combining", "unexplored", "never", "adaptive"],
            "-": ["reproduce", "replicate", "baseline", "known", "standard", "benchmark"],
        },
        "impact": {
            "+": ["chemical accuracy", "advantage", "halves", "order of magnitude", "error correction",
                  "drug", "catalyst", "scal", "fidelity", "reduce"],
            "-": ["toy", "minor", "incremental"],
        },
    }
    DEFAULT_HYPOTHESIS_WEIGHTS = {"feasibility": 0.3, "novelty": 0.3, "impact": 0.4}

    def _heuristic_score(self, statement: str, dim: str, est_weeks=None, est_cost=None):
        text = statement.lower()
        cues = self.SCORE_CUES[dim]
        plus = [c for c in cues["+"] if c in text]
        minus = [c for c in cues["-"] if c in text]
        score = 0.5 + 0.1 * len(plus) - 0.1 * len(minus)
        notes = [f"+{c}" for c in plus] + [f"-{c}" for c in minus]
        if dim == "feasibility":
            if est_weeks is not None:
                adj = 0.15 if est_weeks <= 4 else (-0.15 if est_weeks > 26 else 0.0)
                score += adj
                if adj:
                    notes.append(f"{'+' if adj > 0 else '-'}duration {est_weeks}w")
            if est_cost is not None:
                adj = 0.1 if est_cost <= 1000 else (-0.1 if est_cost > 50000 else 0.0)
                score += adj
                if adj:
                    notes.append(f"{'+' if adj > 0 else '-'}cost ${est_cost:,.0f}")
        return float(np.clip(score, 0.05, 0.95)), notes

    async def _rank_hypotheses(self, inp: Dict) -> AgentResult:
        """Rank hypotheses by weighted feasibility / novelty / impact.

        Scores you supply (0-1) are used as-is; missing ones are estimated
        from transparent keyword cues (see SCORE_CUES) and marked "heuristic".
        """
        hypotheses = inp.get("hypotheses", [])
        if not hypotheses:
            hypotheses = self._hypotheses
        weights = {**self.DEFAULT_HYPOTHESIS_WEIGHTS, **(inp.get("weights") or {})}
        total_w = sum(weights.values()) or 1.0

        scored = []
        n_estimated = 0
        for i, h in enumerate(hypotheses):
            statement = h.get("statement") or h.get("hypothesis") or h.get("text") or ""
            sources, rationale, vals = {}, {}, {}
            for dim in ("feasibility", "novelty", "impact"):
                given = h.get(dim)
                if given is not None and given != "":
                    vals[dim] = float(given)
                    sources[dim] = "user"
                else:
                    vals[dim], rationale[dim] = self._heuristic_score(
                        statement, dim, h.get("est_weeks"), h.get("est_cost_usd"))
                    sources[dim] = "heuristic"
                    n_estimated += 1
            composite = sum(vals[d] * weights.get(d, 0) for d in vals) / total_w
            scored.append({
                **h,
                "hypothesis_id": h.get("hypothesis_id") or f"H{i + 1:02d}",
                "feasibility": round(vals["feasibility"], 3),
                "novelty": round(vals["novelty"], 3),
                "impact": round(vals["impact"], 3),
                "composite_score": round(composite, 3),
                "score_source": sources,
                "heuristic_cues": rationale,
            })

        scored.sort(key=lambda x: x["composite_score"], reverse=True)
        for rank, h in enumerate(scored, start=1):
            h["rank"] = rank

        return AgentResult(
            success=True,
            output={"ranked_hypotheses": scored, "weights": weights},
            confidence=0.85 if n_estimated == 0 else 0.6,
            reasoning=f"Ranked {len(scored)} hypotheses by weighted feasibility, novelty and impact "
                      f"({n_estimated} score(s) estimated heuristically).",
            warnings=[] if n_estimated == 0 else
            ["Some scores were estimated from keyword cues; supply novelty/feasibility/impact for a defensible ranking."],
        )

    async def _create_milestone_plan(self, inp: Dict) -> AgentResult:
        project_id = inp.get("project_id", self.recall("current_project", "proj_0000"))
        project = self._projects.get(project_id, {})
        duration = inp.get("duration_weeks", project.get("duration_weeks", 52))

        start = datetime.now(timezone.utc)
        milestones = []
        for i, phase in enumerate(["Literature review", "Hypothesis refinement",
                                    "Simulation campaign", "Hardware experiments",
                                    "Analysis & publication"]):
            week_offset = int(duration * (i / 5))
            milestones.append({
                "milestone_id": f"M{i+1:02d}",
                "phase": phase,
                "target_date": (start + timedelta(weeks=week_offset)).strftime("%Y-%m-%d"),
                "deliverables": self._phase_deliverables(phase),
                "success_criteria": self._phase_criteria(phase),
            })

        return AgentResult(
            success=True,
            output={"project_id": project_id, "milestones": milestones},
            confidence=0.88,
            reasoning=f"Created {len(milestones)} milestones over {duration} weeks.",
        )

    async def _generate_funding_outline(self, inp: Dict) -> AgentResult:
        project_id = inp.get("project_id", self.recall("current_project", "unknown"))
        agency = inp.get("agency", "NSF")
        program = inp.get("program", "Quantum Information Science")

        outline = {
            "agency": agency,
            "program": program,
            "sections": {
                "executive_summary": (
                    "This proposal develops AI-driven quantum research acceleration tools "
                    "combining multi-agent neural systems with quantum hardware integration."
                ),
                "broader_impact": (
                    "The platform reduces time-to-discovery by an estimated 10×, democratising "
                    "quantum research for under-resourced institutions."
                ),
                "intellectual_merit": (
                    "Novel differentiable quantum–classical interfaces enable end-to-end "
                    "gradient-based optimisation of quantum experiments."
                ),
                "budget_justification": {
                    "personnel": "2 PhD students × $35k/yr stipend + fringe",
                    "compute": "$50k cloud HPC + $20k quantum hardware credits",
                    "travel": "$10k conferences (APS March Meeting, QIP)",
                    "indirect": "University F&A at 54%",
                },
                "evaluation_plan": "Quarterly milestone reviews with external advisory board.",
            },
        }

        return AgentResult(
            success=True,
            output=outline,
            confidence=0.79,
            reasoning=f"Generated funding outline for {agency} {program} program.",
        )

    async def _prioritise_experiments(self, inp: Dict) -> AgentResult:
        """Weighted score: scientific value, success probability, low cost, short duration."""
        experiments = inp.get("experiments", [])
        weights = {"scientific_value": 0.4, "success_probability": 0.25, "cost": 0.2, "time": 0.15,
                   **(inp.get("weights") or {})}
        if not experiments:
            return AgentResult(success=False, output={"error": "No experiments supplied"})

        costs = [float(e.get("cost_usd") or 0) for e in experiments]
        hours = [float(e.get("estimated_hours") or 0) for e in experiments]
        max_cost, max_hours = max(costs) or 1.0, max(hours) or 1.0
        total_w = sum(weights.values()) or 1.0

        prioritised = []
        for exp, c, t in zip(experiments, costs, hours):
            value = float(exp.get("scientific_value", 0.5) if exp.get("scientific_value") is not None else 0.5)
            p_success = float(exp.get("success_probability") if exp.get("success_probability") is not None else 0.7)
            parts = {
                "scientific_value": value,
                "success_probability": p_success,
                "cost": 1 - c / max_cost,
                "time": 1 - t / max_hours,
            }
            score = sum(parts[k] * weights.get(k, 0) for k in parts) / total_w
            drivers = sorted(parts, key=lambda k: -parts[k] * weights.get(k, 0))[:2]
            prioritised.append({
                **exp,
                "priority_score": round(score, 3),
                "score_breakdown": {k: round(v, 3) for k, v in parts.items()},
                "rationale": "Driven by " + " and ".join(d.replace("_", " ") for d in drivers) + ".",
            })

        prioritised.sort(key=lambda x: x["priority_score"], reverse=True)
        for i, p in enumerate(prioritised):
            p["recommended_order"] = i + 1

        return AgentResult(
            success=True,
            output={"prioritised_experiments": prioritised, "weights": weights},
            confidence=0.85,
        )

    async def _summarise_progress(self, inp: Dict) -> AgentResult:
        """Progress from what this agent actually knows: the stored plan, the
        milestones you mark complete (``completed_milestones``), and its task log."""
        project_id = inp.get("project_id", self.recall("current_project", "unknown"))
        project = self._projects.get(project_id, {})
        phases = project.get("phases", [])
        completed = list(inp.get("completed_milestones", []))
        achievements = list(inp.get("achievements", []))
        blockers = list(inp.get("blockers", []))

        n_phases = len(phases)
        completion = round(100.0 * min(len(completed), n_phases) / n_phases, 1) if n_phases else None
        elapsed = None
        if project.get("created_at") and project.get("duration_weeks"):
            started = datetime.fromisoformat(project["created_at"])
            weeks = (datetime.now(timezone.utc) - started).days / 7
            elapsed = round(min(100.0, 100.0 * weeks / project["duration_weeks"]), 1)

        if completion is None:
            status = "no_plan"
        elif elapsed is not None and completion + 10 < elapsed:
            status = "behind_schedule"
        else:
            status = "on_track"

        done_tasks = [t for t in self._task_history if t.output is not None]
        next_actions = [p.get("name", str(p)) for p in phases[len(completed):len(completed) + 3]] if phases else \
            ["Create a research plan (action generate_research_plan)"]

        summary = {
            "project_id": project_id,
            "title": project.get("title"),
            "status": status,
            "completion_percentage": completion,
            "schedule_elapsed_pct": elapsed,
            "n_phases": n_phases,
            "n_hypotheses": len(project.get("hypotheses", [])),
            "completed_milestones": completed,
            "key_achievements": achievements,
            "blockers": blockers,
            "next_actions": next_actions,
            "agent_tasks_completed": len(done_tasks),
            "generated_at": datetime.now(timezone.utc).isoformat(),
        }

        return AgentResult(success=True, output=summary, confidence=0.9 if project else 0.5,
                           warnings=[] if project else [f"No stored plan for project '{project_id}'."])

    # ------------------------------------------------------------------ #
    #  Private helpers                                                     #
    # ------------------------------------------------------------------ #

    def _generate_hypotheses(self, objective: str, domain_info: Dict) -> List[Dict]:
        hypotheses = []
        for i, challenge in enumerate(domain_info["key_challenges"][:4]):
            hypotheses.append({
                "hypothesis_id": f"H{i+1:02d}",
                "statement": f"Addressing {challenge} via neural-assisted quantum optimisation will improve performance by ≥15%.",
                "testable": True,
                "required_methodology": domain_info["methodologies"][i % len(domain_info["methodologies"])],
                "expected_timeline_weeks": 8 + i * 4,
            })
        self._hypotheses = hypotheses
        return hypotheses

    def _build_phases(self, objective: str, duration: int, domain_info: Dict) -> List[Dict]:
        phase_fractions = [0.15, 0.10, 0.35, 0.25, 0.15]
        phase_names = [
            "Literature Review & Gap Analysis",
            "Hypothesis Refinement",
            "Computational Simulation Campaign",
            "Hardware Experiment Execution",
            "Analysis, Validation & Publication",
        ]
        phases = []
        start_week = 0
        for name, frac in zip(phase_names, phase_fractions):
            phase_weeks = max(2, int(duration * frac))
            phases.append({
                "name": name,
                "start_week": start_week,
                "duration_weeks": phase_weeks,
                "agent_assignments": self._assign_agents_to_phase(name),
            })
            start_week += phase_weeks
        return phases

    def _assign_agents_to_phase(self, phase: str) -> List[str]:
        mapping = {
            "Literature Review": ["literature_agent", "knowledge_graph_agent"],
            "Hypothesis Refinement": ["pi_agent", "literature_agent", "explainability_agent"],
            "Computational Simulation": ["circuit_designer", "chemistry_agent", "optimization_agent"],
            "Hardware Experiment": ["hardware_agent", "lab_automation_agent", "analysis_agent"],
            "Analysis": ["analysis_agent", "explainability_agent", "publication_agent"],
        }
        for key, agents in mapping.items():
            if key in phase:
                return agents
        return ["pi_agent"]

    def _assess_risks(self, domain: str, resources: Dict) -> List[Dict]:
        return [
            {"risk": "Hardware queue delays on IBM Quantum", "likelihood": "high", "mitigation": "Use simulators as fallback"},
            {"risk": "Simulation convergence failures for large molecules", "likelihood": "medium", "mitigation": "Adaptive basis set selection"},
            {"risk": "Data quality issues in literature corpus", "likelihood": "low", "mitigation": "Cross-validate with multiple sources"},
        ]

    def _define_kpis(self, domain: str) -> List[Dict]:
        return [
            {"kpi": "Circuit gate count reduction", "target": "≥30%", "measurement": "Automated circuit analysis"},
            {"kpi": "Energy estimation accuracy vs CCSD(T)", "target": "Chemical accuracy ≤1 kcal/mol", "measurement": "Benchmark suite"},
            {"kpi": "Experiment-to-result latency", "target": "≤2h end-to-end", "measurement": "Platform telemetry"},
            {"kpi": "Publications", "target": "≥2 peer-reviewed papers per year", "measurement": "Publication tracker"},
        ]

    def _phase_deliverables(self, phase: str) -> List[str]:
        mapping = {
            "Literature review": ["Structured literature database", "Research gap map", "Annotated bibliography"],
            "Hypothesis refinement": ["Ranked hypothesis list", "Experimental design document"],
            "Simulation campaign": ["Simulation dataset", "Performance benchmarks", "Circuit library"],
            "Hardware experiments": ["Raw hardware results", "Error characterisation report"],
            "Analysis": ["Peer-reviewed paper draft", "Open-source code release", "Final project report"],
        }
        for key, deliverables in mapping.items():
            if key.lower() in phase.lower():
                return deliverables
        return ["Milestone report"]

    def _phase_criteria(self, phase: str) -> List[str]:
        return [f"All deliverables complete and peer-reviewed", f"No critical open issues for {phase}"]
