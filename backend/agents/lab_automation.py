"""
Laboratory Automation Agent

Translates natural-language research intent into:
  - Calibration sequences for superconducting / ion-trap devices
  - Closed-loop optimisation workflows
  - Instrument command sequences
  - Iterative measurement-feedback loops
  - Automated qubit characterisation protocols (T1, T2, RB)
"""
from __future__ import annotations

import asyncio
from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Any, Dict, List, Optional, Tuple

import numpy as np

from .base import BaseAgent, AgentResult
from ..core.logging import get_logger
from ..quantum.characterisation import (
    FITTERS, FitError, SimulatedTransmon, fit_measurements, optimise_pulse,
)

FITTABLE = ("T1", "T2", "T2_echo", "RB")

logger = get_logger(__name__)


@dataclass
class ExperimentStep:
    step_id: str
    description: str
    instrument: str
    command: str
    params: Dict[str, Any] = field(default_factory=dict)
    expected_output: str = ""
    timeout_s: int = 30


@dataclass
class CalibrationProtocol:
    protocol_id: str
    name: str
    target_qubit: int
    steps: List[ExperimentStep] = field(default_factory=list)
    convergence_criterion: str = "δ < 1e-3"
    max_iterations: int = 50


class LabAutomationAgent(BaseAgent):
    """
    Autonomous laboratory controller that interprets high-level research
    intent and translates it into instrument-level experiment plans,
    calibration sequences, and iterative optimisation loops.
    """

    name = "lab_automation_agent"
    description = "Translate research intent → calibration workflows, instrument commands, closed-loop optimisation"
    capabilities = [
        "intent_parsing",
        "calibration_planning",
        "qubit_characterisation",
        "closed_loop_optimisation",
        "instrument_scheduling",
        "robotic_workflow",
        "experiment_reporting",
    ]

    CHARACTERISATION_PROTOCOLS = {
        "T1": "Energy relaxation time measurement via inversion-recovery sequence",
        "T2": "Dephasing time via Ramsey fringe experiment",
        "T2_echo": "Extended coherence via spin-echo (Hahn echo) sequence",
        "RB": "Randomised benchmarking for average gate fidelity",
        "gate_set": "Gate Set Tomography (GST) for complete process characterisation",
        "readout": "Readout fidelity calibration via preparation-measurement matrix",
        "cross_talk": "Cross-talk matrix characterisation via simultaneous single-qubit gates",
        "leakage": "Leakage characterisation via phase randomised benchmarking",
    }

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self._measurements: List[Dict] = []          # validated qubit_measurements rows
        self._devices: Dict[int, SimulatedTransmon] = {}
        self._last_characterisation: Dict[int, Dict] = {}

    def load_measurements(self, rows: List[Dict]) -> Dict:
        """Add validated ``qubit_measurements`` rows (see backend.importers)."""
        self._measurements.extend(rows)
        qubits = sorted({int(r["qubit_id"]) for r in self._measurements})
        return {"n_rows": len(self._measurements), "qubits": qubits}

    def _device(self, qubit_id: int) -> SimulatedTransmon:
        if qubit_id not in self._devices:
            self._devices[qubit_id] = SimulatedTransmon(qubit_id)
        return self._devices[qubit_id]

    async def _execute(self, task_input: Dict[str, Any]) -> AgentResult:
        action = task_input.get("action", "parse_intent")
        dispatch = {
            "load_measurements": self._load_measurements,
            "fit_measurements": self._fit_all,
            "parse_intent": self._parse_and_plan,
            "calibrate_qubit": self._calibrate_qubit,
            "characterise": self._characterise_qubit,
            "closed_loop": self._run_closed_loop,
            "schedule": self._schedule_experiments,
            "generate_report": self._generate_report,
        }
        handler = dispatch.get(action)
        if handler is None:
            return AgentResult(success=False, output={"error": f"Unknown action: {action}"})
        return await handler(task_input)

    # ------------------------------------------------------------------ #

    async def _parse_and_plan(self, inp: Dict) -> AgentResult:
        intent = inp.get("intent", "Optimise superconducting qubit calibration")

        parsed = self._parse_intent(intent)
        protocol = self._build_protocol(parsed)

        return AgentResult(
            success=True,
            output={
                "parsed_intent": parsed,
                "protocol": {
                    "name": protocol["name"],
                    "n_steps": len(protocol["steps"]),
                    "estimated_duration_min": len(protocol["steps"]) * 2,
                    "steps": protocol["steps"][:6],
                    "hardware_requirements": protocol["hardware"],
                },
            },
            confidence=0.87,
            reasoning=f"Parsed intent '{intent[:60]}...' and generated {len(protocol['steps'])}-step protocol.",
        )

    async def _load_measurements(self, inp: Dict) -> AgentResult:
        rows = inp.get("rows") or []
        if not rows:
            return AgentResult(success=False, output={"error": "No measurement rows supplied"})
        info = self.load_measurements(rows)
        return AgentResult(success=True, output=info,
                           reasoning=f"Loaded {len(rows)} measurement points for qubits {info['qubits']}.")

    async def _fit_all(self, inp: Dict) -> AgentResult:
        rows = inp.get("rows") or self._measurements
        if not rows:
            return AgentResult(success=False, output={"error": "No measurement data loaded (import a qubit_measurements file)"})
        fits = fit_measurements(rows)
        table = [self._summarise_fits(q, f, "measured_data") for q, f in fits.items()]
        return AgentResult(success=True, output={"qubits": table, "fits": fits}, confidence=0.95)

    @staticmethod
    def _summarise_fits(qubit_id: int, fits: Dict[str, Dict], source: str) -> Dict:
        return {
            "qubit_id": qubit_id,
            "T1_us": fits.get("T1", {}).get("value"),
            "T2_us": fits.get("T2", {}).get("value"),
            "T2_echo_us": fits.get("T2_echo", {}).get("value"),
            "gate_fidelity": fits.get("RB", {}).get("average_gate_fidelity"),
            "data_source": source,
        }

    async def _calibrate_qubit(self, inp: Dict) -> AgentResult:
        """Sequential calibration against the simulated transmon (no instrument connected).

        Each step does a 1-D scan of one pulse parameter around the current value
        and keeps the best measured fidelity.
        """
        qubit_id = inp.get("qubit_id", 0)
        target_fidelity = inp.get("target_fidelity", 0.999)
        dev = self._device(qubit_id)
        params = {"amplitude": 0.5, "drag_beta": 0.0, "frequency_offset_MHz": 0.0}

        calibration_sequence = [
            ("Fine frequency via Ramsey", "ramsey", "frequency_offset_MHz", np.linspace(-0.5, 0.5, 41)),
            ("Rabi amplitude", "rabi", "amplitude", np.linspace(0.35, 0.65, 61)),
            ("DRAG coefficient", "drag_cal", "drag_beta", np.linspace(-2.0, 2.0, 41)),
            ("Rabi amplitude (fine)", "rabi_fine", "amplitude", None),
        ]

        steps = []
        current_fidelity = 1 - dev.gate_error(**params)
        for name, cmd, key, grid in calibration_sequence:
            if grid is None:
                grid = np.linspace(params[key] - 0.01, params[key] + 0.01, 41)
            scores = [dev.measure_gate_fidelity({**params, key: float(v)}) for v in grid]
            params[key] = float(grid[int(np.argmax(scores))])
            current_fidelity = 1 - dev.gate_error(**params)
            steps.append({
                "step": name,
                "command": cmd,
                "params": {"scan": key, "range": [round(float(grid[0]), 4), round(float(grid[-1]), 4)],
                           "points": len(grid), "best": round(params[key], 5)},
                "result_fidelity": round(current_fidelity, 6),
                "converged": current_fidelity >= target_fidelity,
            })
            if current_fidelity >= target_fidelity:
                break

        result = {
            "qubit_id": qubit_id,
            "target_fidelity": target_fidelity,
            "achieved_fidelity": round(current_fidelity, 6),
            "coherence_limit": round(1 - dev.coherence_limited_error, 6),
            "n_calibration_steps": len(steps),
            "converged": current_fidelity >= target_fidelity,
            "calibrated_params": {k: round(v, 5) for k, v in params.items()},
            "calibration_steps": steps,
            "data_source": "simulated_device",
        }

        return AgentResult(
            success=True,
            output=result,
            confidence=0.9,
            reasoning=f"Calibrated qubit {qubit_id}. Achieved fidelity {current_fidelity:.5f} (target {target_fidelity}).",
            warnings=["Ran against the built-in simulated transmon; connect instrument drivers for a real device."],
        )

    async def _characterise_qubit(self, inp: Dict) -> AgentResult:
        """Fit T1/T2/T2_echo/RB curves.

        Data priority: ``data`` rows in the request, then rows loaded with
        ``load_measurements``, then curves from the simulated device (labelled).
        """
        qubit_id = int(inp.get("qubit_id", 0))
        measurements = inp.get("measurements", list(self.CHARACTERISATION_PROTOCOLS.keys()))
        rows = [r for r in (inp.get("data") or self._measurements) if int(r["qubit_id"]) == qubit_id]

        results: Dict[str, Dict] = {}
        sources = set()
        if rows:
            measured = fit_measurements(rows).get(qubit_id, {})
            for m in measurements:
                if m in measured:
                    results[m] = {**measured[m], "data_source": "measured_data"}
                    sources.add("measured_data")
        dev = self._device(qubit_id)
        for m in measurements:
            if m in results:
                continue
            if m in FITTABLE and not rows:
                x, y = dev.curve(m)
                try:
                    results[m] = {**FITTERS[m](x, y), "data_source": "simulated_device"}
                except FitError as exc:
                    results[m] = {"error": str(exc)}
                sources.add("simulated_device")
            elif m == "readout" and not rows:
                p0, p1 = dev.readout
                results[m] = {"fidelity": round((p0 + p1) / 2, 5), "assignment_error": round(1 - (p0 + p1) / 2, 5),
                              "data_source": "simulated_device"}
                sources.add("simulated_device")
            else:
                results[m] = {"value": None, "status": "not measured (no data for this protocol)"}

        for r in results.values():  # keep the old 'value' key for RB consumers
            if "average_gate_fidelity" in r:
                r.setdefault("value", r["average_gate_fidelity"])

        summary = {
            "qubit_id": qubit_id,
            "T1_us": results.get("T1", {}).get("value"),
            "T2_us": results.get("T2", {}).get("value"),
            "T2_echo_us": results.get("T2_echo", {}).get("value"),
            "gate_fidelity": results.get("RB", {}).get("average_gate_fidelity"),
            "readout_fidelity": results.get("readout", {}).get("fidelity"),
            "data_source": "+".join(sorted(sources)) or "none",
            "full_results": results,
        }
        self._last_characterisation[qubit_id] = summary

        return AgentResult(
            success=True,
            output=summary,
            confidence=0.95 if "measured_data" in sources else 0.6,
            reasoning=f"Characterised qubit {qubit_id} across {len(results)} metrics.",
            warnings=[] if "simulated_device" not in sources else
            ["Some results come from the built-in simulated transmon; import qubit_measurements for real data."],
        )

    async def _run_closed_loop(self, inp: Dict) -> AgentResult:
        objective = inp.get("objective", "maximise_gate_fidelity")
        qubit_id = inp.get("qubit_id", 0)
        max_iterations = inp.get("max_iterations", 50)
        target = inp.get("target", 0.9995)
        start = inp.get("initial_params", {"drag_beta": 0.0, "amplitude": 0.5, "frequency_offset_MHz": 0.0})

        dev = self._device(qubit_id)
        res = optimise_pulse(dev, start, max_iterations=max_iterations, target_fidelity=target)
        warnings = ["Optimised against the built-in simulated transmon (Nelder-Mead on measured fidelity)."]
        if target > res["coherence_limit"]:
            warnings.append(f"Target {target} is above this qubit's coherence limit {res['coherence_limit']}.")

        return AgentResult(
            success=True,
            output={
                "qubit_id": qubit_id,
                "objective": objective,
                "target": target,
                "data_source": "simulated_device",
                **res,
            },
            confidence=0.89,
            warnings=warnings,
        )

    async def _schedule_experiments(self, inp: Dict) -> AgentResult:
        experiments = inp.get("experiments", [])
        hardware = inp.get("hardware", "ibm_nairobi")
        available_windows = inp.get("available_hours", 8)

        scheduled = []
        total_hours = 0.0
        for exp in experiments:
            est_hours = exp.get("estimated_hours", 1.0)
            if total_hours + est_hours <= available_windows:
                scheduled.append({
                    **exp,
                    "scheduled": True,
                    "start_hour": round(total_hours, 1),
                    "end_hour": round(total_hours + est_hours, 1),
                    "hardware": hardware,
                })
                total_hours += est_hours
            else:
                scheduled.append({**exp, "scheduled": False, "reason": "Insufficient time window"})

        return AgentResult(
            success=True,
            output={
                "hardware": hardware,
                "available_hours": available_windows,
                "scheduled_hours": round(total_hours, 1),
                "n_experiments": len([e for e in scheduled if e["scheduled"]]),
                "schedule": scheduled,
            },
            confidence=0.95,
        )

    async def _generate_report(self, inp: Dict) -> AgentResult:
        experiment_id = inp.get("experiment_id", "exp_001")
        qubit_id = inp.get("qubit_id")
        results = dict(inp.get("results") or {})
        if not results and self._last_characterisation:
            qid = qubit_id if qubit_id in self._last_characterisation else next(iter(self._last_characterisation))
            results = self._last_characterisation[qid]
        if not results:
            return AgentResult(success=False, output={
                "error": "No results to report: pass 'results' or run 'characterise' first"})

        metrics = {k: results.get(k) for k in ("T1_us", "T2_us", "T2_echo_us", "gate_fidelity", "readout_fidelity")
                   if results.get(k) is not None}
        recs = []
        t1, t2 = metrics.get("T1_us"), metrics.get("T2_us")
        if t1 and t2 and t2 < 0.5 * t1:
            recs.append(f"T2 ({t2:.1f} us) is well below T1 ({t1:.1f} us): check for low-frequency/flux noise or "
                        "add dynamical decoupling.")
        if t1 and t2 and t2 > 2 * t1 * 1.05:
            recs.append("T2 exceeds 2*T1, which is unphysical: re-check the Ramsey fit and detuning.")
        gf = metrics.get("gate_fidelity")
        if gf is not None and gf < 0.999:
            recs.append(f"Single-qubit gate fidelity {gf:.5f} is below 0.999: rerun amplitude/DRAG calibration.")
        rf = metrics.get("readout_fidelity")
        if rf is not None and rf < 0.97:
            recs.append(f"Readout fidelity {rf:.4f} is below 0.97: re-optimise readout frequency/power and threshold.")
        if not recs:
            recs.append("All tracked metrics are within typical thresholds; keep weekly T1/T2 tracking.")

        summary_bits = [f"{k}={v}" for k, v in metrics.items()]
        report = {
            "experiment_id": experiment_id,
            "report_type": "calibration_summary",
            "generated_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
            "data_source": results.get("data_source", "user_supplied"),
            "executive_summary": "Characterisation results: " + (", ".join(summary_bits) or "no metrics") + ".",
            "key_metrics": metrics,
            "recommendations": recs,
            "figures": ["T1_decay_curve", "ramsey_fringes", "rb_decay_curve"],
        }

        return AgentResult(success=True, output=report, confidence=0.92)

    # ------------------------------------------------------------------ #
    #  Private helpers                                                     #
    # ------------------------------------------------------------------ #

    def _parse_intent(self, intent: str) -> Dict:
        intent_lower = intent.lower()
        protocol_type = "full_characterisation"
        if "calibrat" in intent_lower:
            protocol_type = "calibration"
        elif "t1" in intent_lower or "coherence" in intent_lower:
            protocol_type = "coherence_measurement"
        elif "gate" in intent_lower and "fidel" in intent_lower:
            protocol_type = "gate_benchmarking"
        elif "readout" in intent_lower:
            protocol_type = "readout_calibration"

        return {
            "original_intent": intent,
            "protocol_type": protocol_type,
            "target_qubit": "all" if "all" in intent_lower else 0,
            "priority": "high" if any(w in intent_lower for w in ["urgent", "asap", "critical"]) else "normal",
        }

    def _build_protocol(self, parsed: Dict) -> Dict:
        steps = []
        ptype = parsed["protocol_type"]

        base_steps = {
            "calibration": [
                {"step": "Qubit spectroscopy", "instrument": "VNA", "command": "spectroscopy"},
                {"step": "Rabi oscillations", "instrument": "AWG", "command": "rabi_chevron"},
                {"step": "DRAG calibration", "instrument": "AWG", "command": "drag_cal"},
                {"step": "Ramsey frequency", "instrument": "AWG", "command": "ramsey"},
                {"step": "Readout optimisation", "instrument": "ADC", "command": "readout_cal"},
                {"step": "Single-qubit RB", "instrument": "Controller", "command": "rb_1q"},
            ],
            "coherence_measurement": [
                {"step": "T1 inversion recovery", "instrument": "AWG", "command": "t1_sequence"},
                {"step": "T2 Ramsey", "instrument": "AWG", "command": "ramsey"},
                {"step": "T2 Echo", "instrument": "AWG", "command": "hahn_echo"},
                {"step": "T2* dynamical decoupling", "instrument": "AWG", "command": "dd_sequence"},
            ],
            "gate_benchmarking": [
                {"step": "Single-qubit RB", "instrument": "Controller", "command": "rb_1q"},
                {"step": "Two-qubit RB", "instrument": "Controller", "command": "rb_2q"},
                {"step": "Interleaved RB for CX", "instrument": "Controller", "command": "rb_irb"},
            ],
        }

        steps = base_steps.get(ptype, base_steps["calibration"])

        return {
            "name": ptype.replace("_", " ").title(),
            "steps": steps,
            "hardware": ["AWG", "ADC", "VNA", "qubit_controller"],
        }
