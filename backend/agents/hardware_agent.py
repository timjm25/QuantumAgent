"""
Hardware Integration Agent

Connects with:
  IBM Quantum, IonQ, Quantinuum, Rigetti, D-Wave, Amazon Braket, Azure Quantum, local simulators

Functions:
  - Submit quantum jobs to backend hardware or simulators
  - Monitor job execution and retrieve results
  - Compare hardware performance metrics
  - Recommend optimal backend for a given circuit
  - Manage job queues and priorities
"""
from __future__ import annotations

import asyncio
import uuid
from dataclasses import dataclass, field
from datetime import datetime, timezone
from enum import Enum
from typing import Any, Dict, List, Optional

import numpy as np

from .base import BaseAgent, AgentResult
from ..core.logging import get_logger
from ..quantum.cloud import CloudUnavailable, get_cloud_adapter, provider_status
from ..quantum.qasm import QasmParseError, parse_qasm
from ..quantum.statevector import Circuit, SimulationError, StatevectorSimulator, bell_circuit

logger = get_logger(__name__)


class BackendType(str, Enum):
    IBM_QUANTUM = "ibm_quantum"
    IONQ = "ionq"
    QUANTINUUM = "quantinuum"
    RIGETTI = "rigetti"
    DWAVE = "dwave"
    AWS_BRAKET = "aws_braket"
    AZURE_QUANTUM = "azure_quantum"
    LOCAL_SIMULATOR = "local_simulator"
    AER_SIMULATOR = "aer_simulator"
    USER_DEVICE = "user_device"


# Backend types that run on this machine (no account, no cost)
LOCAL_TYPES = {BackendType.LOCAL_SIMULATOR, BackendType.AER_SIMULATOR}

# Cloud backend type -> provider key in backend.quantum.cloud.PROVIDERS
CLOUD_PROVIDER = {
    BackendType.IBM_QUANTUM: "ibm_quantum",
    BackendType.IONQ: "ionq",
    BackendType.QUANTINUUM: "quantinuum",
    BackendType.RIGETTI: "rigetti",
    BackendType.AWS_BRAKET: "aws_braket",
    BackendType.AZURE_QUANTUM: "azure_quantum",
}


@dataclass
class HardwareBackend:
    backend_id: str
    name: str
    backend_type: BackendType
    n_qubits: int
    connectivity: str
    native_gates: List[str]
    avg_t1_us: float
    avg_t2_us: float
    cx_error: float
    readout_error: float
    max_shots: int = 100000
    queue_depth: int = 0
    available: bool = True
    gate_error_1q: float = 0.0005
    source: str = "catalogue (static reference values)"
    device_name: str = ""  # provider-side device name for cloud submission


@dataclass
class QuantumJob:
    job_id: str = field(default_factory=lambda: f"job_{uuid.uuid4().hex[:12]}")
    circuit_id: str = ""
    backend_id: str = ""
    shots: int = 8192
    status: str = "queued"
    submitted_at: Optional[str] = None
    completed_at: Optional[str] = None
    counts: Optional[Dict[str, int]] = None
    error: Optional[str] = None
    engine: str = ""
    remote_job_id: Optional[str] = None
    provider: Optional[str] = None
    probabilities: Optional[Dict[str, float]] = None


BACKEND_CATALOGUE: Dict[str, HardwareBackend] = {
    "ibm_nairobi": HardwareBackend(
        "ibm_nairobi", "IBM Nairobi (7Q)", BackendType.IBM_QUANTUM,
        7, "H-graph", ["cx", "rz", "sx", "x", "id", "reset"],
        avg_t1_us=120.5, avg_t2_us=85.3, cx_error=0.0058, readout_error=0.017,
    ),
    "ibm_eagle_r3": HardwareBackend(
        "ibm_eagle_r3", "IBM Eagle R3 (127Q)", BackendType.IBM_QUANTUM,
        127, "Heavy-hex", ["ecr", "rz", "sx", "x", "id", "reset"],
        avg_t1_us=245.0, avg_t2_us=167.0, cx_error=0.0042, readout_error=0.012,
    ),
    "ibm_heron_r2": HardwareBackend(
        "ibm_heron_r2", "IBM Heron R2 (133Q)", BackendType.IBM_QUANTUM,
        133, "Heavy-hex", ["cz", "rz", "sx", "x", "id", "reset"],
        avg_t1_us=310.0, avg_t2_us=220.0, cx_error=0.0028, readout_error=0.008,
    ),
    "ionq_aria": HardwareBackend(
        "ionq_aria", "IonQ Aria (25Q)", BackendType.IONQ,
        25, "All-to-all", ["ms", "gpi", "gpi2", "rz"],
        avg_t1_us=10000.0, avg_t2_us=500.0, cx_error=0.0006, readout_error=0.003,
    ),
    "quantinuum_h2": HardwareBackend(
        "quantinuum_h2", "Quantinuum H2 (56Q)", BackendType.QUANTINUUM,
        56, "All-to-all", ["rzz", "rz", "rx", "ry", "u1q"],
        avg_t1_us=100000.0, avg_t2_us=3000.0, cx_error=0.0004, readout_error=0.002,
    ),
    "rigetti_ankaa": HardwareBackend(
        "rigetti_ankaa", "Rigetti Ankaa-3 (84Q)", BackendType.RIGETTI,
        84, "2D-grid", ["cz", "rx", "rz", "xy"],
        avg_t1_us=95.0, avg_t2_us=68.0, cx_error=0.0080, readout_error=0.025,
    ),
    "aer_simulator": HardwareBackend(
        "aer_simulator", "Qiskit Aer Simulator", BackendType.AER_SIMULATOR,
        32, "All-to-all", ["all"],
        avg_t1_us=1e9, avg_t2_us=1e9, cx_error=0.0, readout_error=0.0,
        max_shots=1000000, gate_error_1q=0.0, source="local (qiskit-aer if installed, else NumPy)",
    ),
    "local_simulator": HardwareBackend(
        "local_simulator", "Local NumPy state-vector simulator", BackendType.LOCAL_SIMULATOR,
        24, "All-to-all", ["all"],
        avg_t1_us=1e9, avg_t2_us=1e9, cx_error=0.0, readout_error=0.0,
        max_shots=10000000, gate_error_1q=0.0, source="local (default, no dependencies)",
    ),
}


class HardwareIntegrationAgent(BaseAgent):
    """
    Manages quantum hardware connections, job submission, monitoring,
    and backend selection across IBM Quantum, IonQ, Quantinuum, Rigetti,
    D-Wave, Braket, Azure, and local simulators.
    """

    name = "hardware_agent"
    description = "Quantum hardware integration: job submission, monitoring, backend selection"
    capabilities = [
        "job_submission",
        "job_monitoring",
        "backend_selection",
        "performance_comparison",
        "queue_management",
        "result_retrieval",
        "noise_characterisation",
    ]

    def __init__(self, seed: Optional[int] = None, **kwargs):
        super().__init__(**kwargs)
        self._jobs: Dict[str, QuantumJob] = {}
        # per-instance copy so registered devices don't leak between agents
        self.catalogue: Dict[str, HardwareBackend] = dict(BACKEND_CATALOGUE)
        self._sim = StatevectorSimulator(seed=seed)

    async def _execute(self, task_input: Dict[str, Any]) -> AgentResult:
        action = task_input.get("action", "list_backends")
        dispatch = {
            "list_backends": self._list_backends,
            "submit_job": self._submit_job,
            "monitor_job": self._monitor_job,
            "get_results": self._get_results,
            "select_backend": self._select_backend,
            "compare_backends": self._compare_backends,
            "cancel_job": self._cancel_job,
            "register_device": self._register_device,
            "cloud_status": self._cloud_status,
        }
        handler = dispatch.get(action)
        if handler is None:
            return AgentResult(success=False, output={"error": f"Unknown action: {action}"})
        return await handler(task_input)

    # ------------------------------------------------------------------ #

    async def _list_backends(self, inp: Dict) -> AgentResult:
        backend_type = inp.get("backend_type", None)
        min_qubits = inp.get("min_qubits", 0)

        backends = [
            self._backend_to_dict(b)
            for b in self.catalogue.values()
            if b.n_qubits >= min_qubits
            and (backend_type is None or b.backend_type.value == backend_type)
        ]

        return AgentResult(
            success=True,
            output={"backends": backends, "total": len(backends)},
            confidence=1.0,
        )

    async def _submit_job(self, inp: Dict) -> AgentResult:
        """Run a circuit.

        Inputs: ``qasm3`` / ``qasm`` (OpenQASM 2 or 3 text) or ``circuit``
        ({"n_qubits", "gates": [{"name", "qubits", "params"}]}), ``shots``,
        ``backend_id`` (default ``local_simulator``) and ``emulate`` (run a
        hardware backend locally with its error rates instead of the cloud).
        """
        prev = [o for o in (inp.get("_previous_outputs") or {}).values() if isinstance(o, dict)]
        if not (inp.get("qasm3") or inp.get("qasm") or inp.get("circuit")):
            upstream_qasm = next((o["qasm3"] for o in prev if o.get("qasm3")), None)
            if upstream_qasm:
                inp = {**inp, "qasm3": upstream_qasm}
        if "backend_id" not in inp:
            rec = next((o["recommended_backend"] for o in prev if o.get("recommended_backend")), None)
            if rec:
                inp = {**inp, "backend_id": rec["backend_id"],
                       "emulate": inp.get("emulate", rec.get("type") not in ("local_simulator", "aer_simulator"))}
        circuit_id = inp.get("circuit_id", "circ_001")
        backend_id = inp.get("backend_id", "local_simulator")
        shots = int(inp.get("shots", 8192))
        emulate = bool(inp.get("emulate", False))
        warnings: List[str] = []

        backend = self.catalogue.get(backend_id)
        if not backend:
            return AgentResult(success=False, output={
                "error": f"Backend {backend_id} not found",
                "available_backends": sorted(self.catalogue)})
        if not 1 <= shots <= backend.max_shots:
            return AgentResult(success=False, output={"error": f"shots must be between 1 and {backend.max_shots}"})

        try:
            circuit, qasm_text = self._circuit_from_input(inp)
        except (QasmParseError, SimulationError, ValueError, KeyError, TypeError) as exc:
            return AgentResult(success=False, output={"error": f"Could not read circuit: {exc}"})
        if circuit is None:
            circuit = bell_circuit()
            qasm_text = ""
            warnings.append("No circuit supplied (qasm3 / qasm / circuit); ran a 2-qubit Bell-state demo circuit.")
        if circuit.n_qubits > backend.n_qubits:
            return AgentResult(success=False, output={
                "error": f"Circuit needs {circuit.n_qubits} qubits; {backend.name} has {backend.n_qubits}"})

        job = QuantumJob(
            circuit_id=circuit_id,
            backend_id=backend_id,
            shots=shots,
            submitted_at=datetime.now(timezone.utc).isoformat(),
        )

        run_locally = backend.backend_type in LOCAL_TYPES or backend.backend_type == BackendType.USER_DEVICE or emulate
        if run_locally:
            try:
                result = self._run_simulation(
                    circuit, shots,
                    noise_from=backend if backend.backend_type not in LOCAL_TYPES else None,
                    prefer_aer=backend.backend_type == BackendType.AER_SIMULATOR,
                )
            except SimulationError as exc:
                return AgentResult(success=False, output={"error": f"Simulation failed: {exc}"})
            job.status = "completed"
            job.counts = result["counts"]
            job.probabilities = result.get("probabilities")
            job.engine = result["engine"]
            job.completed_at = datetime.now(timezone.utc).isoformat()
            if backend.backend_type not in LOCAL_TYPES:
                warnings.append(
                    f"Emulated {backend.name} locally with a depolarising + readout noise model "
                    f"built from its error rates ({backend.source}); no cloud job was sent.")
        else:
            provider = CLOUD_PROVIDER.get(backend.backend_type, backend.backend_type.value)
            try:
                adapter = get_cloud_adapter(provider)
                job.remote_job_id = adapter.submit(qasm_text or self._to_qasm2(circuit), shots,
                                                   inp.get("device_name") or backend.device_name)
            except CloudUnavailable as exc:
                return AgentResult(success=False, output={
                    "error": str(exc), "backend": backend_id,
                    "hint": "Pass emulate=true to run this backend's noise model locally, "
                            "or use backend_id='local_simulator'."})
            except Exception as exc:  # pragma: no cover - remote failures
                return AgentResult(success=False, output={"error": f"Cloud submission failed: {exc}"})
            job.provider = provider
            job.status = "queued"
            job.engine = f"cloud:{provider}"

        self._jobs[job.job_id] = job

        return AgentResult(
            success=True,
            output={
                "job_id": job.job_id,
                "backend": backend_id,
                "status": job.status,
                "shots": shots,
                "n_qubits": circuit.n_qubits,
                "engine": job.engine,
                "estimated_queue_time_min": 0.0 if run_locally else self._estimate_queue_time(backend),
                "counts": job.counts,
                "probabilities": job.probabilities,
            },
            confidence=0.99 if run_locally else 0.9,
            reasoning=f"Submitted job {job.job_id} to {backend.name} ({job.engine}).",
            warnings=warnings,
        )

    async def _monitor_job(self, inp: Dict) -> AgentResult:
        job_id = inp.get("job_id")
        job = self._jobs.get(job_id)
        if not job:
            return AgentResult(success=False, output={"error": f"Job {job_id} not found"})

        if job.remote_job_id and job.status not in ("completed", "cancelled", "failed"):
            try:  # pragma: no cover - needs a live cloud account
                adapter = get_cloud_adapter(job.provider)
                remote = adapter.status(job.remote_job_id)
                if "done" in remote or "completed" in remote:
                    job.counts = adapter.counts(job.remote_job_id)
                    job.status = "completed"
                    job.completed_at = datetime.now(timezone.utc).isoformat()
                elif "error" in remote or "fail" in remote or "cancel" in remote:
                    job.status = "failed"
                    job.error = remote
                else:
                    job.status = "running" if "run" in remote else "queued"
            except CloudUnavailable as exc:
                job.error = str(exc)

        return AgentResult(
            success=True,
            output={
                "job_id": job_id,
                "status": job.status,
                "backend": job.backend_id,
                "engine": job.engine,
                "submitted_at": job.submitted_at,
                "completed_at": job.completed_at,
                "error": job.error,
            },
            confidence=1.0,
        )

    async def _get_results(self, inp: Dict) -> AgentResult:
        job_id = inp.get("job_id")
        job = self._jobs.get(job_id)
        if not job:
            return AgentResult(success=False, output={"error": f"Job {job_id} not found"})

        if job.status != "completed":
            return AgentResult(success=False, output={"error": "Job not yet completed", "status": job.status})

        result = {
            "job_id": job_id,
            "backend": job.backend_id,
            "shots": job.shots,
            "counts": job.counts,
            "most_probable_state": max(job.counts, key=job.counts.get) if job.counts else None,
            "quasi_probabilities": {k: v / job.shots for k, v in (job.counts or {}).items()},
            "completed_at": job.completed_at,
        }

        return AgentResult(success=True, output=result, confidence=0.99)

    async def _select_backend(self, inp: Dict) -> AgentResult:
        n_qubits = inp.get("n_qubits", 4)
        priority = inp.get("priority", "fidelity")
        circuit_depth = inp.get("circuit_depth", 20)
        budget = inp.get("budget_usd", None)

        exclude_sims = bool(inp.get("exclude_simulators", False))
        candidates = [
            b for b in self.catalogue.values()
            if b.n_qubits >= n_qubits and b.available
            and not (exclude_sims and b.backend_type in LOCAL_TYPES)
        ]

        scored = []
        for b in candidates:
            score = self._score_backend(b, priority, circuit_depth)
            scored.append({**self._backend_to_dict(b), "selection_score": round(score, 4)})

        scored.sort(key=lambda x: x["selection_score"], reverse=True)

        return AgentResult(
            success=True,
            output={
                "recommended_backend": scored[0] if scored else None,
                "alternatives": scored[1:4],
                "ranking": scored,
                "selection_criterion": priority,
                "n_qubits_required": n_qubits,
            },
            confidence=0.88,
            reasoning=f"Selected {scored[0]['name'] if scored else 'none'} based on {priority} priority.",
        )

    async def _compare_backends(self, inp: Dict) -> AgentResult:
        backend_ids = inp.get("backend_ids", list(self.catalogue.keys()))

        comparison = []
        for bid in backend_ids:
            b = self.catalogue.get(bid)
            if b:
                comparison.append({
                    "backend": b.name,
                    "n_qubits": b.n_qubits,
                    "cx_error": b.cx_error,
                    "T1_us": b.avg_t1_us,
                    "T2_us": b.avg_t2_us,
                    "readout_error": b.readout_error,
                    "type": b.backend_type.value,
                    "native_gates": b.native_gates[:4],
                })

        return AgentResult(
            success=True,
            output={"comparison": comparison, "n_backends": len(comparison)},
            confidence=0.97,
        )

    async def _register_device(self, inp: Dict) -> AgentResult:
        """Register devices from validated ``device_calibration`` rows (see backend.importers)."""
        rows = inp.get("rows") or []
        if not rows:
            return AgentResult(success=False, output={"error": "No calibration rows supplied (import a device_calibration file)"})
        by_device: Dict[str, List[Dict]] = {}
        for r in rows:
            by_device.setdefault(r["device"], []).append(r)

        registered = []
        for name, rs in by_device.items():
            def avg(key: str, default: float) -> float:
                vals = [float(r[key]) for r in rs if r.get(key) is not None]
                return float(np.mean(vals)) if vals else default
            backend_id = "user_" + "".join(ch if ch.isalnum() else "_" for ch in name.lower()).strip("_")
            b = HardwareBackend(
                backend_id, name, BackendType.USER_DEVICE,
                n_qubits=len({int(r["qubit_id"]) for r in rs}),
                connectivity=next((r["connectivity"] for r in rs if r.get("connectivity")), "unknown"),
                native_gates=["all"],
                avg_t1_us=round(avg("T1_us", 0.0), 3), avg_t2_us=round(avg("T2_us", 0.0), 3),
                cx_error=round(avg("cx_error", 0.01), 6), readout_error=round(avg("readout_error", 0.02), 6),
                gate_error_1q=round(avg("gate_error_1q", 0.001), 7),
                source=f"user calibration import ({len(rs)} qubits)",
            )
            self.catalogue[backend_id] = b
            registered.append(self._backend_to_dict(b))

        return AgentResult(
            success=True,
            output={"registered": registered, "n_devices": len(registered)},
            confidence=1.0,
            reasoning=f"Registered {len(registered)} device(s) from calibration data. "
                      "They can be ranked with select_backend and emulated with submit_job.",
        )

    async def _cloud_status(self, inp: Dict) -> AgentResult:
        return AgentResult(success=True, output=provider_status(), confidence=1.0)

    async def _cancel_job(self, inp: Dict) -> AgentResult:
        job_id = inp.get("job_id")
        job = self._jobs.get(job_id)
        if not job:
            return AgentResult(success=False, output={"error": f"Job {job_id} not found"})
        if job.status in ("queued", "running"):
            job.status = "cancelled"
        return AgentResult(success=True, output={"job_id": job_id, "status": job.status})

    # ------------------------------------------------------------------ #
    #  Private helpers                                                     #
    # ------------------------------------------------------------------ #

    def _circuit_from_input(self, inp: Dict):
        qasm_text = inp.get("qasm3") or inp.get("qasm") or ""
        if qasm_text.strip():
            return parse_qasm(qasm_text), qasm_text
        spec = inp.get("circuit")
        if spec:
            c = Circuit(int(spec["n_qubits"]))
            for g in spec.get("gates", []):
                c.add(g["name"], g["qubits"], g.get("params", []))
            c.measured = list(spec.get("measured") or range(c.n_qubits))
            return c, ""
        return None, ""

    def _run_simulation(self, circuit: Circuit, shots: int,
                        noise_from: Optional[HardwareBackend] = None, prefer_aer: bool = False) -> Dict:
        """Exact state-vector run; optionally apply a simple device noise model.

        Noise model (when ``noise_from`` is given): global depolarising channel
        with fidelity prod(1 - e_gate) over all gates, then independent readout
        bit-flips with probability ``readout_error``. It is a coarse estimate,
        good for comparing devices, not a substitute for a pulse-level model.
        """
        if noise_from is None:
            aer = self._run_with_aer(circuit, shots) if prefer_aer else None
            return aer or self._sim.run(circuit, shots)

        psi = self._sim.statevector(circuit)
        probs = self._sim.marginal_probabilities(self._sim.probabilities(psi), circuit.n_qubits, circuit.measured)
        n_meas = len(circuit.measured) or circuit.n_qubits
        n2 = sum(1 for op in circuit.ops if len(op.qubits) >= 2)
        n1 = len(circuit.ops) - n2
        fidelity = (1 - noise_from.gate_error_1q) ** n1 * (1 - noise_from.cx_error) ** n2
        dim = 2 ** n_meas
        dense = np.zeros(dim)
        for bits, p in probs.items():
            dense[int(bits, 2)] = p
        dense = fidelity * dense + (1 - fidelity) / dim
        e = noise_from.readout_error
        for q in range(n_meas):  # readout flips, one bit at a time
            flipped = dense[np.arange(dim) ^ (1 << q)]
            dense = (1 - e) * dense + e * flipped
        noisy = {format(i, f"0{n_meas}b"): float(p) for i, p in enumerate(dense) if p > 1e-12}
        return {
            "counts": self._sim.sample_counts(noisy, shots),
            "probabilities": {k: round(v, 10) for k, v in noisy.items()},
            "engine": "numpy_statevector+depolarising_readout_noise",
            "estimated_circuit_fidelity": round(fidelity, 6),
        }

    def _run_with_aer(self, circuit: Circuit, shots: int) -> Optional[Dict]:
        """Use qiskit-aer when it is installed (optional); returns None otherwise."""
        try:
            from qiskit import QuantumCircuit as QiskitCircuit, transpile  # type: ignore
            from qiskit_aer import AerSimulator  # type: ignore
        except Exception:
            return None
        qc = QiskitCircuit(circuit.n_qubits, len(circuit.measured))
        alias = {"u3": "u", "u1": "p", "phase": "p", "cnot": "cx", "cu1": "cp", "cphase": "cp",
                 "i": "id", "toffoli": "ccx"}
        for op in circuit.ops:
            name = alias.get(op.name, op.name)
            if name == "u2":
                qc.u(np.pi / 2, op.params[0], op.params[1], op.qubits[0])
                continue
            if name == "cu3":
                qc.cu(*op.params, 0.0, *op.qubits)
                continue
            getattr(qc, name)(*op.params, *op.qubits)
        qc.measure(circuit.measured, list(range(len(circuit.measured))))
        sim = AerSimulator(seed_simulator=int(self._sim.rng.integers(2**31)))
        counts = sim.run(transpile(qc, sim), shots=shots).result().get_counts()
        return {"counts": {k.replace(" ", ""): int(v) for k, v in counts.items()}, "engine": "qiskit_aer"}

    @staticmethod
    def _to_qasm2(circuit: Circuit) -> str:
        lines = ["OPENQASM 2.0;", 'include "qelib1.inc";',
                 f"qreg q[{circuit.n_qubits}];", f"creg c[{len(circuit.measured)}];"]
        for op in circuit.ops:
            params = "(" + ",".join(repr(p) for p in op.params) + ")" if op.params else ""
            lines.append(f"{op.name}{params} " + ",".join(f"q[{q}]" for q in op.qubits) + ";")
        for i, q in enumerate(circuit.measured):
            lines.append(f"measure q[{q}] -> c[{i}];")
        return "\n".join(lines)

    def _estimate_queue_time(self, backend: HardwareBackend) -> float:
        if backend.backend_type in LOCAL_TYPES or backend.backend_type == BackendType.USER_DEVICE:
            return 0.0
        base_times = {
            BackendType.IBM_QUANTUM: 15.0,
            BackendType.IONQ: 30.0,
            BackendType.QUANTINUUM: 45.0,
            BackendType.RIGETTI: 20.0,
            BackendType.AWS_BRAKET: 10.0,
        }
        return base_times.get(backend.backend_type, 20.0) + backend.queue_depth * 2.0

    def _score_backend(self, b: HardwareBackend, priority: str, depth: int) -> float:
        fidelity_score = (1 - b.cx_error) * (1 - b.readout_error)
        speed_score = 1 / (1 + self._estimate_queue_time(b) / 60)
        scale_score = min(1.0, b.n_qubits / 50)

        if priority == "fidelity":
            return fidelity_score * 0.7 + speed_score * 0.2 + scale_score * 0.1
        elif priority == "speed":
            return speed_score * 0.6 + fidelity_score * 0.3 + scale_score * 0.1
        elif priority == "scale":
            return scale_score * 0.5 + fidelity_score * 0.3 + speed_score * 0.2
        return fidelity_score * 0.5 + speed_score * 0.3 + scale_score * 0.2

    def _backend_to_dict(self, b: HardwareBackend) -> Dict:
        return {
            "backend_id": b.backend_id,
            "name": b.name,
            "type": b.backend_type.value,
            "n_qubits": b.n_qubits,
            "connectivity": b.connectivity,
            "cx_error": b.cx_error,
            "readout_error": b.readout_error,
            "T1_us": b.avg_t1_us,
            "T2_us": b.avg_t2_us,
            "native_gates": b.native_gates,
            "available": b.available,
            "gate_error_1q": b.gate_error_1q,
            "source": b.source,
            "runs": "locally" if b.backend_type in LOCAL_TYPES else
                    ("locally (emulated from calibration)" if b.backend_type == BackendType.USER_DEVICE
                     else "cloud (opt-in) or emulate=true"),
        }
