"""
Optional cloud quantum backends.

Nothing here runs unless BOTH of these are true:

  1. ``QUANTUMAGENT_ENABLE_CLOUD=true`` is set (an explicit opt-in, because
     cloud providers may bill per shot / per second), and
  2. the provider's credential env var is set (see ``.env.example``) and its
     SDK is installed.

Otherwise :func:`get_cloud_adapter` raises :class:`CloudUnavailable` with a
message telling the user exactly what is missing, and the platform keeps
using the local simulator.

Only IBM Quantum (``qiskit-ibm-runtime``) has a working adapter today; the
other providers are listed so the error message is helpful, and adding one
means implementing :class:`CloudAdapter` for its SDK.
"""
from __future__ import annotations

import importlib
import os
from dataclasses import dataclass
from typing import Dict, Optional


class CloudUnavailable(RuntimeError):
    pass


@dataclass(frozen=True)
class ProviderSpec:
    provider: str
    token_env: str
    sdk_module: str
    pip_package: str
    implemented: bool


PROVIDERS: Dict[str, ProviderSpec] = {
    "ibm_quantum": ProviderSpec("ibm_quantum", "IBM_QUANTUM_TOKEN", "qiskit_ibm_runtime", "qiskit-ibm-runtime", True),
    "ionq": ProviderSpec("ionq", "IONQ_API_KEY", "qiskit_ionq", "qiskit-ionq", False),
    "quantinuum": ProviderSpec("quantinuum", "QUANTINUUM_API_KEY", "pytket.extensions.quantinuum", "pytket-quantinuum", False),
    "rigetti": ProviderSpec("rigetti", "RIGETTI_API_KEY", "pyquil", "pyquil", False),
    "aws_braket": ProviderSpec("aws_braket", "AWS_ACCESS_KEY_ID", "braket.aws", "amazon-braket-sdk", False),
    "azure_quantum": ProviderSpec("azure_quantum", "AZURE_QUANTUM_RESOURCE_ID", "azure.quantum", "azure-quantum", False),
}


def cloud_enabled() -> bool:
    return os.getenv("QUANTUMAGENT_ENABLE_CLOUD", "").strip().lower() in ("1", "true", "yes")


def provider_status() -> Dict[str, Dict]:
    """What is configured, without revealing any secret values."""
    out = {}
    for name, spec in PROVIDERS.items():
        try:
            importlib.import_module(spec.sdk_module)
            sdk = True
        except Exception:
            sdk = False
        out[name] = {
            "credential_env": spec.token_env,
            "credential_set": bool(os.getenv(spec.token_env)),
            "sdk_installed": sdk,
            "adapter_implemented": spec.implemented,
            "usable": cloud_enabled() and spec.implemented and sdk and bool(os.getenv(spec.token_env)),
        }
    return {"cloud_enabled": cloud_enabled(), "providers": out}


class CloudAdapter:
    provider = "base"

    def submit(self, qasm: str, shots: int, device: str) -> str:  # pragma: no cover - interface
        raise NotImplementedError

    def status(self, remote_job_id: str) -> str:  # pragma: no cover - interface
        raise NotImplementedError

    def counts(self, remote_job_id: str) -> Dict[str, int]:  # pragma: no cover - interface
        raise NotImplementedError


class IBMQuantumAdapter(CloudAdapter):  # pragma: no cover - needs network + account
    """Runs circuits through IBM Quantum Runtime's SamplerV2."""

    provider = "ibm_quantum"

    def __init__(self) -> None:
        from qiskit_ibm_runtime import QiskitRuntimeService  # type: ignore

        kwargs = {"token": os.environ["IBM_QUANTUM_TOKEN"]}
        channel = os.getenv("IBM_QUANTUM_CHANNEL")
        instance = os.getenv("IBM_QUANTUM_INSTANCE")
        if channel:
            kwargs["channel"] = channel
        if instance:
            kwargs["instance"] = instance
        self.service = QiskitRuntimeService(**kwargs)
        self._jobs = {}

    def submit(self, qasm: str, shots: int, device: str) -> str:
        from qiskit import qasm2, qasm3, transpile  # type: ignore
        from qiskit_ibm_runtime import SamplerV2  # type: ignore

        circuit = qasm3.loads(qasm) if qasm.lstrip().upper().startswith("OPENQASM 3") else qasm2.loads(qasm)
        backend = self.service.backend(device) if device else self.service.least_busy(operational=True, simulator=False)
        job = SamplerV2(mode=backend).run([transpile(circuit, backend)], shots=shots)
        self._jobs[job.job_id()] = job
        return job.job_id()

    def _job(self, remote_job_id: str):
        return self._jobs.get(remote_job_id) or self.service.job(remote_job_id)

    def status(self, remote_job_id: str) -> str:
        return str(self._job(remote_job_id).status()).lower()

    def counts(self, remote_job_id: str) -> Dict[str, int]:
        result = self._job(remote_job_id).result()[0]
        register = next(iter(result.data.values()))
        return dict(register.get_counts())


_ADAPTERS = {"ibm_quantum": IBMQuantumAdapter}


def get_cloud_adapter(provider: str) -> CloudAdapter:
    spec = PROVIDERS.get(provider)
    if spec is None:
        raise CloudUnavailable(f"Unknown cloud provider '{provider}'")
    if not cloud_enabled():
        raise CloudUnavailable(
            f"Cloud execution is disabled. Use backend 'local_simulator' (default), or set "
            f"QUANTUMAGENT_ENABLE_CLOUD=true and {spec.token_env} in .env to use {provider}."
        )
    if not os.getenv(spec.token_env):
        raise CloudUnavailable(f"{provider}: set {spec.token_env} in .env (see .env.example).")
    if not spec.implemented:
        raise CloudUnavailable(
            f"{provider}: no adapter implemented yet (would use {spec.pip_package}). "
            f"Run on 'local_simulator' or 'ibm_quantum' instead."
        )
    try:
        importlib.import_module(spec.sdk_module)
    except Exception:
        raise CloudUnavailable(f"{provider}: install the SDK first: pip install {spec.pip_package}")
    try:
        return _ADAPTERS[provider]()
    except Exception as exc:  # pragma: no cover - depends on remote service
        raise CloudUnavailable(f"{provider}: could not connect ({type(exc).__name__}: {exc})")
