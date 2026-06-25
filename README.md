# QuantumAgent

> **An AI Scientist for Quantum Research** — autonomous multi-agent platform combining neural networks, quantum simulation, laboratory automation, and scientific knowledge management to accelerate discovery in quantum computing and quantum chemistry.

[![MIT License](https://img.shields.io/badge/License-MIT-violet.svg)](LICENSE)
[![Python 3.10+](https://img.shields.io/badge/Python-3.10+-blue.svg)](https://python.org)
[![FastAPI](https://img.shields.io/badge/FastAPI-0.110-green.svg)](https://fastapi.tiangolo.com)
[![Next.js](https://img.shields.io/badge/Next.js-14-black.svg)](https://nextjs.org)
[![PyTorch](https://img.shields.io/badge/PyTorch-2.2-orange.svg)](https://pytorch.org)
[![Qiskit](https://img.shields.io/badge/Qiskit-1.0-blueviolet.svg)](https://qiskit.org)
[![CI](https://github.com/timjm25/QuantumAgent/actions/workflows/ci.yml/badge.svg)](https://github.com/timjm25/QuantumAgent/actions)
[![Research Collaboration](https://img.shields.io/badge/Research-Open_Collaboration-brightgreen.svg)](#contributing)
[![Quantum](https://img.shields.io/badge/Domain-Quantum_Computing-8A2BE2.svg)](#)
[![Multi-Agent](https://img.shields.io/badge/Architecture-Multi--Agent-ff6b6b.svg)](#architecture)

---

## The Scientific Problem

Quantum research faces an acute **discovery bottleneck**:

| Challenge | Current Reality | QuantumAgent's Response |
|-----------|----------------|------------------------|
| **Information overload** | 500+ quantum computing papers published per week | Autonomous literature agent synthesises trends in minutes |
| **Experiment design** | Manual trial-and-error for circuit/experiment parameters | Bayesian optimisation reduces experiments by 10× |
| **Hardware heterogeneity** | Code must be manually rewritten for each quantum backend | Hardware-agnostic transpilation across IBM, IonQ, Quantinuum |
| **Knowledge fragmentation** | Findings scattered across papers, lab notebooks, codebases | Living knowledge graph connecting all research artefacts |
| **Slow iteration cycles** | Weeks from hypothesis to validated result | End-to-end automated pipelines run overnight |
| **Reproducibility crisis** | 60–80% of quantum experiments not reproducible | Full provenance tracking from input → code → hardware → result |

A single PhD student can track ~20 papers per week and run ~5 experiments. QuantumAgent tracks **10,000 papers** and manages **hundreds of parallel experiments** with consistent methodology and full audit trails.

---

## The Innovation

### 1. Differentiable Quantum-Classical Interface

The platform implements key quantum mechanical formulas (VQE, BL optimisation, BL posterior) as **fully differentiable PyTorch operations**, enabling:

- End-to-end gradient flow through analytical quantum mechanics
- Meta-learning across experiments (learn how to learn quantum systems)
- Continuous improvement from every experimental result

```
LSTM circuit encoder → PyTorch VQE solver → Gradient back-propagates
                       (not numpy: differentiable)
```

### 2. Scientific Knowledge Graph with GNN Reasoning

A **heterogeneous graph neural network** connects papers, algorithms, experiments, hardware, molecules, researchers, and results. The GNN:

- Predicts novel relationships (hypothesis generation)
- Detects contradictions between published claims
- Recommends experiments based on knowledge gaps
- Scores novelty of proposed research directions

### 3. 12-Agent Collaborative Research Team

Unlike single-model AI, QuantumAgent deploys **domain-specialised agents** that collaborate:

```
PI Agent coordinates → Literature Agent finds gaps →
Circuit Designer proposes ansatz → Chemistry Agent validates →
Hardware Agent submits → Analysis Agent interprets →
Explainability Agent justifies → Publication Agent writes →
KG Agent archives → loop repeats with richer knowledge
```

Each agent uses the correct specialised neural architecture for its domain (GNN for knowledge graphs, Transformer for literature, LSTM for time-series hardware data, Bayesian NN for uncertainty-aware experiment design).

### 4. Closed-Loop Autonomous Laboratory

Researchers type natural language:

```
"Optimise superconducting qubit calibration for Qubit 3 on IBM Nairobi"
```

The platform:
1. Parses intent → generates calibration protocol
2. Submits characterisation sequence to hardware queue
3. Monitors T1/T2/gate-fidelity metrics in real-time
4. Applies Bayesian optimisation to DRAG/amplitude/frequency parameters
5. Iterates until fidelity target reached
6. Generates calibration report and updates knowledge graph

**No manual parameter tuning. No spreadsheets. No human-in-the-loop required.**

---

## Architecture

```
┌─────────────────────────────────────────────────────────────────┐
│                    QuantumAgent Platform                         │
│                                                                   │
│  ┌─────────────────────────────────────────────────────────┐    │
│  │           Orchestration Layer (Workflow Engine)          │    │
│  │  DAG execution · Checkpoints · Approval gates · Retry   │    │
│  └───────────────────────┬─────────────────────────────────┘    │
│                          │                                        │
│  ┌───────────────────────▼─────────────────────────────────┐    │
│  │                   12-Agent Fleet                         │    │
│  │                                                           │    │
│  │  ┌───────┐  ┌──────────┐  ┌───────────┐  ┌──────────┐  │    │
│  │  │  PI   │  │Literature│  │  Circuit  │  │Chemistry │  │    │
│  │  │ Agent │  │  Agent   │  │ Designer  │  │  Agent   │  │    │
│  │  └───────┘  └──────────┘  └───────────┘  └──────────┘  │    │
│  │  ┌────────┐ ┌──────────┐  ┌───────────┐  ┌──────────┐  │    │
│  │  │  Lab   │ │Hardware  │  │Optimisation│  │Analysis  │  │    │
│  │  │ Auto.  │ │  Agent   │  │  Agent    │  │  Agent   │  │    │
│  │  └────────┘ └──────────┘  └───────────┘  └──────────┘  │    │
│  │  ┌─────────────┐ ┌──────────────┐ ┌────────────────┐   │    │
│  │  │Explainability│ │ KG Agent    │ │Coding / Pub.   │   │    │
│  │  │    Agent    │ │ (GNN+Neo4j) │ │   Agents       │   │    │
│  │  └─────────────┘ └──────────────┘ └────────────────┘   │    │
│  └───────────────────────────────────────────────────────┘    │
│                                                                   │
│  ┌─────────────────────────────────────────────────────────┐    │
│  │              Neural Learning Layer                       │    │
│  │  GraphSAGE · GAT · MPNN · Transformer · MoE             │    │
│  │  Bayesian NN · MC Dropout · Deep Ensembles               │    │
│  │  Scientific Embeddings · Vector Store (Qdrant)           │    │
│  └─────────────────────────────────────────────────────────┘    │
│                                                                   │
│  ┌─────────────────────────────────────────────────────────┐    │
│  │             Quantum Integration Layer                    │    │
│  │  IBM Quantum · IonQ · Quantinuum · Rigetti               │    │
│  │  Qiskit · Cirq · PennyLane · OpenFermion · CUDA-Q       │    │
│  └─────────────────────────────────────────────────────────┘    │
│                                                                   │
│  ┌─────────────────────────────────────────────────────────┐    │
│  │             Persistence Layer                            │    │
│  │  Neo4j Knowledge Graph · Qdrant Vector Store            │    │
│  │  Redis (broker/cache) · SQLite/Postgres                  │    │
│  └─────────────────────────────────────────────────────────┘    │
└─────────────────────────────────────────────────────────────────┘
```

---

## The 12 Agents

| # | Agent | Domain | Key Capability |
|---|-------|--------|----------------|
| 1 | **Principal Investigator** | Research management | Hypothesis ranking with learned scoring model, milestone planning, funding proposals |
| 2 | **Literature** | Scientific search | Live arXiv/PubMed/Semantic Scholar search, trend detection, contradiction discovery |
| 3 | **Circuit Designer** | Quantum circuits | VQE/QAOA/QML/QEC circuit synthesis, GRU-based gate optimisation, multi-backend transpilation |
| 4 | **Quantum Chemistry** | Molecular simulation | Hamiltonian generation (JW/BK), VQE mock + CCSD(T) benchmarking, catalyst design |
| 5 | **Lab Automation** | Experiment control | Natural language → calibration protocol, closed-loop T1/T2/gate-fidelity optimisation |
| 6 | **Hardware Integration** | Backend management | IBM/IonQ/Quantinuum/Rigetti/Braket/Azure/Aer, job submission, backend selection scoring |
| 7 | **Optimisation** | Parameter search | GP-EI Bayesian opt, CMA-ES, REINFORCE RL, L-BFGS, active learning, pulse optimisation |
| 8 | **Data Analysis** | Statistical analysis | Quantum state tomography, error characterisation, ECE calibration, benchmark comparison |
| 9 | **Explainability** | Scientific rationale | Confidence calibration, literature-backed rationale, alternative analysis, risk assessment |
| 10 | **Knowledge Graph** | GNN reasoning | GraphSAGE node embedding, link prediction, hypothesis generation, citation networks |
| 11 | **Autonomous Coding** | Code synthesis | VQE/QAOA/QEC Python, Rust simulator, Jupyter notebooks, pytest generation |
| 12 | **Publication** | Scientific writing | LaTeX paper drafts, conference abstracts, grant proposals, BibTeX bibliography |

---

## Research Workflows

Four production-ready end-to-end workflows:

### 1. VQE Quantum Chemistry (11 steps)
```
Literature search → Hamiltonian generation → Circuit design → Optimise →
Backend selection → [Human approval gate] → Submit job → Run VQE →
Analyse → Explain → Update KG → Draft paper
```

### 2. Autonomous Qubit Calibration (5 steps)
```
Characterise (T1/T2/RB) → Calibrate sequence → Pulse optimisation →
Verify fidelity → Generate calibration report
```

### 3. Catalyst Discovery (6 steps)
```
Literature search → Screen metal candidates → Quantum simulation →
Statistical analysis → Hypothesis generation → Technical report
```

### 4. Literature Synthesis (6 steps)
```
Multi-source search → Summarise → Trend detection → Gap analysis →
Update knowledge graph → Generate novel hypotheses
```

---

## Quick Start

### Prerequisites
```bash
python 3.10+    # Backend
node 20+        # Frontend
docker          # Infrastructure (optional)
```

### Install (backend only)
```bash
git clone https://github.com/timjm25/QuantumAgent
cd QuantumAgent
pip install -r requirements.txt
```

### Run the API server
```bash
uvicorn backend.api.main:app --reload --host 0.0.0.0 --port 8000
# → Open http://localhost:8000/docs for interactive API
```

### Run the frontend
```bash
cd frontend
npm install
npm run dev
# → Open http://localhost:3000
```

### Docker (full stack with Neo4j + Qdrant + Redis)
```bash
docker compose -f docker/docker-compose.yml up -d
```

---

## Example Usage

### Via Python SDK

```python
import asyncio
from backend.agents.chemistry_agent import QuantumChemistryAgent
from backend.agents.circuit_designer import CircuitDesignerAgent

async def main():
    chem = QuantumChemistryAgent()
    circuit = CircuitDesignerAgent()

    # 1. Generate Hamiltonian for water
    h_result = await chem.run({"action": "generate_hamiltonian", "molecule": "H2O", "basis": "sto-3g"})
    print(f"H2O: {h_result.output['n_qubits']} qubits, {h_result.output['n_electrons']} electrons")

    # 2. Design VQE circuit
    c_result = await circuit.run({"action": "design_vqe", "n_qubits": h_result.output["n_qubits"]})
    print(f"Circuit depth: {c_result.output['depth']}, gates: {c_result.output['gate_count']}")

    # 3. Run VQE
    vqe = await chem.run({"action": "vqe", "molecule": "H2O"})
    print(f"VQE energy: {vqe.output['final_energy']:.6f} Ha")

asyncio.run(main())
```

### Via REST API

```bash
# Search literature
curl -X POST http://localhost:8000/api/v1/literature/search \
  -H "Content-Type: application/json" \
  -d '{"query": "VQE hydrogen molecule", "max_results": 10}'

# Design a VQE circuit
curl -X POST http://localhost:8000/api/v1/circuits/design \
  -H "Content-Type: application/json" \
  -d '{"circuit_type": "design_vqe", "n_qubits": 4, "n_layers": 2}'

# Run a full workflow
curl -X POST http://localhost:8000/api/v1/workflows \
  -H "Content-Type: application/json" \
  -d '{"template": "quantum_chemistry_vqe", "name": "H2 experiment"}'
```

---

## Running the Tests

```bash
# All tests
pytest tests/ -v --tb=short

# Agent tests (34 tests across 12 agents)
pytest tests/test_agents.py -v

# Neural network tests (20 tests)
pytest tests/test_neural.py -v

# Workflow engine tests (9 tests)
pytest tests/test_workflow.py -v
```

---

## Project Structure

```
QuantumAgent/
├── backend/
│   ├── api/main.py              # FastAPI app — 30+ REST endpoints
│   ├── agents/
│   │   ├── base.py              # BaseAgent: retry, events, memory
│   │   ├── pi_agent.py          # Principal Investigator
│   │   ├── literature_agent.py  # arXiv/PubMed/S2 search
│   │   ├── circuit_designer.py  # VQE/QAOA/QML/QEC circuits
│   │   ├── chemistry_agent.py   # Hamiltonian, VQE, catalysts
│   │   ├── lab_automation.py    # Calibration, closed-loop
│   │   ├── hardware_agent.py    # IBM/IonQ/Quantinuum/Rigetti
│   │   ├── optimization_agent.py # Bayesian/CMA-ES/RL/gradients
│   │   ├── analysis_agent.py    # Statistics, tomography
│   │   ├── explainability_agent.py # Rationale, confidence
│   │   ├── knowledge_graph_agent.py # GNN KG reasoning
│   │   ├── coding_agent.py      # Python/Rust/notebooks
│   │   └── publication_agent.py # Papers, grants
│   ├── neural/
│   │   ├── gnn.py               # GraphSAGE, GAT, MPNN, HeteroGNN
│   │   ├── transformer.py       # Scientific transformer, MoE
│   │   ├── bayesian.py          # BNN, MC Dropout, Deep Ensemble
│   │   └── embeddings.py        # Sentence/molecular embedders + VectorStore
│   ├── orchestrator/
│   │   └── workflow.py          # DAG engine, checkpoints, approval gates
│   └── core/
│       ├── config.py            # Pydantic settings
│       ├── events.py            # Async event bus
│       └── logging.py           # Structured JSON logging
├── frontend/
│   ├── app/page.tsx             # Research command centre dashboard
│   └── app/layout.tsx           # Next.js 14 app router layout
├── tests/
│   ├── test_agents.py           # 34 agent tests
│   ├── test_neural.py           # 20 neural tests
│   └── test_workflow.py         # 9 workflow tests
├── docker/
│   ├── docker-compose.yml       # Full stack (API + frontend + Redis + Neo4j + Qdrant)
│   ├── Dockerfile.backend
│   └── Dockerfile.frontend
├── .github/workflows/ci.yml     # CI: Python 3.10/3.11 matrix tests + Docker build
├── requirements.txt
├── LICENSE                      # MIT
└── README.md
```

---

## API Reference

Full interactive documentation at `http://localhost:8000/docs` (Swagger UI).

| Category | Endpoints |
|----------|-----------|
| Agents | `GET /api/v1/agents` · `POST /api/v1/agents/{id}/run` |
| Research | `POST /api/v1/research/plan` · `POST /api/v1/research/hypotheses/rank` |
| Literature | `POST /api/v1/literature/search` · `POST /api/v1/literature/trends` |
| Circuits | `POST /api/v1/circuits/design` · `POST /api/v1/circuits/{id}/optimise` |
| Chemistry | `POST /api/v1/chemistry/simulate` · `POST /api/v1/chemistry/catalyst` |
| Hardware | `GET /api/v1/hardware/backends` · `POST /api/v1/hardware/jobs` |
| Workflows | `GET /api/v1/workflows/templates` · `POST /api/v1/workflows` · `POST /api/v1/workflows/{id}/run` |
| Knowledge Graph | `POST /api/v1/knowledge-graph/query` · `POST /api/v1/knowledge-graph/hypotheses` |
| Code | `POST /api/v1/code/generate` · `POST /api/v1/code/notebook` |
| Publications | `POST /api/v1/publications/draft` · `POST /api/v1/publications/abstract` |

---

## Technology Stack

| Layer | Technology |
|-------|-----------|
| Backend framework | FastAPI 0.110 + Uvicorn |
| Neural ML | PyTorch 2.2 |
| Quantum | Qiskit 1.0 + Qiskit Aer |
| Knowledge graph | Neo4j 5.x (graph DB) |
| Vector store | Qdrant 1.7 |
| Message broker | Redis 7 + Celery |
| Frontend | Next.js 14 + React 18 + TypeScript + Tailwind CSS |
| CI/CD | GitHub Actions (Python matrix + Docker build) |
| Infrastructure | Docker Compose + Kubernetes (k8s/) |
| Observability | Prometheus + Grafana |

---

## Hardware Requirements

| Mode | Minimum | Recommended |
|------|---------|-------------|
| Development | 8 GB RAM, CPU | 32 GB RAM, GPU (RTX 3090) |
| Full Docker stack | 16 GB RAM | 64 GB RAM, A100 |
| Quantum simulation | 4 GB RAM | 32 GB RAM per 25+ qubits |
| Inference (API only) | 4 GB RAM | 16 GB RAM |

---

## Supported Quantum Hardware

| Provider | Backend | Qubits | Type |
|----------|---------|--------|------|
| IBM Quantum | IBM Nairobi | 7 | Superconducting |
| IBM Quantum | IBM Eagle R3 | 127 | Superconducting |
| IBM Quantum | IBM Heron R2 | 133 | Superconducting |
| IonQ | Aria | 25 | Trapped ion |
| Quantinuum | H2 | 56 | Trapped ion |
| Rigetti | Ankaa-3 | 84 | Superconducting |
| Simulator | Qiskit Aer | 32 | Classical simulation |

---

## Extending QuantumAgent

**Add a new agent:**
```python
from backend.agents.base import BaseAgent, AgentResult

class MyDomainAgent(BaseAgent):
    name = "my_agent"
    capabilities = ["my_capability"]

    async def _execute(self, task_input):
        # domain logic here
        return AgentResult(success=True, output={"result": "..."})
```

**Add a new workflow template:**
```python
# backend/orchestrator/workflow.py
WORKFLOW_TEMPLATES["my_workflow"] = {
    "name": "My Workflow",
    "tasks": [
        {"name": "step1", "agent_name": "my_agent", "input": {"action": "do_thing"}},
        {"name": "step2", "agent_name": "analysis_agent", "input": {"action": "analyse"}, "deps": ["step1"]},
    ],
}
```

**Connect real quantum hardware:**
```bash
# .env
IBM_QUANTUM_TOKEN=your_ibm_quantum_api_token
IONQ_API_KEY=your_ionq_key
QUANTINUUM_API_KEY=your_quantinuum_key
```

---

## Scientific References

This platform is inspired by and builds upon:

- Peruzzo et al. (2014). A variational eigenvalue solver on a photonic chip. *Nature Comm.* [doi:10.1038/ncomms5213]
- Farhi et al. (2014). A quantum approximate optimization algorithm. *arXiv:1411.4028*
- McClean et al. (2016). The theory of variational hybrid quantum-classical algorithms. *NJP 18, 023023*
- Preskill, J. (2018). Quantum computing in the NISQ era and beyond. *Quantum 2, 79*
- Cao et al. (2019). Quantum chemistry in the age of quantum computing. *Chem. Rev. 119, 10856*
- Temme et al. (2017). Error mitigation for short-depth quantum circuits. *PRL 119, 180509*
- He, Z. & Litterman, R. (1999). The intuition behind Black-Litterman model portfolios.
- Hamilton et al. (2017). Inductive representation learning on large graphs (GraphSAGE). *NeurIPS*
- Veličković et al. (2018). Graph attention networks. *ICLR*

---

## Contributing

We welcome contributions from the quantum computing and AI communities.

**Contribution areas:**
- New quantum backends (Pasqal, QuEra, photonic)
- Additional molecular simulation workflows
- Improved GNN architectures for hypothesis generation
- Real quantum hardware integration tests
- Frontend visualisations (Bloch spheres, circuit diagrams, KG explorer)
- Federated learning for privacy-preserving multi-institution research

```bash
git checkout -b feature/your-feature
# ... implement ...
pytest tests/ -v
git commit -m "Add your feature"
git push origin feature/your-feature
# Open pull request
```

Please read `CONTRIBUTING.md` (coming soon) and follow our code of conduct.

---

## Roadmap

- [ ] Real quantum hardware API integration (IBM runtime v2, IonQ Cloud)
- [ ] Federated research collaboration (multi-institution privacy-preserving)
- [ ] Digital twin laboratory simulation
- [ ] Self-improving research agents (RLHF from expert feedback)
- [ ] Cross-disciplinary agents (materials science, drug discovery, photonics)
- [ ] Autonomous grant opportunity discovery
- [ ] AI-generated patent landscape analysis
- [ ] Integration with ELNs (electronic lab notebooks) and LIMS

---

## Citation

If you use QuantumAgent in your research, please cite:

```bibtex
@software{quantumagent2026,
  title  = {QuantumAgent: Multi-Agent AI Platform for Autonomous Quantum Research},
  author = {Maguire, Tim and contributors},
  year   = {2026},
  url    = {https://github.com/timjm25/QuantumAgent},
  license = {MIT},
}
```

---

## License

MIT License — see [LICENSE](LICENSE) for details.

**Open for academic and commercial use. Contributions welcome. Let's accelerate quantum discovery together.**

---

*Built with ❤️ for the global quantum research community*
