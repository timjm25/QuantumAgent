"use client";

import React, { useEffect, useState } from "react";

interface AgentStatus {
  agent_id: string;
  name: string;
  description: string;
  status: string;
  tasks_completed: number;
  tasks_failed: number;
  capabilities: string[];
}

interface PlatformInfo {
  platform: string;
  version: string;
  status: string;
  n_agents: number;
}

const STATUS_COLORS: Record<string, string> = {
  idle: "bg-gray-500",
  running: "bg-blue-500 animate-pulse",
  completed: "bg-green-500",
  failed: "bg-red-500",
  paused: "bg-yellow-500",
};

const AGENT_ICONS: Record<string, string> = {
  pi_agent: "🎓",
  literature_agent: "📚",
  circuit_designer: "⭕",
  chemistry_agent: "⚗️",
  lab_automation_agent: "🤖",
  hardware_agent: "💻",
  optimization_agent: "🎯",
  analysis_agent: "📊",
  explainability_agent: "💡",
  knowledge_graph_agent: "🕸️",
  coding_agent: "💻",
  publication_agent: "📄",
};

export default function Dashboard() {
  const [agents, setAgents] = useState<AgentStatus[]>([]);
  const [platform, setPlatform] = useState<PlatformInfo | null>(null);
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState<string | null>(null);

  const API_URL = process.env.NEXT_PUBLIC_API_URL || "http://localhost:8000";

  useEffect(() => {
    const fetchData = async () => {
      try {
        const [infoRes, agentsRes] = await Promise.all([
          fetch(`${API_URL}/`),
          fetch(`${API_URL}/api/v1/agents`),
        ]);
        if (infoRes.ok) setPlatform(await infoRes.json());
        if (agentsRes.ok) {
          const data = await agentsRes.json();
          setAgents(data.agents || []);
        }
      } catch (e) {
        setError("Backend not reachable — start with: uvicorn backend.api.main:app --reload");
      } finally {
        setLoading(false);
      }
    };
    fetchData();
    const interval = setInterval(fetchData, 10000);
    return () => clearInterval(interval);
  }, []);

  const mockAgents: AgentStatus[] = [
    { agent_id: "pi_agent", name: "pi_agent", description: "Principal Investigator — research planning and coordination", status: "idle", tasks_completed: 12, tasks_failed: 0, capabilities: ["research_planning", "hypothesis_generation"] },
    { agent_id: "literature_agent", name: "literature_agent", description: "Scientific literature search and synthesis", status: "idle", tasks_completed: 47, tasks_failed: 2, capabilities: ["arxiv_search", "trend_detection"] },
    { agent_id: "circuit_designer", name: "circuit_designer", description: "Quantum circuit design and optimisation", status: "idle", tasks_completed: 23, tasks_failed: 1, capabilities: ["vqe_design", "qaoa_design", "transpilation"] },
    { agent_id: "chemistry_agent", name: "chemistry_agent", description: "Quantum chemistry simulation", status: "idle", tasks_completed: 8, tasks_failed: 0, capabilities: ["vqe", "hamiltonian_generation"] },
    { agent_id: "hardware_agent", name: "hardware_agent", description: "Quantum hardware integration", status: "idle", tasks_completed: 31, tasks_failed: 3, capabilities: ["ibm_quantum", "ionq", "job_submission"] },
    { agent_id: "optimization_agent", name: "optimization_agent", description: "Bayesian and RL optimisation", status: "idle", tasks_completed: 19, tasks_failed: 0, capabilities: ["bayesian_opt", "cma_es", "rl"] },
    { agent_id: "analysis_agent", name: "analysis_agent", description: "Data analysis and visualisation", status: "idle", tasks_completed: 34, tasks_failed: 1, capabilities: ["statistics", "tomography"] },
    { agent_id: "explainability_agent", name: "explainability_agent", description: "Scientific rationale and confidence", status: "idle", tasks_completed: 28, tasks_failed: 0, capabilities: ["rationale", "references"] },
    { agent_id: "knowledge_graph_agent", name: "knowledge_graph_agent", description: "Scientific knowledge graph", status: "idle", tasks_completed: 15, tasks_failed: 0, capabilities: ["entity_extraction", "gnn_reasoning"] },
    { agent_id: "lab_automation_agent", name: "lab_automation_agent", description: "Laboratory automation and calibration", status: "idle", tasks_completed: 6, tasks_failed: 0, capabilities: ["calibration", "closed_loop"] },
    { agent_id: "coding_agent", name: "coding_agent", description: "Code generation and review", status: "idle", tasks_completed: 22, tasks_failed: 2, capabilities: ["python", "qiskit", "notebooks"] },
    { agent_id: "publication_agent", name: "publication_agent", description: "Paper drafts and publications", status: "idle", tasks_completed: 5, tasks_failed: 0, capabilities: ["paper_draft", "grant_proposal"] },
  ];

  const displayAgents = agents.length > 0 ? agents : mockAgents;

  const totalCompleted = displayAgents.reduce((s, a) => s + a.tasks_completed, 0);
  const totalFailed = displayAgents.reduce((s, a) => s + a.tasks_failed, 0);

  return (
    <div className="p-8">
      {/* Header */}
      <div className="mb-8">
        <h2 className="text-3xl font-bold text-white mb-2">Research Command Centre</h2>
        <p className="text-gray-400">
          {platform
            ? `QuantumAgent v${platform.version} — ${platform.n_agents} agents loaded`
            : "Multi-agent AI platform for autonomous quantum research"}
        </p>
        {error && (
          <div className="mt-3 px-4 py-3 bg-yellow-900/30 border border-yellow-700 rounded-lg text-yellow-300 text-sm">
            ⚠️ {error}
          </div>
        )}
      </div>

      {/* Stats */}
      <div className="grid grid-cols-4 gap-4 mb-8">
        {[
          { label: "Active Agents", value: "12", color: "violet" },
          { label: "Tasks Completed", value: totalCompleted.toString(), color: "green" },
          { label: "Tasks Failed", value: totalFailed.toString(), color: "red" },
          { label: "Workflows Available", value: "4", color: "blue" },
        ].map(({ label, value, color }) => (
          <div key={label} className="bg-gray-900 rounded-xl p-5 border border-gray-800">
            <p className="text-gray-400 text-sm mb-1">{label}</p>
            <p className={`text-3xl font-bold text-${color}-400`}>{value}</p>
          </div>
        ))}
      </div>

      {/* Agent Grid */}
      <h3 className="text-xl font-semibold text-white mb-4">Agent Fleet</h3>
      <div className="grid grid-cols-3 gap-4">
        {displayAgents.map((agent) => (
          <div
            key={agent.agent_id}
            className="bg-gray-900 rounded-xl p-5 border border-gray-800 hover:border-violet-700 transition-colors"
          >
            <div className="flex items-start justify-between mb-3">
              <div className="flex items-center space-x-3">
                <span className="text-2xl">{AGENT_ICONS[agent.name] || "🤖"}</span>
                <div>
                  <h4 className="font-semibold text-white text-sm">
                    {agent.name.replace(/_/g, " ").replace(/\b\w/g, (c) => c.toUpperCase())}
                  </h4>
                  <div className="flex items-center space-x-1 mt-0.5">
                    <div className={`w-1.5 h-1.5 rounded-full ${STATUS_COLORS[agent.status] || "bg-gray-500"}`} />
                    <span className="text-xs text-gray-400 capitalize">{agent.status}</span>
                  </div>
                </div>
              </div>
            </div>
            <p className="text-gray-400 text-xs mb-3 line-clamp-2">{agent.description}</p>
            <div className="flex items-center justify-between text-xs text-gray-500">
              <span className="text-green-400">✓ {agent.tasks_completed}</span>
              <span className={agent.tasks_failed > 0 ? "text-red-400" : "text-gray-500"}>
                ✗ {agent.tasks_failed}
              </span>
            </div>
            <div className="mt-2 flex flex-wrap gap-1">
              {(agent.capabilities || []).slice(0, 3).map((cap) => (
                <span key={cap} className="px-1.5 py-0.5 bg-gray-800 text-gray-400 text-xs rounded">
                  {cap}
                </span>
              ))}
            </div>
          </div>
        ))}
      </div>

      {/* Workflow Quick-start */}
      <div className="mt-8">
        <h3 className="text-xl font-semibold text-white mb-4">Research Workflows</h3>
        <div className="grid grid-cols-2 gap-4">
          {[
            { id: "quantum_chemistry_vqe", name: "VQE Quantum Chemistry", desc: "End-to-end molecular simulation", n: 11, icon: "⚗️", color: "violet" },
            { id: "qubit_calibration", name: "Qubit Calibration", desc: "Autonomous calibration pipeline", n: 5, icon: "🎛️", color: "blue" },
            { id: "catalyst_discovery", name: "Catalyst Discovery", desc: "AI-guided catalyst screening", n: 6, icon: "🔬", color: "green" },
            { id: "literature_synthesis", name: "Literature Synthesis", desc: "Automated literature review", n: 6, icon: "📚", color: "orange" },
          ].map((wf) => (
            <div key={wf.id} className={`bg-gray-900 rounded-xl p-5 border border-gray-800 hover:border-${wf.color}-700 cursor-pointer transition-colors`}>
              <div className="flex items-center space-x-3 mb-2">
                <span className="text-2xl">{wf.icon}</span>
                <h4 className="font-semibold text-white">{wf.name}</h4>
              </div>
              <p className="text-gray-400 text-sm mb-3">{wf.desc}</p>
              <div className="flex items-center justify-between">
                <span className="text-xs text-gray-500">{wf.n} tasks</span>
                <button className={`px-3 py-1 bg-${wf.color}-600 text-white text-xs rounded-lg hover:bg-${wf.color}-500 transition-colors`}>
                  Start Workflow
                </button>
              </div>
            </div>
          ))}
        </div>
      </div>
    </div>
  );
}
