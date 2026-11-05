"""
QuantumAgent command line.

    python -m backend.cli schemas                       # list importable datasets + columns
    python -m backend.cli validate papers my_papers.csv # check a file against its schema
    python -m backend.cli vqe --hamiltonian my_h.csv    # VQE on your Pauli Hamiltonian
    python -m backend.cli vqe --demo                    # VQE on the built-in H2/STO-3G
    python -m backend.cli run-circuit bell.qasm --shots 2000
    python -m backend.cli fit-qubits my_sweeps.csv      # T1 / T2 / T2 echo / RB fits
    python -m backend.cli register-device calib.csv     # rank your device against the catalogue
    python -m backend.cli rank-hypotheses hyps.csv
    python -m backend.cli prioritise experiments.csv
    python -m backend.cli search-papers papers.csv "error mitigation"
    python -m backend.cli demo                          # all of the above on the bundled demo data

Add ``--json`` to any command for machine-readable output. Files may be
.csv, .tsv, .json or .xlsx; templates live in data/templates/.
"""
from __future__ import annotations

import argparse
import asyncio
import json
import os
import sys
from pathlib import Path
from typing import Any, Dict, List, Sequence

from .importers import SCHEMAS, ImportValidationError, hamiltonians_from_rows, load_rows, template_path
from .data_loader import DEMO_FILES, SAMPLE_DIR


def _print(obj: Any, as_json: bool) -> None:
    if as_json:
        print(json.dumps(obj, indent=2, default=str))
    elif isinstance(obj, str):
        print(obj)
    else:
        print(json.dumps(obj, indent=2, default=str))


def _table(rows: List[Dict], cols: Sequence[str]) -> str:
    if not rows:
        return "(no rows)"
    cells = [[("" if r.get(c) is None else (f"{r[c]:.6g}" if isinstance(r.get(c), float) else str(r[c])))
              for c in cols] for r in rows]
    widths = [max(len(c), *(len(row[i]) for row in cells)) for i, c in enumerate(cols)]
    line = "  ".join(c.ljust(w) for c, w in zip(cols, widths))
    sep = "  ".join("-" * w for w in widths)
    body = "\n".join("  ".join(v.ljust(w) for v, w in zip(row, widths)) for row in cells)
    return f"{line}\n{sep}\n{body}"


def _run(agent, payload: Dict):
    return asyncio.run(agent.run(payload))


# --------------------------------------------------------------------------- #
#  Commands                                                                    #
# --------------------------------------------------------------------------- #

def cmd_schemas(args) -> int:
    if args.json:
        _print({k: s.to_dict() for k, s in SCHEMAS.items()}, True)
        return 0
    for kind, s in SCHEMAS.items():
        print(f"\n{kind}  ({s.description})")
        print(f"  template: data/templates/{s.template}   used by: {s.used_by}")
        for c in s.columns:
            req = "required" if c.required else "optional"
            extra = f" one of {list(c.choices)}" if c.choices else ""
            print(f"    - {c.name:<18} {c.type:<5} {req:<8} {c.description}{extra}")
    return 0


def cmd_validate(args) -> int:
    rows = load_rows(args.kind, args.file)
    _print({"kind": args.kind, "file": args.file, "valid_rows": len(rows), "first_row": rows[0]} if args.json
           else f"OK: {len(rows)} valid {args.kind} row(s) in {args.file}", args.json)
    return 0


def cmd_vqe(args) -> int:
    from .quantum.pauli import h2_sto3g
    from .quantum.vqe import run_vqe

    if args.hamiltonian:
        hams = hamiltonians_from_rows(load_rows("hamiltonian", args.hamiltonian))
        if args.molecule:
            if args.molecule not in hams:
                raise SystemExit(f"molecule '{args.molecule}' not in file; found {sorted(hams)}")
            hams = {args.molecule: hams[args.molecule]}
    else:
        hams = {"H2": h2_sto3g()}
    out = []
    for name, ham in hams.items():
        r = run_vqe(ham, layers=args.layers, optimiser=args.optimiser, max_iterations=args.max_iterations,
                    shots=args.shots, restarts=args.restarts)
        d = r.to_dict()
        out.append({"molecule": name, "n_qubits": ham.n_qubits, "n_terms": len(ham.terms),
                    "vqe_total_energy": d["total_energy"], "exact_total_energy": d["exact_total_energy"],
                    "error_kcal": d["error_kcal"], "chem_accuracy": d["within_chemical_accuracy"],
                    "evaluations": d["n_evaluations"], "runtime_s": d["runtime_s"]})
    _print(out if args.json else _table(out, list(out[0].keys())), args.json)
    return 0


def cmd_run_circuit(args) -> int:
    from .agents.hardware_agent import HardwareIntegrationAgent

    qasm = Path(args.file).read_text()
    r = _run(HardwareIntegrationAgent(seed=args.seed),
             {"action": "submit_job", "qasm": qasm, "shots": args.shots,
              "backend_id": args.backend, "emulate": args.emulate})
    if not r.success:
        raise SystemExit(f"error: {r.output.get('error')}\n{r.output.get('hint', '')}".rstrip())
    if args.json:
        _print({**r.output, "warnings": r.warnings}, True)
    else:
        counts = sorted(r.output["counts"].items(), key=lambda kv: -kv[1])
        print(f"{r.output['n_qubits']} qubits, {r.output['shots']} shots on {r.output['backend']} ({r.output['engine']})")
        print(_table([{"bitstring": k, "count": v, "freq": v / r.output["shots"]} for k, v in counts[:16]],
                     ["bitstring", "count", "freq"]))
        for w in r.warnings:
            print("note:", w)
    return 0


def cmd_fit_qubits(args) -> int:
    from .agents.lab_automation import LabAutomationAgent

    rows = load_rows("qubit_measurements", args.file)
    r = _run(LabAutomationAgent(), {"action": "fit_measurements", "rows": rows})
    if args.json:
        _print(r.output, True)
    else:
        print(_table(r.output["qubits"], ["qubit_id", "T1_us", "T2_us", "T2_echo_us", "gate_fidelity"]))
        for q, fits in r.output["fits"].items():
            for exp, f in fits.items():
                if "error" in f:
                    print(f"qubit {q} {exp}: {f['error']}")
    return 0


def cmd_register_device(args) -> int:
    from .agents.hardware_agent import HardwareIntegrationAgent

    agent = HardwareIntegrationAgent()
    rows = load_rows("device_calibration", args.file)
    reg = _run(agent, {"action": "register_device", "rows": rows})
    n = max(d["n_qubits"] for d in reg.output["registered"])
    sel = _run(agent, {"action": "select_backend", "n_qubits": min(n, args.n_qubits or n),
                       "priority": args.priority, "exclude_simulators": True})
    ranking = sel.output["ranking"]
    if args.json:
        _print({"registered": reg.output["registered"], "ranking": ranking}, True)
    else:
        print(_table(reg.output["registered"], ["backend_id", "n_qubits", "T1_us", "T2_us", "cx_error", "readout_error"]))
        print(f"\nRanking for {min(n, args.n_qubits or n)} qubits, priority={args.priority}:")
        print(_table(ranking, ["name", "selection_score", "cx_error", "readout_error", "source"]))
    return 0


def cmd_rank_hypotheses(args) -> int:
    from .agents.pi_agent import PrincipalInvestigatorAgent

    rows = load_rows("hypotheses", args.file)
    r = _run(PrincipalInvestigatorAgent(), {"action": "rank_hypotheses", "hypotheses": rows})
    out = r.output["ranked_hypotheses"]
    _print(out if args.json else _table(out, ["rank", "hypothesis_id", "composite_score", "feasibility",
                                              "novelty", "impact", "statement"]), args.json)
    return 0


def cmd_prioritise(args) -> int:
    from .agents.pi_agent import PrincipalInvestigatorAgent

    rows = load_rows("experiments", args.file)
    r = _run(PrincipalInvestigatorAgent(), {"action": "prioritise_experiments", "experiments": rows})
    out = r.output["prioritised_experiments"]
    _print(out if args.json else _table(out, ["recommended_order", "name", "priority_score", "rationale"]), args.json)
    return 0


def cmd_search_papers(args) -> int:
    from .agents.literature_agent import LiteratureAgent

    agent = LiteratureAgent()
    agent.load_library(load_rows("papers", args.file))
    sources = ["local"] + (["arxiv", "semantic_scholar"] if args.online else [])
    r = _run(agent, {"action": "search", "query": args.query, "sources": sources, "max_results": args.max_results})
    out = r.output["papers"]
    _print(out if args.json else _table(out, ["relevance_score", "published", "title", "source"]), args.json)
    return 0


def cmd_demo(args) -> int:
    print("== VQE on bundled H2 Hamiltonian ==")
    cmd_vqe(argparse.Namespace(hamiltonian=str(SAMPLE_DIR / DEMO_FILES["hamiltonian"]), molecule=None, layers=2,
                               optimiser="COBYLA", max_iterations=500, shots=None, restarts=2, json=args.json))
    print("\n== Qubit fits on demo (synthetic) sweeps ==")
    cmd_fit_qubits(argparse.Namespace(file=str(SAMPLE_DIR / DEMO_FILES["qubit_measurements"]), json=args.json))
    print("\n== Register demo device and rank backends ==")
    cmd_register_device(argparse.Namespace(file=str(SAMPLE_DIR / DEMO_FILES["device_calibration"]), n_qubits=None,
                                           priority="fidelity", json=args.json))
    print("\n== Rank demo hypotheses ==")
    cmd_rank_hypotheses(argparse.Namespace(file=str(SAMPLE_DIR / DEMO_FILES["hypotheses"]), json=args.json))
    print("\n== Prioritise demo experiments ==")
    cmd_prioritise(argparse.Namespace(file=str(SAMPLE_DIR / DEMO_FILES["experiments"]), json=args.json))
    print("\n== Search demo paper library: 'error mitigation' ==")
    cmd_search_papers(argparse.Namespace(file=str(SAMPLE_DIR / DEMO_FILES["papers"]), query="error mitigation",
                                         online=False, max_results=5, json=args.json))
    return 0


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(prog="python -m backend.cli", description="QuantumAgent command line")
    p.add_argument("--json", action="store_true", help="machine-readable output")
    common = argparse.ArgumentParser(add_help=False)
    common.add_argument("--json", action="store_true", default=argparse.SUPPRESS, help="machine-readable output")
    sub = p.add_subparsers(dest="command", required=True)
    _add = sub.add_parser
    sub.add_parser = lambda *a, **kw: _add(*a, parents=[common], **kw)  # every command accepts --json

    sub.add_parser("schemas", help="list importable datasets and their columns").set_defaults(fn=cmd_schemas)

    s = sub.add_parser("validate", help="check a data file against its schema")
    s.add_argument("kind", choices=sorted(SCHEMAS))
    s.add_argument("file")
    s.set_defaults(fn=cmd_validate)

    s = sub.add_parser("vqe", help="run VQE on a Pauli Hamiltonian (local simulator)")
    s.add_argument("--hamiltonian", help="hamiltonian CSV/JSON/XLSX (default: built-in H2/STO-3G)")
    s.add_argument("--demo", action="store_true", help="use the built-in H2/STO-3G Hamiltonian")
    s.add_argument("--molecule", help="only this molecule from the file")
    s.add_argument("--layers", type=int, default=2)
    s.add_argument("--optimiser", default="COBYLA", choices=["COBYLA", "L-BFGS-B", "Nelder-Mead", "Powell", "SLSQP"])
    s.add_argument("--max-iterations", type=int, default=500)
    s.add_argument("--shots", type=int, default=None, help="estimate energies from samples instead of exactly")
    s.add_argument("--restarts", type=int, default=2)
    s.set_defaults(fn=cmd_vqe)

    s = sub.add_parser("run-circuit", help="run an OpenQASM 2/3 file")
    s.add_argument("file")
    s.add_argument("--shots", type=int, default=1024)
    s.add_argument("--backend", default="local_simulator")
    s.add_argument("--emulate", action="store_true", help="run a hardware backend's noise model locally")
    s.add_argument("--seed", type=int, default=None)
    s.set_defaults(fn=cmd_run_circuit)

    s = sub.add_parser("fit-qubits", help="fit T1 / T2 / T2 echo / RB curves")
    s.add_argument("file")
    s.set_defaults(fn=cmd_fit_qubits)

    s = sub.add_parser("register-device", help="register a device from calibration data and rank backends")
    s.add_argument("file")
    s.add_argument("--n-qubits", type=int, default=None)
    s.add_argument("--priority", default="fidelity", choices=["fidelity", "speed", "scale", "balanced"])
    s.set_defaults(fn=cmd_register_device)

    s = sub.add_parser("rank-hypotheses", help="rank hypotheses")
    s.add_argument("file")
    s.set_defaults(fn=cmd_rank_hypotheses)

    s = sub.add_parser("prioritise", help="prioritise candidate experiments")
    s.add_argument("file")
    s.set_defaults(fn=cmd_prioritise)

    s = sub.add_parser("search-papers", help="search your paper library")
    s.add_argument("file")
    s.add_argument("query")
    s.add_argument("--online", action="store_true", help="also query arXiv and Semantic Scholar")
    s.add_argument("--max-results", type=int, default=10)
    s.set_defaults(fn=cmd_search_papers)

    sub.add_parser("demo", help="run every command on the bundled demo data").set_defaults(fn=cmd_demo)
    return p


def main(argv: Sequence[str] = None) -> int:
    args = build_parser().parse_args(argv)
    try:
        return args.fn(args)
    except ImportValidationError as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 2
    except FileNotFoundError as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
