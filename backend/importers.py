"""
Bring-your-own-data importers.

Every dataset the agents can use has a documented column schema (see
``SCHEMAS`` below and ``data/templates/*.csv``). Files can be CSV, TSV,
JSON (a list of row objects, or ``{"rows": [...]}``) or Excel
(.xlsx, first sheet, needs ``openpyxl``).

``load_rows(kind, path_or_bytes, filename)`` returns validated rows with
proper Python types, or raises :class:`ImportValidationError` listing every
problem as "row N, column 'x': ...".
"""
from __future__ import annotations

import csv
import io
import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Sequence, Union

REPO_ROOT = Path(__file__).resolve().parent.parent
TEMPLATE_DIR = REPO_ROOT / "data" / "templates"
MAX_ERRORS_REPORTED = 25


class ImportValidationError(ValueError):
    def __init__(self, kind: str, errors: List[str]) -> None:
        self.kind = kind
        self.errors = errors
        shown = errors[:MAX_ERRORS_REPORTED]
        more = f"\n  ... and {len(errors) - len(shown)} more" if len(errors) > len(shown) else ""
        super().__init__(f"{kind}: {len(errors)} problem(s):\n  " + "\n  ".join(shown) + more)


# --------------------------------------------------------------------------- #
#  Column types                                                                #
# --------------------------------------------------------------------------- #

def _to_float(v: Any) -> float:
    if isinstance(v, bool):
        raise ValueError
    return float(str(v).replace(",", "").strip()) if isinstance(v, str) else float(v)


def _to_int(v: Any) -> int:
    f = _to_float(v)
    if f != int(f):
        raise ValueError
    return int(f)


def _to_str(v: Any) -> str:
    return str(v).strip()


def _to_list(v: Any) -> List[str]:
    if isinstance(v, list):
        return [str(x).strip() for x in v if str(x).strip()]
    return [p.strip() for p in str(v).replace("|", ";").split(";") if p.strip()]


_TYPES: Dict[str, Callable[[Any], Any]] = {
    "float": _to_float, "int": _to_int, "str": _to_str, "list": _to_list,
}
_TYPE_WORDS = {"float": "a number", "int": "a whole number", "str": "text", "list": "a ';'-separated list"}


@dataclass
class Column:
    name: str
    type: str = "str"
    required: bool = True
    description: str = ""
    choices: Optional[Sequence[str]] = None
    min: Optional[float] = None
    max: Optional[float] = None
    aliases: Sequence[str] = ()


@dataclass
class Schema:
    kind: str
    description: str
    columns: List[Column]
    template: str
    used_by: str
    row_check: Optional[Callable[[Dict], Optional[str]]] = None
    notes: List[str] = field(default_factory=list)

    @property
    def header(self) -> List[str]:
        return [c.name for c in self.columns]

    def to_dict(self) -> Dict:
        return {
            "kind": self.kind,
            "description": self.description,
            "used_by": self.used_by,
            "template": f"data/templates/{self.template}",
            "columns": [
                {
                    "name": c.name, "type": c.type, "required": c.required, "description": c.description,
                    **({"choices": list(c.choices)} if c.choices else {}),
                    **({"min": c.min} if c.min is not None else {}),
                    **({"max": c.max} if c.max is not None else {}),
                }
                for c in self.columns
            ],
            "notes": self.notes,
        }


def _check_rb_x(row: Dict) -> Optional[str]:
    if row["experiment"] == "RB" and row["x"] < 1:
        return "column 'x': RB sequence length must be >= 1"
    if row["experiment"] != "RB" and row["x"] < 0:
        return "column 'x': delay must be >= 0 us"
    return None


SCHEMAS: Dict[str, Schema] = {
    "hamiltonian": Schema(
        kind="hamiltonian",
        description="Qubit Hamiltonian as a sum of Pauli terms (export from PySCF/OpenFermion/Qiskit Nature).",
        template="hamiltonian_template.csv",
        used_by="chemistry_agent (simulate / vqe / benchmark), CLI 'vqe'",
        columns=[
            Column("molecule", "str", True, "Name used to look the Hamiltonian up later, e.g. H2, LiH"),
            Column("pauli", "str", True, "Pauli label: dense 'XZI' (rightmost char = qubit 0) or sparse 'X2 Z0'; 'I' for the constant",
                   aliases=("pauli_string", "term", "label")),
            Column("coeff", "float", True, "Real coefficient in Hartree", aliases=("coefficient", "weight")),
            Column("basis", "str", False, "Basis set, e.g. sto-3g"),
            Column("n_electrons", "int", False, "Active electrons (first non-empty value per molecule is used)", min=0),
            Column("nuclear_repulsion", "float", False, "Nuclear repulsion energy (Ha) added to report total energy"),
            Column("notes", "str", False, "Free text, e.g. geometry and mapping"),
        ],
        notes=["Rows with the same 'molecule' form one Hamiltonian; repeated labels are summed."],
    ),
    "qubit_measurements": Schema(
        kind="qubit_measurements",
        description="Raw characterisation sweeps to fit T1, T2 (Ramsey), T2 echo and randomised benchmarking.",
        template="qubit_measurements_template.csv",
        used_by="lab_automation_agent (characterise), CLI 'fit-qubits'",
        columns=[
            Column("qubit_id", "int", True, "Qubit index", min=0),
            Column("experiment", "str", True, "Which sweep this point belongs to", choices=("T1", "T2", "T2_echo", "RB")),
            Column("x", "float", True, "Delay in us (T1/T2/T2_echo) or Clifford length (RB)", aliases=("delay_us", "length")),
            Column("y", "float", True, "Measured population / survival probability", min=0, max=1,
                   aliases=("population", "survival", "p1")),
            Column("device", "str", False, "Device or chip name"),
            Column("measured_at", "str", False, "Timestamp (ISO 8601)"),
        ],
        row_check=_check_rb_x,
        notes=["At least 4 points per curve (6 for Ramsey)."],
    ),
    "device_calibration": Schema(
        kind="device_calibration",
        description="Per-qubit calibration snapshot of a device, to register it as a backend and rank it.",
        template="device_calibration_template.csv",
        used_by="hardware_agent (register_device / select_backend), CLI 'register-device'",
        columns=[
            Column("device", "str", True, "Device name; rows with the same name form one backend"),
            Column("qubit_id", "int", True, "Qubit index", min=0),
            Column("T1_us", "float", True, "Energy relaxation time (us)", min=0),
            Column("T2_us", "float", True, "Dephasing time (us)", min=0),
            Column("readout_error", "float", True, "Assignment error, 0-1", min=0, max=1),
            Column("cx_error", "float", False, "Two-qubit gate error to the next qubit, 0-1", min=0, max=1),
            Column("gate_error_1q", "float", False, "Single-qubit gate error, 0-1", min=0, max=1),
            Column("frequency_ghz", "float", False, "Qubit frequency (GHz)", min=0),
            Column("provider", "str", False, "Vendor / lab"),
            Column("connectivity", "str", False, "e.g. linear, heavy-hex, all-to-all"),
        ],
    ),
    "papers": Schema(
        kind="papers",
        description="Your own literature library (offline search, trends, citation stats).",
        template="papers_template.csv",
        used_by="literature_agent (search with source 'local'), CLI 'search-papers'",
        columns=[
            Column("title", "str", True, "Paper title"),
            Column("authors", "list", True, "Authors separated by ';'"),
            Column("year", "int", True, "Publication year", min=1900, max=2100),
            Column("abstract", "str", False, "Abstract or your own summary", aliases=("summary",)),
            Column("doi", "str", False, "DOI without the https://doi.org/ prefix"),
            Column("url", "str", False, "Link to the paper"),
            Column("venue", "str", False, "Journal / conference"),
            Column("keywords", "list", False, "Keywords separated by ';'"),
            Column("citations", "int", False, "Citation count", min=0),
        ],
    ),
    "hypotheses": Schema(
        kind="hypotheses",
        description="Research hypotheses to rank; leave score columns blank to have them estimated.",
        template="hypotheses_template.csv",
        used_by="pi_agent (rank_hypotheses), CLI 'rank-hypotheses'",
        columns=[
            Column("hypothesis_id", "str", False, "Your ID"),
            Column("statement", "str", True, "The hypothesis in one sentence"),
            Column("novelty", "float", False, "0-1, your estimate", min=0, max=1),
            Column("feasibility", "float", False, "0-1, your estimate", min=0, max=1),
            Column("impact", "float", False, "0-1, your estimate", min=0, max=1),
            Column("est_cost_usd", "float", False, "Estimated cost", min=0),
            Column("est_weeks", "float", False, "Estimated duration", min=0),
        ],
    ),
    "experiments": Schema(
        kind="experiments",
        description="Candidate experiments to prioritise and schedule.",
        template="experiments_template.csv",
        used_by="pi_agent (prioritise_experiments), lab_automation_agent (schedule)",
        columns=[
            Column("name", "str", True, "Experiment name"),
            Column("scientific_value", "float", True, "0-1", min=0, max=1),
            Column("success_probability", "float", False, "0-1", min=0, max=1),
            Column("cost_usd", "float", False, "Estimated cost", min=0),
            Column("estimated_hours", "float", False, "Machine / lab hours", min=0),
            Column("hardware", "str", False, "Backend or instrument needed"),
        ],
    ),
}


# --------------------------------------------------------------------------- #
#  Reading files                                                               #
# --------------------------------------------------------------------------- #

Source = Union[str, Path, bytes]


def _read_raw(source: Source, filename: Optional[str] = None) -> List[Dict[str, Any]]:
    if isinstance(source, (str, Path)) and not isinstance(source, bytes):
        path = Path(source)
        if not path.exists():
            raise FileNotFoundError(f"File not found: {path}")
        filename = filename or path.name
        data = path.read_bytes()
    else:
        data = source
    name = (filename or "upload.csv").lower()
    suffix = Path(name).suffix

    if suffix in (".xlsx", ".xlsm"):
        try:
            import openpyxl  # type: ignore
        except ImportError:
            raise ImportValidationError("file", ["Excel import needs openpyxl: pip install openpyxl (or save as CSV)"])
        wb = openpyxl.load_workbook(io.BytesIO(data), read_only=True, data_only=True)
        ws = wb.worksheets[0]
        rows = list(ws.iter_rows(values_only=True))
        if not rows:
            return []
        header = [str(h).strip() if h is not None else "" for h in rows[0]]
        return [dict(zip(header, r)) for r in rows[1:] if any(v not in (None, "") for v in r)]
    if suffix == ".xls":
        raise ImportValidationError("file", ["Legacy .xls is not supported; save the sheet as .xlsx or .csv"])

    text = data.decode("utf-8-sig") if isinstance(data, bytes) else str(data)
    if suffix == ".json":
        try:
            obj = json.loads(text)
        except json.JSONDecodeError as exc:
            raise ImportValidationError("file", [f"invalid JSON: {exc}"])
        if isinstance(obj, dict):
            obj = obj.get("rows", obj.get("data", obj.get("terms")))
        if not isinstance(obj, list) or not all(isinstance(r, dict) for r in obj):
            raise ImportValidationError("file", ["JSON must be a list of row objects or {\"rows\": [...]}"])
        return obj
    delimiter = "\t" if suffix in (".tsv", ".tab") else ","
    reader = csv.DictReader(io.StringIO(text), delimiter=delimiter)
    return [r for r in reader if any((v or "").strip() for v in r.values() if isinstance(v, str))]


def validate_rows(kind: str, raw_rows: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    schema = SCHEMAS.get(kind)
    if schema is None:
        raise ImportValidationError(kind, [f"unknown data kind '{kind}'; choose one of {sorted(SCHEMAS)}"])
    if not raw_rows:
        raise ImportValidationError(kind, ["file has no data rows"])

    # map headers case-insensitively, with aliases
    present = {str(k).strip().lower(): k for k in raw_rows[0].keys() if k is not None}
    colmap: Dict[str, Any] = {}
    errors: List[str] = []
    for col in schema.columns:
        for cand in (col.name, *col.aliases):
            if cand.lower() in present:
                colmap[col.name] = present[cand.lower()]
                break
        else:
            if col.required:
                errors.append(f"missing required column '{col.name}' ({col.description})")
    if errors:
        errors.append(f"expected header: {','.join(schema.header)} (template: data/templates/{schema.template})")
        raise ImportValidationError(kind, errors)

    out: List[Dict[str, Any]] = []
    for i, raw in enumerate(raw_rows, start=2):  # row 1 is the header
        row: Dict[str, Any] = {}
        for col in schema.columns:
            key = colmap.get(col.name)
            val = raw.get(key) if key is not None else None
            if val is None or (isinstance(val, str) and not val.strip()):
                if col.required:
                    errors.append(f"row {i}, column '{col.name}': value is required")
                row[col.name] = None if col.type != "list" else []
                continue
            try:
                conv = _TYPES[col.type](val)
            except (ValueError, TypeError):
                errors.append(f"row {i}, column '{col.name}': {val!r} is not {_TYPE_WORDS[col.type]}")
                continue
            if col.choices and conv not in col.choices:
                match = next((c for c in col.choices if c.lower() == str(conv).lower()), None)
                if match is None:
                    errors.append(f"row {i}, column '{col.name}': {conv!r} must be one of {list(col.choices)}")
                    continue
                conv = match
            if col.min is not None and conv < col.min:
                errors.append(f"row {i}, column '{col.name}': {conv} is below the minimum {col.min}")
                continue
            if col.max is not None and conv > col.max:
                errors.append(f"row {i}, column '{col.name}': {conv} is above the maximum {col.max}")
                continue
            row[col.name] = conv
        if schema.row_check and not any(e.startswith(f"row {i},") for e in errors):
            msg = schema.row_check(row)
            if msg:
                errors.append(f"row {i}, {msg}")
        out.append(row)
    if errors:
        raise ImportValidationError(kind, errors)
    return out


def load_rows(kind: str, source: Source, filename: Optional[str] = None) -> List[Dict[str, Any]]:
    """Read ``source`` (path or uploaded bytes) and validate it against ``SCHEMAS[kind]``."""
    return validate_rows(kind, _read_raw(source, filename))


def template_path(kind: str) -> Path:
    if kind not in SCHEMAS:
        raise KeyError(kind)
    return TEMPLATE_DIR / SCHEMAS[kind].template


# --------------------------------------------------------------------------- #
#  Domain conversions                                                          #
# --------------------------------------------------------------------------- #

def hamiltonians_from_rows(rows: List[Dict[str, Any]]):
    """Group validated hamiltonian rows into PauliHamiltonian objects keyed by molecule."""
    from .quantum.pauli import HamiltonianError, PauliHamiltonian

    groups: Dict[str, List[Dict[str, Any]]] = {}
    for r in rows:
        groups.setdefault(r["molecule"], []).append(r)
    out = {}
    errors = []
    for mol, rs in groups.items():
        first = lambda k: next((r[k] for r in rs if r.get(k) not in (None, "")), None)  # noqa: E731
        try:
            out[mol] = PauliHamiltonian.from_terms(
                [(r["pauli"], r["coeff"]) for r in rs],
                name=mol,
                basis=first("basis") or "",
                n_electrons=first("n_electrons"),
                nuclear_repulsion=first("nuclear_repulsion") or 0.0,
                source="user_import",
                metadata={"notes": first("notes") or ""},
            )
        except HamiltonianError as exc:
            errors.append(f"molecule '{mol}': {exc}")
    if errors:
        raise ImportValidationError("hamiltonian", errors)
    return out
