"""
Minimal OpenQASM 2.0 / 3.0 reader for the local simulator.

Supports the gate-level subset the platform itself emits (and that most
tools export): register declarations, standard gates with constant
parameter expressions (``pi``, arithmetic), ``measure`` in both 2.0 and 3.0
syntax, and ``barrier``. Custom ``gate`` definitions, classical control,
loops and ``reset`` are rejected with a clear error.
"""
from __future__ import annotations

import ast
import math
import operator
import re
from typing import Dict, List, Tuple

from .statevector import Circuit, SUPPORTED_GATES


class QasmParseError(ValueError):
    def __init__(self, message: str, line_no: int = 0) -> None:
        self.line_no = line_no
        super().__init__(f"line {line_no}: {message}" if line_no else message)


_ALLOWED_OPS = {
    ast.Add: operator.add, ast.Sub: operator.sub, ast.Mult: operator.mul,
    ast.Div: operator.truediv, ast.Pow: operator.pow, ast.USub: operator.neg, ast.UAdd: operator.pos,
}
_CONSTS = {"pi": math.pi, "π": math.pi, "tau": 2 * math.pi, "e": math.e}


def _eval_param(expr: str, line_no: int) -> float:
    expr = expr.strip().replace("π", "pi")
    try:
        tree = ast.parse(expr, mode="eval")
    except SyntaxError:
        raise QasmParseError(f"cannot parse parameter '{expr}'", line_no)

    def _ev(node):
        if isinstance(node, ast.Expression):
            return _ev(node.body)
        if isinstance(node, ast.Constant) and isinstance(node.value, (int, float)):
            return float(node.value)
        if isinstance(node, ast.Name) and node.id in _CONSTS:
            return _CONSTS[node.id]
        if isinstance(node, ast.BinOp) and type(node.op) in _ALLOWED_OPS:
            return _ALLOWED_OPS[type(node.op)](_ev(node.left), _ev(node.right))
        if isinstance(node, ast.UnaryOp) and type(node.op) in _ALLOWED_OPS:
            return _ALLOWED_OPS[type(node.op)](_ev(node.operand))
        raise QasmParseError(
            f"unsupported parameter expression '{expr}' (only numbers, pi and + - * / ** allowed)", line_no
        )

    return _ev(tree)


def _split_params(text: str) -> List[str]:
    parts, depth, cur = [], 0, ""
    for ch in text:
        if ch == "," and depth == 0:
            parts.append(cur)
            cur = ""
            continue
        depth += ch == "("
        depth -= ch == ")"
        cur += ch
    if cur.strip():
        parts.append(cur)
    return parts


_DECL_QREG2 = re.compile(r"^qreg\s+(\w+)\s*\[\s*(\d+)\s*\]$")
_DECL_QREG3 = re.compile(r"^qubit\s*\[\s*(\d+)\s*\]\s*(\w+)$")
_DECL_QUBIT1 = re.compile(r"^qubit\s+(\w+)$")
_DECL_CREG = re.compile(r"^(creg\s+\w+\s*\[\s*\d+\s*\]|bit\s*\[\s*\d+\s*\]\s*\w+|bit\s+\w+)$")
_MEAS2 = re.compile(r"^measure\s+(.+?)\s*->\s*(.+)$")
_MEAS3 = re.compile(r"^(.+?)\s*=\s*measure\s+(.+)$")
_GATE = re.compile(r"^([A-Za-z_][\w]*)\s*(?:\((.*)\))?\s+(.+)$")
_ARG = re.compile(r"^(\w+)\s*(?:\[\s*(\d+)\s*\])?$")


def parse_qasm(text: str) -> Circuit:
    """Parse OpenQASM text into a :class:`Circuit`."""
    if not text or not text.strip():
        raise QasmParseError("empty QASM program")

    # strip comments, then split on ';' while keeping line numbers
    cleaned_lines = []
    in_block = False
    for raw in text.splitlines():
        line = raw
        if in_block:
            if "*/" in line:
                line = line.split("*/", 1)[1]
                in_block = False
            else:
                cleaned_lines.append("")
                continue
        if "/*" in line:
            before, _, after = line.partition("/*")
            if "*/" in after:
                line = before + after.split("*/", 1)[1]
            else:
                line = before
                in_block = True
        cleaned_lines.append(line.split("//", 1)[0])

    statements: List[Tuple[int, str]] = []
    buf, start = "", 1
    for i, line in enumerate(cleaned_lines, start=1):
        for ch in line:
            if not buf.strip():
                start = i
            if ch == ";":
                if buf.strip():
                    statements.append((start, " ".join(buf.split())))
                buf = ""
            else:
                buf += ch
        buf += " "
    if buf.strip():
        raise QasmParseError(f"missing ';' after '{buf.strip()[:40]}'", start)

    registers: Dict[str, Tuple[int, int]] = {}  # name -> (offset, size)
    n_qubits = 0
    ops: List[Tuple[int, str, List[float], List[int]]] = []
    measured: List[int] = []

    def resolve(arg: str, line_no: int) -> List[int]:
        m = _ARG.match(arg.strip())
        if not m:
            raise QasmParseError(f"bad qubit argument '{arg}'", line_no)
        name, idx = m.group(1), m.group(2)
        if name not in registers:
            raise QasmParseError(f"unknown qubit register '{name}'", line_no)
        off, size = registers[name]
        if idx is None:
            return list(range(off, off + size))
        if int(idx) >= size:
            raise QasmParseError(f"qubit index {name}[{idx}] out of range (size {size})", line_no)
        return [off + int(idx)]

    for line_no, st in statements:
        low = st.lower()
        if low.startswith("openqasm") or low.startswith("include"):
            continue
        if low.startswith("gate ") or low.startswith("def ") or low.startswith("if") or low.startswith("for "):
            raise QasmParseError("custom gate definitions / classical control are not supported", line_no)
        if low.startswith("reset"):
            raise QasmParseError("'reset' is not supported by the local simulator", line_no)

        m = _DECL_QREG2.match(st) or _DECL_QREG3.match(st)
        if m:
            if _DECL_QREG2.match(st):
                name, size = m.group(1), int(m.group(2))
            else:
                size, name = int(m.group(1)), m.group(2)
            registers[name] = (n_qubits, size)
            n_qubits += size
            continue
        m = _DECL_QUBIT1.match(st)
        if m:
            registers[m.group(1)] = (n_qubits, 1)
            n_qubits += 1
            continue
        if _DECL_CREG.match(st):
            continue

        m = _MEAS2.match(st)
        if m:
            measured.extend(resolve(m.group(1), line_no))
            continue
        m = _MEAS3.match(st)
        if m:
            measured.extend(resolve(m.group(2), line_no))
            continue
        if low.startswith("measure "):
            measured.extend(resolve(st.split(None, 1)[1], line_no))
            continue

        m = _GATE.match(st)
        if not m:
            raise QasmParseError(f"cannot parse statement '{st[:60]}'", line_no)
        gname = m.group(1).lower()
        if gname == "barrier":
            continue
        if gname not in SUPPORTED_GATES:
            raise QasmParseError(
                f"unsupported gate '{m.group(1)}' (supported: {', '.join(SUPPORTED_GATES)})", line_no
            )
        params = [_eval_param(p, line_no) for p in _split_params(m.group(2))] if m.group(2) else []
        args = [a for a in m.group(3).split(",") if a.strip()]
        qubit_lists = [resolve(a, line_no) for a in args]
        # broadcast register-wide arguments (e.g. "h q;")
        width = max(len(q) for q in qubit_lists)
        for k in range(width):
            qs = [ql[k] if len(ql) > 1 else ql[0] for ql in qubit_lists]
            ops.append((line_no, gname, params, qs))

    if n_qubits == 0:
        raise QasmParseError("no qubits declared (expected 'qreg q[n];' or 'qubit[n] q;')")

    circuit = Circuit(n_qubits)
    for _line_no, gname, params, qs in ops:
        circuit.add(gname, qs, params)
    # de-duplicate while keeping order
    seen = set()
    circuit.measured = [q for q in measured if not (q in seen or seen.add(q))] or list(range(n_qubits))
    return circuit
