"""
Qubit characterisation: curve fits for measured data, plus a simple
simulated transmon used for demos and closed-loop calibration when no
instrument is connected.

Measured data (bring your own) uses the long format in
``data/templates/qubit_measurements_template.csv``:

    qubit_id, experiment, x, y
    experiment = T1 | T2 | T2_echo | RB
    x = delay in microseconds (T1/T2/T2_echo) or Clifford sequence length (RB)
    y = excited-state population (T1), P(|0>) / P(|1>) Ramsey signal (T2),
        echo signal (T2_echo) or survival probability (RB), all in [0, 1]
"""
from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Dict, List, Optional, Sequence

import numpy as np
from scipy.optimize import curve_fit, minimize


class FitError(ValueError):
    pass


def _r_squared(y: np.ndarray, y_fit: np.ndarray) -> float:
    ss_res = float(np.sum((y - y_fit) ** 2))
    ss_tot = float(np.sum((y - np.mean(y)) ** 2)) or 1e-30
    return 1.0 - ss_res / ss_tot


def _prep(x: Sequence[float], y: Sequence[float], min_points: int, name: str):
    x = np.asarray(x, float)
    y = np.asarray(y, float)
    if x.shape != y.shape:
        raise FitError(f"{name}: x and y have different lengths")
    if len(x) < min_points:
        raise FitError(f"{name}: need at least {min_points} points, got {len(x)}")
    if not np.all(np.isfinite(x)) or not np.all(np.isfinite(y)):
        raise FitError(f"{name}: data contains NaN/inf")
    order = np.argsort(x)
    return x[order], y[order]


def fit_exponential_decay(x, y, name: str = "T1") -> Dict:
    """Fit y = A exp(-x/T) + B. Used for T1 (inversion recovery) and T2 echo."""
    x, y = _prep(x, y, 4, name)
    a0 = y[0] - y[-1]
    b0 = y[-1]
    # initial T: where the signal has fallen to 1/e of its span
    target = b0 + a0 / math.e
    idx = int(np.argmin(np.abs(y - target)))
    t0 = max(x[idx], (x[-1] - x[0]) / 5 or 1.0)
    try:
        popt, pcov = curve_fit(
            lambda t, A, T, B: A * np.exp(-t / T) + B, x, y,
            p0=[a0, t0, b0], bounds=([-2, 1e-9, -1], [2, 1e9, 2]), maxfev=20000,
        )
    except Exception as exc:  # pragma: no cover - scipy message passthrough
        raise FitError(f"{name} fit failed: {exc}")
    A, T, B = popt
    perr = np.sqrt(np.clip(np.diag(pcov), 0, None))
    y_fit = A * np.exp(-x / T) + B
    return {
        "value": round(float(T), 4), "stderr": round(float(perr[1]), 4), "unit": "us",
        "amplitude": round(float(A), 5), "offset": round(float(B), 5),
        "r_squared": round(_r_squared(y, y_fit), 5), "n_points": int(len(x)),
        "model": "A*exp(-t/T)+B",
    }


def fit_ramsey(x, y) -> Dict:
    """Fit y = A exp(-x/T2) cos(2 pi f x + phi) + B (Ramsey fringes)."""
    x, y = _prep(x, y, 6, "T2")
    B0 = float(np.mean(y))
    A0 = float((np.max(y) - np.min(y)) / 2) or 0.5
    # frequency guess from FFT of the evenly-resampled signal
    xs = np.linspace(x[0], x[-1], max(64, len(x)))
    ys = np.interp(xs, x, y) - B0
    spec = np.abs(np.fft.rfft(ys))
    freqs = np.fft.rfftfreq(len(xs), d=(xs[1] - xs[0]))
    f0 = float(freqs[1 + int(np.argmax(spec[1:]))]) if len(spec) > 1 else 0.1
    span = x[-1] - x[0] or 1.0

    def model(t, A, T, f, phi, B):
        return A * np.exp(-t / T) * np.cos(2 * np.pi * f * t + phi) + B

    best = None
    for phi0 in (0.0, np.pi / 2, np.pi, -np.pi / 2):
        try:
            popt, pcov = curve_fit(
                model, x, y, p0=[A0, span / 2, f0, phi0, B0],
                bounds=([0, 1e-9, 0, -2 * np.pi, -1], [2, 1e9, np.inf, 2 * np.pi, 2]), maxfev=20000,
            )
        except Exception:
            continue
        resid = float(np.sum((model(x, *popt) - y) ** 2))
        if best is None or resid < best[0]:
            best = (resid, popt, pcov)
    if best is None:
        raise FitError("T2 (Ramsey) fit failed to converge")
    _, popt, pcov = best
    perr = np.sqrt(np.clip(np.diag(pcov), 0, None))
    return {
        "value": round(float(popt[1]), 4), "stderr": round(float(perr[1]), 4), "unit": "us",
        "detuning_MHz": round(float(popt[2]), 5), "amplitude": round(float(popt[0]), 5),
        "r_squared": round(_r_squared(y, model(x, *popt)), 5), "n_points": int(len(x)),
        "model": "A*exp(-t/T2)*cos(2*pi*f*t+phi)+B",
    }


def fit_randomised_benchmarking(m, survival, n_qubits: int = 1) -> Dict:
    """Fit P(m) = A p^m + B and convert to error per Clifford / average gate fidelity."""
    m, y = _prep(m, survival, 4, "RB")
    d = 2 ** n_qubits
    try:
        popt, pcov = curve_fit(
            lambda mm, A, p, B: A * p ** mm + B, m, y,
            p0=[y[0] - y[-1] or 0.5, 0.99, y[-1]], bounds=([0, 0, 0], [1, 1, 1]), maxfev=20000,
        )
    except Exception as exc:  # pragma: no cover
        raise FitError(f"RB fit failed: {exc}")
    A, p, B = popt
    perr = np.sqrt(np.clip(np.diag(pcov), 0, None))
    epc = (1 - p) * (d - 1) / d
    return {
        "depolarising_parameter": round(float(p), 7), "stderr_p": round(float(perr[1]), 7),
        "error_per_clifford": round(float(epc), 7),
        "average_gate_fidelity": round(float(1 - epc), 7),
        "amplitude": round(float(A), 5), "offset": round(float(B), 5),
        "r_squared": round(_r_squared(y, A * p ** m + B), 5), "n_points": int(len(m)),
        "model": "A*p^m+B",
    }


def readout_fidelity(p0_given_0: float, p1_given_1: float) -> Dict:
    for v in (p0_given_0, p1_given_1):
        if not 0 <= v <= 1:
            raise FitError("readout probabilities must be in [0, 1]")
    fid = (p0_given_0 + p1_given_1) / 2
    return {"fidelity": round(fid, 6), "assignment_error": round(1 - fid, 6),
            "p0_given_0": p0_given_0, "p1_given_1": p1_given_1}


FITTERS = {
    "T1": lambda x, y: fit_exponential_decay(x, y, "T1"),
    "T2": fit_ramsey,
    "T2_echo": lambda x, y: fit_exponential_decay(x, y, "T2_echo"),
    "RB": fit_randomised_benchmarking,
}


def fit_measurements(rows: List[Dict]) -> Dict[int, Dict[str, Dict]]:
    """Group validated measurement rows by qubit and experiment and fit each curve."""
    grouped: Dict[int, Dict[str, List]] = {}
    for r in rows:
        grouped.setdefault(int(r["qubit_id"]), {}).setdefault(r["experiment"], []).append((r["x"], r["y"]))
    out: Dict[int, Dict[str, Dict]] = {}
    for qid, exps in sorted(grouped.items()):
        out[qid] = {}
        for exp, pts in exps.items():
            xs, ys = zip(*pts)
            try:
                out[qid][exp] = FITTERS[exp](xs, ys)
            except FitError as exc:
                out[qid][exp] = {"error": str(exc), "n_points": len(xs)}
    return out


# --------------------------------------------------------------------------- #
#  Simulated device (demo / digital twin)                                     #
# --------------------------------------------------------------------------- #

@dataclass
class SimulatedTransmon:
    """A deterministic toy model of one transmon qubit.

    It produces realistic-looking T1 / Ramsey / RB curves and has a
    single-qubit gate error that depends on drive amplitude, DRAG coefficient
    and frequency offset, with a hidden optimum. It stands in for real
    instruments so the calibration loop has something physical to optimise.
    """

    qubit_id: int = 0

    def __post_init__(self) -> None:
        rng = np.random.default_rng(1000 + self.qubit_id)
        self.t1_us = float(rng.uniform(80, 200))
        self.t2_us = float(min(2 * self.t1_us, rng.uniform(40, 140)))
        self.t2_echo_us = float(min(2 * self.t1_us, self.t2_us * rng.uniform(1.2, 1.6)))
        # Ramsey detuning chosen so ~5 fringes fit in the 3*T2 measurement window
        self.detuning_mhz = float(5.0 / (3.0 * self.t2_us))
        self.opt = {
            "amplitude": float(rng.uniform(0.42, 0.58)),
            "drag_beta": float(rng.uniform(-1.0, 1.0)),
            "frequency_offset_MHz": float(rng.uniform(-0.3, 0.3)),
        }
        self.gate_time_ns = 35.0
        self.readout = (float(rng.uniform(0.975, 0.995)), float(rng.uniform(0.95, 0.985)))
        self._rng = rng

    @property
    def coherence_limited_error(self) -> float:
        t = self.gate_time_ns / 1000.0
        return (1 - math.exp(-t / self.t1_us)) / 3 + (1 - math.exp(-t / self.t2_us)) / 3

    def gate_error(self, amplitude: float, drag_beta: float, frequency_offset_MHz: float) -> float:
        o = self.opt
        coherent = (
            0.20 * ((amplitude - o["amplitude"]) / o["amplitude"]) ** 2
            + 0.0008 * (drag_beta - o["drag_beta"]) ** 2
            + 0.0040 * (frequency_offset_MHz - o["frequency_offset_MHz"]) ** 2
        )
        return float(min(0.5, self.coherence_limited_error + coherent))

    def measure_gate_fidelity(self, params: Dict[str, float], shots: int = 4000) -> float:
        err = self.gate_error(params["amplitude"], params["drag_beta"], params["frequency_offset_MHz"])
        noise = self._rng.normal(0, math.sqrt(max(err * (1 - err), 1e-12) / shots) * 0.05)
        return float(np.clip(1 - err + noise, 0, 1))

    def curve(self, experiment: str, n_points: int = 40, noise: float = 0.01):
        rng = np.random.default_rng(self.qubit_id * 31 + len(experiment))
        if experiment == "T1":
            x = np.linspace(0, 4 * self.t1_us, n_points)
            y = 0.97 * np.exp(-x / self.t1_us) + 0.015
        elif experiment == "T2":
            x = np.linspace(0, 3 * self.t2_us, n_points * 2)
            y = 0.48 * np.exp(-x / self.t2_us) * np.cos(2 * np.pi * self.detuning_mhz * x) + 0.5
        elif experiment == "T2_echo":
            x = np.linspace(0, 4 * self.t2_echo_us, n_points)
            y = 0.48 * np.exp(-x / self.t2_echo_us) + 0.5
        elif experiment == "RB":
            x = np.unique(np.geomspace(1, 2000, n_points).astype(int)).astype(float)
            p = 1 - 2 * self.gate_error(**self.opt) * 1.875  # ~1.875 physical gates per Clifford
            y = 0.5 * p ** x + 0.5
        else:
            raise FitError(f"no simulated curve for experiment '{experiment}'")
        return x, np.clip(y + rng.normal(0, noise, len(x)), 0, 1)


def optimise_pulse(device: SimulatedTransmon, start: Dict[str, float], max_iterations: int = 60,
                   target_fidelity: float = 0.9995) -> Dict:
    """Closed-loop Nelder-Mead over (amplitude, drag_beta, frequency_offset)."""
    keys = ["amplitude", "drag_beta", "frequency_offset_MHz"]
    scale = np.array([0.05, 0.5, 0.2])
    history: List[Dict] = []

    def cost(z: np.ndarray) -> float:
        params = {k: float(v) for k, v in zip(keys, z * scale)}
        fid = device.measure_gate_fidelity(params)
        history.append({"iteration": len(history) + 1, "objective": round(fid, 6),
                        "params": {k: round(v, 5) for k, v in params.items()},
                        "converged": fid >= target_fidelity})
        return 1 - fid

    x0 = np.array([start[k] for k in keys]) / scale
    initial = 1 - device.gate_error(**start)

    def _stop(_xk):
        if history and history[-1]["objective"] >= target_fidelity:
            raise StopIteration

    try:
        simplex = np.vstack([x0, x0 + np.eye(len(keys))])  # one scale unit per parameter
        minimize(cost, x0, method="Nelder-Mead", callback=_stop,
                 options={"maxiter": max_iterations, "xatol": 1e-4, "fatol": 1e-7, "initial_simplex": simplex})
    except StopIteration:  # older SciPy re-raises the callback's StopIteration
        pass
    best = max(history, key=lambda h: h["objective"])
    best_params = best["params"]
    final = 1 - device.gate_error(**best_params)
    return {
        "initial_value": round(initial, 6),
        "final_value": round(final, 6),
        "converged": final >= target_fidelity,
        "total_iterations": len(history),
        "optimal_params": best_params,
        "convergence_history": history[-10:],
        "coherence_limit": round(1 - device.coherence_limited_error, 6),
    }
