"""Curve fits and the simulated transmon used for closed-loop calibration."""
import numpy as np
import pytest

from backend.quantum.characterisation import (
    FitError, SimulatedTransmon, fit_exponential_decay, fit_measurements, fit_ramsey,
    fit_randomised_benchmarking, optimise_pulse, readout_fidelity,
)


def test_t1_fit_recovers_value():
    x = np.linspace(0, 400, 30)
    y = 0.95 * np.exp(-x / 120.0) + 0.02
    f = fit_exponential_decay(x, y)
    assert abs(f["value"] - 120.0) < 0.5 and f["r_squared"] > 0.999


def test_ramsey_fit_recovers_t2():
    x = np.linspace(0, 200, 120)
    y = 0.45 * np.exp(-x / 70.0) * np.cos(2 * np.pi * 0.04 * x + 0.3) + 0.5
    f = fit_ramsey(x, y)
    assert abs(f["value"] - 70.0) < 2.0
    assert abs(f["detuning_MHz"] - 0.04) < 1e-3


def test_rb_fit_gives_gate_fidelity():
    m = np.array([1, 10, 50, 100, 200, 500, 1000], float)
    p = 0.998
    f = fit_randomised_benchmarking(m, 0.5 * p ** m + 0.5)
    assert abs(f["depolarising_parameter"] - p) < 1e-5
    assert abs(f["error_per_clifford"] - (1 - p) / 2) < 1e-5


def test_fit_errors():
    with pytest.raises(FitError, match="at least"):
        fit_exponential_decay([0, 1], [1, 0.5])
    with pytest.raises(FitError, match="NaN"):
        fit_exponential_decay([0, 1, 2, 3], [1, float("nan"), 0.3, 0.1])
    with pytest.raises(FitError):
        readout_fidelity(1.2, 0.9)


def test_fit_measurements_groups_by_qubit_and_reports_bad_curves():
    rows = [{"qubit_id": 0, "experiment": "T1", "x": x, "y": float(np.exp(-x / 50))} for x in (0, 25, 50, 100, 200)]
    rows += [{"qubit_id": 1, "experiment": "T1", "x": 0.0, "y": 1.0}]
    out = fit_measurements(rows)
    assert abs(out[0]["T1"]["value"] - 50) < 0.5
    assert "error" in out[1]["T1"]


def test_simulated_device_is_deterministic_and_physical():
    a, b = SimulatedTransmon(2), SimulatedTransmon(2)
    assert a.t1_us == b.t1_us
    assert a.t2_us <= 2 * a.t1_us
    assert a.gate_error(**a.opt) == pytest.approx(a.coherence_limited_error)


def test_optimise_pulse_improves_fidelity():
    dev = SimulatedTransmon(1)
    res = optimise_pulse(dev, {"amplitude": 0.5, "drag_beta": 0.0, "frequency_offset_MHz": 0.0},
                         max_iterations=80, target_fidelity=0.9995)
    assert res["final_value"] > res["initial_value"]
    assert res["final_value"] <= res["coherence_limit"] + 1e-9
