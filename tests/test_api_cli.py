"""REST upload endpoints and the command line."""
import json

import pytest

from backend import cli
from backend.data_loader import SAMPLE_DIR
from backend.importers import template_path

fastapi = pytest.importorskip("fastapi")
from fastapi.testclient import TestClient  # noqa: E402

from backend.api.main import app  # noqa: E402

client = TestClient(app)


def test_schemas_and_template_download():
    s = client.get("/api/v1/data/schemas").json()
    assert "hamiltonian" in s and s["papers"]["columns"][0]["name"] == "title"
    t = client.get("/api/v1/data/templates/hamiltonian")
    assert t.status_code == 200 and t.text.startswith("molecule,pauli,coeff")
    assert client.get("/api/v1/data/templates/nope").status_code == 404


def test_upload_valid_file_routes_to_agent():
    with open(template_path("device_calibration"), "rb") as fh:
        r = client.post("/api/v1/data/import/device_calibration", files={"file": ("cal.csv", fh, "text/csv")})
    body = r.json()
    assert r.status_code == 200 and body["agent"] == "hardware_agent" and body["success"]
    backends = client.get("/api/v1/hardware/backends").json()["backends"]
    assert any(b["type"] == "user_device" for b in backends)


def test_upload_invalid_file_returns_422_with_errors():
    r = client.post("/api/v1/data/import/hamiltonian",
                    files={"file": ("h.csv", b"molecule,pauli,coeff\nX,ZZ,abc\n", "text/csv")})
    assert r.status_code == 422
    assert any("coeff" in e for e in r.json()["errors"])


def test_demo_load_and_simulate():
    out = client.post("/api/v1/data/demo").json()
    assert all(v["success"] for v in out.values())
    sim = client.post("/api/v1/chemistry/simulate", json={"molecule": "SAMPLE_H2"}).json()
    assert sim["hamiltonian_source"] == "user_import" and sim["chemical_accuracy"]


def test_run_agent_returns_warnings():
    r = client.post("/api/v1/agents/chemistry_agent/run", json={"action": "simulate", "molecule": "BeH2"}).json()
    assert r["warnings"] and "SYNTHETIC" in r["warnings"][0]


# -- CLI -----------------------------------------------------------------------

def test_cli_vqe_json(capsys):
    assert cli.main(["vqe", "--hamiltonian", str(template_path("hamiltonian")), "--layers", "1", "--json"]) == 0
    out = json.loads(capsys.readouterr().out)
    assert out[0]["chem_accuracy"] is True


def test_cli_fit_and_rank(capsys):
    assert cli.main(["fit-qubits", str(SAMPLE_DIR / "qubit_measurements_demo.csv")]) == 0
    assert "T1_us" in capsys.readouterr().out
    assert cli.main(["--json", "rank-hypotheses", str(template_path("hypotheses"))]) == 0
    ranked = json.loads(capsys.readouterr().out)
    assert ranked[0]["rank"] == 1


def test_cli_run_circuit(tmp_path, capsys):
    f = tmp_path / "bell.qasm"
    f.write_text("OPENQASM 2.0; qreg q[2]; creg c[2]; h q[0]; cx q[0],q[1]; measure q -> c;")
    assert cli.main(["run-circuit", str(f), "--shots", "200", "--seed", "1", "--json"]) == 0
    assert set(json.loads(capsys.readouterr().out)["counts"]) <= {"00", "11"}


def test_cli_validation_error_exit_code(tmp_path, capsys):
    f = tmp_path / "bad.csv"
    f.write_text("title,year\nX,2020\n")
    assert cli.main(["validate", "papers", str(f)]) == 2
    assert "missing required column 'authors'" in capsys.readouterr().err


def test_cli_demo_runs(capsys):
    assert cli.main(["demo"]) == 0
    assert "Rank demo hypotheses" in capsys.readouterr().out
