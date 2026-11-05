"""Bring-your-own-data importers and the shipped templates."""
import json

import pytest

from backend.importers import (
    SCHEMAS, ImportValidationError, hamiltonians_from_rows, load_rows, template_path, validate_rows,
)


@pytest.mark.parametrize("kind", sorted(SCHEMAS))
def test_every_template_validates_and_header_matches_schema(kind):
    path = template_path(kind)
    assert path.exists(), f"missing template {path}"
    header = path.read_text().splitlines()[0].split(",")
    assert header == SCHEMAS[kind].header
    assert len(load_rows(kind, path)) >= 2


def test_missing_required_column_lists_expected_header():
    with pytest.raises(ImportValidationError) as e:
        load_rows("papers", b"title,year\nA,2020\n", "x.csv")
    msg = str(e.value)
    assert "missing required column 'authors'" in msg
    assert "expected header" in msg


def test_row_level_errors_are_reported_with_row_numbers():
    csv = (b"qubit_id,experiment,x,y\n"
           b"0,T1,0,1.0\n"
           b"zero,T1,1,0.9\n"      # row 3 bad int
           b"0,T9,2,0.8\n"         # row 4 bad choice
           b"0,T1,3,1.5\n"         # row 5 above max
           b"0,RB,0,0.9\n")        # row 6 RB length < 1
    with pytest.raises(ImportValidationError) as e:
        load_rows("qubit_measurements", csv, "m.csv")
    errs = e.value.errors
    assert any(s.startswith("row 3, column 'qubit_id'") for s in errs)
    assert any(s.startswith("row 4, column 'experiment'") for s in errs)
    assert any(s.startswith("row 5, column 'y'") for s in errs)
    assert any(s.startswith("row 6,") and "RB" in s for s in errs)


def test_aliases_case_insensitive_and_choice_normalised():
    rows = load_rows("qubit_measurements", b"Qubit_ID,EXPERIMENT,delay_us,population\n1,t1,5,0.5\n", "m.csv")
    assert rows == [{"qubit_id": 1, "experiment": "T1", "x": 5.0, "y": 0.5, "device": None, "measured_at": None}]


def test_json_and_tsv_inputs():
    data = json.dumps({"rows": [{"name": "exp", "scientific_value": 0.7}]}).encode()
    assert load_rows("experiments", data, "e.json")[0]["scientific_value"] == 0.7
    tsv = b"name\tscientific_value\nexp\t0.2\n"
    assert load_rows("experiments", tsv, "e.tsv")[0]["name"] == "exp"
    with pytest.raises(ImportValidationError, match="invalid JSON"):
        load_rows("experiments", b"{nope", "e.json")


def test_xlsx_input(tmp_path):
    openpyxl = pytest.importorskip("openpyxl")
    wb = openpyxl.Workbook()
    ws = wb.active
    ws.append(["title", "authors", "year"])
    ws.append(["Paper", "A; B", 2021])
    path = tmp_path / "papers.xlsx"
    wb.save(path)
    rows = load_rows("papers", path)
    assert rows[0]["authors"] == ["A", "B"] and rows[0]["year"] == 2021


def test_empty_and_unknown():
    with pytest.raises(ImportValidationError, match="no data rows"):
        load_rows("papers", b"title,authors,year\n", "p.csv")
    with pytest.raises(ImportValidationError, match="unknown data kind"):
        validate_rows("nope", [{"a": 1}])
    with pytest.raises(FileNotFoundError):
        load_rows("papers", "does/not/exist.csv")


def test_hamiltonians_from_rows_groups_and_sums_terms():
    rows = load_rows("hamiltonian", b"molecule,pauli,coeff,nuclear_repulsion\n"
                                    b"A,Z0,0.5,0.1\nA,Z0,0.25,\nA,X1 X0,0.2,\nB,I,1.0,\n", "h.csv")
    hams = hamiltonians_from_rows(rows)
    assert set(hams) == {"A", "B"}
    assert dict(hams["A"].terms) == {"IZ": 0.75, "XX": 0.2}
    assert hams["A"].nuclear_repulsion == 0.1


def test_hamiltonian_bad_label_is_a_validation_error():
    rows = load_rows("hamiltonian", b"molecule,pauli,coeff\nA,Q7,0.5\n", "h.csv")
    with pytest.raises(ImportValidationError, match="molecule 'A'"):
        hamiltonians_from_rows(rows)
