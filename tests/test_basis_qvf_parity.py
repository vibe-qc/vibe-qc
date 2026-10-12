"""The QVF sidecars describe exactly what the runtime files ship.

``basis/<name>.g94`` is what libint reads and ``basis/<name>.ecp`` is what
libecpint reads; ``qvf/<name>.qvf.json`` is the structured record of both.
Until 2026-09 the converter never read the sidecar and the importer dropped
every lmax=5 block, so the shipped dhf-TZVP record carried 18 of 35 ECPs and
linked none of them to an element.  These checks are pure text parsing, so
the whole library is covered in well under a second.
"""

from __future__ import annotations

import json
import re
from pathlib import Path

import pytest
import vibeqc as _vq

LIBRARY = Path(_vq.__file__).resolve().parent / "basis_library"
BASIS_DIR = LIBRARY / "basis"
QVF_DIR = LIBRARY / "qvf"

_ELEMENT_HEADER = re.compile(r"^\s*([A-Z][A-Za-z]?)\s+0\s*$", re.MULTILINE)
_ECP_HEADER = re.compile(r"^\s*([A-Z][A-Za-z]?)-ECP\s+(\d+)\s+(\d+)\s*$", re.MULTILINE)


def _symbols(text: str, pattern: re.Pattern) -> set[str]:
    return {m.group(1).capitalize() for m in pattern.finditer(text)}


def _g94_stems() -> list[str]:
    return sorted(p.stem for p in BASIS_DIR.glob("*.g94"))


@pytest.mark.parametrize("stem", _g94_stems())
def test_every_runtime_basis_has_a_qvf_record_with_the_same_elements(stem):
    qvf = QVF_DIR / f"{stem}.qvf.json"
    assert qvf.is_file(), f"{stem}: no QVF sidecar under basis_library/qvf/"
    record = json.loads(qvf.read_text())
    shipped = _symbols((BASIS_DIR / f"{stem}.g94").read_text(errors="replace"), _ELEMENT_HEADER)
    described = {k.capitalize() for k in record["elements"]}
    assert described == shipped, f"{stem}: QVF elements differ from the .g94"


@pytest.mark.parametrize("stem", sorted(p.stem for p in BASIS_DIR.glob("*.ecp")))
def test_every_sidecar_ecp_is_in_the_qvf_record_and_linked(stem):
    text = (BASIS_DIR / f"{stem}.ecp").read_text(errors="replace")
    sidecar = {
        m.group(1).capitalize(): int(m.group(3)) for m in _ECP_HEADER.finditer(text)
    }
    record = json.loads((QVF_DIR / f"{stem}.qvf.json").read_text())
    ecps = {k.capitalize(): v for k, v in (record.get("ecps") or {}).items()}
    assert set(ecps) == set(sidecar), f"{stem}: QVF ecps differ from the sidecar"
    for symbol, n_core in sidecar.items():
        assert ecps[symbol]["n_core_electrons"] == n_core, f"{stem}: {symbol} core count"
        assert ecps[symbol]["potentials"], f"{stem}: {symbol} has no potential channels"
    elements = {k.capitalize(): v for k, v in record["elements"].items()}
    for symbol in sidecar:
        if symbol in elements:
            assert elements[symbol].get("ecp_id"), f"{stem}: {symbol} not linked to its ECP"


@pytest.fixture(scope="module")
def basis_schema_validator():
    """Use the shipped schema, requiring full JSON Schema validation."""
    import jsonschema

    schema_path = LIBRARY.parent / "basis_toolkit" / "schemas" / "qvf_basis_v1.schema.json"
    schema = json.loads(schema_path.read_text(encoding="utf-8"))
    validator_type = jsonschema.validators.validator_for(schema)
    validator_type.check_schema(schema)
    return validator_type(schema)


def test_bundled_qvf_records_are_present():
    assert list(QVF_DIR.rglob("*.qvf.json")), "No bundled QVF basis records found"


@pytest.mark.parametrize(
    "record_path", sorted(QVF_DIR.rglob("*.qvf.json")), ids=lambda path: path.name
)
def test_every_bundled_qvf_record_validates_against_the_shipped_schema(
    record_path, basis_schema_validator
):
    record = json.loads(record_path.read_text(encoding="utf-8"))
    errors = list(basis_schema_validator.iter_errors(record))
    assert not errors, "\n".join(
        f"{record_path.name}: /{'/'.join(map(str, error.path))}: {error.message}"
        for error in errors
    )


@pytest.fixture
def qvf_record_with_ecp_term():
    """Select an actual bundled term so controls exercise the full record."""
    for path in sorted(QVF_DIR.rglob("*.qvf.json")):
        record = json.loads(path.read_text(encoding="utf-8"))
        for ecp in record.get("ecps", {}).values():
            for potential in ecp["potentials"]:
                if potential["terms"]:
                    return record, potential["terms"][0]
    pytest.fail("No bundled ECP term found")


@pytest.mark.parametrize("power", range(5))
def test_ecp_gaussian_powers_zero_through_four_are_valid(
    power, qvf_record_with_ecp_term, basis_schema_validator
):
    record, term = qvf_record_with_ecp_term
    term["power"] = power
    assert not list(basis_schema_validator.iter_errors(record))


@pytest.mark.parametrize("power", ["2", 1.5, -1, 5, None, True])
def test_invalid_ecp_powers_are_rejected(
    power, qvf_record_with_ecp_term, basis_schema_validator
):
    record, term = qvf_record_with_ecp_term
    term["power"] = power
    errors = list(basis_schema_validator.iter_errors(record))
    assert errors
    assert all(error.path[-1] == "power" for error in errors)


def test_missing_ecp_power_is_rejected(
    qvf_record_with_ecp_term, basis_schema_validator
):
    record, term = qvf_record_with_ecp_term
    del term["power"]
    assert any(
        error.validator == "required" and "power" in error.message
        for error in basis_schema_validator.iter_errors(record)
    )
