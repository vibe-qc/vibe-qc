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
