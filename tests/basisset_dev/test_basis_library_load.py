"""Verify every shipped .g94 basis loads under libint.

For each ``python/vibeqc/basis_library/basis/<name>.g94`` we:

1. Discover the elements the basis covers by parsing the file
   (looking for ``<symbol>     0`` headers).
2. Build a single-atom molecule with the lightest element it
   covers (open-shell when needed; we use the spin multiplicity
   of the ground-state atom).
3. Construct ``vibeqc.BasisSet(mol, name)``. Any libint parse
   error fails the case.
4. For PRIMARY orbital bases (not auxiliary fitting bases),
   additionally run a single-point SCF and assert the energy
   is finite and negative — this catches "loaded but degenerate
   contraction" cases that the parser alone wouldn't notice.

Bases bundled with ECP blocks (vDZP, dhf-*, x2c-*, LANL*) are
expected to fail until libecpint is integrated; they are
``xfail``-tagged with the reason. Aux-fit / RI / JKfit / CABS
files are loaded but not SCF'd (they don't span an orbital
space).

Run::

    .venv/bin/python -m pytest tests/basisset_dev/test_basis_library_load.py -v --noconftest

(``--noconftest`` because the parent ``tests/conftest.py``
expects vibeqc fully built; we shim around it to allow the
import-stage skip to fire cleanly when vibeqc isn't available.)

When vibeqc IS importable, the parametrised cases run
end-to-end. When it isn't, every case is skipped at collection
time with a single informative message.
"""

from __future__ import annotations

import re
import sys
import types
from pathlib import Path
from typing import Optional

import pytest

PKG_PARENT = Path(__file__).resolve().parents[2] / "python"
BASIS_DIR = PKG_PARENT / "vibeqc" / "basis_library" / "basis"


# ---------- Vibeqc availability gate -----------------------------------------
#
# We attempt a real ``import vibeqc``. If that fails (the typical
# state during basisset_dev work without `pip install -e .`), the
# whole module is skipped with one collection-time message rather
# than spamming N parametrised failures.

if str(PKG_PARENT) not in sys.path:
    sys.path.insert(0, str(PKG_PARENT))

# Drop any prior shim from sister tests (e.g. test_basis_opt_stage1_arch).
for _mod_name in list(sys.modules):
    if _mod_name == "vibeqc" or _mod_name.startswith("vibeqc."):
        # Keep vibeqc._vibeqc_core* registered: the single-phase-init C
        # extension never re-runs PyInit on re-import, so deleting those
        # entries would strip the pybind11 def_submodule registrations
        # (...semiempirical.{nddo,xtb,indo}) for the rest of the pytest
        # process and break every later dotted import of them.
        if _mod_name == "vibeqc._vibeqc_core" or _mod_name.startswith(
            "vibeqc._vibeqc_core."
        ):
            continue
        del sys.modules[_mod_name]

# Guard against the 2026-05-09 overnight crash: the full 366-case
# parametrised SCF takes ~8 minutes wall and does ~130 atomic
# UHFs back-to-back — fine in isolation, but cumulative load on a
# 16 GB laptop running other things is the failure mode that
# brought down the host. Skip by default; opt in with
# ``VIBEQC_RUN_HEAVY_TESTS=1`` (env var set automatically by the
# vq daemon on compute-reference via the systemd unit's environment).
import os as _os  # noqa: E402

if _os.environ.get("VIBEQC_RUN_HEAVY_TESTS") not in ("1", "true", "yes"):
    import pytest as _pytest  # noqa: E402
    _pytest.skip(
        "Heavy parametrised test (366 SCF cases, ~8 minutes wall) — "
        "skipped on laptops to avoid cumulative-load OOM. Set "
        "VIBEQC_RUN_HEAVY_TESTS=1 to opt in (recommended only on "
        "compute-reference via vq submit; see reference_vq_queue memory).",
        allow_module_level=True,
    )

try:
    import vibeqc as vq  # type: ignore[import-not-found]
    VIBEQC_AVAILABLE = True
    VIBEQC_REASON = ""
except Exception as exc:  # noqa: BLE001
    VIBEQC_AVAILABLE = False
    VIBEQC_REASON = (
        f"vibeqc not importable: {exc!r}\n"
        f"To run the live load test, set up the worktree-private venv:\n"
        f"  python3 -m venv .venv && .venv/bin/pip install -e .\n"
        f"(this builds the C++ extension + libint; ~5-30 minutes)"
    )

pytestmark = pytest.mark.skipif(
    not VIBEQC_AVAILABLE, reason=VIBEQC_REASON,
)


# ---------- Test-case discovery ---------------------------------------------


# File-name suffixes that mark a non-primary (auxiliary) basis. These
# load fine but are not supposed to span an orbital space, so we skip
# the SCF step on them.
AUX_SUFFIX_RE = re.compile(
    r"-(rifit|ri|jkfit|jfit|c|j|jk|cabs|optri|f12|mini)\.g94$"
    r"|^augmentation-",
)

# Detection of ECP-bearing files: the splitter
# (scripts/basisset_dev/split_ecp_g94.py) leaves a sister `.ecp`
# file alongside any .g94 it had to strip ECP blocks out of. After
# the split, the .g94 itself is orbital-only and libint loads it
# fine — but the SCF still cannot run correctly without inline-ECP
# auto-population (Phase 14e), so test_single_atom_scf still xfails
# on these.


def has_ecp_sidecar(g94: Path) -> bool:
    return g94.with_suffix(".ecp").exists()

# Atomic numbers for the elements in atomic-number order; we look up
# by symbol when the basis lists "Li     0" etc. as block headers.
SYMBOL_TO_Z = {
    "H": 1, "He": 2, "Li": 3, "Be": 4, "B": 5, "C": 6, "N": 7, "O": 8,
    "F": 9, "Ne": 10, "Na": 11, "Mg": 12, "Al": 13, "Si": 14, "P": 15,
    "S": 16, "Cl": 17, "Ar": 18, "K": 19, "Ca": 20, "Sc": 21, "Ti": 22,
    "V": 23, "Cr": 24, "Mn": 25, "Fe": 26, "Co": 27, "Ni": 28, "Cu": 29,
    "Zn": 30, "Ga": 31, "Ge": 32, "As": 33, "Se": 34, "Br": 35, "Kr": 36,
    "Rb": 37, "Sr": 38, "Y": 39, "Zr": 40, "Nb": 41, "Mo": 42, "Tc": 43,
    "Ru": 44, "Rh": 45, "Pd": 46, "Ag": 47, "Cd": 48, "In": 49, "Sn": 50,
    "Sb": 51, "Te": 52, "I": 53, "Xe": 54, "Cs": 55, "Ba": 56,
    "La": 57, "Ce": 58, "Pr": 59, "Nd": 60, "Pm": 61, "Sm": 62, "Eu": 63,
    "Gd": 64, "Tb": 65, "Dy": 66, "Ho": 67, "Er": 68, "Tm": 69, "Yb": 70,
    "Lu": 71, "Hf": 72, "Ta": 73, "W": 74, "Re": 75, "Os": 76, "Ir": 77,
    "Pt": 78, "Au": 79, "Hg": 80, "Tl": 81, "Pb": 82, "Bi": 83, "Po": 84,
    "At": 85, "Rn": 86,
}

# Ground-state spin multiplicities (2S+1) for atoms 1-36. Used so a
# single-atom UHF on N gets multiplicity=4 etc. instead of failing on
# "odd electron count with multiplicity 1".
GROUND_STATE_MULT = {
    1: 2, 2: 1, 3: 2, 4: 1, 5: 2, 6: 3, 7: 4, 8: 3, 9: 2, 10: 1,
    11: 2, 12: 1, 13: 2, 14: 3, 15: 4, 16: 3, 17: 2, 18: 1, 19: 2,
    20: 1, 21: 2, 22: 3, 23: 4, 24: 7, 25: 6, 26: 5, 27: 4, 28: 3,
    29: 2, 30: 1, 31: 2, 32: 3, 33: 4, 34: 3, 35: 2, 36: 1,
    37: 2, 38: 1, 39: 2, 40: 3, 41: 6, 42: 7, 43: 6, 44: 5, 45: 4,
    46: 1, 47: 2, 48: 1, 49: 2, 50: 3, 51: 4, 52: 3, 53: 2, 54: 1,
    55: 2, 56: 1,
    # Lanthanide ground-state multiplicities (4f^n5d^k6s^2). The
    # f-block atomic SCF is notoriously unstable; we list the
    # term-symbol mults but tolerate convergence failures via the
    # try/except around the SCF call below.
    57: 2, 58: 3, 59: 4, 60: 5, 61: 6, 62: 7, 63: 8, 64: 9, 65: 6,
    66: 5, 67: 4, 68: 3, 69: 2, 70: 1, 71: 2,
    72: 3, 73: 4, 74: 5, 75: 6, 76: 5, 77: 4, 78: 3, 79: 2, 80: 1,
    81: 2, 82: 3, 83: 4, 84: 3, 85: 2, 86: 1,
}

_ELEMENT_HEADER_RE = re.compile(r"^([A-Z][a-z]?)\s+0\s*$", re.MULTILINE)


def smallest_element(g94_path: Path) -> Optional[int]:
    """Parse a .g94 and return the smallest atomic number it covers.

    Returns ``None`` if no element block can be parsed (likely a
    malformed or empty file — the test will then xfail).
    """
    text = g94_path.read_text(errors="replace")
    matches = _ELEMENT_HEADER_RE.findall(text)
    z_values = [SYMBOL_TO_Z[s] for s in matches if s in SYMBOL_TO_Z]
    return min(z_values) if z_values else None


def has_ecp_block(g94_path: Path) -> bool:
    """Detect whether a basis file ships an ECP block libint cannot parse."""
    text = g94_path.read_text(errors="replace")
    # Most BSE-emitted ECP blocks include "<Sym>-ECP" headers.
    return bool(re.search(r"^[A-Z][a-z]?-ECP\b", text, re.MULTILINE))


def discover_basis_files() -> list[Path]:
    if not BASIS_DIR.exists():
        return []
    return sorted(BASIS_DIR.glob("*.g94"))


def _id(p: Path) -> str:
    return p.stem


# ---------- Tests ------------------------------------------------------------


@pytest.fixture(scope="module")
def all_basis_files() -> list[Path]:
    files = discover_basis_files()
    if not files:
        pytest.fail(f"no .g94 files discovered under {BASIS_DIR}")
    return files


# Filename prefixes for SAP atomic-density helpers (used by libint's
# SCF guess, NOT orbital bases — running an SCF on them is meaningless).
SAP_PREFIXES = ("sap_",)


def _xfail_if_unsupported(g94: Path, *, scf_path: bool) -> None:
    """Raise pytest.xfail with an actionable reason for known unsupported cases.

    Returns silently if the basis is expected to load (and, when
    ``scf_path=True``, run SCF) cleanly. Distinguishes the load
    path (where ECP-bearing bases now succeed thanks to the
    sidecar split) from the SCF path (where they still need
    Phase 14e — auto-population of ecp_centers from the inline
    ECP definitions in the sidecar — to converge correctly with
    the right number of valence electrons).
    """
    name_lower = g94.stem.lower()
    if any(name_lower.startswith(p) for p in SAP_PREFIXES):
        pytest.xfail("SAP atomic-density helper (SCF guess fixture, not an orbital basis)")
    if scf_path and has_ecp_sidecar(g94):
        pytest.xfail(
            "ECP-bearing basis: load works (orbital-only g94 after split), "
            "but SCF needs Phase 14e (auto-populate ecp_centers from inline "
            "ECP definitions in the sidecar) to use the correct valence "
            "electron count"
        )


@pytest.mark.parametrize("g94", discover_basis_files(), ids=_id)
def test_loads(g94: Path):
    """``vq.BasisSet(mol, name)`` succeeds for every shipped .g94."""
    _xfail_if_unsupported(g94, scf_path=False)
    z = smallest_element(g94)
    if z is None:
        pytest.fail(f"{g94.name}: no element header found")

    mol = vq.Molecule(
        [vq.Atom(z, [0.0, 0.0, 0.0])],
        multiplicity=GROUND_STATE_MULT.get(z, 1),
    )
    # The basis-set name is the file stem (libint normalises lowercase).
    try:
        basis = vq.BasisSet(mol, g94.stem)
    except RuntimeError as exc:
        msg = str(exc)
        if "lmax_exceeded" in msg or "angular momentum" in msg:
            pytest.xfail(
                "libint compiled with insufficient max angular momentum "
                f"(file uses higher l than vibe-qc's libint build supports): {exc}"
            )
        raise
    assert basis is not None


@pytest.mark.parametrize(
    "g94",
    [p for p in discover_basis_files() if not AUX_SUFFIX_RE.search(p.name)],
    ids=_id,
)
def test_single_atom_scf(g94: Path):
    """Single-atom HF / UHF runs and produces a finite, negative energy.

    Only PRIMARY orbital bases get this treatment — auxiliary fitting
    bases (`-rifit`, `-jkfit`, `-cabs`, `-optri`, `-c`, `-j`, `-jk`,
    `augmentation-*`) span the product density rather than the orbital
    space and would fail or be meaningless under SCF.
    """
    _xfail_if_unsupported(g94, scf_path=True)
    z = smallest_element(g94)
    if z is None:
        pytest.fail(f"{g94.name}: no element header found")

    mult = GROUND_STATE_MULT.get(z, 1)
    mol = vq.Molecule([vq.Atom(z, [0.0, 0.0, 0.0])], multiplicity=mult)
    try:
        basis = vq.BasisSet(mol, g94.stem)
    except RuntimeError as exc:
        msg = str(exc)
        if "lmax_exceeded" in msg or "angular momentum" in msg:
            pytest.xfail(
                "libint compiled with insufficient max angular momentum: "
                f"{exc}"
            )
        raise
    runner = vq.run_rhf if mult == 1 else vq.run_uhf
    try:
        result = runner(mol, basis)
    except RuntimeError as exc:
        msg = str(exc)
        if "lmax_exceeded" in msg or "angular momentum" in msg:
            pytest.xfail(
                "SCF needs ERIs at angular momentum higher than libint's "
                f"compiled-in max: {exc}"
            )
        raise
    e = float(result.energy)
    assert e == e, f"NaN energy for {g94.stem} on Z={z}"
    assert e < 0.0, f"non-negative atomic energy {e:.6f} for {g94.stem} on Z={z}"
    # H atom: exact at -0.5; modest sanity bound for everything else.
    if z == 1:
        assert -0.51 < e < -0.40, f"H atom energy {e:.6f} far from -0.5"
