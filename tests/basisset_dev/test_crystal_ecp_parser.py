"""CRYSTAL `INPUT`-keyword ECP block parser tests.

Phase 14f: ``vibeqc.basis_crystal.parse_crystal_atom_basis`` no longer
raises ``NotImplementedError`` on ECP-bearing CRYSTAL files (5th-period
pob: Rb-I, Cs-Po). It now consumes the INPUT block fully into
``CrystalAtomBasis.ecp`` so downstream code (Phase 14e auto-population
of ``ecp_centers``) can build a libecpint setup directly from the
parsed structure.

These tests construct synthetic CRYSTAL files matching the format from
the CRYSTAL23 manual page 84 and verify the round-trip.
"""

from __future__ import annotations

import sys
import textwrap
import types
from pathlib import Path

import pytest

PKG_PARENT = Path(__file__).resolve().parents[2] / "python"
if str(PKG_PARENT) not in sys.path:
    sys.path.insert(0, str(PKG_PARENT))
# Prefer the real (C++-backed) vibeqc whenever it imports, so this module never
# mutates sys.modules for the rest of the session. Only in a build-less dev
# checkout (where ``import vibeqc`` fails) do we shim a namespace package to
# reach the pure-Python ``basis_crystal``; that is safe there precisely because
# no real package exists to clobber. Same guarded shape as the sister files
# test_bdiis_driver.py / test_basis_opt_gradients.py.
#
# This replaced an unconditional shim that snapshotted the loaded vibeqc
# modules and restored them after the import block. Two defects: the names
# imported below were *copies* of the real classes even on a built checkout, so
# an isinstance check against the genuine class silently failed; and any
# exception inside the import block skipped the restore, leaving a bare shim as
# ``vibeqc`` for every later-collected module. Registering copies under the real
# module names is what disarmed the fail-closed ECP-provenance guard in
# tests/test_periodic_ecp.py — see tests/basisset_dev/test_ld_penalty_inmemory.py
# for that failure written out in full.
try:
    import vibeqc  # noqa: F401
except Exception:
    for _mod_name in list(sys.modules):
        # Keep vibeqc._vibeqc_core* registered: the single-phase-init C
        # extension never re-runs PyInit on re-import, so deleting those
        # entries would strip the pybind11 def_submodule registrations
        # (...semiempirical.{nddo,xtb,indo}) for the rest of the pytest
        # process and break every later dotted import of them.
        if _mod_name == "vibeqc._vibeqc_core" or _mod_name.startswith(
            "vibeqc._vibeqc_core."
        ):
            continue
        if _mod_name == "vibeqc" or _mod_name.startswith("vibeqc."):
            del sys.modules[_mod_name]
    _shim = types.ModuleType("vibeqc")
    _shim.__path__ = [str(PKG_PARENT / "vibeqc")]
    sys.modules["vibeqc"] = _shim

from vibeqc.basis_crystal import (  # noqa: E402
    CrystalECP,
    CrystalECPTerm,
    parse_crystal_atom_basis,
)


# A synthetic Rb-like atom file: 200+37 header (ECP-using Rb), 1 valence
# shell (s, single primitive). The INPUT block uses Stuttgart-Köln
# small-core RECP-style numbers — synthetic but realistic shape.
# Layout per CRYSTAL23 manual §"Effective core pseudo-potentials":
#   line 1: 237 1                     — atom header (200+Z=237, NSHELL=1)
#   line 2: INPUT                     — ECP keyword
#   line 3: 9.0 0 1 1 1 0 0           — ZNUC, M, M0..M4 (3 ℓ-projected
#                                       terms in ℓ=0,1,2)
#   lines 4-6: 3 records (α, C, n)    — ℓ=0, ℓ=1, ℓ=2 in source order
#   line 7: 0 0 1 0.0 1.0             — shell header (S, 1 primitive)
#   line 8: 0.5 1.0                   — exponent, coefficient
RB_ECP_FILE = textwrap.dedent("""
    237  1
    INPUT
    9.0 0 1 1 1 0 0
    1.0  0.0  2
    2.0  17.0 2
    3.0  -3.0 2
    0  0  1  0.0  1.0
    0.5  1.0
""").strip() + "\n"


def test_parses_input_block_into_ecp_dataclass():
    atom = parse_crystal_atom_basis(RB_ECP_FILE, source="<rb-test>")
    assert atom.Z == 37        # 200+37 header → Z=37 (Rb)
    assert atom.has_ecp
    assert atom.ecp is not None

    ecp = atom.ecp
    assert isinstance(ecp, CrystalECP)
    assert ecp.znuc == pytest.approx(9.0)   # 9 valence electrons
    assert ecp.m_local == 0
    assert ecp.m_per_ell == (1, 1, 1, 0, 0)
    assert ecp.total_terms == 3
    assert len(ecp.terms) == 3


def test_term_groups_carry_correct_ell_labels():
    atom = parse_crystal_atom_basis(RB_ECP_FILE, source="<rb-test>")
    ells = [term.ell for term in atom.ecp.terms]
    # Source order: M=0 local terms (none here), then M0=1 ell=0 term,
    # M1=1 ell=1 term, M2=1 ell=2 term.
    assert ells == [0, 1, 2]


def test_term_payload_is_alpha_coef_n():
    atom = parse_crystal_atom_basis(RB_ECP_FILE, source="<rb-test>")
    # First payload line is "1.0  0.0  2" — but with M_local=0 that
    # line is the FIRST ell=0 term:    α=1.0, C=0.0, n=2
    # Wait — the first FOUR data lines are: 1.0 0.0 2 / 2.0 17 2 /
    # 3.0 -3 2 / 4.0 1.5 2. With M_local=0 and M0..M2 = 1 each, the
    # parser reads exactly 3 lines (the fourth is treated as the
    # orbital basis). So:
    #   ell=0 term: α=1.0,  C=0.0,  n=2
    #   ell=1 term: α=2.0,  C=17.0, n=2
    #   ell=2 term: α=3.0,  C=-3.0, n=2
    ecp = atom.ecp
    assert (ecp.terms[0].alpha, ecp.terms[0].coefficient, ecp.terms[0].n_pow) \
        == (pytest.approx(1.0), pytest.approx(0.0), 2)
    assert (ecp.terms[1].alpha, ecp.terms[1].coefficient, ecp.terms[1].n_pow) \
        == (pytest.approx(2.0), pytest.approx(17.0), 2)
    assert (ecp.terms[2].alpha, ecp.terms[2].coefficient, ecp.terms[2].n_pow) \
        == (pytest.approx(3.0), pytest.approx(-3.0), 2)


def test_orbital_basis_after_ecp_block_still_parses():
    """Once the ECP INPUT block is consumed, the rest of the file
    (NSHELL × shell records) parses identically to a non-ECP atom.
    """
    atom = parse_crystal_atom_basis(RB_ECP_FILE, source="<rb-test>")
    assert len(atom.shells) == 1
    shell = atom.shells[0]
    assert shell.shell_type == "S"
    assert shell.n_primitives == 1
    assert shell.exponents == [pytest.approx(0.5)]
    assert shell.coefficients == [pytest.approx(1.0)]


def test_atom_without_ecp_still_works_unchanged():
    """Backward-compat: ordinary basis files (no 200+Z) still parse
    cleanly, and ``ecp`` stays None.
    """
    text = textwrap.dedent("""
        1  1
        0  0  3  1.0  1.0
        34.0613410   0.00602520
         5.1235746   0.04502109
         1.1646626   0.20189726
    """).strip() + "\n"
    atom = parse_crystal_atom_basis(text, source="<H-test>")
    assert atom.Z == 1
    assert not atom.has_ecp
    assert atom.ecp is None
    assert len(atom.shells) == 1
    assert atom.shells[0].n_primitives == 3
