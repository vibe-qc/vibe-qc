"""Phase 14g scaffolding: CRYSTAL → libecpint inline-primitive bridge.

Tests the pure-Python ``crystal_ecp_to_libecpint_arrays`` helper that
flattens a parsed :class:`vibeqc.basis_crystal.CrystalECP` into the
exact arrays ``libecpint::ECPIntegrator::set_ecp_basis(...)`` expects.

The full Phase 14g loop is:

    CRYSTAL ``INPUT`` block
        → ``parse_crystal_atom_basis`` (Phase 14f)
        → ``crystal_ecp_to_libecpint_arrays`` (Phase 14g — this file)
        → C++ ``compute_ecp_matrix_inline`` (Phase 14g — TODO)

The Python helper is independently testable: it's just bookkeeping
over the CrystalECP dataclass. The C++ side calls into libecpint's
``set_ecp_basis`` with the same array shapes and runs the integrator
exactly as the XML-fed ``compute_ecp_matrix`` does today.

References:
* CRYSTAL23 manual §"Effective core pseudo-potentials", eqs. 3.18-3.20.
* libecpint api.hpp:118 (set_ecp_basis signature) + ecp.cpp:82
  (ECP::addPrimitive — confirms per-primitive (n, l, α, c) layout).
* third_party/libecpint/install/share/libecpint/xml/ecp10mdf.xml —
  reference for "local lives at lval = max_proj + 1" convention
  (K shell layout: lval=4 placeholder local + lval=0..3 projectors).
"""

from __future__ import annotations

import sys
import types
from pathlib import Path

import pytest

PKG_PARENT = Path(__file__).resolve().parents[2] / "python"
if str(PKG_PARENT) not in sys.path:
    sys.path.insert(0, str(PKG_PARENT))
# Same guarded shape as ``test_crystal_ecp_parser.py``: prefer the real
# (C++-backed) vibeqc whenever it imports, and shim a namespace package only in
# a build-less dev checkout, where there is no real package to clobber. See that
# file for why the previous unconditional shim-and-restore was replaced (it
# handed this module *copies* of the real classes and could leave a bare shim
# behind if an import raised).
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
    LibecpintInlineArrays,
    crystal_ecp_to_libecpint_arrays,
)


def _make_ecp(*, m_local, m_per_ell, terms):
    """Test fixture: hand-build a CrystalECP with explicit groups."""
    return CrystalECP(
        znuc=9.0,
        m_local=m_local,
        m_per_ell=m_per_ell,
        terms=list(terms),
    )


def test_minimal_ecp_with_s_and_p_projectors():
    """1 local + 1 s-projector + 1 p-projector → 3 primitives, local at l=2."""
    ecp = _make_ecp(
        m_local=1,
        m_per_ell=(1, 1, 0, 0, 0),  # M0=1, M1=1, the rest zero
        terms=[
            CrystalECPTerm(ell="local", alpha=10.0, coefficient=4.0, n_pow=2),
            CrystalECPTerm(ell=0,       alpha=20.0, coefficient=5.0, n_pow=2),
            CrystalECPTerm(ell=1,       alpha=30.0, coefficient=6.0, n_pow=2),
        ],
    )
    out = crystal_ecp_to_libecpint_arrays(ecp)
    assert isinstance(out, LibecpintInlineArrays)
    assert out.n_primitives == 3
    # max projector = 1 (p); local sits one above at l=2.
    assert out.max_proj_ell == 1
    assert out.local_ell == 2
    # Source order preserved: local → l=2, then s → l=0, then p → l=1.
    assert out.ams == (2, 0, 1)
    # CRYSTAL r^2 terms; libecpint subtracts two from the power it is given.
    assert out.ns == (4, 4, 4)
    assert out.exponents == (10.0, 20.0, 30.0)
    assert out.coefficients == (4.0, 5.0, 6.0)


def test_ecp_with_full_spdf_projectors():
    """Mirrors the K (Z=19) ecp10mdf XML layout: f-projector present, local
    at lval=4."""
    ecp = _make_ecp(
        m_local=1,
        m_per_ell=(1, 1, 1, 1, 0),   # M0..M3 each 1, M4 = 0
        terms=[
            CrystalECPTerm(ell="local", alpha=1.0,  coefficient=0.0,   n_pow=2),
            CrystalECPTerm(ell=0,       alpha=6.88, coefficient=91.14, n_pow=2),
            CrystalECPTerm(ell=1,       alpha=4.72, coefficient=9.67,  n_pow=2),
            CrystalECPTerm(ell=2,       alpha=8.70, coefficient=-2.54, n_pow=2),
            CrystalECPTerm(ell=3,       alpha=13.93, coefficient=-16.55, n_pow=2),
        ],
    )
    out = crystal_ecp_to_libecpint_arrays(ecp)
    assert out.n_primitives == 5
    assert out.max_proj_ell == 3
    assert out.local_ell == 4
    # ams: local→4, s→0, p→1, d→2, f→3
    assert out.ams == (4, 0, 1, 2, 3)


def test_projectors_without_local_terms_get_a_zero_placeholder():
    """A CRYSTAL record with M = 0 still needs a local channel.

    libecpint reads its local channel off the highest angular momentum
    present, so without a primitive at ``local_ell`` the highest projector
    is silently promoted to the local potential -- which is what every pob
    record hit on the periodic route (#207)."""
    ecp = _make_ecp(
        m_local=0,
        m_per_ell=(1, 1, 0, 0, 0),
        terms=[
            CrystalECPTerm(ell=0, alpha=2.0, coefficient=5.0, n_pow=0),
            CrystalECPTerm(ell=1, alpha=3.0, coefficient=6.0, n_pow=0),
        ],
    )
    out = crystal_ecp_to_libecpint_arrays(ecp)
    assert out.local_ell == 2
    assert out.ams == (0, 1, 2)
    assert out.coefficients == (5.0, 6.0, 0.0)
    assert out.ns == (2, 2, 2)
    assert out.n_primitives == 3


def test_local_only_ecp_no_projectors():
    """All-local ECP (rare): no projector channels at all. local_ell
    falls back to 0 (the single channel becomes max-l by default)."""
    ecp = _make_ecp(
        m_local=2,
        m_per_ell=(0, 0, 0, 0, 0),
        terms=[
            CrystalECPTerm(ell="local", alpha=1.0, coefficient=2.0, n_pow=0),
            CrystalECPTerm(ell="local", alpha=3.0, coefficient=4.0, n_pow=2),
        ],
    )
    out = crystal_ecp_to_libecpint_arrays(ecp)
    assert out.n_primitives == 2
    assert out.max_proj_ell == -1
    assert out.local_ell == 0
    assert out.ams == (0, 0)


def test_n_powers_converted_to_the_libecpint_convention():
    """Each ``n_pow`` is raised by two on the way out.

    CRYSTAL states the term as r^n (CRYSTAL23 eqs. 3.18-3.19), while
    libecpint's ``GaussianECP`` constructor stores ``n - 2`` from the
    Gaussian-format power it is handed. Carrying NKL across unconverted
    turned every Stuttgart r^0 term into a singular r^-2 one (#207)."""
    ecp = _make_ecp(
        m_local=1,
        m_per_ell=(2, 0, 0, 0, 0),
        terms=[
            CrystalECPTerm(ell="local", alpha=1.0, coefficient=1.0, n_pow=0),
            CrystalECPTerm(ell=0,       alpha=2.0, coefficient=2.0, n_pow=1),
            CrystalECPTerm(ell=0,       alpha=3.0, coefficient=3.0, n_pow=2),
        ],
    )
    out = crystal_ecp_to_libecpint_arrays(ecp)
    assert out.ns == (2, 3, 4)


def test_invalid_ell_raises():
    """Out-of-range ell labels are caught before they reach libecpint."""
    ecp = _make_ecp(
        m_local=0,
        m_per_ell=(1, 0, 0, 0, 0),
        terms=[
            # ell=7 is out of the valid 0..4 range.
            CrystalECPTerm(ell=7, alpha=1.0, coefficient=2.0, n_pow=2),
        ],
    )
    with pytest.raises(ValueError, match="unrecognised ell"):
        crystal_ecp_to_libecpint_arrays(ecp)


def test_array_lengths_consistent():
    """All four flat arrays have the same length, matching n_primitives."""
    ecp = _make_ecp(
        m_local=1,
        m_per_ell=(2, 3, 1, 0, 0),
        terms=[
            CrystalECPTerm(ell="local", alpha=1.0, coefficient=0.1, n_pow=2),
            CrystalECPTerm(ell=0,       alpha=2.0, coefficient=0.2, n_pow=2),
            CrystalECPTerm(ell=0,       alpha=3.0, coefficient=0.3, n_pow=2),
            CrystalECPTerm(ell=1,       alpha=4.0, coefficient=0.4, n_pow=2),
            CrystalECPTerm(ell=1,       alpha=5.0, coefficient=0.5, n_pow=2),
            CrystalECPTerm(ell=1,       alpha=6.0, coefficient=0.6, n_pow=2),
            CrystalECPTerm(ell=2,       alpha=7.0, coefficient=0.7, n_pow=2),
        ],
    )
    out = crystal_ecp_to_libecpint_arrays(ecp)
    n = out.n_primitives
    assert n == 7
    assert len(out.exponents) == n
    assert len(out.coefficients) == n
    assert len(out.ams) == n
    assert len(out.ns) == n


def test_returns_frozen_dataclass_with_immutable_fields():
    """LibecpintInlineArrays is frozen — caller can hash it / can't
    accidentally mutate the conversion result."""
    ecp = _make_ecp(
        m_local=1,
        m_per_ell=(1, 0, 0, 0, 0),
        terms=[
            CrystalECPTerm(ell="local", alpha=1.0, coefficient=2.0, n_pow=2),
            CrystalECPTerm(ell=0,       alpha=3.0, coefficient=4.0, n_pow=2),
        ],
    )
    out = crystal_ecp_to_libecpint_arrays(ecp)
    with pytest.raises(Exception):
        out.local_ell = 99  # type: ignore[misc]
    # Tuples, not lists — by construction, can't append.
    assert isinstance(out.exponents, tuple)
    assert isinstance(out.coefficients, tuple)
    assert isinstance(out.ams, tuple)
    assert isinstance(out.ns, tuple)
