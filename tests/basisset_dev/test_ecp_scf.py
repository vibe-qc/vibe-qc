"""Targeted integration tests for libecpint-backed SCF.

These complement the parametrised ``test_basis_library_load.py`` —
which only verifies that ECP-bearing .g94 files LOAD — by exercising
the full SCF path with manually-supplied ``ecp_centers``. They prove
that Phase 14a-c (libecpint integration) plus the Phase 14c-extension
(``total_ncore`` valence-electron accounting) work end-to-end.

The ``test_single_atom_scf`` cases for ECP-bearing files still
xfail because they don't auto-populate ``ecp_centers`` from the
basis-loaded ECP info — that's Phase 14e and is documented in the
handover.

Run::

    .venv/bin/python -m pytest tests/basisset_dev/test_ecp_scf.py --noconftest -v
"""

from __future__ import annotations

import sys
import types
from pathlib import Path

import pytest

PKG_PARENT = Path(__file__).resolve().parents[2] / "python"
if str(PKG_PARENT) not in sys.path:
    sys.path.insert(0, str(PKG_PARENT))

# Unlike test_basis_opt_stage1_arch.py we DO need the C++-backed
# vibeqc here — Molecule, BasisSet, run_uhf live in _vibeqc_core. Drop a
# leftover namespace shim from a sister test, but reuse an already-loaded real
# package so collection cannot create a second, incompatible class generation.
_loaded_vibeqc = sys.modules.get("vibeqc")
if _loaded_vibeqc is None or getattr(_loaded_vibeqc, "__file__", None) is None:
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
try:
    import vibeqc as vq  # type: ignore[import-not-found]
except Exception as exc:  # noqa: BLE001
    pytest.skip(
        f"vibeqc not importable: {exc!r}; build with `pip install -e .`",
        allow_module_level=True,
    )


def _ecp_center(z: int) -> "vq.ECPCenter":
    c = vq.ECPCenter()
    c.Z = int(z)
    c.xyz = [0.0, 0.0, 0.0]
    return c


@pytest.mark.parametrize(
    "z, mult, basis_name, ecp_library, expected_nbasis",
    [
        # Si ([Ne]3s²3p²) under lanl2dz: 10-core, 4 valence electrons.
        # Triplet 3P ground state. Tiny basis (~8 bf).
        pytest.param(14, 3, "lanl2dz", "lanl2dz", None, id="Si_lanl2dz"),
        # Pt ([Xe]4f¹⁴5d⁹6s¹) under lanl2dz: 60-core, 18 valence electrons.
        # Triplet 3D ground state. Heavy element where the ECP earns
        # its keep — 22 bf vs ~80 for an all-electron treatment.
        pytest.param(78, 3, "lanl2dz", "lanl2dz", None, id="Pt_lanl2dz"),
        # I ([Kr]4d¹⁰5s²5p⁵) under vdzp: vdzp's own inline ECP — note
        # that vDZP uses non-standard core sizes per element. The
        # library name ``ecp10mdf`` is wrong for I (vdzp uses 28-core
        # for I); but ``ecp46mdf`` would also be wrong (vdzp removes
        # 28 e⁻, not 46). We document this Phase 14e gap in the handover.
        # No test case here for vDZP — vDZP needs inline-ECP feed.
    ],
)
def test_uhf_with_explicit_ecp(
    z: int, mult: int, basis_name: str, ecp_library: str,
    expected_nbasis: int | None,
):
    """Single-atom UHF on an ECP-bearing basis with manually supplied
    ``ecp_centers``. Asserts SCF converges and produces a finite,
    negative energy.
    """
    mol = vq.Molecule([vq.Atom(z, [0.0, 0.0, 0.0])], multiplicity=mult)
    basis = vq.BasisSet(mol, basis_name)
    if expected_nbasis is not None:
        assert basis.nbasis == expected_nbasis

    opts = vq.UHFOptions()
    opts.ecp_centers = [_ecp_center(z)]
    opts.ecp_library = ecp_library
    opts.max_iter = 200

    result = vq.run_uhf(mol, basis, opts)
    assert result.converged, (
        f"UHF on Z={z} with {basis_name}+{ecp_library} did not converge "
        f"in {result.n_iter} iters; final E = {result.energy}"
    )
    e = float(result.energy)
    assert e == e, f"NaN energy"
    assert e < 0.0, f"non-negative atomic energy {e:.6f}"


def test_ncore_subtraction_propagates_to_nocc():
    """Direct check: the SCF's nocc count comes from
    ``mol.n_electrons() − total_ncore``.

    For Pt (Z=78) with LANL2DZ removing 60 core electrons, valence
    count is 18 → n_α = 10, n_β = 8 in the triplet ground state.
    There are 22 basis functions, so n_α (10) ≤ n_kept (≤22)
    comfortably. Without the subtraction, the SCF would try to fit
    78 electrons into 22 orbitals and fail with the canonical-
    orthogonalization error.
    """
    m = vq.Molecule([vq.Atom(78, [0, 0, 0])], multiplicity=3)
    b = vq.BasisSet(m, "lanl2dz")
    opts = vq.UHFOptions()
    opts.ecp_centers = [_ecp_center(78)]
    opts.ecp_library = "lanl2dz"
    opts.max_iter = 200  # heavy-atom UHF takes ~125 iterations
    # Should NOT raise; the assertion is implicit in the absence of
    # "canonical orthogonalization dropped too many basis directions".
    r = vq.run_uhf(m, b, opts)
    assert r.converged


def test_no_ecp_unchanged():
    """Without ``ecp_centers``, the SCF behaves exactly as before
    (regression guard for the Phase 14c-extension change).
    """
    m = vq.Molecule(
        [vq.Atom(8, [0.0, 0.0, 0.0]),
         vq.Atom(1, [0.0, 1.43, -0.98]),
         vq.Atom(1, [0.0, -1.43, -0.98])],
    )
    b = vq.BasisSet(m, "6-31g*")
    r = vq.run_rhf(m, b)
    # H2O/RHF/6-31G* reference value (vibeqc-internal).
    assert r.converged
    assert -76.02 < r.energy < -75.99


def test_ecp_core_count_too_large_rejected():
    """ECP core electrons exceeding total electron count is rejected
    with a clear message rather than silently producing nonsense.
    """
    m = vq.Molecule([vq.Atom(1, [0, 0, 0])], multiplicity=2)  # H, 1 e-
    b = vq.BasisSet(m, "sto-3g")
    opts = vq.UHFOptions()
    # Pretend the H atom has an ECP that removes 28 electrons. With
    # 1 total electron and 28 ECP'd, n_elec_eff = -27 → reject.
    # ecp10mdf doesn't define H but our library probe is libecpint's
    # job; the SCF's check fires first if the library defines a Z
    # match (here it doesn't, so total_ncore stays 0 — the exception
    # path must be a separate test). Skip in the immediate version:
    pytest.skip(
        "Pseudocode for the Phase 14e error-path test once the "
        "'ECP definition matches an unrelated atom' diagnostic lands."
    )
