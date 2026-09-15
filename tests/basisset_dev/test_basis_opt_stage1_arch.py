"""Stage-1 architecture smoke test for the basis-optimisation engine.

Exercises the parametrise + driver layer end-to-end on a *mock*
energy function (quadratic in the parameters). Does NOT call
vibe-qc SCF — the goal is to verify the optimiser pipeline works
in isolation, before stage 2 plugs in the real SCF energy.

The live single-atom HF stage-1 test (``test_basis_opt_stage1_live.py``,
also under tests/basisset_dev/) is opt-in and requires ``vibeqc``
to be importable; this one runs anywhere that has numpy + scipy.

Why have both:

* This file: catches architectural mistakes in pack/unpack, in the
  scipy/iminuit driver shims, and in the OptResult contract. Fast.
* The live file: validates that the whole stack including SCF
  reaches the published pob-TZVP H exponent. Slow.

Run with:

    .venv/bin/python -m pytest tests/basisset_dev/test_basis_opt_stage1_arch.py -v
"""

from __future__ import annotations

import sys
import types
from pathlib import Path

import numpy as np
import pytest

# Make the in-tree package importable without `pip install -e .`. Done
# at module scope rather than in a conftest because the parent
# ``tests/conftest.py`` does ``from vibeqc import ...`` and would
# fail to load if a fully-built vibeqc isn't available; pytest is
# expected to run this file with ``--noconftest``.
PKG_PARENT = Path(__file__).resolve().parents[2] / "python"
if str(PKG_PARENT) not in sys.path:
    sys.path.insert(0, str(PKG_PARENT))

# Prefer the real (C++-backed) vibeqc whenever it imports, so this module never
# mutates sys.modules for the rest of the session. Only when it cannot be
# imported at all do we substitute a minimal namespace package to reach the
# pure-Python submodules; that is safe there because no real package exists to
# clobber. Same guarded shape as the sister files test_bdiis_driver.py /
# test_basis_opt_gradients.py.
#
# This replaced an unconditional shim that snapshotted the loaded vibeqc modules
# and restored them after the import block. Two defects: the names imported
# below were *copies* of the real classes even on a built checkout, so a
# BasisParametrisation built here was not an instance of the genuine class; and
# any exception inside the import block skipped the restore, leaving a bare shim
# as ``vibeqc`` for every later-collected module. Registering copies under the
# real module names is what disarmed the fail-closed ECP-provenance guard in
# tests/test_periodic_ecp.py — see tests/basisset_dev/test_ld_penalty_inmemory.py
# for that failure written out in full.
try:
    import vibeqc  # noqa: F401
except Exception:
    for mod_name in list(sys.modules):
        # Keep vibeqc._vibeqc_core* registered: the single-phase-init C
        # extension never re-runs PyInit on re-import, so deleting those
        # entries would strip the pybind11 def_submodule registrations
        # (...semiempirical.{nddo,xtb,indo}) for the rest of the pytest
        # process and break every later dotted import of them.
        if mod_name == "vibeqc._vibeqc_core" or mod_name.startswith(
            "vibeqc._vibeqc_core."
        ):
            continue
        if mod_name == "vibeqc" or mod_name.startswith("vibeqc."):
            del sys.modules[mod_name]
    _shim = types.ModuleType("vibeqc")
    _shim.__path__ = [str(PKG_PARENT / "vibeqc")]
    sys.modules["vibeqc"] = _shim

from vibeqc.basis_crystal import (  # noqa: E402
    CrystalShell,
    parse_crystal_atom_basis_file,
)
from vibeqc.basis_optimization import (  # noqa: E402
    BasisParametrisation,
    FreeSpec,
    MockEnergy,
    Transform,
    optimize_scipy,
)


# ---------- Parametrise round-trip -------------------------------------------


@pytest.fixture(scope="module")
def pob_h() -> dict:
    """Parsed pob-TZVP H atom (the simplest real basis case)."""
    src = (
        Path(__file__).resolve().parents[2]
        / "python" / "vibeqc" / "basis_library" / "sources" / "pob-TZVP" / "01_H"
    )
    return {"H": parse_crystal_atom_basis_file(src)}


def test_parametrise_pack_unpack_roundtrip_log(pob_h):
    """pack ∘ unpack ∘ pack must be idempotent in log-space."""
    p = BasisParametrisation(
        atoms=pob_h,
        free=[
            FreeSpec(symbol="H", shell_idx=2, prim_idx=0,
                     field="exponent", transform=Transform.LOG,
                     bounds=(0.05, 1.0)),
        ],
    )
    x0 = p.pack()
    # The diffuse-most s exponent of pob-TZVP H is 0.1795111000.
    assert pytest.approx(np.exp(x0[0]), rel=1e-9) == 0.1795111000
    new_atoms = p.unpack(x0)
    # Read it back: should match the source exactly.
    assert pytest.approx(new_atoms["H"].shells[2].exponents[0], rel=1e-12) == 0.1795111000


def test_parametrise_unpack_modifies_only_target(pob_h):
    """Updating one free param must not perturb anything else."""
    p = BasisParametrisation(
        atoms=pob_h,
        free=[
            FreeSpec(symbol="H", shell_idx=2, prim_idx=0,
                     field="exponent", transform=Transform.LOG),
        ],
    )
    x0 = p.pack()
    new = p.unpack(x0 + 0.05)  # small log-space nudge
    src = pob_h["H"]
    out = new["H"]
    # Untouched: shell 0 (contracted s), shell 1 (single s), shell 3 (p).
    for si in (0, 1, 3):
        assert out.shells[si].exponents == src.shells[si].exponents
        assert out.shells[si].coefficients == src.shells[si].coefficients
    # Touched: shell 2 prim 0 exponent changed; coefficients untouched.
    assert out.shells[2].exponents[0] != src.shells[2].exponents[0]
    assert out.shells[2].coefficients == src.shells[2].coefficients


def test_parametrise_bounds_enforced_on_unpack(pob_h):
    p = BasisParametrisation(
        atoms=pob_h,
        free=[
            FreeSpec(symbol="H", shell_idx=2, prim_idx=0,
                     field="exponent", transform=Transform.LOG,
                     bounds=(0.10, 1.0)),
        ],
    )
    x_below = np.array([np.log(0.05)])  # below the lower bound 0.10
    with pytest.raises(ValueError, match="below lower bound"):
        p.unpack(x_below)


def test_parametrise_unknown_field_raises(pob_h):
    with pytest.raises(ValueError, match="unknown field"):
        BasisParametrisation(
            atoms=pob_h,
            free=[
                FreeSpec(symbol="H", shell_idx=2, prim_idx=0,
                         field="not_a_field"),
            ],
        )


def test_parametrise_coeff_on_sp_shell_disambiguates(pob_h):
    """An SP shell rejects ambiguous 'coeff' field with a useful message."""
    # pob-TZVP has no SP shells, so synthesise one for the test.
    from copy import deepcopy

    atoms = deepcopy(pob_h)
    atoms["H"].shells[0] = CrystalShell(
        shell_type="SP", occupancy=1.0, scale_factor=1.0,
        exponents=[1.0], coefficients=[1.0], coefficients_p=[1.0],
    )
    with pytest.raises(ValueError, match="ambiguous on SP shells"):
        BasisParametrisation(
            atoms=atoms,
            free=[
                FreeSpec(symbol="H", shell_idx=0, prim_idx=0,
                         field="coeff"),
            ],
        )


# ---------- Driver convergence on a mock objective ---------------------------


def test_scipy_driver_recovers_quadratic_minimum_1d():
    obj = MockEnergy(n_params=1, center=np.array([0.42]), e0=-3.14)
    res = optimize_scipy(obj, x0=np.array([1.0]), tol=1e-10)
    assert res.success
    assert res.driver == "scipy"
    assert pytest.approx(res.x[0], abs=1e-5) == 0.42
    assert pytest.approx(res.fun, abs=1e-9) == -3.14
    assert res.n_evaluations >= 2  # at least starting + one step
    assert len(res.history) > 0


def test_scipy_driver_recovers_quadratic_minimum_3d():
    center = np.array([0.5, -1.2, 0.0])
    H = np.diag([1.0, 4.0, 0.25])  # different curvatures per axis
    obj = MockEnergy(n_params=3, center=center, e0=0.0, hessian=H)
    res = optimize_scipy(obj, x0=np.array([2.0, 2.0, 2.0]), tol=1e-10)
    assert res.success
    np.testing.assert_allclose(res.x, center, atol=1e-4)
    assert res.fun < 1e-8


def test_parametrise_drives_mock_objective_through_log_space(pob_h):
    """End-to-end: parametrise → MockEnergy → scipy → recovers center.

    The MockEnergy lives in optimiser space (log of the exponent here).
    The optimiser's job is to find that center; pack/unpack is the
    bridge. The test asserts both that the optimiser converges and
    that the recovered exponent matches in physical space.
    """
    p = BasisParametrisation(
        atoms=pob_h,
        free=[
            FreeSpec(symbol="H", shell_idx=2, prim_idx=0,
                     field="exponent", transform=Transform.LOG,
                     bounds=(0.05, 1.0)),
        ],
    )
    target_phys = 0.30  # arbitrary "true minimum" different from default 0.1795
    target_log = np.log(target_phys)
    obj = MockEnergy(n_params=1, center=np.array([target_log]), e0=-1.0)
    x0 = p.pack()
    res = optimize_scipy(obj, x0=x0, bounds=p.optim_bounds(), tol=1e-10)
    assert res.success
    assert pytest.approx(res.x[0], abs=1e-5) == target_log
    # Round-trip back to physical space and verify pack/unpack is consistent.
    recovered_atoms = p.unpack(res.x)
    recovered_phys = recovered_atoms["H"].shells[2].exponents[0]
    assert pytest.approx(recovered_phys, rel=1e-5) == target_phys
