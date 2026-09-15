"""Tests for the BDIIS / GDIIS basis-optimisation driver.

Pure-Python: exercises ``optimize_bdiis`` and the condition-number
penalty on mock objectives and a parsed real basis, without calling
vibe-qc SCF. Uses the same ``vibeqc`` namespace shim as
``test_basis_opt_stage1_arch.py`` so it runs anywhere numpy + scipy are
available (no built C++ extension needed). Run with:

    .venv/bin/python -m pytest tests/basisset_dev/test_bdiis_driver.py -v --noconftest
"""

from __future__ import annotations

import sys
import types
from pathlib import Path

import numpy as np
import pytest

if sys.version_info >= (3, 11):
    import tomllib
else:  # pragma: no cover — repo requires 3.11+
    import tomli as tomllib

# Prefer the real (C++-backed) vibeqc whenever it imports — e.g. in CI's
# ``pytest --collect-only`` pass or on the build host — so this module
# never mutates sys.modules for the rest of the session. Only in a
# build-less dev checkout (where ``import vibeqc`` fails) do we shim a
# namespace package exposing the pure-Python basis_optimization
# submodules. This is the same trick as test_basis_opt_stage1_arch.py,
# but guarded so it cannot clobber a real vibeqc during shared collection.
REPO_ROOT = Path(__file__).resolve().parents[2]
try:
    import vibeqc  # noqa: F401
except Exception:
    PKG_PARENT = REPO_ROOT / "python"
    if str(PKG_PARENT) not in sys.path:
        sys.path.insert(0, str(PKG_PARENT))
    for mod_name in list(sys.modules):
        # Keep vibeqc._vibeqc_core* registered even here: the single-phase-init
        # C extension never re-runs PyInit on re-import, so deleting those
        # entries would strip the pybind11 def_submodule registrations
        # (...semiempirical.{nddo,xtb,indo}) for the rest of the pytest process.
        # Reachable when ``import vibeqc`` fails *after* the extension itself
        # loaded.
        if mod_name == "vibeqc._vibeqc_core" or mod_name.startswith(
            "vibeqc._vibeqc_core."
        ):
            continue
        if mod_name == "vibeqc" or mod_name.startswith("vibeqc."):
            del sys.modules[mod_name]
    _shim = types.ModuleType("vibeqc")
    _shim.__path__ = [str(PKG_PARENT / "vibeqc")]
    sys.modules["vibeqc"] = _shim

from vibeqc.basis_crystal import parse_crystal_atom_basis_file  # noqa: E402
from vibeqc.basis_optimization import (  # noqa: E402
    BDIIS_CITATION_KEYS,
    MockEnergy,
    cond_penalty_from_atom,
    condition_number_penalty,
    method_citations,
    optimize_bdiis,
    optimize_scipy,
)
from vibeqc.basis_optimization.ld_diagnostics import (  # noqa: E402
    LDDiagnostics,
    compute_overlap_diagnostics_from_atom,
)

DATABASE_TOML = REPO_ROOT / "python" / "vibeqc" / "output" / "citations" / "database.toml"


def _quad_grad(obj: MockEnergy):
    """Analytic gradient of MockEnergy: H·(x − center)."""
    return lambda x: obj.hessian @ (np.asarray(x, dtype=float) - obj.center)


# --------------------------------------------------------------------------
# Driver convergence
# --------------------------------------------------------------------------


def test_bdiis_recovers_quadratic_minimum_1d():
    obj = MockEnergy(n_params=1, center=np.array([0.42]), e0=-3.14)
    res = optimize_bdiis(obj, x0=np.array([1.0]), grad=_quad_grad(obj))
    assert res.success
    assert res.driver == "bdiis"
    assert res.x[0] == pytest.approx(0.42, abs=1e-6)
    assert res.fun == pytest.approx(-3.14, abs=1e-9)
    assert res.n_evaluations >= 2
    assert len(res.history) > 0


def test_bdiis_recovers_quadratic_minimum_3d_anisotropic():
    center = np.array([0.5, -1.2, 0.0])
    H = np.diag([1.0, 4.0, 0.25])  # different curvature per axis
    obj = MockEnergy(n_params=3, center=center, e0=0.0, hessian=H)
    res = optimize_bdiis(obj, x0=np.array([2.0, 2.0, 2.0]), grad=_quad_grad(obj))
    assert res.success
    np.testing.assert_allclose(res.x, center, atol=1e-5)
    assert res.fun < 1e-9


def test_bdiis_with_finite_difference_gradient():
    """No analytic gradient supplied → central-FD path must still converge."""
    center = np.array([0.3, -0.7])
    H = np.array([[2.0, 0.3], [0.3, 1.0]])  # coupled, positive definite
    obj = MockEnergy(n_params=2, center=center, e0=1.0, hessian=H)
    res = optimize_bdiis(obj, x0=np.array([1.5, 1.5]), fd_step=1e-5)
    assert res.success
    np.testing.assert_allclose(res.x, center, atol=1e-4)


def test_bdiis_matches_scipy_on_same_quadratic():
    center = np.array([0.5, -1.2, 0.0])
    H = np.diag([1.0, 4.0, 0.25])
    x0 = np.array([2.0, 2.0, 2.0])
    obj_b = MockEnergy(n_params=3, center=center, e0=0.0, hessian=H)
    obj_s = MockEnergy(n_params=3, center=center, e0=0.0, hessian=H)
    res_b = optimize_bdiis(obj_b, x0=x0, grad=_quad_grad(obj_b))
    res_s = optimize_scipy(obj_s, x0=x0, tol=1e-12)
    np.testing.assert_allclose(res_b.x, res_s.x, atol=1e-4)
    assert res_b.fun == pytest.approx(res_s.fun, abs=1e-8)


def test_bdiis_respects_bounds_constrained_minimum():
    """Unconstrained min at 0.42 lies below the box; BDIIS must stop at the
    lower bound 0.60 and flag convergence via the projected gradient."""
    obj = MockEnergy(n_params=1, center=np.array([0.42]), e0=0.0)
    res = optimize_bdiis(
        obj, x0=np.array([0.9]), bounds=[(0.60, 1.0)], grad=_quad_grad(obj)
    )
    assert res.x[0] == pytest.approx(0.60, abs=1e-6)
    assert res.success  # projected-gradient KKT criterion at the boundary


def test_bdiis_robust_on_rosenbrock():
    """Non-quadratic stress test of the BFGS + trust-region machinery."""

    class Rosenbrock:
        n_params = 2

        def __init__(self) -> None:
            self.n_calls = 0

        def __call__(self, x: np.ndarray) -> float:
            self.n_calls += 1
            x = np.asarray(x, dtype=float)
            return float((1.0 - x[0]) ** 2 + 100.0 * (x[1] - x[0] ** 2) ** 2)

    def rosen_grad(x: np.ndarray) -> np.ndarray:
        x = np.asarray(x, dtype=float)
        return np.array(
            [
                -2.0 * (1.0 - x[0]) - 400.0 * x[0] * (x[1] - x[0] ** 2),
                200.0 * (x[1] - x[0] ** 2),
            ]
        )

    obj = Rosenbrock()
    res = optimize_bdiis(
        obj,
        x0=np.array([-1.2, 1.0]),
        grad=rosen_grad,
        max_iter=2000,
        trust_radius=1.0,
        tol_grad=1e-5,
        tol_energy=1e-14,
    )
    np.testing.assert_allclose(res.x, [1.0, 1.0], atol=1e-3)
    assert res.fun < 1e-6


def test_bdiis_handles_infeasible_start():
    class InfObj:
        n_params = 1
        n_calls = 0

        def __call__(self, x):
            self.n_calls += 1
            return float("inf")

    res = optimize_bdiis(InfObj(), x0=np.array([0.0]))
    assert not res.success
    assert "infeasible" in res.message


# --------------------------------------------------------------------------
# Condition-number penalty (VandeVondele / OPTBASIS objective term)
# --------------------------------------------------------------------------


def _diag_with_kappa(kappa: float) -> LDDiagnostics:
    """Minimal LDDiagnostics carrying a chosen condition number."""
    return LDDiagnostics(
        overlap_eigenvalues=np.array([1.0 / kappa, 1.0]),
        lambda_min=1.0 / kappa,
        condition_number=kappa,
        n_independent=2,
        n_dependent=0,
        ld_detected=False,
    )


def test_condition_number_penalty_is_gamma_log_kappa():
    diag = _diag_with_kappa(1000.0)
    assert condition_number_penalty(diag, gamma=1e-3) == pytest.approx(
        1e-3 * np.log(1000.0)
    )


def test_condition_number_penalty_scales_linearly_in_gamma():
    diag = _diag_with_kappa(500.0)
    p1 = condition_number_penalty(diag, gamma=1e-3)
    p2 = condition_number_penalty(diag, gamma=5e-3)
    assert p2 == pytest.approx(5.0 * p1)


def test_condition_number_penalty_nonnegative_and_floored():
    # κ ≥ 1 → ln κ ≥ 0; the floor guards numerical noise pushing κ below 1.
    assert condition_number_penalty(_diag_with_kappa(1.0), gamma=1e-2) == 0.0
    noisy = _diag_with_kappa(0.999999)
    assert condition_number_penalty(noisy, gamma=1e-2) == 0.0


def test_condition_number_penalty_inf_on_degenerate():
    diag = _diag_with_kappa(1.0)
    diag.condition_number = float("inf")
    assert condition_number_penalty(diag) == float("inf")


def test_cond_penalty_on_real_pob_h_is_small_and_positive():
    """A well-conditioned shipped basis contributes a small, finite penalty."""
    src = (
        REPO_ROOT
        / "python" / "vibeqc" / "basis_library" / "sources" / "pob-TZVP" / "01_H"
    )
    atom = parse_crystal_atom_basis_file(src)
    diag = compute_overlap_diagnostics_from_atom(atom)
    p = cond_penalty_from_atom(atom, gamma=1e-3)
    assert np.isfinite(p)
    assert p >= 0.0
    assert p == pytest.approx(1e-3 * np.log(diag.condition_number))


# --------------------------------------------------------------------------
# Citation provenance — no dead-weight entries (CLAUDE.md § 8)
# --------------------------------------------------------------------------


def test_bdiis_citation_keys_resolve_in_database():
    db = tomllib.loads(DATABASE_TOML.read_text())
    entries = db.get("entries", {})
    missing = [k for k in BDIIS_CITATION_KEYS if k not in entries]
    assert not missing, (
        f"BDIIS cites citation-database keys that do not exist in "
        f"database.toml: {missing}. Add the [entries.<key>] blocks "
        f"(CLAUDE.md § 8)."
    )
    assert method_citations() == BDIIS_CITATION_KEYS
