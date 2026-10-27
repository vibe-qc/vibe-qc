"""Tests for the derivative-free drivers and the CMA-ES-then-BDIIS recipe.

Covers ``optimize_bobyqa`` (Py-BOBYQA), ``optimize_dfols`` (DFO-LS),
``optimize_cmaes`` (pycma) and
``recipes.global_then_polish.optimize_global_then_polish`` on the same mock
objectives the scipy and BDIIS driver tests use, plus the pack/unpack
round trip through log space, bounds enforcement, the ``OptResult`` contract
and the citation provenance of each driver.

Pure-Python: no vibe-qc SCF. Uses the guarded ``vibeqc`` namespace shim of
``test_bdiis_driver.py`` so it runs without a built C++ extension. Every
optimiser backend is optional (vibe-qc's ``[basisopt-drivers]`` extra); a
test that needs one skips when it is not installed, and the
``*_missing_package_raises`` tests pin the error a caller gets instead. Run
with:

    .venv/bin/python -m pytest tests/basisset_dev/test_basis_opt_dfo_drivers.py -v --noconftest
"""

from __future__ import annotations

import sys
import types
from pathlib import Path

import numpy as np
import pytest

if sys.version_info >= (3, 11):
    import tomllib
else:  # pragma: no cover - repo requires 3.11+
    import tomli as tomllib

REPO_ROOT = Path(__file__).resolve().parents[2]
try:
    import vibeqc  # noqa: F401
except Exception:
    PKG_PARENT = REPO_ROOT / "python"
    if str(PKG_PARENT) not in sys.path:
        sys.path.insert(0, str(PKG_PARENT))
    for mod_name in list(sys.modules):
        # Keep vibeqc._vibeqc_core* registered: the single-phase-init C
        # extension never re-runs PyInit on re-import (see the sister files).
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
    CMAES_CITATION_KEYS,
    DFO_CITATION_KEYS,
    BasisParametrisation,
    FreeSpec,
    MockEnergy,
    MockResiduals,
    OptResult,
    Transform,
    cmaes_default_sigma,
    cmaes_method_citations,
    dfo_method_citations,
    optimize_bdiis,
    optimize_bobyqa,
    optimize_cmaes,
    optimize_dfols,
)
from vibeqc.basis_optimization.recipes.global_then_polish import (  # noqa: E402
    GLOBAL_THEN_POLISH_CITATION_KEYS,
    optimize_global_then_polish,
)
from vibeqc.basis_optimization.recipes.global_then_polish import (  # noqa: E402
    method_citations as global_then_polish_citations,
)

DATABASE_TOML = REPO_ROOT / "python" / "vibeqc" / "output" / "citations" / "database.toml"
POB_H = REPO_ROOT / "python" / "vibeqc" / "basis_library" / "sources" / "pob-TZVP" / "01_H"


# --------------------------------------------------------------------------
# Fixtures and helpers
# --------------------------------------------------------------------------


@pytest.fixture
def need_bobyqa():
    return pytest.importorskip("pybobyqa")


@pytest.fixture
def need_dfols():
    return pytest.importorskip("dfols")


@pytest.fixture
def need_cma():
    return pytest.importorskip("cma")


@pytest.fixture(scope="module")
def pob_h() -> dict:
    return {"H": parse_crystal_atom_basis_file(POB_H)}


def _quad_grad(obj: MockEnergy):
    return lambda x: obj.hessian @ (np.asarray(x, dtype=float) - obj.center)


class DoubleWell:
    """f(x) = (x^2 - 1)^2 + 0.3 x: two basins, the global minimum near x = -1.04
    and a higher local one near x = +0.96. Started at x0 = 1.0 a local driver
    stays in the right-hand basin."""

    n_params = 1

    def __init__(self) -> None:
        self.n_calls = 0

    def __call__(self, x: np.ndarray) -> float:
        self.n_calls += 1
        t = float(np.asarray(x, dtype=float)[0])
        return (t * t - 1.0) ** 2 + 0.3 * t

    @staticmethod
    def grad(x: np.ndarray) -> np.ndarray:
        t = float(np.asarray(x, dtype=float)[0])
        return np.array([4.0 * t * (t * t - 1.0) + 0.3])


class InfBelowZero:
    """Forbidden region x < 0 (objective +inf), minimum at x = 0.5."""

    n_params = 1

    def __init__(self) -> None:
        self.n_calls = 0

    def __call__(self, x: np.ndarray) -> float:
        self.n_calls += 1
        t = float(np.asarray(x, dtype=float)[0])
        return float("inf") if t < 0.0 else (t - 0.5) ** 2


def _assert_optresult_contract(res: OptResult, driver: str) -> None:
    assert isinstance(res, OptResult)
    assert res.driver == driver
    assert isinstance(res.x, np.ndarray)
    assert np.isfinite(res.fun)
    assert isinstance(res.success, bool)
    assert isinstance(res.message, str) and res.message
    assert res.n_evaluations >= 1
    assert res.wall_seconds >= 0.0


# --------------------------------------------------------------------------
# Py-BOBYQA
# --------------------------------------------------------------------------


def test_bobyqa_recovers_quadratic_minimum_1d(need_bobyqa):
    obj = MockEnergy(n_params=1, center=np.array([0.42]), e0=-3.14)
    res = optimize_bobyqa(obj, x0=np.array([1.0]), rhoend=1e-10)
    _assert_optresult_contract(res, "pybobyqa")
    assert res.success, res.message
    assert res.x[0] == pytest.approx(0.42, abs=1e-6)
    assert res.fun == pytest.approx(-3.14, abs=1e-9)
    assert len(res.history) == res.n_evaluations == obj.n_calls


def test_bobyqa_recovers_quadratic_minimum_3d_anisotropic(need_bobyqa):
    center = np.array([0.5, -1.2, 0.0])
    H = np.diag([1.0, 4.0, 0.25])
    obj = MockEnergy(n_params=3, center=center, e0=0.0, hessian=H)
    res = optimize_bobyqa(obj, x0=np.array([2.0, 2.0, 2.0]), rhoend=1e-10)
    assert res.success, res.message
    np.testing.assert_allclose(res.x, center, atol=1e-5)
    assert res.fun < 1e-9


def test_bobyqa_uses_fewer_evaluations_than_fd_bdiis_on_3d(need_bobyqa):
    """The point of the driver: a model-based step costs about one
    evaluation, a central-difference gradient costs 2 n per step."""
    center = np.array([0.5, -1.2, 0.0])
    H = np.diag([1.0, 4.0, 0.25])
    x0 = np.array([2.0, 2.0, 2.0])
    obj_b = MockEnergy(n_params=3, center=center, e0=0.0, hessian=H)
    obj_d = MockEnergy(n_params=3, center=center, e0=0.0, hessian=H)
    res_b = optimize_bobyqa(obj_b, x0=x0, rhoend=1e-8)
    res_d = optimize_bdiis(obj_d, x0=x0, fd_step=1e-5, tol_grad=1e-6)
    np.testing.assert_allclose(res_b.x, res_d.x, atol=1e-4)
    assert res_b.n_evaluations < res_d.n_evaluations


def test_bobyqa_respects_bounds_constrained_minimum(need_bobyqa):
    obj = MockEnergy(n_params=1, center=np.array([0.42]), e0=0.0)
    res = optimize_bobyqa(obj, x0=np.array([0.9]), bounds=[(0.60, 1.0)], rhoend=1e-10)
    assert res.success, res.message
    assert res.x[0] == pytest.approx(0.60, abs=1e-8)
    assert all(0.60 - 1e-12 <= xk[0] <= 1.0 + 1e-12 for xk, _ in res.history)


def test_bobyqa_projects_starting_point_into_box(need_bobyqa):
    obj = MockEnergy(n_params=1, center=np.array([0.42]), e0=0.0)
    res = optimize_bobyqa(obj, x0=np.array([2.0]), bounds=[(0.60, 1.0)])
    assert res.x[0] == pytest.approx(0.60, abs=1e-6)


def test_bobyqa_half_open_bounds_and_inf_region(need_bobyqa):
    obj = InfBelowZero()
    res = optimize_bobyqa(obj, x0=np.array([1.5]), bounds=[(None, 5.0)], rhoend=1e-10)
    assert res.success, res.message
    assert res.x[0] == pytest.approx(0.5, abs=1e-6)


def test_bobyqa_bounds_length_mismatch_raises(need_bobyqa):
    obj = MockEnergy(n_params=2, center=np.zeros(2))
    with pytest.raises(ValueError, match="bounds length"):
        optimize_bobyqa(obj, x0=np.zeros(2), bounds=[(0.0, 1.0)])


def test_bobyqa_missing_package_raises(monkeypatch):
    monkeypatch.setitem(sys.modules, "pybobyqa", None)
    obj = MockEnergy(n_params=1, center=np.zeros(1))
    with pytest.raises(RuntimeError, match="pip install Py-BOBYQA"):
        optimize_bobyqa(obj, x0=np.zeros(1))


# --------------------------------------------------------------------------
# DFO-LS
# --------------------------------------------------------------------------


def test_dfols_recovers_least_squares_minimum(need_dfols):
    center = np.array([0.42, -1.2])
    obj = MockResiduals(n_params=2, center=center, matrix=np.array([[1.0, 0.3], [0.0, 2.0]]))
    res = optimize_dfols(obj, x0=np.array([1.0, 1.0]), rhoend=1e-10)
    _assert_optresult_contract(res, "dfols")
    assert res.success, res.message
    np.testing.assert_allclose(res.x, center, atol=1e-6)
    assert res.fun < 1e-10
    assert len(res.history) == res.n_evaluations


def test_dfols_residuals_kwarg_overrides_objective(need_dfols):
    center = np.array([0.42, -1.2, 0.7])
    res = optimize_dfols(
        None, x0=np.zeros(3),
        residuals=lambda x: np.asarray(x, dtype=float) - center, rhoend=1e-10,
    )
    assert res.success, res.message
    np.testing.assert_allclose(res.x, center, atol=1e-6)


def test_dfols_respects_bounds(need_dfols):
    obj = MockResiduals(n_params=1, center=np.array([0.42]))
    res = optimize_dfols(obj, x0=np.array([0.9]), bounds=[(0.60, 1.0)], rhoend=1e-10)
    assert res.x[0] == pytest.approx(0.60, abs=1e-8)
    assert res.fun == pytest.approx((0.60 - 0.42) ** 2, abs=1e-10)


def test_dfols_refuses_a_scalar_objective(need_dfols):
    """A scalar energy is not a residual: a single r = Omega would minimise
    Omega^2 and push a negative energy *up*. The driver must say so."""
    obj = MockEnergy(n_params=1, center=np.array([0.42]), e0=-3.14)
    with pytest.raises(TypeError, match="optimize_bobyqa"):
        optimize_dfols(obj, x0=np.array([1.0]))


def test_dfols_missing_package_raises(monkeypatch):
    monkeypatch.setitem(sys.modules, "dfols", None)
    obj = MockResiduals(n_params=1, center=np.zeros(1))
    with pytest.raises(RuntimeError, match="pip install DFO-LS"):
        optimize_dfols(obj, x0=np.zeros(1))


def test_mock_residuals_objective_is_the_sum_of_squares():
    obj = MockResiduals(n_params=2, center=np.array([1.0, 2.0]), matrix=np.array([[1.0, 0.0], [0.0, 3.0]]))
    x = np.array([2.0, 1.0])
    r = obj.residuals(x)
    np.testing.assert_allclose(r, [1.0, -3.0])
    assert obj(x) == pytest.approx(10.0)
    with pytest.raises(ValueError, match="matrix has shape"):
        MockResiduals(n_params=2, center=np.zeros(2), matrix=np.zeros((2, 3)))


# --------------------------------------------------------------------------
# CMA-ES
# --------------------------------------------------------------------------


def test_cmaes_recovers_quadratic_minimum_1d(need_cma):
    obj = MockEnergy(n_params=1, center=np.array([0.42]), e0=-3.14)
    res = optimize_cmaes(obj, x0=np.array([1.0]), bounds=[(-5.0, 5.0)], seed=1, tol_fun=1e-12)
    _assert_optresult_contract(res, "cmaes")
    assert res.success, res.message
    assert res.x[0] == pytest.approx(0.42, abs=1e-4)
    assert res.fun == pytest.approx(-3.14, abs=1e-8)
    assert len(res.history) == res.n_evaluations == obj.n_calls
    assert res.message.startswith("converged")


def test_cmaes_recovers_quadratic_minimum_3d_anisotropic(need_cma):
    center = np.array([0.5, -1.2, 0.0])
    H = np.diag([1.0, 4.0, 0.25])
    obj = MockEnergy(n_params=3, center=center, e0=0.0, hessian=H)
    res = optimize_cmaes(
        obj, x0=np.array([2.0, 2.0, 2.0]), bounds=[(-5.0, 5.0)] * 3,
        seed=1, tol_fun=1e-12, max_evaluations=4000,
    )
    assert res.success, res.message
    np.testing.assert_allclose(res.x, center, atol=1e-3)
    assert res.fun < 1e-7


def test_cmaes_respects_bounds_constrained_minimum(need_cma):
    obj = MockEnergy(n_params=1, center=np.array([0.42]), e0=0.0)
    res = optimize_cmaes(obj, x0=np.array([0.9]), bounds=[(0.60, 1.0)], seed=3, tol_fun=1e-12)
    assert res.x[0] == pytest.approx(0.60, abs=1e-3)
    assert all(0.60 - 1e-12 <= xk[0] <= 1.0 + 1e-12 for xk, _ in res.history)


def test_cmaes_tolerates_inf_region(need_cma):
    obj = InfBelowZero()
    res = optimize_cmaes(obj, x0=np.array([1.5]), sigma0=0.5, seed=2, tol_fun=1e-12)
    assert res.success, res.message
    assert res.x[0] == pytest.approx(0.5, abs=1e-3)


def test_cmaes_escapes_the_starting_basin_where_bdiis_does_not(need_cma):
    """The roadmap's reason for the driver: from x0 = 1.0 the local BDIIS
    settles in the right-hand basin of the double well, CMA-ES with a step
    sized from the box finds the lower left-hand one."""
    x0 = np.array([1.0])
    bounds = [(-3.0, 3.0)]
    local = optimize_bdiis(DoubleWell(), x0, bounds=bounds, grad=DoubleWell.grad)
    assert local.success
    assert local.x[0] > 0.0                      # stuck in the starting basin
    glob = optimize_cmaes(DoubleWell(), x0, bounds=bounds, seed=1, tol_fun=1e-10)
    assert glob.x[0] < 0.0                       # crossed to the global basin
    assert glob.fun < local.fun - 0.1


def test_cmaes_is_reproducible_with_a_seed(need_cma):
    center = np.array([0.5, -1.2])
    a = optimize_cmaes(MockEnergy(n_params=2, center=center), np.zeros(2), seed=7, max_evaluations=200)
    b = optimize_cmaes(MockEnergy(n_params=2, center=center), np.zeros(2), seed=7, max_evaluations=200)
    np.testing.assert_array_equal(a.x, b.x)
    assert a.n_evaluations == b.n_evaluations


def test_cmaes_budget_exhaustion_is_not_success(need_cma):
    obj = MockEnergy(n_params=3, center=np.array([0.5, -1.2, 0.0]))
    res = optimize_cmaes(obj, x0=np.array([2.0, 2.0, 2.0]), seed=1, max_evaluations=8)
    assert not res.success
    assert "budget" in res.message
    assert res.n_evaluations <= 8 + res.raw.popsize


def test_cmaes_default_sigma_comes_from_the_optimiser_space_box():
    sigma, stds = cmaes_default_sigma([(0.0, 1.0), (0.0, 4.0), (None, None)], 3)
    assert sigma == pytest.approx(0.25)
    assert stds == [1.0, 4.0, 1.0]
    assert cmaes_default_sigma(None, 2) == (0.5, None)
    assert cmaes_default_sigma([(0.0, 2.0), (-1.0, 1.0)], 2) == (0.5, None)
    assert cmaes_default_sigma([(None, 1.0)], 1) == (0.5, None)
    with pytest.raises(ValueError, match="bounds length"):
        cmaes_default_sigma([(0.0, 1.0)], 2)


def test_cmaes_rejects_nonpositive_sigma(need_cma):
    obj = MockEnergy(n_params=1, center=np.zeros(1))
    with pytest.raises(ValueError, match="sigma0"):
        optimize_cmaes(obj, x0=np.zeros(1), sigma0=0.0)


def test_cmaes_missing_package_raises(monkeypatch):
    monkeypatch.setitem(sys.modules, "cma", None)
    obj = MockEnergy(n_params=1, center=np.zeros(1))
    with pytest.raises(RuntimeError, match="pip install cma"):
        optimize_cmaes(obj, x0=np.zeros(1))


# --------------------------------------------------------------------------
# CMA-ES then BDIIS recipe
# --------------------------------------------------------------------------


def test_global_then_polish_finds_the_global_basin(need_cma):
    obj = DoubleWell()
    res = optimize_global_then_polish(
        obj, np.array([1.0]), bounds=[(-3.0, 3.0)], grad=DoubleWell.grad,
        cmaes_max_evaluations=200, cmaes_seed=1,
    )
    _assert_optresult_contract(res, "cmaes+bdiis")
    assert res.success, res.message
    assert res.x[0] < 0.0
    assert DoubleWell.grad(res.x)[0] == pytest.approx(0.0, abs=1e-4)
    stage1, stage2 = res.raw["cmaes"], res.raw["bdiis"]
    assert stage1.driver == "cmaes" and stage2.driver == "bdiis"
    assert res.fun <= stage1.fun
    assert res.n_evaluations == stage1.n_evaluations + stage2.n_evaluations
    assert len(res.history) == len(stage1.history) + len(stage2.history)
    assert res.message.startswith("cmaes: ") and "; bdiis: " in res.message


def test_global_then_polish_reaches_quadratic_minimum_without_gradient(need_cma):
    center = np.array([0.5, -1.2, 0.0])
    obj = MockEnergy(n_params=3, center=center, hessian=np.diag([1.0, 4.0, 0.25]))
    res = optimize_global_then_polish(
        obj, np.array([2.0, 2.0, 2.0]), bounds=[(-5.0, 5.0)] * 3,
        cmaes_max_evaluations=100, cmaes_seed=1, bdiis_tol_grad=1e-6,
    )
    assert res.success, res.message
    np.testing.assert_allclose(res.x, center, atol=1e-4)


def test_global_then_polish_reports_an_infeasible_search(need_cma):
    class Never:
        n_params = 1
        n_calls = 0

        def __call__(self, x):
            self.n_calls += 1
            return float("inf")

    res = optimize_global_then_polish(Never(), np.zeros(1), cmaes_max_evaluations=20, cmaes_seed=1)
    assert not res.success
    assert res.raw["bdiis"] is None
    assert "skipped" in res.message


# --------------------------------------------------------------------------
# Through the parametrisation (log-space exponents, physical bounds)
# --------------------------------------------------------------------------


@pytest.mark.parametrize("driver", ["bobyqa", "cmaes"])
def test_parametrise_drives_mock_objective_through_log_space(pob_h, driver):
    if driver == "bobyqa":
        pytest.importorskip("pybobyqa")
    else:
        pytest.importorskip("cma")
    p = BasisParametrisation(
        atoms=pob_h,
        free=[FreeSpec(symbol="H", shell_idx=2, prim_idx=0, field="exponent",
                       transform=Transform.LOG, bounds=(0.05, 1.0))],
    )
    target_phys = 0.30
    obj = MockEnergy(n_params=1, center=np.array([np.log(target_phys)]), e0=-1.0)
    x0 = p.pack()
    if driver == "bobyqa":
        res = optimize_bobyqa(obj, x0=x0, bounds=p.optim_bounds(), rhoend=1e-10)
        tol = 1e-6
    else:
        res = optimize_cmaes(obj, x0=x0, bounds=p.optim_bounds(), seed=1, tol_fun=1e-12)
        tol = 1e-3
    assert res.success, res.message
    recovered = p.unpack(res.x)["H"].shells[2].exponents[0]
    assert recovered == pytest.approx(target_phys, rel=tol)
    # every sampled point stayed inside the physical box, so unpack never raised
    assert all(np.log(0.05) - 1e-12 <= xk[0] <= np.log(1.0) + 1e-12 for xk, _ in res.history)


# --------------------------------------------------------------------------
# Citation provenance
# --------------------------------------------------------------------------


def test_driver_citation_keys_resolve_in_database():
    db = tomllib.loads(DATABASE_TOML.read_text(encoding="utf-8"))
    entries = db.get("entries", {})
    for keys in (DFO_CITATION_KEYS, CMAES_CITATION_KEYS, GLOBAL_THEN_POLISH_CITATION_KEYS):
        missing = [k for k in keys if k not in entries]
        assert not missing, f"citation keys missing from database.toml: {missing}"
    assert dfo_method_citations() == DFO_CITATION_KEYS
    assert cmaes_method_citations() == CMAES_CITATION_KEYS
    assert global_then_polish_citations() == GLOBAL_THEN_POLISH_CITATION_KEYS
    assert set(GLOBAL_THEN_POLISH_CITATION_KEYS) == set(CMAES_CITATION_KEYS) | set(BDIIS_CITATION_KEYS)
    assert len(GLOBAL_THEN_POLISH_CITATION_KEYS) == len(set(GLOBAL_THEN_POLISH_CITATION_KEYS))
