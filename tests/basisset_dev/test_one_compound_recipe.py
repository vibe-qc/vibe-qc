"""Stage-2 one-compound recipe and the periodic single-point objective.

Pure-Python: the periodic SCF is a mock runner injected through the
``scf_runner`` / ``basis_factory`` seams of ``PeriodicSinglePointEnergy``,
so this file checks the plumbing (what the runner is handed, failure
scoring, the driver dispatch, the result report) without a built core.
The one test that builds the real LiH ``PeriodicSystem`` skips without
``vibeqc``. Same guarded ``vibeqc`` shim as test_bdiis_driver.py. Run with:

    .venv/bin/python -m pytest tests/basisset_dev/test_one_compound_recipe.py -v --noconftest
"""

from __future__ import annotations

import sys
import types
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest

REPO_ROOT = Path(__file__).resolve().parents[2]
try:
    import vibeqc  # noqa: F401
except Exception:
    PKG_PARENT = REPO_ROOT / "python"
    if str(PKG_PARENT) not in sys.path:
        sys.path.insert(0, str(PKG_PARENT))
    for mod_name in list(sys.modules):
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
    DFO_CITATION_KEYS,
    BasisParametrisation,
    FreeSpec,
    Transform,
)
from vibeqc.basis_optimization.io import TempBasisLibrary  # noqa: E402
from vibeqc.basis_optimization.objective import PeriodicSinglePointEnergy  # noqa: E402
from vibeqc.basis_optimization.recipes.one_compound import (  # noqa: E402
    EXPECTED_EXPONENT_WINDOW,
    PT2013_LIH_HF_PER_CELL_HA,
    OneCompoundResult,
    run_lih_smoke,
    run_one_compound,
)

POB = REPO_ROOT / "python" / "vibeqc" / "basis_library" / "sources" / "pob-TZVP"
H_DIFFUSE_PUBLISHED = 0.1795111000
MOCK_OPTIMUM = 0.15          # inside the design's expected window
MOCK_E0 = -8.07


class StubSystem:
    def unit_cell_molecule(self):
        raise AssertionError("the stub basis_factory must bypass unit_cell_molecule")


def stub_basis_factory(system, name, atoms):
    """A 'BasisSet' that just carries the candidate H diffuse exponent."""
    return SimpleNamespace(name=name, h_diffuse=atoms["H"].shells[2].exponents[0])


class MockPeriodicRunner:
    """Quadratic in ln(exponent) around MOCK_OPTIMUM; records every call."""

    def __init__(self, *, converge: bool = True, raise_on: int | None = None) -> None:
        self.calls: list[dict] = []
        self.converge = converge
        self.raise_on = raise_on

    def __call__(self, system, basis, **kwargs):
        self.calls.append({"system": system, "basis": basis, **kwargs})
        if self.raise_on is not None and len(self.calls) == self.raise_on:
            raise RuntimeError("synthetic SCF blow-up")
        d = np.log(basis.h_diffuse) - np.log(MOCK_OPTIMUM)
        return SimpleNamespace(energy=MOCK_E0 + 0.5 * d * d, converged=self.converge, n_iter=7)


def _h_parametrisation(bounds=(0.05, 1.0)) -> BasisParametrisation:
    return BasisParametrisation(
        atoms={"H": parse_crystal_atom_basis_file(POB / "01_H")},
        free=[FreeSpec("H", 2, 0, "exponent", transform=Transform.LOG,
                       bounds=bounds, label="H_diffuse_s")],
    )


# --------------------------------------------------------------------------
# PeriodicSinglePointEnergy
# --------------------------------------------------------------------------


def test_periodic_objective_hands_the_runner_the_candidate_basis_and_settings():
    par = _h_parametrisation()
    runner = MockPeriodicRunner()
    with TempBasisLibrary() as lib:
        obj = PeriodicSinglePointEnergy(
            parametrisation=par, system_factory=StubSystem, library=lib,
            kpoints=(3, 3, 3), method="RHF", runner_kwargs={"conv_tol_energy": 1e-9},
            scf_runner=runner, basis_factory=stub_basis_factory,
        )
        x0 = par.pack()
        e = obj(x0)
        assert obj.n_params == 1 and obj.n_calls == 1
        assert e == pytest.approx(MOCK_E0 + 0.5 * (np.log(H_DIFFUSE_PUBLISHED / MOCK_OPTIMUM)) ** 2)
        call = runner.calls[0]
        # candidate basis written under the temp library, unique per call
        assert (Path(lib.path) / "basis" / f"{call['basis'].name}.g94").is_file()
        assert call["basis"].h_diffuse == pytest.approx(H_DIFFUSE_PUBLISHED)
        # engine defaults plus the caller's settings
        assert call["jk_method"] == "bipole"
        assert call["output_qvf"] is False and call["progress"] is False
        assert call["kpoints"] == (3, 3, 3) and call["method"] == "RHF"
        assert call["functional"] is None
        assert call["conv_tol_energy"] == 1e-9
        assert str(call["output"]).startswith(str(Path(lib.path) / "periodic"))
        assert obj.last_result.n_iter == 7
        # a second call writes a second, differently named basis
        obj(x0 + 0.1)
        assert runner.calls[1]["basis"].name != call["basis"].name
        assert runner.calls[1]["basis"].h_diffuse == pytest.approx(H_DIFFUSE_PUBLISHED * np.exp(0.1))


def test_periodic_objective_scores_failures_as_inf_or_raises():
    par = _h_parametrisation()
    with TempBasisLibrary() as lib:
        nonconv = PeriodicSinglePointEnergy(
            parametrisation=par, system_factory=StubSystem, library=lib,
            scf_runner=MockPeriodicRunner(converge=False), basis_factory=stub_basis_factory,
        )
        assert nonconv(par.pack()) == float("inf")
        blowup = PeriodicSinglePointEnergy(
            parametrisation=par, system_factory=StubSystem, library=lib,
            scf_runner=MockPeriodicRunner(raise_on=1), basis_factory=stub_basis_factory,
        )
        assert blowup(par.pack()) == float("inf")
        strict = PeriodicSinglePointEnergy(
            parametrisation=par, system_factory=StubSystem, library=lib,
            scf_runner=MockPeriodicRunner(converge=False), basis_factory=stub_basis_factory,
            on_failure="raise",
        )
        with pytest.raises(RuntimeError, match="did not converge"):
            strict(par.pack())
    with pytest.raises(ValueError, match="on_failure"):
        PeriodicSinglePointEnergy(
            parametrisation=par, system_factory=StubSystem, library=None, on_failure="nan"
        )


# --------------------------------------------------------------------------
# The recipe through the drivers (mock SCF)
# --------------------------------------------------------------------------


@pytest.mark.parametrize("driver,expected_keys", [
    ("bobyqa", DFO_CITATION_KEYS),
    ("bdiis", BDIIS_CITATION_KEYS),
    ("scipy", ()),
])
def test_lih_smoke_recovers_the_mock_optimum(driver, expected_keys):
    if driver == "bobyqa":
        pytest.importorskip("pybobyqa")
    runner = MockPeriodicRunner()
    res = run_lih_smoke(
        kpoints=(2, 2, 2), driver=driver, perturbation=1.5,
        scf_runner=runner, basis_factory=stub_basis_factory, max_evaluations=60,
    )
    assert isinstance(res, OneCompoundResult)
    assert res.compound == "LiH" and res.method == "RHF" and res.driver == driver
    assert res.kpoints == (2, 2, 2)
    assert res.starting_value == pytest.approx(1.5 * H_DIFFUSE_PUBLISHED)
    assert res.optimal_value == pytest.approx(MOCK_OPTIMUM, rel=1e-3)
    assert res.optimal_energy == pytest.approx(MOCK_E0, abs=1e-6)
    assert res.improvement_mha > 0.0
    assert res.in_expected_window
    assert res.reference_energy == PT2013_LIH_HF_PER_CELL_HA
    assert res.reference_delta_mha == pytest.approx((MOCK_E0 - PT2013_LIH_HF_PER_CELL_HA) * 1e3)
    assert res.n_evaluations == len(runner.calls)
    assert res.citations == tuple(expected_keys)
    assert res.converged, res.message
    text = res.summary()
    assert "one-compound LiH refit" in text and "reference" in text and "design window" in text
    # the runner saw the LiH primitive-cell factory's product on every call
    assert all(c["kpoints"] == (2, 2, 2) and c["method"] == "RHF" for c in runner.calls)


def test_lih_smoke_with_cmaes_prepass():
    pytest.importorskip("cma")
    runner = MockPeriodicRunner()
    res = run_lih_smoke(
        kpoints=(2, 2, 2), driver="cmaes+bdiis", perturbation=1.5,
        scf_runner=runner, basis_factory=stub_basis_factory, max_evaluations=40, cmaes_seed=3,
    )
    assert res.optimal_value == pytest.approx(MOCK_OPTIMUM, rel=1e-3)
    assert res.driver == "cmaes+bdiis"
    assert "hansen_cmaes_2001" in res.citations and "daga_optbasis_2020" in res.citations


def test_lih_smoke_unknown_driver_and_unknown_method_reference():
    with pytest.raises(ValueError, match="unknown driver"):
        run_lih_smoke(driver="newton", scf_runner=MockPeriodicRunner(),
                      basis_factory=stub_basis_factory)
    res = run_lih_smoke(
        method="RKS", driver="scipy", scf_runner=MockPeriodicRunner(),
        basis_factory=stub_basis_factory, max_evaluations=20,
    )
    assert res.reference_energy is None and res.reference_delta_mha is None
    assert "reference" not in res.summary()


def test_cond_penalty_adds_the_atomic_ln_kappa_terms():
    from vibeqc.basis_optimization.ld_diagnostics import cond_penalty_from_atom

    par = _h_parametrisation()
    plain = run_one_compound(
        par, StubSystem, compound="H-only", driver="scipy", max_evaluations=5,
        scf_runner=MockPeriodicRunner(), basis_factory=stub_basis_factory,
    )
    pen = run_one_compound(
        par, StubSystem, compound="H-only", driver="scipy", max_evaluations=5,
        scf_runner=MockPeriodicRunner(), basis_factory=stub_basis_factory,
        use_cond_penalty=True, cond_gamma=1e-3,
    )
    expected = cond_penalty_from_atom(par.atoms["H"], gamma=1e-3)
    assert pen.starting_energy - plain.starting_energy == pytest.approx(expected, abs=1e-12)
    assert expected > 0.0


def test_run_one_compound_needs_exactly_one_free_parameter():
    par = BasisParametrisation(
        atoms={"H": parse_crystal_atom_basis_file(POB / "01_H")},
        free=[FreeSpec("H", 2, 0, "exponent", transform=Transform.LOG),
              FreeSpec("H", 3, 0, "exponent", transform=Transform.LOG)],
    )
    with pytest.raises(ValueError, match="one free parameter"):
        run_one_compound(par, StubSystem, scf_runner=MockPeriodicRunner(),
                         basis_factory=stub_basis_factory)


def test_expected_window_brackets_stage1_and_published():
    from vibeqc.basis_optimization.recipes.one_compound import MEASURED_LIH_RHF_K222_EXPONENT

    lo, hi = EXPECTED_EXPONENT_WINDOW
    assert lo < 0.1308 < H_DIFFUSE_PUBLISHED < hi
    # the first live run landed just above the design's window; the constant
    # records that so the window is read as a hypothesis, not a gate
    assert hi < MEASURED_LIH_RHF_K222_EXPONENT < 0.2


# --------------------------------------------------------------------------
# The real LiH cell (needs a built vibe-qc)
# --------------------------------------------------------------------------


def test_lih_primitive_cell_geometry():
    vq = pytest.importorskip("vibeqc")
    from vibeqc.basis_optimization.recipes.one_compound import (
        ANGSTROM_TO_BOHR,
        LIH_A_ANGSTROM,
        lih_pob_tzvp_parametrisation,
        lih_primitive_cell,
    )

    system = lih_primitive_cell()
    atoms = list(system.unit_cell)
    assert sorted(int(a.Z) for a in atoms) == [1, 3]
    a = LIH_A_ANGSTROM * ANGSTROM_TO_BOHR
    lattice = np.asarray(system.lattice, dtype=float)
    assert abs(np.linalg.det(lattice)) == pytest.approx(a ** 3 / 4.0, rel=1e-12)
    assert hasattr(vq, "PeriodicSystem")
    par = lih_pob_tzvp_parametrisation()
    assert par.labels() == ["H_diffuse_s"]
    assert np.exp(par.pack()[0]) == pytest.approx(H_DIFFUSE_PUBLISHED)
    assert set(par.atoms) == {"Li", "H"}
