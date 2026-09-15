"""Tests for the uniform geometry optimisation framework (``vibeqc.geomopt``).

Coverage:
1. All four Phase-1 optimisers minimise water to the same energy.
2. Convergence policy composability (single-gate, multi-gate, presets).
3. Frozen atoms.
4. Trajectory collection.
5. Backward-compat keyword shim (optimizer_backend → geom_opt).
6. Error handling for invalid keywords.
"""

from __future__ import annotations

import io

import numpy as np
import pytest
from vibeqc._vibeqc_core import Atom, Molecule, PeriodicSystem, monkhorst_pack
from vibeqc.geomopt import (
    CartesianCoordinates,
    ConvergencePolicy,
    MolecularSCFProvider,
    bfgs,
    conjugate_gradient,
    lbfgs,
    run_geomopt,
    steepest_descent,
)
from vibeqc.geomopt.line_search import (
    _StrongWolfeLineSearch,
    resolve_line_search,
)
from vibeqc.geomopt.periodic_providers import (
    CellStrainCoordinates,
    FractionalCoordinates,
    PeriodicSCFProvider,
    run_periodic_geomopt,
)
from vibeqc.output import (
    OutputChannel,
    active_policy,
    set_active_policy,
)
from vibeqc.progress import ProgressLogger

# ---------------------------------------------------------------------------
# Shared fixtures
# ---------------------------------------------------------------------------


@pytest.fixture(scope="module")
def water() -> Molecule:
    """Water near equilibrium (bohr), RHF/STO-3G."""
    return Molecule(
        [
            Atom(8, [0.0000, 0.0000, 0.1173]),
            Atom(1, [0.0000, 1.4315, -0.9388]),
            Atom(1, [0.0000, -1.4315, -0.9388]),
        ],
        charge=0,
        multiplicity=1,
    )


@pytest.fixture(scope="module")
def provider() -> MolecularSCFProvider:
    return MolecularSCFProvider("sto-3g", method="rhf")


# ---------------------------------------------------------------------------
# 1. All four optimisers converge to the same energy
# ---------------------------------------------------------------------------


_WATER_REF_ENERGY = -74.96590  # Ha, RHF/STO-3G


@pytest.mark.parametrize(
    "opt_name,line_search,max_iter",
    [
        ("sd", "brent", 50),
        ("cg", "brent", 50),
        ("bfgs", "backtracking", 30),
        ("lbfgs", "backtracking", 30),
    ],
)
def test_optimizer_converges_water(
    water: Molecule,
    provider: MolecularSCFProvider,
    opt_name: str,
    line_search: str,
    max_iter: int,
) -> None:
    """Every Phase-1 optimiser converges water to the RHF/STO-3G minimum."""
    result = run_geomopt(
        water,
        provider,
        geom_opt=opt_name,
        geom_line_search=line_search,
        geom_max_iter=max_iter,
        geom_conv=ConvergencePolicy(gmax=1e-3),
        record_trajectory=False,
    )
    assert result.converged, f"{opt_name} did not converge"
    assert result.energy == pytest.approx(_WATER_REF_ENERGY, rel=1e-4)
    grad_max = float(np.max(np.abs(result.gradient))) if result.gradient.size else 0.0
    assert grad_max <= 1e-3


def test_optimizers_agree_on_water_energy(
    water: Molecule,
    provider: MolecularSCFProvider,
) -> None:
    """All four optimisers return the same energy within 1e-5 Ha."""
    energies = {}
    for opt_name in ("sd", "cg", "bfgs", "lbfgs"):
        ls = "brent" if opt_name in ("sd", "cg") else "backtracking"
        r = run_geomopt(
            water,
            provider,
            geom_opt=opt_name,
            geom_line_search=ls,
            geom_max_iter=50,
            geom_conv=ConvergencePolicy(gmax=1e-3),
            record_trajectory=False,
        )
        assert r.converged
        energies[opt_name] = r.energy
    ref = energies["bfgs"]
    for name, e in energies.items():
        assert e == pytest.approx(ref, abs=1e-5), f"{name} energy differs from BFGS"


# ---------------------------------------------------------------------------
# 2. Convergence policy
# ---------------------------------------------------------------------------


def test_convergence_policy_single_gate() -> None:
    """Default policy uses gmax only."""
    pol = ConvergencePolicy.default()
    assert pol.gmax == 4.5e-4
    assert pol.grms is None
    assert pol.dmax is None
    assert pol.drms is None
    assert pol.ediff is None

    # Converged
    g = np.array([1e-6, 2e-6, 3e-6])
    r = pol.check(gradient=g)
    assert bool(r)
    assert r.details["gmax"] is True
    assert r.thresholds == {"gmax": 4.5e-4}
    assert r.summary() == "gmax=3.000e-06 <= 4.500e-04 Ha/bohr [pass]"

    # Not converged
    g2 = np.array([1e-3, 2e-3, 3e-3])
    r2 = pol.check(gradient=g2)
    assert not bool(r2)
    assert r2.details["gmax"] is False
    assert r2.summary() == "gmax=3.000e-03 <= 4.500e-04 Ha/bohr [fail]"


def test_convergence_policy_multi_gate() -> None:
    """All five gates active."""
    pol = ConvergencePolicy(gmax=1e-4, grms=1e-4, dmax=1e-3, drms=1e-3, ediff=1e-6)
    g = np.array([1e-6, 2e-6, 3e-6])
    r = pol.check(
        gradient=g,
        x_old=np.zeros(3),
        x_new=np.array([1e-5, 2e-5, 3e-5]),
        e_old=-75.0,
        e_new=-75.0000001,
    )
    assert bool(r)
    for gate in ("gmax", "grms", "dmax", "drms", "ediff"):
        assert r.details[gate] is True, f"gate {gate} should pass"
        assert r.thresholds[gate] == getattr(pol, gate)
    summary = r.summary()
    assert "gmax=3.000e-06 <= 1.000e-04 Ha/bohr [pass]" in summary
    assert "dmax=0.000030 <= 0.001000 bohr [pass]" in summary
    assert "ediff=1.000e-07 <= 1.000e-06 Ha [pass]" in summary
    assert "✓" not in summary and "✗" not in summary


def test_convergence_presets() -> None:
    """Tight and loose presets."""
    tight = ConvergencePolicy.tight()
    assert tight.gmax == 1e-5
    assert tight.grms == 1e-5
    assert tight.dmax == 1e-4
    assert tight.drms == 1e-4
    assert tight.ediff == 1e-8

    loose = ConvergencePolicy.loose()
    assert loose.gmax == 1e-2
    assert loose.grms is None


# ---------------------------------------------------------------------------
# 3. Frozen atoms
# ---------------------------------------------------------------------------


def test_frozen_atoms_two_hydrogens(water: Molecule) -> None:
    """Freezing the two hydrogens at equilibrium → zero-step optimisation."""
    provider = MolecularSCFProvider("sto-3g", method="rhf")
    result = run_geomopt(
        water,
        provider,
        geom_opt="bfgs",
        geom_freeze=[1, 2],  # freeze both H atoms
        geom_max_iter=5,
        geom_conv=ConvergencePolicy(gmax=1e-3),
        record_trajectory=False,
    )
    # The starting geometry is already near equilibrium; freezing H atoms
    # means only the O can move, and it's already at its optimum.
    assert result.converged
    # Energy should be near the reference
    assert result.energy == pytest.approx(_WATER_REF_ENERGY, rel=1e-4)


# ---------------------------------------------------------------------------
# 4. Trajectory collection
# ---------------------------------------------------------------------------


def test_trajectory_collected(
    water: Molecule,
    provider: MolecularSCFProvider,
) -> None:
    """Trajectory frames are collected when record_trajectory=True."""
    result = run_geomopt(
        water,
        provider,
        geom_opt="bfgs",
        geom_max_iter=10,
        geom_conv=ConvergencePolicy(gmax=1e-3),
        record_trajectory=True,
    )
    assert result.converged
    assert len(result.trajectory_frames) >= 2  # start + at least one step
    assert len(result.trajectory_energies) == len(result.trajectory_frames)
    # First frame is the starting geometry
    first_e = result.trajectory_energies[0]
    last_e = result.trajectory_energies[-1]
    assert last_e < first_e  # energy decreased


def test_no_trajectory_when_disabled(
    water: Molecule,
    provider: MolecularSCFProvider,
) -> None:
    """record_trajectory=False gives empty lists."""
    result = run_geomopt(
        water,
        provider,
        geom_opt="bfgs",
        geom_max_iter=10,
        geom_conv=ConvergencePolicy(gmax=1e-3),
        record_trajectory=False,
    )
    assert result.converged
    assert result.trajectory_frames == []
    assert result.trajectory_energies == []


# ---------------------------------------------------------------------------
# 5. Backward-compat keyword shim
# ---------------------------------------------------------------------------


def test_optimizer_backend_brent_maps_to_sd(
    water: Molecule,
    provider: MolecularSCFProvider,
) -> None:
    """optimizer_backend='brent' is mapped to geom_opt='sd'."""
    result = run_geomopt(
        water,
        provider,
        optimizer_backend="brent",
        geom_max_iter=50,
        geom_conv=ConvergencePolicy(gmax=1e-3),
        record_trajectory=False,
    )
    assert result.converged
    assert result.optimizer == "sd"


def test_optimizer_backend_native_maps_to_lbfgs(
    water: Molecule,
    provider: MolecularSCFProvider,
) -> None:
    """optimizer_backend='native' is mapped to geom_opt='lbfgs'."""
    result = run_geomopt(
        water,
        provider,
        optimizer_backend="native",
        geom_max_iter=30,
        geom_conv=ConvergencePolicy(gmax=1e-3),
        record_trajectory=False,
    )
    assert result.converged
    assert result.optimizer == "lbfgs"


# ---------------------------------------------------------------------------
# 6. Error handling
# ---------------------------------------------------------------------------


def test_invalid_geom_opt_raises(
    water: Molecule, provider: MolecularSCFProvider
) -> None:
    with pytest.raises(ValueError, match="Unknown geom_opt"):
        run_geomopt(water, provider, geom_opt="nonexistent", geom_max_iter=1)


def test_unregistered_newton_geom_opt_raises(
    water: Molecule, provider: MolecularSCFProvider
) -> None:
    """SCF Newton finalizers are not geometry optimizers."""
    with pytest.raises(ValueError, match="Unknown geom_opt='newton'"):
        run_geomopt(water, provider, geom_opt="newton", geom_max_iter=1)


def test_invalid_geom_target(water: Molecule, provider: MolecularSCFProvider) -> None:
    with pytest.raises(ValueError, match="geom_target"):
        run_geomopt(water, provider, geom_target="saddle", geom_max_iter=1)


def test_ts_not_implemented(water: Molecule, provider: MolecularSCFProvider) -> None:
    """TS search is now enabled — should auto-select EF optimiser."""
    # TS search delegates to EF; confirm it runs without error
    result = run_geomopt(
        water,
        provider,
        geom_target="transition_state",
        geom_max_iter=5,
        geom_conv=ConvergencePolicy(gmax=1e-2),  # loose for quick test
        record_trajectory=False,
    )
    assert result.optimizer == "ef"


def test_invalid_coords_not_implemented(
    water: Molecule, provider: MolecularSCFProvider
) -> None:
    """Unsupported coordinate systems raise NotImplementedError."""
    with pytest.raises(NotImplementedError, match="geom_coords"):
        run_geomopt(water, provider, geom_coords="zmat", geom_max_iter=1)


# ---------------------------------------------------------------------------
# 7. Coordinate representation
# ---------------------------------------------------------------------------


def test_cartesian_coordinates() -> None:
    mol = Molecule(
        [Atom(1, [0.0, 0.0, 0.0]), Atom(1, [0.0, 0.0, 1.4])],
        charge=0,
        multiplicity=1,
    )
    coords = CartesianCoordinates(2)
    x0 = coords.x0(mol)
    assert len(x0) == 6
    assert x0[0] == 0.0
    assert x0[5] == 1.4

    # Round-trip
    mol2 = coords.to_cartesian(mol, x0 + 0.1)
    assert abs(mol2.atoms[0].xyz[0] - 0.1) < 1e-12
    assert abs(mol2.atoms[1].xyz[2] - 1.5) < 1e-12


def test_cartesian_frozen_gradient() -> None:
    mol = Molecule(
        [Atom(8, [0.0, 0.0, 0.0]), Atom(1, [0.0, 0.0, 1.0]), Atom(1, [0.0, 1.0, 0.0])],
        charge=0,
        multiplicity=1,
    )
    coords = CartesianCoordinates(3, freeze_indices=[0])
    x0 = coords.x0(mol)
    g = np.ones(9)
    g_masked = coords.apply_frozen(x0, g, coords.frozen_set)
    # Oxygen (first 3 coords) should be zero
    assert np.all(g_masked[:3] == 0.0)
    # Hydrogens should still be 1.0
    assert np.all(g_masked[3:] == 1.0)


# ---------------------------------------------------------------------------
# 8. Phase 3+ optimisers: trust-region, RFO
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("opt_name", ["trust", "rfo"])
def test_trust_region_optimizers_converge_water(
    water: Molecule,
    provider: MolecularSCFProvider,
    opt_name: str,
) -> None:
    """Trust-region and RFO converge water to the same minimum."""
    result = run_geomopt(
        water,
        provider,
        geom_opt=opt_name,
        geom_max_iter=50,
        geom_conv=ConvergencePolicy(gmax=1e-3),
        record_trajectory=False,
    )
    assert result.converged, f"{opt_name} did not converge"
    assert result.energy == pytest.approx(_WATER_REF_ENERGY, rel=1e-4)


@pytest.mark.parametrize("opt_name", ["ef", "prfo"])
def test_ef_prfo_converge_water_minimum(
    water: Molecule,
    provider: MolecularSCFProvider,
    opt_name: str,
) -> None:
    """EF and P-RFO converge water when targeting a minimum."""
    result = run_geomopt(
        water,
        provider,
        geom_opt=opt_name,
        geom_target="minimum",
        geom_max_iter=50,
        geom_conv=ConvergencePolicy(gmax=1e-3),
        record_trajectory=False,
    )
    assert result.converged, f"{opt_name} did not converge"
    assert result.optimizer == opt_name
    assert result.energy == pytest.approx(_WATER_REF_ENERGY, rel=1e-4)


@pytest.mark.parametrize("opt_name", ["ef", "prfo"])
def test_ef_prfo_ts_seed_handles_frozen_atoms(
    water: Molecule,
    provider: MolecularSCFProvider,
    opt_name: str,
) -> None:
    """TS Hessian seeding uses the gradient, not the scalar energy."""
    result = run_geomopt(
        water,
        provider,
        geom_opt=opt_name,
        geom_target="transition_state",
        geom_freeze=[0],
        geom_max_iter=0,
        geom_conv=ConvergencePolicy(gmax=1e-10),
        record_trajectory=False,
    )
    assert result.optimizer == opt_name
    assert result.n_energy_evals == 1
    assert result.gradient.shape == (9,)


# ---------------------------------------------------------------------------
# 9. Hessian update formulas
# ---------------------------------------------------------------------------


def test_hessian_updates_preserve_symmetry() -> None:
    """All Hessian updates produce symmetric matrices."""
    from vibeqc.geomopt.hessian_update import (
        bfgs_hessian_update,
        bofill_hessian_update,
        murtagh_sargent_hessian_update,
        powell_hessian_update,
        resolve_hessian_update,
        sr1_hessian_update,
    )

    n = 6
    H = np.eye(n)
    s = np.random.randn(n)
    y = np.random.randn(n) * 0.1 + s  # ensure sTy > 0

    for name in ("bfgs", "sr1", "powell", "bofill", "murtagh_sargent"):
        updater = resolve_hessian_update(name)
        H_new = updater(H, s, y)
        assert H_new.shape == (n, n)
        assert np.allclose(H_new, H_new.T), f"{name} update broke symmetry"


# ---------------------------------------------------------------------------
# 10. Registry covers all expected optimizers
# ---------------------------------------------------------------------------


def test_registry_has_all_optimizers() -> None:
    from vibeqc.geomopt.registry import _OPTIMIZERS

    expected = {
        "sd",
        "cg",
        "bfgs",
        "lbfgs",
        "trust",
        "rfo",
        "ef",
        "prfo",
        "dimer",
        "gdiis",
        "fire",
    }
    missing = expected - set(_OPTIMIZERS)
    assert not missing, f"Missing optimizers: {missing}"


# ---------------------------------------------------------------------------
# 11. GDIIS and FIRE optimisers
# ---------------------------------------------------------------------------


def test_gdiis_converges_water(
    water: Molecule,
    provider: MolecularSCFProvider,
) -> None:
    """GDIIS converges water to the minimum."""
    result = run_geomopt(
        water,
        provider,
        geom_opt="gdiis",
        geom_max_iter=50,
        geom_conv=ConvergencePolicy(gmax=1e-3),
        record_trajectory=False,
    )
    assert result.converged
    assert result.optimizer == "gdiis"
    assert result.energy == pytest.approx(_WATER_REF_ENERGY, rel=1e-4)


def test_fire_converges_water_downhill(
    water: Molecule,
    provider: MolecularSCFProvider,
) -> None:
    """FIRE follows forces (-gradE) and makes downhill progress."""
    result = run_geomopt(
        water,
        provider,
        geom_opt="fire",
        geom_max_iter=80,
        geom_conv=ConvergencePolicy(gmax=1e-3),
        record_trajectory=True,
    )
    assert result.optimizer == "fire"
    assert result.converged
    assert result.trajectory_energies[-1] < result.trajectory_energies[0]
    grad_max = float(np.max(np.abs(result.gradient))) if result.gradient.size else 0.0
    assert grad_max <= 1e-3


# ---------------------------------------------------------------------------
# 11b. Dimer method (TS search)
# ---------------------------------------------------------------------------


def test_dimer_registered() -> None:
    """The dimer optimiser is registered and discoverable."""
    from vibeqc.geomopt.registry import _OPTIMIZERS

    assert "dimer" in _OPTIMIZERS, "dimer optimiser not registered"
    assert callable(_OPTIMIZERS["dimer"])


def test_dimer_requires_transition_state_target(
    water: Molecule,
    provider: MolecularSCFProvider,
) -> None:
    """Dimer is a saddle-search method and should not masquerade as a minimum opt."""
    with pytest.raises(ValueError, match="transition-state"):
        run_geomopt(
            water,
            provider,
            geom_opt="dimer",
            geom_target="minimum",
            geom_max_iter=1,
            record_trajectory=False,
        )


def test_dimer_wrapper_runs(
    water: Molecule,
    provider: MolecularSCFProvider,
) -> None:
    """Dimer wrapper runs on water without crashing (3-iter smoke test)."""
    # Use a geometry displaced toward a symmetric linear TS guess.
    import numpy as np

    linear_h2o = Molecule(
        [
            Atom(8, [0.0, 0.0, 0.0]),
            Atom(1, [0.0, 0.0, -2.3]),  # ~1.217 Å
            Atom(1, [0.0, 0.0, 2.3]),
        ],
        charge=0,
        multiplicity=1,
    )

    # Initial direction: antisymmetric stretch (points toward the TS)
    direction = np.zeros((3, 3), dtype=float)
    direction[1, 2] = 1.0
    direction[2, 2] = -1.0

    result = run_geomopt(
        linear_h2o,
        provider,
        geom_opt="dimer",
        geom_target="transition_state",
        geom_max_iter=3,
        geom_conv=ConvergencePolicy(gmax=1e-1),
        geom_opt_options={"initial_direction": direction},
        record_trajectory=False,
    )
    assert result.optimizer == "dimer"
    assert result.n_iter > 0
    assert result.n_energy_evals > 0


def test_dimer_default_direction_is_negative_gradient(
    water: Molecule,
    provider: MolecularSCFProvider,
) -> None:
    """When no initial_direction is given, the wrapper uses −∇E."""
    result = run_geomopt(
        water,
        provider,
        geom_opt="dimer",
        geom_target="transition_state",
        geom_max_iter=1,
        geom_conv=ConvergencePolicy(gmax=1e-1),
        record_trajectory=False,
    )
    assert result.optimizer == "dimer"
    assert result.n_iter == 1


# ---------------------------------------------------------------------------
# 12. DLC coordinates
# ---------------------------------------------------------------------------


def test_dlc_converges_water(
    water: Molecule,
    provider: MolecularSCFProvider,
) -> None:
    """DLC + BFGS converges water to the same minimum."""
    result = run_geomopt(
        water,
        provider,
        geom_opt="bfgs",
        geom_coords="dlc",
        geom_max_iter=30,
        geom_conv=ConvergencePolicy(gmax=1e-3),
        record_trajectory=False,
    )
    assert result.converged
    assert result.energy == pytest.approx(_WATER_REF_ENERGY, rel=1e-4)


# ---------------------------------------------------------------------------
# 13. Restart / checkpoint
# ---------------------------------------------------------------------------


def test_checkpoint_save_and_load(
    water: Molecule,
    provider: MolecularSCFProvider,
    tmp_path,
) -> None:
    """A checkpoint is written after optimisation and can be loaded back."""
    import json

    from vibeqc.geomopt.history import RestartSerializer

    ckpt = tmp_path / "opt_checkpoint.json"
    result = run_geomopt(
        water,
        provider,
        geom_opt="bfgs",
        geom_max_iter=10,
        geom_conv=ConvergencePolicy(gmax=1e-3),
        geom_checkpoint=str(ckpt),
        record_trajectory=False,
    )
    assert result.converged
    assert ckpt.exists()

    serializer = RestartSerializer(str(ckpt))
    state = serializer.load()
    assert state["converged"] is True
    assert "energy" in state
    assert "optimizer" in state


def test_history_manager_records_steps(
    water: Molecule, provider: MolecularSCFProvider
) -> None:
    """HistoryManager records per-step data."""
    from vibeqc.geomopt.history import HistoryManager

    mgr = HistoryManager()
    x = np.zeros(9)
    for i in range(5):
        mgr.record(i, -75.0 - 0.01 * i, np.ones(9) * (0.1 / (i + 1)), x)

    assert len(mgr.records) == 5
    assert mgr.best is not None
    assert mgr.best.energy == pytest.approx(-75.04)
    assert mgr.last().iteration == 4


def test_history_serialise_roundtrip() -> None:
    """HistoryManager survives serialisation round-trip."""
    from vibeqc.geomopt.history import HistoryManager

    mgr = HistoryManager()
    x = np.array([1.0, 2.0, 3.0])
    mgr.record(0, -75.0, np.array([0.1, 0.2, 0.3]), x)
    mgr.record(1, -75.5, np.array([0.01, 0.02, 0.03]), x + 0.1)

    state = mgr.to_state()
    mgr2 = HistoryManager.from_state(state)
    assert len(mgr2.records) == 2
    assert mgr2.records[0].energy == pytest.approx(-75.0)
    assert np.allclose(mgr2.records[1].x_flat, x + 0.1)


def test_restart_state_build_and_restore() -> None:
    """build_restart_state / restore_optimizer_state round-trip."""
    from vibeqc.geomopt.history import build_restart_state, restore_optimizer_state
    from vibeqc.geomopt.state import OptimizerState

    st = OptimizerState(
        x=np.array([1.0, 2.0, 3.0]),
        energy=-75.0,
        gradient=np.array([0.01, 0.02, 0.03]),
        iteration=5,
        n_energy_evals=20,
    )
    pol = ConvergencePolicy(gmax=1e-3)
    restart = build_restart_state(
        st, optimizer="bfgs", method="rhf", convergence_policy=pol
    )
    st2, extra = restore_optimizer_state(restart)
    assert st2.iteration == 5
    assert st2.energy == pytest.approx(-75.0)
    assert np.allclose(st2.x, [1.0, 2.0, 3.0])
    assert restart["convergence"]["gmax"] == 1e-3


# ---------------------------------------------------------------------------
# 7. Line search strategies
# ---------------------------------------------------------------------------


def test_wolfe_line_search_finds_minimum() -> None:
    """Strong Wolfe line search correctly minimises a 2D quadratic."""

    # 2D quadratic bowl with minimum at (1, -2):
    #   f(x,y) = (x - 1)^2 + 2 * (y + 2)^2 + 3
    #   grad f = [2(x-1), 4(y+2)]
    #   minimum at (1, -2) with f_min = 3
    def f_quadratic(x: np.ndarray) -> tuple[float, np.ndarray]:
        x0, x1 = x[0], x[1]
        dx0 = x0 - 1.0
        dx1 = x1 + 2.0
        e = dx0 * dx0 + 2.0 * dx1 * dx1 + 3.0
        g = np.array([2.0 * dx0, 4.0 * dx1])
        return float(e), g

    x0 = np.array([5.0, 3.0])  # start far from minimum
    f0, g0 = f_quadratic(x0)

    # Steepest-descent direction (normalised).
    direction = -g0 / np.linalg.norm(g0)

    # Directional derivative at x0.
    dphi0 = float(np.dot(g0, direction))
    assert dphi0 < 0, "direction must be descent"

    # Create line search via factory; use N&W defaults.
    ls = _StrongWolfeLineSearch(c1=1e-4, c2=0.9, max_iter=20)
    x_new, f_new, n_eval = ls(x0, f0, direction, f_quadratic, lambda m: m)

    # Evaluate gradient at the new point.
    _, g_new = f_quadratic(x_new)
    dphi_new = float(np.dot(g_new, direction))

    # 1. Energy must decrease.
    assert f_new < f0, f"f_new={f_new} >= f0={f0}"

    # 2. Sufficient decrease (Armijo): φ(α) ≤ φ(0) + c₁·α·φ'(0)
    alpha = float(np.linalg.norm(x_new - x0) / np.linalg.norm(direction))
    armijo_rhs = f0 + 1e-4 * alpha * dphi0
    assert f_new <= armijo_rhs + 1e-12, f"Armijo: f_new={f_new} > rhs={armijo_rhs}"

    # 3. Curvature (strong Wolfe): |φ'(α)| ≤ c₂·|φ'(0)|
    assert abs(dphi_new) <= 0.9 * abs(dphi0) + 1e-12, (
        f"Curvature: |dphi_new|={abs(dphi_new)} > 0.9*|dphi0|={0.9 * abs(dphi0)}"
    )

    # 4. The line search found a point closer to the minimum
    #    than x0 (i.e., gradient magnitude reduced).
    assert np.linalg.norm(g_new) < np.linalg.norm(g0), "gradient norm should decrease"

    # 5. Some function evaluations were used.
    assert n_eval >= 1


def test_wolfe_factory_registration() -> None:
    """resolve_line_search('wolfe') returns a _StrongWolfeLineSearch."""
    ls1 = resolve_line_search("wolfe")
    assert isinstance(ls1, _StrongWolfeLineSearch)

    ls2 = resolve_line_search("strong_wolfe")
    assert isinstance(ls2, _StrongWolfeLineSearch)


def test_wolfe_rejects_bad_c_params() -> None:
    """_StrongWolfeLineSearch validates c1 < c2."""
    with pytest.raises(ValueError, match="0 < c1 < c2 < 1"):
        _StrongWolfeLineSearch(c1=0.5, c2=0.3)


def test_wolfe_non_descent_direction_returns_x0() -> None:
    """Non-descent direction triggers early exit without error."""

    def f_quadratic(x: np.ndarray) -> tuple[float, np.ndarray]:
        dx0 = x[0] - 1.0
        dx1 = x[1] + 2.0
        e = dx0 * dx0 + 2.0 * dx1 * dx1 + 3.0
        g = np.array([2.0 * dx0, 4.0 * dx1])
        return float(e), g

    x0 = np.array([5.0, 3.0])
    f0, _ = f_quadratic(x0)

    # Use an ascent direction.
    direction = np.array([1.0, 0.0])

    ls = _StrongWolfeLineSearch()
    x_new, f_new, n_eval = ls(x0, f0, direction, f_quadratic, lambda m: m)

    assert np.allclose(x_new, x0)
    assert f_new == f0
    assert n_eval == 1


def test_wolfe_quadratic_single_step() -> None:
    """For a 1-D quadratic f(x)=x^2, Wolfe line search satisfies both
    conditions with a step that reduces the function value."""

    # f(x) = x^2 (1-D); minimum at 0.
    def f_quad(x: np.ndarray) -> tuple[float, np.ndarray]:
        e = float(x[0] * x[0])
        g = np.array([2.0 * x[0]])
        return e, g

    x0 = np.array([3.0])
    f0, g0 = f_quad(x0)
    direction = -g0 / np.linalg.norm(g0)
    dphi0 = float(np.dot(g0, direction))
    assert dphi0 < 0

    ls = _StrongWolfeLineSearch(c1=1e-4, c2=0.9, max_iter=20)
    x_new, f_new, n_eval = ls(x0, f0, direction, f_quad, lambda m: m)

    _, g_new = f_quad(x_new)
    dphi_new = float(np.dot(g_new, direction))
    alpha = float(np.linalg.norm(x_new - x0) / np.linalg.norm(direction))

    # 1. Energy decreased (x moved toward minimum).
    assert f_new < f0, f"f_new={f_new} >= f0={f0}"
    assert alpha > 0, "must take a positive step"

    # 2. Armijo satisfied.
    assert f_new <= f0 + 1e-4 * alpha * dphi0 + 1e-12

    # 3. Curvature satisfied.
    assert abs(dphi_new) <= 0.9 * abs(dphi0) + 1e-12

    # 4. The trial step alpha=1 already satisfies the Wolfe conditions
    #    for this quadratic (alpha=1 gives x=2, f=4, |dphi|=4); verify
    #    it landed at or better than alpha=1 (x <= 2).  The algorithm
    #    doubles alpha when conditions are met, so it may overshoot
    #    the first-acceptable point slightly — but still reduces energy.
    assert x_new[0] <= 2.0 + 1e-10, f"x_new[0]={x_new[0]} should be <= starting x0=3.0"


def test_progress_stream_reports_initial_evaluation() -> None:
    """Progress output is flushed before the first expensive provider call."""
    mol = Molecule([Atom(2, [2.0, 0.0, 0.0])], charge=0, multiplicity=1)

    def harmonic(molecule: Molecule) -> tuple[float, np.ndarray]:
        x = np.array([c for atom in molecule.atoms for c in atom.xyz], dtype=float)
        return float(np.dot(x, x)), 2.0 * x.reshape((1, 3))

    progress = io.StringIO()
    logger = ProgressLogger(stream=progress)
    run_geomopt(
        mol,
        harmonic,
        geom_opt="bfgs",
        geom_max_iter=1,
        geom_conv=ConvergencePolicy(gmax=1e-12),
        record_trajectory=False,
        progress=logger,
    )

    text = progress.getvalue()
    assert "evaluating initial energy and gradient" in text
    assert "dE" in text
    assert "|step|" in text
    assert "max|g|" in text


def test_progress_keeps_persistent_and_live_sinks_separate() -> None:
    """Geomopt writes the same trace without receiving the .out stream."""
    mol = Molecule([Atom(2, [2.0, 0.0, 0.0])], charge=0, multiplicity=1)

    def harmonic(molecule: Molecule) -> tuple[float, np.ndarray]:
        x = np.array([c for atom in molecule.atoms for c in atom.xyz], dtype=float)
        return float(np.dot(x, x)), 2.0 * x.reshape((1, 3))

    persistent = io.StringIO()
    live = io.StringIO()
    with OutputChannel.to_stream(persistent):
        run_geomopt(
            mol,
            harmonic,
            geom_opt="bfgs",
            geom_max_iter=1,
            geom_conv=ConvergencePolicy(gmax=1e-12),
            record_trajectory=False,
            progress=ProgressLogger(stream=live),
        )

    assert persistent.getvalue() == live.getvalue()
    assert (
        persistent.getvalue().count("evaluating initial energy and gradient") == 1
    )


def test_progress_converts_trace_and_criteria_with_active_policy() -> None:
    mol = Molecule([Atom(2, [2.0, 0.0, 0.0])], charge=0, multiplicity=1)

    def harmonic(molecule: Molecule) -> tuple[float, np.ndarray]:
        x = np.array([c for atom in molecule.atoms for c in atom.xyz], dtype=float)
        return float(np.dot(x, x)), 2.0 * x.reshape((1, 3))

    saved = active_policy()
    converted = (
        saved.with_unit("energy", "eV")
        .with_unit("gradient", "eV/Angstrom")
        .with_unit("length", "Angstrom")
    )
    live = io.StringIO()
    try:
        set_active_policy(converted)
        run_geomopt(
            mol,
            harmonic,
            geom_opt="bfgs",
            geom_max_iter=1,
            geom_conv=ConvergencePolicy(gmax=1e-12),
            record_trajectory=False,
            progress=ProgressLogger(stream=live),
        )
    finally:
        set_active_policy(saved)

    text = live.getvalue()
    assert "E (eV)" in text
    assert "gmax=5.1e-11 eV/Angstrom" in text
    assert "108.84554498" in text
    assert "eV/Angstrom [fail]" in text
    assert "Ha/bohr" not in text


# ---------------------------------------------------------------------------
# 14. Restart from checkpoint
# ---------------------------------------------------------------------------


def test_restart_from_checkpoint(
    water: Molecule,
    provider: MolecularSCFProvider,
    tmp_path,
) -> None:
    """Run an optimisation for 5 steps, checkpoint, then restart and verify
    the restart reaches the same minimum."""
    ckpt = tmp_path / "restart_test.json"

    result1 = run_geomopt(
        water,
        provider,
        geom_opt="bfgs",
        geom_max_iter=5,
        geom_conv=ConvergencePolicy(gmax=1e-3),
        geom_checkpoint=str(ckpt),
        record_trajectory=False,
    )
    assert ckpt.exists()
    assert result1.n_iter == 5

    result2 = run_geomopt(
        water,
        provider,
        geom_opt="bfgs",
        geom_max_iter=25,
        geom_conv=ConvergencePolicy(gmax=1e-3),
        geom_restart=str(ckpt),
        record_trajectory=False,
    )
    assert result2.n_iter >= 5, (
        f"Restart n_iter={result2.n_iter} should count from the checkpoint"
    )
    assert result2.converged
    assert result2.energy == pytest.approx(_WATER_REF_ENERGY, rel=1e-4)


# ---------------------------------------------------------------------------
# 15. Semiempirical provider
# ---------------------------------------------------------------------------


def test_semiempirical_provider_imports() -> None:
    """SemiempiricalProvider can be imported and constructed."""
    from vibeqc.geomopt.semiempirical_provider import SemiempiricalProvider, _build_model
    from vibeqc.semiempirical import SCCDFTBModel

    p = SemiempiricalProvider("dftb0")
    assert p._method_key == "dftb0"
    assert SemiempiricalProvider(" dftb ")._method_key == "dftb0"
    assert SemiempiricalProvider("dftb-0")._method_key == "dftb0"
    assert SemiempiricalProvider("scc_dftb")._method_key == "scc_dftb"
    assert SemiempiricalProvider("sccdftb")._method_key == "scc_dftb"
    assert SemiempiricalProvider(" GFN2_XTB ")._method_key == "gfn2_xtb"
    assert SemiempiricalProvider("gfn2xtb")._method_key == "gfn2_xtb"
    assert isinstance(
        _build_model(
            "scc-dftb",
            Molecule([
                Atom(1, [0.0, 0.0, 0.0]),
                Atom(1, [1.4, 0.0, 0.0]),
            ]),
        ),
        SCCDFTBModel,
    )


def test_semiempirical_provider_invalid_method() -> None:
    """SemiempiricalProvider raises on unknown method."""
    from vibeqc.geomopt.semiempirical_provider import SemiempiricalProvider

    with pytest.raises(ValueError, match="Unknown semiempirical method"):
        SemiempiricalProvider("bunk")


def test_semiempirical_provider_builds_molecular_route_plan(monkeypatch) -> None:
    """SemiempiricalProvider validates aliases through the shared route plan."""
    from vibeqc.geomopt import semiempirical_provider as provider_module
    from vibeqc.geomopt.semiempirical_provider import SemiempiricalProvider

    calls = []
    real_from_request = provider_module.SemiempiricalRoutePlan.from_request

    def tracked_from_request(method, **kwargs):
        calls.append((method, kwargs))
        return real_from_request(method, **kwargs)

    monkeypatch.setattr(
        provider_module.SemiempiricalRoutePlan,
        "from_request",
        staticmethod(tracked_from_request),
    )

    provider = SemiempiricalProvider("scc-dftb", multiplicity=2)

    assert provider._method_key == "scc_dftb"
    assert calls == [
        (
            "scc-dftb",
            {"boundary": "molecule", "charge": 0, "multiplicity": 2},
        ),
    ]


def test_semiempirical_provider_charge_multiplicity_override(monkeypatch) -> None:
    """SemiempiricalProvider constructor charge/spin knobs are real overrides."""
    from vibeqc.geomopt import semiempirical_provider as provider_module
    from vibeqc.geomopt.semiempirical_provider import SemiempiricalProvider

    seen = []

    class FakeResult:
        energy = -1.0

        def __init__(self, molecule):
            self.molecule = molecule

        def gradient(self):
            return np.zeros((len(list(self.molecule.atoms)), 3))

    def fake_run_semiempirical(method_key, molecule):
        seen.append((method_key, molecule.charge, molecule.multiplicity))
        return FakeResult(molecule)

    monkeypatch.setattr(
        provider_module,
        "run_semiempirical",
        fake_run_semiempirical,
    )
    mol = Molecule([Atom(2, [0.0, 0.0, 0.0])], charge=0, multiplicity=1)

    SemiempiricalProvider("pm6", charge=1, multiplicity=2)(mol)
    SemiempiricalProvider("pm6")(Molecule(
        [Atom(1, [0.0, 0.0, 0.0])],
        charge=0,
        multiplicity=2,
    ))
    SemiempiricalProvider("scc-dftb")(mol)

    assert seen == [("pm6", 1, 2), ("pm6", 0, 2), ("scc_dftb", 0, 1)]


def test_semiempirical_provider_uses_unified_runner_gradient(monkeypatch) -> None:
    """SemiempiricalProvider delegates energy/gradient to run_semiempirical."""
    from vibeqc.geomopt import semiempirical_provider as provider_module
    from vibeqc.geomopt.semiempirical_provider import SemiempiricalProvider

    calls = []

    class FakeResult:
        energy = -2.5

        def gradient(self):
            calls.append("gradient")
            return np.array([[1.0, 2.0, 3.0]])

    def fake_run_semiempirical(method_key, molecule):
        calls.append(("run", method_key, molecule.charge, molecule.multiplicity))
        return FakeResult()

    monkeypatch.setattr(
        provider_module,
        "run_semiempirical",
        fake_run_semiempirical,
    )

    mol = Molecule([Atom(2, [0.0, 0.0, 0.0])], charge=0, multiplicity=1)
    energy, gradient = SemiempiricalProvider("gfn2xtb")(mol)

    assert energy == pytest.approx(-2.5)
    np.testing.assert_allclose(gradient, [1.0, 2.0, 3.0])
    assert calls == [("run", "gfn2_xtb", 0, 1), "gradient"]


def test_semiempirical_provider_rejects_energy_only_runner_result(monkeypatch) -> None:
    """Geometry providers fail closed when a semiempirical route has no forces."""
    from vibeqc.geomopt import semiempirical_provider as provider_module
    from vibeqc.geomopt.semiempirical_provider import SemiempiricalProvider

    class FakeResult:
        energy = -1.0

        def gradient(self):
            return None

    monkeypatch.setattr(
        provider_module,
        "run_semiempirical",
        lambda method_key, molecule: FakeResult(),
    )

    mol = Molecule([Atom(2, [0.0, 0.0, 0.0])], charge=0, multiplicity=1)
    with pytest.raises(NotImplementedError, match="requires an energy gradient"):
        SemiempiricalProvider("pm6")(mol)


def test_semiempirical_provider_keeps_msindo_fd_gradient_fallback(
    monkeypatch,
) -> None:
    """MSINDO provider keeps the FD force path when unified grad is absent."""
    from vibeqc.geomopt import semiempirical_provider as provider_module
    from vibeqc.geomopt.semiempirical_provider import SemiempiricalProvider

    calls = []

    class FakeResult:
        energy = -4.0

        def gradient(self):
            calls.append("unified-gradient")
            return None

    monkeypatch.setattr(
        provider_module,
        "run_semiempirical",
        lambda method_key, molecule: FakeResult(),
    )
    monkeypatch.setattr(
        provider_module,
        "_msindo_fd_gradient",
        lambda molecule: np.array([[0.4, 0.5, 0.6]]),
    )

    mol = Molecule([Atom(2, [0.0, 0.0, 0.0])], charge=0, multiplicity=1)
    energy, gradient = SemiempiricalProvider("msindo")(mol)

    assert energy == pytest.approx(-4.0)
    np.testing.assert_allclose(gradient, [0.4, 0.5, 0.6])
    assert calls == ["unified-gradient"]


# ---------------------------------------------------------------------------
# 16. Dimer L-BFGS acceleration
# ---------------------------------------------------------------------------


def test_dimer_lbfgs_accelerated(
    provider: MolecularSCFProvider,
) -> None:
    """Dimer with lbfgs_acceleration=True runs without error."""
    import numpy as np

    linear_h2o = Molecule(
        [
            Atom(8, [0.0, 0.0, 0.0]),
            Atom(1, [0.0, 0.0, -2.3]),
            Atom(1, [0.0, 0.0, 2.3]),
        ],
        charge=0,
        multiplicity=1,
    )
    direction = np.zeros((3, 3), dtype=float)
    direction[1, 2] = 1.0
    direction[2, 2] = -1.0

    result = run_geomopt(
        linear_h2o,
        provider,
        geom_opt="dimer",
        geom_target="transition_state",
        geom_max_iter=3,
        geom_conv=ConvergencePolicy(gmax=1e-1),
        geom_opt_options={
            "initial_direction": direction,
            "lbfgs_acceleration": True,
            "lbfgs_memory": 5,
        },
        record_trajectory=False,
    )
    assert result.optimizer == "dimer"
    assert result.n_iter > 0
    assert result.n_energy_evals > 0


# ---------------------------------------------------------------------------
# 17. Periodic geometry optimisation
# ---------------------------------------------------------------------------


def test_periodic_provider_imports() -> None:
    """PeriodicSCFProvider can be imported and constructed with a periodic system."""
    lattice = np.array([[3.0, 0.0, 0.0], [0.0, 10.0, 0.0], [0.0, 0.0, 10.0]])
    atoms = [Atom(1, [0.0, 0.0, 0.0]), Atom(1, [0.7, 0.0, 0.0])]
    system = PeriodicSystem(1, lattice, atoms, charge=0, multiplicity=1)
    kmesh = monkhorst_pack(system, [2, 1, 1], [0, 0, 0], False)

    provider = PeriodicSCFProvider("sto-3g", kmesh, method="RHF")
    assert provider.method == "RHF"
    assert provider.basis_name == "sto-3g"


def _low_dimensional_bipole_message(dim):
    # Public periodic_runner BIPOLE contract, including supported remedies.
    return (
        "jk_method='bipole' requires a 3-D periodic system (dim=3); got "
        f"dim={dim}. The exact Ewald-J route uses a 3-D "
        "Ewald/Madelung lattice sum that is not a low-dimensional Coulomb "
        "model. For a 1-D wire use jk_method='auto' or 'gdf'; for a 2-D "
        "surface use jk_method='auto'/'slab_ewald_2d', or explicit GDF "
        "for its supported closed-shell slab envelope."
    )


@pytest.mark.parametrize("force_mode", ["fd", "analytic"])
@pytest.mark.parametrize("dim", [1, 2])
@pytest.mark.parametrize("method", ["RHF", "UHF", "RKS", "UKS"])
def test_periodic_provider_refuses_low_dimensional_bipole_before_scf(
    dim, method, force_mode, monkeypatch
) -> None:
    """The geomopt provider must mirror the public BIPOLE dim=3 gate.

    The direct drivers retain a low-dimensional ``DIRECT_TRUNCATED``
    diagnostic, but that cutoff-dependent energy is not a 1-D/2-D Coulomb
    model and must never become an optimisation objective (IID 542).
    """
    import vibeqc.geomopt.periodic_providers as pp

    lattice = np.diag([8.0, 8.0, 20.0])
    atoms = [Atom(1, [4.0, 4.0, 9.3]), Atom(1, [4.0, 4.0, 10.7])]
    system = PeriodicSystem(dim, lattice, atoms, charge=0, multiplicity=1)
    kmesh = monkhorst_pack(system, [1, 1, 1], [0, 0, 0], False)
    provider = PeriodicSCFProvider(
        "sto-3g",
        kmesh,
        method=method,
        force_mode=force_mode,
        functional="lda" if method.endswith("KS") else None,
    )

    calls = []

    def must_not_run(*args, **kwargs):
        calls.append((args, kwargs))
        raise AssertionError("low-dimensional provider reached a BIPOLE driver")

    for name in (
        "run_pbc_bipole_rhf",
        "run_pbc_bipole_uhf",
        "run_pbc_bipole_rks",
        "run_pbc_bipole_uks",
    ):
        monkeypatch.setattr(pp, name, must_not_run)

    with pytest.raises(NotImplementedError) as exc:
        provider(system)
    assert str(exc.value) == _low_dimensional_bipole_message(dim)
    assert calls == []
    assert provider._reference_system is None
    assert provider._had_good_eval is False


@pytest.mark.parametrize("dim", [1, 2])
def test_run_periodic_geomopt_propagates_low_dimensional_bipole_refusal(
    dim, monkeypatch,
) -> None:
    """The high-level runner cannot bypass the provider's dimension gate."""
    import vibeqc.geomopt.periodic_providers as pp

    lattice = np.diag([3.0, 10.0, 10.0])
    atoms = [Atom(1, [0.0, 0.0, 0.0]), Atom(1, [0.8, 0.0, 0.0])]
    system = PeriodicSystem(dim, lattice, atoms, charge=0, multiplicity=1)
    kmesh = monkhorst_pack(system, [1, 1, 1], [0, 0, 0], False)
    provider = PeriodicSCFProvider("sto-3g", kmesh, method="RHF")

    def must_not_run(*args, **kwargs):
        raise AssertionError("low-dimensional geomopt reached a BIPOLE driver")

    monkeypatch.setattr(pp, "run_pbc_bipole_rhf", must_not_run)
    with pytest.raises(NotImplementedError) as exc:
        run_periodic_geomopt(
            system,
            provider,
            geom_max_iter_atoms=1,
            geom_max_outer=1,
            relax_cell=False,
            record_trajectory=False,
            progress=False,
        )
    assert str(exc.value) == _low_dimensional_bipole_message(dim)
    assert provider._reference_system is None
    assert provider._had_good_eval is False


def test_fractional_coordinates_roundtrip() -> None:
    """FractionalCoordinates converts a PeriodicSystem to fractional coords and back."""
    lattice = np.array([[3.0, 0.0, 0.0], [0.0, 10.0, 0.0], [0.0, 0.0, 10.0]])
    atoms = [Atom(1, [0.0, 0.0, 0.0]), Atom(1, [0.7, 0.0, 0.0])]
    system = PeriodicSystem(1, lattice, atoms, charge=0, multiplicity=1)

    coords = FractionalCoordinates(n_atoms=2, lattice=lattice)
    assert coords.n_params == 6

    x0 = coords.x0(system)
    assert x0.shape == (6,)
    assert x0[0] == pytest.approx(0.0)
    assert x0[3] == pytest.approx(0.7 / 3.0)

    # Round-trip: x0 --> to_cartesian --> x0 should be the same
    restored = coords.to_cartesian(system, x0)
    x0_restored = coords.x0(restored)
    np.testing.assert_allclose(x0_restored, x0, atol=1e-12)


def test_cell_strain_coordinates() -> None:
    """CellStrainCoordinates x0 is zeros, to_cartesian produces different lattice."""
    lattice = np.array([[3.0, 0.0, 0.0], [0.0, 10.0, 0.0], [0.0, 0.0, 10.0]])
    atoms = [Atom(1, [0.0, 0.0, 0.0]), Atom(1, [0.7, 0.0, 0.0])]
    system = PeriodicSystem(1, lattice, atoms, charge=0, multiplicity=1)

    coords = CellStrainCoordinates(reference_lattice=lattice)
    assert coords.n_params == 6

    x0 = coords.x0(system)
    np.testing.assert_array_equal(x0, np.zeros(6))

    # Zero strain recovers the reference lattice exactly
    sys_zero = coords.to_cartesian(system, np.zeros(6))
    np.testing.assert_allclose(np.asarray(sys_zero.lattice), lattice, atol=1e-12)

    # Non-zero strain produces a different lattice
    strain = np.array([0.05, 0.0, 0.0, 0.0, 0.0, 0.0])  # 0.05% in xx
    sys_strained = coords.to_cartesian(system, strain)
    lattice_strained = np.asarray(sys_strained.lattice)
    assert not np.allclose(lattice_strained, lattice, atol=1e-12)
    # The xx component should be stretched by (1 + 5e-4)
    assert lattice_strained[0, 0] == pytest.approx(3.0 * 1.0005)


def _compact_h2_3d_fixture(*, a_bohr=6.0, bond_bohr=1.4):
    """Compact cubic 3-D H₂ workhorse for the BIPOLE geomopt contract."""
    lattice = np.diag([a_bohr, a_bohr, a_bohr])
    centre = a_bohr / 2.0
    atoms = [
        Atom(1, [centre, centre, centre - bond_bohr / 2.0]),
        Atom(1, [centre, centre, centre + bond_bohr / 2.0]),
    ]
    system = PeriodicSystem(3, lattice, atoms, charge=0, multiplicity=1)
    kmesh = monkhorst_pack(system, [1, 1, 1], [0, 0, 0], False)
    return system, kmesh


def test_periodic_provider_default_inherits_the_driver_lattice_cutoffs() -> None:
    """With no ``cutoff_bohr`` the provider runs the SAME lattice sums as the
    BIPOLE drivers it wraps (IID 540).

    Asserted against a fresh ``LatticeSumOptions()`` rather than literals, so
    this pins "same as the drivers", not a number: if the driver default ever
    moves, the provider must move with it. Before the fix the provider
    silently ran both sums at 8.0 bohr (drivers: 15.0 AO / 25.0 nuclear) --
    the same "wrapper runs a tighter operator than the route it wraps" shape
    as IID 307.
    """
    from vibeqc._vibeqc_core import LatticeSumOptions

    _, kmesh = _compact_h2_3d_fixture()
    provider = PeriodicSCFProvider("sto-3g", kmesh, method="RHF")
    ref = LatticeSumOptions()
    assert provider._opts.lattice_opts.cutoff_bohr == ref.cutoff_bohr
    assert provider._opts.lattice_opts.nuclear_cutoff_bohr == ref.nuclear_cutoff_bohr
    assert ref.nuclear_cutoff_bohr != ref.cutoff_bohr, (
        "the fixture would be vacuous if the driver defaults coincided")


def test_periodic_provider_explicit_cutoff_keeps_the_set_both_semantics() -> None:
    """An explicit ``cutoff_bohr`` still applies to BOTH sums -- the historical
    contract every caller that passed the knob relied on -- and is recorded
    as what will run."""
    _, kmesh = _compact_h2_3d_fixture()
    provider = PeriodicSCFProvider("sto-3g", kmesh, method="RHF", cutoff_bohr=8.0)
    assert provider._opts.lattice_opts.cutoff_bohr == 8.0
    assert provider._opts.lattice_opts.nuclear_cutoff_bohr == 8.0
    assert provider._cutoff_bohr == 8.0


def test_periodic_provider_respects_caller_lattice_opts_when_cutoff_unset() -> None:
    """A caller's ``scf_options.lattice_opts`` is honoured, not clobbered.

    The second latent defect behind IID 540: the unconditional 8.0 overwrote
    caller-supplied cutoffs too, so ``scf_options`` could never tighten or
    loosen the sums. Twelve and twenty are chosen to differ from both the
    driver defaults and the old 8.0, so a regression to either shows.
    """
    from vibeqc._vibeqc_core import LatticeSumOptions, PeriodicRHFOptions

    _, kmesh = _compact_h2_3d_fixture()
    opts = PeriodicRHFOptions()
    lo = LatticeSumOptions(); lo.cutoff_bohr = 12.0; lo.nuclear_cutoff_bohr = 20.0
    opts.lattice_opts = lo
    provider = PeriodicSCFProvider("sto-3g", kmesh, method="RHF", scf_options=opts)
    assert provider._opts.lattice_opts.cutoff_bohr == 12.0
    assert provider._opts.lattice_opts.nuclear_cutoff_bohr == 20.0
    # ...and an explicit knob still wins over scf_options, as documented.
    provider2 = PeriodicSCFProvider("sto-3g", kmesh, method="RHF", scf_options=opts, cutoff_bohr=9.0)
    assert provider2._opts.lattice_opts.cutoff_bohr == 9.0
    assert provider2._opts.lattice_opts.nuclear_cutoff_bohr == 9.0


def test_periodic_provider_refusal_at_initial_geometry_aborts(monkeypatch) -> None:
    """A fold-guard refusal at the user's OWN geometry must propagate (IID 536).

    There is nothing to back off to, so the penalty contract must not swallow
    it -- otherwise a job would silently optimise a 1e6 Ha barrier.
    """
    import vibeqc.geomopt.periodic_providers as pp
    from vibeqc.pbc_bipole_common import BipoleFoldUnreliableError

    system, kmesh = _compact_h2_3d_fixture()
    provider = PeriodicSCFProvider("sto-3g", kmesh, method="RHF")

    def refuse(*a, **k):
        raise BipoleFoldUnreliableError("test: refused at the initial geometry")

    monkeypatch.setattr(pp, "run_pbc_bipole_rhf", refuse)
    with pytest.raises(BipoleFoldUnreliableError):
        provider(system)


def test_periodic_provider_refusal_at_trial_geometry_is_a_penalty(monkeypatch) -> None:
    """After one good evaluation a refusal is a finite barrier, not an error.

    Pins the contract ``BipoleFoldUnreliableError`` documents and
    ``bipole_optimize.relax_atoms`` already honours: the shared
    ``BIPOLE_TRIAL_PENALTY_HA`` with a **zero** gradient, and -- the part that
    saves real time -- no finite-difference gradient (6N SCFs) is attempted on
    a geometry the stack has just refused.
    """
    import vibeqc.geomopt.periodic_providers as pp
    from vibeqc.pbc_bipole_common import BIPOLE_TRIAL_PENALTY_HA, BipoleFoldUnreliableError

    system, kmesh = _compact_h2_3d_fixture()
    provider = PeriodicSCFProvider("sto-3g", kmesh, method="RHF")

    calls = {"scf": 0, "fd": 0}

    def scf_then_refuse(*a, **k):
        calls["scf"] += 1
        if calls["scf"] == 1:
            return type("Result", (), {"converged": True, "energy": -1.0})()
        raise BipoleFoldUnreliableError("test: refused at a trial geometry")

    def fd_counter(*a, **k):
        calls["fd"] += 1
        return np.zeros((2, 3))

    monkeypatch.setattr(pp, "run_pbc_bipole_rhf", scf_then_refuse)
    monkeypatch.setattr(pp, "compute_bipole_gradient_fd", fd_counter)

    e0, g0 = provider(system)          # good evaluation: real energy, FD ran
    assert e0 < 0.0 and calls["fd"] == 1

    e1, g1 = provider(system)          # refused trial: penalty, zero slope
    assert e1 == BIPOLE_TRIAL_PENALTY_HA
    assert g1.shape == (6,) and not g1.any()
    assert calls["fd"] == 1, "FD gradient must not run on a refused geometry"


def test_run_periodic_geomopt_recovers_from_a_refused_trial_step(monkeypatch) -> None:
    """A deterministic 3-D trial refusal is backtracked, not accepted.

    The fake SCF surface has its minimum at 1.2 bohr.  From 1.4 bohr, the
    first full BFGS step in fractional coordinates compresses H₂ to 0.68
    bohr and is deliberately refused; the half-step reaches 1.04 bohr and
    lowers the real energy.  This keeps IID 536 pinned without using the
    cutoff-dependent low-dimensional diagnostic as an objective.
    """
    import vibeqc.geomopt.periodic_providers as pp
    from vibeqc.pbc_bipole_common import (
        BIPOLE_TRIAL_PENALTY_HA,
        BipoleFoldUnreliableError,
    )

    system, kmesh = _compact_h2_3d_fixture()
    provider = PeriodicSCFProvider("sto-3g", kmesh, method="RHF")

    equilibrium = 1.2
    force_constant = 0.05
    calls = {"refusals": 0}

    def bond_geometry(trial_system):
        xyz = np.asarray([atom.xyz for atom in trial_system.unit_cell], dtype=float)
        delta = xyz[1] - xyz[0]
        return float(np.linalg.norm(delta)), delta

    def harmonic_scf(trial_system, *args, **kwargs):
        bond, _ = bond_geometry(trial_system)
        if bond < 0.9:
            calls["refusals"] += 1
            raise BipoleFoldUnreliableError("test: compressed trial refused")
        energy = -1.0 + 0.5 * force_constant * (bond - equilibrium) ** 2
        return type("Result", (), {"converged": True, "energy": energy})()

    def harmonic_gradient(trial_system, *args, **kwargs):
        bond, delta = bond_geometry(trial_system)
        unit = delta / bond
        slope = force_constant * (bond - equilibrium)
        return np.asarray([-slope * unit, slope * unit])

    monkeypatch.setattr(pp, "run_pbc_bipole_rhf", harmonic_scf)
    monkeypatch.setattr(pp, "compute_bipole_gradient_fd", harmonic_gradient)
    result = run_periodic_geomopt(
        system, provider, geom_opt="bfgs", geom_max_iter_atoms=1,
        geom_max_outer=1, relax_cell=False, force_mode="fd",
        fd_step_bohr=1e-3, record_trajectory=False, progress=False,
    )

    assert calls["refusals"] == 1
    assert result.energy == pytest.approx(-0.99936)
    assert result.energy < BIPOLE_TRIAL_PENALTY_HA / 2
    assert result.n_energy_evals > 0


@pytest.mark.slow
def test_run_periodic_geomopt_imports_and_runs(monkeypatch) -> None:
    """The public runner completes a real 3-D BIPOLE SCF smoke test.

    Its gradient is replaced with zero after that real SCF so this integration
    pin remains one electronic-structure evaluation rather than 6N+1.
    """
    import vibeqc.geomopt.periodic_providers as pp

    system, kmesh = _compact_h2_3d_fixture(a_bohr=12.0)
    provider = PeriodicSCFProvider("sto-3g", kmesh, method="RHF")
    monkeypatch.setattr(
        pp, "compute_bipole_gradient_fd", lambda *args, **kwargs: np.zeros((2, 3))
    )

    result = run_periodic_geomopt(
        system,
        provider,
        geom_opt="bfgs",
        geom_max_iter_atoms=1,
        geom_max_outer=1,
        relax_cell=False,
        force_mode="fd",
        fd_step_bohr=1e-3,
        record_trajectory=False,
        progress=False,
    )

    assert result.energy < 0
    assert result.n_energy_evals > 0
    assert result.optimizer == "bfgs"
