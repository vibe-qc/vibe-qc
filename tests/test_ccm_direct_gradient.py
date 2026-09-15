"""EXPERIMENTAL: analytic forces on the direct-torus route (milestone 2c).

Gates for :func:`vibeqc.periodic.ccm.direct.run_ccm_direct_gradient`
(2026-08-19): the composition of the route's Fourier-representation
identity with the landed production multi-k rsgdf analytic gradient
(``compute_gradient=True``, public since 2026-07-30). The same-Hamiltonian
premise is VERIFIED per run by the energy-parity gate — never assumed —
so every force this driver returns is attributable to the direct energy
at the measured parity floor.

Measured on this file's fixtures (2026-08-19, h = 2e-4 central FD of the
DIRECT energies, per-cell convention):

* RHF, compact H₂ (2,1,1):      |analytic − FD| = 5.6e-9, parity 1.3e-13
* RKS-PBE / PBE0, same fixture: 1.26e-4 / 9.6e-5 — the production KS
  gradient's own documented XC-grid floor (their slab KRKS FD gate sits
  at 2e-4, "XC-grid limited"), NOT a composition error; parity 1.3e-13.
* UHF, triplet LiH (1,1,1):     6.8e-7, parity 4.4e-12 (force 0.045).
* UKS, triplet LiH (1,1,1): the production KUKS analytic gradient now
  includes atom-centred XC grid motion and matches FD of its own energy
  to 1e-8 Ha/bohr in ``test_periodic_gdf_gradient.py``. The direct route
  is consequently enabled for open-shell KS as well.
"""

from __future__ import annotations

import warnings

import numpy as np
import pytest
from vibeqc import Atom, PeriodicSystem
from vibeqc.periodic.ccm import CCMSystem
from vibeqc.periodic.ccm.direct import (
    run_ccm_direct_gradient,
    run_ccm_rhf_direct,
    run_ccm_rks_direct,
    run_ccm_uhf_direct,
)

pytestmark = pytest.mark.experimental  # Γ-CCM (aiccm2026dev-a) research lane

_H_FD = 2e-4


def _h2_ccm(dz=0.0, nrep=(2, 1, 1)):
    """Asymmetric compact H₂ cube (nonzero z-force on atom 1)."""
    cell = PeriodicSystem(
        3, np.diag([6.0, 6.0, 6.0]),
        [Atom(1, [3.0, 3.0, 2.2]), Atom(1, [3.0, 3.0, 3.8 + dz])], 0, 1)
    return CCMSystem(cell, nrep, "sto-3g")


def _lih_triplet_ccm(dz=0.0):
    """Genuine open-shell fixture with a real force (0.045 Ha/bohr)."""
    cell = PeriodicSystem(
        3, np.diag([7.0, 7.0, 7.0]),
        [Atom(3, [3.5, 3.5, 2.6]), Atom(1, [3.5, 3.5, 5.6 + dz])], 0, 3)
    return CCMSystem(cell, (1, 1, 1), "sto-3g")


def _quiet(fn, *a, **k):
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        return fn(*a, **k)


def _fd_per_cell(runner, mk, *args, atom_z_of=None, **kw):
    ep = _quiet(runner, mk(+_H_FD), *args, **kw)
    em = _quiet(runner, mk(-_H_FD), *args, **kw)
    n_c = int(np.prod(mk(0.0).nrep))
    return (float(ep.energy) - float(em.energy)) / (2 * _H_FD) / n_c


def test_direct_gradient_rhf_matches_fd_of_direct_energy():
    """The decisive gate: analytic == central FD of the DIRECT per-cell
    energy (measured 5.6e-9), with the parity premise recorded on the
    result and the convention fields present."""
    fd = _fd_per_cell(run_ccm_rhf_direct, _h2_ccm)
    g = _quiet(run_ccm_direct_gradient, _h2_ccm(0.0))
    assert abs(g.gradient[1, 2] - fd) < 1e-6
    assert g.parity_residual_ha_per_cell < 1e-10
    assert g.exchange_q0 == "BvK-ewald"
    assert g.exchange_q0_applicability == "active"
    assert g.gradient.shape == (2, 3)


@pytest.mark.slow
@pytest.mark.parametrize("functional,appl", [("pbe", "inactive"),
                                             ("pbe0", "active")])
def test_direct_gradient_ks_matches_fd_at_xc_grid_floor(functional, appl):
    """KS forces land at the production KS gradient's own XC-grid floor
    (measured 1.26e-4 PBE / 9.6e-5 PBE0 vs FD; upstream slab KRKS FD gate
    is 2e-4 'XC-grid limited'). Gate at 5e-4."""
    fd = _fd_per_cell(run_ccm_rks_direct, _h2_ccm, functional)
    g = _quiet(run_ccm_direct_gradient, _h2_ccm(0.0), functional=functional)
    assert abs(g.gradient[1, 2] - fd) < 5e-4
    assert g.parity_residual_ha_per_cell < 1e-10
    assert g.exchange_q0_applicability == appl


@pytest.mark.slow
def test_direct_gradient_uhf_matches_fd_open_shell():
    """Open-shell HF forces on a genuine triplet (dominant force 0.045
    Ha/bohr; measured 6.8e-7 vs FD — near the h=2e-4 FD noise floor)."""
    fd = _fd_per_cell(run_ccm_uhf_direct, _lih_triplet_ccm)
    g = _quiet(run_ccm_direct_gradient, _lih_triplet_ccm(0.0))
    assert abs(g.gradient[1, 2] - fd) < 5e-6
    assert g.parity_residual_ha_per_cell < 1e-10


@pytest.mark.slow
def test_direct_gradient_fail_closed_surface():
    """Boundary: the parity premise gates hard (an impossible parity_tol
    raises rather than returning unattributable forces); strict-zero,
    screened hybrids, and double hybrids all fail closed with pointers."""
    ccm = _h2_ccm(0.0)
    with pytest.raises(ValueError, match="parity"):
        _quiet(run_ccm_direct_gradient, ccm, parity_tol=1e-16)
    with pytest.raises(NotImplementedError, match="ewald"):
        _quiet(run_ccm_direct_gradient, ccm, exxdiv=None)
    with pytest.raises(NotImplementedError, match="screened|range-sep"):
        _quiet(run_ccm_direct_gradient, ccm, functional="hse06")
    with pytest.raises(NotImplementedError, match="double hybrid"):
        _quiet(run_ccm_direct_gradient, ccm, functional="b2plyp")


def test_direct_gradient_routes_open_shell_ks(monkeypatch):
    """Issue #158 closure: genuine open-shell KS reaches the KUKS gradient
    control instead of the former fail-closed defect gate."""
    from types import SimpleNamespace

    sentinel_options = object()
    calls = []
    direct = SimpleNamespace(
        converged=True,
        energy=-4.0,
        exchange_q0="",
        exchange_q0_applicability="inactive",
    )
    control = SimpleNamespace(
        converged=True,
        energy=-4.0,
        exchange_q0="",
        exchange_q0_applicability="inactive",
    )
    expected = np.arange(6, dtype=float).reshape(2, 3)

    def fake_direct(ccm, functional, **kwargs):
        calls.append(("direct", functional, kwargs))
        assert functional == "pbe"
        assert kwargs["initial_guess"] == "AUTO"
        return direct

    def fake_gdf(ccm, functional, **kwargs):
        calls.append(("gdf", functional, kwargs))
        assert functional == "pbe"
        assert kwargs["initial_guess"].name == "HCORE"
        assert kwargs["compute_gradient"] is True
        assert kwargs["options"] is sentinel_options
        return SimpleNamespace(
            **vars(control), raw=SimpleNamespace(gradient=expected)
        )

    monkeypatch.setattr(
        "vibeqc.periodic.ccm.direct.run_ccm_uks_direct", fake_direct
    )
    monkeypatch.setattr(
        "vibeqc.periodic.ccm.ri.run_ccm_uks_gdf", fake_gdf
    )
    result = run_ccm_direct_gradient(
        _lih_triplet_ccm(0.0), functional="pbe", options=sentinel_options
    )
    assert [call[0] for call in calls] == ["direct", "gdf"]
    assert np.array_equal(result.gradient, expected)
    assert result.parity_residual_ha_per_cell == 0.0


@pytest.mark.slow
def test_direct_optimize_relaxes_h2():
    """Geometry relaxation on the direct-torus surface (endpoints mode):
    the asymmetric compact H₂ relaxes 1.6 → ~1.47 bohr in a handful of
    L-BFGS-B steps (measured: 4 steps, final max|grad| 2.1e-5 Ha/bohr,
    endpoint parities 1.3e-13). The relaxed stationary point carries full
    direct-vs-control parity verification at both endpoints."""
    from vibeqc.periodic.ccm.direct import run_ccm_direct_optimize

    r = _quiet(run_ccm_direct_optimize, _h2_ccm(0.0))
    assert r.converged
    assert r.parity_check == "endpoints"
    pos = [np.asarray(a.xyz) for a in r.system_opt.unit_cell]
    bond = abs(pos[1][2] - pos[0][2])
    assert 1.3 < bond < 1.6, bond
    assert np.max(np.abs(r.gradient)) < 1e-4
    assert r.parity_initial.parity_residual_ha_per_cell < 1e-10
    assert r.parity_final.parity_residual_ha_per_cell < 1e-10


def test_direct_optimize_boundary():
    """Cheap boundary pins: unknown parity_check rejected before any SCF."""
    from vibeqc.periodic.ccm.direct import run_ccm_direct_optimize

    with pytest.raises(ValueError, match="parity_check"):
        _quiet(run_ccm_direct_optimize, _h2_ccm(0.0), parity_check="never")


@pytest.mark.parametrize(
    "conv_tol_grad",
    (-1.0, 0.0, float("nan"), float("inf")),
    ids=("negative", "zero", "nan", "positive-infinity"),
)
def test_direct_optimize_rejects_invalid_gradient_tolerance(conv_tol_grad):
    """Invalid geometry thresholds fail before parity or gradient work."""
    from vibeqc.periodic.ccm.direct import run_ccm_direct_optimize

    with pytest.raises(
        ValueError,
        match=(
            r"run_ccm_direct_optimize: conv_tol_grad must be finite and "
            r"positive\."
        ),
    ):
        run_ccm_direct_optimize(None, conv_tol_grad=conv_tol_grad)


@pytest.mark.parametrize("conv_tol_grad", (None, 4.5e-4, 1e-5))
def test_direct_optimize_valid_gradient_tolerance_preserves_boundary(
    conv_tol_grad,
):
    """Default and positive values leave existing route validation intact."""
    from vibeqc.periodic.ccm.direct import run_ccm_direct_optimize

    kwargs = {"parity_check": "never"}
    if conv_tol_grad is not None:
        kwargs["conv_tol_grad"] = conv_tol_grad
    with pytest.raises(ValueError, match="parity_check"):
        run_ccm_direct_optimize(None, **kwargs)


@pytest.mark.parametrize(
    "max_steps",
    (
        -1,
        0,
        -0.5,
        1.5,
        2.0,
        float("nan"),
        float("inf"),
        True,
        False,
        "2",
        None,
    ),
    ids=(
        "negative-integer",
        "zero",
        "negative-fraction",
        "positive-fraction",
        "integral-float",
        "nan",
        "positive-infinity",
        "true",
        "false",
        "string",
        "none",
    ),
)
def test_direct_optimize_rejects_invalid_step_budget(max_steps):
    """Issue 368: malformed budgets fail before parity or gradient work."""
    from vibeqc.periodic.ccm.direct import run_ccm_direct_optimize

    with pytest.raises(
        ValueError,
        match=(
            r"run_ccm_direct_optimize: max_steps must be an integer >= 1;"
        ),
    ):
        run_ccm_direct_optimize(None, max_steps=max_steps)


@pytest.mark.parametrize("max_steps", (None, 1, np.int64(2)))
def test_direct_optimize_valid_step_budget_preserves_boundary(max_steps):
    """The default and integer budgets retain downstream route validation."""
    from vibeqc.periodic.ccm.direct import run_ccm_direct_optimize

    kwargs = {"parity_check": "never"}
    if max_steps is not None:
        kwargs["max_steps"] = max_steps
    with pytest.raises(ValueError, match="parity_check"):
        run_ccm_direct_optimize(None, **kwargs)


@pytest.mark.parametrize("is_ks", (False, True))
def test_gdf_control_guess_options_preserve_caller(is_ks):
    from vibeqc import InitialGuess, PeriodicKSOptions, PeriodicRHFOptions
    from vibeqc.periodic.ccm.ri import _gdf_guess_kwargs

    source = PeriodicKSOptions() if is_ks else PeriodicRHFOptions()
    source.initial_guess = InitialGuess.SAD
    source.max_iter = 37
    source.ecp_total_ncore = 28
    source.ecp_effective_charges = [19.0]
    source.lattice_opts.pair_complete_1e = False
    result = _gdf_guess_kwargs(
        {"options": source, "initial_guess": "CORE", "compute_gradient": True},
        is_ks=is_ks,
    )
    options = result["options"]
    assert options is not source
    assert source.initial_guess == InitialGuess.SAD
    assert options.initial_guess == InitialGuess.HCORE
    assert options.max_iter == 37
    assert options.ecp_total_ncore == 28
    assert list(options.ecp_effective_charges) == [19.0]
    assert not options.lattice_opts.pair_complete_1e
    assert result["compute_gradient"]
    assert "initial_guess" not in result
    with pytest.raises(ValueError, match="initial.guess"):
        _gdf_guess_kwargs({"initial_guess": "BOGUS"}, is_ks=is_ks)
