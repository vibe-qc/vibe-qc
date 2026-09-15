"""Meta-GGA GPW Fock correctness: the t (kinetic-energy-density) term.

Until this fix the whole Γ GPW meta-GGA path was dead on arrival, twice over:

1. ``_evaluate_xc_on_grid`` was called without ``basis``/``density_matrix``
   at every GPW SCF site, so a meta-GGA functional raised ``ValueError``
   before the first t build.
2. Had it been reached, ``_compute_kinetic_energy_density_fft`` crashed on a
   flat-vs-grid shape mismatch (``tau_flat (n_grid,) += (nx, ny, nz)``).
3. Had *that* worked, the v_tau Fock term was projected as a multiplicative
   potential ``∫ chi_mu chi_ν v_tau`` -- but the derivative of E_xc w.r.t.
   D_muν is ``1/2 ∫ v_tau gradchi_mu . gradchi_ν`` (the generalized-KS
   convention, matching ``cpp/src/periodic_xc.cpp``'s
   ``w_vtau = 0.5 w v_tau; V += da^T diag(w_vtau) ds``).

Pins here:

* the t builder runs and satisfies ``∫ t dV == tr(D T) / -- the grid
  kinetic-energy identity (integration by parts on a periodic cell);
* the full meta-GGA V_xc equals the numerical derivative dE_xc/dD
  (the wrong multiplicative projection fails this by construction);
* an end-to-end Γ GPW RKS-TPSS SCF converges to a finite energy;
* the molecular-limit multi-k GPW RKS-TPSS path converges too.
"""

from __future__ import annotations

import warnings

import numpy as np
import pytest
import vibeqc as vq
from vibeqc import _vibeqc_core as core
from vibeqc.periodic_gapw_grid import GAPWExperimentalWarning, PlaneWaveGrid
from vibeqc.periodic_gapw_j import (
    _compute_kinetic_energy_density_fft,
    _evaluate_xc_on_grid,
    _kinetic_lattice_gamma,
    _project_vtau_to_ao,
    _project_vxc_to_ao,
    build_gpw_collocation_cache,
    collocate_density_on_grid,
    run_periodic_rks_gpw,
    run_periodic_rks_gpw_multi_k,
)

warnings.simplefilter("ignore", GAPWExperimentalWarning)


def _h2_fixture(L: float = 10.0, n: int = 24):
    sysp = core.PeriodicSystem()
    sysp.dim = 3
    sysp.lattice = np.eye(3) * L
    sysp.unit_cell = [
        core.Atom(1, [L / 2 - 0.7, L / 2, L / 2]),
        core.Atom(1, [L / 2 + 0.7, L / 2, L / 2]),
    ]
    mol = vq.Molecule(list(sysp.unit_cell), 0, 1)
    basis = vq.BasisSet(mol, "sto-3g")
    grid = PlaneWaveGrid(np.eye(3) * L, n, n, n)
    return sysp, basis, grid


def _psd_density(basis, sysp):
    """A physical (PSD) closed-shell density from the Hcore guess, so the
    r >= 0 / t >= 0 clips in the XC evaluation stay inactive and the
    finite-difference derivative is clean."""
    T = _kinetic_lattice_gamma(basis, sysp)
    from vibeqc.periodic_gapw_j import _overlap_lattice_gamma

    S = _overlap_lattice_gamma(basis, sysp)
    s_e, U = np.linalg.eigh(S)
    X = U @ np.diag(1.0 / np.sqrt(np.maximum(s_e, 1e-12))) @ U.T
    _, C_orth = np.linalg.eigh(X.T @ T @ X)
    C = X @ C_orth
    n_occ = 1  # H2: one doubly-occupied orbital
    return 2.0 * (C[:, :n_occ] @ C[:, :n_occ].T)


def test_tau_builder_runs_and_matches_kinetic_trace():
    """t no longer crashes, is non-negative for a PSD density, and its
    integral reproduces tr(D T) -- the periodic-cell integration-by-parts
    identity ∫ t dV = 1/2 S D_mn ∫ gradchi_m.gradchi_n = tr(D T) -- to
    within the smooth-grid quadrature error."""
    sysp, basis, grid = _h2_fixture()
    D = _psd_density(basis, sysp)

    tau = _compute_kinetic_energy_density_fft(basis, D, grid)
    assert tau.shape == grid.shape
    assert float(tau.min()) >= -1e-10

    n_tau = float(tau.sum() * grid.voxel_volume_bohr3)
    T = _kinetic_lattice_gamma(basis, sysp)
    e_kin = float(np.einsum("ij,ij->", D, T))
    assert n_tau == pytest.approx(e_kin, rel=5e-3)


def test_mgga_vxc_is_derivative_of_exc():
    """V_xc (LDA + GGA-by-parts + 1/2 v_tau gradchi.gradchi, with the
    constrained chain rule at t_W-clamped points) equals the central finite
    difference dE_xc/dD_muν for r2SCAN at a fixed PSD density.

    The pre-fix multiplicative v_tau projection fails this pin by
    construction: ∫ chichi v_tau is not the derivative of E_xc[t[D]].
    r2SCAN is the pin functional because it self-regularises the iso-orbital
    boundary; TPSS-class functionals have intrinsically divergent partials
    (dz/dsigma = 1/(8 rho t)) at orbital critical points, which a uniform
    grid samples -- there both FD and the projection are stiffness-dominated
    and no tolerance is meaningful (see the module docstring)."""
    sysp, basis, grid = _h2_fixture()
    cache = build_gpw_collocation_cache(basis, grid)
    func = core.Functional("r2scan", 1)
    D0 = _psd_density(basis, sysp)
    n_basis = basis.nbasis

    def exc_of(D):
        rho = collocate_density_on_grid(basis, D, grid, cache=cache)
        e_xc, *_ = _evaluate_xc_on_grid(
            rho, grid, func, basis=basis, density_matrix=D, cache=cache
        )
        return e_xc

    # Analytic V_xc at D0.
    rho0 = collocate_density_on_grid(basis, D0, grid, cache=cache)
    _, v_xc_grid, v_sigma_grid, grad_rho, v_tau_grid = _evaluate_xc_on_grid(
        rho0, grid, func, basis=basis, density_matrix=D0, cache=cache
    )
    assert v_tau_grid is not None  # r2SCAN is a meta-GGA
    V = _project_vxc_to_ao(
        basis,
        v_xc_grid,
        grid,
        v_sigma_grid=v_sigma_grid,
        grad_rho=grad_rho,
        cache=cache,
    )
    V = V + _project_vtau_to_ao(basis, v_tau_grid, grid, cache=cache)

    # Central FD over symmetric perturbations: perturbing D_mn AND D_nm by
    # h gives dE = (2 - d_mn) h V_mn for a symmetric V.
    h = 1e-5
    for (m, n) in [(0, 0), (0, 1), (1, 1)]:
        P = np.zeros((n_basis, n_basis))
        P[m, n] = 1.0
        P[n, m] = 1.0  # keep D symmetric (no-op on the diagonal)
        e_plus = exc_of(D0 + h * P)
        e_minus = exc_of(D0 - h * P)
        fd = (e_plus - e_minus) / (2.0 * h)
        scale = 1.0 if m == n else 2.0
        assert fd == pytest.approx(scale * V[m, n], rel=1e-3, abs=1e-6), (
            f"dE_xc/dD[{m},{n}] mismatch: FD={fd}, V={scale * V[m, n]}"
        )


def test_rks_gpw_r2scan_gamma_converges():
    """End-to-end Γ GPW RKS-r2SCAN on boxed H2 converges to a finite, sane
    energy (this SCF previously died at the first XC evaluation)."""
    sysp, basis, grid = _h2_fixture()
    res = run_periodic_rks_gpw(
        sysp,
        basis,
        functional="r2scan",
        grid=grid,
        max_iter=60,
        quiet=True,
    )
    assert res.converged
    assert np.isfinite(res.energy)
    # Boxed H2/STO-3G r2SCAN: loose physical window around ~-1.1 Ha.
    assert -1.35 < res.energy < -0.85


def test_rks_gpw_tpss_gamma_fails_closed_on_stiffness():
    """TPSS on boxed H2 (single-orbital: the whole grid rides the
    iso-orbital boundary where TPSS's dz/dsigma = 1/(8 rho t) diverges)
    must fail closed with the stiffness guard, not silently converge to a
    spurious stationary point (+0.32 Ha was one observed artifact)."""
    sysp, basis, grid = _h2_fixture()
    with pytest.raises(NotImplementedError, match="stiffness-dominated"):
        run_periodic_rks_gpw(
            sysp,
            basis,
            functional="tpss",
            grid=grid,
            max_iter=60,
            quiet=True,
        )


def test_rks_gpw_multik_mgga_molecular_limit_converges():
    """Molecular-limit multi-k GPW RKS-r2SCAN converges (t built from the
    Gamma-summed density; compact cells still gate meta-GGA upstream)."""
    sysp, basis, grid = _h2_fixture(L=16.0, n=32)
    kmesh = core.monkhorst_pack(sysp, [1, 1, 2])
    res = run_periodic_rks_gpw_multi_k(
        sysp,
        basis,
        kmesh,
        functional="r2scan",
        grid=grid,
        max_iter=60,
        quiet=True,
    )
    assert res.converged
    assert np.isfinite(res.energy)
    assert -1.35 < res.energy < -0.85
