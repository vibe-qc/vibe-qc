"""Open-shell (Γ) GPW meta-GGA: per-spin τ through eval_polarised_mgga.

Lifts the gate placed when the closed-shell GPW meta-GGA path was revived:
``run_periodic_uks_gpw(functional=<mGGA>)`` now builds per-spin
kinetic-energy densities, applies the same regularisation chain as the
closed-shell path per spin channel (von Weizsäcker floor + constrained
chain rule, uniform-grid density screen, fail-closed stiffness guard), and
adds the per-spin generalized-KS Fock term ``½∫v_τσ ∇χ·∇χ``.

Pins:

* **Closed-shell reduction** — UKS(r2scan) on an H₂ singlet reproduces the
  RKS(r2scan) energy (the polarised evaluation at ρ_α == ρ_β must match
  the unpolarised one);
* **Fully-polarised channel** — an H-atom doublet keeps the majority
  spin's exchange (a screen bug zeroing exc where *either* spin was small
  gave −0.08 Ha instead of ~−0.46; the screen now requires *both* spins
  small);
* TPSS-class stiffness still fails closed on the iso-orbital worst case;
* multi-k UKS keeps its meta-GGA gate (per-spin Bloch τ not wired).
"""

from __future__ import annotations

import warnings

import numpy as np
import pytest
import vibeqc as vq
from vibeqc import _vibeqc_core as core
from vibeqc.periodic_gapw_grid import GAPWExperimentalWarning, PlaneWaveGrid
from vibeqc.periodic_gapw_j import run_periodic_rks_gpw
from vibeqc.periodic_gapw_open_shell import (
    run_periodic_uks_gpw,
    run_periodic_uks_gpw_multi_k,
)

warnings.simplefilter("ignore", GAPWExperimentalWarning)


def _h2_singlet(L: float = 10.0):
    sysp = core.PeriodicSystem()
    sysp.dim = 3
    sysp.lattice = np.eye(3) * L
    sysp.unit_cell = [
        core.Atom(1, [L / 2 - 0.7, L / 2, L / 2]),
        core.Atom(1, [L / 2 + 0.7, L / 2, L / 2]),
    ]
    mol = vq.Molecule(list(sysp.unit_cell), 0, 1)
    return sysp, vq.BasisSet(mol, "sto-3g"), PlaneWaveGrid(np.eye(3) * L, 24, 24, 24)


def _h_doublet(L: float = 10.0):
    sysp = core.PeriodicSystem()
    sysp.dim = 3
    sysp.lattice = np.eye(3) * L
    sysp.unit_cell = [core.Atom(1, [L / 2, L / 2, L / 2])]
    sysp.multiplicity = 2
    mol = vq.Molecule(list(sysp.unit_cell), 0, 2)
    return sysp, vq.BasisSet(mol, "sto-3g"), PlaneWaveGrid(np.eye(3) * L, 24, 24, 24)


def test_uks_mgga_closed_shell_reduction():
    sysp, basis, grid = _h2_singlet()
    rks = run_periodic_rks_gpw(sysp, basis, functional="r2scan", grid=grid, quiet=True)
    uks = run_periodic_uks_gpw(sysp, basis, functional="r2scan", grid=grid, quiet=True)
    assert rks.converged and uks.converged
    assert uks.energy == pytest.approx(rks.energy, abs=1e-8)


def test_uks_mgga_doublet_keeps_majority_spin_exchange():
    """The fully-polarised channel (rho_beta == 0 everywhere) must retain
    the alpha exchange: E(r2scan) sits BELOW E(lda) for the boxed H atom.
    The either-spin screen bug gave -0.08 Ha here (exc zeroed everywhere)."""
    sysp, basis, grid = _h_doublet()
    lda = run_periodic_uks_gpw(sysp, basis, functional="lda", grid=grid, quiet=True)
    mgga = run_periodic_uks_gpw(sysp, basis, functional="r2scan", grid=grid, quiet=True)
    assert lda.converged and mgga.converged
    assert -0.55 < mgga.energy < -0.40
    assert mgga.energy < lda.energy


def test_uks_mgga_tpss_fails_closed_on_stiffness():
    """TPSS on the single-orbital H2 cell rides the iso-orbital boundary;
    the per-spin stiffness guard must fail closed like the closed-shell
    path (not silently converge to a spurious point)."""
    sysp, basis, grid = _h2_singlet()
    with pytest.raises(NotImplementedError, match="stiffness-dominated"):
        run_periodic_uks_gpw(
            sysp, basis, functional="tpss", grid=grid, quiet=True, max_iter=40
        )


def test_uks_multik_mgga_now_supported():
    """Multi-k spin-polarised meta-GGA is wired on GPW (per-spin Bloch tau +
    generalised-KS V_tau). A closed-shell H2 cell through the multi-k UKS
    driver reproduces the Gamma open-shell GPW r2SCAN energy on the same
    grid -- the gate that used to reject this call is gone.

    Full RKS/UKS parity on both the molecular-limit and compact Bloch
    regimes lives in ``test_periodic_gpw_uks_multik.py``; this pins that the
    r2SCAN call no longer raises from *this* module's driver.
    """
    sysp, basis, grid = _h2_singlet(L=12.0)
    gamma = run_periodic_uks_gpw(
        sysp, basis, functional="r2scan", grid=grid, quiet=True,
        conv_tol_energy=1e-9, conv_tol_density=1e-7,
    )
    kmesh = core.monkhorst_pack(sysp, [1, 1, 1])
    multik = run_periodic_uks_gpw_multi_k(
        sysp, basis, kmesh, functional="r2scan", grid=grid, quiet=True,
        conv_tol_energy=1e-9, conv_tol_density=1e-7,
    )
    assert gamma.converged and multik.converged
    assert np.isfinite(multik.energy)
    assert multik.energy == pytest.approx(gamma.energy, abs=5e-7)
