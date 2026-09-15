"""KRKS GDF XC-density regression gates (the 2026-07-09 KRKS finding).

`run_krks_periodic_gdf` at a Γ-only mesh delegates to the legacy
`run_rhf_periodic_gamma_gdf`, whose KS branch fed `build_xc_periodic` a
HOME-CELL-ONLY density set (`_density_set_gamma`). `build_xc_periodic`
treats such a set as a *molecular-limit* density (rho = chi_0 D chi_0, no
image-cell AO products -- `density_has_cross_cell_blocks` in
cpp/src/periodic_xc.cpp), which pairs with the molecular Becke grid of the
v0.8.x convention. On the default *periodic* Becke grid it drops every
image contribution to rho(r): exact in a vacuum box (which is why the
vacuum gates never saw it), ~0.3 Ha wrong on tight ionic cells.

Measured symptom (handovers/HANDOVER_AICCM_DIRECT_TORUS.md Finding §4, the
CHANGELOG [Unreleased] "Found: production KRKS ..." entry): rocksalt
LiH/STO-3G fcc a=7.72 bohr, (1,1,1), pure PBE --

* external PySCF 2.13.1 KRKS(GDF):  -8.238879724246 Ha
* run_ccm_rks_direct (real-Gamma):  -8.233910335588 Ha
* run_krks_periodic_gdf (pre-fix):  -7.930753392700 Ha  (+308 mHa)

with the pre-fix result's own stored density evaluating 0.24 Ha below the
energy it reported (molecular-limit e_xc vs the physical torus e_xc on the
same density).

The fix pairs the density convention with the grid: periodic-Becke grid
-> Gamma-torus density set (`_density_set_torus_gamma`, D in every lattice
block, build_xc_periodic's validated cross-cell mode -- the N_k=1 limit of
what every multi-k KS driver feeds); molecular grid
(use_periodic_becke=False) -> the historical home-cell-only set.
"""

from __future__ import annotations

import numpy as np
import pytest

from vibeqc import Atom, PeriodicSystem, make_basis, run_krks_periodic_gdf
from vibeqc._vibeqc_core import (
    Functional,
    GridOptions,
    PeriodicKSOptions,
    PeriodicXCDensityDomain,
    build_xc_periodic,
    compute_overlap_lattice,
    direct_lattice_cells,
)
from vibeqc._vibeqc_core import monkhorst_pack as mp_native
from vibeqc.periodic_grid import build_periodic_becke_grid
from vibeqc.periodic_k_gdf import (
    _build_xc_k_from_density,
    _xc_density_cells_for_domain,
)
from vibeqc.periodic_rhf_gdf import (
    _density_set_gamma,
    _density_set_torus_gamma,
)

# External anchor: PySCF 2.13.1 pyscf.pbc dft.KRKS(cell, kpts).density_fit(),
# xc='pbe', rocksalt LiH fcc primitive cell a = 7.72 bohr, STO-3G, (1,1,1)
# (out of process, 2026-07-09; reproduced for this fix -- see the
# CHANGELOG entry). Pure PBE carries no exchange-q=0 convention.
E_PYSCF_KRKS_PBE_LIH = -8.238879724246488
# Shared-machinery HF baseline on the same fixture deviates by ~4.7 mHa
# (ultra-diffuse Li/STO-3G fitting class, documented in
# tests/test_ccm_rks_direct.py) => gate the KS anchor at 6 mHa.
TOL_EXTERNAL_LIH = 6e-3


def test_periodic_xc_density_domain_contains_every_active_pair_difference():
    from vibeqc.pair_resolved_truncation import pair_resolved_domain

    system = PeriodicSystem(
        3,
        np.eye(3) * 5.0,
        [Atom(1, [0.0, 0.0, 0.0])],
    )
    lat_opts = PeriodicKSOptions().lattice_opts
    lat_opts.cutoff_bohr = 5.1
    active = list(
        pair_resolved_domain(system, lat_opts.cutoff_bohr).cells
    )

    automatic = _xc_density_cells_for_domain(
        system,
        lat_opts,
        active,
        PeriodicXCDensityDomain.AUTO,
    )
    extended = _xc_density_cells_for_domain(
        system,
        lat_opts,
        active,
        PeriodicXCDensityDomain.PERIODIC_LATTICE,
    )

    assert automatic is active
    extended_indices = {
        tuple(np.asarray(cell.index, dtype=int)) for cell in extended
    }
    for bra in active:
        bra_index = np.asarray(bra.index, dtype=int)
        for ket in active:
            difference = tuple(
                np.asarray(ket.index, dtype=int) - bra_index
            )
            assert difference in extended_indices


def _lih_system():
    a = 7.72
    lat = 0.5 * a * np.array([[0.0, 1, 1], [1, 0, 1], [1, 1, 0]]).T
    return PeriodicSystem(
        3, lat, [Atom(3, [0, 0, 0]), Atom(1, [0.5 * a, 0.5 * a, 0.5 * a])], 0, 1
    )


def test_density_set_torus_gamma_populates_every_block():
    """The torus setter writes D into every lattice block (cross-cell mode
    trigger), where the legacy setter zeroes all g != 0 blocks
    (molecular-limit mode trigger)."""
    system = _lih_system()
    basis = make_basis(system.unit_cell_molecule(), "sto-3g")
    lat_opts = PeriodicKSOptions().lattice_opts
    rng = np.random.default_rng(7)
    D = rng.standard_normal((basis.nbasis, basis.nbasis))
    D = 0.5 * (D + D.T)

    torus = _density_set_torus_gamma(
        compute_overlap_lattice(basis, system, lat_opts), D
    )
    home = _density_set_gamma(
        compute_overlap_lattice(basis, system, lat_opts), D
    )
    n = len(torus)
    assert n > 1  # tight cell: the lattice sum covers image cells
    for i in range(n):
        assert np.allclose(np.asarray(torus.blocks[i]), D)
    n_zero = sum(
        1
        for i in range(n)
        if np.max(np.abs(np.asarray(home.blocks[i]))) == 0.0
    )
    assert n_zero == n - 1


def test_gamma_torus_xc_matches_multi_k_fold_at_nk1():
    """Internal identity: build_xc_periodic on the Gamma-torus density set
    equals the multi-k fold helper (`_build_xc_k_from_density`) at its
    N_k = 1 limit -- same e_xc, same V_xc(Gamma) -- so the Gamma fast path
    and the multi-k branch now share one XC convention."""
    system = _lih_system()
    basis = make_basis(system.unit_cell_molecule(), "sto-3g")
    opts = PeriodicKSOptions()
    lat_opts = opts.lattice_opts
    grid = build_periodic_becke_grid(
        system,
        grid_options=GridOptions(),
        image_radius_bohr=float(getattr(opts, "becke_image_radius_bohr", 0.0)),
    )
    func = Functional("pbe", 1)

    # A plausible closed-shell density: superposition-free symmetric matrix
    # with the right trace scale (the identity is algebraic -- any Hermitian
    # D exercises it; use a deterministic one).
    rng = np.random.default_rng(11)
    D = rng.standard_normal((basis.nbasis, basis.nbasis))
    D = 0.1 * (D + D.T)

    torus = _density_set_torus_gamma(
        compute_overlap_lattice(basis, system, lat_opts), D
    )
    xc_direct = build_xc_periodic(basis, system, grid, func, torus, lat_opts)

    from vibeqc._vibeqc_core import bloch_sum

    kb = mp_native(system, [1, 1, 1], [0, 0, 0], False)
    cells = direct_lattice_cells(system, float(lat_opts.cutoff_bohr))
    e_xc_fold, vxc_k = _build_xc_k_from_density(
        basis=basis,
        system=system,
        grid=grid,
        func=func,
        density_k=[D.astype(complex)],
        kmesh_bloch=kb,
        cells=cells,
        kpoints_cart=[np.zeros(3)],
        lat_opts=lat_opts,
    )
    v_direct = np.real(bloch_sum(xc_direct.V_xc, np.zeros(3)))
    v_direct = 0.5 * (v_direct + v_direct.T)
    assert float(xc_direct.e_xc) == pytest.approx(e_xc_fold, abs=1e-10)
    assert np.max(np.abs(v_direct - np.real(vxc_k[0]))) < 1e-10


@pytest.mark.slow
def test_krks_gdf_gamma_pbe_lih_matches_external_pyscf():
    """End-to-end anchor: the production KRKS at (1,1,1) on tight ionic LiH
    now lands at the external PySCF KRKS(GDF) PBE value (pre-fix: +308 mHa).
    Pure functional => convention-free comparison. Since the Γ-KS routing
    fix (Finding-§4 residual (a)) this exercises the run_pbc_gdf_rks
    delegation rather than the legacy gamma driver."""
    system = _lih_system()
    basis = make_basis(system.unit_cell_molecule(), "sto-3g")
    r = run_krks_periodic_gdf(system, basis, (1, 1, 1), functional="pbe")
    assert r.converged
    assert r.energy == pytest.approx(E_PYSCF_KRKS_PBE_LIH, abs=TOL_EXTERNAL_LIH)


# External anchor for the Γ-path HYBRID KS (Finding-§4 residual (a)):
# PySCF 2.13.1 pyscf.pbc dft.KRKS(cell, kpts).density_fit(), xc='pbe0',
# default exxdiv='ewald', same LiH fixture (out of process, 2026-07-09).
E_PYSCF_KRKS_PBE0_LIH = -8.291224931282631


@pytest.mark.slow
def test_krks_gdf_gamma_pbe0_hybrid_lih_matches_external_pyscf():
    """Γ-path HYBRID KS anchor (Finding-§4 residual (a), fixed): pre-fix the
    Γ KS fast path delegated to the legacy gamma driver, whose full-range
    real-space K carries NO exxdiv convention — landing +8.3e-2 Ha above the
    exxdiv='ewald' references (not the strict-zero-mode offset
    a_x·ξ·N_e/2 = 0.297 Ha, a distinct truncated-sum gauge). Post-fix the Γ
    closed-shell KS delegates to run_pbc_gdf_rks: a_x-scaled GDF K + the
    exxdiv='ewald' Madelung shift, matching PySCF's convention — an exchange
    convention error would miss this anchor by ~0.1-0.3 Ha, far beyond the
    6 mHa shared-fitting-floor gate."""
    system = _lih_system()
    basis = make_basis(system.unit_cell_molecule(), "sto-3g")
    r = run_krks_periodic_gdf(system, basis, (1, 1, 1), functional="pbe0")
    assert r.converged
    assert float(r.e_hf_exchange) < 0.0  # exact-exchange channel active
    assert r.energy == pytest.approx(E_PYSCF_KRKS_PBE0_LIH, abs=TOL_EXTERNAL_LIH)


# External anchor for the true multi-k KS branch at a non-TRIM mesh:
# PySCF 2.13.1 KRKS(GDF) PBE / KRHF(GDF) on the dim=3-declared H2 chain
# (6 x 15 x 15 bohr box, H at (0,7.5,7.5) and (1.4,7.5,7.5)), kmesh (3,1,1)
# (out of process, 2026-07-09). vibe-qc measured -1.152329373928 (KS,
# delta 29 uHa: the XC-quadrature floor between grid schemes) and
# -1.118680489396 (HF, delta 5e-10).
E_PYSCF_KRKS_PBE_CHAIN3D = -1.152299977555
E_PYSCF_KRHF_CHAIN3D = -1.118680489887


def _chain3d_system():
    return PeriodicSystem(
        3, np.diag([6.0, 15.0, 15.0]),
        [Atom(1, [0.0, 7.5, 7.5]), Atom(1, [1.4, 7.5, 7.5])], 0, 1
    )


@pytest.mark.slow
def test_krks_gdf_multi_k_nontrim_chain_matches_external_pyscf():
    """The true multi-k KS branch (full real-space fold + per-k Bloch-folded
    V_xc) against external PySCF at a NON-TRIM mesh -- the complex-k XC fold
    has no TRIM cancellation to hide behind. Gate at 2e-4 Ha (measured
    29 uHa; grid-scheme floor)."""
    system = _chain3d_system()
    basis = make_basis(system.unit_cell_molecule(), "sto-3g")
    r = run_krks_periodic_gdf(system, basis, (3, 1, 1), functional="pbe")
    assert r.converged
    assert r.energy == pytest.approx(E_PYSCF_KRKS_PBE_CHAIN3D, abs=2e-4)


def test_krks_gdf_multik_dim1_fails_closed():
    """MULTI-K pure-KS must REFUSE ``dim < 3``, not converge on it.

    This test used to pin ``E = -3.894696294`` Ha/cell for the dim=1 H₂ chain
    (6-bohr cell, 15-bohr transverse vacuum, STO-3G, (3,1,1), PBE), justified by
    the real-Γ direct route agreeing to ~1e-10. That agreement was the trap: the
    two routes share the same defect, so they land on each other while both are
    wrong. The same ~1e-16 agreement holds on the H chain, where both are
    over-bound by 0.512 Ha/atom against the four-centre value that reproduces the
    isolated-H₂ anchor to 0.8 uHa. Mutual agreement between two routes sharing a
    defect is not validation.

    Two independent defects, either sufficient on its own:

    * the reciprocal mesh pins every non-periodic axis at ``G_perp = 0``, so the
      kernel it feeds is a transverse-uniform sheet term ``~1/V`` rather than
      ``1/r``, and the electron repulsion vanishes as the vacuum padding grows;
    * bare ``V_ne``/``E_nn`` combined with a neutral (``G+q=0``-dropped) ``J`` leave
      an uncancelled conditional constant ``c``, so the total diverges with the
      nuclear lattice-sum cutoff.

    Fixed 2026-07-10: ``_gauge_lat_opts_for_v_ne_and_e_nuc`` fails closed for
    ``dim < 3``, because for ``dim < 3`` there is no gauge to return. See
    ``docs/aiccm2026dev_a_lowd_greens.md`` section 0.1. The gauge-correct low-D
    Hamiltonian is the mixed-boundary wire kernel
    (``vibeqc.periodic.ccm.lowd_scf.run_ccm_rhf_wire``, ``dim == 1``)."""
    system = PeriodicSystem(
        1, np.diag([6.0, 15.0, 15.0]),
        [Atom(1, [0.0, 7.5, 7.5]), Atom(1, [1.4, 7.5, 7.5])], 0, 1
    )
    basis = make_basis(system.unit_cell_molecule(), "sto-3g")
    with pytest.raises(NotImplementedError, match="no consistent Coulomb gauge"):
        run_krks_periodic_gdf(system, basis, (3, 1, 1), functional="pbe")


def test_gamma_ewald_uks_matches_rks_closed_shell():
    """Closed-shell identity for the Gamma-Ewald pair: UKS(mult=1) == RKS on
    the dim=3 chain box (non-vacuum cross-cell overlap, below the
    dense-ionic fence). Both drivers now feed the torus density set on the
    periodic-Becke grid; pre-fix both fed the molecular-limit set, so the
    identity held for the wrong value."""
    from vibeqc._vibeqc_core import PeriodicKSOptions
    from vibeqc.periodic_rks_ewald import run_rks_periodic_gamma_ewald3d
    from vibeqc.periodic_uks_ewald import run_uks_periodic_gamma_ewald3d

    system = _chain3d_system()
    basis = make_basis(system.unit_cell_molecule(), "sto-3g")
    opts_r = PeriodicKSOptions()
    opts_r.functional = "pbe"
    r = run_rks_periodic_gamma_ewald3d(system, basis, opts_r)
    opts_u = PeriodicKSOptions()
    opts_u.functional = "pbe"
    u = run_uks_periodic_gamma_ewald3d(system, basis, opts_u)
    assert r.converged and u.converged
    assert u.energy == pytest.approx(r.energy, abs=1e-6)
