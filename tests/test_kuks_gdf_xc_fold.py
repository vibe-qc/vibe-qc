"""Multi-k KUKS XC per-spin real-space fold gates (2026-07-09 follow-up).

`run_kuhf_periodic_gdf`'s KS branch historically fed `build_xc_periodic_uks`
the BZ-averaged density in the HOME-CELL block only (`_density_set_gamma`) --
the C++ kernel's molecular-limit mode, one k-independent V_xc(Γ) added to
every k. Same defect class as the fixed Γ KRKS finding (commit b3f74aa9):
exact in the vacuum/molecular limit, wrong by the image density on tight
cells (+3.2e-4 Ha on the dim=3 H2 chain (3,1,1) PBE fixture below).

The fix mirrors the closed-shell multi-k branch: per-spin inverse-Bloch fold
of the per-k densities to the real-space finite-torus set
(`_build_xc_k_from_density_uks`), spin-polarised periodic XC on the pair,
per-k Bloch-folded V_a/V_b. It also repairs the options default that sank
the FIRST fold attempt by -2.4 Ha on this very identity: KS runs through
`run_kuhf_periodic_gdf` used to default to `PeriodicRHFOptions`, which
carries no ``use_periodic_becke``, silently selecting the MOLECULAR Becke
grid -- and a cross-cell density fold on the molecular grid over-counts by
the replicated image density (the b3f74aa9 grid/density-convention pairing,
in reverse). KS runs now default to `PeriodicKSOptions` (periodic-Becke
grid), like the closed-shell `run_krks_periodic_gdf` wrapper always did.

Identity caveat (measured 2026-07-09): the DEFAULT closed-shell pure-DFT
route keeps `use_compcell=False` (Ewald-3D real-space J), while the
open-shell driver always contracts the cached GDF Lpq for J. Those two J
backends legitimately differ at the density-fitting scale (-4.4e-5 Ha on
the chain fixture), so the machine-precision KUKS==KRKS identity is gated
on the like-for-like pairing: KRKS(use_compcell=True, rsgdf) vs
KUKS(rsgdf).
"""

from __future__ import annotations

import numpy as np
import pytest

from vibeqc import Atom, PeriodicSystem, make_basis
from vibeqc._vibeqc_core import (
    GridOptions,
    Functional,
    PeriodicKSOptions,
    direct_lattice_cells,
)
from vibeqc._vibeqc_core import monkhorst_pack as mp_native
from vibeqc.periodic_grid import build_periodic_becke_grid
from vibeqc.periodic_k_gdf import (
    _build_xc_k_from_density,
    _build_xc_k_from_density_uks,
    run_krks_periodic_gdf,
    run_kuks_periodic_gdf,
)


def _chain3d_system(multiplicity: int = 1) -> PeriodicSystem:
    # The dim=3-declared H2 chain of tests/test_krks_gdf_xc_density.py:
    # tight along x (a = 6 bohr), vacuum-padded in y/z. Non-TRIM (3,1,1)
    # meshes give genuinely complex k -- no time-reversal cancellation to
    # hide a broken fold behind.
    return PeriodicSystem(
        3, np.diag([6.0, 15.0, 15.0]),
        [Atom(1, [0.0, 7.5, 7.5]), Atom(1, [1.4, 7.5, 7.5])], 0, multiplicity
    )


def test_uks_fold_matches_closed_shell_fold_at_equal_spins():
    """Algebraic spin identity of the fold helpers, no SCF: with
    D_a(k) = D_b(k) = D(k)/2 the per-spin fold + spin-polarised XC must
    reproduce the closed-shell fold + unpolarised XC -- same e_xc, same
    V_xc(k) per spin -- at a multi-k mesh with complex k."""
    system = _chain3d_system()
    basis = make_basis(system.unit_cell_molecule(), "sto-3g")
    opts = PeriodicKSOptions()
    lat_opts = opts.lattice_opts
    grid = build_periodic_becke_grid(
        system,
        grid_options=GridOptions(),
        image_radius_bohr=float(getattr(opts, "becke_image_radius_bohr", 0.0)),
    )

    kb = mp_native(system, [3, 1, 1], [0, 0, 0], False)
    kpoints = [np.asarray(k, dtype=float).reshape(3) for k in kb.kpoints]
    cells = direct_lattice_cells(system, float(lat_opts.cutoff_bohr))

    # Deterministic Hermitian per-k densities with k-dependent complex
    # structure (the identity is algebraic; any Hermitian set exercises it).
    rng = np.random.default_rng(23)
    n = basis.nbasis
    D_k = []
    for _ in kpoints:
        M = rng.standard_normal((n, n)) + 1j * rng.standard_normal((n, n))
        D_k.append(0.1 * (M + M.conj().T))

    e_r, v_r_k = _build_xc_k_from_density(
        basis=basis, system=system, grid=grid, func=Functional("pbe", 1),
        density_k=D_k, kmesh_bloch=kb, cells=cells,
        kpoints_cart=kpoints, lat_opts=lat_opts,
    )
    e_u, va_k, vb_k = _build_xc_k_from_density_uks(
        basis=basis, system=system, grid=grid, func=Functional("pbe", 2),
        density_alpha_k=[0.5 * D for D in D_k],
        density_beta_k=[0.5 * D for D in D_k],
        kmesh_bloch=kb, cells=cells,
        kpoints_cart=kpoints, lat_opts=lat_opts,
    )
    assert e_u == pytest.approx(e_r, abs=1e-10)
    for i in range(len(kpoints)):
        assert np.max(np.abs(va_k[i] - v_r_k[i])) < 1e-10
        assert np.max(np.abs(vb_k[i] - v_r_k[i])) < 1e-10


@pytest.mark.slow
def test_kuks_krks_identity_multi_k_chain():
    """KUKS(mult=1) == KRKS on the tight chain at (3,1,1) PBE, like-for-like
    J backend (cached-Lpq rsgdf both sides). Measured 4.4e-16 Ha post-fix;
    pre-fix the home-cell XC shortcut left +3.2e-4 Ha. Gated at 1e-12."""
    system = _chain3d_system()
    basis = make_basis(system.unit_cell_molecule(), "sto-3g")
    r = run_krks_periodic_gdf(
        system, basis, (3, 1, 1), functional="pbe",
        use_compcell=True, gdf_method="rsgdf",
    )
    u = run_kuks_periodic_gdf(
        system, basis, (3, 1, 1), functional="pbe", gdf_method="rsgdf",
    )
    assert r.converged and u.converged
    assert abs(u.s_squared) < 1e-10
    assert u.energy == pytest.approx(r.energy, abs=1e-12)
    for i in range(len(r.density)):
        D_tot = np.asarray(u.density_alpha[i]) + np.asarray(u.density_beta[i])
        assert np.max(np.abs(np.asarray(r.density[i]) - D_tot)) < 1e-8


# External anchor: PySCF 2.13.1 pyscf.pbc dft.KUKS(cell, kpts).density_fit(),
# xc='pbe', conv_tol=1e-10, on the chain fixture with mf.nelec = (6, 0) --
# per-cell (2, 0), i.e. the fully-spin-polarised state vibe-qc's per-cell
# multiplicity=3 prescribes. (PySCF's cell.spin distributes the excess over
# the whole BvK supercell -- nelec (4, 2) at (3,1,1) -- which is a DIFFERENT
# state, 0.40 Ha below; the override pins the like-for-like one. Out of
# process per CLAUDE.md §10, 2026-07-09.) The xc-free KUHF twin on the same
# fixture/override gives -0.527888819660 vs vibe-qc -0.527888820777
# (1.1e-9 Ha): the J/K/exxdiv machinery is exact, so the KS deviation is
# the XC-quadrature-scheme floor (periodic-Becke vs PySCF FFT grid;
# measured -1.26e-4 Ha).
E_PYSCF_KUKS_PBE_FULLPOL_CHAIN311 = -0.533376641201


@pytest.mark.slow
def test_kuks_gdf_multi_k_spin_polarized_matches_external_pyscf():
    """The genuinely spin-polarised (Da != Db) multi-k KUKS branch against
    external PySCF on the tight chain at (3,1,1): fully-polarised
    H2 (mult=3 per cell), pure PBE (convention-free). Pre-fix the home-cell
    XC shortcut sat +3.4e-4 from the anchor; post-fix -1.26e-4 (the
    documented XC grid-scheme floor). Gated at 4e-4."""
    system = _chain3d_system(multiplicity=3)
    basis = make_basis(system.unit_cell_molecule(), "sto-3g")
    u = run_kuks_periodic_gdf(system, basis, (3, 1, 1), functional="pbe")
    assert u.converged
    assert u.s_squared == pytest.approx(2.0, abs=1e-8)
    assert u.energy == pytest.approx(
        E_PYSCF_KUKS_PBE_FULLPOL_CHAIN311, abs=4e-4
    )


# External anchor: PySCF 2.13.1 KUKS(GDF) PBE, rocksalt LiH fcc primitive
# a = 7.72 bohr, STO-3G, kmesh (2,1,1), spin=0, conv_tol=1e-10 (out of
# process per CLAUDE.md §10, 2026-07-09; KRKS on the same fixture gives the
# identical value, confirming the spin-symmetric state). vibe-qc measured
# -8.140935209166: +2.1 mHa, inside the documented ultra-diffuse-Li/STO-3G
# shared-fitting floor (the Γ KRKS anchor in
# tests/test_krks_gdf_xc_density.py carries the same class at 6 mHa; the
# HF baseline on the Γ twin deviates 4.7 mHa).
E_PYSCF_KUKS_PBE_LIH_211 = -8.143033032724


@pytest.mark.slow
def test_kuks_gdf_multi_k_tight_ionic_matches_external_pyscf():
    """Multi-k KUKS on the tight ionic rocksalt LiH cell at (2,1,1) PBE --
    the cell class where the home-cell XC shortcut was worst (+0.31 Ha on
    the Γ twin pre-b3f74aa9). Also pins KUKS == KRKS(like-for-like) on the
    same run to 1e-12 (measured 2.3e-14)."""
    a = 7.72
    lat = 0.5 * a * np.array([[0.0, 1, 1], [1, 0, 1], [1, 1, 0]]).T
    system = PeriodicSystem(
        3, lat, [Atom(3, [0, 0, 0]), Atom(1, [0.5 * a, 0.5 * a, 0.5 * a])], 0, 1
    )
    basis = make_basis(system.unit_cell_molecule(), "sto-3g")
    u = run_kuks_periodic_gdf(system, basis, (2, 1, 1), functional="pbe")
    r = run_krks_periodic_gdf(
        system, basis, (2, 1, 1), functional="pbe",
        use_compcell=True, gdf_method="rsgdf",
    )
    assert u.converged and r.converged
    assert u.energy == pytest.approx(E_PYSCF_KUKS_PBE_LIH_211, abs=6e-3)
    assert u.energy == pytest.approx(r.energy, abs=1e-12)
