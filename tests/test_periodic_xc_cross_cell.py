"""Regression: periodic XC uses the full cross-cell (bra-image) density.

Pins the fix for the v0.14.0 P0 dense-crystal periodic-XC bug.
The shipped ``build_xc_periodic``
assembled the grid density with the bra AO anchored in the home cell,

    rho_code(r) = Sum_s  chi_0(r) . P(s) . chi_s(r),

which is NOT lattice-periodic and is pointwise wrong by O(1) in the bonding
region of dense crystals (it still integrates to N electrons, so an integral
check cannot see it). The physical periodic density sums BOTH AOs over cells,

    rho_full(r) = Sum_a Sum_s  chi_a(r) . P(s-a) . chi_s(r),

which restores ``rho(r) = rho(r + R)``. On MgO/STO-3G this is the difference
between vibe-qc's -270.070 and the sealed CRYSTAL23 / PySCF.pbc GDF -270.498
Ha/FU (the full-SCF parity ladder lives in
``examples/regression/crystal_parity/parity_mgo_{svwn,pbe,r2scan}_sto3g.py``).

This test exercises ``build_xc_periodic`` directly (no Ewald-J SCF) on a dense
MgO cell with a real multi-k cross-cell density, and checks the C++ E_xc

  * matches an independent NumPy cross-cell (rho_full) assembly, and
  * differs substantially from the home-bra-only (rho_code) value,

functional-resolved for LDA (rho-only), PBE (GGA, sigma) and r2SCAN (mGGA,
tau) -- so a revert of any one functional's density/Fock terms is caught.
"""
from __future__ import annotations

import numpy as np
import pytest
import scipy.linalg as sla

import vibeqc as vq
from vibeqc import _vibeqc_core as core
from vibeqc import monkhorst_pack
from vibeqc._vibeqc_core import (
    GridOptions,
    LatticeSumOptions,
    bloch_sum,
    build_xc_periodic,
    compute_kinetic_lattice,
    compute_overlap_lattice,
    direct_lattice_cells,
    real_space_density_from_kpoints,
)
from vibeqc.periodic_grid import build_periodic_becke_grid

_ANG = 1.0 / 0.529177210903
CUTOFF = 12.0


def _mgo():
    a = 4.21 * _ANG
    lattice = (a / 2.0) * np.array(
        [[0.0, 1.0, 1.0], [1.0, 0.0, 1.0], [1.0, 1.0, 0.0]]
    )
    atoms = [vq.Atom(12, [0.0, 0.0, 0.0]),
             vq.Atom(8, [a / 2.0, a / 2.0, a / 2.0])]
    system = vq.PeriodicSystem(3, lattice, atoms)
    basis = vq.BasisSet(system.unit_cell_molecule(), "sto-3g")
    return system, basis


def _cross_cell_density(system, basis, lat_opts):
    """Deterministic dense multi-k density from a kinetic-Hamiltonian fold
    (T C = S C eps). Physical enough to carry non-trivial P(g!=0) blocks and
    appreciable image overlap; needs no nuclear/Ewald terms."""
    kmesh = monkhorst_pack(system, [2, 2, 2], use_symmetry=False)
    kpts = list(kmesh.kpoints)
    S_lat = compute_overlap_lattice(basis, system, lat_opts)
    T_lat = compute_kinetic_lattice(basis, system, lat_opts)
    n_occ = system.n_electrons() // 2
    C_per_k = []
    for k in kpts:
        k_arr = np.asarray(k, dtype=float).reshape(3)
        S_k = np.asarray(bloch_sum(S_lat, k_arr))
        T_k = np.asarray(bloch_sum(T_lat, k_arr))
        S_k = 0.5 * (S_k + S_k.conj().T)
        T_k = 0.5 * (T_k + T_k.conj().T)
        _, C = sla.eigh(T_k, S_k)
        C_per_k.append(np.asarray(C, dtype=complex))
    cells = list(direct_lattice_cells(system, 2.0 * lat_opts.cutoff_bohr))
    D_real = real_space_density_from_kpoints(
        C_per_k, [n_occ] * len(kpts), kmesh, cells
    )
    return D_real


def _numpy_exc(system, basis, grid, D_real, functional, *, cross_cell):
    """Independent E_xc oracle assembling rho, grad-rho and tau over lattice
    cells. ``cross_cell=True`` sums bra AND ket over the active cells
    (rho_full, the physics); ``False`` anchors the bra at home (rho_code, the
    shipped bug). Mirrors the C++ rho/grad/tau assembly with NumPy."""
    pts = np.asarray(grid.points, dtype=float)
    w = np.asarray(grid.weights, dtype=float)
    npts = pts.shape[0]
    L = np.asarray(system.lattice, dtype=float)
    atoms = list(system.unit_cell)
    cells = list(D_real.cells)
    idx = [tuple(int(x) for x in np.asarray(c.index)) for c in cells]
    Pg = {idx[i]: np.asarray(D_real.blocks[i], dtype=float)
          for i in range(len(cells))}
    func = core.Functional(functional)
    kind = str(func.kind)
    need_grad = ("GGA" in kind) or ("MGGA" in kind)
    is_mgga = "MGGA" in kind

    chi_cache = {}

    def chi(i):
        if i in chi_cache:
            return chi_cache[i]
        R = L @ np.asarray(i, dtype=float)
        at = [core.Atom(int(a.Z),
                        [a.xyz[0] + R[0], a.xyz[1] + R[1], a.xyz[2] + R[2]])
              for a in atoms]
        bs = core.BasisSet(
            core.Molecule(at, system.charge, system.multiplicity), basis.name)
        if need_grad:
            v, dx, dy, dz = core.evaluate_ao_with_gradient(bs, pts)
            out = (np.asarray(v), np.asarray(dx), np.asarray(dy), np.asarray(dz))
        else:
            out = (np.asarray(core.evaluate_ao(bs, pts)), None, None, None)
        chi_cache[i] = out
        return out

    home = np.array([a.xyz for a in atoms], dtype=float)

    def near(i):
        if i == (0, 0, 0):
            return True
        R = L @ np.asarray(i, dtype=float)
        return any(np.linalg.norm(ti + R - tj) <= CUTOFF
                   for ti in home for tj in home)

    active = [i for i in idx if near(i)]
    bra_cells = active if cross_cell else [(0, 0, 0)]

    rho = np.zeros(npts)
    gx = np.zeros(npts); gy = np.zeros(npts); gz = np.zeros(npts)
    tau = np.zeros(npts)
    for a in bra_cells:
        cha, dax, day, daz = chi(a)
        for s in active:
            g = (s[0] - a[0], s[1] - a[1], s[2] - a[2])
            P = Pg.get(g)
            if P is None:
                continue
            chs, dsx, dsy, dsz = chi(s)
            chaP = cha @ P
            rho += np.einsum("pm,pm->p", chaP, chs)
            if need_grad:
                # grad-rho = P [d_chi_a . chi_s + chi_a . d_chi_s]
                gx += np.einsum("pm,pm->p", chaP, dsx)
                gy += np.einsum("pm,pm->p", chaP, dsy)
                gz += np.einsum("pm,pm->p", chaP, dsz)
                gx += np.einsum("pm,pm->p", dax @ P, chs)
                gy += np.einsum("pm,pm->p", day @ P, chs)
                gz += np.einsum("pm,pm->p", daz @ P, chs)
            if is_mgga:
                tau += np.einsum("pm,pm->p", dax @ P, dsx)
                tau += np.einsum("pm,pm->p", day @ P, dsy)
                tau += np.einsum("pm,pm->p", daz @ P, dsz)
    if is_mgga:
        tau *= 0.5
    rho = np.where(rho < 0.0, 0.0, rho)
    sigma = gx**2 + gy**2 + gz**2

    if is_mgga:
        exc = np.asarray(func.eval_unpolarised_mgga(rho, sigma, tau)[0])
    elif need_grad:
        exc = np.asarray(func.eval_unpolarised(rho, sigma)[0])
    else:
        exc = np.asarray(func.eval_unpolarised(rho, np.zeros(npts))[0])
    return float(w @ exc)


@pytest.mark.parametrize("functional", ["lda", "pbe", "r2scan"])
def test_build_xc_periodic_uses_cross_cell_density(functional):
    system, basis = _mgo()
    lat_opts = LatticeSumOptions()
    lat_opts.cutoff_bohr = CUTOFF
    lat_opts.nuclear_cutoff_bohr = CUTOFF
    # The grid below is built with image_radius=CUTOFF, so the cross-cell bra
    # screen must use the same partition reach (else build_xc_periodic falls
    # back to its 10-bohr default and the bra set differs from the oracle's).
    lat_opts.becke_image_radius_bohr = CUTOFF

    # Coarse but deterministic DFT grid (the cross-cell physics is grid-shape
    # independent; we compare C++ and NumPy on the SAME grid).
    gopts = GridOptions()
    gopts.n_radial = 35
    grid = build_periodic_becke_grid(system, grid_options=gopts,
                                     image_radius_bohr=CUTOFF)

    D_real = _cross_cell_density(system, basis, lat_opts)
    func = core.Functional(functional)
    xc = build_xc_periodic(basis, system, grid, func, D_real, lat_opts)

    e_full = _numpy_exc(system, basis, grid, D_real, functional, cross_cell=True)
    e_code = _numpy_exc(system, basis, grid, D_real, functional, cross_cell=False)

    # C++ build_xc_periodic must reproduce the cross-cell (rho_full) oracle...
    assert xc.e_xc == pytest.approx(e_full, abs=1e-7, rel=1e-7), (
        f"{functional}: C++ E_xc={xc.e_xc:.8f} != cross-cell oracle "
        f"{e_full:.8f} (home-bra value was {e_code:.8f})"
    )
    # ...and that value must differ substantially from the home-bra-only bug
    # (otherwise a revert to the shipped assembly would pass undetected).
    assert abs(e_full - e_code) > 5e-2, (
        f"{functional}: cross-cell and home-bra E_xc are too close "
        f"({e_full:.6f} vs {e_code:.6f}); test would not catch a regression"
    )


def test_cross_cell_density_is_lattice_periodic():
    """The smoking-gun invariant: the cross-cell density rho_full(r) is
    lattice-periodic (rho(r) == rho(r + a_i)); the shipped home-bra rho_code
    is not. Checked pointwise via the independent AO sum."""
    system, basis = _mgo()
    lat_opts = LatticeSumOptions()
    lat_opts.cutoff_bohr = CUTOFF
    lat_opts.nuclear_cutoff_bohr = CUTOFF
    D_real = _cross_cell_density(system, basis, lat_opts)

    L = np.asarray(system.lattice, dtype=float)
    atoms = list(system.unit_cell)
    cells = list(D_real.cells)
    idx = [tuple(int(x) for x in np.asarray(c.index)) for c in cells]
    Pg = {idx[i]: np.asarray(D_real.blocks[i], dtype=float)
          for i in range(len(cells))}
    home = np.array([a.xyz for a in atoms], dtype=float)

    def near(i):
        if i == (0, 0, 0):
            return True
        R = L @ np.asarray(i, dtype=float)
        return any(np.linalg.norm(ti + R - tj) <= CUTOFF
                   for ti in home for tj in home)

    active = [i for i in idx if near(i)]

    def chi(i, points):
        R = L @ np.asarray(i, dtype=float)
        at = [core.Atom(int(a.Z),
                        [a.xyz[0] + R[0], a.xyz[1] + R[1], a.xyz[2] + R[2]])
              for a in atoms]
        return np.asarray(core.evaluate_ao(
            core.BasisSet(core.Molecule(at, system.charge, system.multiplicity),
                          basis.name), points), dtype=float)

    def rho(points, cross_cell):
        out = np.zeros(points.shape[0])
        bras = active if cross_cell else [(0, 0, 0)]
        for a in bras:
            cha = chi(a, points)
            for s in active:
                g = (s[0] - a[0], s[1] - a[1], s[2] - a[2])
                P = Pg.get(g)
                if P is None:
                    continue
                out += np.einsum("pm,mn,pn->p", cha, P, chi(s, points))
        return out

    rng = np.random.default_rng(0)
    # Sample points inside the home Wigner-Seitz region (near the atoms).
    base = home[None, 1] + 0.6 * (rng.random((8, 3)) - 0.5) * 4.0
    for ai in (L[:, 0], L[:, 1], L[:, 2]):
        shifted = base + ai
        full_dev = np.max(np.abs(rho(base, True) - rho(shifted, True)))
        code_dev = np.max(np.abs(rho(base, False) - rho(shifted, False)))
        # The cross-cell density is lattice-periodic up to the finite-cutoff
        # truncation (~2e-6 at CUTOFF bohr); the shipped home-bra density
        # violates periodicity by O(1e-1..1e1) in the bonding region. They are
        # 3+ orders of magnitude apart, so the test cleanly separates them.
        assert full_dev < 1e-4, f"cross-cell rho not periodic: dev={full_dev:.2e}"
        assert code_dev > 1e3 * full_dev and code_dev > 1e-2, (
            f"home-bra rho not clearly aperiodic (code_dev={code_dev:.2e}, "
            f"full_dev={full_dev:.2e}); test not exercising the bug"
        )


@pytest.mark.parametrize("name", ["custom-periodic-xc-no-file", "sto-3g"])
@pytest.mark.parametrize("spin", [1, 2])
@pytest.mark.parametrize("functional", ["lda", "pbe", "r2scan"])
def test_periodic_xc_gradient_preserves_explicit_shells(name, spin, functional):
    """Translated image AOs must use the supplied basis, even with a library name."""
    lattice = np.array([[6., .3, .1], [.4, 8., .2], [.1, .2, 9.]])
    positions = np.array([[.1, .2, .3], [1.4, .3, .1]])

    def displaced(delta):
        atoms = positions.copy()
        atoms[1, 0] += delta
        system = vq.PeriodicSystem(3, lattice, [vq.Atom(1, p) for p in atoms])
        basis = vq.BasisSet(system.unit_cell_molecule(), [
            vq.ShellInfo(i, 0, False, [exponent], [1.], p)
            for i, (exponent, p) in enumerate(zip([.7, 1.1], atoms))
        ], name, False)
        return system, basis

    system, basis = displaced(0.)
    opts = LatticeSumOptions()
    opts.cutoff_bohr = 6.2
    opts.becke_image_radius_bohr = 10.
    cells = list(direct_lattice_cells(system, opts.cutoff_bohr))
    assert len(cells) > 1
    matrix = np.array([[1., .2], [.2, .4]])

    def density(scale):
        return core.make_lattice_matrix_set(2, cells, [
            scale*matrix if np.all(np.asarray(cell.index) == 0) else np.zeros((2, 2))
            for cell in cells
        ])

    grid_opts = GridOptions()
    grid_opts.n_radial = 16
    grid_opts.n_theta = 5
    grid_opts.n_phi = 8
    grid = build_periodic_becke_grid(system, grid_options=grid_opts)
    func = vq.Functional(functional, spin)
    if spin == 1:
        densities = (density(1.),)
        build = core.build_xc_periodic
        differentiate = core.xc_lattice_gradient_contribution
    else:
        densities = (density(.65), density(.35))
        build = core.build_xc_periodic_uks
        differentiate = core.xc_lattice_gradient_contribution_uks
    gradient = np.asarray(differentiate(basis, system, grid, func, *densities, opts))
    # Keep both the density matrix and quadrature fixed: this differentiates
    # precisely the AO-image contribution, without a moving-grid correction.
    h = 2e-5
    energies = []
    for delta in [-h, h]:
        shifted_system, shifted_basis = displaced(delta)
        energies.append(build(shifted_basis, shifted_system, grid, func, *densities, opts).e_xc)
    finite_difference = (energies[1]-energies[0])/(2*h)
    assert np.all(np.isfinite(gradient))
    assert gradient[1, 0] == pytest.approx(finite_difference, rel=0, abs=2e-7)
