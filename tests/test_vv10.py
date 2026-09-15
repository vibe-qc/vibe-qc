"""VV10 nonlocal correlation (Vydrov & Van Voorhis, *J. Chem. Phys.*
**133**, 244103 (2010)) — the nonlocal-correlation kernel that completes
the ωB97X-V / ωB97M-V / standalone-VV10 functionals.

libxc supplies only the *semilocal* part of a VV10-paired functional plus
the (b, C) parameters via ``xc_nlc_coef``; vibe-qc evaluates the nonlocal
double-grid integral itself (``vibeqc.compute_vv10``, ``cpp/src/vv10.cpp``)
and folds the energy + self-consistent potential into the RKS / UKS SCF.

Pins:

  1. The bare kernel ``compute_vv10`` reproduces PySCF's reference
     ``dft.numint._vv10nlc`` — energy *and* per-point potential — on a
     shared grid + density to ~1e-9 (the formulas are identical; this
     pins that vibe-qc's independent C++ port matches bit-for-bit).
  2. The analytic VV10 potential (v_rho, v_sigma) is the finite-difference
     derivative of the VV10 energy on a synthetic grid (self-consistency
     of the energy/potential pair, independent of any reference code).
  3. ``Functional("wb97x-v")`` / ``"vv10"`` / ``"wb97m-v"`` flag VV10 and
     report the right (b, C); plain ``"wb97x"`` does not.
  4. End-to-end: ωB97X-V through ``run_rks`` now *includes* VV10 (the bare
     alias is no longer a silent semilocal-only lie); the total tracks
     PySCF's self-consistent ωB97X-V to grid accuracy, and the VV10 piece
     is a real, nonzero, correctly-signed contribution.
"""

from __future__ import annotations

import numpy as np
import pytest

from vibeqc import (
    Atom,
    BasisSet,
    Functional,
    Molecule,
    RKSOptions,
    UKSOptions,
    compute_vv10,
    run_rks,
    run_uks,
)

from .conftest import GEOMETRIES


# VV10 parameters libxc reports for each paired functional.
WB97XV_B, WB97XV_C = 6.0, 0.01
VV10_B, VV10_C = 5.9, 0.0093


def test_functional_vv10_flags():
    """needs_vv10 / vv10_b / vv10_C are set for VV10-paired functionals
    and clear for the plain (non-VV10) range-separated parent."""
    f = Functional("wb97x-v")
    assert f.needs_vv10 is True
    assert f.vv10_b == pytest.approx(WB97XV_B, abs=1e-10)
    assert f.vv10_C == pytest.approx(WB97XV_C, abs=1e-10)

    vv10 = Functional("vv10")
    assert vv10.needs_vv10 is True
    assert vv10.vv10_b == pytest.approx(VV10_B, abs=1e-10)
    assert vv10.vv10_C == pytest.approx(VV10_C, abs=1e-10)

    # The plain ωB97X (no "-V") has no nonlocal component.
    plain = Functional("wb97x")
    assert plain.needs_vv10 is False
    assert plain.vv10_b == 0.0
    assert plain.vv10_C == 0.0


def _numpy_vv10_reference(coords, weights, rho, sigma, b, C):
    """Independent NumPy evaluation of the VV10 energy + potential —
    the same formulas as cpp/src/vv10.cpp, written from scratch here so
    the C++ port is checked against a second implementation, not itself.
    Returns (energy, v_rho, v_sigma)."""
    thresh = 1e-8
    Pi = np.pi
    Pi43 = 4.0 * Pi / 3.0
    Kvv = b * 1.5 * Pi * (9.0 * Pi) ** (-1.0 / 6.0)
    Beta = (3.0 / (b * b)) ** 0.75 / 32.0

    n = rho.size
    v_rho = np.zeros(n)
    v_sigma = np.zeros(n)
    act = np.where(rho >= thresh)[0]
    r = rho[act]
    s = sigma[act]
    w = weights[act]
    xyz = coords[act]
    w0sq = C * (s / r**2) ** 2
    W0 = np.sqrt(w0sq + Pi43 * r)
    K = Kvv * r ** (1.0 / 6.0)
    dW0dR = (0.5 * Pi43 * r - 2.0 * w0sq) / W0
    dKdR = K / 6.0
    dW0dG = C * s / (r**3 * W0)
    rw = r * w

    energy = 0.0
    for ii in range(act.size):
        dx = xyz[:, 0] - xyz[ii, 0]
        dy = xyz[:, 1] - xyz[ii, 1]
        dz = xyz[:, 2] - xyz[ii, 2]
        R2 = dx * dx + dy * dy + dz * dz
        g = W0[ii] * R2 + K[ii]
        gp = W0 * R2 + K
        gt = g + gp
        T = rw / (g * gp * gt)
        F = T.sum()
        Tk = T * (1.0 / g + 1.0 / gt)
        U = Tk.sum()
        W = (Tk * R2).sum()
        F *= -1.5
        energy += w[ii] * r[ii] * (Beta + 0.5 * F)
        v_rho[act[ii]] = Beta + F + 1.5 * (U * dKdR[ii] + W * dW0dR[ii])
        v_sigma[act[ii]] = 1.5 * W * dW0dG[ii]
    return energy, v_rho, v_sigma


def test_vv10_kernel_matches_numpy_reference():
    """compute_vv10 (C++) == an independent NumPy reimplementation of the
    same Vydrov-Van Voorhis formulas on a small synthetic grid."""
    rng = np.random.default_rng(20260620)
    n = 40
    coords = rng.uniform(-3.0, 3.0, size=(n, 3))
    weights = rng.uniform(0.05, 0.5, size=n)
    rho = rng.uniform(1e-3, 0.8, size=n)
    # Build a plausible sigma = |grad rho|^2 (positive).
    grad = rng.uniform(-0.4, 0.4, size=(n, 3))
    sigma = (grad**2).sum(axis=1)

    e_c, vr_c, vs_c = compute_vv10(coords, weights, rho, sigma, VV10_B, VV10_C)
    e_np, vr_np, vs_np = _numpy_vv10_reference(
        coords, weights, rho, sigma, VV10_B, VV10_C)

    assert e_c == pytest.approx(e_np, rel=1e-11, abs=1e-13)
    np.testing.assert_allclose(vr_c, vr_np, rtol=1e-10, atol=1e-12)
    np.testing.assert_allclose(vs_c, vs_np, rtol=1e-10, atol=1e-12)


def test_vv10_potential_is_finite_difference_of_energy():
    """The analytic VV10 potential is the derivative of the VV10 energy:
    dE_nl/dρ_k = w_k·v_rho[k] and dE_nl/dσ_k = w_k·v_sigma[k] (σ held
    fixed for the ρ test and vice-versa). Pure self-check, no reference
    code — it pins that the self-consistent SCF potential is exact."""
    rng = np.random.default_rng(7)
    n = 25
    coords = rng.uniform(-2.5, 2.5, size=(n, 3))
    weights = rng.uniform(0.1, 0.4, size=n)
    rho = rng.uniform(0.05, 0.7, size=n)
    grad = rng.uniform(-0.3, 0.3, size=(n, 3))
    sigma = (grad**2).sum(axis=1)

    _, v_rho, v_sigma = compute_vv10(coords, weights, rho, sigma,
                                     VV10_B, VV10_C)

    h = 1e-6
    for k in (3, 11, 20):
        rp = rho.copy(); rp[k] += h
        rm = rho.copy(); rm[k] -= h
        e_p, _, _ = compute_vv10(coords, weights, rp, sigma, VV10_B, VV10_C)
        e_m, _, _ = compute_vv10(coords, weights, rm, sigma, VV10_B, VV10_C)
        fd = (e_p - e_m) / (2 * h)
        assert fd == pytest.approx(weights[k] * v_rho[k], rel=1e-5, abs=1e-9)

        sp = sigma.copy(); sp[k] += h
        sm = sigma.copy(); sm[k] -= h
        e_p, _, _ = compute_vv10(coords, weights, rho, sp, VV10_B, VV10_C)
        e_m, _, _ = compute_vv10(coords, weights, rho, sm, VV10_B, VV10_C)
        fd = (e_p - e_m) / (2 * h)
        assert fd == pytest.approx(weights[k] * v_sigma[k], rel=1e-5, abs=1e-9)


def test_vv10_kernel_matches_pyscf_on_shared_grid():
    """compute_vv10 reproduces PySCF's reference ``_vv10nlc`` — energy and
    both potential channels — on the *same* grid and density, pinning the
    independent C++ port against the established implementation."""
    pyscf = pytest.importorskip("pyscf")
    from pyscf import gto, dft
    from pyscf.dft import numint

    atoms = GEOMETRIES["H2O"]
    mol = gto.Mole()
    mol.unit = "Bohr"
    mol.atom = [[Z, tuple(xyz)] for Z, xyz in atoms]
    mol.basis = "cc-pvdz"
    mol.verbose = 0
    mol.build()

    # A converged-ish density to evaluate the kernel on (PBE is fine — the
    # kernel comparison only needs the *same* ρ on both sides).
    mf = dft.RKS(mol, xc="pbe")
    mf.conv_tol = 1e-10
    mf.kernel()
    dm = mf.make_rdm1()

    grids = mf.grids
    ao = numint.eval_ao(mol, grids.coords, deriv=1)
    rho = numint.eval_rho(mol, ao, dm, xctype="GGA")  # (4, n): den + grad

    nlc_pars = (WB97XV_B, WB97XV_C)
    exc_ps, vxc_ps = numint._vv10nlc(
        rho, grids.coords, rho, grids.weights, grids.coords, nlc_pars)
    e_ps = float((grids.weights * rho[0] * exc_ps).sum())

    sigma = rho[1] ** 2 + rho[2] ** 2 + rho[3] ** 2
    e_vq, vr_vq, vs_vq = compute_vv10(
        grids.coords, grids.weights, rho[0], sigma, WB97XV_B, WB97XV_C)

    assert e_vq == pytest.approx(e_ps, rel=1e-8, abs=1e-9), (
        f"VV10 E_nl vibeqc={e_vq:.10f} pyscf={e_ps:.10f}")
    np.testing.assert_allclose(vr_vq, vxc_ps[0], rtol=1e-7, atol=1e-8)
    np.testing.assert_allclose(vs_vq, vxc_ps[1], rtol=1e-7, atol=1e-8)


def _atoms_to_mol_basis(atoms_bohr, basis_name):
    mol = Molecule([Atom(Z, list(xyz)) for Z, xyz in atoms_bohr])
    return mol, BasisSet(mol, basis_name)


def test_wb97xv_scf_preserves_pairs_across_ao_batches():
    """The global VV10 double sum must survive AO-projection batching.

    This one-atom product grid has 6,120 points, so it crosses the native
    4,096-point AO batch boundary. The values were captured on the dense
    pre-fix path; evaluating VV10 independently per slice would drop all
    cross-slice pairs and fail both pins.

    The pins are dense-grid values, so the VV10 grid is pinned to the XC
    grid explicitly (``vv10_grid_factor = 1.0``). Since 6c96a7c66 the
    default factor is 3.0, which coarsens this grid to 5 x 6 x 12 = 360
    points: below the batch boundary, so the batch-crossing premise would
    not even be exercised, and 1.66e-4 Ha away from the pins (#675).
    """
    from vibeqc import _vibeqc_core as core

    mol = Molecule([Atom(2, [0.0, 0.0, 0.0])])
    basis = BasisSet(mol, "sto-3g")
    opts = RKSOptions()
    opts.functional = "wb97x-v"
    opts.density_fit = False
    opts.grid.angular = "product"
    opts.grid.n_radial = 10
    opts.grid.n_theta = 17
    opts.grid.n_phi = 36
    opts.grid.vv10_grid_factor = 1.0  # dense path: VV10 on the XC grid
    opts.conv_tol_energy = 1e-12
    opts.conv_tol_grad = 1e-9

    n_points = opts.grid.n_radial * opts.grid.n_theta * opts.grid.n_phi
    assert n_points > core.MOLECULAR_XC_GRID_BATCH_SIZE
    result = run_rks(mol, basis, opts)

    assert result.converged
    assert result.energy == pytest.approx(-2.841931319453023, abs=1e-10)
    assert result.e_xc == pytest.approx(-0.6475170588567877, abs=1e-10)


def test_uks_wb97xv_preserves_spin_chain_rule_across_ao_batches():
    """Closed-shell UKS must match RKS when the VV10 grid crosses a batch.

    Dense-path pins, so ``vv10_grid_factor = 1.0`` as in the RKS twin
    (#675)."""
    from vibeqc import _vibeqc_core as core

    mol = Molecule([Atom(2, [0.0, 0.0, 0.0])], 0, 1)
    basis = BasisSet(mol, "sto-3g")
    opts = UKSOptions()
    opts.functional = "wb97x-v"
    opts.density_fit = False
    opts.grid.angular = "product"
    opts.grid.n_radial = 10
    opts.grid.n_theta = 17
    opts.grid.n_phi = 36
    opts.grid.vv10_grid_factor = 1.0  # dense path: VV10 on the XC grid
    opts.conv_tol_energy = 1e-12
    opts.conv_tol_grad = 1e-9

    n_points = opts.grid.n_radial * opts.grid.n_theta * opts.grid.n_phi
    assert n_points > core.MOLECULAR_XC_GRID_BATCH_SIZE
    result = run_uks(mol, basis, opts)

    assert result.converged
    assert result.energy == pytest.approx(-2.841931319453023, abs=1e-10)
    assert result.e_xc == pytest.approx(-0.6475170588567877, abs=1e-10)


def test_wb97xv_default_coarse_vv10_grid_is_pinned_and_near_dense():
    """The shipped default (``vv10_grid_factor = 3.0``) evaluates VV10 on a
    separate coarser grid, following the Vydrov and Van Voorhis 2010
    implementation practice (J. Chem. Phys. 133, 244103, Sec. II and III:
    the nonlocal term is evaluated on a coarser grid than the semilocal
    functional because E_c^nl is far less sensitive to grid fineness).

    Coverage the dense-path tests above deliberately do not give (#675):
    the default is 3.0, RKS and closed-shell UKS agree on it, both are
    pinned, and the coarse result sits at the grid-error scale from the
    dense pins (1.66e-4 Ha on this intentionally tiny 10-radial grid,
    where 5 x 6 x 12 = 360 coarse points replace 6,120) rather than
    reproducing them, which would mean the coarse grid was silently
    disabled."""
    dense_energy = -2.841931319453023
    dense_e_xc = -0.6475170588567877

    def make(cls, spin_args):
        mol = Molecule([Atom(2, [0.0, 0.0, 0.0])], *spin_args)
        basis = BasisSet(mol, "sto-3g")
        opts = cls()
        opts.functional = "wb97x-v"
        opts.density_fit = False
        opts.grid.angular = "product"
        opts.grid.n_radial = 10
        opts.grid.n_theta = 17
        opts.grid.n_phi = 36
        opts.conv_tol_energy = 1e-12
        opts.conv_tol_grad = 1e-9
        assert opts.grid.vv10_grid_factor == 3.0
        return mol, basis, opts

    mol, basis, opts = make(RKSOptions, ())
    rks = run_rks(mol, basis, opts)
    molu, basisu, optsu = make(UKSOptions, (0, 1))
    uks = run_uks(molu, basisu, optsu)

    for result in (rks, uks):
        assert result.converged
        assert result.energy == pytest.approx(-2.841764943333874, abs=1e-10)
        assert result.e_xc == pytest.approx(-0.647350682737639, abs=1e-10)
        shift = result.energy - dense_energy
        assert 1e-6 < abs(shift) < 5e-4, shift
        assert result.e_xc - dense_e_xc == pytest.approx(shift, abs=1e-8)
    assert rks.energy == pytest.approx(uks.energy, abs=1e-12)


def test_wb97xv_includes_vv10_and_tracks_pyscf():
    """End-to-end ωB97X-V SCF: the bare ``functional="wb97x-v"`` now
    computes the VV10 nonlocal term (no longer a silent semilocal lie),
    and the total tracks PySCF's self-consistent ωB97X-V to grid
    accuracy. Also confirms VV10 makes a real difference vs the
    semilocal-only (ωB97X) energy."""
    pyscf = pytest.importorskip("pyscf")
    from pyscf import gto, dft

    atoms = GEOMETRIES["H2O"]
    mol, basis = _atoms_to_mol_basis(atoms, "cc-pvdz")
    opts = RKSOptions()
    opts.functional = "wb97x-v"
    opts.density_fit = False  # RSH needs the direct path
    opts.conv_tol_energy = 1e-10
    opts.conv_tol_grad = 1e-7
    res = run_rks(mol, basis, opts)
    assert res.converged

    pm = gto.Mole()
    pm.unit = "Bohr"
    pm.atom = [[Z, tuple(xyz)] for Z, xyz in atoms]
    pm.basis = "cc-pvdz"
    pm.verbose = 0
    pm.build()
    mf = dft.RKS(pm, xc="wb97x-v")
    mf.nlc = "VV10"
    mf.grids.level = 3
    mf.nlcgrids.level = 3
    mf.conv_tol = 1e-10
    e_pyscf = mf.kernel()
    assert mf.converged

    # Cross-code total: vibe-qc's converged grid and PySCF's level-3 grid
    # agree to sub-µHa on this case (observed ~0.04 µHa), so the complete
    # self-consistent ωB97X-V total — semilocal RSH + VV10 — reproduces
    # PySCF essentially exactly. The 5e-5 tolerance leaves headroom for
    # grid-default / BLAS-summation drift across machines; the VV10 kernel
    # itself is pinned tightly by the shared-grid test above.
    assert res.energy == pytest.approx(e_pyscf, abs=5e-5), (
        f"vibeqc ωB97X-V = {res.energy:.8f}, PySCF = {e_pyscf:.8f}, "
        f"gap = {(res.energy - e_pyscf) * 1e6:+.2f} µHa")

    # VV10 must actually change the energy: compare to semilocal-only ωB97X.
    opts_sl = RKSOptions()
    opts_sl.functional = "wb97x"
    opts_sl.density_fit = False
    opts_sl.conv_tol_energy = 1e-10
    res_sl = run_rks(mol, basis, opts_sl)
    # The VV10 contribution is small but non-negligible (a few mHa for a
    # molecule this size) — not zero, which is what the old silent-omit bug
    # effectively delivered.
    assert abs(res.energy - res_sl.energy) > 1e-4
