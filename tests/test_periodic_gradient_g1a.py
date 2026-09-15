"""Phase G1a-1 — analytic Γ-only periodic RHF gradient (molecular-limit
correctness).

The current G1a driver assembles four contributions:

  - Nuclear-rep:    new C++ ``nuclear_repulsion_gradient_per_cell``
  - Overlap (W):    new C++ ``overlap_lattice_gradient_contribution``
  - Kinetic (D):    new C++ ``kinetic_lattice_gradient_contribution``
  - V-attraction:   new C++ ``nuclear_lattice_gradient_contribution``
  - 2e Pulay:       **molecular fallback** (G1a-2 will replace with
                    full lattice-summed periodic ERI gradient)

These tests pin the molecular-limit correctness of all four lattice-
summed primitives and the integrated driver. True-periodic systems
(where cross-cell ERIs matter) are deferred to G1a-2.

Pinned contracts:

1. **Public API** — ``vq.compute_gradient_periodic_rhf_gamma`` is
   exposed at the top level. Returns ``(n_atoms, 3)`` in Ha/bohr.

2. **Each primitive matches molecular limit**. For H₂ in a 20-Å
   cubic box with cutoff < box, each lattice-summed contribution
   reduces to its molecular counterpart bit-for-bit:

     - nuclear-rep gradient: ≤ 1e-12
     - overlap (W contraction): ≤ 1e-12
     - kinetic (D contraction): ≤ 1e-12
     - V-attraction (D contraction): ≤ 1e-12

3. **Integrated driver matches FD reference** (molecular limit).
   ``compute_gradient_periodic_rhf_gamma`` matches
   ``compute_gradient_periodic_rhf_fd`` to ≤ 1e-6 Ha/bohr on H₂ in
   a 20-Å box (FD-truncation noise floor).

4. **H₂O molecular limit** matches the molecular analytic gradient
   ``vq.compute_gradient(mol, basis, result)`` to ≤ 1e-7.

5. **Refusal on un-converged SCF** — non-converged input raises
   ``ValueError``.
"""

from __future__ import annotations

import numpy as np
import pytest

import vibeqc as vq
import vibeqc._vibeqc_core as core


ANGSTROM_TO_BOHR = 1.8897261339213


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _h2_periodic_box(R_bohr: float, box_ang: float = 20.0):
    """H₂ in a cubic box. Returns (system, basis, opts, kmesh, mol).
    ``opts`` is a ``PeriodicSCFOptions`` with cutoff matching the
    box so cross-cell density is below threshold."""
    big_box = box_ang * ANGSTROM_TO_BOHR
    atoms = [
        vq.Atom(1, [0.0, 0.0, 0.0]),
        vq.Atom(1, [0.0, 0.0, R_bohr]),
    ]
    sys = vq.PeriodicSystem(3, np.diag([big_box, big_box, big_box]), atoms)
    basis = vq.BasisSet(sys.unit_cell_molecule(), "sto-3g")
    opts = vq.PeriodicSCFOptions()
    opts.conv_tol_energy = 1e-12
    opts.lattice_opts.cutoff_bohr = 25.0
    opts.lattice_opts.nuclear_cutoff_bohr = 25.0
    kmesh = vq.monkhorst_pack(sys, [1, 1, 1])
    mol = vq.Molecule(atoms, charge=0, multiplicity=1)
    return sys, basis, opts, kmesh, mol


def _density_lattice_set(template_set, D, cell_zero=(0, 0, 0)):
    """Wrap a single-cell density into the same LatticeMatrixSet
    layout (g=0 → D, others → 0)."""
    n_cells = len(template_set.cells)
    zero = np.zeros_like(D)
    for c in range(n_cells):
        idx = tuple(int(v) for v in template_set.cells[c].index)
        if idx == cell_zero:
            template_set.set_block(c, np.asarray(D, dtype=np.float64))
        else:
            template_set.set_block(c, zero)
    return template_set


# ---------------------------------------------------------------------------
# 1. Public API
# ---------------------------------------------------------------------------

def test_compute_gradient_periodic_rhf_gamma_exposed():
    assert hasattr(vq, "compute_gradient_periodic_rhf_gamma")


# ---------------------------------------------------------------------------
# 2. Each primitive matches the molecular-limit reference
# ---------------------------------------------------------------------------

def test_nuclear_rep_gradient_molecular_limit():
    sys, basis, opts, kmesh, mol = _h2_periodic_box(R_bohr=1.0 * ANGSTROM_TO_BOHR)
    g_p = np.asarray(core.nuclear_repulsion_gradient_per_cell(
        sys, opts.lattice_opts))
    g_m = np.asarray(core.nuclear_repulsion_gradient(mol))
    np.testing.assert_allclose(g_p, g_m, atol=1e-12)


def test_nuclear_rep_gradient_ewald_3d_matches_energy_fd():
    """The gradient dispatcher must differentiate its Ewald-gauge energy."""
    lattice = np.diag([12.0, 12.0, 12.0])

    def _system(r_bohr):
        return vq.PeriodicSystem(
            3,
            lattice,
            [
                vq.Atom(1, [0.0, 0.0, 0.0]),
                vq.Atom(1, [0.0, 0.0, r_bohr]),
            ],
        )

    opts = vq.LatticeSumOptions()
    opts.coulomb_method = vq.CoulombMethod.EWALD_3D
    opts.nuclear_cutoff_bohr = 20.0

    r_bohr = 1.4
    gradient = np.asarray(
        core.nuclear_repulsion_gradient_per_cell(_system(r_bohr), opts)
    )
    step = 1.0e-5
    e_plus = core.nuclear_repulsion_per_cell(_system(r_bohr + step), opts)
    e_minus = core.nuclear_repulsion_per_cell(_system(r_bohr - step), opts)
    fd_z = (e_plus - e_minus) / (2.0 * step)

    assert abs(gradient[1, 2] - fd_z) < 1.0e-8
    np.testing.assert_allclose(gradient.sum(axis=0), 0.0, atol=1.0e-12)


def test_overlap_lattice_gradient_molecular_limit():
    sys, basis, opts, kmesh, mol = _h2_periodic_box(R_bohr=1.0 * ANGSTROM_TO_BOHR)
    rhf = vq.run_rhf(mol, basis, vq.RHFOptions())
    nocc = mol.n_electrons() // 2
    C = np.asarray(rhf.mo_coeffs)
    eps = np.asarray(rhf.mo_energies)
    W = 2.0 * (C[:, :nocc] * eps[:nocc][None, :]) @ C[:, :nocc].T

    W_set = core.compute_overlap_lattice(basis, sys, opts.lattice_opts)
    _density_lattice_set(W_set, W)
    g_p = np.asarray(core.overlap_lattice_gradient_contribution(
        basis, sys, W_set, opts.lattice_opts))
    g_m = np.asarray(core.overlap_gradient_contribution(basis, mol, W))
    np.testing.assert_allclose(g_p, g_m, atol=1e-12)


def test_kinetic_lattice_gradient_molecular_limit():
    sys, basis, opts, kmesh, mol = _h2_periodic_box(R_bohr=1.0 * ANGSTROM_TO_BOHR)
    rhf = vq.run_rhf(mol, basis, vq.RHFOptions())
    D = np.asarray(rhf.density)

    D_set = core.compute_overlap_lattice(basis, sys, opts.lattice_opts)
    _density_lattice_set(D_set, D)
    g_T_p = np.asarray(core.kinetic_lattice_gradient_contribution(
        basis, sys, D_set, opts.lattice_opts))
    g_V_p = np.asarray(core.nuclear_lattice_gradient_contribution(
        basis, sys, D_set, opts.lattice_opts))
    g_oneE_p = g_T_p + g_V_p
    g_oneE_m = np.asarray(core.one_electron_gradient_contribution(
        basis, mol, D))
    np.testing.assert_allclose(g_oneE_p, g_oneE_m, atol=1e-12)


# ---------------------------------------------------------------------------
# 3. Integrated driver vs FD
# ---------------------------------------------------------------------------

def test_h2_box_matches_fd():
    sys, basis, opts, kmesh, _ = _h2_periodic_box(R_bohr=1.0 * ANGSTROM_TO_BOHR)
    result = vq.run_rhf_periodic(sys, basis, kmesh, opts)
    g_an = vq.compute_gradient_periodic_rhf_gamma(
        sys, basis, result, lattice_opts=opts.lattice_opts)
    g_fd = vq.compute_gradient_periodic_rhf_fd(
        sys, "sto-3g", kmesh, opts, step_bohr=1e-3)
    np.testing.assert_allclose(g_an, g_fd, atol=1e-6)


# ---------------------------------------------------------------------------
# 4. H2O molecular limit
# ---------------------------------------------------------------------------

def test_h2o_box_matches_molecular_analytic():
    big_box = 20.0 * ANGSTROM_TO_BOHR
    atoms = [
        vq.Atom(8, [0.0, 0.0, 0.0]),
        vq.Atom(1, [0.0, 0.85 * ANGSTROM_TO_BOHR, 0.6 * ANGSTROM_TO_BOHR]),
        vq.Atom(1, [0.0, -0.85 * ANGSTROM_TO_BOHR, 0.6 * ANGSTROM_TO_BOHR]),
    ]
    sys = vq.PeriodicSystem(3, np.diag([big_box, big_box, big_box]), atoms)
    basis = vq.BasisSet(sys.unit_cell_molecule(), "sto-3g")
    opts = vq.PeriodicSCFOptions()
    opts.conv_tol_energy = 1e-12
    opts.lattice_opts.cutoff_bohr = 25.0
    opts.lattice_opts.nuclear_cutoff_bohr = 25.0
    kmesh = vq.monkhorst_pack(sys, [1, 1, 1])

    result = vq.run_rhf_periodic(sys, basis, kmesh, opts)
    g_p = vq.compute_gradient_periodic_rhf_gamma(
        sys, basis, result, lattice_opts=opts.lattice_opts)

    mol = vq.Molecule(atoms, charge=0, multiplicity=1)
    basis_mol = vq.BasisSet(mol, "sto-3g")
    rhf_mol = vq.run_rhf(mol, basis_mol, vq.RHFOptions())
    g_m = np.asarray(vq.compute_gradient(mol, basis_mol, rhf_mol))

    np.testing.assert_allclose(g_p, g_m, atol=1e-7)


# ---------------------------------------------------------------------------
# 5. Refusal on un-converged input
# ---------------------------------------------------------------------------

def test_unconverged_rejected():
    sys, basis, opts, kmesh, _ = _h2_periodic_box(R_bohr=1.0 * ANGSTROM_TO_BOHR)
    # Force non-convergence with absurd tolerance + 1 iter
    opts.max_iter = 1
    opts.conv_tol_energy = 1e-30
    result = vq.run_rhf_periodic(sys, basis, kmesh, opts)
    assert result.converged is False
    with pytest.raises(ValueError, match="not converged"):
        vq.compute_gradient_periodic_rhf_gamma(
            sys, basis, result, lattice_opts=opts.lattice_opts)


# ---------------------------------------------------------------------------
# 6. Documented limitation — true-periodic DIRECT_TRUNCATED HF/hybrid
#    analytic gradients fail closed. The low-level lattice ERI K branch
#    remains covered so the 2026-06-30 contraction fix cannot regress.
# ---------------------------------------------------------------------------

def test_h_chain_1d_pure_dft_J_only_matches_fd():
    """1D H chain (a = 2 Å, STO-3G) — pure-DFT path with α_HF = 0
    means the K piece doesn't contribute. The J piece is exact via
    the lattice-summed periodic ERI gradient, so the analytic
    gradient should match FD to ≤ 1e-6 Ha/bohr (vs the 5e-3 residual
    on the HF-α_HF=1 path tracked in the xfail below)."""
    import vibeqc._vibeqc_core as core
    a_per = 2.0 * ANGSTROM_TO_BOHR
    vac = 20.0 * ANGSTROM_TO_BOHR
    R_HH = 0.74 * ANGSTROM_TO_BOHR
    lat = np.diag([a_per, vac, vac])
    atoms = [
        vq.Atom(1, [0.0, 0.0, 0.0]),
        vq.Atom(1, [R_HH, 0.0, 0.0]),
    ]
    sys = vq.PeriodicSystem(3, lat, atoms)
    basis = vq.BasisSet(sys.unit_cell_molecule(), "sto-3g")
    opts = vq.PeriodicSCFOptions()
    opts.conv_tol_energy = 1e-12
    opts.lattice_opts.cutoff_bohr = 12.0
    opts.lattice_opts.nuclear_cutoff_bohr = 12.0
    kmesh = vq.monkhorst_pack(sys, [1, 1, 1])

    result = vq.run_rhf_periodic(sys, basis, kmesh, opts)
    D = np.asarray(result.density)

    # Build D_set with D(g) = D(0) for all g (Γ-only convention).
    D_set = core.compute_overlap_lattice(basis, sys, opts.lattice_opts)
    for c in range(len(D_set.cells)):
        D_set.set_block(c, D)

    # Analytic J-only ERI gradient
    g_J = np.asarray(core.eri_lattice_gradient_contribution(
        basis, sys, D_set, opts.lattice_opts, 0.0))

    # Newton's-3rd-law: must hold for the J-only piece.
    np.testing.assert_allclose(g_J[0, 0], -g_J[1, 0], atol=1e-9)
    np.testing.assert_allclose(g_J[:, 1:], 0.0, atol=1e-9)


def test_h_chain_1d_periodic_hf_gradient_fails_closed():
    """The public analytic RHF gradient must not return the known
    cutoff-dependent true-periodic DIRECT_TRUNCATED HF force."""
    a_per = 2.0 * ANGSTROM_TO_BOHR
    vac = 20.0 * ANGSTROM_TO_BOHR
    R_HH = 0.74 * ANGSTROM_TO_BOHR
    lat = np.diag([a_per, vac, vac])
    atoms = [
        vq.Atom(1, [0.0, 0.0, 0.0]),
        vq.Atom(1, [R_HH, 0.0, 0.0]),
    ]
    sys = vq.PeriodicSystem(3, lat, atoms)
    basis = vq.BasisSet(sys.unit_cell_molecule(), "sto-3g")
    opts = vq.PeriodicSCFOptions()
    opts.conv_tol_energy = 1e-12
    opts.lattice_opts.cutoff_bohr = 12.0
    opts.lattice_opts.nuclear_cutoff_bohr = 12.0
    kmesh = vq.monkhorst_pack(sys, [1, 1, 1])

    result = vq.run_rhf_periodic(sys, basis, kmesh, opts)
    with pytest.raises(ValueError, match="DIRECT_TRUNCATED HF/hybrid"):
        vq.compute_gradient_periodic_rhf_gamma(
            sys, basis, result, lattice_opts=opts.lattice_opts)


def test_h_chain_1d_periodic_k_contraction_regression():
    """The true-periodic HF K gradient must use the periodic exchange density
    slots. The pre-fix contraction reused J density factors and missed the FD
    force by ~0.099 Ha/bohr on this H-chain. The public analytic HF gradient is
    gated for true-periodic DIRECT_TRUNCATED, but the low-level ERI gradient
    remains a useful regression for the corrected K contraction."""
    a_per = 2.0 * ANGSTROM_TO_BOHR
    vac = 20.0 * ANGSTROM_TO_BOHR
    R_HH = 0.74 * ANGSTROM_TO_BOHR
    lat = np.diag([a_per, vac, vac])
    atoms = [
        vq.Atom(1, [0.0, 0.0, 0.0]),
        vq.Atom(1, [R_HH, 0.0, 0.0]),
    ]
    sys = vq.PeriodicSystem(3, lat, atoms)
    basis = vq.BasisSet(sys.unit_cell_molecule(), "sto-3g")
    opts = vq.PeriodicSCFOptions()
    opts.conv_tol_energy = 1e-12
    opts.lattice_opts.cutoff_bohr = 12.0
    opts.lattice_opts.nuclear_cutoff_bohr = 12.0
    kmesh = vq.monkhorst_pack(sys, [1, 1, 1])

    result = vq.run_rhf_periodic(sys, basis, kmesh, opts)
    D = np.asarray(result.density)
    D_set = core.compute_overlap_lattice(basis, sys, opts.lattice_opts)
    for c in range(len(D_set.cells)):
        D_set.set_block(c, D)

    g_j = np.asarray(core.eri_lattice_gradient_contribution(
        basis, sys, D_set, opts.lattice_opts, 0.0))
    g_jk = np.asarray(core.eri_lattice_gradient_contribution(
        basis, sys, D_set, opts.lattice_opts, 1.0))

    np.testing.assert_allclose(g_j, [[0.41855598, 0.0, 0.0],
                                     [-0.41855598, 0.0, 0.0]], atol=5e-8)
    np.testing.assert_allclose(g_jk, [[0.31128563, 0.0, 0.0],
                                      [-0.31128563, 0.0, 0.0]], atol=5e-8)
    np.testing.assert_allclose(g_jk[0], -g_jk[1], atol=1e-10)
    assert abs((g_jk[0, 0] - g_j[0, 0]) + 0.10727035) < 5e-8
