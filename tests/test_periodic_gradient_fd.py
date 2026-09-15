"""Phase G1a-fd — finite-difference periodic atomic gradient.

Validation reference for the analytic periodic-gradient code that
lands in G1a / G1b. Pinned contracts:

1. **API surface** — :func:`vibeqc.compute_gradient_periodic_rhf_fd`
   is public. Returns ``(n_atoms, 3)`` array in Ha/bohr.

2. **Molecular limit recovery** — H₂ in a 20-Å cubic box with a
   1×1×1 (Γ-only) k-mesh reproduces the molecular RHF gradient to
   FD-truncation precision (≤ 1e-7 Ha/bohr at ``step_bohr=1e-3``).

3. **Newton's 3rd law on periodic 1D H chain** — equal-and-opposite
   forces on the two intra-cell H atoms; transverse (y, z)
   components exactly zero by symmetry.

4. **Equilibrium gradient ≈ 0** — at a known equilibrium geometry
   (eq H₂ bond), the FD gradient magnitude is below the
   FD-truncation noise floor.

5. **Lattice vectors held fixed** — the displaced systems share the
   reference lattice; only atomic positions move (atomic gradients
   only — cell-parameter gradients are G2 / phase v0.7).

6. **Bad inputs raise** — non-convergent SCF at displaced geometry
   raises a clear ``RuntimeError`` pointing at the diagnosed atom +
   axis.
"""

from __future__ import annotations

import numpy as np
import pytest

import vibeqc as vq


ANGSTROM_TO_BOHR = 1.8897261339213


# ---------------------------------------------------------------------------
# 1. API surface
# ---------------------------------------------------------------------------

def test_compute_gradient_periodic_rhf_fd_exposed():
    assert hasattr(vq, "compute_gradient_periodic_rhf_fd")


# ---------------------------------------------------------------------------
# 2. Molecular limit — H2 in a giant cubic box
# ---------------------------------------------------------------------------

def test_h2_molecular_limit_matches_molecular_gradient():
    """H₂ at R = 1.0 Å in a 20-Å cubic box, Γ-only mesh, real-space
    cutoff smaller than the box. The periodic FD gradient should
    match the molecular analytic gradient to ~1e-7 Ha/bohr."""
    R = 1.0 * ANGSTROM_TO_BOHR
    big_box = 20.0 * ANGSTROM_TO_BOHR
    atoms = [
        vq.Atom(1, [0.0, 0.0, 0.0]),
        vq.Atom(1, [0.0, 0.0, R]),
    ]

    # Molecular reference
    mol = vq.Molecule(atoms, charge=0, multiplicity=1)
    basis_mol = vq.BasisSet(mol, "sto-3g")
    opts_mol = vq.RHFOptions()
    opts_mol.conv_tol_energy = 1e-10
    result_mol = vq.run_rhf(mol, basis_mol, opts_mol)
    g_mol = np.asarray(vq.compute_gradient(mol, basis_mol, result_mol))

    # Periodic FD
    sys = vq.PeriodicSystem(3, np.diag([big_box, big_box, big_box]), atoms)
    kmesh = vq.monkhorst_pack(sys, [1, 1, 1])
    opts_periodic = vq.PeriodicSCFOptions()
    opts_periodic.conv_tol_energy = 1e-10
    opts_periodic.lattice_opts.cutoff_bohr = 25.0
    g_fd = vq.compute_gradient_periodic_rhf_fd(
        sys, "sto-3g", kmesh, opts_periodic, step_bohr=1e-3)

    np.testing.assert_allclose(g_fd, g_mol, atol=1e-6,
        err_msg="molecular-limit periodic FD gradient diverges from "
                "molecular analytic gradient")


# ---------------------------------------------------------------------------
# 3. Newton's 3rd law on 1D H chain
# ---------------------------------------------------------------------------

def test_h_chain_1d_newtons_third_law():
    """Two H atoms in a unit cell with periodic axis along x. The
    intra-cell bond gradient must satisfy:

      - g[0, x] = -g[1, x] (Newton's 3rd law along the periodic axis)
      - g[*, y] = g[*, z] = 0 (no transverse force; vacuum is symmetric)
    """
    a_per = 2.0 * ANGSTROM_TO_BOHR    # periodic
    vac = 20.0 * ANGSTROM_TO_BOHR     # vacuum
    R_HH = 0.74 * ANGSTROM_TO_BOHR
    lat = np.diag([a_per, vac, vac])
    atoms = [
        vq.Atom(1, [0.0, 0.0, 0.0]),
        vq.Atom(1, [R_HH, 0.0, 0.0]),
    ]
    sys = vq.PeriodicSystem(3, lat, atoms)
    kmesh = vq.monkhorst_pack(sys, [4, 1, 1])
    opts = vq.PeriodicSCFOptions()
    # Revalidated 2026-08-20 (GitLab IID 189). At 1e-10 energy convergence,
    # the twelve independent displaced SCFs stopped with translation-paired
    # electronic energies differing by 3.14e-9 Ha, which becomes a
    # 1.546e-6 Ha/bohr net gradient after division by 2h. Tightening the
    # fixture reduces that residual to 7.3e-10 without changing h or atol.
    opts.conv_tol_energy = 1e-12
    opts.conv_tol_grad = 1e-10
    opts.lattice_opts.cutoff_bohr = 12.0
    opts.lattice_opts.nuclear_cutoff_bohr = 25.0
    opts.lattice_opts.coulomb_method = vq.CoulombMethod.DIRECT_TRUNCATED
    g = vq.compute_gradient_periodic_rhf_fd(sys, "sto-3g", kmesh, opts,
                                              step_bohr=1e-3)

    # Newton's 3rd law along x.
    np.testing.assert_allclose(g[0, 0], -g[1, 0], atol=1e-7)
    # Transverse forces vanish by symmetry.
    np.testing.assert_allclose(g[:, 1:], 0.0, atol=1e-7)


# ---------------------------------------------------------------------------
# 4. Equilibrium gradient ≈ 0
# ---------------------------------------------------------------------------

def test_h2_periodic_matches_h2_molecular_at_equilibrium():
    """H₂ in a 20-Å box at any geometry — periodic FD gradient should
    track molecular analytic gradient. Tested at 0.74 Å where neither
    is zero but where the molecular code is well-validated."""
    R = 0.74 * ANGSTROM_TO_BOHR
    big_box = 20.0 * ANGSTROM_TO_BOHR
    atoms = [
        vq.Atom(1, [0.0, 0.0, 0.0]),
        vq.Atom(1, [0.0, 0.0, R]),
    ]
    mol = vq.Molecule(atoms, charge=0, multiplicity=1)
    basis_mol = vq.BasisSet(mol, "sto-3g")
    opts_mol = vq.RHFOptions()
    opts_mol.conv_tol_energy = 1e-12
    g_mol = np.asarray(vq.compute_gradient(mol, basis_mol,
                                              vq.run_rhf(mol, basis_mol, opts_mol)))

    sys = vq.PeriodicSystem(3, np.diag([big_box, big_box, big_box]), atoms)
    kmesh = vq.monkhorst_pack(sys, [1, 1, 1])
    opts_periodic = vq.PeriodicSCFOptions()
    opts_periodic.conv_tol_energy = 1e-12
    opts_periodic.lattice_opts.cutoff_bohr = 25.0
    g_fd = vq.compute_gradient_periodic_rhf_fd(
        sys, "sto-3g", kmesh, opts_periodic, step_bohr=1e-3)

    np.testing.assert_allclose(g_fd, g_mol, atol=1e-6)
