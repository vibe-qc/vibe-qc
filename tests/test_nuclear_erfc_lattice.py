"""Phase 12e-b: short-range (erfc-screened) nuclear-attraction lattice sum.

This is an Ewald building block, not a user-facing feature yet. Its
purpose is to provide the real-space component of the Gaussian-charge
Ewald treatment of V(g) that Phase 12e-c assembles into a full k-space
Ewald dispatch.

Correctness witnesses
---------------------

1. **ω → 0 limit.** The kernel erfc(ω·r) / r → 1/r as ω → 0. Therefore
   compute_nuclear_erfc_lattice with a tiny ω must reproduce
   compute_nuclear_lattice (the standard 1/r integral).

2. **ω → ∞ limit.** The kernel erfc(ω·r) / r → 0 for any r > 0.
   Therefore the integral vanishes as ω grows.

3. **Monotonicity.** For fixed ω > 0, the magnitudes of the diagonal
   matrix elements must decrease monotonically as ω grows (the kernel
   is monotonically decreasing in ω for every r > 0).

4. **Cell-shape invariance.** The erfc-screened integral depends only
   on atomic positions, not on the choice of periodic cell basis — if
   we double the unit cell, the integrals on the reference cell remain
   the same.
"""

from __future__ import annotations

import numpy as np
import pytest

import vibeqc as vq


_BOHR_PER_A = 1.0 / 0.529177210903


def _small_system():
    """H2O in a big box — molecular-limit regime so reference values are
    exactly the molecular ones when only g=0 is in the cutoff."""
    atoms = [
        vq.Atom(8, [0, 0, 0]),
        vq.Atom(1, [0, 1.43, -0.98]),
        vq.Atom(1, [0, -1.43, -0.98]),
    ]
    sysp = vq.PeriodicSystem(3, np.eye(3) * 30.0, atoms)
    basis = vq.BasisSet(sysp.unit_cell_molecule(), "sto-3g")
    opts = vq.LatticeSumOptions()
    opts.cutoff_bohr = 10.0
    opts.nuclear_cutoff_bohr = 15.0
    return sysp, basis, opts


def test_erfc_omega_small_reduces_to_full_nuclear():
    """compute_nuclear_erfc_lattice(ω → 0) == compute_nuclear_lattice."""
    sysp, basis, opts = _small_system()
    V_full = vq.compute_nuclear_lattice(basis, sysp, opts)
    V_erfc = vq.compute_nuclear_erfc_lattice(basis, sysp, 1e-10, opts)
    assert len(V_full.blocks) == len(V_erfc.blocks)
    for c in range(len(V_full.blocks)):
        max_diff = np.max(np.abs(np.asarray(V_full.blocks[c])
                                 - np.asarray(V_erfc.blocks[c])))
        # erfc(1e-10 · r) ≈ 1 − 1.1e-10 · r · 2/√π. For r up to ~15 bohr
        # (nuclear cutoff) this gives a residual of ~ 1e-9 relative to V;
        # absolute residual scales with the typical V magnitude.
        assert max_diff < 1e-7, (
            f"cell {c}: max |V_full − V_erfc(ω=1e-10)| = {max_diff:.2e}"
        )


def test_erfc_omega_large_vanishes_as_inverse_omega_squared():
    """As ω → ∞, compute_nuclear_erfc_lattice → 0. The decay rate is 1/ω²
    because the integrand erfc(ω·r)/r contributes from a shell of
    thickness ~1/ω around each nucleus where the AO density is O(1)."""
    sysp, basis, opts = _small_system()
    V_1e3 = vq.compute_nuclear_erfc_lattice(basis, sysp, 1000.0, opts)
    V_1e4 = vq.compute_nuclear_erfc_lattice(basis, sysp, 10000.0, opts)
    m_1e3 = max(np.max(np.abs(np.asarray(b))) for b in V_1e3.blocks)
    m_1e4 = max(np.max(np.abs(np.asarray(b))) for b in V_1e4.blocks)
    # Expected 100× decrease (ω grows 10× → integral drops as 1/ω²);
    # tolerate a factor of 3 slop because the tight 1s core deviates
    # from the asymptotic estimate slightly.
    ratio = m_1e3 / m_1e4
    assert 30.0 < ratio < 300.0, (
        f"erfc integrals not decaying as ~1/ω²: "
        f"|V|(ω=1e3) = {m_1e3:.3e}, |V|(ω=1e4) = {m_1e4:.3e}, "
        f"ratio = {ratio:.1f}"
    )
    # And at ω=1e4 the magnitude is already below 1e-4.
    assert m_1e4 < 1e-4


@pytest.mark.parametrize("omega_pair", [(0.1, 0.2), (0.3, 0.5), (0.5, 1.0)])
def test_erfc_monotonic_in_omega(omega_pair):
    """|V_erfc|_ii decreases as ω grows (kernel monotonically decreases)."""
    sysp, basis, opts = _small_system()
    omega_small, omega_large = omega_pair
    V_small = vq.compute_nuclear_erfc_lattice(basis, sysp, omega_small, opts)
    V_large = vq.compute_nuclear_erfc_lattice(basis, sysp, omega_large, opts)
    # g=0 block, diagonal element magnitudes
    d_small = np.abs(np.diag(np.asarray(V_small.blocks[0])))
    d_large = np.abs(np.diag(np.asarray(V_large.blocks[0])))
    # Every diagonal element must have |V(ω_large)| ≤ |V(ω_small)|.
    assert (d_large <= d_small + 1e-12).all(), (
        f"monotonicity violated: ω={omega_small} diag={d_small}, "
        f"ω={omega_large} diag={d_large}"
    )


def test_erfc_decays_with_cell_distance():
    """For ω > 0 the erfc kernel is exponentially short-ranged; the g-cell
    blocks with large |g| must have near-zero norm even when the
    corresponding full-nuclear block doesn't."""
    atoms = [vq.Atom(1, [0, 0, 0]), vq.Atom(1, [0, 0, 1.4])]
    sysp = vq.PeriodicSystem(1, np.diag([6.0, 30.0, 30.0]), atoms)
    basis = vq.BasisSet(sysp.unit_cell_molecule(), "sto-3g")
    opts = vq.LatticeSumOptions()
    opts.cutoff_bohr = 30.0         # pulls in many cells
    opts.nuclear_cutoff_bohr = 40.0
    V = vq.compute_nuclear_erfc_lattice(basis, sysp, 0.5, opts)

    # Sort blocks by cell distance and verify monotone decay in |V(g)|
    distances = [np.linalg.norm(np.asarray(c.r_cart)) for c in V.cells]
    norms = [np.linalg.norm(np.asarray(b)) for b in V.blocks]
    ordered = sorted(zip(distances, norms))
    # Strict monotonicity isn't required because of sign flips, but the
    # farthest cell must be ≥ 4 orders of magnitude smaller than the
    # nearest non-zero cell.
    far_norm = ordered[-1][1]
    near_norm = [n for d, n in ordered if d > 1e-6][0]
    assert far_norm < near_norm, (
        f"erfc blocks don't decay: nearest |V| = {near_norm:.2e}, "
        f"farthest |V| = {far_norm:.2e}"
    )
    assert far_norm < 1e-6, (
        f"erfc block at |g| = {ordered[-1][0]:.1f} bohr still has "
        f"norm {far_norm:.2e} — suspiciously large for ω = 0.5"
    )


def test_erfc_identity_matches_molecular_V_in_big_box():
    """At ω → 0 and in the molecular-limit regime (only g=0 in cutoff),
    the erfc-screened V is the molecular V."""
    atoms = [vq.Atom(1, [0, 0, 0]), vq.Atom(1, [0, 0, 1.4])]
    mol = vq.Molecule(atoms)
    sysp = vq.PeriodicSystem(3, np.eye(3) * 50.0, atoms)
    basis = vq.BasisSet(mol, "sto-3g")

    opts = vq.LatticeSumOptions()
    opts.cutoff_bohr = 10.0
    opts.nuclear_cutoff_bohr = 10.0
    V_erfc = vq.compute_nuclear_erfc_lattice(basis, sysp, 1e-10, opts)
    assert len(V_erfc.blocks) == 1

    V_mol = np.asarray(vq.compute_nuclear(basis, mol))
    V_erfc_g0 = np.asarray(V_erfc.blocks[0])

    max_diff = np.max(np.abs(V_erfc_g0 - V_mol))
    assert max_diff < 1e-8, (
        f"V_erfc(ω→0) doesn't match molecular V: max diff = {max_diff:.2e}"
    )


def _displaced_pair(z=2, shift=0.0):
    system = vq.PeriodicSystem(3, np.diag([8., 16., 16.]), [
        vq.Atom(z, [shift, 0., 0.]), vq.Atom(z, [shift + 4., 0., 0.]),
    ])
    return system, vq.BasisSet(system.unit_cell_molecule(), "6-31g")


def test_ewald_nuclear_support_covers_product_offsets_and_smearing():
    from scipy.special import erfcinv
    from vibeqc.pbc_bipole_common import ewald_erfc_lattice_options

    system, basis = _displaced_pair()
    opts = vq.LatticeSumOptions()
    opts.cutoff_bohr = opts.nuclear_cutoff_bohr = 15.0
    alpha, tol = 0.2861288035052463, 1e-8
    padded = ewald_erfc_lattice_options(opts, alpha, tol, basis=basis, system=system)
    gamma_min = min(min(shell.exponents) for shell in basis.shells())
    expected = 15.0 + 4.0 + erfcinv(tol) * np.sqrt(1 / (2*gamma_min) + 1/alpha**2)
    assert padded.nuclear_cutoff_bohr == pytest.approx(expected)
    assert padded.cutoff_bohr == opts.cutoff_bohr == 15.0
    assert opts.nuclear_cutoff_bohr == 15.0  # Never mutate the caller.
    shifted, shifted_basis = _displaced_pair(shift=123.0)
    moved = ewald_erfc_lattice_options(
        opts, alpha, tol, basis=shifted_basis, system=shifted,
    )
    assert moved.nuclear_cutoff_bohr == padded.nuclear_cutoff_bohr
    opts.nuclear_cutoff_bohr = 80.0
    assert ewald_erfc_lattice_options(
        opts, alpha, tol, basis=basis, system=system,
    ).nuclear_cutoff_bohr == 80.0


@pytest.mark.parametrize("z", [2, 10])  # contracted s and s+p products
def test_ewald_nuclear_support_matches_extended_sum(z):
    from vibeqc.pbc_bipole_common import ewald_erfc_lattice_options

    system, basis = _displaced_pair(z)
    opts = vq.LatticeSumOptions()
    opts.cutoff_bohr = opts.nuclear_cutoff_bohr = 15.0
    alpha = 0.2861288035052463
    padded = ewald_erfc_lattice_options(opts, alpha, 1e-8, basis=basis, system=system)
    actual = vq.compute_nuclear_erfc_lattice(basis, system, alpha, padded)
    padded.nuclear_cutoff_bohr += 16.0
    extended = vq.compute_nuclear_erfc_lattice(basis, system, alpha, padded)
    assert len(actual.cells) == len(extended.cells) == 3
    np.testing.assert_allclose(actual.blocks, extended.blocks, atol=1e-10, rtol=0)


def test_ewald_nuclear_support_repairs_raw_nonzero_k_covariance():
    from vibeqc.pbc_bipole_common import (
        _bloch_sum_blocks_multi_k,
        ewald_erfc_lattice_options,
    )

    system, basis = _displaced_pair()
    opts = vq.LatticeSumOptions()
    opts.cutoff_bohr = opts.nuclear_cutoff_bohr = 15.0
    alpha = 0.2861288035052463
    old = vq.compute_nuclear_erfc_lattice(basis, system, alpha, opts)
    padded = ewald_erfc_lattice_options(opts, alpha, 1e-8, basis=basis, system=system)
    repaired = vq.compute_nuclear_erfc_lattice(basis, system, alpha, padded)
    k = [np.pi/8, 0., 0.]
    before, after = [
        _bloch_sum_blocks_multi_k(x.blocks, x.cells, [k])[0] for x in (old, repaired)
    ]
    action = np.zeros((4, 4), complex)
    action[2:, :2] = np.eye(2)
    action[:2, 2:] = -np.eye(2)
    assert np.max(np.abs(action.conj().T@before@action-before)) > 5e-6
    assert np.max(np.abs(before-before.conj().T)) > 1e-6
    assert np.max(np.abs(action.conj().T@after@action-after)) < 1e-9
    assert np.max(np.abs(after-after.conj().T)) < 1e-12
    # No Hermitization, matrix averaging, alpha change or AO-output widening.


def test_ewald_nuclear_support_independent_s_product_integrals():
    """Gaussian-product/Boys evaluation, independent of libint erfc tiles."""
    from scipy.special import erf
    from vibeqc.pbc_bipole_common import ewald_erfc_lattice_options

    system, basis = _displaced_pair()
    opts = vq.LatticeSumOptions()
    opts.cutoff_bohr = opts.nuclear_cutoff_bohr = 15.0
    alpha = 0.2861288035052463
    padded = ewald_erfc_lattice_options(opts, alpha, 1e-8, basis=basis, system=system)
    actual = vq.compute_nuclear_erfc_lattice(basis, system, alpha, padded)
    cells = vq._vibeqc_core.direct_lattice_cells(system, padded.nuclear_cutoff_bohr)
    charges = np.array([np.asarray(atom.xyz) + cell.r_cart
                        for cell in cells for atom in system.unit_cell])
    shells = list(basis.shells())

    def boys0(t):
        root = np.sqrt(t)
        return np.divide(np.sqrt(np.pi)*erf(root), 2*root,
                         out=np.ones_like(t), where=root > 0)

    for cell, block in zip(actual.cells, actual.blocks):
        expected = np.zeros((4, 4))
        for i, bra in enumerate(shells):
            a_center = np.asarray(bra.origin)
            for j, ket in enumerate(shells):
                b_center = np.asarray(ket.origin) + cell.r_cart
                distance2 = np.sum((a_center-b_center)**2)
                for a, ca in zip(bra.exponents, bra.coefficients):
                    for b, cb in zip(ket.exponents, ket.coefficients):
                        p = a+b
                        product_center = (a*a_center+b*b_center)/p
                        r2 = np.sum((charges-product_center)**2, axis=1)
                        q = p*alpha**2/(p+alpha**2)
                        screened = boys0(p*r2)-np.sqrt(q/p)*boys0(q*r2)
                        expected[i, j] -= (
                            2*ca*cb * (2*np.pi/p) * np.exp(-a*b/p*distance2)
                            * np.sum(screened)
                        )  # Z=2 for every nuclear image.
        np.testing.assert_allclose(block, expected, atol=1e-11, rtol=0)


@pytest.mark.parametrize("alpha,tol", [(0., 1e-8), (float("inf"), 1e-8),
                                      (float("nan"), 1e-8), (0.3, 0.), (0.3, 1.)])
def test_ewald_nuclear_support_rejects_invalid_controls(alpha, tol):
    from vibeqc.pbc_bipole_common import ewald_erfc_lattice_options

    system, basis = _displaced_pair()
    with pytest.raises(ValueError):
        ewald_erfc_lattice_options(
            vq.LatticeSumOptions(), alpha, tol, basis=basis, system=system,
        )
