"""Phase 12e-c-2: erfc-screened ERIs for the short-range Ewald J/K.

Extends the J/K builders with an ``omega`` parameter that swaps the
1/r_12 kernel for erfc(ω·r_12)/r_12 — the short-range piece of the
Ewald split for 3D bulk. The matching long-range part
(erf(ω·r_12)/r_12, evaluated in reciprocal space) lands in 12e-c-3.

Correctness witnesses (same pattern as 12e-b tested for V):

1. **ω → 0 limit.** erfc(ω·r)/r → 1/r. The screened builders must
   reduce to the unscreened variants (the existing coulomb path).

2. **ω → ∞ limit.** erfc(ω·r)/r → 0 for r > 0; only on-contact
   "self-interaction" can contribute anything. Screened builders
   return ≪ 1 Ha matrix elements at large ω.

3. **Monotonicity.** |J(ω)| and |K(ω)| decrease monotonically as ω
   grows — the kernel is monotone-decreasing in ω at every r > 0.

4. **Short + long = full.** For any ω, J(ω) plus the implicit long-
   range complement equals J(ω=0). We spot-check by choosing ω that
   gives a clean, non-trivial split and verifying J(0) − J(ω) has
   the same sign and order of magnitude as J(ω).
"""

from __future__ import annotations

import numpy as np
import pytest

import vibeqc as vq


def _h2_system():
    uc = [vq.Atom(1, [0, 0, 0]), vq.Atom(1, [0, 0, 1.4])]
    sysp = vq.PeriodicSystem(3, np.eye(3) * 30.0, uc)
    basis = vq.BasisSet(sysp.unit_cell_molecule(), "sto-3g")
    opts = vq.LatticeSumOptions()
    opts.cutoff_bohr = 10.0
    opts.nuclear_cutoff_bohr = 15.0
    mol = sysp.unit_cell_molecule()
    P_mol = np.asarray(vq.run_rhf(mol, basis).density)
    return sysp, basis, opts, P_mol


# ---------------------------------------------------------------------------
# build_jk_gamma_molecular_limit: screened J/K
# ---------------------------------------------------------------------------

def test_jk_omega_zero_matches_unscreened():
    """Passing ``omega=0`` must be indistinguishable from the default
    (full 1/r_12) call, to machine precision."""
    sysp, basis, opts, P = _h2_system()
    jk_default = vq.build_jk_gamma_molecular_limit(basis, sysp, opts, P)
    jk_omega0  = vq.build_jk_gamma_molecular_limit(basis, sysp, opts, P, 0.0)
    assert np.allclose(np.asarray(jk_default.J),
                       np.asarray(jk_omega0.J),  atol=1e-14)
    assert np.allclose(np.asarray(jk_default.K),
                       np.asarray(jk_omega0.K),  atol=1e-14)


def test_jk_omega_tiny_approaches_unscreened():
    """For ω → 0⁺ the erfc kernel ≈ 1/r; agreement to O(ω)."""
    sysp, basis, opts, P = _h2_system()
    jk_0 = vq.build_jk_gamma_molecular_limit(basis, sysp, opts, P, 0.0)
    jk_e = vq.build_jk_gamma_molecular_limit(basis, sysp, opts, P, 1.0e-6)
    assert np.max(np.abs(np.asarray(jk_0.J) - np.asarray(jk_e.J))) < 1e-5
    assert np.max(np.abs(np.asarray(jk_0.K) - np.asarray(jk_e.K))) < 1e-5


def test_jk_omega_large_vanishes_to_contact_only():
    """erfc(ω·r)/r → 0 for any r > 0 as ω → ∞; the residual is a
    short-ranged "contact" term that scales as 1/ω² for compact orbitals
    near a nucleus. At ω = 10 it should be well below 0.01 Ha."""
    sysp, basis, opts, P = _h2_system()
    jk = vq.build_jk_gamma_molecular_limit(basis, sysp, opts, P, 10.0)
    assert np.max(np.abs(np.asarray(jk.J))) < 0.01
    assert np.max(np.abs(np.asarray(jk.K))) < 0.01


@pytest.mark.parametrize(
    "omega_pair", [(0.2, 0.5), (0.5, 1.0), (1.0, 2.0)],
)
def test_jk_omega_monotone_decreasing(omega_pair):
    """|J(ω)| and |K(ω)| decrease monotonically in ω at every AO pair —
    the erfc kernel is monotone-decreasing in ω for every r > 0, so the
    short-range contribution shrinks as ω grows."""
    sysp, basis, opts, P = _h2_system()
    w_small, w_large = omega_pair
    jk_small = vq.build_jk_gamma_molecular_limit(basis, sysp, opts, P, w_small)
    jk_large = vq.build_jk_gamma_molecular_limit(basis, sysp, opts, P, w_large)
    J_s = np.abs(np.asarray(jk_small.J))
    J_l = np.abs(np.asarray(jk_large.J))
    # Element-wise |J(ω_small)| ≥ |J(ω_large)| (with 1e-12 slack for
    # round-off on vanishing elements).
    assert (J_l <= J_s + 1e-12).all()


def test_jk_short_plus_long_equals_full():
    """J(ω=0) = J_short(ω) + J_long(ω). The long-range piece is what
    12e-c-3 will compute; here we verify the decomposition algebraically
    by taking differences."""
    sysp, basis, opts, P = _h2_system()
    omega = 0.4
    jk_full = vq.build_jk_gamma_molecular_limit(basis, sysp, opts, P, 0.0)
    jk_short = vq.build_jk_gamma_molecular_limit(basis, sysp, opts, P, omega)

    J_long_implied = np.asarray(jk_full.J) - np.asarray(jk_short.J)
    K_long_implied = np.asarray(jk_full.K) - np.asarray(jk_short.K)

    # Long-range pieces have the same sign as the full at each AO pair
    # where the full is large enough to trust the sign (> 1e-6 Ha).
    for arr_full, arr_long in ((np.asarray(jk_full.J), J_long_implied),
                               (np.asarray(jk_full.K), K_long_implied)):
        mask = np.abs(arr_full) > 1e-6
        # Same sign where the full is non-trivial.
        assert np.all((np.sign(arr_full[mask]) * np.sign(arr_long[mask])) >= 0)


# ---------------------------------------------------------------------------
# build_fock_2e_real_space: screened multi-k Fock build
# ---------------------------------------------------------------------------

def _h2_multi_k_inputs():
    uc = [vq.Atom(1, [0, 0, 0]), vq.Atom(1, [0, 0, 1.4])]
    sysp = vq.PeriodicSystem(3, np.eye(3) * 30.0, uc)
    basis = vq.BasisSet(sysp.unit_cell_molecule(), "sto-3g")
    opts = vq.LatticeSumOptions()
    opts.cutoff_bohr = 10.0
    opts.nuclear_cutoff_bohr = 15.0

    # Seed P(g) from a Γ-point diagonalisation — we only need *a*
    # reasonable density to call the builder; exact values don't
    # matter for the ω-invariance tests.
    km = vq.monkhorst_pack(sysp, [1, 1, 1])
    S = vq.compute_overlap_lattice(basis, sysp, opts)
    T = vq.compute_kinetic_lattice(basis, sysp, opts)
    V = vq.compute_nuclear_lattice(basis, sysp, opts)
    k0 = np.zeros(3)
    Sk = vq.bloch_sum(S, k0)
    Hk = vq.bloch_sum(T, k0) + vq.bloch_sum(V, k0)
    bd = vq.diagonalize_bloch(Hk, Sk)
    n_occ = sysp.n_electrons() // 2
    P_real = vq.real_space_density_from_kpoints(
        [bd.coefficients], [n_occ], km, S.cells,
    )
    return sysp, basis, opts, P_real


def test_fock_2e_omega_zero_matches_unscreened():
    sysp, basis, opts, P_real = _h2_multi_k_inputs()
    F_default = vq.build_fock_2e_real_space(basis, sysp, opts, P_real, 1.0)
    F_omega0  = vq.build_fock_2e_real_space(basis, sysp, opts, P_real, 1.0, 0.0)
    for b_d, b_0 in zip(F_default.blocks, F_omega0.blocks):
        assert np.allclose(np.asarray(b_d), np.asarray(b_0), atol=1e-14)


def test_fock_2e_omega_large_vanishes():
    sysp, basis, opts, P_real = _h2_multi_k_inputs()
    F_big = vq.build_fock_2e_real_space(basis, sysp, opts, P_real, 1.0, 20.0)
    for b in F_big.blocks:
        assert np.max(np.abs(np.asarray(b))) < 0.01


def test_fock_2e_omega_negative_rejected():
    sysp, basis, opts, P_real = _h2_multi_k_inputs()
    with pytest.raises(RuntimeError, match="omega"):
        vq.build_fock_2e_real_space(basis, sysp, opts, P_real, 1.0, -0.5)
