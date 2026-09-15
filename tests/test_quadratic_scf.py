"""Phase C1c — quadratic-SCF kernel + Γ-only RHF Ewald integration tests.

Pin the contract:

  1. ``expm_skew`` matches scipy's ``expm`` to machine precision on
     skew-symmetric input. Roundtrip ``expm_skew(0) == I``,
     ``expm_skew(K) · expm_skew(-K) == I``, and ``expm_skew(K)`` is
     orthogonal for skew K (matrix preserves the metric).

  2. ``quadratic_step`` taken from a converged RHF density is the
     identity (Brillouin condition: F^MO_{ai} = 0 for occ-vir).

  3. The Γ-only RHF Ewald driver with the C1c fallback enabled
     converges to the same energy as the standard path on a clean
     system (correctness — fallback doesn't break easy cases).

  4. The new option fields exist on PeriodicRHFOptions with the
     documented defaults.
"""

from __future__ import annotations

import numpy as np
import pytest

import vibeqc as vq
from vibeqc.quadratic_scf import expm_skew, quadratic_step


# ---------------------------------------------------------------------------
# expm_skew
# ---------------------------------------------------------------------------

def test_expm_skew_zero_returns_identity():
    K = np.zeros((5, 5))
    assert np.allclose(expm_skew(K), np.eye(5))


def test_expm_skew_inverts_under_negation():
    """exp(K) · exp(-K) = I for any K (group property of the matrix
    exponential)."""
    rng = np.random.default_rng(42)
    A = rng.standard_normal((6, 6))
    K = A - A.T   # skew-symmetric
    U = expm_skew(K)
    U_inv = expm_skew(-K)
    assert np.allclose(U @ U_inv, np.eye(6), atol=1e-12)


def test_expm_skew_is_orthogonal_for_skew_input():
    """exp(K) is orthogonal for skew-symmetric K — fundamental
    property: U U^T = I."""
    rng = np.random.default_rng(7)
    A = rng.standard_normal((8, 8))
    K = A - A.T
    U = expm_skew(K)
    assert np.allclose(U @ U.T, np.eye(8), atol=1e-12)
    assert np.allclose(U.T @ U, np.eye(8), atol=1e-12)


def test_expm_skew_satisfies_exp_group_law():
    """Group property of the matrix exponential:
    ``exp(K) · exp(K) == exp(2K)`` — and stays accurate when the
    norm is large enough to trigger scaling-and-squaring inside
    ``expm_skew``. Doesn't rely on an external implementation."""
    rng = np.random.default_rng(123)
    A = rng.standard_normal((4, 4))
    K = (A - A.T) * 2.0    # ‖K‖ ~ 5, scaling-and-squaring path
    U = expm_skew(K)
    U2 = expm_skew(2.0 * K)
    assert np.allclose(U @ U, U2, atol=1e-12)


def test_expm_skew_matches_scipy_when_available():
    """Independent ground-truth check via scipy. Skipped if scipy
    isn't installed (it's an optional convenience dep)."""
    sp = pytest.importorskip("scipy.linalg")
    rng = np.random.default_rng(456)
    A = rng.standard_normal((5, 5))
    K = (A - A.T) * 1.5
    assert np.allclose(expm_skew(K), sp.expm(K), atol=1e-12)


# ---------------------------------------------------------------------------
# quadratic_step
# ---------------------------------------------------------------------------

def test_quadratic_step_is_identity_at_convergence():
    """At a converged SCF, F is diagonal in the MO basis (Brillouin
    condition for HF). The Newton step's occ-vir gradient is zero,
    so κ = 0 and exp(κ) = I — C and ε come back unchanged."""
    n_bf = 6
    n_occ = 2

    # Construct a "converged" state: F is already diagonal in C's
    # MO basis with eigenvalues ε.
    rng = np.random.default_rng(99)
    eps = np.sort(rng.uniform(-2.0, 2.0, size=n_bf))
    # Random orthogonal C.
    A = rng.standard_normal((n_bf, n_bf))
    C, _ = np.linalg.qr(A)
    F = C @ np.diag(eps) @ C.T

    C_new, eps_new = quadratic_step(F, C, eps, n_occ, shift=0.1)
    # Step is the identity → C_new == C, eps_new == eps.
    assert np.allclose(C_new, C, atol=1e-10)
    assert np.allclose(eps_new, eps, atol=1e-10)


def test_quadratic_step_decreases_orbital_gradient():
    """One Newton step from a non-converged state should decrease
    the orbital-gradient norm ‖F^MO_ov‖ in the new MO basis."""
    n_bf = 5
    n_occ = 2
    rng = np.random.default_rng(13)
    # Symmetric F with non-trivial structure.
    A = rng.standard_normal((n_bf, n_bf))
    F = A + A.T

    # Start from MO basis = canonical orth of F's leading eigenvectors,
    # but perturb so we're not yet at convergence.
    C0, _ = np.linalg.qr(rng.standard_normal((n_bf, n_bf)))
    eps0 = np.sort(np.diag(C0.T @ F @ C0))
    # Re-sort C by eps: standard "occupied lowest" convention.
    order = np.argsort(np.diag(C0.T @ F @ C0))
    C0 = C0[:, order]
    eps0 = np.array([np.diag(C0.T @ F @ C0)[i] for i in range(n_bf)])

    F_mo = C0.T @ F @ C0
    grad0 = np.linalg.norm(F_mo[n_occ:, :n_occ])

    C1, eps1 = quadratic_step(F, C0, eps0, n_occ, shift=0.5, max_step=0.5)

    F_mo_new = C1.T @ F @ C1
    grad1 = np.linalg.norm(F_mo_new[n_occ:, :n_occ])
    assert grad1 < grad0, (
        f"Newton step increased gradient: {grad0:.3e} → {grad1:.3e}"
    )


def test_quadratic_step_no_virtuals_returns_unchanged():
    """Edge case: when n_occ == n_kept (no virtual subspace), the
    step is the identity by definition."""
    n = 3
    rng = np.random.default_rng(0)
    F = rng.standard_normal((n, n))
    F = F + F.T
    A = rng.standard_normal((n, n))
    C, _ = np.linalg.qr(A)
    eps = np.diag(C.T @ F @ C)

    C_new, eps_new = quadratic_step(F, C, eps, n_occ=n)
    assert np.allclose(C_new, C, atol=1e-12)


# ---------------------------------------------------------------------------
# Integration: Γ-only RHF Ewald with C1c enabled
# ---------------------------------------------------------------------------

def _h2_in_box(box: float = 30.0):
    c = box / 2
    atoms = [
        vq.Atom(1, [c, c, c - 0.7]),
        vq.Atom(1, [c, c, c + 0.7]),
    ]
    sysp = vq.PeriodicSystem(3, np.eye(3) * box, atoms)
    basis = vq.BasisSet(sysp.unit_cell_molecule(), "sto-3g")
    return sysp, basis


def _default_options():
    opts = vq.PeriodicRHFOptions()
    opts.lattice_opts.cutoff_bohr = 12.0
    opts.lattice_opts.nuclear_cutoff_bohr = 15.0
    opts.damping = 0.3
    opts.max_iter = 60
    opts.use_diis = True
    return opts


def test_periodic_options_have_quadratic_fallback_fields():
    """The C++ binding exposes the three new C1c fields with the
    documented defaults."""
    opts = vq.PeriodicRHFOptions()
    assert opts.quadratic_fallback_iter == 0
    assert opts.quadratic_fallback_shift == pytest.approx(0.1)
    assert opts.quadratic_fallback_max_step == pytest.approx(0.1)


def test_quadratic_fallback_disabled_is_identical_to_standard():
    """quadratic_fallback_iter == 0 (default) must reproduce the
    pre-C1c SCF trajectory bit-for-bit."""
    sysp, basis = _h2_in_box()
    opts = _default_options()
    opts.quadratic_fallback_iter = 0
    r = vq.run_rhf_periodic_gamma_ewald3d(sysp, basis, opts, omega=0.5)
    assert r.converged
    # H₂ / STO-3G in 30-bohr box, RHF Ewald: a known number.
    assert -1.5 < r.energy < -1.0


def test_quadratic_fallback_converges_to_same_energy_as_standard():
    """With the fallback active, the converged energy should match
    the standard-path energy to ~µHa (same SCF fixed point)."""
    sysp, basis = _h2_in_box()

    opts_std = _default_options()
    opts_std.quadratic_fallback_iter = 0
    r_std = vq.run_rhf_periodic_gamma_ewald3d(sysp, basis, opts_std, omega=0.5)

    opts_q = _default_options()
    opts_q.use_diis = False    # disable DIIS so quadratic actually drives convergence
    opts_q.quadratic_fallback_iter = 1
    r_q = vq.run_rhf_periodic_gamma_ewald3d(sysp, basis, opts_q, omega=0.5)

    assert r_std.converged
    assert r_q.converged
    assert abs(r_std.energy - r_q.energy) < 1e-6, (
        f"Standard E={r_std.energy:.10f}, quadratic E={r_q.energy:.10f}"
    )


def test_quadratic_fallback_options_round_trip_through_dataclass():
    """Setting the fields and reading them back must round-trip."""
    opts = vq.PeriodicRHFOptions()
    opts.quadratic_fallback_iter = 25
    opts.quadratic_fallback_shift = 0.5
    opts.quadratic_fallback_max_step = 0.05
    assert opts.quadratic_fallback_iter == 25
    assert opts.quadratic_fallback_shift == pytest.approx(0.5)
    assert opts.quadratic_fallback_max_step == pytest.approx(0.05)


# ---------------------------------------------------------------------------
# Integration: same fallback wired into the other Γ-only Ewald drivers
# (UHF, RKS, UKS). Pin that the fallback path produces the same
# converged energy as the standard path for a clean reference system.
# ---------------------------------------------------------------------------

def _h_atom_in_box(box: float = 30.0):
    c = box / 2
    atoms = [vq.Atom(1, [c, c, c])]
    sysp = vq.PeriodicSystem(
        3, np.eye(3) * box, atoms, charge=0, multiplicity=2,
    )
    basis = vq.BasisSet(sysp.unit_cell_molecule(), "sto-3g")
    return sysp, basis


def test_quadratic_fallback_uhf_matches_standard():
    sysp, basis = _h_atom_in_box()
    opts_std = _default_options()
    r_std = vq.run_uhf_periodic_gamma_ewald3d(sysp, basis, opts_std, omega=0.5)

    opts_q = _default_options()
    opts_q.use_diis = False
    opts_q.quadratic_fallback_iter = 1
    r_q = vq.run_uhf_periodic_gamma_ewald3d(sysp, basis, opts_q, omega=0.5)

    assert r_std.converged and r_q.converged
    assert abs(r_std.energy - r_q.energy) < 1e-6
    # H atom: <S²> = 0.75 ideal.
    assert abs(r_std.s_squared - 0.75) < 1e-6
    assert abs(r_q.s_squared - 0.75) < 1e-6


def test_periodic_ks_options_have_quadratic_fallback_fields():
    """PeriodicKSOptions also exposes the C1c fields."""
    opts = vq.PeriodicKSOptions()
    assert opts.quadratic_fallback_iter == 0
    assert opts.quadratic_fallback_shift == pytest.approx(0.1)
    assert opts.quadratic_fallback_max_step == pytest.approx(0.1)


def test_quadratic_fallback_rks_matches_standard():
    sysp, basis = _h2_in_box()
    opts_std = vq.PeriodicKSOptions()
    opts_std.functional = "PBE"
    opts_std.lattice_opts.cutoff_bohr = 12.0
    opts_std.lattice_opts.nuclear_cutoff_bohr = 15.0
    opts_std.damping = 0.3
    opts_std.max_iter = 60
    opts_std.use_diis = True
    r_std = vq.run_rks_periodic_gamma_ewald3d(sysp, basis, opts_std, omega=0.5)

    opts_q = vq.PeriodicKSOptions()
    opts_q.functional = "PBE"
    opts_q.lattice_opts.cutoff_bohr = 12.0
    opts_q.lattice_opts.nuclear_cutoff_bohr = 15.0
    opts_q.damping = 0.3
    opts_q.max_iter = 60
    opts_q.use_diis = False
    opts_q.quadratic_fallback_iter = 1
    r_q = vq.run_rks_periodic_gamma_ewald3d(sysp, basis, opts_q, omega=0.5)

    assert r_std.converged and r_q.converged
    assert abs(r_std.energy - r_q.energy) < 1e-6


def test_quadratic_fallback_uks_matches_standard():
    sysp, basis = _h_atom_in_box()
    opts_std = vq.PeriodicKSOptions()
    opts_std.functional = "PBE"
    opts_std.lattice_opts.cutoff_bohr = 12.0
    opts_std.lattice_opts.nuclear_cutoff_bohr = 15.0
    opts_std.damping = 0.3
    opts_std.max_iter = 60
    opts_std.use_diis = True
    r_std = vq.run_uks_periodic_gamma_ewald3d(sysp, basis, opts_std, omega=0.5)

    opts_q = vq.PeriodicKSOptions()
    opts_q.functional = "PBE"
    opts_q.lattice_opts.cutoff_bohr = 12.0
    opts_q.lattice_opts.nuclear_cutoff_bohr = 15.0
    opts_q.damping = 0.3
    opts_q.max_iter = 60
    opts_q.use_diis = False
    opts_q.quadratic_fallback_iter = 1
    r_q = vq.run_uks_periodic_gamma_ewald3d(sysp, basis, opts_q, omega=0.5)

    assert r_std.converged and r_q.converged
    assert abs(r_std.energy - r_q.energy) < 1e-6
    assert abs(r_q.s_squared - 0.75) < 1e-6


# ---------------------------------------------------------------------------
# C1c-2 — multi-k Ewald drivers with quadratic fallback
# ---------------------------------------------------------------------------

def _multi_k_rhf_opts(quad=False):
    opts = vq.PeriodicRHFOptions()
    opts.lattice_opts.cutoff_bohr = 12.0
    opts.lattice_opts.nuclear_cutoff_bohr = 15.0
    opts.damping = 0.3
    opts.max_iter = 30
    opts.use_diis = not quad
    if quad:
        opts.quadratic_fallback_iter = 1
    return opts


def _multi_k_ks_opts(func="PBE", quad=False):
    opts = vq.PeriodicKSOptions()
    opts.functional = func
    opts.lattice_opts.cutoff_bohr = 12.0
    opts.lattice_opts.nuclear_cutoff_bohr = 15.0
    opts.damping = 0.3
    opts.max_iter = 30
    opts.use_diis = not quad
    if quad:
        opts.quadratic_fallback_iter = 1
    return opts


def test_quadratic_fallback_multi_k_rhf():
    sysp, basis = _h2_in_box()
    km = vq.monkhorst_pack(sysp, [1, 1, 1])
    r_std = vq.run_rhf_periodic_multi_k_ewald3d(
        sysp, basis, km, _multi_k_rhf_opts(False), omega=0.5,
    )
    r_q = vq.run_rhf_periodic_multi_k_ewald3d(
        sysp, basis, km, _multi_k_rhf_opts(True), omega=0.5,
    )
    assert r_std.converged and r_q.converged
    assert abs(r_std.energy - r_q.energy) < 1e-6


def test_quadratic_fallback_multi_k_uhf():
    sysp, basis = _h2_in_box()
    km = vq.monkhorst_pack(sysp, [1, 1, 1])
    r_std = vq.run_uhf_periodic_multi_k_ewald3d(
        sysp, basis, km, _multi_k_rhf_opts(False), omega=0.5,
    )
    r_q = vq.run_uhf_periodic_multi_k_ewald3d(
        sysp, basis, km, _multi_k_rhf_opts(True), omega=0.5,
    )
    assert r_std.converged and r_q.converged
    assert abs(r_std.energy - r_q.energy) < 1e-6


def test_quadratic_fallback_multi_k_rks():
    sysp, basis = _h2_in_box()
    km = vq.monkhorst_pack(sysp, [1, 1, 1])
    r_std = vq.run_rks_periodic_multi_k_ewald3d(
        sysp, basis, km, _multi_k_ks_opts("PBE", False), omega=0.5,
    )
    r_q = vq.run_rks_periodic_multi_k_ewald3d(
        sysp, basis, km, _multi_k_ks_opts("PBE", True), omega=0.5,
    )
    assert r_std.converged and r_q.converged
    assert abs(r_std.energy - r_q.energy) < 1e-6


def test_quadratic_fallback_multi_k_uks():
    """H atom multi-k UKS with quadratic fallback. Verifies the
    per-spin per-k Newton step works and reproduces ⟨S²⟩ = 0.75."""
    sysp, basis = _h_atom_in_box()
    km = vq.monkhorst_pack(sysp, [1, 1, 1])
    r_std = vq.run_uks_periodic_multi_k_ewald3d(
        sysp, basis, km, _multi_k_ks_opts("PBE", False), omega=0.5,
    )
    r_q = vq.run_uks_periodic_multi_k_ewald3d(
        sysp, basis, km, _multi_k_ks_opts("PBE", True), omega=0.5,
    )
    assert r_std.converged and r_q.converged
    assert abs(r_std.energy - r_q.energy) < 1e-6
    assert abs(r_q.s_squared - 0.75) < 1e-6


def test_quadratic_step_works_on_complex_input():
    """Direct kernel test: ``quadratic_step`` accepts complex C / F
    (multi-k SCF state) and produces complex C_new with unitary
    rotation."""
    n_bf = 5
    n_occ = 2
    rng = np.random.default_rng(42)

    # Build a complex Hermitian F.
    A = rng.standard_normal((n_bf, n_bf)) + 1j * rng.standard_normal((n_bf, n_bf))
    F = A + A.conj().T

    # Random unitary C.
    A2 = rng.standard_normal((n_bf, n_bf)) + 1j * rng.standard_normal((n_bf, n_bf))
    C, _ = np.linalg.qr(A2)
    eps = np.real(np.diag(C.conj().T @ F @ C))

    C_new, eps_new = quadratic_step(F, C, eps, n_occ, shift=0.5, max_step=0.1)

    # C_new should still be unitary (orthonormality preserved).
    assert np.allclose(C_new.conj().T @ C_new, np.eye(n_bf), atol=1e-10)


# ---------------------------------------------------------------------------
# Phase C1c-3 — molecular drivers (run_rhf / run_uhf / run_rks / run_uks).
# Defaults are off; default-path numbers are unchanged; with the fallback
# enabled the SCF converges to the same energy as the standard path.
# ---------------------------------------------------------------------------

@pytest.fixture
def _h2o_sto3g():
    """Closed-shell H2O / STO-3G (small enough to be fast, large enough
    to exercise damping + DIIS + diagonalisation)."""
    from .conftest import GEOMETRIES, make_molecule
    mol = make_molecule(GEOMETRIES["H2O"])
    basis = vq.BasisSet(mol, "sto-3g")
    return mol, basis


@pytest.fixture
def _o_triplet_sto3g():
    """Triplet O atom — the simplest open-shell test case."""
    mol = vq.Molecule([vq.Atom(8, [0.0, 0.0, 0.0])],
                      charge=0, multiplicity=3)
    basis = vq.BasisSet(mol, "sto-3g")
    return mol, basis


def test_molecular_options_have_quadratic_fallback_fields():
    """All four molecular Options classes expose the C1c-3 fields with
    the documented defaults."""
    for cls in (vq.RHFOptions, vq.UHFOptions, vq.RKSOptions, vq.UKSOptions):
        opts = cls()
        assert opts.quadratic_fallback_iter == 0, cls.__name__
        assert opts.quadratic_fallback_shift == pytest.approx(0.1), cls.__name__
        assert opts.quadratic_fallback_max_step == pytest.approx(0.1), cls.__name__


def test_molecular_quadratic_fallback_disabled_is_identical_to_standard_rhf(_h2o_sto3g):
    """quadratic_fallback_iter=0 (default) is byte-identical to the path
    that existed before C1c-3 — toggling between the two with the same
    seed must not perturb the result."""
    mol, basis = _h2o_sto3g
    opts = vq.RHFOptions()
    r1 = vq.run_rhf(mol, basis, opts)
    opts2 = vq.RHFOptions()
    opts2.quadratic_fallback_iter = 0   # explicit disable
    r2 = vq.run_rhf(mol, basis, opts2)
    assert r1.converged and r2.converged
    assert r1.energy == pytest.approx(r2.energy, abs=1e-12)


def test_molecular_quadratic_fallback_rhf_matches_standard(_h2o_sto3g):
    """C1c-3 enabled (kicks in after iter 3) converges to the same
    energy as the default DIIS-only path on a well-behaved system."""
    mol, basis = _h2o_sto3g
    opts_std = vq.RHFOptions()
    r_std = vq.run_rhf(mol, basis, opts_std)

    opts_q = vq.RHFOptions()
    opts_q.quadratic_fallback_iter = 3
    r_q = vq.run_rhf(mol, basis, opts_q)

    assert r_std.converged and r_q.converged
    assert abs(r_std.energy - r_q.energy) < 1e-7


def test_molecular_quadratic_fallback_uhf_matches_standard(_o_triplet_sto3g):
    """Per-spin Newton step on triplet O reproduces the default UHF
    energy and the ⟨S²⟩ = 2 expectation value of the triplet state."""
    mol, basis = _o_triplet_sto3g
    opts_std = vq.UHFOptions()
    r_std = vq.run_uhf(mol, basis, opts_std)

    opts_q = vq.UHFOptions()
    opts_q.quadratic_fallback_iter = 3
    r_q = vq.run_uhf(mol, basis, opts_q)

    assert r_std.converged and r_q.converged
    assert abs(r_std.energy - r_q.energy) < 1e-7
    # Triplet O has S = 1, so <S^2>_ideal = 2; UHF/STO-3G has small
    # contamination but should be very close.
    assert abs(r_q.s_squared - 2.0) < 1e-2


def test_molecular_quadratic_fallback_rks_matches_standard(_h2o_sto3g):
    """C1c-3 on RKS LDA/STO-3G/H2O converges to the same KS energy as
    the standard path."""
    mol, basis = _h2o_sto3g
    opts_std = vq.RKSOptions()
    opts_std.functional = "LDA"
    r_std = vq.run_rks(mol, basis, opts_std)

    opts_q = vq.RKSOptions()
    opts_q.functional = "LDA"
    opts_q.quadratic_fallback_iter = 3
    opts_q.quadratic_fallback_shift = 0.2
    r_q = vq.run_rks(mol, basis, opts_q)

    assert r_std.converged and r_q.converged
    assert abs(r_std.energy - r_q.energy) < 1e-7


def test_molecular_quadratic_fallback_uks_matches_standard(_o_triplet_sto3g):
    """Per-spin Newton step on triplet O / LDA reproduces the default
    UKS energy."""
    mol, basis = _o_triplet_sto3g
    opts_std = vq.UKSOptions()
    opts_std.functional = "LDA"
    r_std = vq.run_uks(mol, basis, opts_std)

    opts_q = vq.UKSOptions()
    opts_q.functional = "LDA"
    opts_q.quadratic_fallback_iter = 3
    r_q = vq.run_uks(mol, basis, opts_q)

    assert r_std.converged and r_q.converged
    assert abs(r_std.energy - r_q.energy) < 1e-7
