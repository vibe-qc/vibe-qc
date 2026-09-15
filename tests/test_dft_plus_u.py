"""DFT+U Increment 1 — occupation-matrix machinery.

Pinned contracts for [dft_plus_u.py](../python/vibeqc/dft_plus_u.py):

1. ``HubbardSite`` dataclass invariants — frozen, validated,
   ``U_eff = U - J`` in both eV and Hartree.

2. ``ao_group_indices`` — partitions AO indices by
   ``(atom_index, l)`` for a spherical Gaussian basis; rejects
   Cartesian shells with a clean ``NotImplementedError``.

3. ``compute_occupation_matrices`` — ``n^A_l = (S P S)_{(A,l),(A,l)}``
   on hand-constructed P/S and on a real SCF density matrix; the
   per-channel blocks tile to ``tr(S P S)``.

4. ``compute_dudarev_energy`` — closed-form on diagonal n's, agrees
   with the eigenvalue formula ``(U_eff/2) sum_m (λ - λ²)``, sums
   over multiple sites correctly, zero for integer occupations.
"""

from __future__ import annotations

import ast
from pathlib import Path

import numpy as np
import pytest

import vibeqc as vq
from vibeqc._vibeqc_core import (
    _compute_dft_plus_u_cxx,
    _HubbardSiteCxx,
)
from vibeqc.dft_plus_u import _EV_TO_HARTREE, _HARTREE_TO_EV


# ---------------------------------------------------------------------------
# 1. HubbardSite dataclass
# ---------------------------------------------------------------------------


def test_hubbard_site_construction_and_defaults():
    site = vq.HubbardSite(atom_index=0, l=2, U_ev=4.0)
    assert site.atom_index == 0
    assert site.l == 2
    assert site.U_ev == 4.0
    assert site.J_ev == 0.0
    assert site.U_eff_ev == 4.0
    assert site.U_eff_hartree == pytest.approx(4.0 / _HARTREE_TO_EV)


def test_hubbard_site_u_eff_subtracts_j():
    site = vq.HubbardSite(atom_index=2, l=3, U_ev=6.0, J_ev=1.0)
    assert site.U_eff_ev == pytest.approx(5.0)
    assert site.U_eff_hartree == pytest.approx(5.0 * _EV_TO_HARTREE)


def test_hubbard_site_is_frozen():
    site = vq.HubbardSite(0, 2, 4.0)
    with pytest.raises(Exception):
        site.U_ev = 5.0  # type: ignore[misc]


def test_hubbard_site_rejects_negative_atom_index():
    with pytest.raises(ValueError, match="atom_index"):
        vq.HubbardSite(atom_index=-1, l=2, U_ev=4.0)


def test_hubbard_site_rejects_negative_l():
    with pytest.raises(ValueError, match="l must be"):
        vq.HubbardSite(atom_index=0, l=-1, U_ev=4.0)


# ---------------------------------------------------------------------------
# 2. ao_group_indices on a real Gaussian basis
# ---------------------------------------------------------------------------


def _h2_basis():
    mol = vq.Molecule(
        [vq.Atom(1, [0.0, 0.0, 0.0]), vq.Atom(1, [0.0, 0.0, 1.4])],
        charge=0,
        multiplicity=1,
    )
    return mol, vq.BasisSet(mol, "sto-3g")


def _h2o_basis():
    mol = vq.Molecule(
        [
            vq.Atom(8, [0.0, 0.0, 0.0]),
            vq.Atom(1, [0.0, 1.43, 1.11]),
            vq.Atom(1, [0.0, -1.43, 1.11]),
        ],
        charge=0,
        multiplicity=1,
    )
    return mol, vq.BasisSet(mol, "sto-3g")


def test_ao_group_indices_h2_sto3g():
    """H2/STO-3G has one s shell per atom → two single-AO channels."""
    _, basis = _h2_basis()
    groups = vq.ao_group_indices(basis)
    assert groups == {(0, 0): [0], (1, 0): [1]}


def test_ao_group_indices_h2o_sto3g():
    """O/STO-3G: 1s + 2s + 2p (3 shells, 5 AOs); H/STO-3G: 1s.

    Expected:
      atom 0 (O) → l=0: [0, 1] (two s shells), l=1: [2, 3, 4] (p shell)
      atom 1 (H) → l=0: [5]
      atom 2 (H) → l=0: [6]
    """
    _, basis = _h2o_basis()
    groups = vq.ao_group_indices(basis)
    assert groups[(0, 0)] == [0, 1]
    assert groups[(0, 1)] == [2, 3, 4]
    assert groups[(1, 0)] == [5]
    assert groups[(2, 0)] == [6]
    assert (0, 2) not in groups  # no d shell on O in STO-3G


def test_ao_group_indices_partition_is_total():
    """The (atom, l) blocks must tile the full AO basis exactly."""
    _, basis = _h2o_basis()
    groups = vq.ao_group_indices(basis)
    all_indices = sorted(i for idxs in groups.values() for i in idxs)
    nbf = sum(2 * int(sh.l) + 1 for sh in basis.shells())
    assert all_indices == list(range(nbf))


# ---------------------------------------------------------------------------
# 3. compute_occupation_matrices
# ---------------------------------------------------------------------------


def test_occupation_matrix_orthonormal_basis():
    """With S = I, n^A_l reduces to the AO-block of P directly."""
    P = np.diag([1.0, 0.7, 0.3, 0.0, 0.5])
    S = np.eye(5)
    ao_groups = {(0, 0): [0], (0, 1): [1, 2, 3], (1, 0): [4]}
    sites = [
        vq.HubbardSite(0, 1, U_ev=4.0),
        vq.HubbardSite(1, 0, U_ev=2.0),
    ]
    occs = vq.compute_occupation_matrices(sites, P, S, ao_groups)
    np.testing.assert_allclose(occs[(0, 1)], np.diag([0.7, 0.3, 0.0]))
    np.testing.assert_allclose(occs[(1, 0)], np.array([[0.5]]))


def test_occupation_matrix_two_ao_non_orthogonal():
    """Hand-computed 2-AO case: n = (S P S)_block."""
    s = 0.4
    S = np.array([[1.0, s], [s, 1.0]])
    P = np.array([[1.2, 0.3], [0.3, 0.8]])
    ao_groups = {(0, 0): [0, 1]}
    sites = [vq.HubbardSite(0, 0, U_ev=4.0)]
    occs = vq.compute_occupation_matrices(sites, P, S, ao_groups)
    expected = S @ P @ S
    np.testing.assert_allclose(occs[(0, 0)], expected)


def test_occupation_matrix_block_traces_tile_full_trace():
    """Sum of tr(n^A_l) over all channels = tr(S P S) = tr(P S²).

    The (atom, l) blocks partition the AO basis, so their per-block
    traces must sum to the full diagonal of S P S.
    """
    rng = np.random.default_rng(2026_05_25)
    P_raw = rng.normal(size=(5, 5))
    P = 0.5 * (P_raw + P_raw.T)
    S_raw = rng.normal(size=(5, 5))
    S = 0.5 * (S_raw @ S_raw.T) + np.eye(5)  # symmetric PD overlap
    ao_groups = {(0, 0): [0], (0, 1): [1, 2, 3], (1, 0): [4]}
    sites = [
        vq.HubbardSite(0, 0, U_ev=4.0),
        vq.HubbardSite(0, 1, U_ev=4.0),
        vq.HubbardSite(1, 0, U_ev=4.0),
    ]
    occs = vq.compute_occupation_matrices(sites, P, S, ao_groups)
    block_trace_sum = sum(float(np.trace(n)) for n in occs.values())
    assert block_trace_sum == pytest.approx(float(np.trace(S @ P @ S)))


def test_occupation_matrix_missing_channel_raises():
    P = np.eye(2)
    S = np.eye(2)
    ao_groups = {(0, 0): [0], (1, 0): [1]}
    sites = [vq.HubbardSite(0, 2, U_ev=4.0)]  # asks for d on an s-only atom
    with pytest.raises(KeyError, match="No AOs"):
        vq.compute_occupation_matrices(sites, P, S, ao_groups)


def test_occupation_matrix_wrong_shape_raises():
    P = np.eye(2)
    S = np.eye(3)
    with pytest.raises(ValueError, match="square"):
        vq.compute_occupation_matrices([], P, S, {})


def test_occupation_matrix_from_real_scf_h2():
    """End-to-end smoke: H2/STO-3G, converge RHF, project n on each H.

    For closed-shell H2 the AO density is symmetric under atom-swap;
    n^{H1}_s and n^{H2}_s must be equal scalars.
    """
    mol, basis = _h2_basis()
    result = vq.run_rhf(mol, basis)
    assert result.converged
    P = np.asarray(result.density)
    S = np.asarray(vq.compute_overlap(basis))
    ao_groups = vq.ao_group_indices(basis)
    sites = [vq.HubbardSite(0, 0, U_ev=4.0), vq.HubbardSite(1, 0, U_ev=4.0)]
    occs = vq.compute_occupation_matrices(sites, P, S, ao_groups)
    assert occs[(0, 0)].shape == (1, 1)
    assert occs[(1, 0)].shape == (1, 1)
    n0 = float(occs[(0, 0)].item())
    n1 = float(occs[(1, 0)].item())
    assert n0 == pytest.approx(n1, abs=1e-10)
    # Closed-shell H2 has ~1 electron per H — tr(n) tiles tr(S P S)
    # and the per-atom contributions are equal by symmetry.
    assert n0 > 0.0


# ---------------------------------------------------------------------------
# 4. compute_dudarev_energy
# ---------------------------------------------------------------------------


def test_dudarev_energy_zero_for_integer_occupations():
    """A diagonal n with eigenvalues ∈ {0, 1} gives E_U = 0 exactly."""
    n = np.diag([1.0, 1.0, 0.0, 0.0, 1.0])  # fully gapped d-shell
    site = vq.HubbardSite(0, 2, U_ev=4.0)
    E_U = vq.compute_dudarev_energy([site], {(0, 2): n})
    assert E_U == pytest.approx(0.0, abs=1e-14)


def test_dudarev_energy_half_filled_partial():
    """Diagonal n = diag(0.5, 0.5): E_U = (U_eff/2)(1 - 0.5) = U_eff/4."""
    n = np.diag([0.5, 0.5])
    site = vq.HubbardSite(0, 0, U_ev=4.0)  # U_eff = 4 eV
    E_U = vq.compute_dudarev_energy([site], {(0, 0): n})
    expected_hartree = (4.0 * _EV_TO_HARTREE) / 4.0
    assert E_U == pytest.approx(expected_hartree)


def test_dudarev_energy_matches_eigenvalue_formula():
    """E_U = (U_eff/2) sum(λ_m - λ_m²) on a random Hermitian n."""
    rng = np.random.default_rng(42)
    n_raw = rng.normal(size=(5, 5))
    n = 0.5 * (n_raw + n_raw.T)
    site = vq.HubbardSite(0, 2, U_ev=3.5, J_ev=0.5)  # U_eff = 3.0 eV
    E_direct = vq.compute_dudarev_energy([site], {(0, 2): n})
    eigvals = np.linalg.eigvalsh(n)
    E_eig = 0.5 * site.U_eff_hartree * float(
        np.sum(eigvals) - np.sum(eigvals**2)
    )
    assert E_direct == pytest.approx(E_eig, rel=1e-12)


def test_dudarev_energy_sums_multiple_sites():
    """Multi-site E_U is the additive sum of per-site contributions."""
    n0 = np.diag([0.5, 0.5])
    n1 = np.diag([0.25, 0.75])
    site0 = vq.HubbardSite(0, 0, U_ev=4.0)
    site1 = vq.HubbardSite(1, 0, U_ev=2.0)
    occs = {(0, 0): n0, (1, 0): n1}
    E_both = vq.compute_dudarev_energy([site0, site1], occs)
    E0 = vq.compute_dudarev_energy([site0], occs)
    E1 = vq.compute_dudarev_energy([site1], occs)
    assert E_both == pytest.approx(E0 + E1)


def test_dudarev_energy_zero_when_u_eff_zero():
    """U = J cancels: U_eff = 0, so E_U = 0 regardless of n."""
    n = np.diag([0.3, 0.7, 0.5])
    site = vq.HubbardSite(0, 1, U_ev=4.0, J_ev=4.0)
    E_U = vq.compute_dudarev_energy([site], {(0, 1): n})
    assert E_U == pytest.approx(0.0)


# ---------------------------------------------------------------------------
# 5. C++ kernel parity (Increment 2a)
#
# The C++ kernel in cpp/src/dft_plus_u.cpp is the implementation the SCF
# Fock builder will call from Increment 2b onward. It must agree with
# the NumPy reference at machine precision so we have one source of
# truth for the math.
# ---------------------------------------------------------------------------


def _v_u_from_numpy(sites, P, S, ao_groups_by_site):
    """NumPy reference: assemble the variational Fock contribution
    ``S × V_AO × S`` where ``V_AO^A_{mm'} = U_eff (δ_{mm'}/2 − n^A_l_{mm'})``
    is the Dudarev per-site V scattered into the AO basis.

    The S-sandwich is the variational form ``∂E_U/∂P`` for the
    standard Dudarev energy ``E_U = (U/2)(tr n − tr n²)`` with
    ``n = (S P S)_(A,l)`` in a **non-orthogonal AO basis**. Without
    the sandwich, the SCF converges to a non-variational stationary
    point and the analytic gradient mismatches FD (see
    docs/user_guide/dft_plus_u.md). Pins the C++ kernel's
    ``vibeqc::compute_dft_plus_u`` output (which returns the same
    variational form).
    """
    nbf = P.shape[0]
    V_AO = np.zeros((nbf, nbf))
    SPS = S @ P @ S
    for site, idx in zip(sites, ao_groups_by_site):
        idx_arr = np.asarray(idx, dtype=np.int64)
        n = SPS[np.ix_(idx_arr, idx_arr)]
        V_local = -site.U_eff_hartree * n
        V_local[np.diag_indices_from(V_local)] += 0.5 * site.U_eff_hartree
        V_AO[np.ix_(idx_arr, idx_arr)] += V_local
    V_AO = 0.5 * (V_AO + V_AO.T)
    # S-sandwich for variational consistency in non-orthogonal AO basis.
    return S @ V_AO @ S


def _to_cxx(sites):
    return [
        _HubbardSiteCxx(s.atom_index, s.l, s.U_eff_hartree) for s in sites
    ]


def test_cxx_kernel_no_sites_returns_zero():
    """Empty sites list — kernel must return zero energy + zero V."""
    P = np.eye(4)
    S = np.eye(4)
    E, V = _compute_dft_plus_u_cxx([], [], P, S)
    assert E == 0.0
    np.testing.assert_array_equal(V, np.zeros((4, 4)))


def test_cxx_kernel_matches_numpy_orthonormal_basis():
    """C++ ≡ NumPy on diagonal P with S = I."""
    P = np.diag([0.5, 0.3, 0.7, 1.0, 0.2])
    S = np.eye(5)
    sites = [
        vq.HubbardSite(0, 1, U_ev=4.0),
        vq.HubbardSite(1, 0, U_ev=2.0),
    ]
    ao_groups = [[0, 1, 2], [3]]
    sites_dict = {(0, 1): np.diag([0.5, 0.3, 0.7]), (1, 0): np.array([[1.0]])}
    E_np = vq.compute_dudarev_energy(sites, sites_dict)
    V_np = _v_u_from_numpy(sites, P, S, ao_groups)
    E_cxx, V_cxx = _compute_dft_plus_u_cxx(_to_cxx(sites), ao_groups, P, S)
    assert E_cxx == pytest.approx(E_np, rel=1e-14, abs=1e-14)
    np.testing.assert_allclose(V_cxx, V_np, rtol=1e-14, atol=1e-14)


def test_cxx_kernel_matches_numpy_non_orthogonal():
    """C++ ≡ NumPy on a random symmetric P and PD overlap S."""
    rng = np.random.default_rng(2026_05_25)
    nbf = 7
    P_raw = rng.normal(size=(nbf, nbf))
    P = 0.5 * (P_raw + P_raw.T)
    S_raw = rng.normal(size=(nbf, nbf))
    S = 0.5 * (S_raw @ S_raw.T) + np.eye(nbf)
    sites = [
        vq.HubbardSite(0, 2, U_ev=5.0, J_ev=0.5),  # U_eff = 4.5 eV
        vq.HubbardSite(1, 1, U_ev=3.0),
    ]
    ao_groups = [[0, 1, 2, 3, 4], [5, 6]]
    SPS = S @ P @ S
    occs = {
        (0, 2): SPS[np.ix_([0, 1, 2, 3, 4], [0, 1, 2, 3, 4])],
        (1, 1): SPS[np.ix_([5, 6], [5, 6])],
    }
    E_np = vq.compute_dudarev_energy(sites, occs)
    V_np = _v_u_from_numpy(sites, P, S, ao_groups)
    E_cxx, V_cxx = _compute_dft_plus_u_cxx(_to_cxx(sites), ao_groups, P, S)
    assert E_cxx == pytest.approx(E_np, rel=1e-13, abs=1e-14)
    np.testing.assert_allclose(V_cxx, V_np, rtol=1e-13, atol=1e-14)


def test_cxx_kernel_v_u_is_symmetric():
    """V_U is symmetric in real basis — the defensive symmetrization
    in the C++ kernel pins this contract."""
    rng = np.random.default_rng(7)
    nbf = 6
    P_raw = rng.normal(size=(nbf, nbf))
    P = 0.5 * (P_raw + P_raw.T)
    S = np.eye(nbf) + 0.1 * rng.normal(size=(nbf, nbf))
    S = 0.5 * (S + S.T)
    S += np.eye(nbf) * 2  # make PD
    sites = [vq.HubbardSite(0, 2, U_ev=4.0)]
    ao_groups = [[0, 1, 2, 3, 4]]
    _, V = _compute_dft_plus_u_cxx(_to_cxx(sites), ao_groups, P, S)
    np.testing.assert_allclose(V, V.T, rtol=0.0, atol=1e-14)


def test_cxx_kernel_parallel_array_length_mismatch_raises():
    """sites and ao_groups must be parallel — kernel rejects mismatch."""
    sites = [vq.HubbardSite(0, 1, U_ev=4.0)]
    ao_groups = []  # length 0, sites length 1
    P = np.eye(3)
    S = np.eye(3)
    with pytest.raises(ValueError, match="parallel arrays"):
        _compute_dft_plus_u_cxx(_to_cxx(sites), ao_groups, P, S)


def test_cxx_kernel_wrong_shape_raises():
    """P and S must be square and equal-shape."""
    sites = [vq.HubbardSite(0, 0, U_ev=4.0)]
    ao_groups = [[0]]
    with pytest.raises(ValueError, match="square"):
        _compute_dft_plus_u_cxx(_to_cxx(sites), ao_groups, np.eye(2),
                                 np.eye(3))


def test_cxx_kernel_matches_numpy_on_real_scf_density():
    """End-to-end parity through a real RHF run.

    Confirms the C++ kernel agrees with the NumPy reference on a
    density matrix that comes out of the actual SCF, not just synthetic
    test inputs — i.e. the full pipeline (basis → ao_group_indices →
    Eigen marshalling → kernel → result) is wired correctly.
    """
    mol = vq.Molecule(
        [
            vq.Atom(8, [0.0, 0.0, 0.0]),
            vq.Atom(1, [0.0, 1.43, 1.11]),
            vq.Atom(1, [0.0, -1.43, 1.11]),
        ],
        charge=0,
        multiplicity=1,
    )
    basis = vq.BasisSet(mol, "sto-3g")
    result = vq.run_rhf(mol, basis)
    P = np.asarray(result.density)
    S = np.asarray(vq.compute_overlap(basis))
    ao_groups_dict = vq.ao_group_indices(basis)
    sites = [
        vq.HubbardSite(0, 0, U_ev=4.0),
        vq.HubbardSite(0, 1, U_ev=4.0, J_ev=0.5),
        vq.HubbardSite(1, 0, U_ev=2.0),
    ]
    ao_groups_list = [ao_groups_dict[(s.atom_index, s.l)] for s in sites]
    occs = vq.compute_occupation_matrices(sites, P, S, ao_groups_dict)
    E_np = vq.compute_dudarev_energy(sites, occs)
    V_np = _v_u_from_numpy(sites, P, S, ao_groups_list)
    E_cxx, V_cxx = _compute_dft_plus_u_cxx(_to_cxx(sites), ao_groups_list,
                                            P, S)
    assert E_cxx == pytest.approx(E_np, rel=1e-13, abs=1e-14)
    np.testing.assert_allclose(V_cxx, V_np, rtol=1e-13, atol=1e-13)


# ---------------------------------------------------------------------------
# 6. RHF SCF integration (Increment 2b)
#
# The C++ SCF Fock-build hook in cpp/src/rhf.cpp:278+ adds V_U to the AO
# Fock per iteration when ``RHFOptions.dft_plus_u_sites`` is non-empty.
# These tests pin the end-to-end contract through the Python wrapper.
# ---------------------------------------------------------------------------


def test_run_rhf_no_dft_plus_u_unchanged():
    """Baseline: omitting dft_plus_u must give the same SCF result as
    the bare C++ run_rhf — no spurious side-effects."""
    mol = vq.Molecule(
        [vq.Atom(1, [0.0, 0.0, 0.0]), vq.Atom(1, [0.0, 0.0, 1.4])],
        charge=0, multiplicity=1,
    )
    basis = vq.BasisSet(mol, "cc-pvdz")
    r_a = vq.run_rhf(mol, basis)
    r_b = vq.run_rhf(mol, basis, dft_plus_u=[])
    assert r_a.converged and r_b.converged
    assert r_a.energy == pytest.approx(r_b.energy, rel=0.0, abs=1e-14)
    assert r_a.e_dft_plus_u == 0.0
    assert r_b.e_dft_plus_u == 0.0


def test_run_rhf_dft_plus_u_zero_U_is_no_op():
    """U=0 (or J=U) gives U_eff=0, which makes V_U=0 and E_U=0
    irrespective of the occupation matrix. SCF energy must match the
    no-+U baseline at machine precision."""
    mol = vq.Molecule(
        [vq.Atom(1, [0.0, 0.0, 0.0]), vq.Atom(1, [0.0, 0.0, 1.4])],
        charge=0, multiplicity=1,
    )
    basis = vq.BasisSet(mol, "cc-pvdz")
    baseline = vq.run_rhf(mol, basis)
    with_zero_u = vq.run_rhf(
        mol, basis,
        dft_plus_u=[vq.HubbardSite(0, 1, U_ev=4.0, J_ev=4.0)],
    )
    assert with_zero_u.converged
    assert with_zero_u.e_dft_plus_u == pytest.approx(0.0, abs=1e-12)
    assert with_zero_u.energy == pytest.approx(baseline.energy,
                                                 rel=0.0, abs=1e-10)


def test_run_rhf_dft_plus_u_positive_energy_and_total_shift():
    """A small +U on H's p polarization channel of H2/cc-pVDZ creates
    a finite-positive Dudarev energy and shifts the total energy
    upward relative to the no-+U baseline."""
    mol = vq.Molecule(
        [vq.Atom(1, [0.0, 0.0, 0.0]), vq.Atom(1, [0.0, 0.0, 1.4])],
        charge=0, multiplicity=1,
    )
    basis = vq.BasisSet(mol, "cc-pvdz")
    baseline = vq.run_rhf(mol, basis)
    plus_u = vq.run_rhf(
        mol, basis,
        dft_plus_u=[vq.HubbardSite(0, 1, U_ev=4.0)],
    )
    assert plus_u.converged
    # Partial AO occupation on a p polarization channel ⇒ tr n − tr n²
    # is strictly positive, so E_U > 0 strictly.
    assert plus_u.e_dft_plus_u > 0.0
    assert plus_u.energy > baseline.energy
    # The reported decomposition must be self-consistent: total energy
    # equals (HF electronic) + nuclear + Dudarev, with the HF
    # electronic piece computed from the +U-free Fock.
    assert plus_u.energy == pytest.approx(
        plus_u.e_electronic + plus_u.e_dft_plus_u
        + (baseline.energy - baseline.e_electronic),
        rel=0.0, abs=5e-9,
    )


def test_run_rhf_dft_plus_u_energy_matches_python_reference():
    """The RHF-loop +U contribution must equal the per-spin reference
    formula applied to the converged density. This pins the closed-
    shell ``E_U = 2 × per-spin`` convention end-to-end."""
    mol = vq.Molecule(
        [vq.Atom(1, [0.0, 0.0, 0.0]), vq.Atom(1, [0.0, 0.0, 1.4])],
        charge=0, multiplicity=1,
    )
    basis = vq.BasisSet(mol, "cc-pvdz")
    sites = [vq.HubbardSite(0, 1, U_ev=4.0)]
    result = vq.run_rhf(mol, basis, dft_plus_u=sites)
    P = np.asarray(result.density)
    S = np.asarray(vq.compute_overlap(basis))
    ao_groups = vq.ao_group_indices(basis)
    # Per-spin convention: pass P_σ = P/2.
    occs_sigma = vq.compute_occupation_matrices(sites, 0.5 * P, S, ao_groups)
    E_U_per_spin = vq.compute_dudarev_energy(sites, occs_sigma)
    E_U_total_expected = 2.0 * E_U_per_spin  # closed-shell sum over spins
    assert result.e_dft_plus_u == pytest.approx(
        E_U_total_expected, rel=1e-9, abs=1e-12,
    )


def test_run_rhf_dft_plus_u_rejects_invalid_channel():
    """Asking for d-AOs on an H atom (sto-3g has only s) must error
    out cleanly at the Python wrapper, before any C++ call."""
    mol = vq.Molecule(
        [vq.Atom(1, [0.0, 0.0, 0.0]), vq.Atom(1, [0.0, 0.0, 1.4])],
        charge=0, multiplicity=1,
    )
    basis = vq.BasisSet(mol, "sto-3g")
    with pytest.raises(ValueError, match="no AOs"):
        vq.run_rhf(
            mol, basis,
            dft_plus_u=[vq.HubbardSite(0, 2, U_ev=4.0)],  # d on H/STO-3G
        )


# ---------------------------------------------------------------------------
# 7. RKS SCF integration (Increment 2c)
#
# RKS mirrors the RHF integration (Increment 2b) — same kernel, same
# per-spin convention, same energy-decomposition discipline (the
# Dudarev term is added separately to E_total, not folded into
# E_core + E_J + E_K + E_xc via the (1/2) tr(D F) trace).
# ---------------------------------------------------------------------------


def _h2_ccpvdz_mol_basis():
    mol = vq.Molecule(
        [vq.Atom(1, [0.0, 0.0, 0.0]), vq.Atom(1, [0.0, 0.0, 1.4])],
        charge=0, multiplicity=1,
    )
    return mol, vq.BasisSet(mol, "cc-pvdz")


def _rks_opts(functional: str = "lda"):
    opts = vq.RKSOptions()
    opts.functional = functional
    return opts


def test_run_rks_no_dft_plus_u_unchanged():
    """Baseline: omitting dft_plus_u must give the same SCF result as
    the bare C++ run_rks."""
    mol, basis = _h2_ccpvdz_mol_basis()
    r_a = vq.run_rks(mol, basis, _rks_opts("lda"))
    r_b = vq.run_rks(mol, basis, _rks_opts("lda"), dft_plus_u=[])
    assert r_a.converged and r_b.converged
    assert r_a.energy == pytest.approx(r_b.energy, rel=0.0, abs=1e-14)
    assert r_a.e_dft_plus_u == 0.0
    assert r_b.e_dft_plus_u == 0.0


def test_run_rks_dft_plus_u_zero_U_is_no_op():
    """U=J ⇒ U_eff=0 ⇒ E_U=0; energy matches the U-free baseline."""
    mol, basis = _h2_ccpvdz_mol_basis()
    baseline = vq.run_rks(mol, basis, _rks_opts("lda"))
    with_zero_u = vq.run_rks(
        mol, basis, _rks_opts("lda"),
        dft_plus_u=[vq.HubbardSite(0, 1, U_ev=4.0, J_ev=4.0)],
    )
    assert with_zero_u.converged
    assert with_zero_u.e_dft_plus_u == pytest.approx(0.0, abs=1e-12)
    assert with_zero_u.energy == pytest.approx(baseline.energy,
                                                 rel=0.0, abs=1e-10)


def test_run_rks_dft_plus_u_positive_energy_and_total_shift():
    """LDA on H2/cc-pVDZ + U on H's p-channel ⇒ E_U > 0 and the total
    energy shifts upward relative to the LDA baseline."""
    mol, basis = _h2_ccpvdz_mol_basis()
    baseline = vq.run_rks(mol, basis, _rks_opts("lda"))
    plus_u = vq.run_rks(
        mol, basis, _rks_opts("lda"),
        dft_plus_u=[vq.HubbardSite(0, 1, U_ev=4.0)],
    )
    assert plus_u.converged
    assert plus_u.e_dft_plus_u > 0.0
    assert plus_u.energy > baseline.energy


def test_run_rks_dft_plus_u_energy_matches_python_reference():
    """The RKS-loop +U contribution must equal the per-spin reference
    formula applied to the converged density, doubled for the
    closed-shell sum over spins."""
    mol, basis = _h2_ccpvdz_mol_basis()
    sites = [vq.HubbardSite(0, 1, U_ev=4.0)]
    result = vq.run_rks(mol, basis, _rks_opts("lda"), dft_plus_u=sites)
    P = np.asarray(result.density)
    S = np.asarray(vq.compute_overlap(basis))
    ao_groups = vq.ao_group_indices(basis)
    occs_sigma = vq.compute_occupation_matrices(sites, 0.5 * P, S, ao_groups)
    E_U_per_spin = vq.compute_dudarev_energy(sites, occs_sigma)
    E_U_total_expected = 2.0 * E_U_per_spin
    assert result.e_dft_plus_u == pytest.approx(
        E_U_total_expected, rel=1e-9, abs=1e-12,
    )


def test_run_rks_dft_plus_u_works_on_hybrid_functional():
    """The hybrid-functional path (HF-exchange + V_xc) is the same
    kernel + same hook — confirm B3LYP convergence + positive E_U."""
    mol, basis = _h2_ccpvdz_mol_basis()
    plus_u = vq.run_rks(
        mol, basis, _rks_opts("b3lyp"),
        dft_plus_u=[vq.HubbardSite(0, 1, U_ev=4.0)],
    )
    assert plus_u.converged
    assert plus_u.e_dft_plus_u > 0.0


# ---------------------------------------------------------------------------
# 8. run_job integration (Increment 2c)
#
# run_job is the path that emits citation files (.bibtex / .references)
# alongside the .out file. Pinning the citation surface here verifies
# the full vertical slice — Dudarev + Cococcioni land in .bibtex
# whenever the user passes ``dft_plus_u=`` to run_job.
# ---------------------------------------------------------------------------


def test_run_job_dft_plus_u_writes_bibtex(tmp_path):
    """run_job(... dft_plus_u=[HubbardSite(...)]) must (a) converge,
    (b) populate result.e_dft_plus_u, and (c) emit a .bibtex file
    containing the Dudarev + Cococcioni keys."""
    mol, basis = _h2_ccpvdz_mol_basis()
    output_stem = tmp_path / "h2_plus_u"
    result = vq.run_job(
        mol,
        basis="cc-pvdz",
        method="rks",
        functional="lda",
        output=str(output_stem),
        dft_plus_u=[vq.HubbardSite(0, 1, U_ev=4.0)],
        # Trim output verbosity for the test.
        write_molden_file=False,
        write_xyz_file=False,
        write_population_file=False,
    )
    assert getattr(result, "converged", False)
    assert result.e_dft_plus_u > 0.0

    bibtex_path = tmp_path / "h2_plus_u.bibtex"
    assert bibtex_path.exists(), (
        "run_job must emit a .bibtex file when citations are on"
    )
    bibtex = bibtex_path.read_text()
    assert "dudarev_dft_plus_u_1998" in bibtex
    assert "cococcioni_gironcoli_2005" in bibtex


def test_run_job_no_dft_plus_u_omits_dudarev_in_bibtex(tmp_path):
    """Symmetric check: omitting dft_plus_u must NOT pull the Dudarev
    + Cococcioni entries into the .bibtex output."""
    mol, basis = _h2_ccpvdz_mol_basis()
    output_stem = tmp_path / "h2_baseline"
    vq.run_job(
        mol,
        basis="cc-pvdz",
        method="rks",
        functional="lda",
        output=str(output_stem),
        write_molden_file=False,
        write_xyz_file=False,
        write_population_file=False,
    )
    bibtex = (tmp_path / "h2_baseline.bibtex").read_text()
    assert "dudarev_dft_plus_u_1998" not in bibtex
    assert "cococcioni_gironcoli_2005" not in bibtex


def test_run_job_dft_plus_u_with_solvent_raises(tmp_path):
    """Solvent + +U is not yet supported; the guard must fire."""
    mol, _ = _h2_ccpvdz_mol_basis()
    with pytest.raises(NotImplementedError, match="solvation"):
        vq.run_job(
            mol,
            basis="cc-pvdz",
            method="rks",
            functional="lda",
            output=str(tmp_path / "h2_solv_plus_u"),
            dft_plus_u=[vq.HubbardSite(0, 1, U_ev=4.0)],
            solvent="water",  # rough placeholder; the guard fires first
            write_molden_file=False,
            write_xyz_file=False,
            write_population_file=False,
        )


# Note: the prior ``test_run_job_dft_plus_u_with_optimize_raises``
# is retired in Increment 3 — geometry optimization with +U now
# runs end-to-end through the ASE / BFGS path. See § 14 below for
# the analytic-vs-FD gradient parity test and the optimize+ +U
# end-to-end smoke test.


# Note: the prior ``test_run_job_dft_plus_u_on_uhf_raises`` (Increment
# 2c) is removed in Increment 2d — UHF + UKS now have real +U support
# and the guard is gone. Open-shell run_job tests live in § 10 below.


# ---------------------------------------------------------------------------
# 9. UHF + UKS SCF integration (Increment 2d)
#
# Open-shell molecular +U. The per-spin convention from Increment 2a
# applies natively here — the SCF Fock builder calls the kernel
# separately on each spin's density and adds the per-spin V_U to that
# spin's Fock. Energies sum directly (no factor of 2 — each call
# returns one spin's contribution).
# ---------------------------------------------------------------------------


def _h_doublet_mol_basis():
    """Isolated H atom in cc-pVDZ, multiplicity=2 (one electron).
    Spherically-symmetric → no polarization → n_α^H_p is identically 0
    and E_U vanishes on the p-channel. Useful for U=0 / parity checks
    where the SCF needs to converge but the magnitude of E_U doesn't
    matter."""
    mol = vq.Molecule([vq.Atom(1, [0.0, 0.0, 0.0])],
                       charge=0, multiplicity=2)
    return mol, vq.BasisSet(mol, "cc-pvdz")


def _h2plus_doublet_mol_basis():
    """H2+ cation, doublet (one electron, charge=+1). The single α
    electron sits in a σ_g bonding orbital ≈ (φ_H1 + φ_H2)/√2, so
    n_α^H1_s ≈ 0.5 (∈ (0, 1)) — the cleanest per-spin partial-AO-
    occupation setup that gives a strictly positive Dudarev energy
    on each H's s-channel without the AO-projector-eigenvalue > 1
    pathology that plagues atoms with multiple occupied shells of
    the same l (e.g. Li 1s² 2s¹ projected on the full s-channel)."""
    mol = vq.Molecule(
        [vq.Atom(1, [0.0, 0.0, 0.0]), vq.Atom(1, [0.0, 0.0, 1.4])],
        charge=1, multiplicity=2,
    )
    return mol, vq.BasisSet(mol, "cc-pvdz")


def _li_doublet_mol_basis():
    """Li atom in cc-pVDZ, multiplicity=2. 1s² 2s¹ on a basis with
    multiple s shells on Li ⇒ ``n_α^Li_s`` is a multi-AO block whose
    eigenvalues can exceed 1 (a known artifact of the AO-projector
    convention when the +U-active channel concentrates much of an
    atom's density). Use this fixture only for the C++/Python
    parity tests where the *sign* of E_U is irrelevant — not for
    "E_U > 0" assertions, since the AO projector doesn't guarantee
    that without Löwdin orthogonalisation (which is deferred to a
    v1.0 sophistication per the Increment-1 docstring)."""
    mol = vq.Molecule([vq.Atom(3, [0.0, 0.0, 0.0])],
                       charge=0, multiplicity=2)
    return mol, vq.BasisSet(mol, "cc-pvdz")


def test_run_uhf_no_dft_plus_u_unchanged():
    """Baseline: omitting dft_plus_u must give the same UHF result as
    the bare C++ run_uhf."""
    mol, basis = _h_doublet_mol_basis()
    r_a = vq.run_uhf(mol, basis)
    r_b = vq.run_uhf(mol, basis, dft_plus_u=[])
    assert r_a.converged and r_b.converged
    assert r_a.energy == pytest.approx(r_b.energy, rel=0.0, abs=1e-14)
    assert r_a.e_dft_plus_u == 0.0
    assert r_b.e_dft_plus_u == 0.0


def test_run_uhf_dft_plus_u_zero_U_is_no_op():
    """U=J ⇒ U_eff=0 ⇒ E_U=0 to machine precision."""
    mol, basis = _h_doublet_mol_basis()
    baseline = vq.run_uhf(mol, basis)
    with_zero = vq.run_uhf(
        mol, basis,
        dft_plus_u=[vq.HubbardSite(0, 1, U_ev=4.0, J_ev=4.0)],
    )
    assert with_zero.converged
    assert with_zero.e_dft_plus_u == pytest.approx(0.0, abs=1e-12)
    assert with_zero.energy == pytest.approx(baseline.energy,
                                              rel=0.0, abs=1e-10)


def test_run_uhf_dft_plus_u_positive_energy_and_total_shift():
    """+U on H1's p-channel of H2+ doublet — the σ_g bonding orbital
    has only polarization-function (small) projection onto H's p
    AOs, so eigenvalues stay well below 1 and Dudarev gives a
    strictly positive per-spin E_U. (The s-channel would project the
    bonding electron itself, where AO-projector eigenvalues can
    exceed 1 in non-orthonormal bases — a known limitation of the
    pure-AO projector documented in Increment 1.)"""
    mol, basis = _h2plus_doublet_mol_basis()
    baseline = vq.run_uhf(mol, basis)
    plus_u = vq.run_uhf(
        mol, basis,
        dft_plus_u=[vq.HubbardSite(0, 1, U_ev=4.0)],
    )
    assert plus_u.converged
    assert plus_u.e_dft_plus_u > 0.0
    assert plus_u.energy > baseline.energy


def test_run_uhf_dft_plus_u_energy_matches_python_reference():
    """The UHF-loop +U contribution must equal the sum of the two
    per-spin reference formulas applied to the converged densities.
    Pins the open-shell ``E_U = E_U_α + E_U_β`` convention end-to-end
    (no factor of 2 — the per-spin formula already gives one term per
    spin, in contrast to the closed-shell case where one call covers
    both spins)."""
    mol, basis = _li_doublet_mol_basis()
    sites = [vq.HubbardSite(0, 0, U_ev=4.0)]  # +U on Li's s-channel
    result = vq.run_uhf(mol, basis, dft_plus_u=sites)
    Pa = np.asarray(result.density_alpha)
    Pb = np.asarray(result.density_beta)
    S = np.asarray(vq.compute_overlap(basis))
    ao_groups = vq.ao_group_indices(basis)
    occs_a = vq.compute_occupation_matrices(sites, Pa, S, ao_groups)
    occs_b = vq.compute_occupation_matrices(sites, Pb, S, ao_groups)
    E_a = vq.compute_dudarev_energy(sites, occs_a)
    E_b = vq.compute_dudarev_energy(sites, occs_b)
    assert result.e_dft_plus_u == pytest.approx(
        E_a + E_b, rel=1e-9, abs=1e-12,
    )


def test_run_uks_no_dft_plus_u_unchanged():
    """Baseline UKS smoke."""
    mol, basis = _h_doublet_mol_basis()
    opts = vq.UKSOptions()
    opts.functional = "lda"
    r_a = vq.run_uks(mol, basis, opts)
    opts_b = vq.UKSOptions()
    opts_b.functional = "lda"
    r_b = vq.run_uks(mol, basis, opts_b, dft_plus_u=[])
    assert r_a.converged and r_b.converged
    assert r_a.energy == pytest.approx(r_b.energy, rel=0.0, abs=1e-14)
    assert r_a.e_dft_plus_u == 0.0
    assert r_b.e_dft_plus_u == 0.0


def test_run_uks_dft_plus_u_positive_energy_and_total_shift():
    """Same setup as the UHF case but on the UKS / LDA path."""
    mol, basis = _h2plus_doublet_mol_basis()
    opts_base = vq.UKSOptions()
    opts_base.functional = "lda"
    baseline = vq.run_uks(mol, basis, opts_base)
    opts_u = vq.UKSOptions()
    opts_u.functional = "lda"
    plus_u = vq.run_uks(
        mol, basis, opts_u,
        dft_plus_u=[vq.HubbardSite(0, 1, U_ev=4.0)],
    )
    assert plus_u.converged
    assert plus_u.e_dft_plus_u > 0.0
    assert plus_u.energy > baseline.energy


def test_run_uks_dft_plus_u_energy_matches_python_reference():
    """UKS open-shell parity vs the Python per-spin reference formula."""
    mol, basis = _li_doublet_mol_basis()
    sites = [vq.HubbardSite(0, 0, U_ev=4.0)]
    opts = vq.UKSOptions()
    opts.functional = "lda"
    result = vq.run_uks(mol, basis, opts, dft_plus_u=sites)
    Pa = np.asarray(result.density_alpha)
    Pb = np.asarray(result.density_beta)
    S = np.asarray(vq.compute_overlap(basis))
    ao_groups = vq.ao_group_indices(basis)
    occs_a = vq.compute_occupation_matrices(sites, Pa, S, ao_groups)
    occs_b = vq.compute_occupation_matrices(sites, Pb, S, ao_groups)
    E_a = vq.compute_dudarev_energy(sites, occs_a)
    E_b = vq.compute_dudarev_energy(sites, occs_b)
    assert result.e_dft_plus_u == pytest.approx(
        E_a + E_b, rel=1e-9, abs=1e-12,
    )


# ---------------------------------------------------------------------------
# 10. run_job for open-shell +U (Increment 2d completion)
#
# The runner-level guard that rejected method ∈ {uhf, uks} when
# dft_plus_u was set is lifted now that the SCF integration is real
# on both paths.
# ---------------------------------------------------------------------------


def test_run_job_uhf_dft_plus_u_writes_bibtex(tmp_path):
    """End-to-end open-shell HF +U through run_job — converges,
    populates result.e_dft_plus_u, and emits Dudarev + Cococcioni
    into the .bibtex output."""
    mol, _ = _h2plus_doublet_mol_basis()
    output_stem = tmp_path / "h2plus_uhf_plus_u"
    result = vq.run_job(
        mol,
        basis="cc-pvdz",
        method="uhf",
        output=str(output_stem),
        dft_plus_u=[vq.HubbardSite(0, 1, U_ev=4.0)],
        write_molden_file=False,
        write_xyz_file=False,
        write_population_file=False,
    )
    assert getattr(result, "converged", False)
    assert result.e_dft_plus_u > 0.0
    bibtex = (tmp_path / "h2plus_uhf_plus_u.bibtex").read_text()
    assert "dudarev_dft_plus_u_1998" in bibtex
    assert "cococcioni_gironcoli_2005" in bibtex


def test_run_job_uks_dft_plus_u_writes_bibtex(tmp_path):
    """End-to-end open-shell LDA +U through run_job."""
    mol, _ = _h2plus_doublet_mol_basis()
    output_stem = tmp_path / "h2plus_uks_plus_u"
    result = vq.run_job(
        mol,
        basis="cc-pvdz",
        method="uks",
        functional="lda",
        output=str(output_stem),
        dft_plus_u=[vq.HubbardSite(0, 1, U_ev=4.0)],
        write_molden_file=False,
        write_xyz_file=False,
        write_population_file=False,
    )
    assert getattr(result, "converged", False)
    assert result.e_dft_plus_u > 0.0
    bibtex = (tmp_path / "h2plus_uks_plus_u.bibtex").read_text()
    assert "dudarev_dft_plus_u_1998" in bibtex
    assert "cococcioni_gironcoli_2005" in bibtex


# ---------------------------------------------------------------------------
# 11. Γ-only periodic RHF SCF integration (Increment 4a)
#
# The Γ-only periodic SCF carries a single real density matrix and a
# single real AO overlap — same shape as the molecular case — so the
# +U kernel is reused unchanged. In the molecular-limit regime
# (atoms in a large box, lattice-summed integrals see only g=0), the
# periodic +U energy must match the molecular RHF +U energy at
# machine precision. That's the cleanest parity check; it pins the
# wiring + math + AO indexing through the periodic basis.
#
# Multi-k periodic +U (the actual user goal — bulk transition-metal-
# oxide bandgaps, Fe-slab adsorption) lands in Increment 4c.
# ---------------------------------------------------------------------------


def _h2o_periodic_in_box(box: float = 50.0, *, center: bool = False):
    """H2O in a box that defaults to the 50³-bohr molecular-limit fixture."""
    c = box / 2.0 if center else 0.0
    h2o = [
        vq.Atom(8, [c, c, c]),
        vq.Atom(1, [c, c + 1.43, c - 0.98]),
        vq.Atom(1, [c, c - 1.43, c - 0.98]),
    ]
    sysp = vq.PeriodicSystem(3, np.eye(3) * box, h2o)
    basis = vq.BasisSet(sysp.unit_cell_molecule(), "sto-3g")
    return sysp, basis, h2o


def _tight_periodic_rhf_opts():
    opts = vq.PeriodicRHFOptions()
    opts.conv_tol_energy = 1e-12
    opts.conv_tol_grad = 1e-10
    return opts


def test_run_rhf_periodic_gamma_no_dft_plus_u_unchanged():
    sysp, basis, _ = _h2o_periodic_in_box()
    r_a = vq.run_rhf_periodic_gamma(sysp, basis, _tight_periodic_rhf_opts())
    r_b = vq.run_rhf_periodic_gamma(
        sysp, basis, _tight_periodic_rhf_opts(), dft_plus_u=[],
    )
    assert r_a.converged and r_b.converged
    assert r_a.energy == pytest.approx(r_b.energy, rel=0.0, abs=1e-14)
    assert r_a.e_dft_plus_u == 0.0
    assert r_b.e_dft_plus_u == 0.0


def test_run_rhf_periodic_gamma_dft_plus_u_zero_U_is_no_op():
    sysp, basis, _ = _h2o_periodic_in_box()
    baseline = vq.run_rhf_periodic_gamma(sysp, basis, _tight_periodic_rhf_opts())
    with_zero = vq.run_rhf_periodic_gamma(
        sysp, basis, _tight_periodic_rhf_opts(),
        dft_plus_u=[vq.HubbardSite(0, 1, U_ev=4.0, J_ev=4.0)],
    )
    assert with_zero.converged
    assert with_zero.e_dft_plus_u == pytest.approx(0.0, abs=1e-12)
    assert with_zero.energy == pytest.approx(baseline.energy,
                                              rel=0.0, abs=1e-10)


def test_run_rhf_periodic_gamma_dft_plus_u_matches_molecular_rhf():
    """The Γ-only periodic +U in the molecular-limit regime must agree
    with the molecular RHF +U at machine precision — exact parity is
    the strongest possible test that the periodic wiring is right.
    Tests Increment 4a end-to-end: PeriodicRHFOptions.dft_plus_u_*
    fields, the new branch in cpp/src/periodic_rhf.cpp, the Python
    wrapper, and the AO-indexing reuse."""
    sysp, basis, h2o_atoms = _h2o_periodic_in_box()
    sites = [vq.HubbardSite(0, 1, U_ev=4.0)]  # O's p-channel
    p_result = vq.run_rhf_periodic_gamma(
        sysp, basis, _tight_periodic_rhf_opts(), dft_plus_u=sites,
    )
    # Molecular reference at the same geometry + basis.
    mol = vq.Molecule(h2o_atoms, 0, 1)
    mbasis = vq.BasisSet(mol, "sto-3g")
    mopts = vq.RHFOptions()
    mopts.conv_tol_energy = 1e-12
    mopts.conv_tol_grad = 1e-10
    m_result = vq.run_rhf(mol, mbasis, mopts, dft_plus_u=sites)
    assert p_result.converged and m_result.converged
    # Compare the electronic (non-nuclear) energy and the +U
    # contribution. Nuclear repulsion uses a slightly different
    # lattice-sum convention in the periodic driver, so we strip it
    # to get a clean parity number.
    p_elec = p_result.e_electronic
    m_elec = m_result.e_electronic
    assert p_elec == pytest.approx(m_elec, rel=0.0, abs=5e-10)
    assert p_result.e_dft_plus_u == pytest.approx(
        m_result.e_dft_plus_u, rel=0.0, abs=5e-10,
    )


def test_run_rhf_periodic_gamma_dft_plus_u_positive_shift():
    """+U on H2O/sto-3g's O p-channel gives a strictly positive
    Dudarev contribution and shifts the total cell energy upward."""
    sysp, basis, _ = _h2o_periodic_in_box()
    baseline = vq.run_rhf_periodic_gamma(sysp, basis, _tight_periodic_rhf_opts())
    plus_u = vq.run_rhf_periodic_gamma(
        sysp, basis, _tight_periodic_rhf_opts(),
        dft_plus_u=[vq.HubbardSite(0, 1, U_ev=4.0)],
    )
    assert plus_u.converged
    assert plus_u.e_dft_plus_u > 0.0
    assert plus_u.energy > baseline.energy


# ---------------------------------------------------------------------------
# 12. Γ-only periodic RKS SCF integration (Increment 4b)
#
# The closed-shell-DFT periodic SCF lives in cpp/src/periodic_scf.cpp
# (multi-k driver in general, used here at Γ-only). Same Γ-folded
# real-density convention as periodic_rhf.cpp; multi-k +U lives in
# Increment 4c.
# ---------------------------------------------------------------------------


def _tight_periodic_ks_opts(functional: str = "lda"):
    opts = vq.PeriodicKSOptions()
    opts.functional = functional
    opts.conv_tol_energy = 1e-12
    opts.conv_tol_grad = 1e-10
    return opts


def _gamma_kmesh(sysp):
    return vq.KPoints.monkhorst_pack(sysp, (1, 1, 1))


def test_run_rks_periodic_gamma_dft_plus_u_zero_U_is_no_op():
    sysp, basis, _ = _h2o_periodic_in_box()
    kmesh = _gamma_kmesh(sysp)
    baseline = vq.run_rks_periodic(sysp, basis, kmesh, _tight_periodic_ks_opts())
    with_zero = vq.run_rks_periodic(
        sysp, basis, kmesh, _tight_periodic_ks_opts(),
        dft_plus_u=[vq.HubbardSite(0, 1, U_ev=4.0, J_ev=4.0)],
    )
    assert with_zero.converged
    assert with_zero.e_dft_plus_u == pytest.approx(0.0, abs=1e-12)
    assert with_zero.energy == pytest.approx(baseline.energy, rel=0.0, abs=1e-10)


def test_run_rks_periodic_gamma_dft_plus_u_matches_molecular_rks():
    """Strongest test: Γ-only periodic RKS +U on H2O/STO-3G in a 50³
    bohr box must match molecular RKS +U on the same molecule with
    the same functional + +U setup. Pins Increment 4b end-to-end."""
    sysp, basis, h2o_atoms = _h2o_periodic_in_box()
    kmesh = _gamma_kmesh(sysp)
    sites = [vq.HubbardSite(0, 1, U_ev=4.0)]
    p_result = vq.run_rks_periodic(
        sysp, basis, kmesh, _tight_periodic_ks_opts("lda"), dft_plus_u=sites,
    )
    mol = vq.Molecule(h2o_atoms, 0, 1)
    mbasis = vq.BasisSet(mol, "sto-3g")
    mopts = vq.RKSOptions()
    mopts.functional = "lda"
    mopts.conv_tol_energy = 1e-12
    mopts.conv_tol_grad = 1e-10
    m_result = vq.run_rks(mol, mbasis, mopts, dft_plus_u=sites)
    assert p_result.converged and m_result.converged
    # Both code paths compute e_electronic from the +U-free Fock
    # (E_core + E_J + E_K + E_xc) and report e_dft_plus_u separately,
    # so the parity holds on both fields independently.
    p_elec = p_result.e_electronic
    m_elec = m_result.e_electronic
    assert p_elec == pytest.approx(m_elec, rel=0.0, abs=5e-10)
    assert p_result.e_dft_plus_u == pytest.approx(
        m_result.e_dft_plus_u, rel=0.0, abs=5e-10,
    )


def test_run_rks_periodic_gamma_dft_plus_u_positive_shift():
    sysp, basis, _ = _h2o_periodic_in_box()
    kmesh = _gamma_kmesh(sysp)
    baseline = vq.run_rks_periodic(sysp, basis, kmesh, _tight_periodic_ks_opts())
    plus_u = vq.run_rks_periodic(
        sysp, basis, kmesh, _tight_periodic_ks_opts(),
        dft_plus_u=[vq.HubbardSite(0, 1, U_ev=4.0)],
    )
    assert plus_u.converged
    assert plus_u.e_dft_plus_u > 0.0
    assert plus_u.energy > baseline.energy


def test_run_rks_periodic_multi_k_dft_plus_u_runs():
    """Multi-k periodic +U landed in Increment 4c — the wrapper no
    longer raises. On the H2O/STO-3G molecular-limit fixture (50³
    bohr box) a multi-k mesh should give essentially the same
    energy as Γ-only, since the unit cell is large enough that
    cross-cell density blocks are zero."""
    sysp, basis, _ = _h2o_periodic_in_box()
    kmesh_g = vq.KPoints.monkhorst_pack(sysp, (1, 1, 1))
    kmesh_multi = vq.KPoints.monkhorst_pack(sysp, (2, 1, 1))
    sites = [vq.HubbardSite(0, 1, U_ev=4.0)]
    r_g = vq.run_rks_periodic(
        sysp, basis, kmesh_g, _tight_periodic_ks_opts(),
        dft_plus_u=sites,
    )
    r_mk = vq.run_rks_periodic(
        sysp, basis, kmesh_multi, _tight_periodic_ks_opts(),
        dft_plus_u=sites,
    )
    assert r_g.converged and r_mk.converged
    # Molecular-limit cell: multi-k folds to Γ-only at floor-level
    # tolerance (the lattice cutoff sees only g=0 either way).
    assert r_g.energy == pytest.approx(r_mk.energy, rel=0.0, abs=1e-9)
    assert r_g.e_dft_plus_u == pytest.approx(
        r_mk.e_dft_plus_u, rel=0.0, abs=1e-9,
    )


# ---------------------------------------------------------------------------
# 13. run_periodic_job DFT+U plumbing (Increment 4d-light)
#
# run_periodic_job(... dft_plus_u=[HubbardSite(...)]) surfaces Dudarev +
# Cococcioni in the .bibtex / .references output on every +U route. Γ-only
# AUTO Gamma RHF/RKS/UHF/UKS plan the exact BIPOLE route before preflight;
# this avoids the legacy post-selection swap to a truncated DIRECT Coulomb
# Hamiltonian. Closed-shell true multi-k runs natively on the GDF drivers
# (landed 2026-07-04, a40feea0).
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    ("module_name", "driver_name", "commutator_name", "fock_attributes"),
    [
        (
            "pbc_bipole.py",
            "run_pbc_bipole_rhf",
            "restricted_bipole_commutator_norm",
            {"f_k_list"},
        ),
        (
            "pbc_bipole_rks.py",
            "run_pbc_bipole_rks",
            "restricted_bipole_commutator_norm",
            {"f_k_list"},
        ),
        (
            "pbc_bipole_uhf.py",
            "run_pbc_bipole_uhf",
            "unrestricted_bipole_commutator_norm",
            {"f_alpha_k_list", "f_beta_k_list"},
        ),
        (
            "pbc_bipole_uks.py",
            "run_pbc_bipole_uks",
            "unrestricted_bipole_commutator_norm",
            {"f_alpha_k_list", "f_beta_k_list"},
        ),
    ],
)
def test_bipole_exact_confirmation_reuses_one_dft_plus_u_operator(
    module_name,
    driver_name,
    commutator_name,
    fock_attributes,
):
    """The provisional and terminal IID-514 gates share one +U Fock.

    ``_apply_dft_plus_u`` mutates its Fock-list arguments.  Therefore the
    exact-density confirmation must augment ``_confirm_fb`` before measuring
    its commutator, and finalisation must reuse the associated +U energy
    instead of applying the potential to that same Fock a second time.
    """
    module_path = Path(vq.__file__).resolve().parent / module_name
    tree = ast.parse(module_path.read_text(encoding="utf-8"))
    driver = next(
        node
        for node in tree.body
        if isinstance(node, ast.FunctionDef) and node.name == driver_name
    )

    confirm_build = next(
        node
        for node in ast.walk(driver)
        if isinstance(node, ast.Assign)
        and any(
            isinstance(target, ast.Name) and target.id == "_confirm_fb"
            for target in node.targets
        )
        and isinstance(node.value, ast.Call)
        and isinstance(node.value.func, ast.Name)
        and node.value.func.id == "_build_fock_for_density"
    )
    confirm_commutator = min(
        (
            node
            for node in ast.walk(driver)
            if isinstance(node, ast.Call)
            and isinstance(node.func, ast.Name)
            and node.func.id == commutator_name
            and node.lineno > confirm_build.lineno
        ),
        key=lambda node: node.lineno,
    )
    provisional_plus_u = [
        node
        for node in ast.walk(driver)
        if isinstance(node, ast.Call)
        and isinstance(node.func, ast.Name)
        and node.func.id == "_apply_dft_plus_u"
        and confirm_build.lineno < node.lineno < confirm_commutator.lineno
    ]
    assert len(provisional_plus_u) == 1, (
        f"{module_name} must add +U to the exact confirmation Fock before "
        "measuring its commutator"
    )
    confirmation_attributes = {
        node.attr
        for node in ast.walk(provisional_plus_u[0])
        if isinstance(node, ast.Attribute)
        and isinstance(node.value, ast.Name)
        and node.value.id == "_confirm_fb"
    }
    assert confirmation_attributes == fock_attributes

    terminal_reuse = [
        node
        for node in ast.walk(driver)
        if isinstance(node, ast.If)
        and node.lineno > confirm_commutator.lineno
        and "_confirm_fb is not None" in ast.unparse(node.test)
        and any(
            isinstance(child, ast.Assign)
            and any(
                isinstance(target, ast.Name) and target.id == "e_dft_plus_u"
                for target in child.targets
            )
            and isinstance(child.value, ast.Name)
            and child.value.id == "_confirm_e_dft_plus_u"
            for statement in node.body
            for child in ast.walk(statement)
        )
        and any(
            isinstance(child, ast.Call)
            and isinstance(child.func, ast.Name)
            and child.func.id == "_apply_dft_plus_u"
            for statement in node.orelse
            for child in ast.walk(statement)
        )
    ]
    assert len(terminal_reuse) == 1, (
        f"{module_name} must reuse the confirmed +U energy and only apply "
        "+U in the fresh-rebuild branch"
    )


def test_run_periodic_job_rhf_dft_plus_u_writes_bibtex(tmp_path):
    """Γ-only periodic RHF +U via run_periodic_job — converges,
    populates result.e_dft_plus_u, and emits Dudarev + Cococcioni
    into .bibtex."""
    sysp, basis, _ = _h2o_periodic_in_box(box=12.0, center=True)
    output_stem = tmp_path / "h2o_periodic_rhf_plus_u"
    result = vq.run_periodic_job(
        sysp, basis,
        method="RHF",
        output=str(output_stem),
        dft_plus_u=[vq.HubbardSite(0, 1, U_ev=4.0)],
        write_molden_file=False, write_xyz_file=False,
        write_poscar_file=False, write_xsf_structure_file=False,
        write_cif_file=False, write_population_file=False,
    )
    assert getattr(result, "converged", False)
    assert result.e_dft_plus_u > 0.0
    bibtex = (tmp_path / "h2o_periodic_rhf_plus_u.bibtex").read_text()
    assert "dudarev_dft_plus_u_1998" in bibtex
    assert "cococcioni_gironcoli_2005" in bibtex


def test_run_periodic_job_rks_dft_plus_u_writes_bibtex(tmp_path):
    """Γ-only periodic RKS +U via run_periodic_job."""
    sysp, basis, _ = _h2o_periodic_in_box(box=12.0, center=True)
    output_stem = tmp_path / "h2o_periodic_rks_plus_u"
    result = vq.run_periodic_job(
        sysp, basis,
        method="RKS", functional="lda",
        output=str(output_stem),
        dft_plus_u=[vq.HubbardSite(0, 1, U_ev=4.0)],
        write_molden_file=False, write_xyz_file=False,
        write_poscar_file=False, write_xsf_structure_file=False,
        write_cif_file=False, write_population_file=False,
    )
    assert getattr(result, "converged", False)
    assert result.e_dft_plus_u > 0.0
    bibtex = (tmp_path / "h2o_periodic_rks_plus_u.bibtex").read_text()
    assert "dudarev_dft_plus_u_1998" in bibtex
    assert "cococcioni_gironcoli_2005" in bibtex


def test_run_periodic_job_uhf_dft_plus_u_runs(tmp_path):
    """Open-shell periodic +U on the BIPOLE UHF driver (Increment
    4d-bipole). H2+ doublet in a compact 12³ box; the SCF converges with
    +U on H's p-channel and emits Dudarev + Cococcioni in the .bibtex."""
    c = 6.0
    sysp = vq.PeriodicSystem(
        3, np.eye(3) * 12.0,
        [vq.Atom(1, [c, c, c - 0.7]),
         vq.Atom(1, [c, c, c + 0.7])],
        charge=1, multiplicity=2,
    )
    basis = vq.BasisSet(sysp.unit_cell_molecule(), "cc-pvdz")
    output_stem = tmp_path / "h2plus_uhf_bipole_plus_u"
    result = vq.run_periodic_job(
        sysp, basis,
        method="UHF", jk_method="bipole",
        output=str(output_stem),
        dft_plus_u=[vq.HubbardSite(0, 1, U_ev=4.0)],
        write_molden_file=False, write_xyz_file=False,
        write_poscar_file=False, write_xsf_structure_file=False,
        write_cif_file=False, write_population_file=False,
    )
    assert getattr(result, "converged", False)
    assert result.e_dft_plus_u > 0.0
    bibtex = (tmp_path / "h2plus_uhf_bipole_plus_u.bibtex").read_text()
    assert "dudarev_dft_plus_u_1998" in bibtex
    assert "cococcioni_gironcoli_2005" in bibtex


def test_run_periodic_job_uks_dft_plus_u_runs(tmp_path):
    """Open-shell periodic UKS +U via the BIPOLE driver (Increment
    4d-bipole, UKS commit). H2+ doublet in a compact 12³ bohr box; LDA +
    U on H's p-channel converges and emits Dudarev + Cococcioni
    in the .bibtex."""
    c = 6.0
    sysp = vq.PeriodicSystem(
        3, np.eye(3) * 12.0,
        [vq.Atom(1, [c, c, c - 0.7]),
         vq.Atom(1, [c, c, c + 0.7])],
        charge=1, multiplicity=2,
    )
    basis = vq.BasisSet(sysp.unit_cell_molecule(), "cc-pvdz")
    output_stem = tmp_path / "h2plus_uks_bipole_plus_u"
    result = vq.run_periodic_job(
        sysp, basis,
        method="UKS", functional="lda", jk_method="bipole",
        output=str(output_stem),
        dft_plus_u=[vq.HubbardSite(0, 1, U_ev=4.0)],
        write_molden_file=False, write_xyz_file=False,
        write_poscar_file=False, write_xsf_structure_file=False,
        write_cif_file=False, write_population_file=False,
    )
    assert getattr(result, "converged", False)
    assert result.e_dft_plus_u > 0.0
    bibtex = (tmp_path / "h2plus_uks_bipole_plus_u.bibtex").read_text()
    assert "dudarev_dft_plus_u_1998" in bibtex
    assert "cococcioni_gironcoli_2005" in bibtex


def test_run_periodic_job_multik_dft_plus_u_runs_native_gdf(tmp_path):
    """Multi-k closed-shell +U via run_periodic_job runs natively on
    the GDF drivers — auto jk_method resolves RHF to GDF, and with a
    true multi-k mesh the +U request is NOT intercepted by the legacy
    Γ-only guard (which raised NotImplementedError before the multi-k
    GDF +U landing, a40feea0). H2/STO-3G in a 16³ bohr box with U on
    H's s-channel: the SCF converges, reports a strictly positive
    Dudarev energy, records the gdf_rhf_multi_k route, and emits
    Dudarev + Cococcioni in the .bibtex."""
    L = 16.0
    sysp = vq.PeriodicSystem(
        3, np.eye(3) * L,
        [vq.Atom(1, [L / 2 - 0.7, L / 2, L / 2]),
         vq.Atom(1, [L / 2 + 0.7, L / 2, L / 2])],
    )
    basis = vq.BasisSet(sysp.unit_cell_molecule(), "sto-3g")
    output_stem = tmp_path / "h2_multik_plus_u"
    result = vq.run_periodic_job(
        sysp, basis,
        method="RHF",
        kpoints=(2, 1, 1),
        output=str(output_stem),
        dft_plus_u=[vq.HubbardSite(0, 0, U_ev=2.7)],
        write_molden_file=False, write_xyz_file=False,
        write_poscar_file=False, write_xsf_structure_file=False,
        write_cif_file=False, write_population_file=False,
    )
    assert getattr(result, "converged", False)
    assert result.e_dft_plus_u > 0.0
    out_text = output_stem.with_suffix(".out").read_text()
    assert "dft_plus_u_route    = gdf_rhf_multi_k" in out_text
    bibtex = output_stem.with_suffix(".bibtex").read_text()
    assert "dudarev_dft_plus_u_1998" in bibtex
    assert "cococcioni_gironcoli_2005" in bibtex


# ---------------------------------------------------------------------------
# 14. Analytic +U gradient (Increment 3)
#
# The C++ kernel returns the variational Fock contribution
# V_U_for_Fock = S V_AO S, so the SCF converges to a true stationary
# point of E_total = E_HF + E_U with n = (SPS)_(A,l). The orbital-
# response Pulay piece of dE_U/dR is then captured automatically by
# the standard energy-weighted-density gradient term in
# compute_gradient(result); the explicit dS-piece is added by the
# Python wrapper via the dft_plus_u= kwarg.
#
# Verified against full FD of dE_total/dR at sub-1e-10 Ha/bohr.
# ---------------------------------------------------------------------------


def test_compute_gradient_rhf_dft_plus_u_matches_fd():
    """Analytic dE_total/dR from compute_gradient(..., dft_plus_u=)
    must match finite-difference dE_total/dR on H2O/STO-3G with
    U=4 eV on O's p-channel."""
    ANGSTROM_TO_BOHR = 1.8897261339213
    def make_mol(disp=0.0):
        return vq.Molecule([
            vq.Atom(8, [0.0, 0.0, 0.0]),
            vq.Atom(1, [0.0,  0.7572 * ANGSTROM_TO_BOHR,
                       0.5868 * ANGSTROM_TO_BOHR + disp]),
            vq.Atom(1, [0.0, -0.7572 * ANGSTROM_TO_BOHR,
                       0.5868 * ANGSTROM_TO_BOHR]),
        ], charge=0, multiplicity=1)

    def fresh_opts():
        o = vq.RHFOptions()
        o.conv_tol_energy = 1e-13
        o.conv_tol_grad = 1e-11
        o.max_iter = 200
        return o

    sites = [vq.HubbardSite(0, 1, U_ev=4.0)]
    mol0 = make_mol(0.0)
    basis0 = vq.BasisSet(mol0, "sto-3g")
    result = vq.run_rhf(mol0, basis0, fresh_opts(), dft_plus_u=sites)
    g_analytic = np.asarray(vq.compute_gradient(
        mol0, basis0, result, dft_plus_u=sites,
    ))

    # FD of E_total along H1's z direction.
    h = 1e-4
    mol_p = make_mol(+h)
    basis_p = vq.BasisSet(mol_p, "sto-3g")
    r_p = vq.run_rhf(mol_p, basis_p, fresh_opts(), dft_plus_u=sites)
    mol_m = make_mol(-h)
    basis_m = vq.BasisSet(mol_m, "sto-3g")
    r_m = vq.run_rhf(mol_m, basis_m, fresh_opts(), dft_plus_u=sites)
    g_fd = (r_p.energy - r_m.energy) / (2 * h)
    assert g_analytic[1, 2] == pytest.approx(g_fd, rel=0.0, abs=1e-7)


def test_compute_gradient_no_dft_plus_u_unchanged():
    """Without dft_plus_u kwarg, compute_gradient returns the bare
    HF gradient — back-compat preserved."""
    ANGSTROM_TO_BOHR = 1.8897261339213
    mol = vq.Molecule([
        vq.Atom(8, [0.0, 0.0, 0.0]),
        vq.Atom(1, [0.0,  0.7572 * ANGSTROM_TO_BOHR,
                   0.5868 * ANGSTROM_TO_BOHR]),
        vq.Atom(1, [0.0, -0.7572 * ANGSTROM_TO_BOHR,
                   0.5868 * ANGSTROM_TO_BOHR]),
    ], charge=0, multiplicity=1)
    basis = vq.BasisSet(mol, "sto-3g")
    opts = vq.RHFOptions()
    opts.conv_tol_energy = 1e-12
    opts.conv_tol_grad = 1e-10
    result = vq.run_rhf(mol, basis, opts)
    g_default = np.array(vq.compute_gradient(mol, basis, result), copy=True)
    g_no_u = np.array(
        vq.compute_gradient(mol, basis, result, dft_plus_u=None), copy=True,
    )
    # Tight but not bit-exact — pybind11/Eigen marshalling can shuffle
    # the underlying memory but the math is the same; back-compat means
    # numerical agreement at the level of the converged SCF.
    np.testing.assert_allclose(g_default, g_no_u, rtol=0.0, atol=1e-13)


def test_compute_gradient_rks_dft_plus_u_matches_fd():
    """RKS analog of the RHF +U gradient FD-parity test."""
    ANGSTROM_TO_BOHR = 1.8897261339213
    def make_mol(disp=0.0):
        return vq.Molecule([
            vq.Atom(8, [0.0, 0.0, 0.0]),
            vq.Atom(1, [0.0,  0.7572 * ANGSTROM_TO_BOHR,
                       0.5868 * ANGSTROM_TO_BOHR + disp]),
            vq.Atom(1, [0.0, -0.7572 * ANGSTROM_TO_BOHR,
                       0.5868 * ANGSTROM_TO_BOHR]),
        ], charge=0, multiplicity=1)

    def fresh_opts():
        o = vq.RKSOptions()
        o.functional = "lda"
        o.conv_tol_energy = 1e-13
        o.conv_tol_grad = 1e-11
        o.max_iter = 200
        return o

    sites = [vq.HubbardSite(0, 1, U_ev=4.0)]
    mol0 = make_mol(0.0)
    basis0 = vq.BasisSet(mol0, "sto-3g")
    result = vq.run_rks(mol0, basis0, fresh_opts(), dft_plus_u=sites)
    g_analytic = np.asarray(vq.compute_gradient_rks(
        mol0, basis0, result, dft_plus_u=sites,
    ))

    h = 1e-4
    mol_p = make_mol(+h)
    basis_p = vq.BasisSet(mol_p, "sto-3g")
    r_p = vq.run_rks(mol_p, basis_p, fresh_opts(), dft_plus_u=sites)
    mol_m = make_mol(-h)
    basis_m = vq.BasisSet(mol_m, "sto-3g")
    r_m = vq.run_rks(mol_m, basis_m, fresh_opts(), dft_plus_u=sites)
    g_fd = (r_p.energy - r_m.energy) / (2 * h)
    # RKS gradient has XC-grid integration noise; loosen tolerance.
    assert g_analytic[1, 2] == pytest.approx(g_fd, rel=0.0, abs=1e-5)


def test_run_job_capped_optimize_with_dft_plus_u_fails_closed(tmp_path):
    """The historical eight-step +U smoke run is not geometry-converged."""
    ANGSTROM_TO_BOHR = 1.8897261339213
    mol = vq.Molecule([
        vq.Atom(8, [0.0, 0.0, 0.0]),
        vq.Atom(1, [0.0,  0.7572 * ANGSTROM_TO_BOHR,
                   0.5868 * ANGSTROM_TO_BOHR]),
        vq.Atom(1, [0.0, -0.7572 * ANGSTROM_TO_BOHR,
                   0.5868 * ANGSTROM_TO_BOHR]),
    ], charge=0, multiplicity=1)
    output_stem = tmp_path / "h2o_opt_plus_u"
    with pytest.raises(
        RuntimeError,
        match="did not converge after 8 of 8 allowed steps",
    ):
        vq.run_job(
            mol, basis="sto-3g", method="rhf",
            output=str(output_stem),
            optimize=True,
            fmax=0.1,
            max_opt_steps=8,
            dft_plus_u=[vq.HubbardSite(0, 1, U_ev=4.0)],
            write_molden_file=False,
            write_xyz_file=False,
            write_population_file=False,
        )

    assert (tmp_path / "h2o_opt_plus_u.traj").exists()
    assert 'status           = "crashed"' in output_stem.with_suffix(
        ".system"
    ).read_text()


# ---------------------------------------------------------------------------
# §15 — Γ-only periodic +U gradient (Pulay overlap-derivative contribution)
# ---------------------------------------------------------------------------

def test_compute_gradient_periodic_rhf_gamma_dft_plus_u_matches_molecular():
    """Periodic Γ-only RHF +U gradient in the molecular-limit (20 Å box)
    must match the molecular RHF +U gradient on H2O / STO-3G with U=4 eV
    on O's p-channel."""
    ANGSTROM_TO_BOHR = 1.8897261339213
    big_box = 20.0 * ANGSTROM_TO_BOHR
    atoms = [
        vq.Atom(8, [0.0, 0.0, 0.0]),
        vq.Atom(1, [0.0,  0.7572 * ANGSTROM_TO_BOHR,
                   0.5868 * ANGSTROM_TO_BOHR]),
        vq.Atom(1, [0.0, -0.7572 * ANGSTROM_TO_BOHR,
                   0.5868 * ANGSTROM_TO_BOHR]),
    ]
    sites = [vq.HubbardSite(0, 1, U_ev=4.0)]

    # Periodic Γ-only with +U via the dedicated Γ-only driver.
    sysp = vq.PeriodicSystem(3, np.diag([big_box, big_box, big_box]), atoms)
    basisp = vq.BasisSet(sysp.unit_cell_molecule(), "sto-3g")
    optsp = vq.PeriodicRHFOptions()
    optsp.conv_tol_energy = 1e-12
    optsp.lattice_opts.cutoff_bohr = 25.0
    optsp.lattice_opts.nuclear_cutoff_bohr = 25.0
    rp = vq.run_rhf_periodic_gamma(sysp, basisp, optsp, dft_plus_u=sites)
    g_p = vq.compute_gradient_periodic_rhf_gamma(
        sysp, basisp, rp, lattice_opts=optsp.lattice_opts,
        dft_plus_u=sites,
    )

    # Molecular reference with +U.
    mol = vq.Molecule(atoms, charge=0, multiplicity=1)
    basism = vq.BasisSet(mol, "sto-3g")
    rhf_opts = vq.RHFOptions()
    rhf_opts.conv_tol_energy = 1e-13
    rhf_opts.conv_tol_grad = 1e-11
    rhf_opts.max_iter = 200
    rm = vq.run_rhf(mol, basism, rhf_opts, dft_plus_u=sites)
    g_m = np.asarray(vq.compute_gradient(mol, basism, rm, dft_plus_u=sites))

    # Molecular-limit parity: large box + matching cutoffs → analytic
    # equality at the few-µHa/bohr level. (Slight slack vs vanilla
    # molecular-limit tests because the periodic +U Pulay tile differs
    # from the molecular libint call by O(periodic-image) noise.)
    np.testing.assert_allclose(g_p, g_m, atol=5e-6)


def test_compute_gradient_periodic_rks_gamma_dft_plus_u_matches_molecular():
    """RKS analog of the periodic Γ-only +U gradient molecular-limit
    parity test. LDA, H2O / STO-3G, U=4 eV on O-p."""
    ANGSTROM_TO_BOHR = 1.8897261339213
    big_box = 20.0 * ANGSTROM_TO_BOHR
    atoms = [
        vq.Atom(8, [0.0, 0.0, 0.0]),
        vq.Atom(1, [0.0,  0.7572 * ANGSTROM_TO_BOHR,
                   0.5868 * ANGSTROM_TO_BOHR]),
        vq.Atom(1, [0.0, -0.7572 * ANGSTROM_TO_BOHR,
                   0.5868 * ANGSTROM_TO_BOHR]),
    ]
    sites = [vq.HubbardSite(0, 1, U_ev=4.0)]

    sysp = vq.PeriodicSystem(3, np.diag([big_box, big_box, big_box]), atoms)
    basisp = vq.BasisSet(sysp.unit_cell_molecule(), "sto-3g")
    ks_opts = vq.PeriodicKSOptions()
    ks_opts.functional = "lda"
    ks_opts.conv_tol_energy = 1e-12
    ks_opts.lattice_opts.cutoff_bohr = 25.0
    ks_opts.lattice_opts.nuclear_cutoff_bohr = 25.0
    kmesh = vq.monkhorst_pack(sysp, [1, 1, 1])
    rp = vq.run_rks_periodic(sysp, basisp, kmesh, ks_opts, dft_plus_u=sites)
    g_p = vq.compute_gradient_periodic_rks_gamma(
        sysp, basisp, rp, lattice_opts=ks_opts.lattice_opts,
        dft_plus_u=sites,
    )

    mol = vq.Molecule(atoms, charge=0, multiplicity=1)
    basism = vq.BasisSet(mol, "sto-3g")
    rks_opts = vq.RKSOptions()
    rks_opts.functional = "lda"
    rks_opts.conv_tol_energy = 1e-13
    rks_opts.conv_tol_grad = 1e-11
    rks_opts.max_iter = 200
    rm = vq.run_rks(mol, basism, rks_opts, dft_plus_u=sites)
    g_m = np.asarray(vq.compute_gradient_rks(mol, basism, rm, dft_plus_u=sites))

    np.testing.assert_allclose(g_p, g_m, atol=1e-4)


def test_run_periodic_job_optimize_dft_plus_u_rhf_bipole_runs(tmp_path):
    """``run_periodic_job(optimize=True, method="RHF", jk_method="bipole",
    dft_plus_u=[...])`` now runs end-to-end — the closed-shell BIPOLE
    drivers grew ``dft_plus_u=`` in the same commit that lifted the
    guard. Pre-fix this raised ``NotImplementedError``."""
    c = 6.0
    sysp = vq.PeriodicSystem(
        3, np.diag([12.0] * 3),
        [vq.Atom(1, [c, c, c - 0.7]),
         vq.Atom(1, [c, c, c + 0.7])],
    )
    basis = vq.BasisSet(sysp.unit_cell_molecule(), "cc-pvdz")
    result = vq.run_periodic_job(
        sysp, basis,
        method="RHF", jk_method="bipole", kpoints=(1, 1, 1),
        optimize=True, optimize_max_iter=2,
        output=str(tmp_path / "h2_opt_rhf_plus_u"),
        dft_plus_u=[vq.HubbardSite(0, 1, U_ev=4.0)],
        write_molden_file=False, write_xyz_file=False,
        write_poscar_file=False, write_xsf_structure_file=False,
        write_cif_file=False, write_population_file=False,
    )
    # The single-point SCF inside the first optimize step must converge
    # (energy is finite) and Dudarev + Cococcioni must land in .bibtex.
    assert np.isfinite(result.energy)
    bibtex = (tmp_path / "h2_opt_rhf_plus_u.bibtex").read_text()
    assert "dudarev_dft_plus_u_1998" in bibtex
    assert "cococcioni_gironcoli_2005" in bibtex


# ---------------------------------------------------------------------------
# §16 — Multi-k BIPOLE periodic +U gradient (Pulay overlap-derivative)
# ---------------------------------------------------------------------------

def test_compute_bipole_gradient_uhf_dft_plus_u_contribution_matches_molecular():
    """**+U contribution** to the multi-k UHF BIPOLE gradient — i.e.
    ``g_with_U − g_without_U`` — must match the molecular UHF +U
    contribution in molecular-limit to within the BIPOLE gradient's
    intrinsic Γ-W tile noise (~1 mHa/bohr per existing
    ``test_bipole_gradient`` tolerance bound).

    The BIPOLE gradient (without +U) is already only ~0.01-0.1 Ha/bohr
    accurate vs FD because it tiles ``W_gamma`` at every cell rather
    than proper Bloch-folding W(k). My +U Pulay piece uses the proper
    Bloch fold ``M(g) = Σ_k w_k Re[exp(-i k·g) V_AO S(k) P(k)]`` so it
    introduces no additional approximation; both the +U-shifted ε in
    the W-tile and the explicit +U Pulay propagate consistently.

    System: H₂⁺ doublet / cc-pVDZ in a 20 Å box with U=4 eV on H-p.
    """
    ANGSTROM_TO_BOHR = 1.8897261339213
    big_box = 20.0 * ANGSTROM_TO_BOHR
    atoms = [
        vq.Atom(1, [0.0, 0.0, 0.0]),
        vq.Atom(1, [0.0, 0.0, 1.4]),
    ]
    sites = [vq.HubbardSite(0, 1, U_ev=4.0)]

    sysp = vq.PeriodicSystem(
        3, np.diag([big_box, big_box, big_box]), atoms,
        charge=1, multiplicity=2,
    )
    basisp = vq.BasisSet(sysp.unit_cell_molecule(), "cc-pvdz")
    from vibeqc._vibeqc_core import (
        PeriodicKSOptions as _PKSOpts,
        monkhorst_pack as _mp,
    )
    opts = _PKSOpts()
    opts.conv_tol_energy = 1e-9
    opts.max_iter = 80
    opts.lattice_opts.cutoff_bohr = 25.0
    opts.lattice_opts.nuclear_cutoff_bohr = 25.0
    km = _mp(sysp, [1, 1, 1])
    from vibeqc.pbc_bipole_uhf import run_pbc_bipole_uhf

    # Legacy Γ-local gauge and pre-M5 traversal pinned explicitly: the
    # analytic BIPOLE gradient implements those historical domains only.
    rp_u = run_pbc_bipole_uhf(
        sysp, basisp, km, opts, ewald_precision=1e-8, dft_plus_u=sites,
        use_exchange_ewald_split=False,
        sr_image_precision=None,
        use_fock_symmetry_reduce=False,
    )
    assert rp_u.converged
    g_p_u = np.asarray(vq.compute_bipole_gradient_uhf(
        sysp, basisp, rp_u, lattice_opts=opts.lattice_opts,
        kmesh=km, dft_plus_u=sites,
    ))
    rp_0 = run_pbc_bipole_uhf(
        sysp, basisp, km, opts, ewald_precision=1e-8,
        use_exchange_ewald_split=False,
        sr_image_precision=None,
        use_fock_symmetry_reduce=False,
    )
    assert rp_0.converged
    g_p_0 = np.asarray(vq.compute_bipole_gradient_uhf(
        sysp, basisp, rp_0, lattice_opts=opts.lattice_opts,
    ))

    mol = vq.Molecule(atoms, charge=1, multiplicity=2)
    basism = vq.BasisSet(mol, "cc-pvdz")
    uhf_opts = vq.UHFOptions()
    uhf_opts.conv_tol_energy = 1e-13
    uhf_opts.conv_tol_grad = 1e-11
    uhf_opts.max_iter = 200
    rm_u = vq.run_uhf(mol, basism, uhf_opts, dft_plus_u=sites)
    g_m_u = np.asarray(
        vq.compute_gradient_uhf(mol, basism, rm_u, dft_plus_u=sites)
    )
    rm_0 = vq.run_uhf(mol, basism, uhf_opts)
    g_m_0 = np.asarray(vq.compute_gradient_uhf(mol, basism, rm_0))

    # The +U contribution matches molecular within BIPOLE gradient
    # accuracy (~1 mHa/bohr per existing test_bipole_gradient bound).
    delta_p = g_p_u - g_p_0
    delta_m = g_m_u - g_m_0
    np.testing.assert_allclose(delta_p, delta_m, atol=1e-3)


def test_compute_bipole_gradient_uks_dft_plus_u_contribution_matches_molecular():
    """RKS-DFT analog of the UHF +U contribution test — same H₂⁺
    system, this time with LDA via the UKS BIPOLE driver. Validates
    the per-spin Pulay term for the closed-but-open-shell driver
    used for real Fe-slab UKS+U work."""
    ANGSTROM_TO_BOHR = 1.8897261339213
    big_box = 20.0 * ANGSTROM_TO_BOHR
    atoms = [
        vq.Atom(1, [0.0, 0.0, 0.0]),
        vq.Atom(1, [0.0, 0.0, 1.4]),
    ]
    sites = [vq.HubbardSite(0, 1, U_ev=4.0)]
    sysp = vq.PeriodicSystem(
        3, np.diag([big_box] * 3), atoms,
        charge=1, multiplicity=2,
    )
    basisp = vq.BasisSet(sysp.unit_cell_molecule(), "cc-pvdz")
    from vibeqc._vibeqc_core import (
        PeriodicKSOptions as _PKSOpts,
        monkhorst_pack as _mp,
    )
    from vibeqc.pbc_bipole_uks import run_pbc_bipole_uks
    ks_opts = _PKSOpts()
    ks_opts.functional = "lda"
    ks_opts.conv_tol_energy = 1e-8
    ks_opts.max_iter = 80
    ks_opts.lattice_opts.cutoff_bohr = 25.0
    ks_opts.lattice_opts.nuclear_cutoff_bohr = 25.0
    km = _mp(sysp, [1, 1, 1])

    # Legacy Γ-local gauge and pre-M5 traversal pinned explicitly (see the
    # UHF sibling above).
    rp_u = run_pbc_bipole_uks(
        sysp, basisp, km, ks_opts,
        ewald_precision=1e-8, dft_plus_u=sites,
        use_exchange_ewald_split=False,
        sr_image_precision=None,
        use_fock_symmetry_reduce=False,
    )
    assert rp_u.converged
    g_p_u = np.asarray(vq.compute_bipole_gradient_uks(
        sysp, basisp, rp_u, lattice_opts=ks_opts.lattice_opts,
        kmesh=km, dft_plus_u=sites,
    ))
    rp_0 = run_pbc_bipole_uks(
        sysp, basisp, km, ks_opts, ewald_precision=1e-8,
        use_exchange_ewald_split=False,
        sr_image_precision=None,
        use_fock_symmetry_reduce=False,
    )
    assert rp_0.converged
    g_p_0 = np.asarray(vq.compute_bipole_gradient_uks(
        sysp, basisp, rp_0, lattice_opts=ks_opts.lattice_opts,
    ))

    mol = vq.Molecule(atoms, charge=1, multiplicity=2)
    basism = vq.BasisSet(mol, "cc-pvdz")
    uks_opts = vq.UKSOptions()
    uks_opts.functional = "lda"
    uks_opts.conv_tol_energy = 1e-11
    uks_opts.conv_tol_grad = 1e-9
    uks_opts.max_iter = 200
    rm_u = vq.run_uks(mol, basism, uks_opts, dft_plus_u=sites)
    g_m_u = np.asarray(
        vq.compute_gradient_uks(mol, basism, rm_u, dft_plus_u=sites)
    )
    rm_0 = vq.run_uks(mol, basism, uks_opts)
    g_m_0 = np.asarray(vq.compute_gradient_uks(mol, basism, rm_0))

    delta_p = g_p_u - g_p_0
    delta_m = g_m_u - g_m_0
    # BIPOLE Γ-W noise scale (see UHF test above).
    np.testing.assert_allclose(delta_p, delta_m, atol=2e-3)


def test_run_periodic_job_optimize_dft_plus_u_uhf_runs(tmp_path):
    """``run_periodic_job(optimize=True, method="UHF", jk_method="bipole",
    dft_plus_u=...)`` now runs end-to-end (no NotImplementedError)
    because the multi-k BIPOLE +U Pulay gradient is wired."""
    c = 6.0
    sysp = vq.PeriodicSystem(
        3, np.diag([12.0] * 3),
        [vq.Atom(1, [c, c, c - 0.7]),
         vq.Atom(1, [c, c, c + 0.7])],
        charge=1, multiplicity=2,
    )
    basis = vq.BasisSet(sysp.unit_cell_molecule(), "cc-pvdz")
    # Pre-fix this raised NotImplementedError. Post-fix it runs the
    # SCF, then the optimize loop (may or may not converge in a few
    # iters — what matters is no exception, and the SCF inside the
    # first geometry step converged so result.energy was returned).
    result = vq.run_periodic_job(
        sysp, basis,
        method="UHF", jk_method="bipole", kpoints=(1, 1, 1),
        optimize=True, optimize_max_iter=2,
        output=str(tmp_path / "h2plus_opt_uhf_plus_u"),
        dft_plus_u=[vq.HubbardSite(0, 1, U_ev=4.0)],
        write_molden_file=False, write_xyz_file=False,
        write_poscar_file=False, write_xsf_structure_file=False,
        write_cif_file=False, write_population_file=False,
    )
    # The point of this test is that the run completed; the SCF inside
    # the first geometry step converged → result.energy is finite.
    assert np.isfinite(result.energy)


def test_run_pbc_bipole_rks_dft_plus_u_runs():
    """Direct smoke test of run_pbc_bipole_rks(..., dft_plus_u=[...])
    — the closed-shell RKS BIPOLE +U surface."""
    from vibeqc.pbc_bipole_rks import run_pbc_bipole_rks
    from vibeqc._vibeqc_core import (
        monkhorst_pack as _mp, PeriodicKSOptions as _PKSOpts,
    )
    ANGSTROM_TO_BOHR = 1.8897261339213
    sysp = vq.PeriodicSystem(
        3, np.diag([20.0 * ANGSTROM_TO_BOHR] * 3),
        [vq.Atom(1, [0.0, 0.0, 0.0]),
         vq.Atom(1, [0.0, 0.0, 1.4])],
    )
    basis = vq.BasisSet(sysp.unit_cell_molecule(), "cc-pvdz")
    opts = _PKSOpts()
    opts.functional = "lda"
    opts.conv_tol_energy = 1e-8
    opts.max_iter = 80
    opts.lattice_opts.cutoff_bohr = 25.0
    opts.lattice_opts.nuclear_cutoff_bohr = 25.0
    km = _mp(sysp, [1, 1, 1])
    sites = [vq.HubbardSite(0, 1, U_ev=4.0)]
    r = run_pbc_bipole_rks(
        sysp, basis, km, opts,
        ewald_precision=1e-8, dft_plus_u=sites,
    )
    assert r.converged
    assert r.e_dft_plus_u > 0.0
