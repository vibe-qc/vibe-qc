"""Analytic NEVPT2 CI Lagrangian via transition density matrices.

Computes dE^2/dc for SC-NEVPT2 using the Dyall H_0 effective density
derivatives contracted with transition 1,2-RDMs between the ground state
and each excited CI direction.

This is O(n_det * n_act^4) instead of O(n_det) PT2 solves (the numerical
FD approach).
"""

from __future__ import annotations

import numpy as np


def _make_transition_rdm12(c_bra, c_ket, det_list, norb):
    """Transition 1,2-RDMs between two CI vectors for the active space.

    rdm1[p,q] = <c_bra|E_pq|c_ket>
    rdm2[p,q,r,s] = <c_bra|a+_p a+_r a_s a_q|c_ket>

    Uses the same E_pq machinery as vibeqc.solvers._rdm.make_rdm12.
    """
    from vibeqc.solvers._rdm import _index_map, _reorder_rdm2, apply_E

    idx = _index_map(det_list)
    Ec_ket = {}
    EcT_bra = {}
    rdm1 = np.zeros((norb, norb))
    for r in range(norb):
        for s in range(norb):
            Ec_ket[(r, s)] = apply_E(r, s, c_ket, det_list, idx)
            EcT_bra[(r, s)] = apply_E(s, r, c_bra, det_list, idx)
            rdm1[r, s] = float(c_bra @ Ec_ket[(r, s)])

    T2 = np.zeros((norb, norb, norb, norb))
    for r in range(norb):
        for s in range(norb):
            v_ket = Ec_ket[(r, s)]
            for p in range(norb):
                for q in range(norb):
                    T2[p, q, r, s] = float(EcT_bra[(p, q)] @ v_ket)

    rdm2 = _reorder_rdm2(rdm1, T2)
    return rdm1, rdm2


def compute_nevpt2_ci_lagrangian_analytic(
    h1e_mo: np.ndarray,
    h2e_mo: np.ndarray,
    ci_coeffs: np.ndarray,
    determinants: list,
    n_core: int,
    n_act: int,
    n_act_elec: int,
) -> np.ndarray:
    """Analytic CI Lagrangian dE^2/dc for SC-NEVPT2 via transition RDMs.

    Uses the chain rule through the Dyall H_0 effective density derivatives:

        dE^2/dc_k = Tr(D_eff . gamma_trans) + Tr(Gamma_eff . Gamma_trans)

    where D_eff, Gamma_eff are the NEVPT2 effective 1,2-RDM corrections
    (computed once via compute_nevpt2_effective_density), and
    gamma_trans, Gamma_trans are transition RDMs between the excited
    CI direction |k> and the ground state |0>.

    Returns dE^2/dc of shape (n_det-1,) in the CI Hamiltonian eigenbasis.
    """
    from vibeqc.gradient._casscf import _build_ci_hamiltonian
    from vibeqc.gradient._pt2_density import compute_nevpt2_effective_density
    from vibeqc.solvers._mrpt import _semicanonical_prep

    # CI Hamiltonian eigenbasis
    H_det = _build_ci_hamiltonian(h1e_mo, h2e_mo, determinants, n_core, n_act)
    _E_det, U_det = np.linalg.eigh(H_det)
    n_det = len(determinants)
    if n_det <= 1:
        return np.zeros(0)

    # Effective density derivatives (active-space only)
    P0 = _semicanonical_prep(h1e_mo, h2e_mo, n_core, n_act, n_act_elec, 0)
    DeltaD_full, DeltaGamma_full = compute_nevpt2_effective_density(P0, n_core, n_act)

    act_s = slice(n_core, n_core + n_act)
    D_eff = DeltaD_full[act_s, act_s]
    G_eff = DeltaGamma_full[act_s, act_s, act_s, act_s]

    # Active-space determinant list (determinants are in active-relative
    # indexing, so they index 0..n_act-1 directly).
    dets_act = [(tuple(a), tuple(b)) for a, b in determinants]
    ci_act = ci_coeffs.copy()

    # For each excited eigenvector direction k, compute transition RDMs
    # and contract with effective density derivatives.
    dE2_dc = np.zeros(n_det - 1)
    for k in range(1, n_det):
        ek_vec = U_det[:, k]

        rdm1_t, rdm2_t = _make_transition_rdm12(ek_vec, ci_act, dets_act, n_act)

        val = float(np.sum(D_eff * rdm1_t))
        val += float(np.sum(G_eff * rdm2_t))
        dE2_dc[k - 1] = val

    return dE2_dc
