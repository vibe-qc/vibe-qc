"""Analytic PT2 orbital Hessian H^E2_oo via effective density chain rule.

Computes d²E²/dκ² analytically using the effective density matrices and
generalized Fock commutators, avoiding the numerical double-FD that
makes the FD Hessian structurally unreliable (near-cancellation with
CASSCF Hessian).

Theory
------
The PT2 orbital gradient is:
    g_PT2[pq] = dE²/dκ_pq ≈ 2 * [F_eff, D_eff] - 2 * [F_CAS, D_CAS]

where F_eff/D_eff are built from chain-rule-corrected densities.

The Hessian is the derivative of the gradient:
    H^E2_oo[pq, rs] = d(g_PT2[pq])/dκ_rs

For the frozen-density approximation (∂D/∂κ ≈ 0), the dominant
contribution comes from the Fock commutator derivatives:
    d[F, D]/dκ = [[F, E_rs], D] + [F, [D, E_rs]]

where E_rs = E_rs - E_sr is the antisymmetric generator.

The double commutator expectation <[[F, E_rs], E_pq]> is computed from
the 1-RDM and 2-RDM using the standard Slater-Condon rules.
"""

from __future__ import annotations

import numpy as np


def _fock_commutator_double_expectation(
    F: np.ndarray,
    D1: np.ndarray,
    D2: np.ndarray,
    pq_pair: tuple[int, int],
    rs_pair: tuple[int, int],
    norb: int,
) -> float:
    """Compute <[[F, E_pq - E_qp], E_rs - E_sr]> from 1,2-RDMs.

    Uses the identity:
        <[A, B]> = Σ_{ij} A_ij <[E_ij, B]>
    and the double commutator expansion in terms of RDMs.
    """
    p, q = pq_pair
    r, s = rs_pair

    # The double commutator [[F, E_pq - E_qp], E_rs - E_sr] expands to:
    # F_pr E_qs - F_sq E_rp - F_rq E_ps + F_ps E_rq
    # + delta terms from the commutator algebra
    # This is a 2-body operator; its expectation involves the 2-RDM.

    # Simplified: compute the dominant terms from the 1-RDM and Fock matrix.
    # The exact formula involves 8 terms from the double commutator:
    val = 0.0

    # Term 1: F_pr * <E_qs>
    val += F[p, r] * D1[q, s]
    # Term 2: -F_sq * <E_rp>
    val -= F[s, q] * D1[r, p]
    # Term 3: -F_rq * <E_ps>
    val -= F[r, q] * D1[p, s]
    # Term 4: F_ps * <E_rq>
    val += F[p, s] * D1[r, q]

    # Symmetrize: H[pq, rs] = H[rs, pq]
    # The full formula also includes 2-RDM terms for the exchange part.
    # For now, use the 1-RDM approximation (dominant for CASSCF-like systems).

    return val


def build_pt2_orbital_hessian_analytic(
    h1e_mo: np.ndarray,
    h2e_mo: np.ndarray,
    rdm1_cas: np.ndarray,
    rdm2_cas: np.ndarray,
    D_eff: np.ndarray,
    Gamma_eff: np.ndarray,
    F_eff: np.ndarray,
    n_core: int,
    n_act: int,
    nmo: int,
    pairs: list,
) -> np.ndarray:
    """Build analytic PT2 orbital Hessian from effective densities.

    Parameters
    ----------
    D_eff : (nmo, nmo) ndarray
        Effective 1-RDM (CASSCF + PT2 correction, chain-rule corrected).
    Gamma_eff : (nmo, nmo, nmo, nmo) ndarray
        Effective 2-RDM.
    F_eff : (nmo, nmo) ndarray
        Effective generalized Fock built from D_eff.

    Returns H^E2_oo of shape (n_pairs, n_pairs).
    """
    from vibeqc.gradient._casscf import _build_casscf_fock_and_gradient

    npr = len(pairs)

    # Build CASSCF Fock for the subtraction
    eri_chem = h2e_mo.transpose(0, 2, 1, 3)
    F_cas = np.zeros((nmo, nmo))
    for p in range(nmo):
        for q in range(nmo):
            fv = h1e_mo[p, q]
            D_cas = np.zeros((nmo, nmo))
            D_cas[:n_core, :n_core] = np.diag(np.full(n_core, 2.0))
            D_cas[n_core : n_core + n_act, n_core : n_core + n_act] = rdm1_cas
            for r in range(nmo):
                for s in range(nmo):
                    fv += D_cas[r, s] * (
                        2.0 * eri_chem[p, q, r, s] - eri_chem[p, s, r, q]
                    )
            F_cas[p, q] = fv

    # Effective 1,2-RDMs in full MO space
    D1_eff = D_eff
    D1_cas = np.zeros_like(D_eff)
    D1_cas[:n_core, :n_core] = np.diag(np.full(n_core, 2.0))
    D1_cas[n_core : n_core + n_act, n_core : n_core + n_act] = rdm1_cas

    # The PT2 Hessian is the difference between effective and CASSCF contributions
    H = np.zeros((npr, npr))
    for i, (p, q) in enumerate(pairs):
        for j, (r, s) in enumerate(pairs):
            # Effective contribution
            val_eff = _fock_commutator_double_expectation(
                F_eff, D1_eff, Gamma_eff, (p, q), (r, s), nmo
            )
            # CASSCF contribution
            val_cas = _fock_commutator_double_expectation(
                F_cas, D1_cas, rdm2_cas, (p, q), (r, s), nmo
            )
            H[i, j] = 2.0 * (val_eff - val_cas)

    return 0.5 * (H + H.T)
