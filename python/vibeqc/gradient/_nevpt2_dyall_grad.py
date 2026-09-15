# EXPERIMENTAL - validated Dyall Fock, pending chain-rule integration.
# See handovers/HANDOVER_GATED_ITEMS.md. Numerical FD in _nevpt2.py is production.
"""Analytic NEVPT2 orbital gradient dE^2/dk via Dyall Fock commutators.

For NEVPT2 with Dyall H_0, the orbital gradient has a simple analytic form:

    dE^2/dk_{pq} = Sum_g (2*N_g/Delta_g^2) *
        [(F^D_g_{qp} - F^D_g_{pq})/N_g - (F^D_{qp} - F^D_{pq})]

where F^D is the Dyall Fock built from the reference 1,2-RDMs and F^D_g
from the perturber-group 1,2-RDMs.  The Dyall Fock is:

    F^D[p,q] = epsilon_p * delta_{pq}  (core + virtual, diagonal)
    F^D[t,q] = Sum_u D1[t,u] * h1a[u,q] + Sum_{uvw} D2[t,u,v,w] * (q u|v w)

This is O(1) instead of O(n_pairs) numerical FD, and is exact for Dyall H_0
(which does not depend on CI coefficients or the generalized Fock chain rule).
"""

from __future__ import annotations

import numpy as np


def compute_nevpt2_orbital_gradient(
    prep: dict,
    h1e_mo: np.ndarray,
    h2e_mo: np.ndarray,
    rdm1_ref: np.ndarray,
    rdm2_ref: np.ndarray,
    n_core: int,
    n_act: int,
    nmo: int,
    pairs: list,
) -> np.ndarray:
    """Analytic dE^2/dk for SC-NEVPT2 via Dyall Fock commutator.

    Parameters
    ----------
    prep : dict
        From _semicanonical_prep.
    Returns grad of shape (n_pairs,).
    """
    from vibeqc.gradient._pt2_density import (
        _full_1rdm_from_state,
        _full_2rdm_from_state,
    )
    from vibeqc.solvers._mrpt import _add, _dot, apply_1body, apply_2body

    norb = prep["norb"]
    ref = prep["ref"]
    groups = prep["groups"]
    act = prep["act"]
    aidx = prep["aidx"]

    # Dyall H_0: diagonal eps on core/virtual, full h1a + 2e on active
    h1D = np.zeros((norb, norb))
    for p in range(n_core):
        h1D[p, p] = prep["eps"][p]
    for p in range(n_core + n_act, norb):
        h1D[p, p] = prep["eps"][p]
    h1D[act, act] = prep["h1a"]

    eriD = np.zeros_like(prep["eri"])
    eriD[act, act, act, act] = prep["eri"][act, act, act, act]

    def apply_HD(state):
        return _add(
            apply_1body(state, h1D, norb),
            apply_2body(state, eriD, norb, idx=aidx),
        )

    E0 = _dot(ref, apply_HD(ref))

    # Precompute Dyall Fock for reference (F^D from rdm1_ref, rdm2_ref)
    FD_ref = _build_dyall_fock(h1D, eriD, rdm1_ref, rdm2_ref, n_core, n_act, norb)

    # Compute w_g = N_g / Delta_g^2 for each group
    # and accumulate the Fock-weighted gradient
    grad = np.zeros(len(pairs))
    for V in groups.values():
        N_g = _dot(V, V)
        if N_g < 1e-14:
            continue
        F_g = _dot(V, apply_HD(V)) / N_g
        denom = F_g - E0
        if abs(denom) < 1e-12:
            continue
        w_g = N_g / (denom * denom)  # corrected v35

        # Dyall Fock for perturber group
        # Use full 1,2-RDMs from the sparse state
        from vibeqc.gradient._pt2_density import (
            _full_1rdm_from_state,
            _full_2rdm_from_state,
        )

        D_g_full = _full_1rdm_from_state(V, norb)
        D_g = D_g_full[act, act]
        # 2-RDM of perturber group: expensive but only needed here
        G_g = _full_2rdm_from_state(V, norb)
        G_g_act = G_g[act, act, act, act]

        FD_g = _build_dyall_fock(h1D, eriD, D_g, G_g_act, n_core, n_act, norb)

        # Accumulate: grad[pq] += 2*w_g * [(FD_g_{qp} - FD_g_{pq})/N_g - (FD_ref_{qp} - FD_ref_{pq})]
        for idx, (p, q) in enumerate(pairs):
            dg_ref = 2.0 * (FD_ref[q, p] - FD_ref[p, q])
            dg_g = 2.0 * (FD_g[q, p] - FD_g[p, q]) / max(N_g, 1e-14)
            grad[idx] += w_g * (dg_g - dg_ref)

    return grad


def _build_dyall_fock(
    h1D: np.ndarray,
    eriD: np.ndarray,
    dm1_act: np.ndarray,
    dm2_act: np.ndarray,
    n_core: int,
    n_act: int,
    norb: int,
) -> np.ndarray:
    """Build the Dyall Fock matrix F^D for the orbital gradient.

    F^D[p,q] = h1D[p,q] for core/virtual (diagonal eps)
    F^D[t,q] = Sum_u dm1[t,u] * h1D[u,q] + Sum_{uvw} dm2[t,u,v,w] * (q u|v w)

    The core/virtual blocks are diagonal (F^D[i,i] = eps_i, F^D[a,a] = eps_a).
    For the off-diagonal blocks (core-active, active-virtual, etc.), only
    the Fock commutator 2*(F^D_{qp} - F^D_{pq}) enters the gradient, so the
    diagonal nature makes these contributions zero.

    Returns F^D of shape (norb, norb).
    """
    act = slice(n_core, n_core + n_act)
    FD = np.zeros((norb, norb))

    # Core/virtual diagonal
    FD[:n_core, :n_core] = np.diag(np.diag(h1D[:n_core, :n_core]))
    FD[n_core + n_act :, n_core + n_act :] = np.diag(
        np.diag(h1D[n_core + n_act :, n_core + n_act :])
    )

    # Active rows: F^D[t,q] = dm1_tu · h1D_uq + dm2_tuvw · (q u|v w)
    if n_act > 0:
        # 1-body part: dm1 @ h1D[act, :]
        FD[act, :] = dm1_act @ h1D[act, :]
        # 2-body part: dm2[t,u,v,w] * (q,u|v,w) in chemist's notation
        # where (q,u|v,w) = g_phys[q,v,u,w] = g_chem[q,u,v,w]
        # eriD[act, act, act, act] is chemist's (tu|vw) on active indices
        # We need (q,u|v,w) for q in full space, u,v,w in active space
        # This is dm2[t,u,v,w] * eriD[q,u,v,w] over u,v,w
        for t_idx, t in enumerate(range(n_core, n_core + n_act)):
            for q in range(norb):
                val = 0.0
                for u_idx, u in enumerate(range(n_core, n_core + n_act)):
                    for v_idx, v in enumerate(range(n_core, n_core + n_act)):
                        for w_idx, w in enumerate(range(n_core, n_core + n_act)):
                            d2 = dm2_act[t_idx, u_idx, v_idx, w_idx]
                            if abs(d2) > 1e-15:
                                val += d2 * eriD[q, u, v, w]
                if abs(val) > 1e-15:
                    FD[t, q] += val

    return FD
