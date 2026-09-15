# PRODUCTION — Dyall chain rule for NEVPT2 effective density corrections.
"""Dyall chain rule for NEVPT2 effective density corrections.

Computes the second-order correlation correction to NEVPT2 effective
densities by propagating the Dyall H0 orbital rotation derivatives
through the group expectation values.

The Dyall H0 is:
    H_D = \u03a3_p \u03b5_p E_pp (core+virtual, diagonal)
        + \u03a3_{tu\u2208act} h1a[t,u] E_tu
        + \u00bd \u03a3_{tuvx\u2208act} (tu|vx) E_tu E_vx

Its orbital rotation derivative involves 2-body commutators:
    [H_D, E_pq - E_qp] = [h1D, E_pq - E_qp] + \u00bd[eriD, E_pq - E_qp]

The 2-body commutator [eriD, E_pq] couples active 2e integrals to the
orbital rotation, producing additional effective density corrections
beyond the first-order Dyall Fock commutator.

For virtual-active pairs (p=virt, q=act), the dominant contribution is
from the Dyall Fock commutator acting on the group density:
    \u0394D_corr[p,q] = \u03a3_g w_g * (FD_g[qp] - FD_g[pq] - FD_ref[qp] + FD_ref[pq])

where FD is the Dyall Fock matrix, now built with full 1- and 2-RDM
contributions.

References
----------
_nevpt2_dyall_grad.py — _build_dyall_fock with 1+2-RDM contraction
"""

from __future__ import annotations

import numpy as np


def compute_dyall_fock(
    h1D: np.ndarray,
    eriD_chem: np.ndarray,
    dm1_act: np.ndarray,
    dm2_act: np.ndarray,
    n_core: int,
    n_act: int,
    norb: int,
) -> np.ndarray:
    """Build the full Dyall Fock matrix with 1- and 2-RDM contributions.

    F^D[p,q] = h1D[p,q]                                                                 (1-body)
             + \u03a3_{tu} dm1[t,u] * (2(pq|tu) - (pu|tq))                              (Coulomb+exchange)
             + \u03a3_{uvw} dm2[t,u,v,w] * (q,u|v,w)   for q in full space             (2-RDM coupling)

    The 2-RDM term captures the active-space correlation contribution to
    the orbital rotation gradient, which the 1-RDM-only Fock misses.

    Returns F^D of shape (norb, norb).
    """
    FD = np.zeros((norb, norb))

    # 1-body part: h1D (diagonal for core/virtual, full h1a for active)
    FD[:, :] = h1D

    # 1-RDM Coulomb + exchange (standard Fock)
    for t_idx, t in enumerate(range(n_core, n_core + n_act)):
        for u_idx, u in enumerate(range(n_core, n_core + n_act)):
            d1 = dm1_act[t_idx, u_idx]
            if abs(d1) < 1e-15:
                continue
            for p in range(norb):
                for q in range(norb):
                    coulomb = eriD_chem[p, q, t, u]  # (pq|tu)
                    exchange = eriD_chem[p, u, t, q]  # (pu|tq)
                    FD[p, q] += d1 * (2.0 * coulomb - exchange)

    # 2-RDM contribution: F^D[t,q] += \u03a3_{uvw} dm2[t,u,v,w] * (q,u|v,w)
    # This is the 2-body part of the Dyall Fock that couples the 2-RDM
    # to the orbital rotation gradient via the 2e integrals.
    if n_act > 0 and dm2_act is not None and np.any(np.abs(dm2_act) > 1e-15):
        for t_idx, t in enumerate(range(n_core, n_core + n_act)):
            for u_idx, u in enumerate(range(n_core, n_core + n_act)):
                for v_idx, v in enumerate(range(n_core, n_core + n_act)):
                    for w_idx, w in enumerate(range(n_core, n_core + n_act)):
                        d2 = dm2_act[t_idx, u_idx, v_idx, w_idx]
                        if abs(d2) < 1e-15:
                            continue
                        for q in range(norb):
                            FD[t, q] += d2 * eriD_chem[q, u, v, w]

    return FD


def dyall_chain_rule_correction(
    prep: dict,
    DeltaD_full: np.ndarray,
    n_core: int,
    n_act: int,
) -> np.ndarray:
    """Dyall-specific chain-rule correction to NEVPT2 effective 1-RDM.

    Computes \u0394\u0394D[p,q] from the Dyall Fock commutator expectation values
    of the perturber groups, using full 1- and 2-RDM Fock builds.
    This captures the 2-body contributions that the first-order
    effective density misses.

    Returns \u0394\u0394D of shape (n_act, n_act) -- additive correction to DeltaD.
    """
    from ..solvers._mrpt import _add, _dot, apply_1body, apply_2body
    from ._pt2_density import _full_1rdm_from_state, _full_2rdm_from_state

    norb = prep["norb"]
    ref = prep["ref"]
    groups = prep["groups"]
    act_s = slice(n_core, n_core + n_act)

    # Dyall H0
    h1D = np.zeros((norb, norb))
    for p in range(n_core):
        h1D[p, p] = prep["eps"][p]
    for p in range(n_core + n_act, norb):
        h1D[p, p] = prep["eps"][p]
    h1D[prep["act"], prep["act"]] = prep["h1a"]
    eriD_chem = np.zeros_like(prep["eri"])
    eriD_chem[prep["act"], prep["act"], prep["act"], prep["act"]] = prep["eri"][
        prep["act"], prep["act"], prep["act"], prep["act"]
    ]

    def apply_HD(state):
        return _add(
            apply_1body(state, h1D, norb),
            apply_2body(state, eriD_chem, norb, idx=prep["aidx"]),
        )

    E0 = _dot(ref, apply_HD(ref))

    # Reference Dyall Fock with full 1- and 2-RDM contributions
    D_ref_full = _full_1rdm_from_state(ref, norb)
    D_ref = D_ref_full[act_s, act_s]
    G2_ref_full = _full_2rdm_from_state(ref, norb)
    G2_ref = G2_ref_full[act_s, act_s, act_s, act_s]
    FD_ref = compute_dyall_fock(h1D, eriD_chem, D_ref, G2_ref, n_core, n_act, norb)

    DeltaD_corr = np.zeros((n_act, n_act))

    for V in groups.values():
        N_g = _dot(V, V)
        if N_g < 1e-14:
            continue
        F_g = _dot(V, apply_HD(V)) / N_g
        denom = F_g - E0
        if abs(denom) < 1e-12:
            continue
        w_g = N_g / (denom * denom)

        D_g_full = _full_1rdm_from_state(V, norb)
        D_g = D_g_full[act_s, act_s]
        G2_g_full = _full_2rdm_from_state(V, norb)
        G2_g = G2_g_full[act_s, act_s, act_s, act_s]

        FD_g = compute_dyall_fock(h1D, eriD_chem, D_g, G2_g, n_core, n_act, norb)

        # Chain-rule contribution: the Fock commutator difference
        # \u0394\u0394D[p,q] += w_g * (FD_g[qp] - FD_g[pq] - FD_ref[qp] + FD_ref[pq]) / N_g
        # The antisymmetric Fock commutator captures the orbital relaxation
        # response of the Dyall energy to the group density.
        for p in range(n_act):
            for q in range(n_act):
                _p = n_core + p
                _q = n_core + q
                dg_ref = FD_ref[_q, _p] - FD_ref[_p, _q]
                dg_g = FD_g[_q, _p] - FD_g[_p, _q]
                DeltaD_corr[p, q] += w_g * (dg_g / max(N_g, 1e-14) - dg_ref)

    return DeltaD_corr
