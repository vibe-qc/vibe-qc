"""PT2-specific orbital Hessian H^E2_oo via numerical finite differences.

Computes the second derivative of E^2 w.r.t. orbital rotations:

    H^E2_oo[i,j] = (g(+e_j)[i] - g(-e_j)[i]) / (2e)

where g(±e_j) is the PT2 orbital gradient dE^2/dk evaluated at integrals
rotated by ±e along pair j.  This is O(n_pairs^2) PT2 solves -- feasible
for small active spaces (tested up to CAS(4,4)).
"""

from __future__ import annotations

import numpy as np
from scipy.linalg import expm


def build_pt2_orbital_hessian_fd(
    h1e_mo: np.ndarray,
    h2e_mo: np.ndarray,
    rdm1: np.ndarray,
    rdm2: np.ndarray,
    n_core: int,
    n_act: int,
    n_act_elec: int,
    nmo: int,
    pairs: list,
    *,
    variant: str = "caspt2",
    fd_eps_hess: float = 1e-3,
    fd_eps_grad: float = 1e-3,
) -> np.ndarray:
    """Build PT2-specific orbital-orbital Hessian via FD.

    Parameters
    ----------
    variant : str
        "caspt2" (generalized Fock H_0) or "nevpt2" (Dyall H_0).
    fd_eps_hess : float
        FD step for the Hessian columns.
    fd_eps_grad : float
        FD step for the orbital gradient dE^2/dk within each column.

    Returns H of shape (n_pairs, n_pairs).
    """
    from vibeqc.solvers._mrpt import (
        _add,
        _pt2_correction,
        _semicanonical_prep,
        apply_1body,
        apply_2body,
    )

    npr = len(pairs)
    g_phys = h2e_mo

    # E^2 evaluation at given integrals
    if variant == "nevpt2":

        def e2_at(h1, h2):
            P = _semicanonical_prep(h1, h2, n_core, n_act, n_act_elec, 0)
            norb = P["norb"]
            h1D = np.zeros((norb, norb))
            for p in range(n_core):
                h1D[p, p] = P["eps"][p]
            for p in range(n_core + n_act, norb):
                h1D[p, p] = P["eps"][p]
            h1D[P["act"], P["act"]] = P["h1a"]
            eriD = np.zeros_like(P["eri"])
            eriD[P["act"], P["act"], P["act"], P["act"]] = P["eri"][
                P["act"], P["act"], P["act"], P["act"]
            ]

            def apply_HD(state):
                return _add(
                    apply_1body(state, h1D, norb),
                    apply_2body(state, eriD, norb, idx=P["aidx"]),
                )

            return _pt2_correction(P, apply_HD)
    else:

        def e2_at(h1, h2):
            P = _semicanonical_prep(h1, h2, n_core, n_act, n_act_elec, 0)
            return _pt2_correction(P, lambda s: apply_1body(s, P["F"], P["norb"]))

    # dE^2/dk at given integrals via numerical FD
    def grad_at(h1, h2):
        g = np.zeros(npr)
        for i in range(npr):
            ei = np.zeros(npr)
            ei[i] = 1.0
            K = np.zeros((nmo, nmo))
            for j, (p, q) in enumerate(pairs):
                K[p, q] = fd_eps_grad * ei[j]
                K[q, p] = -fd_eps_grad * ei[j]
            Up = expm(K)
            Um = expm(-K)
            h1p = Up.T @ h1 @ Up
            h1m = Um.T @ h1 @ Um
            gp = np.einsum("ap,bq,cr,ds,abcd->pqrs", Up, Up, Up, Up, g_phys)
            gm = np.einsum("ap,bq,cr,ds,abcd->pqrs", Um, Um, Um, Um, g_phys)
            g[i] = (e2_at(h1p, gp) - e2_at(h1m, gm)) / (2.0 * fd_eps_grad)
        return g

    # Hessian via FD of the gradient
    H = np.zeros((npr, npr))
    for j in range(npr):
        ej = np.zeros(npr)
        ej[j] = 1.0
        Kp = np.zeros((nmo, nmo))
        Km = np.zeros((nmo, nmo))
        for k, (p, q) in enumerate(pairs):
            Kp[p, q] = fd_eps_hess * ej[k]
            Kp[q, p] = -fd_eps_hess * ej[k]
            Km[p, q] = -fd_eps_hess * ej[k]
            Km[q, p] = fd_eps_hess * ej[k]
        Up = expm(Kp)
        Um = expm(Km)
        h1p = Up.T @ h1e_mo @ Up
        h1m = Um.T @ h1e_mo @ Um
        gp = np.einsum("ap,bq,cr,ds,abcd->pqrs", Up, Up, Up, Up, g_phys)
        gm = np.einsum("ap,bq,cr,ds,abcd->pqrs", Um, Um, Um, Um, g_phys)
        gp_grad = grad_at(h1p, gp)
        gm_grad = grad_at(h1m, gm)
        H[:, j] = (gp_grad - gm_grad) / (2.0 * fd_eps_hess)

    return 0.5 * (H + H.T)
