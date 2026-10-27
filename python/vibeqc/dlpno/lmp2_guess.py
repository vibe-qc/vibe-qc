"""Coupled local MP2 in pair natural orbital spaces, for the PNO guess.

The local CCSD route builds its pair natural orbitals (PNOs) from
semicanonical local MP2 amplitudes by default: the occupied Fock matrix of
the localized orbitals is taken as diagonal. ``pno_guess="lmp2"`` instead
iterates the full local MP2 equations, whose stationary amplitudes are
invariant to the occupied localization:

    R_ij = K_ij - D_ij o T_ij - sum_k' [ f_ik S T_kj S^T + f_kj S T_ik S^T ] = 0

with ``D_ij[a, b] = f_ii + f_jj - e_a - e_b`` in the pair's semicanonical PNO
basis, the primed sums excluding ``k = i`` and ``k = j`` respectively, and
``S`` the overlap between the two pairs' PNO spaces. This is the residual of
Pulay and Saebo's local MP2 as Liakos, Sparta, Kesharwani, Martin and Neese
write it for DLPNO, J. Chem. Theory Comput. 11, 1525 (2015),
doi:10.1021/ct501129s, Sec. 2.2, where full local MP2 is the TightPNO option
for "the initial guess, PNO construction and perturbative correction for PNO
and weak-pair errors".

Every PNO here is a combination of projected atomic orbitals, so it lies in
the canonical virtual space and ``S_pq = U_p^T U_q`` with ``U`` the PNO
coefficients over the canonical virtual orbitals. The default sweep uses
that to apply the occupied coupling once per sweep in the canonical virtual
basis (one ``n_act x n_act`` matrix product over every pair's amplitudes)
instead of once per pair link. That dense path holds two
``n_act^2 n_vir^2`` arrays; past ``dense_max_gb`` the solver uses the
compiled per-link sweep of the DLPNO-MP2 route, ``dlpno_mp2_iterate``.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

try:
    from .._vibeqc_core import dlpno_mp2_iterate as _cpp_lmp2_iterate
except ImportError:  # pragma: no cover - depends on the compiled core
    _cpp_lmp2_iterate = None


def pair_energy(K: np.ndarray, T: np.ndarray, i: int, j: int) -> float:
    """Closed-shell pair energy ``(2 - delta_ij) sum K o (2T - T^T)``."""
    w = 1.0 if i == j else 2.0
    return w * float(np.sum(K * (2.0 * T - T.T)))


@dataclass
class LMP2Solution:
    #: Converged amplitudes per pair, in the PNO basis they were given in.
    T: dict
    #: Pair energies of those amplitudes.
    e_pair: dict
    e_corr: float = 0.0
    n_iter: int = 0
    converged: bool = False
    #: ``"dense"`` or ``"compiled"``: which sweep ran.
    sweep: str = ""
    trace: list = field(default_factory=list)


def _cut_fock(f_oo: np.ndarray, fock_cut: float) -> np.ndarray:
    """Off-diagonal occupied Fock coupling with ``|f_ik| < fock_cut`` dropped
    (ORCA's LMP2 Fock cutoff); the diagonal is zeroed."""
    F = np.array(f_oo, dtype=float, copy=True)
    np.fill_diagonal(F, 0.0)
    if fock_cut > 0.0:
        F[np.abs(F) < fock_cut] = 0.0
    return F


def solve_local_lmp2(
    keys,
    K: dict,
    eps: dict,
    U: dict,
    T0: dict,
    f_oo: np.ndarray,
    *,
    fock_cut: float = 1e-5,
    max_iter: int = 50,
    conv_tol_energy: float = 1e-7,
    conv_tol_residual: float = 5e-7,
    dense_max_gb: float = 16.0,
    C_vir: np.ndarray | None = None,
    S_ao: np.ndarray | None = None,
) -> LMP2Solution:
    """Iterate the coupled local MP2 equations to self-consistency.

    Parameters
    ----------
    keys
        Pairs ``(i, j)`` with ``i <= j`` over the localized active
        occupieds. A pair absent from ``keys`` has zero amplitudes.
    K, eps, U, T0
        Per pair: the exchange block ``(ia|jb)`` in the pair's PNO basis, the
        PNO orbital energies (the basis must be semicanonical), the PNO
        coefficients over the canonical virtual orbitals, and the starting
        amplitudes.
    f_oo
        Fock matrix over the localized active occupieds.
    fock_cut
        Off-diagonal ``|f_ik|`` below this do not couple pairs. 0 keeps them
        all, the exact setting.
    dense_max_gb
        Memory bound of the dense sweep. Larger systems, or 0, use the
        compiled sweep, which then needs ``C_vir`` and ``S_ao``.
    """
    keys = [tuple(int(x) for x in k) for k in keys]
    f_dd = np.diag(f_oo)
    F_off = _cut_fock(f_oo, float(fock_cut))
    n_act = f_oo.shape[0]
    T = {k: np.array(T0[k], dtype=float, copy=True) for k in keys}
    denom = {
        (i, j): f_dd[i] + f_dd[j] - eps[(i, j)][:, None] - eps[(i, j)][None, :]
        for i, j in keys
    }
    sol = LMP2Solution(T=T, e_pair={})
    if not keys:
        sol.converged = True
        return sol

    n_vir = next(iter(U.values())).shape[0]
    dense_bytes = 2.0 * 8.0 * n_act * n_act * n_vir * n_vir
    use_dense = dense_bytes <= float(dense_max_gb) * 1e9
    if not use_dense and _cpp_lmp2_iterate is None:  # pragma: no cover
        raise ValueError(
            "the coupled LMP2 guess needs "
            f"{dense_bytes / 1e9:.1f} GB for its dense sweep, above "
            f"dense_max_gb={dense_max_gb:g}, and this core has no compiled "
            "dlpno_mp2_iterate; raise lmp2_dense_max_gb"
        )
    sol.sweep = "dense" if use_dense else "compiled"
    if not use_dense:
        if C_vir is None or S_ao is None:
            raise ValueError("the compiled LMP2 sweep needs C_vir and S_ao")
        # dlpno_mp2_iterate takes the diagonal through F_oo as well.
        F_cpp = np.asfortranarray(F_off + np.diag(f_dd))
        S_cpp = np.asfortranarray(S_ao)
        pi = [k[0] for k in keys]
        pj = [k[1] for k in keys]
        K_list = [np.asfortranarray(K[k]) for k in keys]
        eps_list = [np.asarray(eps[k], dtype=float) for k in keys]
        V_list = [np.asfortranarray(C_vir @ U[k]) for k in keys]

    e_prev = None
    for it in range(int(max_iter)):
        if use_dense:
            # X[k, l] = U_kl T_kl U_kl^T over the canonical virtuals, with
            # X[l, k] = X[k, l]^T; then G1[i, j] = sum_k f_ik X[k, j] in one
            # product, and the residual's coupling block for pair (i, j) is
            # G1[i, j] + G1[j, i]^T (the second sum, by F's symmetry).
            X = np.zeros((n_act, n_act, n_vir, n_vir))
            for i, j in keys:
                Uij = U[(i, j)]
                Xij = Uij @ T[(i, j)] @ Uij.T
                X[i, j] = Xij
                if i != j:
                    X[j, i] = Xij.T
            G1 = (F_off @ X.reshape(n_act, -1)).reshape(n_act, n_act, n_vir, n_vir)
            del X
            max_r = 0.0
            new_T = {}
            for i, j in keys:
                Uij = U[(i, j)]
                G = G1[i, j] + G1[j, i].T
                coupling = Uij.T @ G @ Uij
                Tij = T[(i, j)]
                R = K[(i, j)] - denom[(i, j)] * Tij - coupling
                max_r = max(max_r, float(np.max(np.abs(R))))
                Tn = (K[(i, j)] - coupling) / denom[(i, j)]
                if i == j:
                    Tn = 0.5 * (Tn + Tn.T)
                new_T[(i, j)] = Tn
            del G1
            T.update(new_T)
        else:
            T_new, max_r, _ = _cpp_lmp2_iterate(
                pi,
                pj,
                K_list,
                eps_list,
                [np.asfortranarray(T[k]) for k in keys],
                V_list,
                F_cpp,
                S_cpp,
                1.0,
            )
            for idx, k in enumerate(keys):
                Tn = np.asarray(T_new[idx])
                if k[0] == k[1]:
                    Tn = 0.5 * (Tn + Tn.T)
                T[k] = Tn
        e_corr = sum(pair_energy(K[k], T[k], *k) for k in keys)
        delta = float("inf") if e_prev is None else e_corr - e_prev
        sol.trace.append(
            {"iter": it + 1, "e_corr": e_corr, "delta_e": delta, "max_r": max_r}
        )
        e_prev = e_corr
        if abs(delta) < conv_tol_energy and max_r < conv_tol_residual:
            sol.converged = True
            sol.n_iter = it + 1
            break
    else:
        sol.n_iter = int(max_iter)

    sol.e_pair = {k: pair_energy(K[k], T[k], *k) for k in keys}
    sol.e_corr = float(sum(sol.e_pair.values()))
    return sol
