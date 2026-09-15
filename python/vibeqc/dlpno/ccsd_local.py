"""Local DLPNO-CCSD integral blocks -- M3c step 4 (foundation).

The O(N⁶) correctness pilot (`dlpno.ccsd`) expands every pair's
amplitudes to the full spin-orbital space each iteration. The
reduced-scaling DLPNO-CCSD instead evaluates the closed-shell residual
*inside each pair's PNO basis*, never forming a full-system tensor.

Every integral block the closed-shell DF-CCSD residual needs (DePrince
& Sherrill, J. Chem. Phys. 139, 174102 (2013)) is a contraction of two
density-fitted three-index tensors:

    (pq|rs) = S_P B^P_{pq} B^P_{rs}.

So the local foundation is a small set of per-pair PNO-basis B-tensors:

    B_i  [naux, n_pno]            B^P_{i a}     (occ i, pair PNOs a)
    B_j  [naux, n_pno]            B^P_{j b}
    B_ij [naux]                   B^P_{i j}
    B_vv [naux, n_pno, n_pno]     B^P_{a b}     (pair PNOs a,b)

from which K=(ia|jb), J=(ij|ab), the 3-external (ia|bc) and the
4-external ladder (ab|cd) follow by contraction -- the last two kept
*factored* (never materialised as n_pno^3/n_pno⁴ dense tensors larger
than they must be). This module builds and validates that foundation;
remaining production gates live in ``handovers/HANDOVER_GATED_ITEMS.md``.

These integrals are the global RI in the pair's PNO basis (exact RI).
Composing with `local_df`-style domain-restricted fitting is a later
refinement; the residual physics is validated against the pilot first.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np


@dataclass
class PairCCIntegrals:
    """Density-fitted CCSD integral primitives for one pair, PNO basis.

    All virtual indices run over the pair's ``n_pno`` PNOs. Higher
    integral blocks are exposed as methods so the n_pno^3/n_pno⁴ tensors
    are formed only where a contraction genuinely needs them.
    """

    i: int
    j: int
    n_pno: int
    B_i: np.ndarray   # (naux, n_pno)   B^P_{i a}
    B_j: np.ndarray   # (naux, n_pno)   B^P_{j b}
    B_ij: np.ndarray  # (naux,)         B^P_{i j}
    B_vv: np.ndarray  # (naux, n_pno, n_pno)  B^P_{a b}

    def K(self) -> np.ndarray:
        """Exchange block K_{ij}^{ab} = (ia|jb), shape (n_pno, n_pno)."""
        return self.B_i.T @ self.B_j

    def J(self) -> np.ndarray:
        """Coulomb block J_{ij}^{ab} = (ij|ab), shape (n_pno, n_pno)."""
        return np.einsum("P,Pab->ab", self.B_ij, self.B_vv)

    def three_ext_i(self) -> np.ndarray:
        """3-external (ia|bc) for occ i, shape (n_pno, n_pno, n_pno)."""
        return np.einsum("Pa,Pbc->abc", self.B_i, self.B_vv)

    def four_ext(self) -> np.ndarray:
        """4-external ladder (ab|cd), shape (n_pno,)*4. Use sparingly."""
        return np.einsum("Pab,Pcd->abcd", self.B_vv, self.B_vv)


def build_pair_occ_pno(df, C_occ: np.ndarray, V_pno: np.ndarray) -> np.ndarray:
    """Occupied->PNO half-transformed B-tensor B^P_{m a}, all occupieds.

    Shape (n_aux, n_occ, n_pno). From it every CCSD integral block with
    one occupied and one pair-PNO virtual index follows by contraction:
    (ma|nb) = S_P B^P_{ma} B^P_{nb}, (mf|ab) = S_P B^P_{mf} B^P_{ab}, ...
    """
    return np.asarray(df.mo_transform(C_occ, V_pno))  # (n_aux, n_occ, n_pno)


def pno_overlap(V_p: np.ndarray, V_q: np.ndarray, S_ao: np.ndarray) -> np.ndarray:
    """Cross-pair PNO overlap S_pq = V_pᵀ S V_q, shape (n_p, n_q).

    Projects pair q's amplitudes into pair p's PNO basis:
    ``project_amplitude(S_pq, T_q)``.
    """
    return V_p.T @ S_ao @ V_q


def project_amplitude(S_pq: np.ndarray, T_q: np.ndarray) -> np.ndarray:
    """Project a doubles amplitude from pair q's PNO basis into pair p's.

    T_p^{ab} = S_{cd} S_pq[a,c] T_q^{cd} S_pq[b,d]  (both virtual indices).
    Exact when both PNO spaces are subspaces of the same virtual space.
    """
    return S_pq @ T_q @ S_pq.T


def local_doubles_ladder_residual(
    i: int,
    j: int,
    B_occ: np.ndarray,
    B_vv: np.ndarray,
    B_oo: np.ndarray,
    T_ij: np.ndarray,
    project_into_ij,
) -> np.ndarray:
    """Hole/particle-ladder part of the local closed-shell T2 residual.

    The three terms of the closed-shell DF-CCSD T2 residual that carry no
    Ph (pair-exchange) symmetriser -- evaluated entirely in pair (i,j)'s
    PNO basis with cross-pair amplitudes projected in:

        R_ij^ab = (ia|jb)
                + S_mn  P_ij(t_mn)^ab  W_mnij
                + S_ef  t_ij^ef  W_abef                              (t = T at t1=0)

        W_mnij = (mi|nj) + 1/2 S_ef t_ij^ef (me|nf)
        W_abef = (ae|bf) + 1/2 S_mn P_ij(t_mn)^ab (me|nf)

    where ``project_into_ij(m, n) -> P_ij(t_mn)`` returns the (m,n) pair's
    amplitudes projected into [ij] (n_pno x n_pno). Validated to machine
    precision against the full-space residual projected into [ij]
    (``tests/test_dlpno_ccsd_local.py``). The Ph-half (Fock ladders +
    W1/W2/WX rings) and singles complete the residual -- next M3c step.

    Parameters
    ----------
    B_occ : (n_aux, n_occ, n_pno)   B^P_{m a}  (occupied->[ij] PNO)
    B_vv  : (n_aux, n_pno, n_pno)   B^P_{a b}  ([ij] PNO pair)
    B_oo  : (n_aux, n_occ, n_occ)   B^P_{m n}  (occupied global)
    T_ij  : (n_pno, n_pno)          this pair's doubles amplitudes
    """
    n_occ = B_occ.shape[1]
    # (me|nf):  [m, n, e, f]
    me_nf = np.einsum("Pme,Pnf->mnef", B_occ, B_occ, optimize=True)
    # (mi|nj):  [m, n]
    mi_nj = np.einsum("Pm,Pn->mn", B_oo[:, :, i], B_oo[:, :, j], optimize=True)
    # (ae|bf):  [a, e, b, f]
    ae_bf = np.einsum("Pae,Pbf->aebf", B_vv, B_vv, optimize=True)

    Pt = np.stack(
        [np.stack([project_into_ij(m, n) for n in range(n_occ)]) for m in range(n_occ)]
    )  # (m, n, a, b)

    R = B_occ[:, i, :].T @ B_occ[:, j, :]  # (ia|jb)
    W_mnij = mi_nj + 0.5 * np.einsum("ef,mnef->mn", T_ij, me_nf, optimize=True)
    R += np.einsum("mn,mnab->ab", W_mnij, Pt, optimize=True)
    W_abef = ae_bf + 0.5 * np.einsum("mnab,mnef->aebf", Pt, me_nf, optimize=True)
    R += np.einsum("ef,aebf->ab", T_ij, W_abef, optimize=True)
    return R


def local_t2_residual_doubles(
    i: int,
    j: int,
    B_occ: np.ndarray,
    B_vv: np.ndarray,
    B_oo: np.ndarray,
    f_oo: np.ndarray,
    f_vv_ij: np.ndarray,
    T_ij: np.ndarray,
    project,
) -> np.ndarray:
    """Complete closed-shell T2 (doubles / CCD-level, t1=0) residual for a
    pair, evaluated entirely in its PNO basis with cross-pair amplitudes
    projected in -- the reduced-scaling counterpart of `cs_ccsd_residual`'s
    T2 residual.

    Includes every T2 term that survives at t1=0: the ladders
    (`local_doubles_ladder_residual`), the Fock ladders
    `Ph[S_e t_ij^ae F_be - S_m t_im^ab F_mj]`, and the three ring
    intermediates W1/W2/WX. Adding the singles (t1) terms upgrades this
    to full CCSD -- the final M3c-5b-ii slice.

    **Exact in the full-domain limit** (every pair's PNO space = the full
    virtual space => all projections are identity): then this reproduces
    canonical closed-shell CCD bit-for-bit (validated to ~5e-16,
    `tests/test_dlpno_ccsd_local.py`). With truncated PNO domains the
    cross-domain contractions in the intermediates are projected into
    [ij] -- the controlled DLPNO domain approximation, validated by energy
    recovery rather than exact parity.

    Parameters
    ----------
    B_occ : (n_aux, n_occ, n_pno)   B^P_{ma}     (occupied -> [ij] PNO)
    B_vv  : (n_aux, n_pno, n_pno)   B^P_{ab}     ([ij] PNO pair)
    B_oo  : (n_aux, n_occ, n_occ)   B^P_{mn}     (occupied, global)
    f_oo  : (n_occ, n_occ)          occupied Fock block
    f_vv_ij : (n_pno, n_pno)        virtual Fock in [ij]'s PNO basis
    T_ij  : (n_pno, n_pno)          this pair's amplitudes
    project : callable(m, n) -> (n_pno, n_pno)
        Pair (m,n)'s amplitudes projected into [ij].
    """
    n_occ = B_occ.shape[1]
    R = local_doubles_ladder_residual(i, j, B_occ, B_vv, B_oo, T_ij, project)

    me_nf = np.einsum("Pme,Pnf->mnef", B_occ, B_occ, optimize=True)
    L = 2.0 * me_nf - me_nf.transpose(0, 1, 3, 2)              # 2(me|nf)-(mf|ne)
    me_jb = np.einsum("Pme,Pjb->mejb", B_occ, B_occ, optimize=True)
    mj_be = np.einsum("Pmj,Pbe->mejb", B_oo, B_vv, optimize=True)
    P = {(m, n): project(m, n) for m in range(n_occ) for n in range(n_occ)}

    # F_ae[a,e] = f_vv_ij - S_mnf P(t_mn)[a,f] L[m,n,e,f]
    F_ae = f_vv_ij.copy()
    for m in range(n_occ):
        for n in range(n_occ):
            F_ae -= np.einsum("af,ef->ae", P[(m, n)], L[m, n], optimize=True)
    # F_mi[m,i'] = f_oo + S_n P(t_{i' n})[e,f] L[m,n,e,f]
    F_mi = f_oo.copy()
    for m in range(n_occ):
        for ip in range(n_occ):
            for n in range(n_occ):
                F_mi[m, ip] += float(np.sum(P[(ip, n)] * L[m, n]))

    def half(a_i, a_j):
        h = np.einsum("ae,be->ab", _T(a_i, a_j, T_ij, i, j, project), F_ae, optimize=True)
        h -= np.einsum("mab,m->ab",
                       np.stack([project(a_i, m) for m in range(n_occ)]),
                       F_mi[:, a_j], optimize=True)
        W1 = me_jb.copy()
        W2 = me_jb - mj_be
        WX = -mj_be.copy()
        for jp in range(n_occ):
            for n in range(n_occ):
                Pnj = project(n, jp)
                Pjn = project(jp, n)
                tss = 0.5 * (Pjn - Pjn.T)
                W1[:, :, jp, :] += np.einsum("fb,mef->meb", Pnj - 0.5 * Pjn, me_nf[:, n], optimize=True)
                W1[:, :, jp, :] -= 0.5 * np.einsum("fb,mfe->meb", Pnj, me_nf[:, n], optimize=True)
                W2[:, :, jp, :] += (
                    -np.einsum("fb,mef->meb", tss,
                               me_nf[:, n] - me_nf[:, n].transpose(0, 2, 1), optimize=True)
                    + 0.5 * np.einsum("fb,mef->meb", Pnj, me_nf[:, n], optimize=True)
                )
                WX[:, :, jp, :] += 0.5 * np.einsum("fb,mfe->meb", Pjn, me_nf[:, n], optimize=True)
        h += np.einsum("mae,meb->ab",
                       np.stack([project(a_i, m) - project(m, a_i) for m in range(n_occ)]),
                       W1[:, :, a_j, :], optimize=True)
        h += np.einsum("mae,meb->ab",
                       np.stack([project(a_i, m) for m in range(n_occ)]),
                       W2[:, :, a_j, :], optimize=True)
        h += np.einsum("mae,meb->ab",
                       np.stack([project(m, a_j) for m in range(n_occ)]),
                       WX[:, :, a_i, :], optimize=True)
        return h

    R += half(i, j) + half(j, i).T
    return R


def _T(a_i, a_j, T_ij, i, j, project):
    """This pair's own amplitudes in (a_i,a_j) orientation."""
    if (a_i, a_j) == (i, j):
        return T_ij
    return T_ij.T  # orientation (j,i)


def build_pair_cc_integrals(
    df,
    C_occ: np.ndarray,
    i: int,
    j: int,
    V_pno: np.ndarray,
) -> PairCCIntegrals:
    """Build a pair's PNO-basis DF integral primitives.

    Parameters
    ----------
    df : DensityFitting
        DF object for the orbital basis (its global RI metric is baked
        into ``mo_transform``).
    C_occ : ndarray, shape (nbf, n_occ)
        Occupied MO coefficients (canonical or localised).
    i, j : int
        Occupied orbital indices for the pair.
    V_pno : ndarray, shape (nbf, n_pno)
        AO expansion of the pair's PNOs (S-orthonormal columns).
    """
    n_pno = V_pno.shape[1]
    Ci = C_occ[:, [i]]
    Cj = C_occ[:, [j]]
    # B^P_{i a}, B^P_{j b}: (naux, 1, n_pno) -> (naux, n_pno)
    B_i = np.asarray(df.mo_transform(Ci, V_pno))[:, 0, :]
    B_j = np.asarray(df.mo_transform(Cj, V_pno))[:, 0, :]
    # B^P_{i j}: (naux, 1, 1) -> (naux,)
    B_ij = np.asarray(df.mo_transform(Ci, Cj))[:, 0, 0]
    # B^P_{a b}: (naux, n_pno, n_pno)
    B_vv = np.asarray(df.mo_transform(V_pno, V_pno))
    return PairCCIntegrals(
        i=i, j=j, n_pno=n_pno, B_i=B_i, B_j=B_j, B_ij=B_ij, B_vv=B_vv
    )
