"""DLPNO-UMP2 -- open-shell (UHF-reference) local MP2 (M1).

The open-shell analogue of :mod:`vibeqc.dlpno.mp2`. The correlation energy
is resolved into the three spin channels of unrestricted MP2 (Szabo &
Ostlund Sec.6.7, the same decomposition as :func:`vibeqc.correlation.ump2_energy`
and ``cpp/src/ump2.cpp``):

    E_aa = sum_{i<j in a} sum_ab <ij||ab> T^aa_ij,ab          (same-spin a)
    E_bb = sum_{i<j in b} ...                                 (same-spin b)
    E_ab = sum_{i in a, j in b} sum_ab (ia|jb) T^ab_ij,ab     (opposite-spin)

with same-spin amplitudes antisymmetrised, <ij||ab> = (ia|jb) - (ib|ja),
and opposite-spin Coulomb-only. The (ia|jb) come from the density-fitted
B-tensors of each spin: (ia|jb) = sum_P B^s_{ia,P} B^s'_{jb,P}, so the
mixed alpha-beta block is a single contraction of the alpha and beta
B-tensors over the auxiliary index.

**M1 (``localise="none"``) -- canonical occupieds.** Each pair gets its own
pair natural orbitals, built from the pair density and truncated by
occupation number (``tcut_pno``); same-spin pairs carry a single PNO set
(both indices live in one spin's virtual space), opposite-spin pairs carry
a separate alpha-PNO and beta-PNO set. With ``tcut_pno=0`` (no truncation)
the energy reproduces canonical UMP2 to machine precision (the exactness
gate).

**M1b (``localise="boys"``) -- localised coupled residual.** The active
alpha/beta occupieds are Foster-Boys localised and each spin channel's
coupled LMP2 residual is solved in the full virtual space (the off-diagonal
localised Fock couples each pair to its neighbours). The three channels are
independent at first order (aa couples only aa via F^a; ab couples T_ij to
T_kj/T_il via F^a/F^b; never across channels), so they are solved
separately. This reproduces canonical UMP2 to machine precision too (LMP2
with the full virtual space == canonical MP2).

**M1c -- localised PNO truncation.** ``localise="boys"`` with ``tcut_pno>0``
adds per-pair PNO truncation on top of the localised coupled residual:
each pair carries its own (truncated) PNO basis and the coupling sums
project a neighbour's amplitude into the target pair's PNO space through the
PNO overlap ``S = U_target^T U_src`` (same-spin: one set, sign-carried for
the pair antisymmetry; opposite-spin: separate alpha/beta sets, dual-sided
projection). Because the PNOs are pair-specific rotations even at
``tcut_pno=0``, the full-domain limit exercises the projections and stays
exact vs canonical UMP2 -- the projection ratchet.

**Locality (``tcut_pairs``).** A pair whose semicanonical MP2 estimate is
below ``tcut_pairs`` is dropped from the coupled solve and contributes that
cheap full-virtual estimate directly. With Boys-localised occupieds distant
pairs are weak, so the pair list becomes O(N) for extended systems: a no-op
on compact molecules (every pair strong) and lossless on separated fragments
(an OH radical + a water 8 A away screens all ~90 inter-fragment pairs for
< 0.001 kcal/mol). 0 disables (exact). PAO virtual domains (the other half
of DLPNO locality) remain a later step. The stored non-zero threshold is the
cross-route published NormalPNO coordinate and is active only for localised
modes; the canonical-occupied default path records it as inactive.

References: Riplinger & Neese, J. Chem. Phys. 138, 034106 (2013), and
Pinski & Neese, J. Chem. Phys. 150, 164102 (2019), for the local-pair and
DLPNO-MP2 threshold foundations. The unrestricted alpha/beta spin-channel
extension here is a vibe-qc implementation: it retains independent UHF
orbital and PNO spaces and therefore is not the single-spatial-orbital,
NEV-PNO open-shell coupled-cluster construction of Saitow et al. (2017).
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np


@dataclass
class DLPNOUMP2Options:
    """Options for the open-shell DLPNO-UMP2 driver.

    Attributes
    ----------
    localise : str
        "none" (default) keeps canonical UHF occupieds: the semicanonical
        amplitudes are exact without iteration, and per-pair PNOs are built
        directly from the canonical-occupied pair densities. "boys" (M1b)
        Foster-Boys localises the alpha/beta occupieds and solves the
        coupled LMP2 residual per spin channel (the off-diagonal localised
        Fock couples each pair to its neighbours). With ``tcut_pno=0`` it
        runs in the full virtual space (M1b); with ``tcut_pno>0`` it adds
        per-pair PNO truncation with cross-pair PNO projections (M1c). Both
        reproduce canonical UMP2 at full domains. "external" consumes an
        already-localized occupied gauge and runs the same coupled equations
        without applying a molecular position-operator localization.  It is
        used by periodic finite-torus callers. This is the foundation
        for locality (weak-pair screening, O(N) pair lists -- the next rung).
    n_frozen : int | None
        Number of frozen-core spatial orbitals, removed from both the
        alpha and beta occupied sets (and all pair lists). ``None`` selects
        the shared published chemical-core convention; 0 is explicitly
        all-electron.
    tcut_pno : float
        PNO occupation-number truncation. 0 disables (the exactness
        configuration that reproduces canonical UMP2 on either path).
    tcut_pairs : float
        Weak-pair screening (Ha, ``localise="boys"`` only). A pair whose
        semicanonical MP2 estimate is below ``tcut_pairs`` is dropped from
        the coupled solve and contributes that (cheap, full-virtual)
        estimate directly -- the pair list becomes O(N) for extended
        systems (distant localised pairs are weak). 0 disables (exact). It
        is a no-op while ``localise="none"``; the stored default is the
        cross-route NormalPNO project convention for localised modes.
    max_iter : int
        Maximum Jacobi macro-iterations for the coupled LMP2 residual
        (``localise="boys"``).
    conv_tol : float
        Convergence on both the correlation-energy change and the max
        residual element (``localise="boys"``).
    ss_scale, os_scale : float
        Same-spin / opposite-spin component scaling (1, 1 = plain UMP2;
        6/5, 1/3 = SCS-UMP2; 0, 1.3 = SOS). Applied to the reported
        ``e_corr`` only; the per-channel energies are unscaled.
    aux_basis : str | None
        RI fitting basis requested for the correlation integrals. ``None``
        lets :func:`vibeqc.run_job` resolve the orbital-basis default.
    """

    localise: str = "none"
    n_frozen: int | None = None
    # This is deliberately the Liakos 2015 CC-family NormalPNO subset used
    # across vibe-qc routes. Pinski et al.'s MP2-specific published setting
    # uses TCutPNO=1e-8 and remains available as an explicit custom value.
    tcut_pno: float = 3.33e-7
    # Implemented only by the Boys/external coupled routes; inactive for the
    # default canonical-occupied route, a fact reported by threshold policy.
    tcut_pairs: float = 1e-4
    max_iter: int = 200
    conv_tol: float = 1e-9
    ss_scale: float = 1.0
    os_scale: float = 1.0
    aux_basis: str | None = None


@dataclass
class DLPNOUMP2Result:
    """Result of an open-shell DLPNO-UMP2 calculation.

    ``e_corr = os_scale*e_ab + ss_scale*(e_aa + e_bb)``. The per-channel
    energies are the truncated (DLPNO) values; with ``tcut_pno=0`` they
    equal the canonical UMP2 channels.
    """

    e_hf: float = 0.0
    e_corr: float = 0.0
    e_aa: float = 0.0
    e_bb: float = 0.0
    e_ab: float = 0.0
    e_total: float = 0.0
    n_frozen: int = 0
    localise: str = "none"
    converged: bool = True
    n_screened: int = 0  # weak pairs treated at the semicanonical estimate
    n_pairs_aa: int = 0
    n_pairs_bb: int = 0
    n_pairs_ab: int = 0
    avg_pno_aa: float = 0.0
    avg_pno_bb: float = 0.0
    avg_pno_ab: float = 0.0


def _pnos(D_pair: np.ndarray, eps_vir: np.ndarray, tcut: float):
    """Pair natural orbitals for one virtual space.

    Diagonalise the (symmetric) pair density ``D_pair``, keep the natural
    orbitals with occupation above ``tcut`` (always at least one), and
    quasi-canonicalise the retained space against ``diag(eps_vir)`` so the
    MP2 denominators stay diagonal. Returns ``(U, eps_pno)`` with ``U`` the
    (n_vir x n_pno) transform and ``eps_pno`` the quasi-canonical energies.
    """
    occ, d = np.linalg.eigh(0.5 * (D_pair + D_pair.T))
    order = np.argsort(-occ)
    occ, d = occ[order], d[:, order]
    keep = occ > tcut if tcut > 0.0 else np.ones_like(occ, dtype=bool)
    if not np.any(keep):
        keep[0] = True
    d = d[:, keep]
    F_pno = d.T @ np.diag(eps_vir) @ d
    eps_pno, u = np.linalg.eigh(0.5 * (F_pno + F_pno.T))
    return d @ u, eps_pno


def _same_spin_channel(B, eps_occ, eps_vir, tcut):
    """Same-spin channel energy + PNO stats. ``B`` is (naux, n_occ, n_vir)."""
    n_occ = B.shape[1]
    e = 0.0
    npno = []
    for i in range(n_occ):
        Bi = B[:, i, :]
        for j in range(i + 1, n_occ):
            K = Bi.T @ B[:, j, :]              # (ia|jb)
            anti = K - K.T                     # <ij||ab> = (ia|jb) - (ib|ja)
            denom = eps_occ[i] + eps_occ[j] - eps_vir[:, None] - eps_vir[None, :]
            T = anti / denom
            # Pair density (PSD for antisymmetric T too) -> one PNO set.
            D_pair = T @ T.T + T.T @ T
            U, eps_pno = _pnos(D_pair, eps_vir, tcut)
            Kp = U.T @ K @ U
            antip = Kp - Kp.T
            dp = eps_occ[i] + eps_occ[j] - eps_pno[:, None] - eps_pno[None, :]
            Tp = antip / dp
            e += 0.5 * float(np.sum(antip * Tp))
            npno.append(U.shape[1])
    return e, npno


def _opp_spin_channel(Ba, eao, eva, Bb, ebo, evb, tcut):
    """Opposite-spin channel: alpha(ia) x beta(jb), separate alpha/beta PNOs."""
    e = 0.0
    npno = []
    for i in range(Ba.shape[1]):
        Bai = Ba[:, i, :]
        for j in range(Bb.shape[1]):
            K = Bai.T @ Bb[:, j, :]            # (nva, nvb) = (ia|jb)
            denom = eao[i] + ebo[j] - eva[:, None] - evb[None, :]
            T = K / denom
            Ua, ea_pno = _pnos(T @ T.T, eva, tcut)     # alpha PNOs
            Ub, eb_pno = _pnos(T.T @ T, evb, tcut)     # beta PNOs
            Kp = Ua.T @ K @ Ub
            dp = eao[i] + ebo[j] - ea_pno[:, None] - eb_pno[None, :]
            Tp = Kp / dp
            e += float(np.sum(Kp * Tp))
            npno.append(0.5 * (Ua.shape[1] + Ub.shape[1]))
    return e, npno


def _boys_localise(C_act, basis):
    from vibeqc import compute_dipole
    from vibeqc.localise import foster_boys_localise

    nbf = C_act.shape[0]
    dip = compute_dipole(basis)
    D = np.zeros((nbf, nbf, 3))
    D[:, :, 0] = np.asarray(dip.x)
    D[:, :, 1] = np.asarray(dip.y)
    D[:, :, 2] = np.asarray(dip.z)
    return foster_boys_localise(C_act, D, max_iter=200)


def _screen_pairs(all_pairs, e_sc, tcut_pairs):
    """Partition pairs by their semicanonical MP2 estimate.

    A pair whose |estimate| is below ``tcut_pairs`` is screened: dropped from
    the coupled solve, its estimate accumulated into ``e_screened`` (the cheap
    full-virtual semicanonical pair energy, accurate for a weak/distant pair).
    Returns ``(kept_pairs, screened_set, e_screened)``. tcut_pairs<=0 keeps all.
    """
    if tcut_pairs <= 0.0:
        return list(all_pairs), set(), 0.0
    screened = {p for p in all_pairs if abs(e_sc[p]) < tcut_pairs}
    kept = [p for p in all_pairs if p not in screened]
    return kept, screened, float(sum(e_sc[p] for p in screened))


def _same_spin_localized(B, F_oo, eps_vir, max_iter, tol, tcut_pairs=0.0):
    """Coupled same-spin LMP2 residual in the full virtual space.

    Localised occupieds make ``F_oo`` non-diagonal, so the first-order
    amplitudes obey the coupled residual
        R_ij = <ij||ab> + (e_a+e_b) T_ij - S_k [F_ik T_kj + F_kj T_ik]
    (the k=i / k=j diagonal terms combine with (e_a+e_b) into the Jacobi
    denominator). Same-spin amplitudes are antisymmetric in both the pair
    (T_kj = -T_jk, diagonal vanishes) and the virtual indices.
    """
    no = B.shape[1]
    all_pairs = [(i, j) for i in range(no) for j in range(i + 1, no)]
    fdd = np.diag(F_oo)
    K, T, e_sc = {}, {}, {}
    for (i, j) in all_pairs:
        anti = B[:, i, :].T @ B[:, j, :]
        anti = anti - anti.T
        K[(i, j)] = anti
        Tij = anti / (fdd[i] + fdd[j] - eps_vir[:, None] - eps_vir[None, :])
        T[(i, j)] = Tij
        e_sc[(i, j)] = 0.5 * float(np.sum(anti * Tij))
    pairs, screened, e_screened = _screen_pairs(all_pairs, e_sc, tcut_pairs)

    def getT(k, j):
        if k == j:
            return None
        key = (k, j) if k < j else (j, k)
        if key in screened:
            return None
        return T[(k, j)] if k < j else -T[(j, k)]

    e_prev, converged, used = 0.0, False, 0
    for it in range(max_iter):
        new_T, max_r = {}, 0.0
        for (i, j) in pairs:
            R = K[(i, j)] + (eps_vir[:, None] + eps_vir[None, :]) * T[(i, j)]
            for k in range(no):
                if abs(F_oo[i, k]) > 1e-14:
                    tkj = getT(k, j)
                    if tkj is not None:
                        R -= F_oo[i, k] * tkj
                if abs(F_oo[k, j]) > 1e-14:
                    tik = getT(i, k)
                    if tik is not None:
                        R -= F_oo[k, j] * tik
            max_r = max(max_r, float(np.max(np.abs(R))))
            new_T[(i, j)] = T[(i, j)] + R / (
                fdd[i] + fdd[j] - eps_vir[:, None] - eps_vir[None, :]
            )
        for key in pairs:
            t = new_T[key]
            T[key] = 0.5 * (t - t.T)  # keep antisymmetric in (a,b)
        e = sum(0.5 * float(np.sum(K[p] * T[p])) for p in pairs) + e_screened
        used = it + 1
        if it > 0 and abs(e - e_prev) < tol and max_r < tol:
            converged = True
            break
        e_prev = e
    return e, used, converged, len(screened)


def _opp_spin_localized(Ba, Fa, eva, Bb, Fb, evb, max_iter, tol, tcut_pairs=0.0):
    """Coupled opposite-spin LMP2 residual: alpha(ia) x beta(jb).

    T^ab_ij couples to T^ab_kj (k in alpha, via F^a) and T^ab_il (l in
    beta, via F^b) -- but never to the same-spin channels, so the three
    UMP2 channels are independent at first order and solved separately.
    """
    na, nb = Ba.shape[1], Bb.shape[1]
    fda, fdb = np.diag(Fa), np.diag(Fb)
    all_pairs = [(i, j) for i in range(na) for j in range(nb)]
    K, T, e_sc = {}, {}, {}
    for (i, j) in all_pairs:
        Kij = Ba[:, i, :].T @ Bb[:, j, :]
        K[(i, j)] = Kij
        Tij = Kij / (fda[i] + fdb[j] - eva[:, None] - evb[None, :])
        T[(i, j)] = Tij
        e_sc[(i, j)] = float(np.sum(Kij * Tij))
    pairs, screened, e_screened = _screen_pairs(all_pairs, e_sc, tcut_pairs)
    e_prev, converged, used = 0.0, False, 0
    for it in range(max_iter):
        new_T, max_r = {}, 0.0
        for (i, j) in pairs:
            R = K[(i, j)] + (eva[:, None] + evb[None, :]) * T[(i, j)]
            for k in range(na):
                if abs(Fa[i, k]) > 1e-14 and (k, j) not in screened:
                    R -= Fa[i, k] * T[(k, j)]
            for l in range(nb):
                if abs(Fb[j, l]) > 1e-14 and (i, l) not in screened:
                    R -= Fb[j, l] * T[(i, l)]
            max_r = max(max_r, float(np.max(np.abs(R))))
            new_T[(i, j)] = T[(i, j)] + R / (
                fda[i] + fdb[j] - eva[:, None] - evb[None, :]
            )
        for key in pairs:
            T[key] = new_T[key]
        e = sum(float(np.sum(K[p] * T[p])) for p in pairs) + e_screened
        used = it + 1
        if it > 0 and abs(e - e_prev) < tol and max_r < tol:
            converged = True
            break
        e_prev = e
    return e, used, converged, len(screened)


def _same_spin_localized_pno(B, F_oo, eps_vir, tcut, max_iter, tol, tcut_pairs=0.0):
    """Same-spin LMP2 with per-pair PNO truncation (M1c).

    Like ``_same_spin_localized`` but each pair carries its own (truncated)
    PNO basis ``U`` (n_vir x n_pno). The coupling sums project a neighbour's
    amplitude into the target pair's PNO space via the PNO overlap
    ``S = U_target^T U_src`` (small, no full-virtual expansion); the
    antisymmetry ``T_kj = -T_jk`` is carried as a sign on that projection.
    With ``tcut=0`` the PNOs are full-rank rotations, so the projections are
    exact unitaries and the result is canonical UMP2 (the projection test).
    """
    no = B.shape[1]
    all_pairs = [(i, j) for i in range(no) for j in range(i + 1, no)]
    fdd = np.diag(F_oo)
    U, Kp, epsp, T, e_sc = {}, {}, {}, {}, {}
    for (i, j) in all_pairs:
        anti = B[:, i, :].T @ B[:, j, :]
        anti = anti - anti.T
        Tsc = anti / (fdd[i] + fdd[j] - eps_vir[:, None] - eps_vir[None, :])
        # Semicanonical estimate (full virtual) for pair screening.
        e_sc[(i, j)] = 0.5 * float(np.sum(anti * Tsc))
        Uij, ep = _pnos(Tsc @ Tsc.T + Tsc.T @ Tsc, eps_vir, tcut)
        U[(i, j)], epsp[(i, j)] = Uij, ep
        Kp[(i, j)] = Uij.T @ anti @ Uij
        T[(i, j)] = Kp[(i, j)] / (fdd[i] + fdd[j] - ep[:, None] - ep[None, :])
    pairs, screened, e_screened = _screen_pairs(all_pairs, e_sc, tcut_pairs)

    def proj(k, l, tgt):
        if k == l:
            return None
        src, sign = ((k, l), 1.0) if k < l else ((l, k), -1.0)
        if src in screened:
            return None
        S = U[tgt].T @ U[src]
        return sign * (S @ T[src] @ S.T)

    e_prev, converged, used = 0.0, False, 0
    for it in range(max_iter):
        new_T, max_r = {}, 0.0
        for (i, j) in pairs:
            ep = epsp[(i, j)]
            R = Kp[(i, j)] + (ep[:, None] + ep[None, :]) * T[(i, j)]
            for k in range(no):
                if abs(F_oo[i, k]) > 1e-14:
                    t = proj(k, j, (i, j))
                    if t is not None:
                        R -= F_oo[i, k] * t
                if abs(F_oo[k, j]) > 1e-14:
                    t = proj(i, k, (i, j))
                    if t is not None:
                        R -= F_oo[k, j] * t
            max_r = max(max_r, float(np.max(np.abs(R))))
            new_T[(i, j)] = T[(i, j)] + R / (fdd[i] + fdd[j] - ep[:, None] - ep[None, :])
        for key in pairs:
            t = new_T[key]
            T[key] = 0.5 * (t - t.T)
        e = sum(0.5 * float(np.sum(Kp[p] * T[p])) for p in pairs) + e_screened
        used = it + 1
        if it > 0 and abs(e - e_prev) < tol and max_r < tol:
            converged = True
            break
        e_prev = e
    npno = [U[p].shape[1] for p in pairs]
    return e, used, converged, npno, len(screened)


def _opp_spin_localized_pno(Ba, Fa, eva, Bb, Fb, evb, tcut, max_iter, tol, tcut_pairs=0.0):
    """Opposite-spin LMP2 with per-pair PNO truncation (M1c).

    Each pair carries a separate alpha-PNO and beta-PNO set, so projecting a
    neighbour's amplitude into the target pair is *dual-sided*:
    ``T_proj = (Ua_t^T Ua_s) T_src (Ub_t^T Ub_s)^T``. With ``tcut=0`` both
    sides are full rotations and the result is canonical UMP2.
    """
    na, nb = Ba.shape[1], Bb.shape[1]
    fda, fdb = np.diag(Fa), np.diag(Fb)
    all_pairs = [(i, j) for i in range(na) for j in range(nb)]
    Ua, Ub, Kp, epa, epb, T, e_sc = {}, {}, {}, {}, {}, {}, {}
    for (i, j) in all_pairs:
        Kij = Ba[:, i, :].T @ Bb[:, j, :]
        Tsc = Kij / (fda[i] + fdb[j] - eva[:, None] - evb[None, :])
        e_sc[(i, j)] = float(np.sum(Kij * Tsc))
        Uaij, ea = _pnos(Tsc @ Tsc.T, eva, tcut)
        Ubij, eb = _pnos(Tsc.T @ Tsc, evb, tcut)
        Ua[(i, j)], Ub[(i, j)], epa[(i, j)], epb[(i, j)] = Uaij, Ubij, ea, eb
        Kp[(i, j)] = Uaij.T @ Kij @ Ubij
        T[(i, j)] = Kp[(i, j)] / (fda[i] + fdb[j] - ea[:, None] - eb[None, :])
    pairs, screened, e_screened = _screen_pairs(all_pairs, e_sc, tcut_pairs)
    e_prev, converged, used = 0.0, False, 0
    for it in range(max_iter):
        new_T, max_r = {}, 0.0
        for (i, j) in pairs:
            ea, eb = epa[(i, j)], epb[(i, j)]
            R = Kp[(i, j)] + (ea[:, None] + eb[None, :]) * T[(i, j)]
            for k in range(na):
                if abs(Fa[i, k]) > 1e-14 and (k, j) not in screened:
                    Sa = Ua[(i, j)].T @ Ua[(k, j)]
                    Sb = Ub[(i, j)].T @ Ub[(k, j)]
                    R -= Fa[i, k] * (Sa @ T[(k, j)] @ Sb.T)
            for l in range(nb):
                if abs(Fb[j, l]) > 1e-14 and (i, l) not in screened:
                    Sa = Ua[(i, j)].T @ Ua[(i, l)]
                    Sb = Ub[(i, j)].T @ Ub[(i, l)]
                    R -= Fb[j, l] * (Sa @ T[(i, l)] @ Sb.T)
            max_r = max(max_r, float(np.max(np.abs(R))))
            new_T[(i, j)] = T[(i, j)] + R / (fda[i] + fdb[j] - ea[:, None] - eb[None, :])
        for key in pairs:
            T[key] = new_T[key]
        e = sum(float(np.sum(Kp[p] * T[p])) for p in pairs) + e_screened
        used = it + 1
        if it > 0 and abs(e - e_prev) < tol and max_r < tol:
            converged = True
            break
        e_prev = e
    npno = [0.5 * (Ua[p].shape[1] + Ub[p].shape[1]) for p in pairs]
    return e, used, converged, npno, len(screened)


def run_dlpno_ump2(molecule, basis, uhf, df, options=None):
    """Run open-shell DLPNO-UMP2 on a converged UHF reference.

    Parameters
    ----------
    molecule, basis : Molecule, BasisSet
    uhf : UHF result with ``mo_coeffs_alpha/beta``, ``mo_energies_alpha/beta``,
        ``energy``, ``converged``.
    df : vibeqc.density_fitting.DensityFitting with a real RI fitting basis.
    options : DLPNOUMP2Options
    """
    if options is None:
        options = DLPNOUMP2Options()
    from vibeqc.correlation_conventions import effective_electron_count
    if options.localise not in ("none", "boys", "external"):
        raise ValueError(f"unknown localise option: {options.localise!r}")
    Ca = np.asarray(uhf.mo_coeffs_alpha)
    Cb = np.asarray(uhf.mo_coeffs_beta)
    ea = np.asarray(uhf.mo_energies_alpha)
    eb = np.asarray(uhf.mo_energies_beta)

    ne = effective_electron_count(molecule, uhf)
    two_s = molecule.multiplicity - 1
    na = (ne + two_s) // 2
    nb = (ne - two_s) // 2
    from vibeqc.correlation_conventions import resolve_frozen_core_count

    nf = resolve_frozen_core_count(molecule, options.n_frozen, reference=uhf)
    if nf < 0 or nf > nb:
        raise ValueError(f"n_frozen={nf} out of range for n_beta={nb}")

    Cao, Cav = Ca[:, nf:na], Ca[:, na:]
    Cbo, Cbv = Cb[:, nf:nb], Cb[:, nb:]
    eao, eva = ea[nf:na], ea[na:]
    ebo, evb = eb[nf:nb], eb[nb:]

    converged, n_screened = True, 0
    if options.localise in ("boys", "external"):
        # Localise the active alpha/beta occupieds and solve each spin
        # channel's coupled LMP2 residual. The off-diagonal localised Fock
        # F_oo couples each pair to its neighbours. tcut_pno=0 uses the simple
        # full-virtual solver (M1b); tcut_pno>0 the per-pair PNO solver with
        # cross-pair projections (M1c) -- both reproduce canonical UMP2 at
        # full domains (independently validated against each other).
        Fa = np.asarray(uhf.fock_alpha)
        Fb = np.asarray(uhf.fock_beta)
        if options.localise == "boys":
            Cao_l = _boys_localise(Cao, basis)
            Cbo_l = _boys_localise(Cbo, basis)
        else:
            Cao_l, Cbo_l = Cao, Cbo
        Fa_oo = Cao_l.T @ Fa @ Cao_l
        Fb_oo = Cbo_l.T @ Fb @ Cbo_l
        Ba = np.ascontiguousarray(np.asarray(df.mo_transform(Cao_l, Cav)))
        Bb = np.ascontiguousarray(np.asarray(df.mo_transform(Cbo_l, Cbv)))
        mi, tol = options.max_iter, options.conv_tol
        tp = options.tcut_pairs
        if options.tcut_pno > 0.0:
            tc = options.tcut_pno
            e_aa, _, c_aa, npno_aa, s_aa = _same_spin_localized_pno(
                Ba, Fa_oo, eva, tc, mi, tol, tp
            )
            e_bb, _, c_bb, npno_bb, s_bb = _same_spin_localized_pno(
                Bb, Fb_oo, evb, tc, mi, tol, tp
            )
            e_ab, _, c_ab, npno_ab, s_ab = _opp_spin_localized_pno(
                Ba, Fa_oo, eva, Bb, Fb_oo, evb, tc, mi, tol, tp
            )
        else:
            e_aa, _, c_aa, s_aa = _same_spin_localized(Ba, Fa_oo, eva, mi, tol, tp)
            e_bb, _, c_bb, s_bb = _same_spin_localized(Bb, Fb_oo, evb, mi, tol, tp)
            e_ab, _, c_ab, s_ab = _opp_spin_localized(
                Ba, Fa_oo, eva, Bb, Fb_oo, evb, mi, tol, tp
            )
            npno_aa = [eva.size] * (Ba.shape[1] * (Ba.shape[1] - 1) // 2)
            npno_bb = [evb.size] * (Bb.shape[1] * (Bb.shape[1] - 1) // 2)
            npno_ab = [0.5 * (eva.size + evb.size)] * (Ba.shape[1] * Bb.shape[1])
        converged = bool(c_aa and c_bb and c_ab)
        n_screened = s_aa + s_bb + s_ab
    else:
        # DF B-tensors B^s[P, i, a]; (ia|jb) = sum_P B^s B^s'. Canonical
        # occupieds -> semicanonical (one-shot) amplitudes per pair.
        Ba = np.ascontiguousarray(np.asarray(df.mo_transform(Cao, Cav)))
        Bb = np.ascontiguousarray(np.asarray(df.mo_transform(Cbo, Cbv)))
        e_aa, npno_aa = _same_spin_channel(Ba, eao, eva, options.tcut_pno)
        e_bb, npno_bb = _same_spin_channel(Bb, ebo, evb, options.tcut_pno)
        e_ab, npno_ab = _opp_spin_channel(Ba, eao, eva, Bb, ebo, evb, options.tcut_pno)

    e_corr = options.os_scale * e_ab + options.ss_scale * (e_aa + e_bb)
    res = DLPNOUMP2Result(
        e_hf=float(uhf.energy),
        e_corr=e_corr,
        e_aa=e_aa,
        e_bb=e_bb,
        e_ab=e_ab,
        e_total=float(uhf.energy) + e_corr,
        n_frozen=nf,
        localise=options.localise,
        converged=converged,
        n_screened=n_screened,
        n_pairs_aa=len(npno_aa),
        n_pairs_bb=len(npno_bb),
        n_pairs_ab=len(npno_ab),
        avg_pno_aa=float(np.mean(npno_aa)) if npno_aa else 0.0,
        avg_pno_bb=float(np.mean(npno_bb)) if npno_bb else 0.0,
        avg_pno_ab=float(np.mean(npno_ab)) if npno_ab else 0.0,
    )
    return res
