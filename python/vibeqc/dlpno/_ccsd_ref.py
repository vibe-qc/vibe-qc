"""Spin-orbital CCSD reference kernel (internal validation anchor for M3).

Implements the canonical spin-orbital CCSD equations of Stanton, Gauss,
Watts & Bartlett (J. Chem. Phys. 94, 4334 (1991)) with density-fitted
integrals, dense einsum contractions, full MO spaces -- no locality, no
truncation, no cleverness. Deliberately the most-transcribed equation
set in quantum chemistry so it can serve as an unambiguous in-house
anchor for the DLPNO-CCSD work.

Validation of the anchor itself: for two-electron systems CCSD == FCI,
so `tests/test_dlpno_ccsd.py` pins this kernel against vibe-qc's FCI
solver before anything else trusts it.

Not user-facing and not performance-relevant: O(n⁶) dense over spin
orbitals, fine for the <= ~50-spin-orbital validation molecules.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np


@dataclass
class RefCCSDResult:
    e_corr: float = 0.0
    e_hf: float = 0.0
    e_total: float = 0.0
    e_t: float = 0.0
    n_iter: int = 0
    converged: bool = False
    t1_norm: float = 0.0
    trace: list = field(default_factory=list)


@dataclass
class RefMP2Result:
    """Second-order Moller-Plesset energy on a (semicanonical) reference.

    ``e_singles`` is the off-diagonal-Fock (Brillouin-violation) term that
    is non-zero for an ROHF reference and vanishes for a canonical UHF/RHF
    reference; ``e_doubles`` is the usual pair term. ``e_corr`` is their sum.
    ``e_os`` / ``e_ss`` split the doubles into opposite-spin / same-spin
    components (the levers SCS-/SOS-MP2 and double hybrids scale).
    """

    e_corr: float = 0.0
    e_singles: float = 0.0
    e_doubles: float = 0.0
    e_os: float = 0.0
    e_ss: float = 0.0
    e_hf: float = 0.0
    e_total: float = 0.0

    @property
    def e_correlation(self) -> float:
        """Alias for :attr:`e_corr` -- matches the C++ MP2Result surface so
        the run_job output writer / `_MP2Augmented` wrapper treat both alike."""
        return self.e_corr


def _spin_orbital_eri(B_mo: np.ndarray, n_mo: int) -> np.ndarray:
    """Antisymmetrised spin-orbital integrals <pq||rs> from a DF B-tensor.

    Spin orbitals are ordered (spatial p, spin s) -> index 2p+s with
    s in {0 (a), 1 (b)}. Chemists' (pr|qs) from DF, then
    <pq||rs> = (pr|qs)d_{sp sr}d_{sq ss} - (ps|qr)d_{sp ss}d_{sq sr}.
    """
    # Spatial chemists' integrals (pq|rs) from DF.
    V = np.einsum("Ppq,Prs->pqrs", B_mo, B_mo, optimize=True)
    n_so = 2 * n_mo
    eri = np.zeros((n_so, n_so, n_so, n_so))
    p, q, r, s = np.meshgrid(
        np.arange(n_so),
        np.arange(n_so),
        np.arange(n_so),
        np.arange(n_so),
        indexing="ij",
    )
    sp, sq, sr, ss = p % 2, q % 2, r % 2, s % 2
    P, Q, R, S = p // 2, q // 2, r // 2, s // 2
    # physicists' <pq|rs> = chemists' (pr|qs)
    coul = V[P, R, Q, S] * (sp == sr) * (sq == ss)
    exch = V[P, S, Q, R] * (sp == ss) * (sq == sr)
    eri = coul - exch
    return eri


def _spin_orbital_eri_uhf(B_a: np.ndarray, B_b: np.ndarray, n_mo: int) -> np.ndarray:
    """Antisymmetrised spin-orbital integrals <pq||rs> from UHF DF B-tensors.

    Open-shell sibling of `_spin_orbital_eri`: the (p,r) and (q,s) charge
    densities are each formed from the alpha or beta MO set according to the
    spin of the index pair, so the alpha and beta spin orbitals carry distinct
    spatial integrals. Spin orbitals ordered (spatial p, spin s) -> 2p+s with
    s in {0 (a), 1 (b)}; chemists' (pr|qs)^{s1s2} = S_P B^{s1}_pr B^{s2}_qs,
    then <pq||rs> = (pr|qs)d_{spsr}d_{sqss} - (ps|qr)d_{spss}d_{sqsr}.

    For B_a == B_b this is identical to `_spin_orbital_eri(B_a, n_mo)`.
    """
    B = (np.asarray(B_a), np.asarray(B_b))
    # Spatial chemists' (ab|cd) per (spin-of-ab-pair, spin-of-cd-pair).
    Vpair = {
        (s1, s2): np.einsum("Pab,Pcd->abcd", B[s1], B[s2], optimize=True)
        for s1 in (0, 1)
        for s2 in (0, 1)
    }
    n_so = 2 * n_mo
    idx = np.arange(n_so)
    p, q, r, s = np.meshgrid(idx, idx, idx, idx, indexing="ij")
    sp, sq, sr, ss = p % 2, q % 2, r % 2, s % 2
    P, Q, R, S = p // 2, q // 2, r // 2, s // 2
    eri = np.zeros((n_so, n_so, n_so, n_so))
    for s1 in (0, 1):
        for s2 in (0, 1):
            V = Vpair[(s1, s2)]
            # <pq||rs> Coulomb (pr|qs): (p,r) spin s1, (q,s) spin s2.
            mc = (sp == s1) & (sr == s1) & (sq == s2) & (ss == s2)
            eri[mc] += V[P[mc], R[mc], Q[mc], S[mc]]
            # exchange (ps|qr): (p,s) spin s1, (q,r) spin s2.
            mx = (sp == s1) & (ss == s1) & (sq == s2) & (sr == s2)
            eri[mx] -= V[P[mx], S[mx], Q[mx], R[mx]]
    return eri


def tau_tilde_so(t1, t2):
    tt = 0.5 * (np.einsum("ia,jb->ijab", t1, t1) - np.einsum("ib,ja->ijab", t1, t1))
    return t2 + tt


def tau_full_so(t1, t2):
    tt = np.einsum("ia,jb->ijab", t1, t1) - np.einsum("ib,ja->ijab", t1, t1)
    return t2 + tt


def so_energy(f_so, eri, t1, t2, o, v):
    e = np.einsum("ia,ia->", f_so[o, v], t1)
    e += 0.25 * np.einsum("ijab,ijab->", eri[o, o, v, v], t2)
    e += 0.5 * np.einsum("ijab,ia,jb->", eri[o, o, v, v], t1, t1)
    return float(e)


def so_residuals(f_so, fock_od, eri, t1, t2, o, v, D1, D2):
    """SGB-1991 spin-orbital CCSD residuals (r1, r2) for amplitudes (t1, t2).

    Returns the true residuals (RHS - t.D form): both vanish at the CCSD
    fixed point. Valid for non-canonical orbitals (off-diagonal Fock in
    fock_od; diagonal absorbed in D1/D2).
    """
    if True:
        tt = tau_tilde_so(t1, t2)
        tf = tau_full_so(t1, t2)

        # ---- intermediates (SGB 1991 eqs. 3-8) ----
        Fae = (
            fock_od[v, v]
            - 0.5 * np.einsum("me,ma->ae", f_so[o, v], t1)
            + np.einsum("mf,mafe->ae", t1, eri[o, v, v, v])
            - 0.5 * np.einsum("mnaf,mnef->ae", tt, eri[o, o, v, v])
        )
        Fmi = (
            fock_od[o, o]
            + 0.5 * np.einsum("ie,me->mi", t1, f_so[o, v])
            + np.einsum("ne,mnie->mi", t1, eri[o, o, o, v])
            + 0.5 * np.einsum("inef,mnef->mi", tt, eri[o, o, v, v])
        )
        Fme = f_so[o, v] + np.einsum("nf,mnef->me", t1, eri[o, o, v, v])

        Wmnij = (
            eri[o, o, o, o]
            + np.einsum("je,mnie->mnij", t1, eri[o, o, o, v])
            - np.einsum("ie,mnje->mnij", t1, eri[o, o, o, v])
            + 0.25 * np.einsum("ijef,mnef->mnij", tf, eri[o, o, v, v])
        )
        Wabef = (
            eri[v, v, v, v]
            - np.einsum("mb,amef->abef", t1, eri[v, o, v, v])
            + np.einsum("ma,bmef->abef", t1, eri[v, o, v, v])
            + 0.25 * np.einsum("mnab,mnef->abef", tf, eri[o, o, v, v])
        )
        Wmbej = (
            eri[o, v, v, o]
            + np.einsum("jf,mbef->mbej", t1, eri[o, v, v, v])
            - np.einsum("nb,mnej->mbej", t1, eri[o, o, v, o])
            - np.einsum(
                "jnfb,mnef->mbej",
                0.5 * t2 + np.einsum("jf,nb->jnfb", t1, t1),
                eri[o, o, v, v],
            )
        )

        # ---- T1 residual (SGB eq. 1, moved-to-RHS form) ----
        r1 = (
            f_so[o, v]
            + np.einsum("ie,ae->ia", t1, Fae)
            - np.einsum("ma,mi->ia", t1, Fmi)
            + np.einsum("imae,me->ia", t2, Fme)
            - np.einsum("nf,naif->ia", t1, eri[o, v, o, v])
            - 0.5 * np.einsum("imef,maef->ia", t2, eri[o, v, v, v])
            - 0.5 * np.einsum("mnae,nmei->ia", t2, eri[o, o, v, o])
            - t1 * D1
        )

        # ---- T2 residual (SGB eq. 2, moved-to-RHS form) ----
        Fae_h = Fae - 0.5 * np.einsum("mb,me->be", t1, Fme)
        Fmi_h = Fmi + 0.5 * np.einsum("je,me->mj", t1, Fme)
        r2 = eri[o, o, v, v].copy()
        r2 += np.einsum("ijae,be->ijab", t2, Fae_h)
        r2 -= np.einsum("ijbe,ae->ijab", t2, Fae_h)
        r2 -= np.einsum("imab,mj->ijab", t2, Fmi_h)
        r2 += np.einsum("jmab,mi->ijab", t2, Fmi_h)
        r2 += 0.5 * np.einsum("mnab,mnij->ijab", tf, Wmnij)
        r2 += 0.5 * np.einsum("ijef,abef->ijab", tf, Wabef)
        ring = np.einsum("imae,mbej->ijab", t2, Wmbej)
        ring -= np.einsum("ie,ma,mbej->ijab", t1, t1, eri[o, v, v, o])
        r2 += ring
        r2 -= np.einsum("jmae,mbei->ijab", t2, Wmbej) - np.einsum(
            "je,ma,mbei->ijab", t1, t1, eri[o, v, v, o]
        )
        r2 -= np.einsum("imbe,maej->ijab", t2, Wmbej) - np.einsum(
            "ie,mb,maej->ijab", t1, t1, eri[o, v, v, o]
        )
        r2 += np.einsum("jmbe,maei->ijab", t2, Wmbej) - np.einsum(
            "je,mb,maei->ijab", t1, t1, eri[o, v, v, o]
        )
        r2 += np.einsum("ie,abej->ijab", t1, eri[v, v, v, o])
        r2 -= np.einsum("je,abei->ijab", t1, eri[v, v, v, o])
        r2 -= np.einsum("ma,mbij->ijab", t1, eri[o, v, o, o])
        r2 += np.einsum("mb,maij->ijab", t1, eri[o, v, o, o])
        r2 -= t2 * D2
        return r1, r2


def _iterate_so_ccsd(
    f_so: np.ndarray,
    eri: np.ndarray,
    no: int,
    nv: int,
    e_hf: float,
    *,
    max_iter: int = 100,
    conv_tol: float = 1e-10,
    diis_size: int = 6,
    compute_triples: bool = False,
) -> RefCCSDResult:
    """Spin-orbital CCSD(+(T)) iteration on prepared spin-orbital tensors.

    The reference-agnostic core shared by the closed-shell `run_ref_ccsd`
    and unrestricted `run_ref_uccsd` drivers: each builds the spin-orbital
    Fock ``f_so`` and antisymmetrised integrals ``eri`` (occupied block
    first, then virtual) from its reference and hands them here. Valid for
    non-canonical orbitals: the off-diagonal Fock is retained in
    ``fock_od`` and the diagonal is absorbed into the MP denominators.
    """
    o = slice(0, no)
    v = slice(no, no + nv)

    fock_od = f_so.copy()
    np.fill_diagonal(fock_od, 0.0)
    eps = np.diag(f_so)
    D1 = eps[o, None] - eps[None, v]
    D2 = (
        eps[o][:, None, None, None]
        + eps[o][None, :, None, None]
        - eps[v][None, None, :, None]
        - eps[v][None, None, None, :]
    )

    t1 = f_so[o, v] / D1
    t2 = eri[o, o, v, v] / D2

    def energy(t1, t2):
        return so_energy(f_so, eri, t1, t2, o, v)

    # DIIS storage (amplitude + residual history of the joint (t1, t2) vector)
    ampl_hist: list = []
    r_hist: list = []

    result = RefCCSDResult(e_hf=float(e_hf))
    e_prev = energy(t1, t2)

    for iteration in range(max_iter):
        r1, r2 = so_residuals(f_so, fock_od, eri, t1, t2, o, v, D1, D2)

        # ---- Jacobi + DIIS update ----
        t1_new = t1 + r1 / D1
        t2_new = t2 + r2 / D2

        ampl_hist.append(np.concatenate([t1_new.ravel(), t2_new.ravel()]))
        r_hist.append(np.concatenate([r1.ravel(), r2.ravel()]))
        if len(ampl_hist) > diis_size:
            ampl_hist.pop(0)
            r_hist.pop(0)
        n = len(ampl_hist)
        if n >= 2:
            Bm = np.empty((n + 1, n + 1))
            Bm[-1, :] = -1.0
            Bm[:, -1] = -1.0
            Bm[-1, -1] = 0.0
            for a_ in range(n):
                for b_ in range(n):
                    Bm[a_, b_] = float(np.dot(r_hist[a_], r_hist[b_]))
            rhs = np.zeros(n + 1)
            rhs[-1] = -1.0
            try:
                c = np.linalg.solve(Bm, rhs)[:n]
                flat = sum(c[k] * ampl_hist[k] for k in range(n))
                t1_new = flat[: t1.size].reshape(t1.shape)
                t2_new = flat[t1.size :].reshape(t2.shape)
            except np.linalg.LinAlgError:
                pass

        t1, t2 = t1_new, t2_new
        e_corr = energy(t1, t2)
        r_norm = float(np.linalg.norm(r1)) + float(np.linalg.norm(r2))
        result.trace.append({"iter": iteration + 1, "e_corr": e_corr, "r": r_norm})
        if abs(e_corr - e_prev) < conv_tol and r_norm < 1e-7:
            result.converged = True
            result.n_iter = iteration + 1
            break
        e_prev = e_corr

    result.e_corr = energy(t1, t2)
    result.t1_norm = float(np.linalg.norm(t1))
    if compute_triples:
        result.e_t = so_triples_correction(eps, eri, t1, t2, o, v)
    result.e_total = result.e_hf + result.e_corr + result.e_t
    return result


def run_ref_ccsd(
    f_mo: np.ndarray,
    B_mo: np.ndarray,
    n_occ: int,
    e_hf: float = 0.0,
    *,
    max_iter: int = 100,
    conv_tol: float = 1e-10,
    diis_size: int = 6,
    compute_triples: bool = False,
) -> RefCCSDResult:
    """Spin-orbital CCSD on a closed-shell reference.

    Parameters
    ----------
    f_mo : ndarray, shape (n_mo, n_mo)
        MO-basis Fock matrix (canonical or localised -- any orbitals with
        a converged closed-shell density; f_ov need not vanish).
    B_mo : ndarray, shape (n_aux, n_mo, n_mo)
        DF B-tensor in the same MO basis: (pq|rs) = S_P B_pq B_rs.
    n_occ : int
        Number of doubly occupied spatial orbitals.
    e_hf : float
        Reference energy (passed through to the result).
    """
    n_mo = f_mo.shape[0]
    no = 2 * n_occ
    nv = 2 * (n_mo - n_occ)

    # Spin-orbital Fock: f_so[2p+s, 2q+s'] = f[p,q] d_ss'
    f_so = np.zeros((2 * n_mo, 2 * n_mo))
    f_so[0::2, 0::2] = f_mo
    f_so[1::2, 1::2] = f_mo

    eri = _spin_orbital_eri(np.asarray(B_mo), n_mo)

    # occupied spin orbitals first: order (2i, 2i+1) for i < n_occ, then virtuals
    occ = np.concatenate([(2 * np.arange(n_occ)), (2 * np.arange(n_occ) + 1)])
    vir = np.concatenate(
        [(2 * np.arange(n_occ, n_mo)), (2 * np.arange(n_occ, n_mo) + 1)]
    )
    order = np.concatenate([np.sort(occ), np.sort(vir)])
    f_so = f_so[np.ix_(order, order)]
    eri = eri[np.ix_(order, order, order, order)]

    return _iterate_so_ccsd(
        f_so,
        eri,
        no,
        nv,
        e_hf,
        max_iter=max_iter,
        conv_tol=conv_tol,
        diis_size=diis_size,
        compute_triples=compute_triples,
    )


def run_ref_uccsd(
    f_mo_a: np.ndarray,
    f_mo_b: np.ndarray,
    B_mo_a: np.ndarray,
    B_mo_b: np.ndarray,
    n_occ_a: int,
    n_occ_b: int,
    e_hf: float = 0.0,
    *,
    max_iter: int = 100,
    conv_tol: float = 1e-10,
    diis_size: int = 6,
    compute_triples: bool = False,
) -> RefCCSDResult:
    """Spin-orbital CCSD on an unrestricted (UHF) reference.

    The open-shell sibling of `run_ref_ccsd`: identical spin-orbital
    equations (SGWB-1991 residuals + Raghavachari-1989 (T)), differing only
    in that the alpha and beta spin orbitals carry distinct spatial MOs. The
    spin-orbital form is reference-agnostic, so this is correct-by-
    construction for any reference whose alpha/beta Fock and DF tensors are
    supplied: UHF (canonical, f_ov = 0) and ROHF/semicanonical (off-diagonal
    f_ov retained in fock_od) alike. It is the in-repo anchor for the C++
    unrestricted kernel (cpp/src/uccsd.cpp), mirroring run_ref_ccsd's role
    for cpp/src/ccsd.cpp.

    Parameters
    ----------
    f_mo_a, f_mo_b : ndarray, shape (n_mo, n_mo)
        Alpha / beta MO-basis Fock matrices.
    B_mo_a, B_mo_b : ndarray, shape (n_aux, n_mo, n_mo)
        Alpha / beta DF B-tensors in the respective MO bases:
        (pq|rs)^{s1s2} = S_P B^{s1}_pq B^{s2}_rs.
    n_occ_a, n_occ_b : int
        Number of occupied alpha / beta spatial orbitals.
    e_hf : float
        Reference energy (passed through to the result).
    """
    n_mo = f_mo_a.shape[0]
    no = n_occ_a + n_occ_b
    nv = (n_mo - n_occ_a) + (n_mo - n_occ_b)

    # Spin-orbital Fock: alpha orbitals at even indices, beta at odd.
    f_so = np.zeros((2 * n_mo, 2 * n_mo))
    f_so[0::2, 0::2] = f_mo_a
    f_so[1::2, 1::2] = f_mo_b

    eri = _spin_orbital_eri_uhf(np.asarray(B_mo_a), np.asarray(B_mo_b), n_mo)

    # occupied spin orbitals first: alpha occ = even < 2*n_occ_a,
    # beta occ = odd < 2*n_occ_b+1; the complement is virtual.
    occ = np.concatenate([2 * np.arange(n_occ_a), 2 * np.arange(n_occ_b) + 1])
    vir = np.concatenate(
        [2 * np.arange(n_occ_a, n_mo), 2 * np.arange(n_occ_b, n_mo) + 1]
    )
    order = np.concatenate([np.sort(occ), np.sort(vir)])
    f_so = f_so[np.ix_(order, order)]
    eri = eri[np.ix_(order, order, order, order)]

    return _iterate_so_ccsd(
        f_so,
        eri,
        no,
        nv,
        e_hf,
        max_iter=max_iter,
        conv_tol=conv_tol,
        diis_size=diis_size,
        compute_triples=compute_triples,
    )


def run_ref_rohf_ccsd(
    f_mo_a: np.ndarray,
    f_mo_b: np.ndarray,
    B_mo_a: np.ndarray,
    B_mo_b: np.ndarray,
    n_occ_a: int,
    n_occ_b: int,
    e_hf: float = 0.0,
    *,
    max_iter: int = 100,
    conv_tol: float = 1e-10,
    diis_size: int = 6,
    compute_triples: bool = False,
) -> RefCCSDResult:
    """ROHF-reference spin-orbital CCSD -- convenience wrapper around :func:`run_ref_uccsd`.

    For ROHF, the alpha and beta spatial orbitals are identical
    (``C_alpha == C_beta``), so ``B_mo_a == B_mo_b``.  The per-spin Fock
    matrices differ (``F_alpha != F_beta``, different exchange terms), and
    ``n_occ_a > n_occ_b`` for an open-shell system.  This function is a
    discoverability alias that delegates to the reference-agnostic
    spin-orbital kernel.
    """
    return run_ref_uccsd(
        f_mo_a,
        f_mo_b,
        B_mo_a,
        B_mo_b,
        n_occ_a,
        n_occ_b,
        e_hf,
        max_iter=max_iter,
        conv_tol=conv_tol,
        diis_size=diis_size,
        compute_triples=compute_triples,
    )


def _semicanonicalize_spin(f_mo: np.ndarray, B_mo: np.ndarray, n_occ: int):
    """Diagonalise the occ-occ and vir-vir Fock blocks (semicanonicalisation).

    Returns ``(f_sc, B_sc, eps)``: the MO-basis Fock rotated so its occupied
    and virtual diagonal blocks are diagonal (the off-diagonal occ-vir block
    ``f_ov`` is *retained* -- it is what makes the MP2 singles term non-zero
    for an ROHF reference), the DF B-tensor rotated into the same basis, and
    the semicanonical orbital energies ``eps = [eps_occ, eps_vir]``.

    The block-diagonal rotation never mixes occupied with virtual, so the
    occupied/virtual partition (and therefore the reference determinant) is
    unchanged.
    """
    f_mo = np.asarray(f_mo, dtype=float)
    B_mo = np.asarray(B_mo, dtype=float)
    n_mo = f_mo.shape[0]
    U = np.zeros((n_mo, n_mo))
    eps = np.zeros(n_mo)
    eo, Vo = np.linalg.eigh(f_mo[:n_occ, :n_occ])
    U[:n_occ, :n_occ] = Vo
    eps[:n_occ] = eo
    ev, Vv = np.linalg.eigh(f_mo[n_occ:, n_occ:])
    U[n_occ:, n_occ:] = Vv
    eps[n_occ:] = ev
    f_sc = U.T @ f_mo @ U
    B_sc = np.einsum("Ppq,pi,qj->Pij", B_mo, U, U, optimize=True)
    return f_sc, B_sc, eps


def run_ref_ump2(
    f_mo_a: np.ndarray,
    f_mo_b: np.ndarray,
    B_mo_a: np.ndarray,
    B_mo_b: np.ndarray,
    n_occ_a: int,
    n_occ_b: int,
    e_hf: float = 0.0,
) -> RefMP2Result:
    """Semicanonical second-order Moller-Plesset on an unrestricted reference.

    The MP2 sibling of :func:`run_ref_uccsd`: a reference-agnostic
    spin-orbital MP2 correct-by-construction for any reference whose per-spin
    MO Fock and DF tensors are supplied -- UHF (canonical, ``f_ov = 0`` ->
    singles vanish, standard UMP2) and ROHF (semicanonical, ``f_ov != 0`` ->
    singles contribute) alike. It is the in-repo anchor for an eventual C++
    ROHF-MP2, mirroring :func:`run_ref_uccsd`'s role for the CC kernel.

    Semicanonical ROHF-MP2 (a.k.a. RMP2 in the GAUSSIAN/MOLPRO sense):
    Knowles, Andrews, Amos, Handy & Pople, *Chem. Phys. Lett.* **186**, 130
    (1991), doi:10.1016/S0009-2614(91)85118-G. The orbitals are first
    semicanonicalised (occ-occ and vir-vir Fock blocks diagonalised per
    spin); the correlation energy is then

        E2 = S_ia |f_ia|^2 / (e_i - e_a)                          (singles)
           + 1/4 S_ijab |<ij‖ab>|^2 / (e_i + e_j - e_a - e_b)      (doubles)

    over spin orbitals with e the semicanonical orbital energies. For
    canonical orbitals f_ia = 0 (Brillouin) and the singles term vanishes,
    recovering ordinary UMP2.

    Parameters
    ----------
    f_mo_a, f_mo_b : ndarray, shape (n_mo, n_mo)
        Alpha / beta MO-basis Fock matrices (need not be diagonal).
    B_mo_a, B_mo_b : ndarray, shape (n_aux, n_mo, n_mo)
        Alpha / beta DF B-tensors: (pq|rs)^{s1s2} = S_P B^{s1}_pq B^{s2}_rs.
        For ROHF the alpha/beta spatial orbitals coincide so B_mo_a == B_mo_b.
    n_occ_a, n_occ_b : int
        Number of occupied alpha / beta spatial orbitals.
    e_hf : float
        Reference energy (passed through to the result).
    """
    fa, Ba, eps_a = _semicanonicalize_spin(f_mo_a, B_mo_a, n_occ_a)
    fb, Bb, eps_b = _semicanonicalize_spin(f_mo_b, B_mo_b, n_occ_b)
    n_mo = fa.shape[0]
    no = n_occ_a + n_occ_b
    nv = (n_mo - n_occ_a) + (n_mo - n_occ_b)

    # Spin-orbital Fock + antisymmetrised integrals, alpha at even indices,
    # beta at odd -- the exact setup run_ref_uccsd hands to the CC iterator.
    f_so = np.zeros((2 * n_mo, 2 * n_mo))
    f_so[0::2, 0::2] = fa
    f_so[1::2, 1::2] = fb
    eri = _spin_orbital_eri_uhf(Ba, Bb, n_mo)

    occ = np.concatenate([2 * np.arange(n_occ_a), 2 * np.arange(n_occ_b) + 1])
    vir = np.concatenate(
        [2 * np.arange(n_occ_a, n_mo), 2 * np.arange(n_occ_b, n_mo) + 1]
    )
    order = np.concatenate([np.sort(occ), np.sort(vir)])
    f_so = f_so[np.ix_(order, order)]
    eri = eri[np.ix_(order, order, order, order)]
    spin = order % 2  # 0 = alpha, 1 = beta, per reordered spin orbital

    o = slice(0, no)
    v = slice(no, no + nv)
    eps = np.diag(f_so)  # diagonal in the o-o / v-v blocks after semicanon
    d1 = eps[o, None] - eps[None, v]
    d2 = (
        eps[o][:, None, None, None]
        + eps[o][None, :, None, None]
        - eps[v][None, None, :, None]
        - eps[v][None, None, None, :]
    )

    f_ov = f_so[o, v]
    eri_oovv = eri[o, o, v, v]
    t1 = f_ov / d1
    t2 = eri_oovv / d2

    e_singles = float(np.einsum("ia,ia->", f_ov, t1))
    # Per-pair doubles, split by the spin relationship of the (i, j) pair.
    dbl_ij = 0.25 * np.einsum("ijab,ijab->ij", eri_oovv, t2)
    same_spin = spin[o][:, None] == spin[o][None, :]
    e_doubles = float(dbl_ij.sum())
    e_ss = float(dbl_ij[same_spin].sum())
    e_os = float(dbl_ij[~same_spin].sum())

    e_corr = e_singles + e_doubles
    return RefMP2Result(
        e_corr=e_corr,
        e_singles=e_singles,
        e_doubles=e_doubles,
        e_os=e_os,
        e_ss=e_ss,
        e_hf=float(e_hf),
        e_total=float(e_hf) + e_corr,
    )


def run_ref_rohf_mp2(
    f_mo_a: np.ndarray,
    f_mo_b: np.ndarray,
    B_mo_a: np.ndarray,
    B_mo_b: np.ndarray,
    n_occ_a: int,
    n_occ_b: int,
    e_hf: float = 0.0,
) -> RefMP2Result:
    """ROHF-reference semicanonical MP2 -- discoverability alias for
    :func:`run_ref_ump2`.

    For ROHF the alpha/beta spatial orbitals are identical (``B_mo_a ==
    B_mo_b``); the per-spin Fock matrices differ and ``n_occ_a > n_occ_b``
    for an open-shell system. The off-diagonal ``f_ov`` blocks (non-zero for
    ROHF) drive the singles term.
    """
    return run_ref_ump2(
        f_mo_a, f_mo_b, B_mo_a, B_mo_b, n_occ_a, n_occ_b, e_hf
    )


def so_triples_correction(eps_so, eri, t1, t2, o, v):
    """Perturbative (T) correction (Raghavachari et al. 1989), spin-orbital.

    Standard connected/disconnected form (e.g. Crawford & Schaefer):

        W^c_{ijk}^{abc} = P(i/jk) P(a/bc) [ S_e t_{jk}^{ae} <ei||bc>
                                           - S_m t_{im}^{bc} <ma||jk> ]
        W^d_{ijk}^{abc} = P(i/jk) P(a/bc) [ t_i^a <jk||bc> ]
        E_(T) = (1/36) S_{ijkabc} W^c (W^c + W^d) / D_{ijk}^{abc}

    Evaluated triple-by-triple over distinct (i<j<k) with the full
    (i,j,k) permutation sum restored by symmetry -- the t3 tensor is
    never materialised. Exact full-triples-space (T); locality
    truncations are a later, separate concern.
    """
    no = o.stop - o.start
    eps_o = eps_so[o]
    eps_v = eps_so[v]

    eri_vovv = eri[v, o, v, v]  # <ei||bc> -> [e,i,b,c]
    eri_ovoo = eri[o, v, o, o]  # <ma||jk> -> [m,a,j,k]
    eri_oovv = eri[o, o, v, v]

    e_t = 0.0
    for i in range(no):
        for j in range(i + 1, no):
            for k in range(j + 1, no):
                e_t += so_triple_energy(
                    i, j, k, t1, t2, eri_vovv, eri_ovoo, eri_oovv, eps_o, eps_v
                )
    return e_t


def so_triple_energy(i, j, k, t1, t2, eri_vovv, eri_ovoo, eri_oovv, eps_o, eps_v):
    """(T) energy contribution of one distinct occupied triple (i<j<k).

    The reusable per-triple kernel underneath `so_triples_correction`: the
    virtual indices (a,b,c,e) span whatever space the integral blocks /
    `eps_v` are given in -- the *full* virtual space for the exact (T), or a
    triple's local (TNO) domain for the reduced-scaling DLPNO-(T). The
    occupied sum `m` in the connected term runs over the occupied dimension
    of `t2`/`eri_ovoo`. Spin-orbital; no t3 tensor is materialised.
    """

    # X(ijk)abc = S_e t_{jk}^{ae} <ei||bc> - S_m t_{im}^{bc} <ma||jk>
    def X(i_, j_, k_):
        term1 = np.einsum("ae,ebc->abc", t2[j_, k_], eri_vovv[:, i_])
        term2 = np.einsum("mbc,ma->abc", t2[i_], eri_ovoo[:, :, j_, k_])
        return term1 - term2

    Wc = X(i, j, k) - X(j, i, k) - X(k, j, i)  # P(i/jk) on occupied
    Wc = Wc - Wc.transpose(1, 0, 2) - Wc.transpose(2, 1, 0)  # P(a/bc) on virtual

    def Y(i_, j_, k_):
        return np.einsum("a,bc->abc", t1[i_], eri_oovv[j_, k_])

    Wd = Y(i, j, k) - Y(j, i, k) - Y(k, j, i)
    Wd = Wd - Wd.transpose(1, 0, 2) - Wd.transpose(2, 1, 0)

    D = (
        eps_o[i]
        + eps_o[j]
        + eps_o[k]
        - eps_v[:, None, None]
        - eps_v[None, :, None]
        - eps_v[None, None, :]
    )
    # (1/36) over the full ijk sum = (1/6) per distinct triple; the abc sum
    # stays full (the 1/6 absorbs only the ijk permutations).
    return float(np.sum(Wc * (Wc + Wd) / D)) / 6.0
