"""Per-pair-coupled DLPNO-CCSD on the cyclic cluster (AICCM Task 2, M3b).

The *genuine* per-pair-coupled local DLPNO-CCSD, the reduced-scaling successor to
the union-PNO subspace projection of :func:`~vibeqc.periodic.ccm.dlpno_ccsd.
ccm_dlpno_ccsd` (M3). Each occupied pair ``(i,j)`` keeps **its own** PNO basis; the
closed-shell CCSD residual is evaluated in that basis and coupled to the other
pairs by projecting their amplitudes through the PNO-overlap ``S_pq = U_pᵀ U_q``
(``U`` in canonical-virtual coordinates, which are ``S^CCM``-orthonormal). This is
the CCM transcription of the molecular local solver
:func:`vibeqc.dlpno.ccsd_local_solver.run_local_dlpno_ccsd` -- same coupled loop
and the same validated closed-shell residual :func:`vibeqc.dlpno._ccsd_cs.
cs_ccsd_residual`, but on the CCM **neutral** reference (integrals density-fit from
the neutral cderi ``L``, overlap ``S^CCM``).

Identity note: the implemented Hamiltonian is a neutral fitted-torus correlation
control. The ``ccm_*`` API ancestry assigns neither Γ-CCM nor χ-CCM construction
identity to it.

**When to use vs the union (M3).** The union :func:`ccm_dlpno_ccsd` merges every
pair's PNOs into one virtual space, so on small clusters that union spans the full
virtual and the union energy is *exact* at any ``tcut_pno`` -- more accurate than
this per-pair method there. The per-pair-coupled method's advantage is **scaling**:
each pair keeps only its own (small) PNO space and couples to only its local
occupieds (``coupling_radius``), so the cost grows ~linearly with the number of
strong pairs rather than with a union space that swells with system size. Prefer
this for large clusters; the union for small/validation cells.

Correctness gate: at ``tcut_pno = tcut_mkn = 0`` and full coupling
(``coupling_radius <= 0``) each pair's PNOs span the full virtual space and every
pair couples to every occupied, so the coupled residual is the canonical CCSD
residual in a per-pair unitary-rotated virtual basis -- the energy equals the
**canonical neutral-control CCSD** on the same neutral four-center to machine ε.

Reference = neutral (the bare-1/r Madelung background shifts the occ-virt
denominators; see :mod:`vibeqc.periodic.ccm.dlpno`). Riplinger, Sandhoefer,
Hansen & Neese, J. Chem. Phys. 139, 134101 (2013) (DLPNO-CCSD).
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

import numpy as np

from ...dlpno._ccsd_cs import _blocks, cs_ccsd_residual
from ...dlpno.triples_local import local_triples_correction
from .ccsd import _spin_block_eri_phys, _triples
from .dlpno import _build_ccm_pno_pairs


class _NeutralCderiDF:
    """Minimal ``df`` shim: ``mo_transform(Ca, Cb)`` = the neutral-cderi B-tensor
    ``B[P,a,b] = Σ_μν C_a[μ] L[P,μν] C_b[ν]`` (naux-first), the only integral access
    the molecular local-(T) needs -- so it runs on the CCM neutral reference."""

    def __init__(self, L):
        self._L = np.asarray(L, dtype=float)

    def mo_transform(self, Ca, Cb):
        return np.einsum("Pmn,ma,nb->Pab", self._L,
                         np.asarray(Ca, dtype=float), np.asarray(Cb, dtype=float),
                         optimize=True)


def _spatial_to_so_amplitudes(t1, t2, nocc, nvir):
    """Lift closed-shell spatial amplitudes ``t1[i,a]`` / ``t2[i,j,a,b]`` to the
    spin-orbital ``ts[a,i]`` / ``td[a,b,i,j]`` convention of ``_ccsd_iterate``
    (SO index ``P`` -> spatial ``P//2``, spin ``P%2``; occ block then vir block)."""
    nso_o, nso_v = 2 * nocc, 2 * nvir
    so_o = np.arange(nso_o)
    so_v = np.arange(nso_v)
    io, si_o = so_o // 2, so_o % 2
    av, si_v = so_v // 2, so_v % 2
    # ts[A_v, I_o] = t1[i,a] if spins match.
    ts = np.where(si_v[:, None] == si_o[None, :], t1[io[None, :], av[:, None]], 0.0)
    # td[A,B,I,J] = <ab||ij>-style antisymmetrised spin-orbital amplitude.
    a, b = av[:, None, None, None], av[None, :, None, None]
    i, j = io[None, None, :, None], io[None, None, None, :]
    sa, sb = si_v[:, None, None, None], si_v[None, :, None, None]
    sii, sj = si_o[None, None, :, None], si_o[None, None, None, :]
    direct = (sa == sii) & (sb == sj)
    exch = (sa == sj) & (sb == sii)
    td = np.where(direct, t2[i, j, a, b], 0.0) - np.where(exch, t2[i, j, b, a], 0.0)
    return ts, td


def _coupled_triples(ccm, scf_result, L, bundle, U, T2, t1):
    """Perturbative (T) for the per-pair-coupled amplitudes.

    Reconstructs the full canonical spatial ``t1``/``t2`` from the per-pair PNO
    amplitudes (dense; small-cluster/gate path), then evaluates the spin-orbital
    (T) on the neutral MO integrals via :func:`~vibeqc.periodic.ccm.ccsd._triples`.
    At no truncation the reconstruction is exact, so ``e_t`` equals the canonical
    neutral-control CCSD(T) (T)."""
    C = np.asarray(scf_result.mo_coeffs)
    if np.iscomplexobj(C):
        C = np.real_if_close(C, tol=1000).real
    C = np.asarray(C, dtype=float)
    eps = np.asarray(scf_result.mo_energies, dtype=float)
    S = np.asarray(bundle.S, dtype=float)
    n_occ = int(ccm.supercell.n_electrons()) // 2
    nf, n_act = bundle.nf, bundle.n_act
    C_act_can = C[:, nf:n_occ]
    C_vir = C[:, n_occ:]
    nvir = C_vir.shape[1]

    # localized active-occ -> canonical active-occ unitary V[p,i] = <can_p|loc_i>.
    Vloc = C_act_can.T @ S @ bundle.C_loc                      # (n_act, n_act)

    # Reconstruct localized-occ, canonical-virtual amplitudes.
    t2_lo = np.zeros((n_act, n_act, nvir, nvir))
    for (i, j), T in T2.items():
        blk = U[(i, j)] @ T @ U[(i, j)].T
        t2_lo[i, j] = blk
        if i != j:
            t2_lo[j, i] = blk.T
    t1_lo = np.zeros((n_act, nvir))
    for i, t in t1.items():
        t1_lo[i] = U[(i, i)] @ t
    # -> canonical active-occ.
    t1_can = Vloc @ t1_lo
    t2_can = np.einsum("pi,qj,ijab->pqab", Vloc, Vloc, t2_lo, optimize=True)

    # Full canonical MO set (frozen occ carry zero amplitudes) for the (T).
    C_all = np.concatenate([C[:, :n_occ], C_vir], axis=1)      # occ then vir
    t1_full = np.zeros((n_occ, nvir))
    t2_full = np.zeros((n_occ, n_occ, nvir, nvir))
    t1_full[nf:] = t1_can
    t2_full[nf:, nf:] = t2_can

    g = np.einsum("Pmn,Prs->mnrs", L, L, optimize=True)        # neutral four-center
    mo = np.einsum("mnrs,mp,nq,rt,su->pqtu", g, C_all, C_all, C_all, C_all, optimize=True)
    spinints = _spin_block_eri_phys(mo)
    fs = np.diag(np.repeat(np.concatenate([eps[:n_occ], eps[n_occ:]]), 2))
    ts, td = _spatial_to_so_amplitudes(t1_full, t2_full, n_occ, nvir)
    return float(_triples(spinints, fs, ts, td, 2 * n_occ))

__all__ = ["CCMDLPNOCoupledCCSDResult", "ccm_dlpno_ccsd_coupled"]


@dataclass
class CCMDLPNOCoupledCCSDResult:
    """Result of :func:`ccm_dlpno_ccsd_coupled`."""

    e_hf: float
    e_ccsd_correlation: float
    e_t: float
    e_correlation: float
    e_total: float
    e_correlation_per_atom: float
    e_total_per_atom: float
    n_pairs: int
    avg_pno_per_pair: float
    avg_coupled_occ: float
    localize: str
    converged: bool
    n_iter: int
    guess_selection: object = None


def ccm_dlpno_ccsd_coupled(
    ccm,
    scf_result,
    *,
    cderi: Optional[np.ndarray] = None,
    localize: str = "pm",
    tcut_pno: float = 0.0,
    tcut_mkn: float = 0.0,
    n_frozen: int = 0,
    coupling_radius: float = -1.0,
    compute_triples: bool = False,
    triples_mode: str = "local",
    tcut_tno: float = 0.0,
    lindep: float = 1e-8,
    ke_cutoff: float = 200.0,
    max_iter: int = 128,
    conv_tol: float = 1e-9,
    diis_size: int = 8,
) -> CCMDLPNOCoupledCCSDResult:
    """Per-pair-coupled DLPNO-CCSD on the cyclic cluster (neutral reference).

    ``scf_result`` must be the converged CCM SCF on the neutral four-center
    (e.g. ``run_ccm_rhf(ccm, eri=ccm_eri_neutral(ccm))`` or
    ``run_ccm_rhf_ri_neutral(ccm, cderi=L)``). ``cderi`` is the neutral cderi
    ``L[P,μν]`` (built if omitted). ``coupling_radius <= 0`` couples every pair to
    every occupied (the exact-at-no-truncation limit); a positive Bohr radius
    restricts each pair's coupled-occupied set (locality). ``compute_triples`` adds
    the perturbative (T), via ``triples_mode``:

    * ``"local"`` (default) -- the **scalable per-triple-TNO (T)** (reuses the
      molecular :func:`~vibeqc.dlpno.triples_local.local_triples_correction` through
      the neutral-cderi ``df`` shim). Each triple's TNO domain is truncated by
      ``tcut_tno``. Exact vs canonical (T) for **canonical occupieds**
      (``localize="none"``); with localized occupieds it carries a small
      semicanonical (T0) error (Riplinger 2013) -- the reduced-scaling default.
    * ``"dense"`` -- reconstructs the full canonical amplitudes and runs the
      spin-orbital (T) on the dense neutral MO integrals; exact vs canonical (T) for
      any gauge (machine ε), but O(n**6) -- the small-cluster/validation path.
    """
    if not scf_result.converged:
        raise ValueError("coupled DLPNO-CCSD requires a converged CCM SCF reference.")

    from .neutral import ccm_neutral_cderi

    L = np.asarray(cderi if cderi is not None else
                   ccm_neutral_cderi(ccm, ke_cutoff=ke_cutoff), dtype=float)

    # Per-pair PNOs on the neutral reference (shared PAO->domain->PNO build).
    bundle = _build_ccm_pno_pairs(
        ccm, scf_result, cderi=L, localize=localize, tcut_pno=tcut_pno,
        tcut_mkn=tcut_mkn, n_frozen=n_frozen, lindep=lindep, ke_cutoff=ke_cutoff)
    pairs, S, C_loc, n_act, nf = (
        bundle.pairs, bundle.S, bundle.C_loc, bundle.n_act, bundle.nf)
    F_oo = bundle.F_oo
    f_dd = np.diag(F_oo)
    if not pairs:
        e_hf = float(scf_result.energy)
        return CCMDLPNOCoupledCCSDResult(
            e_hf, 0.0, 0.0, 0.0, e_hf, 0.0, e_hf / ccm.n_atoms, 0, 0.0, 0.0,
            bundle.localize, True, 0, guess_selection=getattr(scf_result, "guess_selection", None))

    # Canonical virtuals + neutral DF B-tensors in the canonical/localized MO basis.
    C = np.asarray(scf_result.mo_coeffs)
    if np.iscomplexobj(C):
        C = np.real_if_close(C, tol=1000).real
    C = np.asarray(C, dtype=float)
    n_occ = int(ccm.supercell.n_electrons()) // 2
    C_vir = np.ascontiguousarray(C[:, n_occ:])
    F = np.asarray(scf_result.fock, dtype=float)
    if np.iscomplexobj(F):
        F = np.real_if_close(F, tol=1000).real
    f_vv_full = C_vir.T @ F @ C_vir
    f_ov_full = C_loc.T @ F @ C_vir
    # B[P, p, q] with p in {loc-occ, canonical-vir}.
    B_ov = np.einsum("mi,Pmn,na->Pia", C_loc, L, C_vir, optimize=True)   # (naux, n_act, n_vir)
    B_vv = np.einsum("ma,Pmn,nb->Pab", C_vir, L, C_vir, optimize=True)   # (naux, n_vir, n_vir)
    B_oo = np.einsum("mi,Pmn,nj->Pij", C_loc, L, C_loc, optimize=True)   # (naux, n_act, n_act)

    # U[(i,j)] = canonical-virtual -> pair-PNO map (S^CCM-orthonormal virtuals).
    U = {k: np.ascontiguousarray(C_vir.T @ S @ p.V) for k, p in pairs.items()}
    T2 = {k: np.array(p.T, dtype=float) for k, p in pairs.items()}
    eps_pno = {k: np.asarray(p.eps, dtype=float) for k, p in pairs.items()}

    # Locality screen: couple each pair only to occupieds within coupling_radius
    # of i or j (the O(N)-pairs lever). Centroids are the position expectation of
    # the *same* localized occupieds the pairs were built from (consistent gauge),
    # and the pairwise distance is the **minimum-image** (BvK-torus) distance -- so
    # occupieds that are neighbours across a cluster boundary (e.g. the two ends of
    # a chain) are correctly coupled, not treated as far.
    _all_occ = np.arange(n_act, dtype=int)
    _occ_dist = None
    if coupling_radius > 0.0:
        from vibeqc import compute_dipole

        from .wigner_seitz import _minkowski_reduce, minimum_image

        _dip = compute_dipole(ccm.basis)
        _rops = [np.asarray(_dip.x, float), np.asarray(_dip.y, float),
                 np.asarray(_dip.z, float)]
        _cent = np.column_stack(
            [np.einsum("mi,mn,ni->i", C_loc, R, C_loc, optimize=True) for R in _rops])
        _Lc = np.asarray(ccm.cluster_vectors, dtype=float)          # supercell (torus) lattice
        _rb = _minkowski_reduce(_Lc)
        _disp = (_cent[:, None, :] - _cent[None, :, :]).reshape(-1, 3)
        _cells, _ = minimum_image(_disp, _Lc, reduced_basis=_rb)
        _mi = np.array([_disp[m] - _cells[m][0] @ _Lc for m in range(_disp.shape[0])])
        _occ_dist = np.linalg.norm(_mi, axis=1).reshape(n_act, n_act)

    def coupled_occ(i, j):
        if coupling_radius <= 0.0:
            return _all_occ
        near = (_occ_dist[i] < coupling_radius) | (_occ_dist[j] < coupling_radius)
        near[i] = near[j] = True
        return np.where(near)[0]

    # Static per-pair integral data in each pair's PNO basis over its coupled set.
    pdata = {}
    for (i, j), p in pairs.items():
        Uij = U[(i, j)]
        Lset = coupled_occ(i, j)
        B_ov_ij = np.einsum("Pmv,va->Pma", B_ov, Uij, optimize=True)     # (naux, n_act, n_pno)
        B_vv_ij = np.einsum("Puv,ua,vb->Pab", B_vv, Uij, Uij, optimize=True)
        f_ov_ij = f_ov_full @ Uij
        pdata[(i, j)] = dict(
            L=Lset,
            li=int(np.searchsorted(Lset, i)),
            lj=int(np.searchsorted(Lset, j)),
            f_oo=np.ascontiguousarray(F_oo[np.ix_(Lset, Lset)]),
            f_vv=np.ascontiguousarray(Uij.T @ f_vv_full @ Uij),
            f_ov=np.ascontiguousarray(f_ov_ij[Lset]),
            f_ovi=f_ov_ij[i],
            K=B_ov_ij[:, i, :].T @ B_ov_ij[:, j, :],
            V=_blocks(B_ov_ij[:, Lset, :], B_vv_ij, B_oo[:, Lset][:, :, Lset]),
        )

    t1 = {i: np.zeros(U[(i, i)].shape[1]) for i in range(n_act) if (i, i) in U}

    def Uof(k, l):
        return U[(k, l)] if (k, l) in U else U[(l, k)]

    def Tof(k, l):
        if (k, l) in T2:
            return T2[(k, l)]
        return T2[(l, k)].T

    def energy():
        e = 0.0
        for (i, j), T in T2.items():
            Uij = U[(i, j)]
            t1i = (Uij.T @ U[(i, i)]) @ t1[i]
            t1j = (Uij.T @ U[(j, j)]) @ t1[j]
            tau = T + np.outer(t1i, t1j)
            K = pdata[(i, j)]["K"]
            w = 1.0 if i == j else 2.0
            e += w * float(np.sum(tau * (2.0 * K - K.T)))
        return e

    amp_hist, res_hist = [], []

    def flatten():
        return np.concatenate(
            [t1[i].ravel() for i in sorted(t1)] + [T2[k].ravel() for k in sorted(T2)])

    def unflatten(vec):
        off = 0
        for i in sorted(t1):
            n = t1[i].size
            t1[i] = vec[off:off + n].copy(); off += n
        for k in sorted(T2):
            n = T2[k].size
            T2[k] = vec[off:off + n].reshape(T2[k].shape).copy(); off += n

    e_prev = energy()
    converged = False
    n_iter = 0
    for iteration in range(max_iter):
        n_iter = iteration + 1
        x_old = flatten()
        new_T2, new_t1 = {}, {}
        for (i, j) in T2:
            Uij = U[(i, j)]
            n = Uij.shape[1]
            pd = pdata[(i, j)]
            Lset, li, lj = pd["L"], pd["li"], pd["lj"]
            nL = len(Lset)
            # Gather coupled amplitudes into this pair's PNO basis.
            t2L = np.zeros((nL, nL, n, n))
            for a_, m in enumerate(Lset):
                for b_, nn in enumerate(Lset):
                    Sm = Uij.T @ Uof(m, nn)
                    t2L[a_, b_] = Sm @ Tof(m, nn) @ Sm.T
            t1L = np.zeros((nL, n))
            for a_, m in enumerate(Lset):
                if m in t1:
                    t1L[a_] = (Uij.T @ U[(m, m)]) @ t1[m]
            R1L, R2L = cs_ccsd_residual(t1L, t2L, pd["f_oo"], pd["f_vv"], pd["f_ov"], pd["V"])
            den = f_dd[i] + f_dd[j] - eps_pno[(i, j)][:, None] - eps_pno[(i, j)][None, :]
            new_T2[(i, j)] = T2[(i, j)] + R2L[li, lj] / den
            if i == j and i in t1:
                new_t1[i] = t1[i] + R1L[li] / (f_dd[i] - eps_pno[(i, i)])
        for k, T in new_T2.items():
            T2[k] = 0.5 * (T + T.T) if k[0] == k[1] else T
        for i, t in new_t1.items():
            t1[i] = t

        amp_hist.append(flatten())
        res_hist.append(amp_hist[-1] - x_old)
        if len(amp_hist) > diis_size:
            amp_hist.pop(0); res_hist.pop(0)
        nh = len(amp_hist)
        if nh >= 2:
            Bm = np.full((nh + 1, nh + 1), -1.0)
            Bm[-1, -1] = 0.0
            for a_ in range(nh):
                for b_ in range(nh):
                    Bm[a_, b_] = float(np.dot(res_hist[a_], res_hist[b_]))
            rhs = np.zeros(nh + 1); rhs[-1] = -1.0
            try:
                c = np.linalg.solve(Bm, rhs)[:nh]
                unflatten(sum(c[k] * amp_hist[k] for k in range(nh)))
            except np.linalg.LinAlgError:
                pass

        e = energy()
        r_norm = float(np.max(np.abs(res_hist[-1])))
        if iteration > 0 and abs(e - e_prev) < conv_tol and r_norm < 1e-7:
            converged = True
            break
        e_prev = e

    e_ccsd = energy()
    if not compute_triples:
        e_t = 0.0
    elif triples_mode == "local":
        # Scalable per-triple-TNO (T): reuses the molecular local-(T) via the
        # neutral-cderi df shim (== the dense (T) at tcut_tno=0 / full PNO).
        C_vir = np.ascontiguousarray(C[:, n_occ:])
        f_vv_full = C_vir.T @ F @ C_vir
        e_t = float(local_triples_correction(
            U, T2, t1, C_loc, C_vir, S, f_vv_full, f_dd, _NeutralCderiDF(L), n_act,
            tcut_tno=tcut_tno, occ_dist=_occ_dist,
            coupling_radius=(coupling_radius if coupling_radius > 0.0 else 0.0)))
    elif triples_mode == "dense":
        e_t = _coupled_triples(ccm, scf_result, L, bundle, U, T2, t1)
    else:
        raise ValueError(f"triples_mode must be 'local' or 'dense'; got {triples_mode!r}")
    e_hf = float(scf_result.energy)
    e_corr = e_ccsd + e_t
    e_tot = e_hf + e_corr
    return CCMDLPNOCoupledCCSDResult(
        e_hf=e_hf, e_ccsd_correlation=e_ccsd, e_t=e_t, e_correlation=e_corr,
        e_total=e_tot, e_correlation_per_atom=e_corr / ccm.n_atoms,
        e_total_per_atom=e_tot / ccm.n_atoms, n_pairs=len(T2),
        avg_pno_per_pair=float(np.mean([U[k].shape[1] for k in T2])),
        avg_coupled_occ=float(np.mean([len(pdata[k]["L"]) for k in T2])),
        localize=bundle.localize, converged=converged, n_iter=n_iter, guess_selection=getattr(scf_result, "guess_selection", None))
