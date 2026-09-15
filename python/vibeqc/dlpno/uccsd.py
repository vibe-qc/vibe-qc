"""DLPNO-UCCSD(T) pilot -- spin-orbital subspace-projected open-shell CC (M2).

The open-shell sibling of :func:`vibeqc.dlpno.ccsd.run_dlpno_ccsd_pilot`. It
is the correctness oracle for open-shell local coupled cluster: physically
exact DLPNO-UCCSD(T) on validation-scale radicals, built -- like its
closed-shell cousin -- to eliminate equation-transcription risk.

The residuals are evaluated by the FCI-anchored spin-orbital engine
(:mod:`vibeqc.dlpno._ccsd_ref`, Stanton-Gauss-Watts-Bartlett 1991, the same
kernel that anchors ``cpp/src/uccsd.cpp``), while the *ansatz* is the DLPNO
one: T2 amplitudes live in per-pair PNO spaces and T1 in the full
spin-orbital singles space. Because the reference is unrestricted, the
spin-orbital integrals already encode spin -- there is **one** virtual space
and **no** explicit aa/bb/ab channel split (far simpler than re-deriving the
spin-adapted open-shell residuals). Saitow, Becker, Riplinger, Valeev & Neese,
J. Chem. Phys. 146, 164105 (2017), is cited as open-shell DLPNO-CCSD
background and motivation. This pilot instead starts from independent UHF
alpha/beta orbital spaces and does not implement or claim that paper's single
spatial-orbital NEV-PNO construction or its SOMO-pair threshold scale.

Each macro-iteration expands the constrained amplitudes to the full
spin-orbital space, evaluates the exact SGWB residuals, and projects them
back into the retained subspaces (the stationarity conditions of
subspace-constrained UCCSD). With untruncated PNO spaces every projection is
a unitary rotation, so the pilot is canonical UCCSD by construction (the
exactness/projection ratchet); with truncated spaces it is the textbook
DLPNO variational-subspace ansatz.

This is an O(N⁶) full-space-contraction *pilot*, capped at ``max_nbf`` basis
functions -- the correctness oracle the reduced-scaling per-pair engine will
be validated against, exactly as the closed-shell ``run_dlpno_ccsd_pilot``
relates to the local solver. Remaining production gates live in
``handovers/HANDOVER_GATED_ITEMS.md``.

Diagonal spin-orbital pairs (P, P) vanish by the Pauli antisymmetry
``<PP||AB> = 0``, so there is no diagonal-pair PNO set; the singles use the
full (untruncated) spin-orbital virtual space (it is small at pilot scale --
production would use orbital domains).
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

from ._ccsd_ref import (
    _spin_orbital_eri_uhf,
    so_energy,
    so_residuals,
    so_triples_correction,
)

# The C++ on-the-fly (T) kernel is ~20-50x faster than the pure-Python
# so_triples_correction for the spin-orbital case.  It expects t1/t2 in
# virtual-first layout: t1(nv, no), t2(nv, nv, no, no).
try:
    from .._vibeqc_core import ccsd_t_triples as _cpp_ccsd_t_triples

    _HAVE_CPP_TRIPLES = True
except ImportError:
    _cpp_ccsd_t_triples = None
    _HAVE_CPP_TRIPLES = False


@dataclass
class DLPNOUCCSDPilotOptions:
    """Options for the open-shell DLPNO-UCCSD correctness pilot.

    Mirrors :class:`vibeqc.dlpno.ccsd.DLPNOCCSDPilotOptions`. No weak-pair or
    distant-pair screening yet (pairs are cheap at pilot scale).

    Attributes
    ----------
    localise : str
        "none" (default) keeps the canonical UHF occupieds; the spin-orbital
        Fock is then block-diagonal in the occupied space and the pilot is the
        exact-(T) oracle. "boys" Foster-Boys localises the active alpha/beta
        occupieds (the off-diagonal localised Fock is retained in the residual
        and the (T) rotates the occupieds back to canonical first). Both
        reproduce canonical UCCSD at full domains. "external" consumes an
        already-localized occupied gauge without applying the molecular Boys
        operator and is intended for periodic finite-torus callers.
    n_frozen : int | None
        Frozen-core spatial orbitals removed from both spin occupied sets.
        ``None`` selects the shared published chemical-core convention; 0 is
        explicitly all-electron.
    tcut_pno : float
        PNO occupation-number truncation. 0 disables (the exactness
        configuration that reproduces canonical UCCSD).
    max_iter, conv_tol_energy, conv_tol_residual, diis_size :
        UCCSD macro-iteration controls.
    compute_triples : bool
        Evaluate the perturbative (T) correction on the converged amplitudes.
    max_nbf : int
        Pilot-cost guard: the expand/project oracle is O(N⁶) in the full
        spin-orbital space. Raise rather than silently launch huge jobs.
    aux_basis : str | None
        RI fitting basis for the correlation integrals. None lets the runner
        auto-resolve it from the orbital basis; set it explicitly for orbital
        bases that ship no default RI aux.
    """

    localise: str = "none"
    n_frozen: int | None = None
    # This pilot consumes only TCutPNO from the Liakos 2015 NormalPNO
    # cross-route policy. TCutPairs and TCutMKN are absent because the pilot
    # has no such algorithms; Pinski's MP2-specific preset does not govern CC.
    tcut_pno: float = 3.33e-7
    max_iter: int = 80
    conv_tol_energy: float = 1e-9
    conv_tol_residual: float = 1e-8
    diis_size: int = 6
    compute_triples: bool = False
    max_nbf: int = 64
    # RI fitting basis for the correlation integrals. None -> the runner
    # auto-resolves it from the orbital basis (density_fitting.
    # default_aux_basis_for(basis, kind="ri")). Set explicitly (e.g.
    # "cc-pvdz-ri") for orbital bases that ship no default RI aux.
    aux_basis: str | None = None


@dataclass
class DLPNOUCCSDPilotResult:
    e_hf: float = 0.0
    e_corr: float = 0.0
    e_t: float = 0.0
    e_total: float = 0.0
    n_iter: int = 0
    converged: bool = False
    n_pairs: int = 0
    n_frozen: int = 0
    localise: str = "none"
    t1_norm: float = 0.0
    avg_pno: float = 0.0
    pno_per_pair: dict = field(default_factory=dict)
    trace: list = field(default_factory=list)
    triples_executed: bool = False


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


def run_dlpno_uccsd_pilot(
    molecule,
    basis,
    uhf,
    df,
    options: DLPNOUCCSDPilotOptions | None = None,
) -> DLPNOUCCSDPilotResult:
    """Run the subspace-projected open-shell DLPNO-UCCSD(T) pilot.

    Parameters mirror :func:`vibeqc.dlpno.ump2.run_dlpno_ump2`: a converged
    UHF reference (``mo_coeffs_alpha/beta``, ``mo_energies_alpha/beta``,
    ``fock_alpha/beta``, ``energy``) and a python ``DensityFitting`` with a
    real RI fitting basis.
    """
    from vibeqc._vibeqc_core import compute_overlap
    from vibeqc.correlation_conventions import effective_electron_count

    if options is None:
        options = DLPNOUCCSDPilotOptions()
    if options.localise not in ("none", "boys", "external"):
        raise ValueError(f"unknown localise option: {options.localise!r}")

    Ca = np.asarray(uhf.mo_coeffs_alpha)
    Cb = np.asarray(uhf.mo_coeffs_beta)
    Fa = np.asarray(uhf.fock_alpha)
    Fb = np.asarray(uhf.fock_beta)
    S_ao = np.asarray(compute_overlap(basis))
    nbf = Ca.shape[0]
    if nbf > options.max_nbf:
        raise ValueError(
            f"DLPNO-UCCSD pilot is an O(N^6) correctness oracle, capped at "
            f"max_nbf={options.max_nbf} basis functions (got {nbf}). The "
            f"reduced-scaling engine is later DLPNO work; see "
            f"docs/tutorial/dlpno_local_correlation.md. Raise "
            f"DLPNOUCCSDPilotOptions.max_nbf "
            f"deliberately to override."
        )

    ne = effective_electron_count(molecule, uhf)
    two_s = molecule.multiplicity - 1
    na = (ne + two_s) // 2
    nb = (ne - two_s) // 2
    from vibeqc.correlation_conventions import resolve_frozen_core_count

    nf = resolve_frozen_core_count(molecule, options.n_frozen, reference=uhf)
    if nf < 0 or nf > nb:
        raise ValueError(f"n_frozen={nf} out of range for n_beta={nb}")

    n_act_a = na - nf
    n_act_b = nb - nf
    n_mo = nbf - nf  # active spatial MOs per spin (occ_act + vir), equal a/b

    Cao, Cav = Ca[:, nf:na], Ca[:, na:]
    Cbo, Cbv = Cb[:, nf:nb], Cb[:, nb:]

    # ----- occupied orbitals -----------------------------------------------
    if options.localise == "boys":
        Cao_use = _boys_localise(Cao, basis)
        Cbo_use = _boys_localise(Cbo, basis)
    else:
        Cao_use, Cbo_use = Cao, Cbo

    # ----- spin-orbital engine (mirrors _ccsd_ref.run_ref_uccsd lines ~402) -
    C_eng_a = np.hstack([Cao_use, Cav])  # (nbf, n_mo)
    C_eng_b = np.hstack([Cbo_use, Cbv])
    f_a = C_eng_a.T @ Fa @ C_eng_a
    f_b = C_eng_b.T @ Fb @ C_eng_b
    B_a = np.asarray(df.mo_transform(C_eng_a, C_eng_a))
    B_b = np.asarray(df.mo_transform(C_eng_b, C_eng_b))

    no = n_act_a + n_act_b
    nv = (n_mo - n_act_a) + (n_mo - n_act_b)
    f_so = np.zeros((2 * n_mo, 2 * n_mo))
    f_so[0::2, 0::2] = f_a  # alpha spin orbitals at even indices
    f_so[1::2, 1::2] = f_b  # beta at odd
    eri = _spin_orbital_eri_uhf(B_a, B_b, n_mo)
    # occupied spin orbitals first (alpha occ = even < 2 n_act_a; beta occ =
    # odd < 2 n_act_b + 1); the complement is virtual.
    occ = np.concatenate([2 * np.arange(n_act_a), 2 * np.arange(n_act_b) + 1])
    vir = np.concatenate(
        [2 * np.arange(n_act_a, n_mo), 2 * np.arange(n_act_b, n_mo) + 1]
    )
    order = np.concatenate([np.sort(occ), np.sort(vir)])
    f_so = f_so[np.ix_(order, order)]
    eri = eri[np.ix_(order, order, order, order)]
    o = slice(0, no)
    v = slice(no, no + nv)
    fock_od = f_so.copy()
    np.fill_diagonal(fock_od, 0.0)
    eps_so = np.diag(f_so)
    eps_o, eps_v = eps_so[o], eps_so[v]
    D1 = eps_o[:, None] - eps_v[None, :]
    D2 = (
        eps_o[:, None, None, None]
        + eps_o[None, :, None, None]
        - eps_v[None, None, :, None]
        - eps_v[None, None, None, :]
    )

    # ----- per-pair PNO spaces ---------------------------------------------
    # For each occupied spin-orbital pair P<Q: the spin-orbital MP2 amplitude
    # t2_mp2 = <PQ||AB> / D2 is antisymmetric in (A,B); the pair density
    # D = T Tᵀ + Tᵀ T is PSD; its leading natural orbitals (occupation >
    # tcut_pno) span the pair PNO domain, quasi-canonicalised against the
    # diagonal virtual Fock so the denominators stay diagonal.
    eri_oovv = eri[o, o, v, v]
    pair_keys = [(P, Q) for P in range(no) for Q in range(P + 1, no)]
    U: dict = {}
    eps_pno: dict = {}
    T2: dict = {}
    for (P, Q) in pair_keys:
        t2_mp2 = eri_oovv[P, Q] / D2[P, Q]
        D_pair = t2_mp2 @ t2_mp2.T + t2_mp2.T @ t2_mp2
        D_pair = 0.5 * (D_pair + D_pair.T)
        occs, d = np.linalg.eigh(D_pair)
        ordd = np.argsort(-occs)
        occs, d = occs[ordd], d[:, ordd]
        keep = (
            occs > options.tcut_pno
            if options.tcut_pno > 0.0
            else np.ones_like(occs, dtype=bool)
        )
        if not np.any(keep):
            keep[0] = True
        d = d[:, keep]
        F_p = d.T @ np.diag(eps_v) @ d
        F_p = 0.5 * (F_p + F_p.T)
        e_p, u_rot = np.linalg.eigh(F_p)
        d = d @ u_rot
        U[(P, Q)] = d  # (nv, n_pno) -- coordinates in the spin-orbital vir basis
        eps_pno[(P, Q)] = e_p
        K_p = d.T @ eri_oovv[P, Q] @ d
        T2[(P, Q)] = K_p / (eps_o[P] + eps_o[Q] - e_p[:, None] - e_p[None, :])

    t1_full = np.zeros((no, nv))  # singles in the full spin-orbital virtual space

    result = DLPNOUCCSDPilotResult(
        e_hf=float(uhf.energy), n_frozen=nf, localise=options.localise
    )
    result.n_pairs = len(pair_keys)
    result.pno_per_pair = {k: U[k].shape[1] for k in pair_keys}
    result.avg_pno = (
        float(np.mean([U[k].shape[1] for k in pair_keys])) if pair_keys else 0.0
    )
    if not pair_keys:
        result.converged = True
        result.e_total = result.e_hf
        return result

    def expand_t2():
        """Constrained per-pair amplitudes -> full antisymmetric t2[P,Q,A,B]."""
        t2f = np.zeros((no, no, nv, nv))
        for (P, Q), T in T2.items():
            blk = U[(P, Q)] @ T @ U[(P, Q)].T
            t2f[P, Q] = blk
            t2f[Q, P] = -blk
        return t2f

    def flatten(t1, T2d):
        return np.concatenate([t1.ravel()] + [T2d[k].ravel() for k in pair_keys])

    def unflatten(vec):
        off = t1_full.size
        t1 = vec[:off].reshape(t1_full.shape).copy()
        T2d = {}
        for k in pair_keys:
            n = T2[k].size
            T2d[k] = vec[off : off + n].reshape(T2[k].shape).copy()
            off += n
        return t1, T2d

    # DIIS over the concatenated [t1_full, per-pair T2] vector.
    amp_hist: list = []
    res_hist: list = []
    e_prev = 0.0
    for iteration in range(options.max_iter):
        t2f = expand_t2()
        r1, r2 = so_residuals(f_so, fock_od, eri, t1_full, t2f, o, v, D1, D2)

        new_t1 = t1_full + r1 / D1
        max_r = float(np.max(np.abs(r1 / D1)))
        new_T2: dict = {}
        for (P, Q), T in T2.items():
            Rp = U[(P, Q)].T @ r2[P, Q] @ U[(P, Q)]
            denom = (
                eps_o[P]
                + eps_o[Q]
                - eps_pno[(P, Q)][:, None]
                - eps_pno[(P, Q)][None, :]
            )
            step = Rp / denom
            max_r = max(max_r, float(np.max(np.abs(step))))
            Tn = T + step
            new_T2[(P, Q)] = 0.5 * (Tn - Tn.T)  # keep antisymmetric

        amp_hist.append(flatten(new_t1, new_T2))
        res_hist.append(amp_hist[-1] - flatten(t1_full, T2))
        if len(amp_hist) > options.diis_size:
            amp_hist.pop(0)
            res_hist.pop(0)
        n_h = len(amp_hist)
        if n_h >= 2:
            Bm = np.empty((n_h + 1, n_h + 1))
            Bm[-1, :] = -1.0
            Bm[:, -1] = -1.0
            Bm[-1, -1] = 0.0
            for a_ in range(n_h):
                for b_ in range(n_h):
                    Bm[a_, b_] = float(np.dot(res_hist[a_], res_hist[b_]))
            rhs = np.zeros(n_h + 1)
            rhs[-1] = -1.0
            try:
                c = np.linalg.solve(Bm, rhs)[:n_h]
                t1_full, T2 = unflatten(
                    sum(c[k] * amp_hist[k] for k in range(n_h))
                )
            except np.linalg.LinAlgError:
                t1_full, T2 = new_t1, new_T2
        else:
            t1_full, T2 = new_t1, new_T2

        t2f = expand_t2()
        e_corr = so_energy(f_so, eri, t1_full, t2f, o, v)
        delta_e = e_corr - e_prev
        result.trace.append(
            {"iter": iteration + 1, "e_corr": e_corr, "delta_e": delta_e, "max_r": max_r}
        )
        if (
            iteration > 0
            and abs(delta_e) < options.conv_tol_energy
            and max_r < options.conv_tol_residual
        ):
            result.converged = True
            result.n_iter = iteration + 1
            break
        e_prev = e_corr
    else:
        result.n_iter = options.max_iter

    t2f = expand_t2()
    result.e_corr = so_energy(f_so, eri, t1_full, t2f, o, v)
    result.t1_norm = float(np.linalg.norm(t1_full))

    if options.compute_triples:
        result.e_t = _triples(
            options.localise, eps_so, eri, t1_full, t2f, o, v, order, no, n_mo,
            Cao, Cbo, Cao_use, Cbo_use, Cav, Cbv, Fa, Fb, S_ao, df,
        )
        result.triples_executed = True
    result.e_total = result.e_hf + result.e_corr + result.e_t
    return result


def _triples(
    localise, eps_so, eri, t1_full, t2f, o, v, order, no, n_mo,
    Cao, Cbo, Cao_use, Cbo_use, Cav, Cbv, Fa, Fb, S_ao, df,
):
    """Perturbative (T) on the converged amplitudes.

    For ``localise="none"`` the occupieds are canonical and the (T) is exact
    in the spin-orbital basis. For ``localise="boys"`` the converged
    amplitudes are first rotated localised->canonical (occupied indices only)
    and the (T) is evaluated in the canonical spin-orbital basis, so the
    triples denominators are exact rather than semicanonical -- mirroring the
    closed-shell pilot's exact-(T) path (ccsd.py).
    """
    if localise == "none":
        if _HAVE_CPP_TRIPLES:
            no_int = o.stop - o.start if isinstance(o, slice) else len(o)
            nv_int = v.stop - v.start if isinstance(v, slice) else len(v)
            nso = no_int + nv_int
            # C++ layout: t1(nv, no), t2(nv, nv, no, no).  The Python
            # reference stores occupied-first; transpose before calling.
            t1_cpp = np.ascontiguousarray(np.asarray(t1_full).T)
            t2_cpp = np.ascontiguousarray(np.asarray(t2f).transpose(2, 3, 0, 1))
            return _cpp_ccsd_t_triples(
                t1_cpp, t2_cpp,
                np.ascontiguousarray(eri),
                np.ascontiguousarray(eps_so[o]),
                np.ascontiguousarray(eps_so[v]),
                no_int, nv_int, nso,
            )
        return so_triples_correction(eps_so, eri, t1_full, t2f, o, v)

    # localised->canonical occupied rotation, per spin: C_loc = C_can @ Uo.
    Uo_a = Cao.T @ S_ao @ Cao_use  # (n_act_a, n_act_a)
    Uo_b = Cbo.T @ S_ao @ Cbo_use  # (n_act_b, n_act_b)
    Uo_occ = np.zeros((no, no))
    for m in range(no):
        om = order[m]  # original spin-orbital index of occupied slot m
        for n_ in range(no):
            on = order[n_]
            if om % 2 == 0 and on % 2 == 0:  # both alpha
                Uo_occ[m, n_] = Uo_a[om // 2, on // 2]
            elif om % 2 == 1 and on % 2 == 1:  # both beta
                Uo_occ[m, n_] = Uo_b[om // 2, on // 2]
    t1_can = Uo_occ @ t1_full
    t2_can = np.einsum("Ii,Jj,ijab->IJab", Uo_occ, Uo_occ, t2f, optimize=True)

    # canonical spin-orbital engine (same virtuals, canonical occupieds)
    Cca = np.hstack([Cao, Cav])
    Ccb = np.hstack([Cbo, Cbv])
    f_can = np.zeros((2 * n_mo, 2 * n_mo))
    f_can[0::2, 0::2] = Cca.T @ Fa @ Cca
    f_can[1::2, 1::2] = Ccb.T @ Fb @ Ccb
    eri_can = _spin_orbital_eri_uhf(
        np.asarray(df.mo_transform(Cca, Cca)),
        np.asarray(df.mo_transform(Ccb, Ccb)),
        n_mo,
    )
    f_can = f_can[np.ix_(order, order)]
    eri_can = eri_can[np.ix_(order, order, order, order)]
    if _HAVE_CPP_TRIPLES:
        t1_cpp = np.ascontiguousarray(np.asarray(t1_can).T)
        t2_cpp = np.ascontiguousarray(np.asarray(t2_can).transpose(2, 3, 0, 1))
        nv_int = v.stop - v.start if isinstance(v, slice) else len(v)
        return _cpp_ccsd_t_triples(
            t1_cpp, t2_cpp,
            np.ascontiguousarray(eri_can),
            np.ascontiguousarray(np.diag(f_can)[o]),
            np.ascontiguousarray(np.diag(f_can)[v]),
            no, nv_int, 2 * n_mo,
        )
    return so_triples_correction(np.diag(f_can), eri_can, t1_can, t2_can, o, v)
