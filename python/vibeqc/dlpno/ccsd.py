"""DLPNO-CCSD pilot -- subspace-projected coupled cluster (M3a).

Physically exact DLPNO-CCSD on validation-scale molecules, built to
eliminate equation-transcription risk: the CCSD residuals are evaluated
by the FCI-anchored spin-orbital engine (`dlpno._ccsd_ref`, Stanton-
Gauss-Watts-Bartlett 1991), while the *ansatz* is the DLPNO one -- T2
amplitudes live in per-pair PNO spaces and T1 in orbital-specific
singles spaces (the diagonal-pair PNOs), exactly as in Riplinger &
Neese 2013. Each iteration expands the constrained amplitudes to the
full space, evaluates the exact residuals, and projects them back into
the retained subspaces (the stationarity conditions of subspace-
constrained CCSD).

This is the correctness pilot: O(N⁶) full-space contractions make it a
*pilot*, not a production method -- the reduced-scaling per-pair
contraction engine (M3c) will be validated against the recovery tiers
this pilot establishes. The expand-project structure is exact: with
untruncated spaces it is canonical CCSD by construction, and with
truncated spaces it is the textbook DLPNO variational-subspace ansatz.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

from ._ccsd_cs import _blocks as _cs_blocks, cs_ccsd_residual
from ._ccsd_ref import (
    _spin_orbital_eri,
    so_energy,
    so_residuals,
    so_triples_correction,
)
from .pao import (
    build_atom_basis_map,
    build_projection_matrix,
    select_domain_atoms_mulliken,
    semicanonical_pao_basis,
)
from .pno_density import pair_density, resolve_pno_norm


@dataclass
class DLPNOCCSDPilotOptions:
    """Options for the DLPNO-CCSD correctness pilot.

    Mirrors `DLPNOMP2Options` where meaningful; no weak-pair tier or
    distant-pair screening yet (pairs are cheap at pilot scale -- those
    join in M3b together with the recovery-tier measurements).
    """

    localise: str = "boys"
    n_frozen: int | None = None
    # Liakos 2015 NormalPNO is the cross-route project policy. This pilot
    # consumes TCutPNO and TCutMKN; it has no pair-screening implementation,
    # so TCutPairs remains explicitly unsupported rather than decorative.
    # Pinski's tighter MP2-specific PNO preset is not a CCSD default.
    tcut_pno: float = 3.33e-7
    tcut_mkn: float = 1e-3
    #: Pair density whose eigenvalues are compared against ``tcut_pno``.
    #: ``"mp2"`` (default) is Riplinger and Neese 2013 Eq. 23, ORCA's
    #: ``PNONorm MP2Norm`` default and the density the published ``TCutPNO``
    #: presets are calibrated against; ``"legacy"`` is vibe-qc's historical
    #: density of the bare amplitudes, ``"iepa"`` the pre-2013 LPNO
    #: convention. Defaulted to ``"mp2"`` by the #65 (old #701) ruling. See
    #: :mod:`vibeqc.dlpno.pno_density`.
    pno_norm: str = "mp2"
    lindep: float = 1e-8
    max_iter: int = 80
    conv_tol_energy: float = 1e-9
    conv_tol_residual: float = 1e-7
    diis_size: int = 6
    compute_triples: bool = False
    # Pilot-cost guard: the expand/project oracle is O(N^6) in the full
    # space. Raise rather than silently launch week-long jobs; the
    # reduced-scaling engine (M3c) will lift this.
    max_nbf: int = 64
    # RI fitting basis for the correlation integrals. None -> the runner
    # auto-resolves it from the orbital basis (density_fitting.
    # default_aux_basis_for(basis, kind="ri")). Set explicitly (e.g.
    # "cc-pvdz-ri") for orbital bases that ship no default RI aux.
    aux_basis: str | None = None


@dataclass
class DLPNOCCSDPilotResult:
    e_hf: float = 0.0
    e_corr: float = 0.0
    e_t: float = 0.0
    e_total: float = 0.0
    n_iter: int = 0
    converged: bool = False
    n_pairs: int = 0
    n_frozen: int = 0
    t1_norm: float = 0.0
    pno_per_pair: dict = field(default_factory=dict)
    trace: list = field(default_factory=list)
    triples_executed: bool = False


def run_dlpno_ccsd_pilot(
    molecule,
    basis,
    rhf,
    df,
    options: DLPNOCCSDPilotOptions | None = None,
) -> DLPNOCCSDPilotResult:
    """Run the subspace-projected DLPNO-CCSD pilot.

    Parameters mirror `dlpno.mp2.run_dlpno_mp2` (closed-shell RHF
    reference, python `DensityFitting` with a real RI fitting basis).
    """
    from vibeqc._vibeqc_core import compute_overlap
    from vibeqc.correlation_conventions import effective_electron_count

    if options is None:
        options = DLPNOCCSDPilotOptions()
    pno_norm = resolve_pno_norm(getattr(options, "pno_norm", "mp2"))

    F_ao = np.asarray(rhf.fock).copy()
    S_ao = np.asarray(compute_overlap(basis)).copy()
    C = np.asarray(rhf.mo_coeffs).copy()
    n_occ = effective_electron_count(molecule, rhf) // 2
    nbf = C.shape[0]
    if nbf > options.max_nbf:
        raise ValueError(
            f"DLPNO-CCSD pilot is an O(N^6) correctness oracle, capped at "
            f"max_nbf={options.max_nbf} basis functions (got {nbf}). The "
            f"production reduced-scaling engine has its own local route. "
            f"Raise DLPNOCCSDPilotOptions.max_nbf deliberately to override."
        )

    from vibeqc.correlation_conventions import resolve_frozen_core_count

    nf = resolve_frozen_core_count(molecule, options.n_frozen, reference=rhf)
    if nf < 0 or nf >= n_occ:
        raise ValueError(f"n_frozen={nf} out of range for n_occ={n_occ}")
    n_act = n_occ - nf
    C_occ_full = C[:, :n_occ]
    C_act = C[:, nf:n_occ]
    C_vir = C[:, n_occ:]
    nv = C_vir.shape[1]

    # ----- occupied orbitals -----------------------------------------------
    if options.localise == "boys":
        from vibeqc import compute_dipole
        from vibeqc.localise import foster_boys_localise

        dip = compute_dipole(basis)
        dipoles = np.zeros((nbf, nbf, 3))
        dipoles[:, :, 0] = np.asarray(dip.x)
        dipoles[:, :, 1] = np.asarray(dip.y)
        dipoles[:, :, 2] = np.asarray(dip.z)
        C_loc = foster_boys_localise(C_act, dipoles, max_iter=200)
    elif options.localise == "none":
        C_loc = C_act
    else:
        raise ValueError(f"unknown localise option: {options.localise!r}")

    F_oo = C_loc.T @ F_ao @ C_loc

    # ----- engine setup: active-occupied + full-virtual MO space -----------
    C_eng = np.hstack([C_loc, C_vir])  # (nbf, n_act + nv)
    f_eng = C_eng.T @ F_ao @ C_eng
    B_eng = np.asarray(df.mo_transform(C_eng, C_eng))
    n_mo = n_act + nv

    no_so = 2 * n_act
    nv_so = 2 * nv
    f_so = np.zeros((2 * n_mo, 2 * n_mo))
    f_so[0::2, 0::2] = f_eng
    f_so[1::2, 1::2] = f_eng
    eri = _spin_orbital_eri(B_eng, n_mo)
    # occupied-first interleaved (a,b per spatial orbital) -- matches the
    # ordering run_ref_ccsd uses, with spatial occupieds already first.
    o = slice(0, no_so)
    v = slice(no_so, no_so + nv_so)
    fock_od = f_so.copy()
    np.fill_diagonal(fock_od, 0.0)
    eps_so = np.diag(f_so)

    # Spatial closed-shell engine: the per-iteration residual is the
    # spatial cs_ccsd_residual (no spin-orbital expansion), 16x cheaper
    # per step than so_residuals. The spin-orbital arrays above are kept
    # only for the end-of-run (T) correction.
    f_oo_eng = f_eng[:n_act, :n_act]
    f_vv_eng = f_eng[n_act:, n_act:]
    f_ov_eng = f_eng[:n_act, n_act:]
    B_ov_eng = np.ascontiguousarray(B_eng[:, :n_act, n_act:])
    B_vv_eng = np.ascontiguousarray(B_eng[:, n_act:, n_act:])
    B_oo_eng = np.ascontiguousarray(B_eng[:, :n_act, :n_act])
    V_eng = _cs_blocks(B_ov_eng, B_vv_eng, B_oo_eng)
    _ovov_eng = V_eng["ovov"]
    _Loovv_eng = (
        2.0 * _ovov_eng.transpose(0, 2, 1, 3)
        - _ovov_eng.transpose(0, 2, 3, 1)
    )

    def cs_energy(t1f, T2f):
        tau = T2f + np.einsum("ia,jb->ijab", t1f, t1f)
        return float(
            2.0 * np.einsum("ia,ia->", f_ov_eng, t1f)
            + np.einsum("ijab,ijab->", tau, _Loovv_eng)
        )
    D1 = eps_so[o, None] - eps_so[None, v]
    D2 = (
        eps_so[o][:, None, None, None]
        + eps_so[o][None, :, None, None]
        - eps_so[v][None, None, :, None]
        - eps_so[v][None, None, None, :]
    )

    # ----- pair PNO spaces (M2 machinery, duplicated for the pilot) --------
    atom_first, _ = build_atom_basis_map(molecule, basis)
    natom = len(atom_first) - 1
    Q_vir = build_projection_matrix(C_occ_full, S_ao)
    B_half = df.mo_transform(C_loc, np.eye(nbf))  # (naux, n_act, nbf)

    pair_keys = [(i, j) for i in range(n_act) for j in range(i, n_act)]
    U: dict = {}  # PNO coordinates in the canonical-virtual basis (nv, n_p)
    eps_pno: dict = {}
    T2: dict = {}
    f_dd = np.diag(F_oo)

    for (i, j) in pair_keys:
        if options.tcut_mkn > 0.0:
            domain_atoms = select_domain_atoms_mulliken(
                C_loc, S_ao, atom_first, i, j, options.tcut_mkn
            )
        else:
            domain_atoms = np.arange(natom, dtype=int)
        ao_mask = np.zeros(nbf, dtype=bool)
        for a in domain_atoms:
            ao_mask[atom_first[a] : atom_first[a + 1]] = True

        V_semi, eps_pao = semicanonical_pao_basis(
            F_ao, S_ao, Q_vir, np.where(ao_mask)[0], options.lindep
        )
        if V_semi.shape[1] == 0:
            continue
        B_i = B_half[:, i, :] @ V_semi
        B_j = B_half[:, j, :] @ V_semi
        K_pao = B_i.T @ B_j
        denom = f_dd[i] + f_dd[j] - eps_pao[:, None] - eps_pao[None, :]
        T_pao = K_pao / denom

        delta = 1.0 if i == j else 0.0
        D_pair = pair_density(T_pao, delta, pno_norm)
        occs, d = np.linalg.eigh(D_pair)
        order = np.argsort(-occs)
        occs, d = occs[order], d[:, order]
        keep = (
            occs > options.tcut_pno
            if options.tcut_pno > 0.0
            else np.ones_like(occs, dtype=bool)
        )
        if not np.any(keep):
            keep[0] = True
        d = d[:, keep]

        F_p = d.T @ np.diag(eps_pao) @ d
        F_p = 0.5 * (F_p + F_p.T)
        e_p, u_rot = np.linalg.eigh(F_p)
        d = d @ u_rot

        V_pno = V_semi @ d  # (nbf, n_p)
        U[(i, j)] = C_vir.T @ S_ao @ V_pno  # exact coords: PNOs subset span(C_vir)
        eps_pno[(i, j)] = e_p
        K_p = d.T @ K_pao @ d
        T2[(i, j)] = K_p / (f_dd[i] + f_dd[j] - e_p[:, None] - e_p[None, :])

    t1: dict = {i: np.zeros(U[(i, i)].shape[1]) for i in range(n_act) if (i, i) in U}

    result = DLPNOCCSDPilotResult(e_hf=float(rhf.energy), n_frozen=nf)
    result.n_pairs = len(U)
    result.pno_per_pair = {
        (k[0] + nf, k[1] + nf): U[k].shape[1] for k in U
    }
    if not U:
        result.converged = True
        result.e_total = result.e_hf
        return result

    def expand_spatial():
        """Constrained amplitudes -> full spatial arrays (t1f, T2f)."""
        T2f = np.zeros((n_act, n_act, nv, nv))
        for (i, j), T in T2.items():
            blk = U[(i, j)] @ T @ U[(i, j)].T
            T2f[i, j] = blk
            T2f[j, i] = blk.T
        t1f = np.zeros((n_act, nv))
        for i, ti in t1.items():
            t1f[i] = U[(i, i)] @ ti
        return t1f, T2f

    def expand_full():
        """Constrained amplitudes -> full spatial -> spin-orbital arrays."""
        t1f, T2f = expand_spatial()

        t1_so = np.zeros((no_so, nv_so))
        t1_so[0::2, 0::2] = t1f
        t1_so[1::2, 1::2] = t1f
        # t2_so[IJAB] = T[i,j,a,b] d(sI,sA)d(sJ,sB) - T[i,j,b,a] d(sI,sB)d(sJ,sA)
        t2_so = np.zeros((no_so, no_so, nv_so, nv_so))
        for si in (0, 1):
            for sj in (0, 1):
                t2_so[si::2, sj::2, si::2, sj::2] += T2f
                t2_so[si::2, sj::2, sj::2, si::2] -= T2f.transpose(0, 1, 3, 2)
        return t1f, T2f, t1_so, t2_so

    # DIIS over the concatenated constrained amplitudes.
    amp_hist: list = []
    res_hist: list = []

    def flatten(t1d, T2d):
        parts = [t1d[i].ravel() for i in sorted(t1d)] + [
            T2d[k].ravel() for k in sorted(T2d)
        ]
        return np.concatenate(parts)

    def unflatten(vec):
        t1n, T2n, off = {}, {}, 0
        for i in sorted(t1):
            n = t1[i].size
            t1n[i] = vec[off : off + n].copy()
            off += n
        for k in sorted(T2):
            n = T2[k].size
            T2n[k] = vec[off : off + n].reshape(T2[k].shape).copy()
            off += n
        return t1n, T2n

    e_prev = 0.0
    for iteration in range(options.max_iter):
        t1f, T2f = expand_spatial()
        r1_sp, r2_sp = cs_ccsd_residual(
            t1f, T2f, f_oo_eng, f_vv_eng, f_ov_eng, V_eng
        )

        max_r = 0.0
        new_t1: dict = {}
        new_T2: dict = {}
        for (i, j), T in T2.items():
            Rp = U[(i, j)].T @ r2_sp[i, j] @ U[(i, j)]
            max_r = max(max_r, float(np.max(np.abs(Rp))))
            denom = (
                f_dd[i]
                + f_dd[j]
                - eps_pno[(i, j)][:, None]
                - eps_pno[(i, j)][None, :]
            )
            new_T2[(i, j)] = T + Rp / denom
        for i, ti in t1.items():
            rp = U[(i, i)].T @ r1_sp[i]
            max_r = max(max_r, float(np.max(np.abs(rp))))
            new_t1[i] = ti + rp / (f_dd[i] - eps_pno[(i, i)])

        # Diagonal pairs stay symmetric.
        for (i, j) in new_T2:
            if i == j:
                new_T2[(i, j)] = 0.5 * (new_T2[(i, j)] + new_T2[(i, j)].T)

        # DIIS on the constrained amplitude vector.
        amp_hist.append(flatten(new_t1, new_T2))
        res_hist.append(amp_hist[-1] - flatten(t1, T2))
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
                t1, T2 = unflatten(sum(c[k] * amp_hist[k] for k in range(n_h)))
            except np.linalg.LinAlgError:
                t1, T2 = new_t1, new_T2
        else:
            t1, T2 = new_t1, new_T2

        t1f, T2f = expand_spatial()
        e_corr = cs_energy(t1f, T2f)
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

    t1f, T2f = expand_spatial()
    result.e_corr = cs_energy(t1f, T2f)
    if options.compute_triples:
        # (T) still runs in the spin-orbital triples space; expand once.
        _, _, t1_so, t2_so = expand_full()
        # Exact (T) on the converged DLPNO amplitudes in the full triples
        # space: occupied indices are rotated back to the canonical basis
        # first, so the triples denominators are exact rather than
        # semicanonical. (The production T0 shortcut -- diagonal localised
        # Fock in the denominators -- is an M3c tradeoff; the pilot is the
        # oracle, so it evaluates the exact quantity. Triples-space
        # locality truncation is likewise M3c scope.)
        if options.localise == "none":
            result.e_t = so_triples_correction(eps_so, eri, t1_so, t2_so, o, v)
        else:
            Uo = C_act.T @ S_ao @ C_loc  # C_loc = C_act @ Uo
            Uo_so = np.zeros((no_so, no_so))
            Uo_so[0::2, 0::2] = Uo
            Uo_so[1::2, 1::2] = Uo
            t1_can = Uo_so @ t1_so
            t2_can = np.einsum(
                "Ii,Jj,ijab->IJab", Uo_so, Uo_so, t2_so, optimize=True
            )
            C_can = np.hstack([C_act, C_vir])
            f_can = C_can.T @ F_ao @ C_can
            B_can = np.asarray(df.mo_transform(C_can, C_can))
            eri_can = _spin_orbital_eri(B_can, n_mo)
            f_can_so = np.zeros((2 * n_mo, 2 * n_mo))
            f_can_so[0::2, 0::2] = f_can
            f_can_so[1::2, 1::2] = f_can
            result.e_t = so_triples_correction(
                np.diag(f_can_so), eri_can, t1_can, t2_can, o, v
            )
        result.triples_executed = True
    result.e_total = result.e_hf + result.e_corr + result.e_t
    result.t1_norm = float(
        np.sqrt(sum(float(np.dot(x, x)) for x in t1.values()))
    )
    return result
