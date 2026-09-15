"""DLPNO local correlation on the cyclic cluster (Task 2).

The local-correlation pipeline for periodic DLPNO on the CCM: projected atomic
orbitals (PAO) -> pair domains -> pair natural orbitals (PNO) -> DLPNO-MP2 (and, in
:mod:`vibeqc.periodic.ccm.dlpno_ccsd`, DLPNO-CCSD(T)). It reuses the molecular
DLPNO machinery (:mod:`vibeqc.dlpno.pao`, :mod:`vibeqc.dlpno.mp2`) on the
cyclic-cluster occupied space + ``S^CCM``, with the **neutral** four-center as the
correlation reference.

Identity note: this is a neutral fitted-torus correlation control. The legacy
``ccm_*`` API name describes its cyclic-cluster container and ancestry; it does
not assign either the union-and-weight Γ-CCM construction or the finite-character
χ-CCM construction to this Hamiltonian.

PAO (Pulay, Chem. Phys. Lett. 100, 151 (1983); Riplinger & Neese,
J. Chem. Phys. 138, 034106 (2013) Sec.II.D): the AO basis projected out of the
occupied space,

    C_PAO = (I - C_occ C_occᵀ S^CCM) . A ,

canonical-orthogonalized within a domain (eigenvectors of ``Q S Q`` above a
linear-dependence threshold). The occupied **space** projector is invariant under
occupied rotations, so the full-cell PAO set is the same whether the canonical or
the localized (Task 1) occupieds are used; the localized occupieds enter at the
pair-domain stage. Foundations from Task 1 (localized occupieds) + Task 3
(symmetry-unique pairs) feed the domain/PNO steps.

**Correlation reference = neutral.** Following the `-b`-critique correction
(main ``301f5fcc``; ``docs/aiccm2026dev_a_followon.md`` Sec. 2.0 decision log), the
ionic Madelung background shifts the occupied-virtual *denominators*, so the
neutral four-center (the RI-consistent ``ccm_neutral_cderi``) is the required
reference for CCM correlation -- *not* the bare-1/r four-center. The DLPNO-MP2
driver here density-fits the neutral cderi for its ``(ia|jb)`` integrals and runs
on the matching neutral SCF reference; the no-truncation limit reproduces the
canonical neutral-control MP2 on the *same* neutral four-center (the hard gate).
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Optional

import numpy as np

__all__ = [
    "ccm_pao",
    "pao_occupied_orthogonality",
    "CCMDLPNOMP2Result",
    "ccm_dlpno_mp2",
]


def _occ(ccm_result, n_occ: int) -> np.ndarray:
    C = np.asarray(ccm_result.mo_coeffs)
    if np.iscomplexobj(C):
        C = np.real_if_close(C, tol=1000).real
    return np.array(C[:, :n_occ], dtype=float)


def ccm_pao(ccm_result, ccm, *, n_occ: int | None = None, lindep_thresh: float = 1e-8):
    """Full-cell projected atomic orbitals of the cyclic cluster.

    Parameters
    ----------
    ccm_result
        A converged CCM SCF result (``mo_coeffs`` + ``overlap`` = ``S^CCM``).
    ccm
        The :class:`~vibeqc.periodic.ccm.CCMSystem`.
    n_occ
        Occupied count; defaults to the supercell closed-shell ``n_electrons//2``.
    lindep_thresh
        Eigenvalue floor for the canonical orthogonalization (drops redundant
        PAOs from linear dependencies in the AO basis).

    Returns
    -------
    (C_pao, n_pao)
        ``C_pao`` (nbf x n_pao) PAO coefficients in the supercell AO basis,
        S-orthogonal to the occupied space; ``n_pao`` = retained count
        (= virtual dimension absent linear dependencies).
    """
    from ...dlpno.pao import build_pao_coeffs, build_projection_matrix

    if n_occ is None:
        n_occ = int(ccm.supercell.n_electrons()) // 2
    S = np.asarray(ccm_result.overlap, dtype=float)
    C_occ = _occ(ccm_result, n_occ)
    Q = build_projection_matrix(C_occ, S)
    C_pao, n_pao = build_pao_coeffs(
        Q, np.ones(int(ccm.nbf), dtype=bool), S, lindep_thresh=lindep_thresh)
    return C_pao, int(n_pao)


def pao_occupied_orthogonality(ccm_result, ccm, C_pao, *, n_occ: int | None = None) -> float:
    """``max|C_occᵀ S^CCM C_pao|`` -- PAOs must be S-orthogonal to the occupied space (~0)."""
    if n_occ is None:
        n_occ = int(ccm.supercell.n_electrons()) // 2
    S = np.asarray(ccm_result.overlap, dtype=float)
    C_occ = _occ(ccm_result, n_occ)
    return float(np.max(np.abs(C_occ.T @ S @ np.asarray(C_pao, dtype=float))))


# ===========================================================================
# DLPNO-MP2 on the cyclic cluster (M2)
# ===========================================================================
#
# Mirrors the validated molecular DLPNO-MP2 (vibeqc.dlpno.mp2.run_dlpno_mp2),
# reusing its component functions (build_atom_basis_map, build_projection_matrix,
# select_domain_atoms_mulliken, semicanonical_pao_basis) but with the cyclic-
# cluster overlap S^CCM and Fock F^CCM injected, and the (ia|jb) integrals
# density-fit from the *neutral* cderi L[P,muν] (g_eff = S_P L⊗L). The pieces:
#
#   PAO -> per-pair Mulliken domain -> semicanonical PAOs (eigck of F in the
#   domain, S^CCM-orthonormal) -> pair amplitudes T_ab = K_ab/(f_ii+f_jj-e_a-e_b)
#   -> PNOs (eigvecs of the pair density, occupation-threshold truncation) ->
#   coupled LMP2 residual (needed when the occupieds are localized; for canonical
#   occupieds F_oo is diagonal and it converges in one step to canonical MP2).
#
# Reference: Riplinger & Neese, J. Chem. Phys. 138, 034106 (2013); Pinski et al.,
# J. Chem. Phys. 143, 034108 (2015).


def _semicanonical_T(K: np.ndarray, f_ii: float, f_jj: float, eps: np.ndarray) -> np.ndarray:
    """First-order amplitudes ``T_ab = K_ab / (f_ii + f_jj - e_a - e_b)``."""
    denom = f_ii + f_jj - eps[:, None] - eps[None, :]
    return K / denom


def _pair_energy(K: np.ndarray, T: np.ndarray, i: int, j: int) -> float:
    """Closed-shell pair energy ``e_ij = (2-d_ij).S_ab K_ab (2 T_ab - T_ba)``."""
    w = 1.0 if i == j else 2.0
    return w * float(np.sum(K * (2.0 * T - T.T)))


@dataclass
class _CCMPairData:
    i: int
    j: int
    V: np.ndarray        # PNO coefficients in the supercell AO basis (nbf x n_pno)
    K: np.ndarray        # (ia|jb) in the quasi-canonical PNO basis
    eps: np.ndarray      # PNO orbital energies
    T: np.ndarray        # amplitudes
    n_pno: int
    n_pao: int


@dataclass
class CCMDLPNOMP2Result:
    """Result of :func:`ccm_dlpno_mp2`."""

    e_hf: float
    e_corr: float
    e_total: float
    e_corr_per_atom: float
    e_pno_correction: float            # S_pair (e_full_PAO - e_trunc_PNO), semicanonical
    n_pairs: int
    n_occ_active: int
    localize: str
    converged: bool
    n_iter: int
    pno_per_pair: dict = field(default_factory=dict)
    pair_energies: dict = field(default_factory=dict)
    guess_selection: object = None
    # Per-call citation key; see CCMMP2Result.backend for why this is not a
    # class default. The DLPNO drivers ride the NEUTRAL cderi, so they are
    # the neutral lineage and never the bare four-centre one.
    backend: str = ""


# Accepted occupied-gauge names -> the localizer's method string.
_LOCALIZE_ALIASES = {
    "none": None,
    "pm": "pipek-mezey",
    "pipek-mezey": "pipek-mezey",
    "boys": "boys",
}


def _localised_occupieds(ccm_result, ccm, n_occ, localize):
    """Active occupied coefficients (canonical or localized) on the AO basis."""
    method = _LOCALIZE_ALIASES[localize]
    if method is None:
        return _occ(ccm_result, n_occ)
    from .localize import localise_ccm

    w = localise_ccm(ccm_result, ccm, method=method)
    C_loc = np.asarray(w.C_loc, dtype=float)
    if np.iscomplexobj(C_loc):
        C_loc = np.real_if_close(C_loc, tol=1000).real
    return np.array(C_loc[:, :n_occ], dtype=float)


@dataclass
class _CCMPairBundle:
    """Shared output of the PAO->domain->PNO pair build (used by MP2 and CCSD)."""

    pairs: dict          # (i,j) -> _CCMPairData  (i,j active-occupied indices)
    e_pno_correction: float
    F_oo: np.ndarray     # active occ-occ Fock block (localized gauge)
    S: np.ndarray        # S^CCM
    C_loc: np.ndarray    # active occupied coefficients (nbf x n_act)
    n_act: int
    nf: int
    e_hf: float
    localize: str


def _build_ccm_pno_pairs(
    ccm, scf_result, *, cderi, localize, tcut_pno, tcut_mkn, n_frozen, lindep,
    ke_cutoff,
) -> _CCMPairBundle:
    """Build per-pair PNOs on the cyclic cluster (neutral reference).

    Shared by :func:`ccm_dlpno_mp2` and :func:`~vibeqc.periodic.ccm.dlpno_ccsd.
    ccm_dlpno_ccsd`. For each active occupied pair ``(i,j)``: Mulliken domain ->
    semicanonical PAOs (S^CCM-orthonormal) -> first-order amplitudes -> PNOs
    (pair-density eigvecs, ``tcut_pno`` truncation) -> quasi-canonicalize. The
    ``(ia|jb)`` integrals are density-fit from the neutral cderi.
    """
    from ...dlpno.pao import (
        build_atom_basis_map,
        build_projection_matrix,
        select_domain_atoms_mulliken,
        semicanonical_pao_basis,
    )
    from .neutral import ccm_neutral_cderi

    if localize not in _LOCALIZE_ALIASES:
        raise ValueError(
            f"unknown localize option {localize!r}; "
            f"choose from {sorted(_LOCALIZE_ALIASES)}")

    S = np.asarray(scf_result.overlap, dtype=float)
    F = np.asarray(scf_result.fock, dtype=float)
    if np.iscomplexobj(F):
        F = np.real_if_close(F, tol=1000).real
    nbf = int(ccm.nbf)

    n_occ = int(ccm.supercell.n_electrons()) // 2
    nf = int(n_frozen)
    if nf < 0 or nf >= n_occ:
        raise ValueError(f"n_frozen={nf} out of range for n_occ={n_occ}")
    n_act = n_occ - nf

    C_occ_full = _occ(scf_result, n_occ)               # all occupieds (for the projector)
    C_loc_all = _localised_occupieds(scf_result, ccm, n_occ, localize)
    C_loc = np.array(C_loc_all[:, nf:n_occ], dtype=float)   # active occupieds

    # Neutral cderi -> occupied half-transform B_half[P,i,ν] = S_mu C_loc[mu,i] L[P,mu,ν].
    L = np.asarray(cderi if cderi is not None else
                   ccm_neutral_cderi(ccm, ke_cutoff=ke_cutoff), dtype=float)
    B_half = np.einsum("mi,Pmn->Pin", C_loc, L, optimize=True)   # (naux, n_act, nbf)

    F_oo = C_loc.T @ F @ C_loc                          # active occ-occ Fock block
    atom_first, _ = build_atom_basis_map(ccm.supercell, ccm.basis)
    natom = len(atom_first) - 1
    Q_vir = build_projection_matrix(C_occ_full, S)      # project out ALL occupieds

    pairs: dict = {}
    e_pno_correction = 0.0
    all_pairs = [(i, j) for i in range(n_act) for j in range(i, n_act)]

    for (i, j) in all_pairs:
        if tcut_mkn > 0.0:
            domain_atoms = select_domain_atoms_mulliken(
                C_loc, S, atom_first, i, j, tcut_mkn)
        else:
            domain_atoms = np.arange(natom, dtype=int)
        ao_mask = np.zeros(nbf, dtype=bool)
        for a in domain_atoms:
            ao_mask[atom_first[a] : atom_first[a + 1]] = True
        ao_idx = np.where(ao_mask)[0]

        V_semi, eps_pao = semicanonical_pao_basis(F, S, Q_vir, ao_idx, lindep_thresh=lindep)
        n_pao = V_semi.shape[1]
        if n_pao == 0:
            continue

        B_i = B_half[:, i, :] @ V_semi                  # (naux, n_pao)
        B_j = B_half[:, j, :] @ V_semi
        K_pao = B_i.T @ B_j                             # (ia|jb)
        f_ii, f_jj = float(F_oo[i, i]), float(F_oo[j, j])
        T_pao = _semicanonical_T(K_pao, f_ii, f_jj, eps_pao)
        e_full = _pair_energy(K_pao, T_pao, i, j)

        # PNOs from the semicanonical pair density.
        delta = 1.0 if i == j else 0.0
        D_pair = (T_pao @ T_pao.T + T_pao.T @ T_pao) / (1.0 + delta)
        D_pair = 0.5 * (D_pair + D_pair.T)
        occs, d = np.linalg.eigh(D_pair)
        order = np.argsort(-occs)
        occs, d = occs[order], d[:, order]
        keep = occs > tcut_pno if tcut_pno > 0.0 else np.ones_like(occs, dtype=bool)
        n_pno = int(np.sum(keep))
        if n_pno == 0:
            keep[0] = True
            n_pno = 1
        d = d[:, keep]

        # Quasi-canonicalize within the retained PNO space.
        F_pno = d.T @ np.diag(eps_pao) @ d
        F_pno = 0.5 * (F_pno + F_pno.T)
        eps_pno, u = np.linalg.eigh(F_pno)
        d = d @ u

        V_pno = V_semi @ d
        K_pno = d.T @ K_pao @ d
        T_pno = _semicanonical_T(K_pno, f_ii, f_jj, eps_pno)
        e_pno_correction += e_full - _pair_energy(K_pno, T_pno, i, j)

        pairs[(i, j)] = _CCMPairData(
            i=i, j=j, V=V_pno, K=K_pno, eps=eps_pno, T=T_pno,
            n_pno=n_pno, n_pao=n_pao)

    return _CCMPairBundle(
        pairs=pairs, e_pno_correction=e_pno_correction, F_oo=F_oo, S=S,
        C_loc=C_loc, n_act=n_act, nf=nf, e_hf=float(scf_result.energy),
        localize=localize)


def ccm_dlpno_mp2(
    ccm,
    scf_result,
    *,
    cderi: Optional[np.ndarray] = None,
    localize: str = "none",
    tcut_pno: float = 0.0,
    tcut_mkn: float = 0.0,
    n_frozen: int = 0,
    max_iter: int = 100,
    conv_tol: float = 1e-10,
    damping: float = 1.0,
    lindep: float = 1e-8,
    ke_cutoff: float = 200.0,
) -> CCMDLPNOMP2Result:
    """DLPNO-MP2 on the cyclic cluster, on the neutral correlation reference.

    Parameters
    ----------
    ccm : CCMSystem
    scf_result : CCMSCFResult
        A converged CCM SCF on the **neutral** four-center (so its Fock /
        orbital energies match the neutral correlation reference). Build it with
        ``run_ccm_rhf(ccm, eri=ccm_eri_neutral(ccm))`` (small clusters) or the GDF
        route. ``mo_coeffs`` / ``mo_energies`` / ``fock`` / ``overlap`` (= S^CCM)
        / ``energy`` are consumed.
    cderi : ndarray, optional
        The neutral cderi ``L[P,mu,ν]`` (:func:`ccm_neutral_cderi`). Computed from
        ``ccm`` if omitted.
    localize : {"none", "pm", "boys"}
        Occupied gauge. ``"none"`` keeps the canonical occupieds (F diagonal ->
        the coupled residual converges in one step). ``"pm"`` / ``"boys"`` use the
        Task-1 localizer (PBC-safe Pipek-Mezey is the recommended DLPNO gauge).
    tcut_pno : float
        PNO occupation-number truncation threshold (0 -> keep all PNOs, the
        canonical limit).
    tcut_mkn : float
        Mulliken pair-domain threshold (0 -> full domains = all atoms, the
        canonical limit).
    n_frozen : int
        Frozen (core) occupied count.

    Returns
    -------
    CCMDLPNOMP2Result

    Notes
    -----
    No-truncation limit (``tcut_pno = tcut_mkn = 0``) reproduces the canonical
    neutral-control MP2 on the same neutral four-center to ~µHa -- the correctness gate
    (``tests/test_ccm_dlpno_mp2.py``).
    """
    b = _build_ccm_pno_pairs(
        ccm, scf_result, cderi=cderi, localize=localize, tcut_pno=tcut_pno,
        tcut_mkn=tcut_mkn, n_frozen=n_frozen, lindep=lindep, ke_cutoff=ke_cutoff)
    pairs, F_oo, S = b.pairs, b.F_oo, b.S
    n_act, nf, e_hf, e_pno_correction = b.n_act, b.nf, b.e_hf, b.e_pno_correction

    if not pairs:
        return CCMDLPNOMP2Result(
            e_hf=e_hf, e_corr=0.0, e_total=e_hf, e_corr_per_atom=0.0,
            e_pno_correction=0.0, n_pairs=0, n_occ_active=n_act,
            localize=localize, converged=True, n_iter=0,
            guess_selection=getattr(scf_result, "guess_selection", None),
            backend="aiccm2026dev-a-dlpno-mp2")

    # ----- coupled LMP2 residual (one step for canonical/diagonal F_oo) -----
    # R_ij = K_ij + (e_a+e_b)∘T_ij - S_k [F_ik.P(T_kj) + F_kj.P(T_ik)], with the
    # neighbour amplitudes projected into pair (i,j)'s PNO basis via S_pq.
    overlap_cache: dict = {}

    def S_pq(p, q):
        key = (p, q)
        if key not in overlap_cache:
            overlap_cache[key] = pairs[p].V.T @ S @ pairs[q].V
        return overlap_cache[key]

    def get_T(k, l):
        key = (k, l) if k <= l else (l, k)
        p = pairs.get(key)
        if p is None:
            return None
        return (p.T if k <= l else p.T.T), key

    converged = False
    n_iter = 0
    e_corr_prev = 0.0
    for iteration in range(max_iter):
        max_r = 0.0
        new_T = {}
        for (i, j), p in pairs.items():
            R = p.K + (p.eps[:, None] + p.eps[None, :]) * p.T
            for k in range(n_act):
                f_ik = float(F_oo[i, k])
                if abs(f_ik) > 1e-14:
                    got = get_T(k, j)
                    if got is not None:
                        T_kj, q = got
                        R -= f_ik * (T_kj if q == (i, j)
                                     else S_pq((i, j), q) @ T_kj @ S_pq((i, j), q).T)
                f_kj = float(F_oo[k, j])
                if abs(f_kj) > 1e-14:
                    got = get_T(i, k)
                    if got is not None:
                        T_ik, q = got
                        R -= f_kj * (T_ik if q == (i, j)
                                     else S_pq((i, j), q) @ T_ik @ S_pq((i, j), q).T)
            max_r = max(max_r, float(np.max(np.abs(R))))
            denom = (float(F_oo[i, i]) + float(F_oo[j, j])
                     - p.eps[:, None] - p.eps[None, :])
            new_T[(i, j)] = p.T + damping * R / denom
        for key, T in new_T.items():
            pairs[key].T = T
        for (i, j), p in pairs.items():
            if i == j:
                p.T = 0.5 * (p.T + p.T.T)
        e_corr_it = sum(_pair_energy(p.K, p.T, p.i, p.j) for p in pairs.values())
        n_iter = iteration + 1
        if iteration > 0 and abs(e_corr_it - e_corr_prev) < conv_tol and max_r < 1e-8:
            converged = True
            break
        e_corr_prev = e_corr_it

    e_corr_iterated = sum(_pair_energy(p.K, p.T, p.i, p.j) for p in pairs.values())
    e_corr = e_corr_iterated + e_pno_correction
    return CCMDLPNOMP2Result(
        e_hf=e_hf, e_corr=e_corr, e_total=e_hf + e_corr,
        e_corr_per_atom=e_corr / ccm.n_atoms,
        e_pno_correction=e_pno_correction, n_pairs=len(pairs),
        n_occ_active=n_act, localize=localize, converged=converged, n_iter=n_iter,
        pno_per_pair={(p.i + nf, p.j + nf): p.n_pno for p in pairs.values()},
        pair_energies={(p.i + nf, p.j + nf): _pair_energy(p.K, p.T, p.i, p.j)
                       for p in pairs.values()},
        guess_selection=getattr(scf_result, "guess_selection", None),
        backend="aiccm2026dev-a-dlpno-mp2")
