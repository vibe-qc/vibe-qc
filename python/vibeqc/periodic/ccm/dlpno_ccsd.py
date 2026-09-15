"""DLPNO-CCSD(T) on the cyclic cluster (AICCM Task 2, M3).

Subspace-projected DLPNO-CCSD(T): the PNO pairs of :func:`~vibeqc.periodic.ccm.
dlpno._build_ccm_pno_pairs` are merged into a single **union PNO virtual space**
``V_trunc`` (the local virtual subspace that carries the correlation), and the
coupled-cluster amplitudes are solved *within* ``{occupied, V_trunc}`` using
vibe-qc's own CCM CCSD engine (:mod:`vibeqc.periodic.ccm.ccsd`). This is the
closed-shell analogue of the molecular DLPNO-CCSD pilot
(:func:`vibeqc.dlpno.ccsd.run_dlpno_ccsd_pilot`).

Identity note: this module uses the neutral fitted-torus correlation reference.
Its legacy ``ccm_*`` namespace does not make it the union-and-weight Γ-CCM
construction or the finite-character χ-CCM construction.

Why it is exact at no truncation
--------------------------------
The engine ``_ccsd_iterate`` / ``_triples`` assumes a **diagonal Fock** (canonical
or semicanonical orbitals). We therefore use the *canonical* occupieds (so the
occ-occ Fock block is diagonal) and a **semicanonical** ``V_trunc`` (diagonalize
the Fock within the union space). Because every PNO is built from PAOs projected
out of the full occupied space, ``V_trunc`` is ``S^CCM``-orthogonal to the
occupieds, so the occ-virt Fock block vanishes and the full MO set
``[C_occ | V_trunc]`` is orthonormal with a block-diagonal Fock.

At ``tcut_pno = tcut_mkn = 0`` each pair's PNOs already span the full virtual
space, so the union spans it too: ``V_trunc`` is then a unitary rotation of the
canonical virtuals, and CCSD(T) -- invariant to unitary rotations within the
occupied and virtual blocks -- equals the **canonical neutral-control CCSD(T)**
on the same neutral four-center (the hard correctness gate). PNO truncation shrinks
``V_trunc`` and trades accuracy for cost.

**Reference = neutral** (per the `-b`-critique correction; see
:mod:`vibeqc.periodic.ccm.dlpno`): the SCF reference and the four-center are the
neutral ``g_eff = S_P L⊗L``, because the bare-1/r Madelung background shifts the
occupied-virtual denominators.

Reference: Riplinger & Neese, J. Chem. Phys. 138, 034106 (2013); Riplinger,
Sandhoefer, Hansen, Neese, J. Chem. Phys. 139, 134101 (2013) (DLPNO-CCSD(T)).
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

import numpy as np

__all__ = ["CCMDLPNOCCSDResult", "ccm_dlpno_ccsd"]


@dataclass
class CCMDLPNOCCSDResult:
    """Result of :func:`ccm_dlpno_ccsd`."""

    e_hf: float
    e_ccsd_correlation: float
    e_t: float                          # perturbative (T)
    e_correlation: float                # CCSD + (T)
    e_total: float
    e_correlation_per_atom: float
    e_total_per_atom: float
    n_virtual_pno: int                  # dim of the truncated (union) virtual space
    n_virtual_full: int                 # full virtual dimension (nbf - n_occ)
    localize: str
    converged: bool
    n_iter: int
    guess_selection: object = None
    # Per-call citation key; see CCMMP2Result.backend for why this is not a
    # class default. The DLPNO drivers ride the NEUTRAL cderi, so they are
    # the neutral lineage and never the bare four-centre one.
    backend: str = ""


def _union_pno_virtual_space(pairs, S, F, *, lindep: float):
    """Merge the per-pair PNOs into one S^CCM-orthonormal, semicanonical space.

    Stack every pair's PNO coefficients ``V`` into ``W`` and
    canonical-orthogonalize against ``S^CCM`` (eigvecs of the Gram ``WᵀSW`` above
    ``lindep``); then diagonalize the Fock within the resulting space
    (semicanonical). Returns ``(V_trunc, eps_v)``.
    """
    W = np.hstack([p.V for p in pairs.values()])        # (nbf, S n_pno)
    G = W.T @ S @ W
    G = 0.5 * (G + G.T)
    s, U = np.linalg.eigh(G)
    keep = s > lindep
    V_union = W @ (U[:, keep] / np.sqrt(s[keep]))       # (nbf, n_vtrunc), SᵀV-orthonormal
    # Semicanonicalize (diagonal Fock within the union virtual space).
    Fv = V_union.T @ F @ V_union
    Fv = 0.5 * (Fv + Fv.T)
    eps_v, R = np.linalg.eigh(Fv)
    return V_union @ R, eps_v


def ccm_dlpno_ccsd(
    ccm,
    scf_result,
    *,
    cderi: Optional[np.ndarray] = None,
    g_neutral: Optional[np.ndarray] = None,
    localize: str = "pm",
    tcut_pno: float = 0.0,
    tcut_mkn: float = 0.0,
    n_frozen: int = 0,
    compute_triples: bool = True,
    pno_lindep: float = 1e-8,
    lindep: float = 1e-8,
    ke_cutoff: float = 200.0,
    max_iter: int = 128,
    conv_tol: float = 1e-9,
) -> CCMDLPNOCCSDResult:
    """DLPNO-CCSD(T) on the cyclic cluster, on the neutral correlation reference.

    Parameters
    ----------
    ccm : CCMSystem
    scf_result : CCMSCFResult
        Converged CCM SCF on the **neutral** four-center (e.g.
        ``run_ccm_rhf(ccm, eri=ccm_eri_neutral(ccm))``).
    cderi : ndarray, optional
        Neutral cderi ``L[P,mu,ν]`` for the PNO pair build (computed if omitted).
    g_neutral : ndarray, optional
        Neutral four-center ``g_eff[muν,rs]`` for the CCSD MO transform. Built from
        ``cderi`` (``S_P L⊗L``) if omitted.
    localize : {"none", "pm", "boys"}
        Occupied gauge for the *PNO domains* (the CCSD itself uses the canonical
        occupieds). ``"pm"`` (Pipek-Mezey) is the recommended local gauge.
    tcut_pno, tcut_mkn : float
        PNO occupation and Mulliken domain thresholds (0 -> canonical limit).
    compute_triples : bool
        Include the perturbative ``(T)`` correction.

    Returns
    -------
    CCMDLPNOCCSDResult

    Notes
    -----
    No-truncation limit (``tcut_pno = tcut_mkn = 0``) reproduces the canonical
    neutral-control CCSD(T) on the same neutral four-center to ~µHa
    (``tests/test_ccm_dlpno_ccsd.py``).
    """
    from .ccsd import _ccsd_iterate, _spin_block_eri_phys, _triples
    from .dlpno import _build_ccm_pno_pairs, _occ
    from .neutral import ccm_eri_neutral, ccm_neutral_cderi

    if not getattr(scf_result, "converged", True):
        raise ValueError("DLPNO-CCSD-CCM requires a converged SCF reference.")

    S = np.asarray(scf_result.overlap, dtype=float)
    F = np.asarray(scf_result.fock, dtype=float)
    if np.iscomplexobj(F):
        F = np.real_if_close(F, tol=1000).real

    n_elec = ccm.supercell.n_electrons()
    if n_elec % 2 != 0:
        raise ValueError("ccm_dlpno_ccsd is closed-shell; use an even-electron cluster.")
    n_occ = n_elec // 2
    nbf = int(ccm.nbf)

    # cderi (for the pair build) and g (for the CCSD MO transform), both neutral.
    L = np.asarray(cderi if cderi is not None else
                   ccm_neutral_cderi(ccm, ke_cutoff=ke_cutoff), dtype=float)
    if g_neutral is None:
        g = np.einsum("Pmn,Prs->mnrs", L, L, optimize=True)
        g = 0.5 * (g + np.transpose(g, (1, 0, 3, 2)))
        g = 0.5 * (g + np.transpose(g, (2, 3, 0, 1)))
    else:
        g = np.asarray(g_neutral, dtype=float)

    # 1) per-pair PNOs (locality from the chosen occupied gauge).
    bundle = _build_ccm_pno_pairs(
        ccm, scf_result, cderi=L, localize=localize, tcut_pno=tcut_pno,
        tcut_mkn=tcut_mkn, n_frozen=n_frozen, lindep=lindep, ke_cutoff=ke_cutoff)
    nf = bundle.nf

    # 2) union PNO virtual space (S^CCM-orthonormal, semicanonical).
    if not bundle.pairs:
        raise ValueError("no correlation pairs were built (empty active space).")
    V_trunc, eps_v = _union_pno_virtual_space(bundle.pairs, S, F, lindep=pno_lindep)
    n_vtrunc = V_trunc.shape[1]

    # 3) MO set [frozen-core | canonical active-occ | V_trunc]. The CCSD engine
    # assumes a diagonal Fock; canonical occupieds + semicanonical V_trunc give a
    # block-diagonal Fock (occ-virt block vanishes since PNOs ⊥ occupied space).
    C_all = _occ(scf_result, n_occ)                     # canonical occupieds (incl. core)
    eps_all = np.asarray(scf_result.mo_energies, dtype=float)
    C_sub = np.hstack([C_all, V_trunc])                 # (nbf, n_occ + n_vtrunc)
    eps_sub = np.concatenate([eps_all[:n_occ], eps_v])

    # 4) transform the neutral four-center into the (occ ⊕ V_trunc) MO basis.
    mo = np.einsum("mnls,mp,nq,lr,st->pqrt", g, C_sub, C_sub, C_sub, C_sub, optimize=True)
    spinints = _spin_block_eri_phys(mo)
    fs = np.diag(np.repeat(eps_sub, 2))
    nocc_so = n_elec                                    # 2 . n_occ (frozen core handled below)

    # Frozen core: drop the lowest nf spatial occupieds from the active CCSD.
    if nf > 0:
        keep = np.concatenate([np.arange(2 * nf, 2 * n_occ),
                               np.arange(2 * n_occ, 2 * (n_occ + n_vtrunc))])
        spinints = spinints[np.ix_(keep, keep, keep, keep)]
        fs = fs[np.ix_(keep, keep)]
        nocc_so = 2 * (n_occ - nf)

    e_ccsd, ts, td, conv, n_iter = _ccsd_iterate(
        spinints, fs, nocc_so, max_iter=max_iter, conv_tol=conv_tol)
    e_t = _triples(spinints, fs, ts, td, nocc_so) if compute_triples else 0.0

    e_corr = e_ccsd + e_t
    e_hf = float(scf_result.energy)
    e_tot = e_hf + e_corr
    # Plain local, not an inline expression: tests/test_citations.py resolves
    # backend= labels out of the AST and understands exactly this shape.
    _dlpno_label = "dlpno-ccsd(t)" if compute_triples else "dlpno-ccsd"
    return CCMDLPNOCCSDResult(
        e_hf=e_hf, e_ccsd_correlation=e_ccsd, e_t=e_t, e_correlation=e_corr,
        e_total=e_tot, e_correlation_per_atom=e_corr / ccm.n_atoms,
        e_total_per_atom=e_tot / ccm.n_atoms,
        n_virtual_pno=n_vtrunc, n_virtual_full=nbf - n_occ,
        localize=localize, converged=conv, n_iter=n_iter,
        guess_selection=getattr(scf_result, "guess_selection", None),
        backend=f"aiccm2026dev-a-{_dlpno_label}")
