"""DLPNO-UCCSD(T) on the cyclic cluster (AICCM Task D, U-DLPNO).

The open-shell sibling of :func:`vibeqc.periodic.ccm.dlpno_ccsd.ccm_dlpno_ccsd`:
subspace-projected unrestricted coupled cluster on the **neutral**-reference
UHF-CCM wavefunction. The per-channel PNO pairs of
:func:`vibeqc.periodic.ccm.dlpno_ump2._build_uccm_pno_channels` are merged into a
**per-spin union PNO virtual space** (``V_trunc^α`` from the αα pairs + the α side
of the αβ pairs; ``V_trunc^β`` from the ββ pairs + the β side of the αβ pairs),
and the unrestricted CC amplitudes are solved *within* ``{occ^σ, V_trunc^σ}`` by
vibe-qc's own spin-orbital UCCSD(T) engine
(:func:`vibeqc.dlpno._ccsd_ref.run_ref_uccsd`) — the same engine
:func:`vibeqc.periodic.ccm.uccsd.run_ccm_uccsd` runs on the full virtual space.

Identity note: this routine uses the neutral fitted-torus correlation reference.
The legacy ``ccm_*`` API namespace assigns neither Γ-CCM nor χ-CCM construction
identity to it.

Why it is exact at no truncation
--------------------------------
The engine consumes diagonal α/β MO Fock matrices, so each spin's truncated MO
set must be ``[occ^σ | V_trunc^σ]`` with a block-diagonal Fock. The canonical
occupieds are Fock-diagonal; ``V_trunc^σ`` is built **semicanonical** (the Fock is
diagonalised within the union space) and is ``S^CCM``-orthonormal and orthogonal
to the occupieds (every PNO is a rotation of the canonical virtual MOs, which are
``S^CCM``-orthogonal to the occupieds). So the occ–virt Fock block vanishes and
``[occ^σ | V_trunc^σ]`` is orthonormal with a diagonal Fock.

At ``tcut_pno = 0`` each pair's PNOs span the full virtual space, so the union
spans it too: ``V_trunc^σ`` is then a unitary rotation of the canonical σ-virtuals
and UCCSD(T) — invariant to rotations within the occupied and virtual blocks —
equals the **canonical neutral-control UCCSD(T)** (:func:`run_ccm_uccsd`) on the
same neutral four-center (the hard correctness gate). PNO truncation shrinks
trades accuracy for cost.

A subtlety of the spin-orbital engine: it indexes α and β spatial MOs by a single
shared range ``[0, n_mo)``, so both spins must present the same total MO count.
At full rank both equal ``nbf``; when the per-spin truncated virtual spaces differ
in size the smaller is padded with extra (real, semicanonical) complement
virtuals — exact-safe, since enlarging a truncated space only reduces truncation.

**Reference = neutral** (per the ``-b``-critique correction; see
:mod:`vibeqc.periodic.ccm.dlpno`): the SCF reference and the four-center are the
neutral ``g_eff = Σ_P L⊗L`` (the bare-1/r kernel has no consistent cderi and the
Madelung background shifts the occ–virt denominators).

Reference: Riplinger, Sandhoefer, Hansen & Neese, J. Chem. Phys. 139, 134101
(2013) (DLPNO-CCSD(T)); Saitow, Becker, Riplinger, Valeev & Neese, J. Chem. Phys.
146, 164105 (2017), as open-shell DLPNO-CCSD background (the present
independent-UHF-space solver does not implement its single-spatial-orbital
NEV-PNO construction or SOMO-pair scale); Stanton et al., J. Chem. Phys. 94,
4334 (1991) (spin-orbital UCCSD).
"""

from __future__ import annotations

from .scf import _ccm_initial_guess

from dataclasses import dataclass
from typing import Optional

import numpy as np

from .uhf import CCMUHFResult, run_ccm_uhf

__all__ = ["CCMDLPNOUCCSDResult", "ccm_dlpno_uccsd"]


@dataclass
class CCMDLPNOUCCSDResult:
    """Result of :func:`ccm_dlpno_uccsd`."""

    e_hf: float
    e_ccsd_correlation: float           # UCCSD singles + doubles
    e_t: float                          # perturbative (T)
    e_correlation: float                # UCCSD + (T)
    e_total: float
    e_correlation_per_atom: float
    e_total_per_atom: float
    n_virtual_pno_alpha: int            # dim of V_trunc^α (pre-padding)
    n_virtual_pno_beta: int             # dim of V_trunc^β (pre-padding)
    n_virtual_full_alpha: int
    n_virtual_full_beta: int
    converged: bool
    n_iter: int
    avg_pao_alpha: float = 0.0
    avg_pao_beta: float = 0.0
    guess_selection: object = None
    # Per-call citation key; see CCMMP2Result.backend for why this is not a
    # class default. The DLPNO drivers ride the NEUTRAL cderi, so they are
    # the neutral lineage and never the bare four-centre one.
    backend: str = ""


def _union_semicanonical(blocks, ev, lindep):
    """Union the canonical-virtual-coord PNO blocks → semicanonical ``Q``.

    ``blocks`` is a list of ``(n_vir × n_pno)`` transforms within one spin's
    canonical virtual MO space (an ``S^CCM``-orthonormal, Fock-diagonal basis, so
    the metric is the identity here). Stack, canonical-orthogonalise (drop
    eigenvalues below ``lindep``), then diagonalise ``diag(ev)`` within the
    resulting space. Returns ``(Q, eps)`` with ``Q`` an ``(n_vir × n_trunc)``
    orthonormal transform and ``eps`` its semicanonical energies.
    """
    nv = int(ev.shape[0])
    if not blocks:
        return np.zeros((nv, 0)), np.zeros(0)
    W = np.hstack(blocks)
    G = 0.5 * (W.T @ W + (W.T @ W).T)
    s, U = np.linalg.eigh(G)
    keep = s > lindep
    Q = W @ (U[:, keep] / np.sqrt(s[keep]))
    Fv = 0.5 * (Q.T @ (ev[:, None] * Q) + (Q.T @ (ev[:, None] * Q)).T)
    eps, R = np.linalg.eigh(Fv)
    return Q @ R, eps


def _pad_virtual_to(Q, ev, n_target):
    """Extend orthonormal ``Q`` (n_vir × m) to n_vir × n_target, semicanonical.

    Appends ``n_target − m`` orthonormal complement virtuals (real canonical-
    virtual directions not already spanned by ``Q``) and re-diagonalises the Fock
    over the enlarged space. Exact-safe: the appended directions are genuine
    virtual orbitals, so the only effect is *less* truncation for that spin.
    """
    nv, m = int(ev.shape[0]), Q.shape[1]
    if n_target <= m:
        return Q, _semicanonical_energies(Q, ev)
    P = 0.5 * (np.eye(nv) - Q @ Q.T + (np.eye(nv) - Q @ Q.T).T)
    w, V = np.linalg.eigh(P)
    comp = V[:, w > 0.5]                       # orthonormal basis of the complement
    Qp = np.hstack([Q, comp[:, : n_target - m]])
    Fv = 0.5 * (Qp.T @ (ev[:, None] * Qp) + (Qp.T @ (ev[:, None] * Qp)).T)
    eps, R = np.linalg.eigh(Fv)
    return Qp @ R, eps


def _semicanonical_energies(Q, ev):
    """Diagonal Fock energies of the (already semicanonical) space ``Q``."""
    if Q.shape[1] == 0:
        return np.zeros(0)
    Fv = 0.5 * (Q.T @ (ev[:, None] * Q) + (Q.T @ (ev[:, None] * Q)).T)
    return np.linalg.eigvalsh(Fv)


def ccm_dlpno_uccsd(
    ccm,
    uhf_result: Optional[CCMUHFResult] = None,
    *, initial_guess: object = "AUTO",
    cderi: Optional[np.ndarray] = None,
    tcut_pno: float = 0.0,
    tcut_mkn: float = 0.0,
    n_frozen: int = 0,
    compute_triples: bool = True,
    pno_lindep: float = 1e-8,
    ke_cutoff: float = 200.0,
    max_iter: int = 128,
    conv_tol: float = 1e-9,
) -> CCMDLPNOUCCSDResult:
    """Open-shell DLPNO-UCCSD(T) on the cyclic cluster, on the neutral reference.

    Parameters
    ----------
    ccm : CCMSystem
    uhf_result : CCMUHFResult, optional
        Converged UHF-CCM reference on the **neutral** four-center. Built from
        ``ccm`` (neutral) if omitted.
    cderi : ndarray, optional
        Neutral cderi ``L[P,μ,ν]`` (:func:`ccm_neutral_cderi`); computed if omitted.
    tcut_pno : float
        PNO occupation-number truncation (0 → canonical limit, exact gate).
    tcut_mkn : float
        Mulliken population threshold for per-spin PAO pair domains. Zero
        bypasses PAO construction and retains the canonical virtual space.
    pno_lindep : float
        Linear-dependency threshold for both PAO redundancy removal and the
        per-spin union-PNO orthogonalization.
    n_frozen : int
        Frozen (core) occupied count, removed from both spins.
    compute_triples : bool
        Include the perturbative ``(T)`` correction.

    Returns
    -------
    CCMDLPNOUCCSDResult

    Notes
    -----
    No-truncation limit (``tcut_pno = 0``) reproduces the canonical neutral-control
    UCCSD(T) (:func:`run_ccm_uccsd`) on the same neutral four-center to ~µHa —
    the correctness gate (``tests/test_ccm_dlpno_uccsd.py``).
    """
    guess_selection = _ccm_initial_guess(
        ccm, initial_guess, driver='ccm_dlpno_uccsd', reference=uhf_result,
    )
    from ...dlpno._ccsd_ref import run_ref_uccsd
    from .dlpno_ump2 import _build_uccm_pno_channels
    from .neutral import ccm_neutral_cderi

    if tcut_pno < 0.0:
        raise ValueError("tcut_pno must be non-negative")
    if tcut_mkn < 0.0:
        raise ValueError("tcut_mkn must be non-negative")
    if pno_lindep < 0.0:
        raise ValueError("pno_lindep must be non-negative")

    L = np.asarray(cderi if cderi is not None else
                   ccm_neutral_cderi(ccm, ke_cutoff=ke_cutoff), dtype=float)
    if uhf_result is None:
        g = np.einsum("Pmn,Prs->mnrs", L, L, optimize=True)
        g = 0.5 * (g + np.transpose(g, (1, 0, 3, 2)))
        g = 0.5 * (g + np.transpose(g, (2, 3, 0, 1)))
        uhf_result = run_ccm_uhf(ccm, eri=g, initial_guess=initial_guess)
    if not getattr(uhf_result, "converged", True):
        raise ValueError("DLPNO-UCCSD-CCM requires a converged UHF-CCM reference.")

    b = _build_uccm_pno_channels(
        ccm, uhf_result, cderi=L, tcut_pno=tcut_pno,
        tcut_mkn=tcut_mkn, pao_lindep=pno_lindep,
        n_frozen=n_frozen, ke_cutoff=ke_cutoff)

    # Per-spin union PNO virtual spaces (canonical-virtual coordinates).
    a_blocks = ([p.Ua for p in b.aa_pairs] + [p.Ua for p in b.ab_pairs])
    b_blocks = ([p.Ua for p in b.bb_pairs] + [p.Ub for p in b.ab_pairs])
    Qa, eps_va = _union_semicanonical(a_blocks, b.eva, pno_lindep)
    Qb, eps_vb = _union_semicanonical(b_blocks, b.evb, pno_lindep)
    nvt_a, nvt_b = Qa.shape[1], Qb.shape[1]
    if nvt_a == 0 or nvt_b == 0:
        raise ValueError("no correlation pairs were built (empty active space).")

    # The spin-orbital engine indexes α/β spatial MOs by a single shared range,
    # so pad the smaller per-spin MO set to the common total.
    n_mo = max(b.na_act + nvt_a, b.nb_act + nvt_b)
    Qa, eps_va = _pad_virtual_to(Qa, b.eva, n_mo - b.na_act)
    Qb, eps_vb = _pad_virtual_to(Qb, b.evb, n_mo - b.nb_act)

    # Truncated MO sets [active occ^σ | V_trunc^σ] in the AO basis; diagonal Fock.
    C_sub_a = np.hstack([b.Coa, b.Cva @ Qa])
    C_sub_b = np.hstack([b.Cob, b.Cvb @ Qb])
    eps_a = np.concatenate([b.eoa, eps_va])
    eps_b = np.concatenate([b.eob, eps_vb])

    B_a = np.einsum("Pmn,mp,nq->Ppq", L, C_sub_a, C_sub_a, optimize=True)
    B_b = np.einsum("Pmn,mp,nq->Ppq", L, C_sub_b, C_sub_b, optimize=True)
    f_a = np.diag(eps_a)
    f_b = np.diag(eps_b)

    e_hf = b.e_hf
    ref = run_ref_uccsd(
        f_a, f_b, B_a, B_b, b.na_act, b.nb_act, e_hf=e_hf,
        max_iter=max_iter, conv_tol=conv_tol, compute_triples=compute_triples)

    e_ccsd = float(ref.e_corr)
    e_t = float(ref.e_t)
    e_corr = e_ccsd + e_t
    e_tot = e_hf + e_corr

    a_pao = [p.n_pao_a for p in b.aa_pairs + b.ab_pairs]
    b_pao = (
        [p.n_pao_a for p in b.bb_pairs]
        + [p.n_pao_b for p in b.ab_pairs]
    )
    _dlpno_label = "dlpno-uccsd(t)" if compute_triples else "dlpno-uccsd"
    return CCMDLPNOUCCSDResult(
        e_hf=e_hf, e_ccsd_correlation=e_ccsd, e_t=e_t, e_correlation=e_corr,
        e_total=e_tot, e_correlation_per_atom=e_corr / ccm.n_atoms,
        e_total_per_atom=e_tot / ccm.n_atoms,
        n_virtual_pno_alpha=nvt_a, n_virtual_pno_beta=nvt_b,
        n_virtual_full_alpha=int(b.eva.shape[0]),
        n_virtual_full_beta=int(b.evb.shape[0]),
        converged=bool(ref.converged), n_iter=int(ref.n_iter),
        avg_pao_alpha=float(np.mean(a_pao)) if a_pao else 0.0,
        avg_pao_beta=float(np.mean(b_pao)) if b_pao else 0.0,
        guess_selection=getattr(uhf_result, "guess_selection", None),
        backend=f"aiccm2026dev-a-{_dlpno_label}")
