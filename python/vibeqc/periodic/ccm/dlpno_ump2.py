"""Open-shell DLPNO-UMP2 on the cyclic cluster (AICCM Task D, U-DLPNO).

The unrestricted sibling of :func:`vibeqc.periodic.ccm.dlpno.ccm_dlpno_mp2`: local
DLPNO second-order Møller–Plesset on the **neutral**-reference UHF-CCM wavefunction
(:func:`vibeqc.periodic.ccm.uhf.run_ccm_uhf`).

Identity note: this routine is a neutral fitted-torus correlation control. The
legacy ``ccm_*`` API name does not classify it as the union-and-weight Γ-CCM
construction or the finite-character χ-CCM construction.

The correlation energy is resolved into the three spin channels of unrestricted
MP2 (Szabo & Ostlund Sec. 6.7, the same decomposition as
:func:`vibeqc.periodic.ccm.ump2.run_ccm_ump2`):

    E_aa = Σ_{i<j∈α} Σ_ab ½ |⟨ij||ab⟩|² / D      (same-spin α)
    E_bb = Σ_{i<j∈β} ...                          (same-spin β)
    E_ab = Σ_{i∈α, j∈β} Σ_ab (ia|jb)² / D        (opposite-spin)

with ``⟨ij||ab⟩ = (ia|jb) − (ib|ja)`` antisymmetrised for the same-spin channels
and a Coulomb-only opposite-spin channel, every ``(ia|jb)`` density-fit from the
neutral cyclic-cluster cderi (``g_eff = Σ_P L⊗L``). Each pair carries its own
**pair natural orbitals** (PNOs), built from the pair density and truncated by
occupation number (``tcut_pno``); same-spin pairs need a single PNO set, the
opposite-spin pairs a separate α-PNO and β-PNO set. With ``tcut_pno = 0`` the
PNOs span the full virtual space and the energy reproduces the canonical
neutral-control UMP2 to machine precision — the correctness gate
(``tests/test_ccm_dlpno_ump2.py``).

**Correlation reference = neutral** (per the ``-b``-critique correction; see
:mod:`vibeqc.periodic.ccm.dlpno` and ``docs/aiccm2026dev_a_followon.md`` Sec. 2.0):
the ionic Madelung background shifts the occupied–virtual *denominators*, so the
neutral four-center is the required reference for CCM correlation. Both the SCF
reference and the integrals are the neutral kernel here.

With ``tcut_mkn > 0``, each occupied pair first selects atoms from its
``S^CCM`` Mulliken populations and builds separate α/β semicanonical projected
atomic-orbital (PAO) spaces. Pair densities and PNOs are formed inside those
bounded spaces. The resulting PNO transforms are mapped back to canonical
virtual coordinates, preserving the shared UCCSD union-space contract.
``tcut_mkn = 0`` bypasses PAO construction exactly and retains the canonical
virtual M1 correctness path bit for bit.

References: Riplinger & Neese, J. Chem. Phys. 138, 034106 (2013), and Pinski &
Neese, J. Chem. Phys. 150, 164102 (2019), for DLPNO-MP2; Peintinger & Bredow,
J. Comput. Chem. 35, 839 (2014), for CCM. The unrestricted alpha/beta
spin-channel extension is a vibe-qc implementation with independent UHF
orbital and PNO spaces.
"""

from __future__ import annotations

from .scf import _ccm_initial_guess

from dataclasses import dataclass, field
from typing import Optional

import numpy as np

from .uhf import CCMUHFResult, run_ccm_uhf

__all__ = ["CCMDLPNOUMP2Result", "ccm_dlpno_ump2"]


@dataclass
class _UPNOPair:
    """One correlation pair's PNO data, in *canonical-virtual* coordinates.

    ``Ua`` / ``Ub`` are the PNO transforms (n_vir × n_pno) within the canonical
    virtual MO space of the relevant spin(s); ``e_pair`` is the truncated pair
    energy. Same-spin pairs carry one set in ``Ua`` (``Ub`` is ``None``);
    opposite-spin pairs carry both.
    """

    i: int
    j: int
    Ua: np.ndarray
    Ub: Optional[np.ndarray]
    e_pair: float
    n_pao_a: int
    n_pao_b: int


@dataclass
class _UPNOBundle:
    """Per-channel PNO pairs of the open-shell cyclic-cluster build.

    Shared by :func:`ccm_dlpno_ump2` and :func:`~vibeqc.periodic.ccm.dlpno_uccsd.
    ccm_dlpno_uccsd`. The PNO transforms live in canonical-virtual coordinates
    (the canonical α/β virtual MOs are ``S^CCM``-orthonormal, so that metric is
    the identity); ``Cva`` / ``Cvb`` map them back to the AO basis.
    """

    e_aa: float
    e_bb: float
    e_ab: float
    aa_pairs: list           # list[_UPNOPair]  (Ua only)
    bb_pairs: list           # list[_UPNOPair]  (Ua only, β space)
    ab_pairs: list           # list[_UPNOPair]  (Ua = α, Ub = β)
    Coa: np.ndarray          # active α occupied MO coeffs (nbf × na_act)
    Cob: np.ndarray          # active β occupied MO coeffs (nbf × nb_act)
    Cva: np.ndarray          # α virtual MO coeffs (nbf × nva)
    Cvb: np.ndarray          # β virtual MO coeffs (nbf × nvb)
    eoa: np.ndarray          # active α occupied energies
    eob: np.ndarray
    eva: np.ndarray          # α virtual energies
    evb: np.ndarray
    na_act: int
    nb_act: int
    nf: int
    e_hf: float


def _channel_b_tensor(L: np.ndarray, Co: np.ndarray, Cv: np.ndarray) -> np.ndarray:
    """Neutral-cderi B-tensor ``B[P,i,a] = Σ_μν C_o[μ,i] L[P,μ,ν] C_v[ν,a]``."""
    return np.einsum("mi,Pmn,na->Pia", Co, L, Cv, optimize=True)


def _same_spin_pairs(B, eo, ev, tcut, pao_basis=None):
    """Same-spin (σσ) PNO pairs + channel energy. ``B`` is (naux, n_occ, n_vir).

    Mirrors :func:`vibeqc.dlpno.ump2._same_spin_channel` (one PNO set per pair,
    antisymmetrised exchange, ½ weight) but also returns the per-pair PNO
    transform for the UCCSD union. Sums to the canonical neutral-control UMP2 same-spin
    channel at ``tcut = 0``.
    """
    from ...dlpno.ump2 import _pnos

    no = B.shape[1]
    e = 0.0
    pairs: list = []
    for i in range(no):
        Bi = B[:, i, :]
        for j in range(i + 1, no):
            K = Bi.T @ B[:, j, :]
            if pao_basis is None:
                W, ev_pair = None, ev
                K_pair = K
            else:
                W, ev_pair = pao_basis(i, j)
                K_pair = W.T @ K @ W
            anti = K_pair - K_pair.T            # ⟨ij||ab⟩ = (ia|jb) − (ib|ja)
            denom = eo[i] + eo[j] - ev_pair[:, None] - ev_pair[None, :]
            T = anti / denom
            U_pair, eps_pno = _pnos(
                T @ T.T + T.T @ T, ev_pair, tcut
            )
            Kp = U_pair.T @ K_pair @ U_pair
            antip = Kp - Kp.T
            dp = eo[i] + eo[j] - eps_pno[:, None] - eps_pno[None, :]
            Tp = antip / dp
            e_pair = 0.5 * float(np.sum(antip * Tp))
            e += e_pair
            U = U_pair if W is None else W @ U_pair
            pairs.append(_UPNOPair(
                i=i, j=j, Ua=U, Ub=None, e_pair=e_pair,
                n_pao_a=int(ev_pair.size), n_pao_b=0,
            ))
    return e, pairs


def _opp_spin_pairs(Ba, eao, eva, Bb, ebo, evb, tcut, pao_basis=None):
    """Opposite-spin (αβ) PNO pairs + channel energy.

    Mirrors :func:`vibeqc.dlpno.ump2._opp_spin_channel`: α(ia) × β(jb), Coulomb
    only, separate α-PNO and β-PNO set per pair. Sums to the canonical
    neutral-control UMP2
    opposite-spin channel at ``tcut = 0``.
    """
    from ...dlpno.ump2 import _pnos

    e = 0.0
    pairs: list = []
    for i in range(Ba.shape[1]):
        Bai = Ba[:, i, :]
        for j in range(Bb.shape[1]):
            K = Bai.T @ Bb[:, j, :]             # (nva, nvb) = (ia|jb)
            if pao_basis is None:
                Wa, Wb = None, None
                eva_pair, evb_pair = eva, evb
                K_pair = K
            else:
                Wa, eva_pair, Wb, evb_pair = pao_basis(i, j)
                K_pair = Wa.T @ K @ Wb
            denom = (
                eao[i] + ebo[j]
                - eva_pair[:, None] - evb_pair[None, :]
            )
            T = K_pair / denom
            Ua_pair, ea_pno = _pnos(
                T @ T.T, eva_pair, tcut
            )                                             # α PNOs
            Ub_pair, eb_pno = _pnos(
                T.T @ T, evb_pair, tcut
            )                                             # β PNOs
            Kp = Ua_pair.T @ K_pair @ Ub_pair
            dp = eao[i] + ebo[j] - ea_pno[:, None] - eb_pno[None, :]
            Tp = Kp / dp
            e_pair = float(np.sum(Kp * Tp))
            e += e_pair
            Ua = Ua_pair if Wa is None else Wa @ Ua_pair
            Ub = Ub_pair if Wb is None else Wb @ Ub_pair
            pairs.append(_UPNOPair(
                i=i, j=j, Ua=Ua, Ub=Ub, e_pair=e_pair,
                n_pao_a=int(eva_pair.size), n_pao_b=int(evb_pair.size),
            ))
    return e, pairs


def _build_pao_pair_bases(
    ccm, uhf_result, Coa, Cob, Cva, Cvb, *, tcut_mkn, lindep,
):
    """Return per-channel semicanonical PAO builders.

    The alpha and beta projectors remove each spin's complete occupied space.
    Opposite-spin domains use the union of the atoms selected independently by
    the alpha and beta occupied orbitals.
    """
    from ...dlpno.pao import (
        build_atom_basis_map,
        build_projection_matrix,
        select_domain_atoms_mulliken,
        semicanonical_pao_basis,
    )

    S = np.asarray(uhf_result.overlap, dtype=float)
    Ca = np.asarray(uhf_result.mo_coeffs_alpha, dtype=float)
    Cb = np.asarray(uhf_result.mo_coeffs_beta, dtype=float)
    ea = np.asarray(uhf_result.mo_energies_alpha, dtype=float)
    eb = np.asarray(uhf_result.mo_energies_beta, dtype=float)
    na, nb = int(uhf_result.n_alpha), int(uhf_result.n_beta)
    Qa = build_projection_matrix(Ca[:, :na], S)
    Qb = build_projection_matrix(Cb[:, :nb], S)
    atom_first, _ = build_atom_basis_map(ccm.supercell, ccm.basis)

    # From F C = S C eps and C.T S C = I:
    # F = S C diag(eps) C.T S for a complete generalized eigenbasis.
    Fa = S @ (Ca * ea) @ Ca.T @ S
    Fb = S @ (Cb * eb) @ Cb.T @ S
    Fa = 0.5 * (Fa + Fa.T)
    Fb = 0.5 * (Fb + Fb.T)

    def _ao_indices(atoms, pair):
        if atoms.size == 0:
            raise ValueError(
                f"empty PAO atom domain for occupied pair {pair} at "
                f"tcut_mkn={tcut_mkn}"
            )
        return np.concatenate([
            np.arange(atom_first[a], atom_first[a + 1], dtype=int)
            for a in atoms
        ])

    def _spin_basis(C_occ, Cvir, F, Q, i, j, label):
        atoms = select_domain_atoms_mulliken(
            C_occ, S, atom_first, i, j, tcut_mkn
        )
        V, eps = semicanonical_pao_basis(
            F, S, Q, _ao_indices(atoms, (label, i, j)), lindep
        )
        if eps.size == 0:
            raise ValueError(
                f"empty {label}-spin PAO virtual domain for occupied pair "
                f"{(i, j)} at tcut_mkn={tcut_mkn}"
            )
        return np.ascontiguousarray(Cvir.T @ S @ V), eps

    def aa(i, j):
        return _spin_basis(Coa, Cva, Fa, Qa, i, j, "alpha")

    def bb(i, j):
        return _spin_basis(Cob, Cvb, Fb, Qb, i, j, "beta")

    def ab(i, j):
        atoms_a = select_domain_atoms_mulliken(
            Coa, S, atom_first, i, i, tcut_mkn
        )
        atoms_b = select_domain_atoms_mulliken(
            Cob, S, atom_first, j, j, tcut_mkn
        )
        atoms = np.union1d(atoms_a, atoms_b)
        ao = _ao_indices(atoms, ("alpha-beta", i, j))
        Va, epa = semicanonical_pao_basis(Fa, S, Qa, ao, lindep)
        Vb, epb = semicanonical_pao_basis(Fb, S, Qb, ao, lindep)
        if epa.size == 0 or epb.size == 0:
            raise ValueError(
                f"empty alpha/beta PAO virtual domain for occupied pair "
                f"{(i, j)} at tcut_mkn={tcut_mkn}"
            )
        Wa = np.ascontiguousarray(Cva.T @ S @ Va)
        Wb = np.ascontiguousarray(Cvb.T @ S @ Vb)
        return Wa, epa, Wb, epb

    return aa, bb, ab


def _build_uccm_pno_channels(
    ccm, uhf_result, *, cderi, tcut_pno, tcut_mkn, pao_lindep,
    n_frozen, ke_cutoff,
) -> _UPNOBundle:
    """Build the αα / ββ / αβ canonical-virtual PNO pairs on the neutral reference.

    Shared by the open-shell DLPNO-UMP2 and DLPNO-UCCSD(T) drivers (the open-shell
    analogue of :func:`vibeqc.periodic.ccm.dlpno._build_ccm_pno_pairs`).
    """
    from .neutral import ccm_neutral_cderi

    Ca = np.asarray(uhf_result.mo_coeffs_alpha, dtype=float)
    Cb = np.asarray(uhf_result.mo_coeffs_beta, dtype=float)
    ea = np.asarray(uhf_result.mo_energies_alpha, dtype=float)
    eb = np.asarray(uhf_result.mo_energies_beta, dtype=float)
    na, nb = int(uhf_result.n_alpha), int(uhf_result.n_beta)

    nf = int(n_frozen)
    if nf < 0 or nf >= max(nb, 1):
        raise ValueError(f"n_frozen={nf} out of range for n_beta={nb}")

    Coa, Cva, eoa, eva = Ca[:, nf:na], Ca[:, na:], ea[nf:na], ea[na:]
    Cob, Cvb, eob, evb = Cb[:, nf:nb], Cb[:, nb:], eb[nf:nb], eb[nb:]

    L = np.asarray(cderi if cderi is not None else
                   ccm_neutral_cderi(ccm, ke_cutoff=ke_cutoff), dtype=float)
    Ba = _channel_b_tensor(L, Coa, Cva)
    Bb = _channel_b_tensor(L, Cob, Cvb)

    aa_pao = bb_pao = ab_pao = None
    if tcut_mkn > 0.0:
        aa_pao, bb_pao, ab_pao = _build_pao_pair_bases(
            ccm, uhf_result, Coa, Cob, Cva, Cvb,
            tcut_mkn=tcut_mkn, lindep=pao_lindep,
        )

    e_aa, aa_pairs = _same_spin_pairs(
        Ba, eoa, eva, tcut_pno, aa_pao
    )
    e_bb, bb_pairs = _same_spin_pairs(
        Bb, eob, evb, tcut_pno, bb_pao
    )
    e_ab, ab_pairs = _opp_spin_pairs(
        Ba, eoa, eva, Bb, eob, evb, tcut_pno, ab_pao
    )

    return _UPNOBundle(
        e_aa=e_aa, e_bb=e_bb, e_ab=e_ab,
        aa_pairs=aa_pairs, bb_pairs=bb_pairs, ab_pairs=ab_pairs,
        Coa=Coa, Cob=Cob, Cva=Cva, Cvb=Cvb,
        eoa=eoa, eob=eob, eva=eva, evb=evb,
        na_act=na - nf, nb_act=nb - nf, nf=nf, e_hf=float(uhf_result.energy))


@dataclass
class CCMDLPNOUMP2Result:
    """Result of :func:`ccm_dlpno_ump2`.

    ``e_corr = os_scale·e_ab + ss_scale·(e_aa + e_bb)``. The per-channel energies
    are the truncated (DLPNO) values; with ``tcut_pno = 0`` they equal the
    canonical neutral-control UMP2 channels.
    """

    e_hf: float
    e_corr: float
    e_aa: float
    e_bb: float
    e_ab: float
    e_total: float
    e_corr_per_atom: float
    e_total_per_atom: float
    n_frozen: int
    n_pairs_aa: int
    n_pairs_bb: int
    n_pairs_ab: int
    avg_pno_aa: float = 0.0
    avg_pno_bb: float = 0.0
    avg_pno_ab: float = 0.0
    avg_pao_aa: float = 0.0
    avg_pao_bb: float = 0.0
    avg_pao_ab: float = 0.0
    pair_energies: dict = field(default_factory=dict)
    pao_per_pair: dict = field(default_factory=dict)
    guess_selection: object = None
    # Per-call citation key; see CCMMP2Result.backend for why this is not a
    # class default. The DLPNO drivers ride the NEUTRAL cderi, so they are
    # the neutral lineage and never the bare four-centre one.
    backend: str = ""


def ccm_dlpno_ump2(
    ccm,
    uhf_result: Optional[CCMUHFResult] = None,
    *, initial_guess: object = "AUTO",
    cderi: Optional[np.ndarray] = None,
    tcut_pno: float = 0.0,
    tcut_mkn: float = 0.0,
    pao_lindep: float = 1e-8,
    n_frozen: int = 0,
    ss_scale: float = 1.0,
    os_scale: float = 1.0,
    ke_cutoff: float = 200.0,
) -> CCMDLPNOUMP2Result:
    """Open-shell DLPNO-UMP2 on the cyclic cluster, on the neutral reference.

    Parameters
    ----------
    ccm : CCMSystem
    uhf_result : CCMUHFResult, optional
        Converged UHF-CCM reference on the **neutral** four-center (so its Fock /
        orbital energies match the neutral correlation reference). Build it with
        ``run_ccm_uhf(ccm, eri=ccm_eri_neutral(ccm))``; constructed from ``ccm``
        (neutral) if omitted.
    cderi : ndarray, optional
        Neutral cderi ``L[P,μ,ν]`` (:func:`ccm_neutral_cderi`); computed if omitted.
    tcut_pno : float
        PNO occupation-number truncation threshold (0 → keep all PNOs, the
        canonical limit). Applies to every channel.
    tcut_mkn : float
        Mulliken population threshold for per-spin PAO pair domains. Zero
        bypasses PAO construction and retains the canonical virtual space.
    pao_lindep : float
        PAO overlap-eigenvalue threshold for removing redundant projected AOs.
    n_frozen : int
        Frozen (core) occupied count, removed from both spins.
    ss_scale, os_scale : float
        Same-spin / opposite-spin scaling (1, 1 = plain UMP2; 6/5, 1/3 =
        SCS-UMP2). Applied to the reported ``e_corr`` only.

    Returns
    -------
    CCMDLPNOUMP2Result

    Notes
    -----
    No-truncation limit (``tcut_pno = 0``, ``ss_scale = os_scale = 1``) reproduces
    the canonical neutral-control UMP2 (:func:`run_ccm_ump2`) on the same neutral
    four-center to machine ε — the correctness gate
    (``tests/test_ccm_dlpno_ump2.py``).
    """
    guess_selection = _ccm_initial_guess(
        ccm, initial_guess, driver='ccm_dlpno_ump2', reference=uhf_result,
    )
    from .neutral import ccm_neutral_cderi

    if tcut_pno < 0.0:
        raise ValueError("tcut_pno must be non-negative")
    if tcut_mkn < 0.0:
        raise ValueError("tcut_mkn must be non-negative")
    if pao_lindep < 0.0:
        raise ValueError("pao_lindep must be non-negative")

    L = np.asarray(cderi if cderi is not None else
                   ccm_neutral_cderi(ccm, ke_cutoff=ke_cutoff), dtype=float)
    if uhf_result is None:
        g = np.einsum("Pmn,Prs->mnrs", L, L, optimize=True)
        g = 0.5 * (g + np.transpose(g, (1, 0, 3, 2)))
        g = 0.5 * (g + np.transpose(g, (2, 3, 0, 1)))
        uhf_result = run_ccm_uhf(ccm, eri=g, initial_guess=initial_guess)
    if not uhf_result.converged:
        raise ValueError(
            "DLPNO-UMP2-CCM requires a converged UHF-CCM reference; SCF did not converge.")

    b = _build_uccm_pno_channels(
        ccm, uhf_result, cderi=L, tcut_pno=tcut_pno,
        tcut_mkn=tcut_mkn, pao_lindep=pao_lindep,
        n_frozen=n_frozen, ke_cutoff=ke_cutoff)

    e_corr = os_scale * b.e_ab + ss_scale * (b.e_aa + b.e_bb)
    e_hf = b.e_hf
    e_tot = e_hf + e_corr

    def _avg(pairs):
        if not pairs:
            return 0.0
        return float(np.mean([p.Ua.shape[1] if p.Ub is None
                              else 0.5 * (p.Ua.shape[1] + p.Ub.shape[1])
                              for p in pairs]))

    pair_e = {("aa", p.i + b.nf, p.j + b.nf): p.e_pair for p in b.aa_pairs}
    pair_e.update({("bb", p.i + b.nf, p.j + b.nf): p.e_pair for p in b.bb_pairs})
    pair_e.update({("ab", p.i + b.nf, p.j + b.nf): p.e_pair for p in b.ab_pairs})
    pao_per_pair = {
        ("aa", p.i + b.nf, p.j + b.nf): p.n_pao_a for p in b.aa_pairs
    }
    pao_per_pair.update({
        ("bb", p.i + b.nf, p.j + b.nf): p.n_pao_a for p in b.bb_pairs
    })
    pao_per_pair.update({
        ("ab", p.i + b.nf, p.j + b.nf): (p.n_pao_a, p.n_pao_b)
        for p in b.ab_pairs
    })

    def _avg_pao(pairs):
        if not pairs:
            return 0.0
        return float(np.mean([
            p.n_pao_a if p.n_pao_b == 0
            else 0.5 * (p.n_pao_a + p.n_pao_b)
            for p in pairs
        ]))

    return CCMDLPNOUMP2Result(
        e_hf=e_hf, e_corr=e_corr, e_aa=b.e_aa, e_bb=b.e_bb, e_ab=b.e_ab,
        e_total=e_tot, e_corr_per_atom=e_corr / ccm.n_atoms,
        e_total_per_atom=e_tot / ccm.n_atoms, n_frozen=b.nf,
        n_pairs_aa=len(b.aa_pairs), n_pairs_bb=len(b.bb_pairs),
        n_pairs_ab=len(b.ab_pairs),
        avg_pno_aa=_avg(b.aa_pairs), avg_pno_bb=_avg(b.bb_pairs),
        avg_pno_ab=_avg(b.ab_pairs),
        avg_pao_aa=_avg_pao(b.aa_pairs),
        avg_pao_bb=_avg_pao(b.bb_pairs),
        avg_pao_ab=_avg_pao(b.ab_pairs),
        pair_energies=pair_e, pao_per_pair=pao_per_pair,
        guess_selection=getattr(uhf_result, "guess_selection", None),
        backend="aiccm2026dev-a-dlpno-ump2")
