"""UMP2-CCM -- open-shell second-order Moller-Plesset on a Cyclic Cluster Model
reference (AICCM open-shell correlation).

The unrestricted sibling of :func:`vibeqc.periodic.ccm.mp2.run_ccm_mp2`: it takes
the spin-resolved UHF-CCM reference (:func:`vibeqc.periodic.ccm.uhf.run_ccm_uhf`)
and the (spin-independent) CCM-weighted effective ERI tensor, transforms the
occupied-virtual blocks to the a and b MO bases, and evaluates the spin-blocked
UMP2 correlation energy. As with the closed-shell driver the *algebra* is the
molecular one -- only the AO ERI source is the CCM-weighted tensor -- so vibe-qc's
C++ ``run_ump2`` (which recomputes canonical integrals) cannot be reused; the
energy expression is evaluated here.

    # Spin-blocked UMP2 (e.g. Szabo & Ostlund Sec.6.7.1), every ERI the CCM one:
    #   E_ss = 1/4 S_{ijabins} [(ia|jb) - (ib|ja)]^2 / (e_i+e_j-e_a-e_b)   (s=a,b)
    #   E_ab = S_{i,aina; j,binb} (ia|jb)^2 / (e_i^a+e_j^b-e_a^a-e_b^b)
    #   E_corr = E_aa + E_bb + E_ab
    # (ia|jb) in Mulliken order, transformed in the respective spin MO basis.

Reduces to the closed-shell :func:`run_ccm_mp2` for a closed shell, and the
isolated-cell limit reproduces vibe-qc's molecular ``run_ump2`` exactly (the CCM
ERI is the molecular ERI there). Small / low-dimensional clusters only (dense
``n_ref_ao**4`` tensor). Reference: Peintinger & Bredow, J. Comput. Chem. 35, 839
(2014), doi:10.1002/jcc.23550.
"""

from __future__ import annotations

from .scf import _ccm_initial_guess

from dataclasses import dataclass

import numpy as np

from .scf import _ccm_eri_for_method
from .uhf import CCMUHFResult, run_ccm_uhf

__all__ = ["CCMUMP2Result", "run_ccm_ump2"]


@dataclass
class CCMUMP2Result:
    # MEASURED 2026-09-08, not read: these are TOTAL finite cyclic-cluster
    # energies, like every other driver in this package -- H2/STO-3G in a
    # 6-bohr cube gives e_correlation -0.0131578701 at nrep=(1,1,1) and
    # -0.0263506898 at (2,1,1), a clean doubling. The comments here used to
    # say "per reference cell", the identical error that CCMKSResult and
    # CCMUHFResult carried (see periodic/ccm/four_center_runner.py's module
    # docstring). Divide by ``ccm.n_cells`` to compare with a primitive-cell
    # method; the runner adapter does.
    e_correlation: float           # UMP2 correlation energy, TOTAL cyclic (Ha)
    e_hf: float                    # UHF-CCM energy, TOTAL cyclic (Ha)
    e_total: float                 # e_hf + e_correlation
    e_aa: float                    # same-spin aa component
    e_bb: float                    # same-spin bb component
    e_ab: float                    # opposite-spin ab component
    e_correlation_per_atom: float
    e_total_per_atom: float
    uhf: CCMUHFResult              # the underlying UHF-CCM reference
    # Per-call citation key; see CCMMP2Result.backend for why this is not a
    # class default.
    backend: str = ""

    @property
    def guess_selection(self):
        """The actual SCF reference selection, forwarded without re-resolution."""
        return getattr(self.uhf, "guess_selection", None)


def _ovov(eri, Co_bra, Cv_bra, Co_ket, Cv_ket):
    """(ia|jb) in Mulliken order from the AO tensor eri[m,n,l,s]=(mn|ls)."""
    return np.einsum("mnls,mi,na,lj,sb->iajb", eri, Co_bra, Cv_bra, Co_ket, Cv_ket,
                     optimize=True)


def _bmo(L, Co, Cv):
    """Occupied-virtual cderi B-tensor ``B[P,i,a] = Σ_μν C_o[μ,i] L[P,μν] C_v[ν,a]``."""
    return np.einsum("Pmn,mi,na->Pia", L, Co, Cv, optimize=True)


def _ovov_ri(B_bra, B_ket):
    """(ia|jb) = ``Σ_P B_bra[P,i,a] B_ket[P,j,b]`` from the cderi B-tensors."""
    return np.einsum("Pia,Pjb->iajb", B_bra, B_ket, optimize=True)


def _same_spin(ovov, eo, ev):
    """E_ss = 1/4 S [(ia|jb)-(ib|ja)]^2 / D (full unrestricted index sum)."""
    if ovov.size == 0:
        return 0.0
    denom = (eo[:, None, None, None] - ev[None, :, None, None]
             + eo[None, None, :, None] - ev[None, None, None, :])
    anti = ovov - ovov.transpose(0, 3, 2, 1)            # (ia|jb) - (ib|ja)
    return float(0.25 * np.sum(anti * anti / denom))


def run_ccm_ump2(ccm, uhf_result: "CCMUHFResult | None" = None, *, initial_guess: object = "AUTO",
                 method="union12", eri=None, cderi=None) -> CCMUMP2Result:
    """Open-shell UMP2-CCM on ``ccm`` (a :class:`CCMSystem`).

    ``method`` selects the effective four-center (``"union12"`` default, or
    ``"aiccm2026dev-a"`` -- see :func:`vibeqc.periodic.ccm.scf.run_ccm_rhf`). The same
    ``eri`` tensor drives the reference UHF and the correlation (``eri`` overrides
    ``method``). Returns a :class:`CCMUMP2Result`.

    **RI-UMP2 (``cderi=L``).** Given the neutral GDF cderi ``L[P,μν]``
    (:func:`~vibeqc.periodic.ccm.neutral.ccm_neutral_cderi`), each spin channel's
    integrals are contracted from per-spin B-tensors ``B^σ[P,i,a] = Σ_μν
    C^σ_o[μ,i] L[P,μν] C^σ_v[ν,a]`` -- ``(ia|jb)^{σσ'} = Σ_P B^σ B^σ'`` -- so the
    dense ``n_ref_ao**4`` AO four-center is **never formed** and the open-shell
    correlation reaches genuine 3-D (the sibling of the RI-MP2/RI-CCSD levers).
    ``uhf_result`` is required and must be the matching neutral reference; equals
    the dense UMP2 on ``g_eff = Σ_P L⊗L`` to machine ε.

    Validated: isolated-cell limit reproduces vibe-qc's molecular ``run_ump2``;
    reduces to :func:`run_ccm_mp2` for a closed shell.
    """
    guess_selection = _ccm_initial_guess(
        ccm, initial_guess, driver='run_ccm_ump2', reference=uhf_result,
    )
    if cderi is not None:
        if uhf_result is None:
            raise ValueError(
                "RI-UMP2 (cderi=) requires a converged uhf_result on the matching "
                "neutral reference (run_ccm_uhf(eri=ccm_eri_neutral)).")
    else:
        if eri is None:
            eri = _ccm_eri_for_method(ccm, method)
        if uhf_result is None:
            uhf_result = run_ccm_uhf(ccm, eri=eri, initial_guess=initial_guess)
    if not uhf_result.converged:
        raise ValueError("UMP2-CCM requires a converged UHF-CCM reference; SCF did not converge.")

    Ca = np.asarray(uhf_result.mo_coeffs_alpha)
    Cb = np.asarray(uhf_result.mo_coeffs_beta)
    if np.iscomplexobj(Ca) or np.iscomplexobj(Cb):
        Ca = np.real_if_close(Ca, tol=1000).real
        Cb = np.real_if_close(Cb, tol=1000).real
    Ca = np.asarray(Ca, dtype=float)
    Cb = np.asarray(Cb, dtype=float)
    ea = np.asarray(uhf_result.mo_energies_alpha, dtype=float)
    eb = np.asarray(uhf_result.mo_energies_beta, dtype=float)
    na, nb = uhf_result.n_alpha, uhf_result.n_beta

    Coa, Cva, eoa, eva = Ca[:, :na], Ca[:, na:], ea[:na], ea[na:]
    Cob, Cvb, eob, evb = Cb[:, :nb], Cb[:, nb:], eb[:nb], eb[nb:]

    has_ab = bool(Cva.shape[1] and Cvb.shape[1] and na and nb)
    if cderi is not None:
        L = np.asarray(cderi, dtype=float)
        Ba = _bmo(L, Coa, Cva)
        Bb = _bmo(L, Cob, Cvb)
        ovov_aa = _ovov_ri(Ba, Ba)
        ovov_bb = _ovov_ri(Bb, Bb)
        ab = _ovov_ri(Ba, Bb) if has_ab else None
    else:
        ovov_aa = _ovov(eri, Coa, Cva, Coa, Cva)
        ovov_bb = _ovov(eri, Cob, Cvb, Cob, Cvb)
        ab = _ovov(eri, Coa, Cva, Cob, Cvb) if has_ab else None

    # Same-spin aa and bb.
    e_aa = _same_spin(ovov_aa, eoa, eva)
    e_bb = _same_spin(ovov_bb, eob, evb)

    # Opposite-spin ab: no exchange, no 1/4.
    e_ab = 0.0
    if ab is not None:
        denom = (eoa[:, None, None, None] - eva[None, :, None, None]
                 + eob[None, None, :, None] - evb[None, None, None, :])
        e_ab = float(np.sum(ab * ab / denom))

    e_corr = e_aa + e_bb + e_ab
    e_hf = float(uhf_result.energy)
    e_tot = e_hf + e_corr
    return CCMUMP2Result(
        e_correlation=e_corr, e_hf=e_hf, e_total=e_tot,
        e_aa=e_aa, e_bb=e_bb, e_ab=e_ab,
        e_correlation_per_atom=e_corr / ccm.n_atoms,
        e_total_per_atom=e_tot / ccm.n_atoms, uhf=uhf_result,
        # Bare four-centre open-shell correlation; run_ccm_ri_mp2 replaces
        # this with the neutral-RI key when it is the caller.
        backend="aiccm2026dev-a-ump2",
    )
