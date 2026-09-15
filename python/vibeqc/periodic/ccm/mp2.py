"""MP2-CCM -- second-order Moller-Plesset correlation on a Cyclic Cluster Model
reference (AICCM roadmap Phase 3 / milestone M4).

The CCM delivers periodic correlation through the *molecular* post-HF code path:
the WSSC-weighted AO ERIs (:func:`vibeqc.periodic.ccm.padded.ccm_eri`) and the
HF-CCM canonical MOs (:func:`vibeqc.periodic.ccm.scf.run_ccm_rhf`) are fed to the
ordinary closed-shell RMP2 energy expression. The correlation *algebra* is the
molecular one unchanged; only the *integral source* is the CCM-weighted tensor
(handovers/HANDOVER_AICCM.md Sec.7) -- vibe-qc's C++ ``run_mp2`` recomputes *canonical* AO ERIs
internally and would silently ignore the weighting, so the energy expression is
evaluated here instead.

    # Closed-shell RMP2 (e.g. Szabo & Ostlund, Modern Quantum Chemistry, eq.
    # 6.74), with every ERI replaced by its CCM-weighted counterpart:
    #   E_corr = S_{ijab} (ia|jb) [ 2 (ia|jb) - (ib|ja) ]
    #                      / (e_i + e_j - e_a - e_b)
    # i,j in occupied CCM MOs; a,b in virtual; (ia|jb) in Mulliken order.

As with the HF-CCM driver this is the small-cluster / 1-D (feasibly 2-D)
validation path: it consumes the dense ``ccm_eri`` tensor, so 3-D supercells need
the scalable weighted four-center (still open -- see handovers/HANDOVER_AICCM.md M3). In the
isolated-cell limit the weighted ERI is the molecular ERI, so this reproduces
vibe-qc's molecular ``run_mp2`` exactly; for a periodic cluster the correlation
energy per cell converges with cluster size (Peintinger & Bredow, J. Comput.
Chem. 35, 839 (2014), doi:10.1002/jcc.23550).
"""

from __future__ import annotations

from .scf import _ccm_initial_guess

from dataclasses import dataclass, replace

import numpy as np

from .scf import CCMSCFResult, _ccm_eri_for_method, run_ccm_rhf

__all__ = ["CCMMP2Result", "run_ccm_mp2", "run_ccm_ri_mp2",
           "run_ccm_ri_mp2_on_reference"]


@dataclass
class CCMMP2Result:
    # MEASURED 2026-09-08, not read: these are TOTAL finite cyclic-cluster
    # energies, like every other driver in this package -- H2/STO-3G in a
    # 6-bohr cube gives e_correlation -0.0131578701 at nrep=(1,1,1) and
    # -0.0263506898 at (2,1,1), a clean doubling. The comments here used to
    # say "per reference cell", the identical error that CCMKSResult and
    # CCMUHFResult carried (see periodic/ccm/four_center_runner.py's module
    # docstring). Divide by ``ccm.n_cells`` to compare with a primitive-cell
    # method; the runner adapter does.
    e_correlation: float           # MP2 correlation energy, TOTAL cyclic (Ha)
    e_hf: float                    # HF-CCM energy, TOTAL cyclic (Ha)
    e_total: float                 # e_hf + e_correlation
    e_os: float                    # opposite-spin component
    e_ss: float                    # same-spin component
    e_correlation_per_atom: float
    e_total_per_atom: float
    scf: CCMSCFResult              # the underlying HF-CCM reference
    # Citation key for THIS call, not for the class (maintainer 2026-09-06).
    # One class serves two lineages -- run_ccm_mp2 is the bare four-centre
    # correlation, run_ccm_ri_mp2 the neutral-RI one -- so a class default
    # would cite the bare four-centre papers for a neutral-RI number, the
    # R1/D89 conflation the route taxonomy exists to prevent. Each producer
    # stamps what it actually built; "" means unstamped, and assemble() then
    # falls back to the bare lineage row.
    backend: str = ""

    @property
    def guess_selection(self):
        """The actual SCF reference selection, forwarded without re-resolution."""
        return getattr(self.scf, "guess_selection", None)


def run_ccm_mp2(ccm, scf_result: "CCMSCFResult | None" = None, *, initial_guess: object = "AUTO",
                method="union12", eri=None, cderi=None) -> CCMMP2Result:
    """Closed-shell MP2-CCM on ``ccm`` (a :class:`CCMSystem`).

    Builds (or reuses) the WSSC-weighted effective ERI tensor and the HF-CCM
    reference, transforms the occupied-virtual ERIs to the CCM MO basis, and
    evaluates the RMP2 correlation energy. Returns a :class:`CCMMP2Result`.

    ``method`` selects the effective four-center (``"union12"`` default, or
    ``"aiccm2026dev-a"`` for the symmetric four-center -- see
    :func:`vibeqc.periodic.ccm.scf.run_ccm_rhf`). The same ``eri`` tensor is used
    for the reference SCF and the correlation so the two are consistent (``eri``
    overrides ``method`` if given). Status mirrors :func:`run_ccm_rhf`: exact in
    the molecular limit, ~1e-5 for 1-D periodic clusters with arbitrary orbitals,
    dense ``n_ref_ao**4`` (small / low-dimensional clusters only).

    **Neutral-control RI-MP2 (``cderi=L``).** Given the neutral GDF cderi ``L[P,μν]``
    (:func:`~vibeqc.periodic.ccm.neutral.ccm_neutral_cderi`), the occupied-virtual
    integrals are contracted *directly* from the density fit --
    ``(ia|jb) = Σ_P B^o_{ia,P} B^o_{jb,P}`` with ``B^o[P,i,a] = Σ_μν C_o[μ,i]
    L[P,μν] C_v[ν,a]`` -- so the dense ``n_ref_ao**4`` AO four-center is **never
    formed** and the correlation reaches genuine 3-D (the AO-tensor wall, not the
    MO-amplitude tensor, is what blocks the dense path; cf. the scalability lane's
    integral-direct HF). ``scf_result`` is required and must be the matching
    neutral reference (e.g. :func:`~vibeqc.periodic.ccm.ri.run_ccm_rhf_gdf` or
    ``run_ccm_rhf(eri=ccm_eri_neutral(ccm))``). The no-AO-tensor limit equals the
    dense MP2 on ``g_eff = Σ_P L⊗L`` to machine ε.
    """
    guess_selection = _ccm_initial_guess(
        ccm, initial_guess, driver='run_ccm_mp2', reference=scf_result,
    )
    if cderi is not None:
        if scf_result is None:
            raise ValueError(
                "RI-MP2 (cderi=) requires a converged scf_result on the matching "
                "neutral reference (run_ccm_rhf_gdf or run_ccm_rhf(eri=ccm_eri_neutral)).")
    else:
        if eri is None:
            eri = _ccm_eri_for_method(ccm, method)
        if scf_result is None:
            scf_result = run_ccm_rhf(ccm, eri=eri, initial_guess=initial_guess)
    if not scf_result.converged:
        raise ValueError("MP2-CCM requires a converged HF-CCM reference; SCF did not converge.")

    C = np.asarray(scf_result.mo_coeffs)
    eps = np.asarray(scf_result.mo_energies)

    n_elec = ccm.supercell.n_electrons()
    if n_elec % 2 != 0:
        raise ValueError(
            f"run_ccm_mp2 is closed-shell but the cluster has {n_elec} electrons; "
            "use an even-electron cluster (UMP2-CCM is a later milestone)."
        )
    n_occ = n_elec // 2
    nbf = C.shape[1]
    if n_occ == 0 or n_occ == nbf:
        # No occupied or no virtual space -> zero correlation.
        e_hf = float(scf_result.energy)
        return CCMMP2Result(0.0, e_hf, e_hf, 0.0, 0.0, 0.0, e_hf / ccm.n_atoms,
                            scf_result, "aiccm2026dev-a-mp2")

    if np.iscomplexobj(C):
        C = np.real_if_close(C, tol=1000).real
    Co = np.asarray(C[:, :n_occ], dtype=float)
    Cv = np.asarray(C[:, n_occ:], dtype=float)
    eo = np.asarray(eps[:n_occ], dtype=float)
    ev = np.asarray(eps[n_occ:], dtype=float)

    if cderi is not None:
        # RI-MP2: (ia|jb) = Σ_P B^o_{ia,P} B^o_{jb,P} from the neutral cderi --
        # the dense n_ref_ao**4 AO four-center is never formed.
        L = np.asarray(cderi, dtype=float)
        B = np.einsum("Pmn,mi,na->Pia", L, Co, Cv, optimize=True)  # (naux, n_occ, n_vir)
        ovov = np.einsum("Pia,Pjb->iajb", B, B, optimize=True)
    else:
        # (ia|jb) in Mulliken order from the CCM-weighted AO tensor eri[m,n,l,s]=(mn|ls).
        ovov = np.einsum("mnls,mi,na,lj,sb->iajb", eri, Co, Cv, Co, Cv, optimize=True)

    # Orbital-energy denominators e_i - e_a + e_j - e_b.
    denom = (eo[:, None, None, None] - ev[None, :, None, None]
             + eo[None, None, :, None] - ev[None, None, None, :])

    # Opposite-spin: (ia|jb)^2 / D ; same-spin: (ia|jb)[(ia|jb) - (ib|ja)] / D.
    exch = ovov.transpose(0, 3, 2, 1)                    # (ib|ja)
    e_os = float(np.sum(ovov * ovov / denom))
    e_ss = float(np.sum(ovov * (ovov - exch) / denom))
    e_corr = e_os + e_ss                                 # == S (ia|jb)[2(ia|jb)-(ib|ja)]/D

    e_hf = float(scf_result.energy)
    e_tot = e_hf + e_corr
    return CCMMP2Result(
        e_correlation=e_corr, e_hf=e_hf, e_total=e_tot, e_os=e_os, e_ss=e_ss,
        e_correlation_per_atom=e_corr / ccm.n_atoms,
        e_total_per_atom=e_tot / ccm.n_atoms, scf=scf_result,
        # The bare four-centre correlation. run_ccm_ri_mp2 replaces this with
        # the neutral-RI key when it is the caller, because it built its own
        # matching reference; see CCMMP2Result.backend.
        backend="aiccm2026dev-a-mp2",
    )


def run_ccm_ri_mp2(ccm, *, initial_guess: object = "AUTO", ke_cutoff: float = 200.0, aux_basis=None,
                   reference: str = "direct", symmetry=None):
    """One-call neutral canonical RI-MP2 -- build the neutral cderi ``L`` once, the
    *matching* lean SCF reference, and the RI correlation, with no dense
    ``n_ref_ao**4`` four-center anywhere (reaches moderate 3-D).

    This is a neutral fitted-torus correlation control. The legacy function
    name does not assign Γ-CCM or χ-CCM construction identity.

    Returns a :class:`CCMMP2Result` (closed shell) or a
    :class:`~vibeqc.periodic.ccm.ump2.CCMUMP2Result` (open shell). This is the safe
    entry point: it guarantees the SCF reference and ``L`` are the *same* neutral
    kernel -- calling ``run_ccm_mp2(scf, cderi=L)`` with a mismatched ``scf`` (e.g. a
    bare-four-center reference) is silently wrong.

    ``reference`` selects the complete SCF reference route. Both routes ride the
    same ``L``, but they do not differ only at exchange q=0: the neutral route
    carries WSSC overlap, one-electron, and nuclear terms, while the direct route
    carries their folded/Ewald counterparts. Their energy difference therefore
    does not isolate the seam.

    * ``"neutral"`` -- the lean strict-zero-mode SCF
      (:func:`~vibeqc.periodic.ccm.ri.run_ccm_rhf_ri_neutral`). A historical
      2026-07-10 diagnostic reported c-diamond (2,2,2) correlation of
      -59.0 mHa/atom versus external KMP2 -29.9 (ewald) / -41.1 (None).
      That provenance-incomplete value is not a reportable benchmark, and the
      neutral/direct route difference cannot assign its cause to the seam.
      **This is no longer the default** (D-5, maintainer 2026-09-05): its error
      grows with the cluster rather than converging. Measured on H2/STO-3G in
      ``diag(6, 15, 15)`` bohr at ``ke_cutoff=40``, both legs riding the same
      ``L`` so only this route differs, ``E_hf(direct) - E_hf(neutral)`` is
      +0.956443 Ha at ``nrep=(2,1,1)``, +2.449583 at (3,1,1) and +4.070138 at
      (4,1,1) -- about 1.5 Ha per added cell, with the per-cell figure still
      climbing (0.478, 0.817, 1.018 Ha/cell). Over the same range the
      correlation itself moves only 6.1 to 7.8 mHa, so the defect is in this
      reference, not in the correlation built on it.
    * ``"direct"`` (default) -- the BvK-ewald direct-torus SCF
      (:func:`~vibeqc.periodic.ccm.direct.run_ccm_rhf_direct` closed /
      :func:`~vibeqc.periodic.ccm.direct.run_ccm_uhf_direct` open, on the same
      ``L``, exchange-q=0 seam included). Measured against out-of-process PySCF
      KMP2 at exxdiv='ewald' (same anchor). Historical diagnostics reported
      residuals of 3e-6 Ha/cell for (1,1,1) c-diamond and 0.24 mHa/atom for
      LiH (2,2,2), but those values are not reportable benchmarks: the diamond
      result was not archived and the producer lacks the complete source,
      native-core, numerical-input, and external-default fingerprint. They
      compare a neutral fitted-torus control with external KMP2, not Γ-CCM with
      χ-CCM. Rerun under the current fingerprint contract before citation.

    ``symmetry`` is forwarded to the ``ccm_neutral_cderi`` build (space-group
    pair reduction; ``None``/``False`` off (default), ``True``/"auto", or a
    pre-computed ``CCMSymmetry``) -- the same knob the SCF drivers expose. The
    reduced fit equals the un-reduced build elementwise, so the correlation is
    symmetry-invariant.
    """
    guess_selection = _ccm_initial_guess(
        ccm, initial_guess, driver='run_ccm_ri_mp2',
    )
    from .neutral import ccm_neutral_cderi
    from .ri import run_ccm_rhf_ri_neutral

    if reference not in ("neutral", "direct"):
        raise ValueError(
            f"run_ccm_ri_mp2: reference must be 'neutral' or 'direct', "
            f"got {reference!r}")
    L = ccm_neutral_cderi(ccm, ke_cutoff=ke_cutoff, aux_basis=aux_basis,
                          symmetry=symmetry)
    n_elec = int(ccm.supercell.n_electrons())
    closed = (n_elec % 2 == 0) and int(ccm.supercell.multiplicity) == 1
    if closed:
        if reference == "direct":
            from .direct import run_ccm_rhf_direct

            scf = run_ccm_rhf_direct(ccm, cderi=L, initial_guess=initial_guess)
        else:
            scf = run_ccm_rhf_ri_neutral(ccm, cderi=L, initial_guess=initial_guess)
        # Stamp the NEUTRAL-RI lineage, overriding whatever run_ccm_mp2 set
        # for the bare four-centre path: this call built its own matching
        # neutral reference and `L`, so the citation owed is the RI one.
        # Plain literal on purpose -- tests/test_citations.py reads these
        # labels out of the AST and cannot follow a computed expression.
        return replace(
            run_ccm_mp2(ccm, scf, cderi=L, initial_guess=initial_guess),
            backend="aiccm2026dev-a-ri-mp2",
        )
    from .ump2 import run_ccm_ump2

    if reference == "direct":
        from .direct import run_ccm_uhf_direct

        uhf = run_ccm_uhf_direct(ccm, cderi=L, initial_guess=initial_guess)
    else:
        from .uhf import run_ccm_uhf

        uhf = run_ccm_uhf(ccm, cderi=L, initial_guess=initial_guess)
    return replace(
        run_ccm_ump2(ccm, uhf, cderi=L, initial_guess=initial_guess),
        backend="aiccm2026dev-a-ri-ump2",
    )


def run_ccm_ri_mp2_on_reference(ccm, scf_result, cderi):
    """Neutral-RI MP2/UMP2 on a reference and ``cderi`` the CALLER built.

    The same correlation :func:`run_ccm_ri_mp2` performs, minus the SCF: for
    a caller that has ALREADY converged the matching neutral reference and
    still holds the ``L`` it rode. The runner's ``variant="real-gamma"`` arm
    is that caller -- it converges the direct-torus SCF itself, so having
    ``run_ccm_ri_mp2`` build a second one would both double the cost and
    report a correlation belonging to a different SCF than the energy beside
    it.

    **The caller owns the consistency guarantee here.** ``run_ccm_ri_mp2``
    exists precisely to remove that burden, and this entry hands it back:
    ``scf_result`` MUST be converged on this same ``cderi``. Pairing it with
    a reference built on another kernel is silently wrong. Use
    :func:`run_ccm_ri_mp2` unless you are holding the very array the
    reference converged on.

    The saving is the SCF, not accuracy. Measured on H2/STO-3G in a 6-bohr
    cube, the two neutral builders agree closely -- ``run_ccm_ri_mp2`` rides
    ``ccm_neutral_cderi`` and the direct drivers default to
    ``ccm_neutral_cderi_fold``, yet their correlation differs by 2.6e-17 at
    ``nrep=(1,1,1)`` and 4.1e-12 at (2,1,1) -- so this entry is not here to
    dodge a builder discrepancy. It is here because a caller that has
    already converged the reference would otherwise pay for a second SCF,
    and because the correlation then belongs to the energy printed beside it
    by identity rather than by two numbers happening to agree.

    There is deliberately **no** ``initial_guess`` parameter. This entry runs
    no SCF, so there is no guess for it to make: the one that produced
    ``scf_result`` is already recorded on that result. Accepting a selector
    here would advertise an influence the call does not have, and it would
    also (rightly) be swept by the guard in ``tests/test_ccm_mp2.py`` that
    requires every ``run_ccm_*`` entry point taking ``initial_guess`` to
    validate it before touching integrals.

    Stamps the neutral-RI citation lineage, never the bare four-centre one:
    the labels are the plain literals ``tests/test_citations.py`` reads out
    of this module's AST, and they are the same two keys
    :func:`run_ccm_ri_mp2` stamps, so the route/producer bijection is
    unchanged.
    """
    n_elec = int(ccm.supercell.n_electrons())
    closed = (n_elec % 2 == 0) and int(ccm.supercell.multiplicity) == 1
    if closed:
        return replace(
            run_ccm_mp2(ccm, scf_result, cderi=cderi),
            backend="aiccm2026dev-a-ri-mp2",
        )
    from .ump2 import run_ccm_ump2

    return replace(
        run_ccm_ump2(ccm, scf_result, cderi=cderi),
        backend="aiccm2026dev-a-ri-ump2",
    )
