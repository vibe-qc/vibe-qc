"""CCSD(T)-CCM -- coupled-cluster correlation on a Cyclic Cluster Model reference.

The capstone of the CCM post-HF stack. As with MP2-CCM, the correlation is
reached through the *molecular* algebra fed the CCM-weighted MO integrals -- but
vibe-qc's C++ ``run_ccsd`` recomputes (density-fitted) canonical integrals from
the molecule/basis and would ignore the WSSC weighting, and there is no external-
integral CC kernel. So a compact **spin-orbital CCSD (and perturbative (T))** is
evaluated here directly on the CCM MO integrals.

Formulation: the standard spin-orbital CCSD intermediates of Stanton, Gauss,
Watts & Bartlett, J. Chem. Phys. 94, 4334 (1991) (the "Programmer's Guide"
equations); the (T) correction follows Crawford & Schaefer, Rev. Comput. Chem.
14, 33 (2000). For a canonical CCM reference the Fock is diagonal (f_ia = 0), so
only the antisymmetrised two-electron integrals <pq‖rs> -- built from the CCM
effective ERI tensor transformed to the CCM MO basis -- enter.

Validation: the isolated-cell limit reproduces vibe-qc's molecular ``run_ccsd``
(conventional, ``density_fit=False``) to µHa; for H₂ (2 e⁻) CCSD = FCI. Small /
low-dimensional clusters only (dense ``n_so**6`` (T) step). Reference for the CCM
itself: Peintinger & Bredow, J. Comput. Chem. 35, 839 (2014).
"""

from __future__ import annotations

from .scf import _ccm_initial_guess

from dataclasses import dataclass, replace

import numpy as np

from .padded import ccm_eri  # noqa: F401  (kept for parity with sibling modules)
from .scf import CCMSCFResult, _ccm_eri_for_method, run_ccm_rhf

__all__ = ["CCMCCSDResult", "run_ccm_ccsd", "run_ccm_ri_ccsd",
           "run_ccm_ri_ccsd_on_reference"]


@dataclass
class CCMCCSDResult:
    # MEASURED 2026-09-09, not read: TOTAL finite cyclic-cluster energies,
    # like every other driver in this package. H2/STO-3G in a 6-bohr cube
    # gives e_correlation -0.0205616181 at nrep=(1,1,1) and -0.0411635299 at
    # (2,1,1), a clean doubling. The comments here used to say "per reference
    # cell", the same error CCMMP2Result, CCMKSResult and CCMUHFResult
    # carried. Divide by ``ccm.n_cells`` to compare with a primitive-cell
    # method; the runner adapters do.
    converged: bool
    n_iter: int
    e_ccsd_correlation: float  # CCSD singles+doubles, TOTAL cyclic (Ha)
    e_t: float  # perturbative (T) correction (0.0 if not computed)
    e_correlation: float  # e_ccsd_correlation + e_t
    e_hf: float  # HF-CCM energy, TOTAL cyclic (Ha)
    e_total: float  # e_hf + e_correlation
    e_correlation_per_atom: float
    e_total_per_atom: float
    scf: CCMSCFResult  # the underlying HF-CCM reference
    # Per-call citation key; see CCMMP2Result.backend for why this is not a
    # class default. run_ccm_ccsd is the bare four-centre lineage,
    # run_ccm_ri_ccsd the neutral-RI one, and they must never share a row.
    backend: str = ""

    @property
    def guess_selection(self):
        """The actual SCF reference selection, forwarded without re-resolution."""
        return getattr(self.scf, "guess_selection", None)


def _spin_block_eri_phys(mo_mulliken):
    """Antisymmetrised spin-orbital integrals <pq‖rs> from spatial MO ERIs.

    ``mo_mulliken[p,q,r,s] = (pq|rs)`` spatial (chemists'). Returns <pq‖rs> over
    spin-orbitals interleaved (spatial p -> 2p,2p+1 = a,b).
    """
    nmo = mo_mulliken.shape[0]
    nso = 2 * nmo
    # spin mask: spin(p) == p % 2.
    spin = np.arange(nso) % 2
    # Physicist <pq|rs> = (pr|qs) chemists'; lift to spin orbitals.
    idx = np.arange(nso) // 2
    chem = mo_mulliken[np.ix_(idx, idx, idx, idx)]  # (pq|rs) spatial-lifted
    phys = chem.transpose(0, 2, 1, 3)  # <pq|rs> = (pr|qs)
    # spin delta: <pq|rs> nonzero only if spin(p)=spin(r) and spin(q)=spin(s).
    sp_pr = spin[:, None, None, None] == spin[None, None, :, None]
    sp_qs = spin[None, :, None, None] == spin[None, None, None, :]
    phys = phys * sp_pr * sp_qs
    return phys - phys.transpose(0, 1, 3, 2)  # <pq||rs>


def _ccsd_energy_so(ts, td, fs, spinints, nocc_so):
    """Spin-orbital CCSD correlation energy (Stanton 1991). ts[a,i], td[a,b,i,j]."""
    o = slice(0, nocc_so)
    v = slice(nocc_so, spinints.shape[0])
    e = np.einsum("ia,ai->", fs[o, v], ts)
    e += 0.25 * np.einsum("ijab,abij->", spinints[o, o, v, v], td)
    e += 0.5 * np.einsum("ijab,ai,bj->", spinints[o, o, v, v], ts, ts)
    return e


def _ccsd_residual_so(ts, td, fs, spinints, nocc_so):
    """Projected spin-orbital CCSD residual Ω1[a,i], Ω2[a,b,i,j] (Ω = 0 at convergence).

    ``Ω = numerator − D⊙t`` for the Stanton (1991) T1/T2 equations, written for a
    **general (non-canonical) Fock** ``fs`` (the off-diagonal ``f_vv``/``f_oo`` enter
    ``fae``/``fmi``; the bare ``f_ai`` drives T1) so the analytic-gradient machinery
    can differentiate the CCSD Lagrangian with respect to the Fock. Reduces to the
    canonical Stanton residual when ``fs`` is diagonal. The single source of truth for
    both :func:`_ccsd_iterate` (energy) and the gradient
    (:func:`~vibeqc.periodic.ccm.gradient_analytic.run_ccm_ccsd_gradient`).

    Reference: Stanton, Gauss, Watts & Bartlett, J. Chem. Phys. 94, 4334 (1991)
    (the "Programmer's Guide" T1/T2 equations, eqs 1-2 and the F/W intermediates).
    """
    nso = spinints.shape[0]
    o = slice(0, nocc_so)
    v = slice(nocc_so, nso)
    eps = np.diag(fs)
    fov = fs[o, v]
    fvv = fs[v, v] - np.diag(np.diag(fs[v, v]))          # off-diagonal f_ab
    foo = fs[o, o] - np.diag(np.diag(fs[o, o]))          # off-diagonal f_ij

    ts_t = td + 0.5 * (np.einsum("ai,bj->abij", ts, ts) - np.einsum("bi,aj->abij", ts, ts))
    ta = td + (np.einsum("ai,bj->abij", ts, ts) - np.einsum("bi,aj->abij", ts, ts))

    fae = (
        fvv
        - 0.5 * np.einsum("me,am->ae", fov, ts)
        + np.einsum("fm,mafe->ae", ts, spinints[o, v, v, v])
        - 0.5 * np.einsum("afmn,mnef->ae", ts_t, spinints[o, o, v, v])
    )
    fmi = (
        foo
        + 0.5 * np.einsum("me,ei->mi", fov, ts)
        + np.einsum("en,mnie->mi", ts, spinints[o, o, o, v])
        + 0.5 * np.einsum("efin,mnef->mi", ts_t, spinints[o, o, v, v])
    )
    fme = fov + np.einsum("fn,mnef->me", ts, spinints[o, o, v, v])

    wmnij = (
        spinints[o, o, o, o]
        + np.einsum("ej,mnie->mnij", ts, spinints[o, o, o, v])
        - np.einsum("ei,mnje->mnij", ts, spinints[o, o, o, v])
        + 0.25 * np.einsum("efij,mnef->mnij", ta, spinints[o, o, v, v])
    )
    wabef = (
        spinints[v, v, v, v]
        - np.einsum("bm,amef->abef", ts, spinints[v, o, v, v])
        + np.einsum("am,bmef->abef", ts, spinints[v, o, v, v])
        + 0.25 * np.einsum("abmn,mnef->abef", ta, spinints[o, o, v, v])
    )
    wmbej = (
        spinints[o, v, v, o]
        + np.einsum("fj,mbef->mbej", ts, spinints[o, v, v, v])
        - np.einsum("bn,mnej->mbej", ts, spinints[o, o, v, o])
        - np.einsum("fbjn,mnef->mbej",
                    (0.5 * td + np.einsum("fj,bn->fbjn", ts, ts)),
                    spinints[o, o, v, v])
    )

    # T1 numerator (Stanton eq 1) — the bare f_ai drive is nonzero off-canonical.
    ts_num = (
        fs[v, o]
        + np.einsum("ei,ae->ai", ts, fae)
        - np.einsum("am,mi->ai", ts, fmi)
        + np.einsum("aeim,me->ai", td, fme)
        - np.einsum("fn,naif->ai", ts, spinints[o, v, o, v])
        - 0.5 * np.einsum("efim,maef->ai", td, spinints[o, v, v, v])
        - 0.5 * np.einsum("aemn,nmei->ai", td, spinints[o, o, v, o])
    )
    Dai = (eps[o, None] - eps[None, v]).T                # [a,i] = εi − εa
    omega1 = ts_num - Dai * ts

    # T2 numerator (Stanton eq 2).
    t2 = spinints[v, v, o, o].copy()
    tmp = fae - 0.5 * np.einsum("bm,me->be", ts, fme)
    t2 += np.einsum("aeij,be->abij", td, tmp)
    t2 -= np.einsum("beij,ae->abij", td, tmp)            # P(ab)
    tmp = fmi + 0.5 * np.einsum("ej,me->mj", ts, fme)
    t2 -= np.einsum("abim,mj->abij", td, tmp)
    t2 += np.einsum("abjm,mi->abij", td, tmp)            # P(ij)
    t2 += 0.5 * np.einsum("abmn,mnij->abij", ta, wmnij)
    t2 += 0.5 * np.einsum("efij,abef->abij", ta, wabef)
    ring = np.einsum("aeim,mbej->abij", td, wmbej) - np.einsum(
        "ei,am,mbej->abij", ts, ts, spinints[o, v, v, o])
    t2 += ring
    t2 -= ring.transpose(0, 1, 3, 2)                     # P(ij)
    t2 -= ring.transpose(1, 0, 2, 3)                     # P(ab)
    t2 += ring.transpose(1, 0, 3, 2)                     # P(ij)P(ab)
    dt = np.einsum("ei,abej->abij", ts, spinints[v, v, v, o])
    t2 += dt - dt.transpose(0, 1, 3, 2)                  # P(ij)
    dm = np.einsum("am,mbij->abij", ts, spinints[o, v, o, o])
    t2 -= dm - dm.transpose(1, 0, 2, 3)                  # P(ab)
    Dabij = (
        eps[o, None, None, None] + eps[None, o, None, None]
        - eps[None, None, v, None] - eps[None, None, None, v]
    ).transpose(2, 3, 0, 1)                              # [a,b,i,j]
    omega2 = t2 - Dabij * td
    return omega1, omega2


def _ccsd_iterate(spinints, fs, nocc_so, *, max_iter, conv_tol):
    """Spin-orbital CCSD (Stanton 1991) fixed-point solve on :func:`_ccsd_residual_so`.

    Returns (E_corr, ts, td, conv, n_iter). The Jacobi update ``t += Ω/D`` is
    identical to the earlier inline form (``t = numerator/D``, since
    ``Ω = numerator − D⊙t``); the residual now lives in one reusable function shared
    with the analytic gradient.
    """
    nso = spinints.shape[0]
    o = slice(0, nocc_so)
    v = slice(nocc_so, nso)
    eps = np.diag(fs)
    Dai = (eps[o, None] - eps[None, v]).T                # [a,i]
    Dabij = (
        eps[o, None, None, None] + eps[None, o, None, None]
        - eps[None, None, v, None] - eps[None, None, None, v]
    ).transpose(2, 3, 0, 1)                              # [a,b,i,j]

    ts = np.zeros((nso - nocc_so, nocc_so))              # [a,i]
    td = spinints[v, v, o, o] / Dabij                    # MP2 guess [a,b,i,j]

    e_old = 0.0
    conv = False
    it = 0
    for it in range(1, max_iter + 1):
        o1, o2 = _ccsd_residual_so(ts, td, fs, spinints, nocc_so)
        ts = ts + o1 / Dai
        td = td + o2 / Dabij
        e_new = _ccsd_energy_so(ts, td, fs, spinints, nocc_so)
        if abs(e_new - e_old) < conv_tol and it > 1:
            conv = True
            e_old = e_new
            break
        e_old = e_new
    return e_old, ts, td, conv, it


def _triples(spinints, fs, ts, td, nocc_so):
    """Perturbative (T) correction (spin-orbital, Crawford & Schaefer 2000).

    Uses the fused C++ on-the-fly contraction when available, which avoids
    materialising 6-index intermediate tensors and lifts the ~20 AO cap.
    """
    nso = spinints.shape[0]
    nv = nso - nocc_so
    o = slice(0, nocc_so)
    v = slice(nocc_so, nso)
    eps = np.diag(fs)
    eo = eps[:nocc_so]
    ev = eps[nocc_so:]

    # Try C++ kernel: scalar output, no 6-index intermediates.
    try:
        from ..._vibeqc_core import ccsd_t_triples as _triples_cpp
    except ImportError:
        pass
    else:
        t1_flat = np.ascontiguousarray(ts.ravel(), dtype=float)
        t2_flat = np.ascontiguousarray(td.ravel(), dtype=float)
        eri_flat = np.ascontiguousarray(spinints.ravel(), dtype=float)
        return float(
            _triples_cpp(
                t1_flat,
                t2_flat,
                eri_flat,
                np.ascontiguousarray(eo, dtype=float),
                np.ascontiguousarray(ev, dtype=float),
                nocc_so,
                nv,
                nso,
            )
        )

    # Python fallback: 6-index einsum (small systems only).
    # Disconnected and connected triples amplitudes, contracted on the fly.
    # t3c[abc,ijk] connected; t3d disconnected. Use explicit denominators.
    # Build via einsum over the full triples (small systems only).
    # Connected: P(i/jk)P(a/bc) sum_e t_jk^ae <ei||bc> - P(k/ij)P(c/ab) sum_m t_im^bc <ma||jk>
    t3c = np.einsum("aejk,eibc->abcijk", td, spinints[v, o, v, v]) - np.einsum(
        "bcim,majk->abcijk", td, spinints[o, v, o, o]
    )

    # Antisymmetrise connected amplitude over i/jk and a/bc (permutation operators).
    def asym_ijk(x):
        return (
            x
            - x.transpose(0, 1, 2, 4, 3, 5)  # j<->i
            - x.transpose(0, 1, 2, 5, 4, 3)
        )  # k<->i

    def asym_abc(x):
        return (
            x
            - x.transpose(1, 0, 2, 3, 4, 5)  # b<->a
            - x.transpose(2, 1, 0, 3, 4, 5)
        )  # c<->a

    t3c = asym_abc(asym_ijk(t3c))
    # Disconnected: P(i/jk)P(a/bc) t_i^a <jk||bc>
    t3d = np.einsum("ai,jkbc->abcijk", ts, spinints[o, o, v, v])
    t3d = asym_abc(asym_ijk(t3d))

    Dabcijk = (
        eo[None, None, None, :, None, None]
        + eo[None, None, None, None, :, None]
        + eo[None, None, None, None, None, :]
        - ev[:, None, None, None, None, None]
        - ev[None, :, None, None, None, None]
        - ev[None, None, :, None, None, None]
    )
    et = (1.0 / 36.0) * np.einsum("abcijk,abcijk->", t3c, (t3c + t3d) / Dabcijk)
    return float(et)


def run_ccm_ccsd(
    ccm,
    scf_result: "CCMSCFResult | None" = None,
    *, initial_guess: object = "AUTO",
    method="union12",
    eri=None,
    cderi=None,
    compute_triples=True,
    max_iter=128,
    conv_tol=1e-9,
) -> CCMCCSDResult:
    """Closed-shell CCSD(T)-CCM on ``ccm`` (a :class:`CCMSystem`).

    Builds (or reuses) the CCM-weighted effective ERI tensor and the HF-CCM
    reference, transforms the ERIs to the canonical CCM MO basis, and runs a
    spin-orbital CCSD (with the perturbative (T) correction unless
    ``compute_triples=False``). ``method`` selects the four-center weighting
    (``"union12"`` default, ``"aiccm2026dev-a"`` for the symmetric four-center --
    :func:`vibeqc.periodic.ccm.scf.run_ccm_rhf`). ``eri`` overrides ``method``.

    **Neutral-control RI-CCSD (``cderi=L``).** Given the neutral GDF cderi ``L[P,μν]``
    (:func:`~vibeqc.periodic.ccm.neutral.ccm_neutral_cderi`), the MO four-center is
    contracted from the density fit -- ``(pq|rt) = Σ_P B_{pq,P} B_{rt,P}`` with
    ``B[P,p,q] = Σ_μν C[μ,p] L[P,μν] C[ν,q]`` -- so the dense ``n_ref_ao**4`` AO
    four-center is **never formed** and the correlation reaches genuine 3-D (the
    sibling of the RI-MP2 lever; the AO-tensor wall was the blocker). ``scf_result``
    is required and must be the matching neutral reference (``run_ccm_rhf_gdf`` /
    ``run_ccm_rhf(eri=ccm_eri_neutral(ccm))``); equals the dense CCSD(T) on
    ``g_eff = Σ_P L⊗L`` to machine ε.

    Validated: isolated-cell limit reproduces vibe-qc's molecular ``run_ccsd``
    (``density_fit=False``); H₂ CCSD = FCI. Returns a :class:`CCMCCSDResult`.
    """
    guess_selection = _ccm_initial_guess(
        ccm, initial_guess, driver='run_ccm_ccsd', reference=scf_result,
    )
    if cderi is not None:
        if scf_result is None:
            raise ValueError(
                "RI-CCSD (cderi=) requires a converged scf_result on the matching "
                "neutral reference (run_ccm_rhf_gdf or run_ccm_rhf(eri=ccm_eri_neutral)).")
    else:
        if eri is None:
            eri = _ccm_eri_for_method(ccm, method)
        if scf_result is None:
            scf_result = run_ccm_rhf(ccm, eri=eri, initial_guess=initial_guess)
    if not scf_result.converged:
        raise ValueError(
            "CCSD-CCM requires a converged HF-CCM reference; SCF did not converge."
        )

    C = np.asarray(scf_result.mo_coeffs)
    if np.iscomplexobj(C):
        C = np.real_if_close(C, tol=1000).real
    C = np.asarray(C, dtype=float)
    eps = np.asarray(scf_result.mo_energies, dtype=float)
    n_elec = ccm.supercell.n_electrons()
    if n_elec % 2 != 0:
        raise ValueError("run_ccm_ccsd is closed-shell; use an even-electron cluster.")
    nmo = C.shape[1]

    # Spatial MO ERIs (chemists' (pq|rs)).
    if cderi is not None:
        # RI: (pq|rt) = Σ_P B_{pq,P} B_{rt,P} from the neutral cderi — the dense
        # n_ref_ao**4 AO four-center is never formed.
        L = np.asarray(cderi, dtype=float)
        B = np.einsum("Pmn,mp,nq->Ppq", L, C, C, optimize=True)  # (naux, nmo, nmo)
        mo = np.einsum("Ppq,Prt->pqrt", B, B, optimize=True)
    else:
        mo = np.einsum("mnls,mp,nq,lr,st->pqrt", eri, C, C, C, C, optimize=True)
    spinints = _spin_block_eri_phys(mo)
    fs = np.diag(np.repeat(eps, 2))  # spin-orbital Fock
    nocc_so = n_elec  # 2 * n_occ

    e_ccsd, ts, td, conv, n_iter = _ccsd_iterate(
        spinints, fs, nocc_so, max_iter=max_iter, conv_tol=conv_tol
    )
    e_t = _triples(spinints, fs, ts, td, nocc_so) if compute_triples else 0.0

    e_corr = e_ccsd + e_t
    e_hf = float(scf_result.energy)
    e_tot = e_hf + e_corr
    # Plain locals, not an inline expression: tests/test_citations.py resolves
    # backend= labels out of the AST and understands exactly this shape.
    _method_label = "ccsd(t)" if compute_triples else "ccsd"
    return CCMCCSDResult(
        converged=conv,
        n_iter=n_iter,
        e_ccsd_correlation=e_ccsd,
        e_t=e_t,
        e_correlation=e_corr,
        e_hf=e_hf,
        e_total=e_tot,
        e_correlation_per_atom=e_corr / ccm.n_atoms,
        e_total_per_atom=e_tot / ccm.n_atoms,
        scf=scf_result,
        # The bare four-centre correlation. run_ccm_ri_ccsd replaces this with
        # the neutral-RI key when it is the caller, because it builds its own
        # matching reference; see CCMCCSDResult.backend. The (T) half of the
        # key tracks what this call ACTUALLY computed: compute_triples is a
        # runtime flag, and citing Raghavachari for a run with no triples
        # would be the misattribution per-call stamping exists to prevent.
        backend=f"aiccm2026dev-a-{_method_label}",
    )


def run_ccm_ri_ccsd(ccm, *, initial_guess: object = "AUTO", ke_cutoff: float = 200.0, aux_basis=None,
                    compute_triples: bool = True,
                    reference: str = "direct", symmetry=None) -> CCMCCSDResult:
    """One-call neutral canonical RI-CCSD(T) (closed shell) -- build the neutral
    cderi ``L`` once, the matching lean SCF reference, and the RI-CCSD(T), with no
    dense ``n_ref_ao**4`` four-center anywhere (reaches moderate 3-D).

    This is a neutral fitted-torus correlation control. The legacy function
    name does not assign Γ-CCM or χ-CCM construction identity.

    The safe entry point (the SCF reference and ``L`` are the *same* neutral kernel;
    a mismatched ``scf`` would be silently wrong). For open-shell CCSD use
    :func:`~vibeqc.periodic.ccm.dlpno_uccsd.ccm_dlpno_uccsd` /
    :func:`~vibeqc.periodic.ccm.uccsd.run_ccm_uccsd` on the lean UHF
    (``run_ccm_uhf(ccm, cderi=L)``).

    ``reference`` selects the complete SCF reference route, exactly as on
    :func:`~vibeqc.periodic.ccm.mp2.run_ccm_ri_mp2`: ``"direct"`` (default since
    D-5, maintainer 2026-09-05; the folded/Ewald route via
    :func:`~vibeqc.periodic.ccm.direct.run_ccm_rhf_direct`, on the same ``L``)
    or ``"neutral"`` (strict-zero mode, whose reference error grows with the
    cluster instead of converging -- see the measurement in that function's
    docstring).
    The routes also differ in overlap, one-electron, and nuclear terms, so their
    correlation difference is not a clean exchange-q=0 seam audit.
    ``symmetry`` is forwarded to the ``ccm_neutral_cderi`` build (space-group
    pair reduction), as on :func:`~vibeqc.periodic.ccm.mp2.run_ccm_ri_mp2`.
    """
    guess_selection = _ccm_initial_guess(
        ccm, initial_guess, driver='run_ccm_ri_ccsd',
    )
    from .neutral import ccm_neutral_cderi
    from .ri import run_ccm_rhf_ri_neutral

    if reference not in ("neutral", "direct"):
        raise ValueError(
            f"run_ccm_ri_ccsd: reference must be 'neutral' or 'direct', "
            f"got {reference!r}")
    L = ccm_neutral_cderi(ccm, ke_cutoff=ke_cutoff, aux_basis=aux_basis,
                          symmetry=symmetry)
    if reference == "direct":
        from .direct import run_ccm_rhf_direct

        scf = run_ccm_rhf_direct(ccm, cderi=L, initial_guess=initial_guess)
    else:
        scf = run_ccm_rhf_ri_neutral(ccm, cderi=L, initial_guess=initial_guess)
    # Stamp the NEUTRAL-RI lineage, overriding the bare four-centre key
    # run_ccm_ccsd set: this call built its own matching neutral reference and
    # ``L``. Plain literal on purpose -- tests/test_citations.py reads these
    # labels out of the AST and cannot follow a computed expression.
    _ri_label = "ri-ccsd(t)" if compute_triples else "ri-ccsd"
    return replace(
        run_ccm_ccsd(ccm, scf, cderi=L, compute_triples=compute_triples,
                     initial_guess=initial_guess),
        backend=f"aiccm2026dev-a-{_ri_label}",
    )


def run_ccm_ri_ccsd_on_reference(ccm, scf_result, cderi, *,
                                 compute_triples: bool = True):
    """Neutral-RI CCSD(T) on a reference and ``cderi`` the CALLER built.

    The CCSD sibling of
    :func:`~vibeqc.periodic.ccm.mp2.run_ccm_ri_mp2_on_reference`, and it
    carries the same contract and the same caveat: ``scf_result`` MUST be
    converged on this same ``cderi``, because this entry runs no SCF and
    cannot check. Use :func:`run_ccm_ri_ccsd` unless you are holding the very
    array the reference converged on.

    Closed and open shell, dispatched on the cluster like
    :func:`~vibeqc.periodic.ccm.mp2.run_ccm_ri_mp2_on_reference`: the
    open-shell leg is :func:`~vibeqc.periodic.ccm.uccsd.run_ccm_uccsd`, which
    is neutral-only and therefore exactly right on this reference. (The
    one-call :func:`run_ccm_ri_ccsd` stays closed-shell, because it builds its
    own reference and has no open-shell SCF leg.) There is deliberately no
    ``initial_guess`` parameter: no SCF runs here, so there is no guess to
    make, and the one that produced ``scf_result`` is already recorded on it.
    """
    n_elec = int(ccm.supercell.n_electrons())
    closed = (n_elec % 2 == 0) and int(ccm.supercell.multiplicity) == 1
    if closed:
        _ri_label = "ri-ccsd(t)" if compute_triples else "ri-ccsd"
        return replace(
            run_ccm_ccsd(ccm, scf_result, cderi=cderi,
                         compute_triples=compute_triples),
            backend=f"aiccm2026dev-a-{_ri_label}",
        )
    from .uccsd import run_ccm_uccsd

    _ri_ulabel = "ri-uccsd(t)" if compute_triples else "ri-uccsd"
    return replace(
        run_ccm_uccsd(ccm, scf_result, cderi=cderi,
                      compute_triples=compute_triples),
        backend=f"aiccm2026dev-a-{_ri_ulabel}",
    )
