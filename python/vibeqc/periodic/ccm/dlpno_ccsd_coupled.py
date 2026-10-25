"""Shared DLPNO-CCSD(T) engine on the experimental neutral fitted CCM torus.

The historical per-pair Python iteration is retired (#508, decision 1).
The CCM reference, metric and factors enter through ``CorrelationSource``;
the molecular engine owns pair construction, iteration and triples.

Pair interactions now use the term-by-term PNO contraction of Riplinger
and Neese, J. Chem. Phys. 138, 034106 (2013), Sec. II C, Eqs. (26)-(30),
doi:10.1063/1.4773581. Projecting a source amplitude into the target pair
before contraction loses contributions under PNO truncation. The shared
engine avoids that projection and agrees with its full-virtual residual
oracle at the same retained pair spaces.

This API retains its explicit Mulliken domains, legacy PNO density,
unscreened pair list and uncorrected PNO tail. Its default local (T0)
remains the semicanonical approximation of Riplinger et al., J. Chem.
Phys. 139, 134101 (2013); ``triples_mode="dense"`` selects canonical (T).
The union-PNO ``ccm_dlpno_ccsd`` stays as a small-cell reference. None of
these neutral-control calls assigns Gamma-CCM or chi-CCM construction
identity or qualifies the periodic production route.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Optional

import numpy as np

__all__ = ["CCMDLPNOCoupledCCSDResult", "ccm_dlpno_ccsd_coupled"]


@dataclass
class CCMDLPNOCoupledCCSDResult:
    """Result of :func:`ccm_dlpno_ccsd_coupled`, in total cyclic energies."""

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
    source_kind: str = ""
    source_identity: str = ""
    same_source: bool = False
    exchange_q0: str = "unspecified"
    cc_result: object = field(default=None, repr=False)


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
    """Per-pair DLPNO-CCSD(T) through the shared compiled PNO-space engine.

    ``scf_result`` must be a converged closed-shell HF reference on the
    neutral fitted torus. A Bloch result is folded by ``as_real_reference``.
    The reference's retained ``cderi`` is reused unless explicitly replaced;
    legacy references without retained factors rebuild them. Only retained
    (or identical explicit) factors certify ``same_source``.

    ``coupling_radius <= 0`` couples every pair to every occupied; positive
    radii use the supercell minimum-image distance. ``n_frozen`` counts
    supercell orbitals, held fixed during active-space localisation.

    ``triples_mode="local"`` keeps the historical per-triple TNO (T0), with
    ``tcut_tno`` controlling TNO truncation. ``"dense"`` expands the pair
    amplitudes and evaluates canonical (T) in the full virtual space.
    It is invariant to active occupied localisation and ignores ``tcut_tno``.
    At full domains both CCSD and dense (T) reproduce the canonical
    neutral-control calculation on the same factors.
    """
    from ...dlpno.ccsd_local_solver import LocalCCSDOptions, run_local_dlpno_ccsd
    from .dlpno_source import ccm_neutral_source

    if triples_mode not in ("local", "dense"):
        raise ValueError(f"triples_mode must be 'local' or 'dense'; got {triples_mode!r}")
    source = ccm_neutral_source(
        ccm, scf_result, cderi=cderi, localize=localize, n_frozen=n_frozen,
        ke_cutoff=ke_cutoff,
    )
    # Retain this research API's pair-space convention, not the unrelated
    # NormalPNO preset. Only the erroneous target-pair-first residual is
    # replaced: contractions use the common term-by-term PNO kernel.
    options = LocalCCSDOptions(
        localise="none", n_frozen=int(n_frozen), tcut_pno=tcut_pno,
        tcut_pno_diag_scale=1.0, pno_norm="legacy", pno_correction=False,
        domain_criterion="mulliken", tcut_mkn=tcut_mkn, tcut_pairs=0.0,
        coupling_radius=coupling_radius, lindep=lindep, residual_domain="pno",
        compute_triples=compute_triples, tcut_tno=tcut_tno,
        triples_mode="exact" if triples_mode == "dense" else "local",
        max_iter=max_iter, conv_tol_energy=conv_tol, conv_tol_residual=1e-7,
        diis_size=diis_size,
        # The legacy API accepts an explicitly constructed CCM of any size.
        max_nbf=int(ccm.nbf),
    )
    result = run_local_dlpno_ccsd(None, None, None, None, options, source=source)
    e_corr = result.e_corr + result.e_t
    return CCMDLPNOCoupledCCSDResult(
        e_hf=result.e_hf, e_ccsd_correlation=result.e_corr, e_t=result.e_t,
        e_correlation=e_corr, e_total=result.e_total,
        e_correlation_per_atom=e_corr / ccm.n_atoms,
        e_total_per_atom=result.e_total / ccm.n_atoms, n_pairs=result.n_pairs,
        avg_pno_per_pair=float(np.mean(list(result.pno_per_pair.values()))) if result.n_pairs else 0.0,
        avg_coupled_occ=result.avg_coupled_occ, localize=localize,
        converged=result.converged, n_iter=result.n_iter,
        guess_selection=getattr(source.reference, "guess_selection", None),
        source_kind=source.kind, source_identity=source.identity,
        same_source=source.conventions.same_source,
        exchange_q0=source.conventions.exchange_q0, cc_result=result,
    )
