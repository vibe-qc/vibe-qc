"""Open-shell UCCSD(T) for the neutral fitted-torus cyclic-cluster control.

The open-shell sibling of :func:`vibeqc.periodic.ccm.ccsd.run_ccm_ccsd`. It takes
the spin-resolved UHF-CCM reference (:func:`vibeqc.periodic.ccm.uhf.run_ccm_uhf`)
on the **neutral** four-center and runs vibe-qc's own spin-orbital UCCSD(T) engine
(:func:`vibeqc.dlpno._ccsd_ref.run_ref_uccsd`, the in-repo anchor for the C++
``uccsd.cpp``), fed the CCM neutral cderi transformed into the a / b MO bases.

Identity note: this is a neutral fitted-torus correlation control. The legacy
``run_ccm_uccsd`` name records API ancestry; it does not identify this
Hamiltonian with the union-and-weight Γ-CCM or finite-character χ-CCM
construction.

Why the neutral route is the natural one here
---------------------------------------------
``run_ref_uccsd`` consumes density-fitted B-tensors with
``(pq|rs)^{s1s2} = S_P B^{s1}_pq B^{s2}_rs``; the CCM kernel that *is* an exact
RI factorization is the neutral four-center ``g_eff = S_P L⊗L`` (``ccm_neutral_cderi``).
The bare-1/r four-center is non-separable (no consistent cderi) and, per the
`-b`-critique correction, is in any case not the right correlation reference for
ionic systems (the Madelung background shifts the occ-virt denominators). So both
the reference and the integrals are the neutral four-center.

Validation
----------
* Closed-shell consistency: for an even-electron closed-shell cluster
  ``run_ccm_uccsd`` reproduces :func:`run_ccm_ccsd` on the same neutral
  four-center (aa = bb, the UHF collapses to RHF).
* Isolated limit: a ``(1,1,1)`` cluster in a large box reproduces vibe-qc's
  molecular ``run_uccsd`` for the same open-shell system.

Reference: Stanton et al., J. Chem. Phys. 94, 4334 (1991) (spin-orbital CCSD);
Raghavachari et al., Chem. Phys. Lett. 157, 479 (1989) ((T)); Peintinger & Bredow,
J. Comput. Chem. 35, 839 (2014) (CCM).
"""

from __future__ import annotations

from .scf import _ccm_initial_guess

from dataclasses import dataclass
from typing import Optional

import numpy as np

from .uhf import CCMUHFResult

__all__ = ["CCMUCCSDResult", "run_ccm_uccsd"]


@dataclass
class CCMUCCSDResult:
    converged: bool
    n_iter: int
    e_ccsd_correlation: float       # UCCSD singles + doubles
    e_t: float                      # perturbative (T)
    e_correlation: float            # UCCSD + (T)
    e_hf: float
    e_total: float
    e_correlation_per_atom: float
    e_total_per_atom: float
    uhf: CCMUHFResult
    # Per-call citation key; see CCMMP2Result.backend. This driver is
    # neutral-only -- it takes no ``method=`` and so cannot build the
    # union-and-weight reference -- so the bare four-centre lineage is not
    # reachable here and only the RI wrappers stamp it.
    backend: str = ""

    @property
    def guess_selection(self):
        """The actual SCF reference selection, forwarded without re-resolution."""
        return getattr(self.uhf, "guess_selection", None)


def _neutral_four_center(L: np.ndarray) -> np.ndarray:
    g = np.einsum("Pmn,Prs->mnrs", L, L, optimize=True)
    g = 0.5 * (g + np.transpose(g, (1, 0, 3, 2)))
    return 0.5 * (g + np.transpose(g, (2, 3, 0, 1)))


def run_ccm_uccsd(
    ccm,
    uhf_result: Optional[CCMUHFResult] = None,
    *, initial_guess: object = "AUTO",
    cderi: Optional[np.ndarray] = None,
    compute_triples: bool = True,
    ke_cutoff: float = 200.0,
    max_iter: int = 128,
    conv_tol: float = 1e-9,
) -> CCMUCCSDResult:
    """Open-shell UCCSD(T) on the neutral fitted-torus ``ccm`` control.

    Parameters
    ----------
    ccm : CCMSystem
    uhf_result : CCMUHFResult, optional
        Converged UHF-CCM reference on the **neutral** four-center. Built with
        ``run_ccm_uhf(ccm, eri=neutral_g)`` if omitted.
    cderi : ndarray, optional
        Neutral cderi ``L[P,mu,ν]`` (:func:`ccm_neutral_cderi`); computed if omitted.
    compute_triples : bool
        Include the perturbative ``(T)`` correction.

    Returns
    -------
    CCMUCCSDResult
    """
    guess_selection = _ccm_initial_guess(
        ccm, initial_guess, driver='run_ccm_uccsd', reference=uhf_result,
    )
    from ..ccm.uhf import run_ccm_uhf
    from .neutral import ccm_neutral_cderi

    try:
        from ...dlpno._ccsd_ref import run_ref_uccsd
    except Exception as exc:  # pragma: no cover
        raise ImportError(
            "run_ccm_uccsd needs the spin-orbital UCCSD engine "
            "vibeqc.dlpno._ccsd_ref.run_ref_uccsd") from exc

    L = np.asarray(cderi if cderi is not None else
                   ccm_neutral_cderi(ccm, ke_cutoff=ke_cutoff), dtype=float)
    if uhf_result is None:
        g = _neutral_four_center(L)
        uhf_result = run_ccm_uhf(ccm, eri=g, initial_guess=initial_guess)
    if not uhf_result.converged:
        raise ValueError("UCCSD-CCM requires a converged UHF-CCM reference.")

    Ca = np.asarray(uhf_result.mo_coeffs_alpha, dtype=float)
    Cb = np.asarray(uhf_result.mo_coeffs_beta, dtype=float)
    ea = np.asarray(uhf_result.mo_energies_alpha, dtype=float)
    eb = np.asarray(uhf_result.mo_energies_beta, dtype=float)
    na, nb = int(uhf_result.n_alpha), int(uhf_result.n_beta)

    # Neutral cderi -> a / b MO-basis B-tensors: (pq|rs)^{ss'} = S_P B^s_pq B^s'_rs.
    B_mo_a = np.einsum("Pmn,mp,nq->Ppq", L, Ca, Ca, optimize=True)
    B_mo_b = np.einsum("Pmn,mp,nq->Ppq", L, Cb, Cb, optimize=True)
    # Canonical UHF -> diagonal MO Fock.
    f_mo_a = np.diag(ea)
    f_mo_b = np.diag(eb)

    e_hf = float(uhf_result.energy)
    ref = run_ref_uccsd(
        f_mo_a, f_mo_b, B_mo_a, B_mo_b, na, nb, e_hf=e_hf,
        max_iter=max_iter, conv_tol=conv_tol, compute_triples=compute_triples)

    e_ccsd = float(ref.e_corr)
    e_t = float(ref.e_t)
    e_corr = e_ccsd + e_t
    e_tot = e_hf + e_corr
    return CCMUCCSDResult(
        converged=bool(ref.converged), n_iter=int(ref.n_iter),
        e_ccsd_correlation=e_ccsd, e_t=e_t, e_correlation=e_corr,
        e_hf=e_hf, e_total=e_tot,
        e_correlation_per_atom=e_corr / ccm.n_atoms,
        e_total_per_atom=e_tot / ccm.n_atoms, uhf=uhf_result)
