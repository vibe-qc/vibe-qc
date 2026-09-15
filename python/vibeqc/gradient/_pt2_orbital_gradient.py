"""Analytic PT2 orbital gradient dE^2/dk via effective densities.

Replaces O(n_pairs) numerical FD PT2 solves with an O(1) Fock
construction, using the same effective density corrections already
computed for the PT2-corrected g^R.
"""

from __future__ import annotations

import numpy as np


def compute_pt2_orbital_gradient(
    prep: dict,
    h1e_cas: np.ndarray,
    h2e_cas: np.ndarray,
    rdm1_cas: np.ndarray,
    rdm2_cas: np.ndarray,
    n_core: int,
    n_act: int,
    nmo: int,
    pairs: list,
    *,
    variant: str = "caspt2",
) -> np.ndarray:
    """Analytic dE^2/dk via effective densities.

    dE^2/dk_{pq} = grad_eff_{pq} - grad_CAS_{pq}

    where grad_eff uses D_CASSCF + DeltaD_PT2 and Gamma_CASSCF + DeltaGamma_PT2
    in the generalized Fock construction.

    This is O(1) instead of O(n_pairs) numerical FD.  The approximation
    (frozen effective density w.r.t. orbital rotation) is the same as used
    for the PT2-corrected g^R.

    Parameters
    ----------
    variant : str
        "caspt2" (generalized Fock H_0) or "nevpt2" (Dyall H_0).
    """
    from vibeqc.gradient._casscf import _build_casscf_fock_and_gradient
    from vibeqc.gradient._pt2_density import (
        compute_nevpt2_effective_density,
        compute_pt2_effective_density,
    )

    # Effective densities
    if variant == "nevpt2":
        DeltaD_full, DeltaGamma_full = compute_nevpt2_effective_density(
            prep, n_core, n_act
        )
    else:
        DeltaD_full, DeltaGamma_full = compute_pt2_effective_density(
            prep, n_core, n_act
        )

    eri_chem = h2e_cas.transpose(0, 2, 1, 3)  # physicist -> chemist
    act_s = slice(n_core, n_core + n_act)

    # CASSCF baseline gradient
    grad_cas, _F_cas, _ = _build_casscf_fock_and_gradient(
        h1e_cas, eri_chem, rdm1_cas, rdm2_cas, n_core, n_act, nmo, pairs
    )

    # Effective gradient (CASSCF + PT2 corrections)
    dm1_eff = rdm1_cas.copy()
    dm1_eff += DeltaD_full[act_s, act_s]
    dm2_eff = rdm2_cas.copy()
    dm2_eff += DeltaGamma_full[act_s, act_s, act_s, act_s]

    grad_eff, _F_eff, _ = _build_casscf_fock_and_gradient(
        h1e_cas, eri_chem, dm1_eff, dm2_eff, n_core, n_act, nmo, pairs
    )

    return grad_eff - grad_cas
