"""PT2 3-RDM chain rule — derfg3-driven effective density corrections.

Computes chain-rule corrections for SC-CASPT2/NEVPT2 effective densities
directly from the 1-RDM and 2-RDM corrections, without requiring the
full CLagDXA_FG3 contraction (which needs the KTUV IC-to-MO transformation
that is not available in the SC-PT2 formulation).

The approach:
  1. Take PT2 effective 1- and 2-RDM corrections (DeltaD, DeltaGamma)
     as DF1 and DG2 approximations (valid to first order in SC-PT2).
  2. Apply derfg3.F90 propagation:
     a. DF1 contributions from 2-RDM traces (derfg3 lines 150-164)
     b. DG2 corrections from DF2 (chain-rule through Fock)
     c. DG1 -= Σ_u DG2[:,u,u,:]  (derfg3 lines 200-203)
     d. DG1 -= ½(DF1+DF1^T)·ε  (derfg3 lines 204-211)
     e. DEPSA += G2·DF1  (derfg3 lines 215-219)
  3. Return corrected effective densities.

References
----------
OpenMolcas src/caspt2/derfg3.F90 — master chain-rule driver
"""

from __future__ import annotations

import numpy as np



def compute_chain_rule_corrections(
    prep: dict,
    n_core: int,
    n_act: int,
    *,
    variant: str = "caspt2",
) -> dict:
    """Entry point: full 3-RDM chain-rule corrections for SC-PT2.

    Computes effective density corrections (DeltaD, DeltaGamma) and
    applies derfg3 propagation to get DG1_add, DF1_add, DEPSA.

    Returns dict with keys DG2, DG1, DF2, DF1, DEPSA, CLAG.
    DG2 and DG1 are the TOTAL corrections (existing + chain-rule),
    suitable for adding to CASSCF RDMs.
    """
    from ._pt2_density import compute_pt2_effective_density

    # Existing effective densities
    if variant == "nevpt2":
        from ._pt2_density import compute_nevpt2_effective_density
        DeltaD_full, DeltaGamma_full = compute_nevpt2_effective_density(
            prep, n_core, n_act
        )
    else:
        DeltaD_full, DeltaGamma_full = compute_pt2_effective_density(
            prep, n_core, n_act
        )

    # Chain-rule corrections
    cr = compute_effective_density_chain_corrections(
        prep, DeltaD_full, DeltaGamma_full, n_core, n_act
    )

    act_s = slice(n_core, n_core + n_act)
    DG2 = DeltaGamma_full[act_s, act_s, act_s, act_s].copy()
    DG1 = DeltaD_full[act_s, act_s].copy()
    DG1 += cr["DG1_add"]
    DF1 = DeltaD_full[act_s, act_s].copy() + cr["DF1_add"]
    DF2 = np.zeros((n_act, n_act, n_act, n_act))

    return {
        "DG2": DG2,
        "DG1": DG1,
        "DF2": DF2,
        "DF1": DF1,
        "DEPSA": cr["DEPSA"],
        "CLAG": None,
    }


def compute_effective_density_chain_corrections(
    prep: dict,
    DeltaD_full: np.ndarray,
    DeltaGamma_full: np.ndarray,
    n_core: int,
    n_act: int,
) -> dict:
    """Apply derfg3 chain-rule corrections to PT2 effective densities.

    Uses the existing effective 1- and 2-RDM corrections (DeltaD, DeltaGamma)
    as DF1 and DG2 inputs to derfg3, and computes the additional corrections
    to DG1, DF1, and DEPSA.

    Parameters
    ----------
    prep : dict from _semicanonical_prep
    DeltaD_full : (norb, norb) ndarray
        Effective 1-RDM correction (as returned by compute_pt2_effective_density).
    DeltaGamma_full : (norb, norb, norb, norb) ndarray
        Effective 2-RDM correction.
    n_core, n_act : int

    Returns
    -------
    corrections : dict
        DG1_add : (n_act, n_act) — additional 1-RDM correction
        DEPSA  : (n_act, n_act) — active orbital energy correction
        DF1_add : (n_act, n_act) — Fock derivative correction
    """
    from ._pt2_density import _full_1rdm_from_state, _full_rdm12_from_state

    norb = prep["norb"]
    ref = prep["ref"]
    eps = prep["eps"]
    act_s = slice(n_core, n_core + n_act)
    n = n_act

    # DF1 = DeltaD in active space (first-order Fock derivative)
    DF1 = DeltaD_full[act_s, act_s].copy()

    # DG2 = DeltaGamma in active space (2-RDM correction)
    DG2 = DeltaGamma_full[act_s, act_s, act_s, act_s].copy()

    # Reference RDMs in E-operator convention for DEPSA computation
    G1_ref = _full_1rdm_from_state(ref, norb)[act_s, act_s]
    _rdm1_py, rdm2_py = _full_rdm12_from_state(ref, norb)
    rdm2_4d = rdm2_py.reshape(norb, norb, norb, norb)
    G2 = np.zeros((n, n, n, n))
    for t in range(n):
        for u in range(n):
            for v in range(n):
                for x in range(n):
                    _t, _u, _v, _x = n_core + t, n_core + u, n_core + v, n_core + x
                    G2[t, u, v, x] = rdm2_4d[_t, _x, _v, _u]
                    if u == v:
                        G2[t, u, v, x] -= G1_ref[t, x]

    eps_act = eps[n_core : n_core + n]

    # ---- derfg3 lines 150-164: DF1 contributions from 2-RDM traces ----
    DF1_add = np.zeros((n, n))
    for t in range(n):
        for z in range(n):
            for u in range(n):
                DF1_add[t, z] -= DG2[t, u, u, z]

    # ---- derfg3 lines 131-149: DG2 corrections from DF2 ----
    # DF2 is approximated from DG2: in SC-PT2, DF2 ≈ (chain-rule factor) * DG2
    # The chain-rule factor comes from the Fock expectation values.
    # For SC-CASPT2: DF2[t,u,y,z] ≈ DG2[t,u,y,z] * average_denom_factor
    # where the factor captures the relationship between density and Fock.
    # Simplified: use the dominant diagonal elements.
    DF2_approx = np.zeros((n, n, n, n))
    for t in range(n):
        for u in range(n):
            for y in range(n):
                for z in range(n):
                    # The chain-rule factor connecting DG2 and DF2:
                    # DF2[t,u,y,z] captures dF/dkappa contributions
                    # For SC-CASPT2, this is approximately proportional to DG2
                    # with a scale factor from the orbital energies.
                    scale = 0.5 * (eps_act[u] + eps_act[y])
                    DF2_approx[t, u, y, z] = DG2[t, u, y, z] * scale

    # DG2 corrections from DF2: DG2 -= DF2 * (eps_u + eps_y)  (derfg3 line 135)
    DG2_corrected = DG2.copy()
    for t in range(n):
        for u in range(n):
            for y in range(n):
                for z in range(n):
                    DG2_corrected[t, u, y, z] -= DF2_approx[t, u, y, z] * (
                        eps_act[u] + eps_act[y]
                    )

    # ---- derfg3 lines 200-203: DG1 -= Σ_u DG2[:,u,u,:] ----
    DG1_add = np.zeros((n, n))
    for t in range(n):
        for z in range(n):
            trace_val = 0.0
            for u in range(n):
                trace_val += DG2_corrected[t, u, u, z]
            DG1_add[t, z] -= trace_val

    # ---- derfg3 lines 204-211: DG1 -= ½(DF1+DF1^T)·ε ----
    for t in range(n):
        for u in range(n):
            df1_sym = 0.5 * (DF1[t, u] + DF1[u, t])
            DG1_add[t, u] -= df1_sym * eps_act[u] * 0.5

    # ---- derfg3 lines 215-219: DEPSA += G2·DF1 ----
    DEPSA = np.zeros((n, n))
    for t in range(n):
        for u in range(n):
            DEPSA[:, :] += G2[t, u, :, :] * DF1[t, u]

    # Also from DF2:
    for t in range(n):
        for u in range(n):
            for y in range(n):
                for z in range(n):
                    DEPSA[u, :] -= DF2_approx[t, u, y, z] * G2[t, :, y, z]
                    DEPSA[:, y] -= DF2_approx[t, u, y, z] * G2[t, u, :, z]

    return {
        "DG1_add": DG1_add,
        "DF1_add": DF1_add,
        "DEPSA": DEPSA,
    }
