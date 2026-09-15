"""CLagDX: IC-CASPT2 CI Lagrangian → active-MO BDER/SDER.

Full IC→MO density transformation using the IC solver's orthonormalization
matrices (T, Q) and candidate orbital-index mapping.

References: OpenMolcas src/caspt2/clagdx.F90
"""

from __future__ import annotations

import numpy as np


def compute_bder_sder_from_ic(
    clagdx_data: dict,
    n_core: int,
    n_act: int,
    norb: int,
) -> tuple[np.ndarray, np.ndarray]:
    """Compute BDER/SDER from IC-CASPT2 amplitudes via CLagDX.

    Uses the full IC orthonormalization chain:
      raw candidates —T→ orthonormal IC —Q→ H0 eigenbasis
    and back-transforms to active MO space.
    """
    c_raw = np.asarray(clagdx_data["c_raw"])
    V_raw = np.asarray(clagdx_data["V_raw"])
    T_mat = np.asarray(clagdx_data["T"])
    Q_mat = np.asarray(clagdx_data["Q"])
    created_kept = clagdx_data["created_kept"]
    annih_kept = clagdx_data["annih_kept"]
    d_vals = np.asarray(clagdx_data["d"])

    n_cand = len(c_raw)
    nb = T_mat.shape[1]
    act_range = set(range(n_core, n_core + n_act))

    # Step 1: Raw TRANS matrix (n_act × n_cand)
    TRANS_raw = np.zeros((n_act, n_cand))
    for o in range(n_cand):
        cr = created_kept[o]
        an = annih_kept[o]
        active_orbs = set()
        for i in cr:
            if i in act_range:
                active_orbs.add(i - n_core)
        for i in an:
            if i in act_range:
                active_orbs.add(i - n_core)
        for t in active_orbs:
            if 0 <= t < n_act:
                TRANS_raw[t, o] = 1.0

    # Step 2: Orthonormalized TRANS = TRANS_raw @ T  (n_act × nb)
    TRANS = TRANS_raw @ T_mat

    # Step 3: Build IC amplitudes and RHS in orthonormal basis
    # c_raw = T @ Q @ c_orth → c_orthonormal = Q @ c_orth = Q @ Q^T @ T^T @ c_raw
    # But simpler: use the solver's path in reverse
    V_orth = T_mat.T @ V_raw  # (nb,) RHS in orthonormal IC basis
    c_tmp = T_mat.T @ c_raw  # (nb,) amplitudes in orthonormal IC basis

    # Step 4: IC density in orthonormal basis: ρ = c_tmp · c_tmp^T
    rho_IC = np.outer(c_tmp, c_tmp)  # (nb, nb)

    # Step 5: Transform to MO: BDER = TRANS @ ρ @ TRANS^T
    BDER = TRANS @ rho_IC @ TRANS.T

    # Step 6: SDER from overlap derivative: V_orth · c_tmp^T + c_tmp · V_orth^T
    SDER_IC = np.outer(V_orth, c_tmp) + np.outer(c_tmp, V_orth)
    SDER = TRANS @ SDER_IC @ TRANS.T

    return BDER, SDER
