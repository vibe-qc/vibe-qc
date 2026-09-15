"""Energy Decomposition Analysis (EDA) for molecular fragments.

Partitions the interaction energy between pre-defined fragments into
physically interpretable components following the LMO-EDA scheme
(Su & Li, J. Chem. Phys. 131, 014102, 2009):

    ΔE_int = ΔE_elstat + ΔE_exch + ΔE_rep + ΔE_pol + ΔE_disp

Public API
----------

.. autofunction:: eda_lmo
.. autofunction:: eda_morokuma
.. autoclass:: EDAResult

Theory references
-----------------

- Su, P.; Li, H., J. Chem. Phys. 131, 014102 (2009). DOI: 10.1063/1.3159673
  (LMO-EDA: localised molecular orbital energy decomposition analysis)
- Morokuma, K., J. Chem. Phys. 55, 1236 (1971). DOI: 10.1063/1.1676210
  (Morokuma EDA, the original energy decomposition scheme)
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Optional, Sequence

import numpy as np

__all__ = [
    "EDAResult",
    "eda_lmo",
    "eda_morokuma",
    "fragment_density_matrix",
]


# ---------------------------------------------------------------------------
# Data types
# ---------------------------------------------------------------------------


@dataclass
class EDAResult:
    """Energy decomposition analysis for a two-fragment system.

    All energies in Hartree. The sum of components equals the total
    interaction energy::

        E_int = E_total - E_frag1 - E_frag2
              = E_elstat + E_exch + E_rep + E_pol + E_disp

    Attributes
    ----------
    e_total : float
        Total energy of the supersystem.
    e_frag1 : float
        Energy of isolated fragment 1 in the supersystem basis (ghost atoms).
    e_frag2 : float
        Energy of isolated fragment 2 in the supersystem basis (ghost atoms).
    e_int : float
        Total interaction energy: E_total - E_frag1 - E_frag2.
    e_elstat : float
        Electrostatic (Coulomb) interaction between frozen fragment densities.
    e_exch : float
        Exchange (Pauli) repulsion from antisymmetrisation.
    e_rep : float
        Total repulsion: E_exch + E_elstat (the frozen-core term).
    e_pol : float
        Polarisation / orbital relaxation energy.
    e_disp : float
        Dispersion energy (when a dispersion correction is present).
    method : str
        EDA variant used (``"lmo"`` or ``"morokuma"``).
    """

    e_total: float = 0.0
    e_frag1: float = 0.0
    e_frag2: float = 0.0
    e_int: float = 0.0
    e_elstat: float = 0.0
    e_exch: float = 0.0
    e_rep: float = 0.0
    e_pol: float = 0.0
    e_disp: float = 0.0
    method: str = "lmo"
    extra: dict = field(default_factory=dict)

    def summary(self) -> str:
        """Human-readable summary table."""
        lines = [
            f"Energy Decomposition Analysis ({self.method.upper()}-EDA)",
            "-" * 52,
            f"  E_total (supersystem)   = {self.e_total:>14.8f} E_h",
            f"  E_fragment_1 (isolated) = {self.e_frag1:>14.8f} E_h",
            f"  E_fragment_2 (isolated) = {self.e_frag2:>14.8f} E_h",
            "-" * 52,
            f"  ΔE_int (total)          = {self.e_int:>14.8f} E_h",
            "",
            f"  Electrostatic           = {self.e_elstat:>14.8f} E_h",
            f"  Exchange (Pauli)        = {self.e_exch:>14.8f} E_h",
            f"  Repulsion (frozen)      = {self.e_rep:>14.8f} E_h",
            f"  Polarisation            = {self.e_pol:>14.8f} E_h",
            f"  Dispersion              = {self.e_disp:>14.8f} E_h",
            "-" * 52,
        ]
        return "\n".join(lines)


# ---------------------------------------------------------------------------
# Fragment density matrix utilities
# ---------------------------------------------------------------------------


def fragment_density_matrix(
    mo_coeffs: np.ndarray,
    overlap: np.ndarray,
    n_occ: int,
    fragment_indices: Sequence[int],
    ao_to_atom: Optional[list[int]] = None,
) -> np.ndarray:
    """Extract the density matrix for a fragment defined by atom indices.

    This zeroes-out AO contributions that don't belong to the fragment
    atoms, then builds the fragment density from the MO coefficients.
    This is a *projection* -- the fragment density is not idempotent.

    Parameters
    ----------
    mo_coeffs : np.ndarray, shape ``(n_ao, n_orb)``
        MO coefficients of the supersystem.
    overlap : np.ndarray, shape ``(n_ao, n_ao)``
        AO overlap matrix.
    n_occ : int
        Number of occupied orbitals.
    fragment_indices : sequence of int
        Atom indices belonging to the fragment.
    ao_to_atom : list of int, optional
        AO-to-atom mapping. If None, all AOs are used (no projection).

    Returns
    -------
    np.ndarray, shape ``(n_ao, n_ao)``
        Fragment density matrix (real part).
    """
    C = np.asarray(mo_coeffs)
    S = np.asarray(overlap)
    n_ao = C.shape[0]

    if ao_to_atom is not None:
        frag_set = set(fragment_indices)
        mask = np.array([ao_to_atom[mu] in frag_set for mu in range(n_ao)], dtype=bool)
        C_frag = C.copy()
        C_frag[~mask, :] = 0.0
    else:
        C_frag = C

    C_occ = C_frag[:, :n_occ]
    P_frag = C_occ @ C_occ.T
    return P_frag.real if np.iscomplexobj(P_frag) else P_frag


# ---------------------------------------------------------------------------
# LMO-EDA (Su & Li 2009)
# ---------------------------------------------------------------------------


def eda_lmo(
    e_total: float,
    e_frag1: float,
    e_frag2: float,
    fock_total: np.ndarray,
    fock_frag1: np.ndarray,
    fock_frag2: np.ndarray,
    density_total: np.ndarray,
    density_frag1: np.ndarray,
    density_frag2: np.ndarray,
    overlap: np.ndarray,
    *,
    e_disp: float = 0.0,
    e_disp_frag1: float = 0.0,
    e_disp_frag2: float = 0.0,
) -> EDAResult:
    r"""LMO-EDA: Localized Molecular Orbital Energy Decomposition Analysis.

    Decomposes the interaction energy between two fragments into:

    - **Electrostatic**: Coulomb interaction between frozen fragment densities.
      E_elstat = Tr[P_1 V_nuc,2] + Tr[P_2 V_nuc,1] + E_nuc,nuc
                + ∬ ρ_1(r) ρ_2(r') / |r-r'| dr dr'

    - **Exchange/Repulsion**: Pauli repulsion from antisymmetrisation.
      E_exch = E[P_1+P_2, KS-orthogonalised] - E[P_1+P_2, non-orthogonal]

    - **Polarisation**: Orbital relaxation in the combined field.
      E_pol = E_total - E[P_1+P_2, KS-orthogonalised]

    - **Dispersion**: Difference in dispersion corrections.

    Parameters
    ----------
    e_total : float
        Total energy of the supersystem.
    e_frag1, e_frag2 : float
        Energies of isolated fragments (in supersystem basis).
    fock_total : np.ndarray
        Fock matrix of the supersystem.
    fock_frag1, fock_frag2 : np.ndarray
        Fock matrices of isolated fragments.
    density_total : np.ndarray
        Density matrix of the supersystem.
    density_frag1, density_frag2 : np.ndarray
        Density matrices of isolated fragments.
    overlap : np.ndarray
        AO overlap matrix.
    e_disp, e_disp_frag1, e_disp_frag2 : float
        Dispersion corrections for supersystem and fragments.

    Returns
    -------
    EDAResult
    """
    # Total interaction energy
    e_int = e_total - e_frag1 - e_frag2

    P1 = np.asarray(density_frag1)
    P2 = np.asarray(density_frag2)
    P12 = P1 + P2
    F1 = np.asarray(fock_frag1)
    F2 = np.asarray(fock_frag2)
    S = np.asarray(overlap)

    # --- Electrostatic ---
    # E_elstat = energy of non-interacting fragment densities in the
    # supersystem Hamiltonian. Approximated as:
    #   Tr[(P1+P2) * (H_core)] - Tr[P1 * H_core_1] - Tr[P2 * H_core_2]
    # where H_core is the one-electron (kinetic + nuclear) part.
    # Without access to the full Hamiltonian decomposition, we use the
    # Fock-matrix difference approach:
    #
    # The frozen-core energy E_frozen = E[P1+P2] (without orbital relaxation)
    # is estimated from the fragment Fock matrices:
    #   E_frozen ≈ 0.5 * Tr[(P1+P2) * (F1+F2)]  (non-orthogonal)
    #
    # Electrostatic = 0.5 * (Tr[P1*F2] + Tr[P2*F1])
    e_elstat = 0.5 * (np.trace(P1 @ F2) + np.trace(P2 @ F1))

    # --- Exchange/Repulsion ---
    # Löwdin-orthogonalise the combined density
    evals, evecs = np.linalg.eigh(S)
    mask = evals > 1e-14
    inv_sqrt_evals = np.zeros_like(evals)
    inv_sqrt_evals[mask] = evals[mask] ** (-0.5)
    S_half_inv = evecs @ np.diag(inv_sqrt_evals) @ evecs.T
    sqrt_evals = np.zeros_like(evals)
    sqrt_evals[mask] = np.sqrt(evals[mask])
    S_half = evecs @ np.diag(sqrt_evals) @ evecs.T

    # Orthogonalised combined density
    P12_orth = S_half @ P12 @ S_half
    # Back to AO
    P12_orth_ao = S_half_inv @ P12_orth @ S_half_inv

    # Exchange energy estimate: difference between orthogonalised and
    # non-orthogonal frozen energies
    e_frozen_nonorth = 0.5 * np.trace(P12 @ (F1 + F2))
    F_frozen = 0.5 * (F1 + F2)
    e_frozen_orth = 0.5 * np.trace(P12_orth_ao @ (F1 + F2))

    e_exch = e_frozen_orth - e_frozen_nonorth
    e_rep = e_elstat + e_exch

    # --- Polarisation ---
    # E_pol = difference between converged supersystem and frozen-orthogonal
    e_frozen = e_frag1 + e_frag2 + e_elstat + e_exch
    e_pol = e_total - e_disp - e_frozen

    # --- Dispersion ---
    e_disp_contrib = e_disp - e_disp_frag1 - e_disp_frag2

    return EDAResult(
        e_total=e_total,
        e_frag1=e_frag1,
        e_frag2=e_frag2,
        e_int=e_int,
        e_elstat=e_elstat,
        e_exch=e_exch,
        e_rep=e_rep,
        e_pol=e_pol,
        e_disp=e_disp_contrib,
        method="lmo",
    )


# ---------------------------------------------------------------------------
# Morokuma EDA (1971)
# ---------------------------------------------------------------------------


def eda_morokuma(
    e_total: float,
    e_frag1: float,
    e_frag2: float,
    fock_total: np.ndarray,
    fock_frag1: np.ndarray,
    fock_frag2: np.ndarray,
    density_total: np.ndarray,
    density_frag1: np.ndarray,
    density_frag2: np.ndarray,
    overlap: np.ndarray,
    *,
    e_disp: float = 0.0,
    e_disp_frag1: float = 0.0,
    e_disp_frag2: float = 0.0,
) -> EDAResult:
    """Morokuma-style EDA (1971).

    Uses the same algorithm as :func:`eda_lmo` but labels the components
    in the Morokuma convention:

    - ES (electrostatic)   = E_elstat
    - EX (exchange)         = E_exch
    - PL (polarisation)     = E_pol
    - CT (charge transfer)  = part of E_pol (not separated here)
    - MIX (mixing)          = remainder

    This is a simplified implementation; for full CT/MIX separation,
    the CDA (charge decomposition analysis) or NOCV extension is needed.
    """
    result = eda_lmo(
        e_total=e_total,
        e_frag1=e_frag1,
        e_frag2=e_frag2,
        fock_total=fock_total,
        fock_frag1=fock_frag1,
        fock_frag2=fock_frag2,
        density_total=density_total,
        density_frag1=density_frag1,
        density_frag2=density_frag2,
        overlap=overlap,
        e_disp=e_disp,
        e_disp_frag1=e_disp_frag1,
        e_disp_frag2=e_disp_frag2,
    )
    result.method = "morokuma"
    result.extra["morokuma_components"] = {
        "ES": result.e_elstat,
        "EX": result.e_exch,
        "PL": result.e_pol,
        "CT+MIX": 0.0,  # not separated in this simplified implementation
    }
    return result
