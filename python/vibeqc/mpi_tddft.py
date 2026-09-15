"""MPI-aware TDDFT helpers -- parallel excitation analysis.

Integrates ``vibeqc.mpi`` with ``vibeqc.tddft`` for distributed
excitation-energy computation.  When MPI is active, independent
work items (k-points, excitation states, or AO shell pairs for
the ERI transform) are farmed across ranks.

Usage
-----
When launched under ``mpirun -np N``, the module automatically
distributes work.  In serial mode (no mpirun), it delegates
directly to the standard ``vibeqc.tddft`` functions.
"""

from __future__ import annotations

from typing import List, Optional

import numpy as np

from .mpi import distribute_tasks, mpi_allreduce, mpi_available, mpi_gather, mpi_rank
from .tddft import TDDFTResult, TDDFTState, oscillator_strength

__all__ = [
    "mpi_tddft_analysis",
]


def _transition_dipole_from_X(
    X: np.ndarray,
    n_occ: int,
    n_virt: int,
    mo_coeff: np.ndarray,
    dipole_ao,
) -> np.ndarray:
    """Compute transition dipole for a single excitation vector."""
    C_occ = mo_coeff[:, :n_occ]
    C_virt = mo_coeff[:, n_occ:]
    X_ia = X.reshape(n_occ, n_virt)
    P_ao = C_occ @ X_ia @ C_virt.T + C_virt @ X_ia.T @ C_occ.T
    mu = np.zeros(3)
    for d, comp in enumerate([dipole_ao.x, dipole_ao.y, dipole_ao.z]):
        mu[d] = np.trace(P_ao @ np.asarray(comp))
    return mu


def mpi_tddft_analysis(
    eigenvals: np.ndarray,
    eigenvecs: np.ndarray,
    mo_coeff: np.ndarray,
    n_occ: int,
    dipole_ao,
) -> List[TDDFTState]:
    """Compute TDDFT state properties (transition dipoles, oscillator
    strengths) in parallel across MPI ranks.

    Each rank processes a subset of excitation states and computes
    transition properties locally.  Results are gathered to rank 0
    and broadcast to all ranks.

    Parameters
    ----------
    eigenvals : np.ndarray
        Excitation energies in Hartree, shape ``(n_states,)``.
    eigenvecs : np.ndarray
        Excitation vectors, shape ``(n_pair, n_states)``.
    mo_coeff : np.ndarray
        MO coefficients, shape ``(n_basis, n_basis)``.
    n_occ : int
        Number of occupied orbitals.
    dipole_ao
        Dipole integrals object with ``.x``, ``.y``, ``.z`` attributes.

    Returns
    -------
    list of TDDFTState
        Computed excited states, sorted by energy.  Identical across
        all ranks.
    """
    n_states = len(eigenvals)
    n_virt = mo_coeff.shape[0] - n_occ

    if not mpi_available():
        # Serial path -- compute all states locally
        states = []
        for k in range(n_states):
            omega = float(abs(eigenvals[k]))
            X = eigenvecs[:, k]
            td = _transition_dipole_from_X(X, n_occ, n_virt, mo_coeff, dipole_ao)
            f_osc = oscillator_strength(omega, td)
            states.append(
                TDDFTState(
                    index=k + 1,
                    excitation_energy=omega,
                    excitation_energy_ev=omega * 27.211386245988,
                    wavelength_nm=45.5633526 / max(omega, 1e-12),
                    oscillator_strength=f_osc,
                    transition_dipole=td,
                )
            )
        return states

    # MPI path -- distribute states across ranks
    start, end = distribute_tasks(n_states)
    local_states = []

    for k in range(start, end):
        omega = float(abs(eigenvals[k]))
        X = eigenvecs[:, k]
        td = _transition_dipole_from_X(X, n_occ, n_virt, mo_coeff, dipole_ao)
        f_osc = oscillator_strength(omega, td)
        local_states.append(
            TDDFTState(
                index=k + 1,
                excitation_energy=omega,
                excitation_energy_ev=omega * 27.211386245988,
                wavelength_nm=45.5633526 / max(omega, 1e-12),
                oscillator_strength=f_osc,
                transition_dipole=td,
            )
        )

    # Gather all states to rank 0, then broadcast
    all_states = mpi_gather(local_states, root=0)
    if mpi_rank() == 0:
        # Flatten and sort by energy
        flat = []
        for sublist in all_states:
            flat.extend(sublist)
        flat.sort(key=lambda s: s.excitation_energy)
        from .mpi import mpi_bcast

        return mpi_bcast(flat, root=0)

    from .mpi import mpi_bcast

    return mpi_bcast(None, root=0)
