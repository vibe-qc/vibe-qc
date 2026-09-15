"""Finite-difference nuclear Hessian on the GAPW / GPW periodic route.

Builds the 3N x 3N atomic Hessian by central differences on the
*analytic atomic gradient* produced by
:func:`vibeqc.periodic_gapw_gradient.compute_gradient_gpw`. Cheaper
and noticeably less noisy than central differences on the total
energy (one fewer numerical-differentiation step is taken, and the
analytic gradient pieces of each force evaluation carry their full
accuracy through).

Diagonalising the mass-weighted Hessian gives harmonic vibrational
frequencies in cm⁻¹ via the same convention the molecular Hessian
code (:mod:`vibeqc.hessian`) uses: positive wavenumbers for real
modes, negative wavenumbers for imaginary (saddle-point) modes.

API
---
::

    import vibeqc

    H = vibeqc.compute_hessian_gpw(
        system, basis, result,
        fd_step_bohr=0.02,
        functional=None,
    )
    freqs_cm1, modes = vibeqc.compute_vibrational_frequencies(
        H, masses_amu=[1.008, 1.008],
    )

Algorithm
---------
For each Cartesian DOF ``q = (atom_i, axis_j)``:

1.  Displace atom ``i`` by ``+fd_step`` along axis ``j``.
2.  Re-run the GPW SCF, seeded with the central-geometry density as
    the initial guess (substantial speedup; usually 2-4 SCF
    iterations to reconverge).
3.  Compute the analytic atomic gradient at the displaced geometry
    via :func:`compute_gradient_gpw`.
4.  Repeat for ``-fd_step``.
5.  Row ``3i+j`` of the Hessian is
    ``H[q, :] = -(F(+q) - F(-q)) / (2 . fd_step)``
    where ``F = -dE/dR`` is the force. Equivalently,
    ``H[q, :] = (G(+q) - G(-q)) / (2 . fd_step)`` with ``G = dE/dR``.

Finally symmetrise ``H = 0.5 . (H + H.T)`` to wash out the
asymmetric FD truncation.

Cost
----
``6 . N`` SCF + analytic-gradient evaluations. Each SCF is warm-
started so the per-displacement SCF cost is much lower than the
central-geometry one. For STO-3G H2 / He cells this runs in a few
seconds.

Scope (v0.12 R3 first cut)
--------------------------
* Γ-only closed-shell RHF / RKS (the scope of the analytic GPW
  gradient itself). Open-shell + multi-k Hessian is post-R3.
* No translation/rotation projection -- for periodic cells in vacuum-
  padded boxes the three translational acoustic modes naturally
  collect near zero (this is the standard non-projected convention
  used by VASP / Quantum Espresso / GPAW for solid-state phonons).

Convention notes
----------------
* Imaginary modes (saddle points) are reported as **negative**
  wavenumbers (matches :mod:`vibeqc.hessian` + Gaussian / NWChem /
  ORCA / PySCF / ASE).
* Mass weighting uses atomic-mass units (amu); the user supplies
  the per-atom mass list. For isotope effects, swap entries (e.g.
  ``1.008 -> 2.014`` for deuterium).
"""

from __future__ import annotations

from typing import Any, Optional, Sequence

import numpy as np

from ._vibeqc_core import (
    Atom,
    BasisSet,
    Molecule,
    PeriodicSystem,
)
from .periodic_gapw_gradient import compute_gradient_gpw
from .periodic_gapw_j import run_periodic_rhf_gpw


__all__ = [
    "compute_hessian_gpw",
    "compute_vibrational_frequencies",
]


# CODATA 2018: 1 a.u. of angular frequency in cm⁻¹.
# Matches :mod:`vibeqc.hessian` (1 Ha = 219474.6313632 cm⁻¹).
_HARTREE_TO_CM_INV: float = 219474.6313632

# 1 atomic mass unit in electron-mass units (so Hessian [Ha/bohr^2]
# divided by mass [m_e] gives w^2 [a.u.^2]). Same constant the
# molecular Hessian uses.
_AMU_TO_ELECTRON_MASS: float = 1822.888486209


def _displaced_system(
    system: PeriodicSystem, atom_idx: int, cart: int, delta: float,
) -> PeriodicSystem:
    """Return a copy of ``system`` with atom ``atom_idx`` shifted by
    ``delta`` bohr along Cartesian axis ``cart``. Lattice unchanged.
    Mirrors :func:`vibeqc.periodic_gapw_gradient._displaced_system`.
    """
    new_atoms = []
    for i, atom in enumerate(system.unit_cell):
        xyz = list(atom.xyz)
        if i == atom_idx:
            xyz[cart] += float(delta)
        new_atoms.append(Atom(int(atom.Z), xyz))
    return PeriodicSystem(
        system.dim,
        np.asarray(system.lattice, dtype=np.float64),
        new_atoms,
        charge=system.charge,
        multiplicity=system.multiplicity,
    )


def _displaced_basis(system: PeriodicSystem, basis_name: str) -> BasisSet:
    """Rebuild a basis on ``system`` using ``basis_name``."""
    mol = Molecule(list(system.unit_cell), 0, 1)
    return BasisSet(mol, basis_name)


def _gradient_at_displacement(
    system: PeriodicSystem,
    basis_name: str,
    atom_idx: int,
    cart: int,
    delta: float,
    *,
    initial_density: np.ndarray,
    cutoff_ha: float,
    functional: Optional[str],
    v_ne_convention: str,
    smearing_alpha: Optional[float],
    fd_step_gradient_bohr: float,
) -> np.ndarray:
    """Reconverge SCF at the displaced geometry, return analytic
    gradient (Ha/bohr) shape ``(n_atoms, 3)``. ``initial_density`` is
    the converged central-geometry density used as a warm start.
    """
    sys_disp = _displaced_system(system, atom_idx, cart, delta)
    basis_disp = _displaced_basis(sys_disp, basis_name)
    res = run_periodic_rhf_gpw(
        sys_disp,
        basis_disp,
        cutoff_ha=cutoff_ha,
        functional=functional,
        v_ne_convention=v_ne_convention,
        smearing_alpha=smearing_alpha,
        initial_density=initial_density,
        quiet=True,
    )
    if not res.converged:
        raise RuntimeError(
            f"compute_hessian_gpw: displaced SCF did not converge "
            f"(atom {atom_idx}, axis {cart}, step {delta:+.4f} bohr)."
        )
    grad = compute_gradient_gpw(
        sys_disp,
        basis_disp,
        res,
        basis_name=basis_name,
        v_ne_convention=v_ne_convention,
        smearing_alpha=smearing_alpha,
        functional=functional,
        fd_step_bohr=fd_step_gradient_bohr,
    )
    return np.asarray(grad, dtype=np.float64)


def compute_hessian_gpw(
    system: PeriodicSystem,
    basis: BasisSet,
    result: Any,
    *,
    basis_name: str = "sto-3g",
    fd_step_bohr: float = 0.02,
    functional: Optional[str] = None,
    v_ne_convention: str = "ewald",
    smearing_alpha: Optional[float] = None,
    fd_step_gradient_bohr: float = 1e-3,
) -> np.ndarray:
    """Build the 3N x 3N atomic Hessian by central differences on
    the analytic GPW gradient.

    Parameters
    ----------
    system, basis
        :class:`PeriodicSystem` and :class:`BasisSet` the SCF ran on.
    result
        Converged :class:`GpwScfResult` at the central geometry. Used
        only to warm-start the per-displacement SCFs (its converged
        density is the initial guess).
    basis_name
        Basis-set name; needed so per-displacement basis sets can be
        rebuilt. Defaults to ``"sto-3g"``; pass the actual name used
        by the SCF.
    fd_step_bohr
        Central-difference step on atomic positions for the Hessian
        (default 0.02 bohr ≈ 0.01 Å). Larger than the gradient FD
        step because we differentiate the *gradient* here, which is
        a tidier numerical signal than the energy.
    functional, v_ne_convention, smearing_alpha
        Same kwargs the central SCF used. Threaded to the per-
        displacement SCFs and the analytic-gradient call.
    fd_step_gradient_bohr
        Step the analytic GPW gradient uses internally for its
        Hellmann-Feynman FD piece. Default ``1e-3`` matches
        :func:`compute_gradient_gpw`'s default.

    Returns
    -------
    np.ndarray, shape ``(3 * n_atoms, 3 * n_atoms)``
        The atomic Hessian in Hartree/bohr^2. Symmetrised
        ``H = 0.5 . (H + H.T)`` to absorb FD asymmetry.

    Notes
    -----
    Index convention: row ``3*i + j`` corresponds to ``(atom i,
    axis j)`` -- standard "atoms outer, axes inner" packing matching
    the ASE Vibrations module and :mod:`vibeqc.hessian`.

    Cost: ``6 . N`` warm-started SCFs + ``6 . N`` analytic-gradient
    builds. For closed-shell H2 / He STO-3G cells the whole pass
    completes in a few seconds.
    """
    if not result.converged:
        raise ValueError(
            "compute_hessian_gpw: central GpwScfResult is not converged."
        )
    if system.dim != 3:
        raise NotImplementedError(
            f"compute_hessian_gpw: only 3D periodic cells supported; "
            f"got dim = {system.dim}."
        )

    n_atoms = len(system.unit_cell)
    n_dof = 3 * n_atoms

    # Warm-start density and PW cutoff inherited from the central
    # SCF (so per-displacement SCFs use the same Hartree grid).
    D_central = np.asarray(result.density, dtype=np.float64).copy()
    cutoff_ha = 300.0
    grid = getattr(result, "grid", None)
    if grid is not None and hasattr(grid, "cutoff_ha"):
        cutoff_ha = float(grid.cutoff_ha)

    H = np.zeros((n_dof, n_dof), dtype=np.float64)
    h = float(fd_step_bohr)

    for a in range(n_atoms):
        for d in range(3):
            grad_plus = _gradient_at_displacement(
                system, basis_name, a, d, +h,
                initial_density=D_central,
                cutoff_ha=cutoff_ha,
                functional=functional,
                v_ne_convention=v_ne_convention,
                smearing_alpha=smearing_alpha,
                fd_step_gradient_bohr=fd_step_gradient_bohr,
            )
            grad_minus = _gradient_at_displacement(
                system, basis_name, a, d, -h,
                initial_density=D_central,
                cutoff_ha=cutoff_ha,
                functional=functional,
                v_ne_convention=v_ne_convention,
                smearing_alpha=smearing_alpha,
                fd_step_gradient_bohr=fd_step_gradient_bohr,
            )
            # d^2E / dq dr = d(dE/dr) / dq via central difference.
            # grad has shape (n_atoms, 3); flatten to (3 n_atoms,).
            row = (grad_plus - grad_minus).reshape(-1) / (2.0 * h)
            H[3 * a + d, :] = row

    # Symmetrise to absorb FD asymmetry: the exact mixed second
    # derivatives commute, so any asymmetry is finite-step noise.
    H = 0.5 * (H + H.T)
    return H


def compute_vibrational_frequencies(
    hessian: np.ndarray,
    masses_amu: Sequence[float],
) -> tuple[np.ndarray, np.ndarray]:
    """Diagonalise the mass-weighted Hessian and return wavenumbers.

    Parameters
    ----------
    hessian
        ``(3N, 3N)`` atomic Hessian in Hartree/bohr^2 (the return
        value of :func:`compute_hessian_gpw`).
    masses_amu
        Per-atom masses in amu, length ``N``. Repeated three times
        per atom to mass-weight the Cartesian DOFs.

    Returns
    -------
    eigenvalues_cm1 : np.ndarray, shape ``(3N,)``
        Harmonic wavenumbers in cm⁻¹, sorted ascending. Imaginary
        modes (negative w^2) are reported as **negative** wavenumbers
        per the convention used by Gaussian / NWChem / ORCA / PySCF
        / :mod:`vibeqc.hessian`.
    eigenvectors : np.ndarray, shape ``(3N, 3N)``
        Mass-weighted normal-mode eigenvectors as columns (output of
        :func:`numpy.linalg.eigh`). Column ``k`` corresponds to
        ``eigenvalues_cm1[k]``.

    Notes
    -----
    The mass-weighted Hessian is

        H_mw[i, j] = H[i, j] / sqrt(m_i . m_j)

    with ``m_i`` the mass of the atom owning DOF ``i`` in *electron-
    mass* units (so the eigenvalues come out in a.u.^2 of w^2, which
    converts cleanly to cm⁻¹ via the Hartree -> cm⁻¹ factor -- same
    derivation as :mod:`vibeqc.hessian`).
    """
    H = np.asarray(hessian, dtype=np.float64)
    if H.ndim != 2 or H.shape[0] != H.shape[1]:
        raise ValueError(
            f"compute_vibrational_frequencies: expected square 2D Hessian; "
            f"got shape {H.shape}."
        )
    n_dof = H.shape[0]
    n_atoms = n_dof // 3
    if 3 * n_atoms != n_dof:
        raise ValueError(
            f"compute_vibrational_frequencies: Hessian DOF count "
            f"{n_dof} is not a multiple of 3."
        )
    if len(masses_amu) != n_atoms:
        raise ValueError(
            f"compute_vibrational_frequencies: got {len(masses_amu)} "
            f"masses for {n_atoms} atoms."
        )

    # Build per-DOF mass vector in electron-mass units.
    masses_me = np.empty(n_dof, dtype=np.float64)
    for i, m in enumerate(masses_amu):
        masses_me[3 * i:3 * i + 3] = float(m) * _AMU_TO_ELECTRON_MASS

    # Mass-weight: H_mw = M^{-1/2} H M^{-1/2}
    inv_sqrt_m = 1.0 / np.sqrt(masses_me)
    H_mw = H * np.outer(inv_sqrt_m, inv_sqrt_m)
    # Re-symmetrise after the outer-product weighting (harmless if
    # H is already symmetric -- guards against float round-off).
    H_mw = 0.5 * (H_mw + H_mw.T)

    omega2, eigvecs = np.linalg.eigh(H_mw)
    # w^2 (a.u.^2) -> w (cm⁻¹), preserving sign so imaginary modes come
    # back as negative wavenumbers.
    out = np.empty_like(omega2)
    pos = omega2 >= 0
    out[pos] = np.sqrt(omega2[pos]) * _HARTREE_TO_CM_INV
    out[~pos] = -np.sqrt(-omega2[~pos]) * _HARTREE_TO_CM_INV
    return out, eigvecs
