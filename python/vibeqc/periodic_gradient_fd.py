"""Phase G1a-fd -- finite-difference periodic atomic gradient.

A simple validation reference for the analytic periodic gradients
landing in G1a. Computes dE/dR_{A,d} via central-difference of
``run_rhf_periodic`` (or the multi-k Ewald variant) at displaced
geometries.

Useful for:

  * **Validation** -- analytic gradient code in G1a / G1b / G1c
    should reproduce this FD reference to <1e-6 Ha/bohr on small
    test cells.
  * **Sanity** -- converged geometries should have FD-gradient
    components ≈ 0; non-equilibrium starting geometries should
    have a gradient pointing away from the strain.
  * **Slow-but-correct fallback** -- for systems where the analytic
    path hasn't been wired up yet (e.g. multi-k UKS Ewald), users
    can still get forces, just at 6N SCF cost per Hessian column.

Cost: 6N full periodic SCF runs per gradient (one displaced + and
one displaced - per atom-coord). For a small unit cell (~10 atoms,
STO-3G, Γ-only) each SCF takes a few seconds, so a full FD gradient
on a 10-atom cell completes in ~5 minutes. The analytic path will
do the same in <1 second per gradient.
"""

from __future__ import annotations

from copy import deepcopy
from typing import Optional, Sequence

import numpy as np

from ._vibeqc_core import (
    Atom,
    BasisSet,
    BlochKMesh,
    PeriodicSCFOptions,
    PeriodicSystem,
    monkhorst_pack,
    run_rhf_periodic,
)


__all__ = ["compute_gradient_periodic_rhf_fd"]


def _displaced_periodic_system(
    sys: PeriodicSystem, atom_idx: int, cart: int, delta: float
) -> PeriodicSystem:
    """Return a copy of ``sys`` with atom ``atom_idx`` displaced by
    ``delta`` along Cartesian axis ``cart``. The lattice vectors are
    held fixed -- only atomic positions move (this is what we want for
    the atomic gradient; cell-parameter gradients live in G2)."""
    new_atoms = []
    for i, atom in enumerate(sys.unit_cell):
        xyz = list(atom.xyz)
        if i == atom_idx:
            xyz[cart] += delta
        new_atoms.append(Atom(int(atom.Z), xyz))
    return PeriodicSystem(
        sys.dim,
        np.asarray(sys.lattice, dtype=np.float64),
        new_atoms,
        charge=sys.charge,
        multiplicity=sys.multiplicity,
    )


def compute_gradient_periodic_rhf_fd(
    system: PeriodicSystem,
    basis_name: str,
    kmesh: BlochKMesh,
    options: Optional[PeriodicSCFOptions] = None,
    *,
    step_bohr: float = 1e-3,
    rebuild_basis: bool = True,
) -> np.ndarray:
    """Central-difference periodic atomic gradient via repeated SCF.

    Parameters
    ----------
    system, basis_name, kmesh, options
        Same arguments as :func:`vibeqc.run_rhf_periodic`. ``basis_name``
        is the basis-set name (string), not a pre-built ``BasisSet``,
        because each displaced geometry needs its own ``BasisSet``
        rebuilt against the displaced atomic positions (same shells +
        contraction coefficients but a different shell-origin set).
    step_bohr
        Half-step for the central difference. Default 1e-3 bohr --
        large enough to be well above SCF convergence noise (1e-8 Ha
        round-trips to ~1e-5 Ha/bohr at this step), small enough to
        stay in the linear regime. Tighter steps (1e-5) trip on SCF
        convergence noise; wider steps (1e-1) trip on cubic
        truncation error.
    rebuild_basis
        If ``True`` (default), re-build a fresh ``BasisSet`` per
        displaced geometry. Disable only for systems where you've
        cached a basis with shell positions tied to the *reference*
        (e.g. one-shot dipole / overlap-derivative debugging).

    Returns
    -------
    np.ndarray
        ``(n_atoms, 3)`` gradient in Ha/bohr. Indexing matches the
        atom order in ``system.unit_cell``.

    Examples
    --------
    >>> import vibeqc as vq
    >>> sys = vq.PeriodicSystem(3, lattice, atoms)
    >>> kmesh = vq.monkhorst_pack(sys, [1, 1, 1])
    >>> g = vq.compute_gradient_periodic_rhf_fd(sys, "sto-3g", kmesh)
    >>> g.shape
    (n_atoms, 3)
    """
    if options is None:
        options = PeriodicSCFOptions()

    n_atoms = len(system.unit_cell)
    grad = np.zeros((n_atoms, 3), dtype=np.float64)

    for ia in range(n_atoms):
        for d in range(3):
            sys_p = _displaced_periodic_system(system, ia, d, +step_bohr)
            sys_m = _displaced_periodic_system(system, ia, d, -step_bohr)
            basis_p = BasisSet(sys_p.unit_cell_molecule(), basis_name)
            basis_m = BasisSet(sys_m.unit_cell_molecule(), basis_name)
            r_p = run_rhf_periodic(sys_p, basis_p, kmesh, options)
            r_m = run_rhf_periodic(sys_m, basis_m, kmesh, options)
            if not r_p.converged or not r_m.converged:
                raise RuntimeError(
                    f"compute_gradient_periodic_rhf_fd: SCF failed at "
                    f"displaced geometry (atom {ia}, axis {d}). "
                    f"Converged ±: {r_p.converged} / {r_m.converged}. "
                    f"Try a larger step_bohr or relax SCF tolerances."
                )
            grad[ia, d] = (r_p.energy - r_m.energy) / (2.0 * step_bohr)

    return grad
