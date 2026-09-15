"""Phase SYM3b: compute-reduced one-electron periodic integral evaluation.

Builds on the SYM2c atom-pair orbit machinery to evaluate one-electron
integrals (overlap, kinetic, nuclear) only at orbit-representative
lattice cells, then scatters the results to all orbit members via
Wigner-D rotations.

This is the **compute** reduction (SYM3b) -- the wall-clock sibling of the
**storage** reduction (SYM3a) in :mod:`vibeqc.symmetry_integrals`.

Public API
----------
  - :func:`compression_summary` -- diagnostics dict from an orbit partition.
  - :func:`compute_overlap_lattice_reduced` -- overlap with reduced cells.
  - :func:`compute_kinetic_lattice_reduced` -- kinetic with reduced cells.
  - :func:`compute_nuclear_lattice_reduced` -- nuclear with reduced cells.
"""

from __future__ import annotations

from typing import List, Sequence, Tuple

import numpy as np

from ._vibeqc_core import (
    BasisSet,
    LatticeCell,
    LatticeSumOptions,
    PeriodicSystem,
    SymmetryOp,
    compute_kinetic_lattice_explicit,
    compute_nuclear_lattice_explicit,
    compute_overlap_lattice_explicit,
    direct_lattice_cells,
    pair_complete_lattice_cells,
)
from .symmetry_integrals import (
    OrbitReducedLatticeMatrix,
    symmorphic_operations,
)
from .symmetry_lattice_c import (
    AtomPairOrbits,
    _cell_tuple,
    compress_lattice_matrix_set_c,
    identify_atom_pair_orbits,
    reconstruct_lattice_matrix_set_c,
)

__all__ = [
    "compression_summary",
    "compute_overlap_lattice_reduced",
    "compute_kinetic_lattice_reduced",
    "compute_nuclear_lattice_reduced",
]


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _unique_rep_cells(
    orbits: AtomPairOrbits,
    full_cells: Sequence[LatticeCell],
) -> List[LatticeCell]:
    """Collect the unique cell objects for orbit representatives.

    Different atom-pair orbits may share the same cell index.  We
    return the unique ``LatticeCell`` objects in first-appearance order.
    """
    cell_index_to_cell = {_cell_tuple(c.index): c for c in full_cells}
    seen: set = set()
    unique: List[LatticeCell] = []
    for orb in orbits.orbits:
        _, _, h = orb.representative
        if h not in seen:
            seen.add(h)
            unique.append(cell_index_to_cell[h])
    return unique


def compression_summary(orbits: AtomPairOrbits) -> dict:
    """One-line diagnostics dict for an orbit partition."""
    n_rep_cells = len({orb.representative[2] for orb in orbits.orbits})
    return {
        "n_orbits": orbits.n_orbits,
        "n_triples": orbits.n_triples,
        "n_cells_full": orbits.n_cells,
        "n_cells_reduced": n_rep_cells,
        "compression_cells": (orbits.n_cells / n_rep_cells if n_rep_cells > 0 else 1.0),
        "compression_triples": orbits.compression_ratio,
    }


def _compute_reduced_impl(
    basis: BasisSet,
    system: PeriodicSystem,
    options: LatticeSumOptions,
    operations: Sequence[SymmetryOp],
    explicit_fn,
    *,
    require_closed: bool = False,
) -> Tuple[OrbitReducedLatticeMatrix, list]:
    """Shared implementation for the three compute_reduced functions."""
    sym_ops = symmorphic_operations(operations)
    full_cells = one_electron_lattice_cells(basis, system, options)
    orbits = identify_atom_pair_orbits(
        full_cells,
        sym_ops,
        system,
        require_closed=require_closed,
    )
    rep_cells = _unique_rep_cells(orbits, full_cells)

    # Compute integrals only on representative cells.
    reduced_lms = explicit_fn(basis, system, rep_cells)

    reps = compress_lattice_matrix_set_c(reduced_lms, orbits, basis)
    recon_blocks = reconstruct_lattice_matrix_set_c(
        reps,
        orbits,
        basis,
        system,
        sym_ops,
        cells=full_cells,
    )
    return (
        OrbitReducedLatticeMatrix(orbits=orbits, representatives=reps),
        recon_blocks,
    )


# ---------------------------------------------------------------------------
# Public entry points
# ---------------------------------------------------------------------------


def one_electron_lattice_cells(
    basis: BasisSet,
    system: PeriodicSystem,
    options: LatticeSumOptions,
) -> list:
    """The cell list the one-electron lattice sums ride under ``options``.

    The plain ``|g| <= cutoff`` ball by default; under
    ``LatticeSumOptions.pair_complete_1e`` (#429) the pair-complete
    enumeration of ``cpp/include/vibeqc/lattice_pair_cells.hpp``, of
    which the ball is an exact prefix. Every consumer that templates a
    lattice matrix on the one-electron list (the reduced S/T
    reconstruction here, the BIPOLE symmetric-integral assembly) takes
    it from this one place so the two agree cell for cell.
    """
    if bool(getattr(options, "pair_complete_1e", False)):
        return list(
            pair_complete_lattice_cells(basis, system, float(options.cutoff_bohr))
        )
    return list(direct_lattice_cells(system, float(options.cutoff_bohr)))


def _pair_cutoff(options: LatticeSumOptions) -> float:
    """The explicit builders' per-pair cutoff under ``options``.

    ``0.0`` (no pair filter) on the plain ball; the interaction cutoff
    under ``pair_complete_1e``, so the representative-cell integrals
    bound the same physical pair separation the full builder bounds.
    A symmorphic operation preserves that separation, so the Wigner-D
    reconstruction over an orbit stays consistent with the filter.
    """
    if bool(getattr(options, "pair_complete_1e", False)):
        return float(options.cutoff_bohr)
    return 0.0


def compute_overlap_lattice_reduced(
    basis: BasisSet,
    system: PeriodicSystem,
    options: LatticeSumOptions,
    operations: Sequence[SymmetryOp],
    *,
    require_closed: bool = False,
) -> Tuple[OrbitReducedLatticeMatrix, list]:
    """Compute S(g) with symmetry-reduced integral evaluation.

    Returns ``(reduced, recon_blocks)`` where ``recon_blocks`` is a
    list of ``(nbf, nbf)`` blocks in full-cell-list order, equivalent
    to ``compute_overlap_lattice(basis, system, options).blocks`` to
    machine precision.
    """
    return _compute_reduced_impl(
        basis,
        system,
        options,
        operations,
        lambda b, s, c: compute_overlap_lattice_explicit(
            b, s, c, _pair_cutoff(options)
        ),
        require_closed=require_closed,
    )


def compute_kinetic_lattice_reduced(
    basis: BasisSet,
    system: PeriodicSystem,
    options: LatticeSumOptions,
    operations: Sequence[SymmetryOp],
    *,
    require_closed: bool = False,
) -> Tuple[OrbitReducedLatticeMatrix, list]:
    """Compute T(g) with symmetry-reduced integral evaluation."""
    return _compute_reduced_impl(
        basis,
        system,
        options,
        operations,
        lambda b, s, c: compute_kinetic_lattice_explicit(
            b, s, c, _pair_cutoff(options)
        ),
        require_closed=require_closed,
    )


def compute_nuclear_lattice_reduced(
    basis: BasisSet,
    system: PeriodicSystem,
    options: LatticeSumOptions,
    operations: Sequence[SymmetryOp],
    *,
    require_closed: bool = False,
) -> Tuple[OrbitReducedLatticeMatrix, list]:
    """Compute V(g) with symmetry-reduced integral evaluation.

    .. warning::

       The nuclear-attraction integral involves the lattice-summed
       nuclear potential, which is NOT translation-invariant.  The
       Wigner-D reconstruction formula that works for overlap and
       kinetic integrals does not correctly capture the position-
       dependent features of V_nuc.  This function currently falls
       back to the full (non-reduced) computation via
       :func:`vibeqc.compute_nuclear_lattice` and compresses the
       result for storage only (SYM3a path).
    """
    from ._vibeqc_core import compute_nuclear_lattice as _full

    full = _full(basis, system, options)
    from .symmetry_integrals import _compress_with_orbits

    orbits, reps = _compress_with_orbits(
        full, basis, system, operations, require_closed=require_closed
    )
    return OrbitReducedLatticeMatrix(orbits=orbits, representatives=reps), list(
        full.blocks
    )
