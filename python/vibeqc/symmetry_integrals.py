"""Phase SYM3: symmetry-reduced lattice-integral helpers.

Builds on the SYM2c atom-pair / cell-index orbit machinery
(:mod:`vibeqc.symmetry_lattice_c`) to provide compressed
representations of one-electron periodic integrals: overlap S(g),
kinetic T(g), nuclear V(g). For each operator-closed cell list the
orbit partition lets us store one ``(n_AO_a, n_AO_b)`` representative
sub-block per orbit instead of the full ``(n_bf x n_bf)`` block at
every cell -- a ``|G|``-fold memory saving on high-symmetry crystals
(e.g. simple cubic He at Pm-3m gives orbit compression ~7x; NaCl in
Fm-3m gives the same up to atom-pair structure).

This module ships **storage** reduction. The integrals are still
computed in full by the existing C++ kernels and then compressed --
the |G|-fold *compute* reduction (skip non-representative shell-pair
x cell triples in the kernel itself) is SYM3b, deferred to a
follow-up commit. The compress / reconstruct round-trip provided
here is the validation substrate that SYM3b's compute-reduced kernel
must agree with to machine precision.

The other deliverable here is :func:`verify_lattice_matrix_set_symmetry`
-- a debug witness that any user-supplied :class:`LatticeMatrixSet`
satisfies the SYM2c symmetry relations to within a tolerance. Useful
as a contract check on densities / Fock matrices reconstructed from
multi-k SCF, where numerical drift could otherwise accumulate
silently across iterations.

Convention note: SYM2c (and therefore everything here) works on
**symmorphic** symmetry operations only -- ``translation = 0``. Many
common space groups (Im-3m for the BCC cubic Wigner-Seitz lattice,
Fd-3m for diamond, etc.) carry non-symmorphic glide-translations
and require additional bookkeeping deferred to a later phase.
:func:`symmorphic_operations` filters a full operator list down to
the symmorphic subset and is exposed as a helper for callers that
want to use whatever symmorphic operators happen to be present.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import List, Sequence, Tuple

import numpy as np

from ._vibeqc_core import (
    BasisSet,
    LatticeMatrixSet,
    LatticeSumOptions,
    PeriodicSystem,
    SymmetryOp,
    compute_kinetic_lattice,
    compute_nuclear_lattice,
    compute_overlap_lattice,
)
from .symmetry_lattice_c import (
    AtomPairOrbits,
    compress_lattice_matrix_set_c,
    identify_atom_pair_orbits,
    reconstruct_lattice_matrix_set_c,
)


__all__ = [
    "OrbitReducedLatticeMatrix",
    "SymmetrisedOrbitIntegralResult",
    "compute_overlap_lattice_with_orbits",
    "compute_kinetic_lattice_with_orbits",
    "compute_nuclear_lattice_with_orbits",
    "compute_overlap_lattice_symmetrised_with_orbits",
    "compute_kinetic_lattice_symmetrised_with_orbits",
    "compute_nuclear_lattice_symmetrised_with_orbits",
    "symmorphic_operations",
    "verify_lattice_matrix_set_symmetry",
]


@dataclass
class OrbitReducedLatticeMatrix:
    """Symmetry-compressed view of a periodic lattice matrix.

    Storage shape:
      - ``orbits`` -- :class:`AtomPairOrbits` partition of (atom_a,
        atom_b, cell_index) triples under the symmorphic operator set.
      - ``representatives`` -- list of ``(n_AO_a, n_AO_b)`` sub-blocks,
        one per orbit, in orbit order.

    Reconstruction back to a full ``LatticeMatrixSet`` lives in
    :func:`vibeqc.reconstruct_lattice_matrix_set_c`. The
    ``compression_ratio`` property is the average orbit size -- the
    factor by which storage shrinks vs the dense LatticeMatrixSet.
    """

    orbits: AtomPairOrbits
    representatives: List[np.ndarray]

    @property
    def compression_ratio(self) -> float:
        return self.orbits.compression_ratio

    @property
    def n_orbits(self) -> int:
        return self.orbits.n_orbits


@dataclass
class SymmetrisedOrbitIntegralResult:
    """Result bundle for GS3 standardise-then-compress helpers.

    ``system`` is the standardised/idealised periodic system used for
    the integral build, ``basis`` is the rebuilt basis on that system,
    ``report`` is the :func:`vibeqc.symmetrise` metadata, ``operations``
    are the symmorphic operations consumed by the orbit partition, and
    ``reduced`` / ``full`` are the compressed and original lattice
    matrix representations.
    """

    system: PeriodicSystem
    basis: BasisSet
    report: "SymmetriseReport"
    operations: List[SymmetryOp]
    reduced: OrbitReducedLatticeMatrix
    full: LatticeMatrixSet

    def __iter__(self):
        """Allow ``reduced, full = compute_*_symmetrised_with_orbits(...)``."""
        yield self.reduced
        yield self.full


def symmorphic_operations(
    operations: Sequence[SymmetryOp],
    *,
    tol: float = 1e-8,
) -> List[SymmetryOp]:
    """Return the subset of ``operations`` whose fractional translation
    is zero (symmorphic).

    Useful preprocessing step for SYM2c / SYM3 -- non-symmorphic
    operations carry glide / screw translations that the current orbit
    machinery doesn't yet handle. Identity-with-zero-translation is
    always included (it's symmorphic).
    """
    out: List[SymmetryOp] = []
    for op in operations:
        t = np.asarray(op.translation, dtype=float)
        if np.all(np.abs(t) < tol):
            out.append(op)
    return out


def _compress_with_orbits(
    full_lms: LatticeMatrixSet,
    basis: BasisSet,
    system: PeriodicSystem,
    operations: Sequence[SymmetryOp],
    *,
    require_closed: bool = True,
) -> Tuple[AtomPairOrbits, List[np.ndarray]]:
    """Internal helper: build orbits, compress an LMS, return both.

    Filters ``operations`` to the symmorphic subset before calling
    :func:`identify_atom_pair_orbits` so callers don't need to know
    about symmorphic-vs-non-symmorphic plumbing.

    ``require_closed=False`` is useful when the cell list comes from a
    spherical real-space cutoff that may clip some orbit partners
    near the boundary. The compress / reconstruct round-trip remains
    exact for the triples that *are* in the cell list; non-closed
    triples are simply not stored.
    """
    sym_ops = symmorphic_operations(operations)
    orbits = identify_atom_pair_orbits(
        full_lms.cells, sym_ops, system, require_closed=require_closed,
    )
    reps = compress_lattice_matrix_set_c(full_lms, orbits, basis)
    return orbits, reps


def compute_overlap_lattice_with_orbits(
    basis: BasisSet,
    system: PeriodicSystem,
    options: LatticeSumOptions,
    operations: Sequence[SymmetryOp],
    *,
    require_closed: bool = True,
) -> Tuple[OrbitReducedLatticeMatrix, LatticeMatrixSet]:
    """Periodic overlap S(g) plus its SYM2c orbit-compressed view.

    Returns ``(reduced, full_lms)`` where ``reduced`` carries the
    orbit partition + per-orbit representative sub-blocks, and
    ``full_lms`` is the original :class:`LatticeMatrixSet` from the
    C++ ``compute_overlap_lattice``. The two are equivalent
    representations to machine precision.

    For a high-symmetry simple-cubic cell with the full Pm-3m point
    group (|G| = 48) and a single-atom basis at the origin, the
    compression ratio approaches |G| as the cell list grows.
    """
    full = compute_overlap_lattice(basis, system, options)
    orbits, reps = _compress_with_orbits(full, basis, system, operations, require_closed=require_closed)
    return OrbitReducedLatticeMatrix(orbits=orbits, representatives=reps), full


def compute_kinetic_lattice_with_orbits(
    basis: BasisSet,
    system: PeriodicSystem,
    options: LatticeSumOptions,
    operations: Sequence[SymmetryOp],
    *,
    require_closed: bool = True,
) -> Tuple[OrbitReducedLatticeMatrix, LatticeMatrixSet]:
    """Periodic kinetic T(g) plus its SYM2c orbit-compressed view."""
    full = compute_kinetic_lattice(basis, system, options)
    orbits, reps = _compress_with_orbits(full, basis, system, operations, require_closed=require_closed)
    return OrbitReducedLatticeMatrix(orbits=orbits, representatives=reps), full


def compute_nuclear_lattice_with_orbits(
    basis: BasisSet,
    system: PeriodicSystem,
    options: LatticeSumOptions,
    operations: Sequence[SymmetryOp],
    *,
    require_closed: bool = True,
) -> Tuple[OrbitReducedLatticeMatrix, LatticeMatrixSet]:
    """Periodic nuclear V(g) plus its SYM2c orbit-compressed view.

    Note: ``compute_nuclear_lattice`` dispatches on
    ``options.coulomb_method``. The orbit partition itself is
    operator-only and method-independent, so this works identically
    for ``DIRECT_TRUNCATED`` and ``EWALD_3D`` nuclear builds.
    """
    full = compute_nuclear_lattice(basis, system, options)
    orbits, reps = _compress_with_orbits(full, basis, system, operations, require_closed=require_closed)
    return OrbitReducedLatticeMatrix(orbits=orbits, representatives=reps), full


def compute_overlap_lattice_symmetrised_with_orbits(
    system: PeriodicSystem,
    basis_name: str,
    options: LatticeSumOptions,
    *,
    symprec: float = 1.0e-4,
    to_primitive: bool = False,
    idealize: bool = True,
    require_closed: bool = True,
) -> SymmetrisedOrbitIntegralResult:
    """Standardise ``system`` and build orbit-compressed overlap S(g).

    This is the GS3 convenience path: imported/near-symmetric structures
    are first cleaned with :func:`vibeqc.symmetrise`, then the basis is
    rebuilt on the standardised cell, and the SYM3 storage-compressed
    lattice matrix is built from the attached symmorphic operations.
    """
    clean, basis, report, ops = _standardised_orbit_inputs(
        system,
        basis_name,
        symprec=symprec,
        to_primitive=to_primitive,
        idealize=idealize,
    )
    reduced, full = compute_overlap_lattice_with_orbits(
        basis, clean, options, ops, require_closed=require_closed,
    )
    return SymmetrisedOrbitIntegralResult(
        system=clean,
        basis=basis,
        report=report,
        operations=ops,
        reduced=reduced,
        full=full,
    )


def compute_kinetic_lattice_symmetrised_with_orbits(
    system: PeriodicSystem,
    basis_name: str,
    options: LatticeSumOptions,
    *,
    symprec: float = 1.0e-4,
    to_primitive: bool = False,
    idealize: bool = True,
    require_closed: bool = True,
) -> SymmetrisedOrbitIntegralResult:
    """Standardise ``system`` and build orbit-compressed kinetic T(g)."""
    clean, basis, report, ops = _standardised_orbit_inputs(
        system,
        basis_name,
        symprec=symprec,
        to_primitive=to_primitive,
        idealize=idealize,
    )
    reduced, full = compute_kinetic_lattice_with_orbits(
        basis, clean, options, ops, require_closed=require_closed,
    )
    return SymmetrisedOrbitIntegralResult(
        system=clean,
        basis=basis,
        report=report,
        operations=ops,
        reduced=reduced,
        full=full,
    )


def compute_nuclear_lattice_symmetrised_with_orbits(
    system: PeriodicSystem,
    basis_name: str,
    options: LatticeSumOptions,
    *,
    symprec: float = 1.0e-4,
    to_primitive: bool = False,
    idealize: bool = True,
    require_closed: bool = True,
) -> SymmetrisedOrbitIntegralResult:
    """Standardise ``system`` and build orbit-compressed nuclear V(g)."""
    clean, basis, report, ops = _standardised_orbit_inputs(
        system,
        basis_name,
        symprec=symprec,
        to_primitive=to_primitive,
        idealize=idealize,
    )
    reduced, full = compute_nuclear_lattice_with_orbits(
        basis, clean, options, ops, require_closed=require_closed,
    )
    return SymmetrisedOrbitIntegralResult(
        system=clean,
        basis=basis,
        report=report,
        operations=ops,
        reduced=reduced,
        full=full,
    )


def _standardised_orbit_inputs(
    system: PeriodicSystem,
    basis_name: str,
    *,
    symprec: float,
    to_primitive: bool,
    idealize: bool,
) -> tuple[PeriodicSystem, BasisSet, "SymmetriseReport", List[SymmetryOp]]:
    from .periodic_symmetrize import SymmetriseReport, symmetrise

    result = symmetrise(
        system,
        symprec=symprec,
        to_primitive=to_primitive,
        idealize=idealize,
    )
    clean = result.system
    basis = BasisSet(clean.unit_cell_molecule(), str(basis_name))
    if clean.symmetry is None:
        raise RuntimeError(
            "compute_*_lattice_symmetrised_with_orbits: symmetrise did "
            "not attach symmetry to the standardised system"
        )
    ops = symmorphic_operations(clean.symmetry.operations)
    if not ops:
        raise ValueError(
            "compute_*_lattice_symmetrised_with_orbits: the standardised "
            "system has no symmorphic symmetry operations available for "
            "orbit compression"
        )
    report: SymmetriseReport = result.report
    return clean, basis, report, ops


def verify_lattice_matrix_set_symmetry(
    lms: LatticeMatrixSet,
    basis: BasisSet,
    system: PeriodicSystem,
    operations: Sequence[SymmetryOp],
    *,
    atol: float = 1e-10,
    require_closed: bool = True,
) -> dict:
    """Verify ``lms`` respects the SYM2c symmetry relations.

    Compresses to per-orbit representatives and reconstructs; the
    reconstructed full LMS must equal the input to within ``atol``.
    Returns a diagnostics dict::

        {
            "n_orbits":          int,
            "compression_ratio": float,
            "max_residual":      float,
            "passes":            bool,
        }

    A failing ``passes`` flag indicates the LMS does **not** transform
    correctly under the (symmorphic) symmetry operations -- useful for
    catching numerical drift in densities reconstructed from multi-k
    SCF, or for catching bugs in routines that should produce
    symmetry-respecting outputs.

    Raises if the operator list contains no symmorphic operators (no
    symmetry to check against).
    """
    sym_ops = symmorphic_operations(operations)
    if not sym_ops:
        raise ValueError(
            "verify_lattice_matrix_set_symmetry: operator list contains "
            "no symmorphic operations; nothing to check."
        )
    orbits = identify_atom_pair_orbits(
        lms.cells, sym_ops, system, require_closed=require_closed,
    )
    reps = compress_lattice_matrix_set_c(lms, orbits, basis)
    recon = reconstruct_lattice_matrix_set_c(
        reps, orbits, basis, system, sym_ops, cells=list(lms.cells),
    )
    max_residual = 0.0
    for slot, recon_blk in enumerate(recon):
        full_blk = np.asarray(lms.blocks[slot], dtype=float)
        diff = float(np.linalg.norm(full_blk - recon_blk))
        if diff > max_residual:
            max_residual = diff
    return {
        "n_orbits": orbits.n_orbits,
        "compression_ratio": orbits.compression_ratio,
        "max_residual": max_residual,
        "passes": max_residual < atol,
    }
