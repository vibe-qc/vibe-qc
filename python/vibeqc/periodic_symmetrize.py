"""Periodic-structure space-group detection and symmetrisation helpers."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Iterator

import numpy as np

from ._vibeqc_core import (
    Atom,
    Crystal,
    PeriodicSystem,
    SpaceGroup,
    analyze_symmetry,
    attach_symmetry,
)


@dataclass(frozen=True)
class SymmetriseReport:
    """Metadata describing a periodic-structure symmetrisation."""

    symprec: float
    spacegroup_before: SpaceGroup
    spacegroup_after: SpaceGroup
    n_atoms_before: int
    n_atoms_after: int
    volume_before_bohr3: float
    volume_after_bohr3: float
    rms_displacement_bohr: float
    max_displacement_bohr: float
    to_primitive: bool
    idealized: bool


@dataclass(frozen=True)
class SymmetriseResult:
    """Return value of :func:`symmetrise`."""

    system: PeriodicSystem
    report: SymmetriseReport

    def __iter__(self) -> Iterator[PeriodicSystem | SymmetriseReport]:
        """Allow ``system, report = symmetrise(...)`` unpacking."""
        yield self.system
        yield self.report


def detect_spacegroup(
    system: PeriodicSystem | Crystal,
    symprec: float = 1.0e-4,
) -> SpaceGroup:
    """Return spglib's space-group analysis without mutating ``system``.

    ``attach_symmetry(system)`` remains the calculation-side helper when
    downstream code should see ``system.symmetry``. This function is for
    introspection and reports.
    """
    crystal = _as_crystal(system)
    return analyze_symmetry(crystal, symprec=symprec)


def symmetrise(
    system: PeriodicSystem,
    symprec: float = 1.0e-4,
    *,
    to_primitive: bool = False,
    idealize: bool = True,
) -> SymmetriseResult:
    """Standardise and idealise a 3D periodic structure via spglib.

    Returns a new :class:`PeriodicSystem` and a report containing the
    before/after space groups plus the RMS and maximum atomic displacement
    between spglib's non-idealized and idealized standardized cells.
    The input system is not modified.
    """
    if int(system.dim) != 3:
        raise ValueError(
            "symmetrise currently supports only 3D bulk PeriodicSystem "
            f"objects; got dim={system.dim}"
        )

    before = detect_spacegroup(system, symprec=symprec)
    cell = _system_to_spglib_cell(system)

    raw = _standardize_cell(
        cell,
        to_primitive=to_primitive,
        no_idealize=True,
        symprec=symprec,
    )
    standardized = (
        _standardize_cell(
            cell,
            to_primitive=to_primitive,
            no_idealize=not idealize,
            symprec=symprec,
        )
        if idealize
        else raw
    )

    out = _spglib_cell_to_system(
        standardized,
        charge=system.charge,
        multiplicity=system.multiplicity,
    )
    attach_symmetry(out, symprec=symprec)
    after = out.symmetry
    if after is None:
        raise RuntimeError("symmetrise: attach_symmetry did not populate output")

    rms, max_disp = _matched_displacement_stats(raw, standardized)
    report = SymmetriseReport(
        symprec=float(symprec),
        spacegroup_before=before,
        spacegroup_after=after,
        n_atoms_before=len(system.unit_cell),
        n_atoms_after=len(out.unit_cell),
        volume_before_bohr3=float(abs(np.linalg.det(np.asarray(system.lattice)))),
        volume_after_bohr3=float(abs(np.linalg.det(np.asarray(out.lattice)))),
        rms_displacement_bohr=float(rms),
        max_displacement_bohr=float(max_disp),
        to_primitive=bool(to_primitive),
        idealized=bool(idealize),
    )
    return SymmetriseResult(system=out, report=report)


def _as_crystal(system: PeriodicSystem | Crystal) -> Crystal:
    if isinstance(system, Crystal):
        return system
    if isinstance(system, PeriodicSystem):
        return _system_to_crystal(system)
    raise TypeError(
        "detect_spacegroup expects a PeriodicSystem or Crystal, "
        f"got {type(system).__name__}"
    )


def _system_to_crystal(system: PeriodicSystem) -> Crystal:
    lattice = np.asarray(system.lattice, dtype=float, order="F")
    inv_lattice = np.linalg.inv(lattice)
    frac = np.empty((3, len(system.unit_cell)), dtype=float, order="F")
    species: list[int] = []
    for i, atom in enumerate(system.unit_cell):
        frac[:, i] = inv_lattice @ np.asarray(atom.xyz, dtype=float)
        species.append(int(atom.Z))
    return Crystal(lattice, frac, species)


def _system_to_spglib_cell(
    system: PeriodicSystem,
) -> tuple[np.ndarray, np.ndarray, list[int]]:
    crystal = _system_to_crystal(system)
    return _crystal_to_spglib_cell(crystal)


def _crystal_to_spglib_cell(
    crystal: Crystal,
) -> tuple[np.ndarray, np.ndarray, list[int]]:
    lattice_rows = np.asarray(crystal.lattice, dtype=float).T
    positions = np.asarray(crystal.fractional_coords, dtype=float).T % 1.0
    numbers = [int(z) for z in crystal.species]
    return lattice_rows, positions, numbers


def _standardize_cell(
    cell: tuple[np.ndarray, np.ndarray, list[int]],
    *,
    to_primitive: bool,
    no_idealize: bool,
    symprec: float,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    try:
        import spglib
    except ImportError as exc:  # pragma: no cover
        raise ImportError(
            "symmetrise requires the Python spglib package. "
            "Install vibe-qc with its runtime dependencies."
        ) from exc

    out = spglib.standardize_cell(
        cell,
        to_primitive=bool(to_primitive),
        no_idealize=bool(no_idealize),
        symprec=float(symprec),
    )
    if out is None:
        try:
            detail = spglib.get_error_message()
        except TypeError:  # pragma: no cover - older spglib signature
            detail = ""
        msg = "symmetrise: spglib.standardize_cell failed"
        if detail:
            msg += f" ({detail})"
        raise RuntimeError(msg)
    lattice_rows, positions, numbers = out
    return (
        np.asarray(lattice_rows, dtype=float),
        np.asarray(positions, dtype=float) % 1.0,
        np.asarray(numbers, dtype=int),
    )


def _spglib_cell_to_system(
    cell: tuple[np.ndarray, np.ndarray, np.ndarray],
    *,
    charge: int,
    multiplicity: int,
) -> PeriodicSystem:
    lattice_rows, positions, numbers = cell
    lattice = np.asarray(lattice_rows, dtype=float).T
    atoms = [
        Atom(int(z), (lattice @ np.asarray(frac, dtype=float)).tolist())
        for frac, z in zip(np.asarray(positions, dtype=float), numbers)
    ]
    return PeriodicSystem(
        dim=3,
        lattice=np.asarray(lattice, dtype=float, order="F"),
        unit_cell=atoms,
        charge=int(charge),
        multiplicity=int(multiplicity),
    )


def _matched_displacement_stats(
    raw: tuple[np.ndarray, np.ndarray, np.ndarray],
    ideal: tuple[np.ndarray, np.ndarray, np.ndarray],
) -> tuple[float, float]:
    _raw_lattice, raw_pos, raw_numbers = raw
    ideal_lattice, ideal_pos, ideal_numbers = ideal
    if len(raw_numbers) != len(ideal_numbers):
        return float("nan"), float("nan")

    displacements: list[float] = []
    for z in sorted(set(int(x) for x in ideal_numbers)):
        raw_idx = np.where(raw_numbers == z)[0]
        ideal_idx = np.where(ideal_numbers == z)[0]
        if len(raw_idx) != len(ideal_idx):
            return float("nan"), float("nan")
        if len(raw_idx) == 0:
            continue
        costs = np.empty((len(raw_idx), len(ideal_idx)), dtype=float)
        for i, iraw in enumerate(raw_idx):
            for j, jideal in enumerate(ideal_idx):
                costs[i, j] = _periodic_distance_bohr(
                    raw_pos[iraw],
                    ideal_pos[jideal],
                    ideal_lattice,
                )
        rows, cols = _linear_sum_assignment(costs)
        displacements.extend(float(costs[i, j]) for i, j in zip(rows, cols))

    if not displacements:
        return 0.0, 0.0
    arr = np.asarray(displacements, dtype=float)
    return float(np.sqrt(np.mean(arr * arr))), float(np.max(arr))


def _periodic_distance_bohr(
    frac_a: np.ndarray,
    frac_b: np.ndarray,
    lattice_rows: np.ndarray,
) -> float:
    delta = np.asarray(frac_b, dtype=float) - np.asarray(frac_a, dtype=float)
    delta -= np.rint(delta)
    cart = delta @ np.asarray(lattice_rows, dtype=float)
    return float(np.linalg.norm(cart))


def _linear_sum_assignment(costs: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    try:
        from scipy.optimize import linear_sum_assignment
    except ImportError:  # pragma: no cover - scipy is a runtime dependency
        n = costs.shape[0]
        rows: list[int] = []
        cols: list[int] = []
        remaining = set(range(n))
        for i in range(n):
            j = min(remaining, key=lambda col: costs[i, col])
            rows.append(i)
            cols.append(j)
            remaining.remove(j)
        return np.asarray(rows), np.asarray(cols)
    return linear_sum_assignment(costs)
