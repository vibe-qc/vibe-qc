"""Relaxed coordinate scans -- molecular and periodic.

Drive one internal coordinate (bond / angle / dihedral) across a
sequence of target values, relaxing all other degrees of freedom at
each step. This is the canonical "scan the reaction coordinate"
workflow used to locate TS guesses and map reaction energy profiles.

Implementation: the "rigid step" approach used by most QC codes.
For each target value, the geometry is mutated so the constrained
internal coordinate hits the target exactly, then the atoms taking
part in the constraint are added to ``freeze_indices`` and the
existing optimizer (:func:`vibeqc.optimize_molecule` for molecules,
:func:`vibeqc.relax_atoms` for periodic systems) relaxes everything
else around them. Constrained-gradient projection -- slightly more
elegant but rarely necessary in practice -- is deferred to a follow-up.

Usage::

    import vibeqc as vq

    # Molecular: H2O O-H stretch
    result = vq.relaxed_scan(
        h2o, "sto-3g",
        coordinate=("bond", 0, 1),
        values=np.linspace(1.6, 2.4, 5),  # bohr
        method="RHF",
        output="oh_stretch",
    )
    result.write_qvf("oh_stretch.qvf")

    # Periodic: H2 in a 15 bohr box
    result = vq.relaxed_scan(
        h2_in_box, "sto-3g",
        coordinate=("bond", 0, 1),
        values=np.linspace(1.2, 2.0, 5),
        method="RHF",
        kpoints=(1, 1, 1),
    )
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import TYPE_CHECKING, Any, Optional, Sequence, Tuple, Union

import numpy as np

if TYPE_CHECKING:
    from .kpoints import KPoints

from ._vibeqc_core import (
    Atom,
    BlochKMesh,
    Molecule,
    PeriodicSystem,
    monkhorst_pack,
)
from .progress import ProgressLogger, resolve_progress
from .spin_channels import spin_densities

__all__ = [
    "ScanResult",
    "ScanResult2D",
    "relaxed_scan",
    "relaxed_scan_2d",
]

CoordinateSpec = Union[
    Tuple[str, int, int],
    Tuple[str, int, int, int],
    Tuple[str, int, int, int, int],
]


def _compute_frame_volumes(
    geometries: Sequence[Any],
    frame_indices: Sequence[int],
    *,
    method: str,
    basis: str,
    functional: Optional[str],
    spacing: float = 0.25,
    padding: float = 4.0,
) -> Tuple[np.ndarray, dict[str, Any], list[int]]:
    """Evaluate the electron density on a shared grid for the selected
    (decimated) scan frames -- the per-frame volumes for QVF animation
    (W1). Molecular only.

    Returns ``(vol_array, grid_descriptor, frame_index)`` where
    ``vol_array`` is float32 ``[n_emitted, nx, ny, nz]``, the grid
    descriptor is an ``{origin, voxel_vectors, shape}`` dict in bohr
    (shared across frames so the isosurface animates in a fixed box),
    and ``frame_index`` echoes the emitted path-frame indices.

    Periodic scans are not supported here -- collocating a periodic
    density on a real-space grid per frame is a heavier, dim==3-only
    path; raise a clear error so the caller can fall back to a
    geometry-only animation.
    """
    from .cube import CubeGrid, _density_on_grid
    from .runner import _run_single_point
    from ._vibeqc_core import BasisSet

    frames = [geometries[i] for i in frame_indices]
    if any(isinstance(f, PeriodicSystem) for f in frames):
        raise NotImplementedError(
            "emit_volumes_every is only supported for molecular scans; "
            "periodic per-frame densities (dim==3 GPW collocation) are a "
            "future enhancement. Re-run write_qvf without emit_volumes_every "
            "for a geometry-only periodic animation."
        )

    # Shared grid: union bounding box over all emitted geometries so the
    # density never clips as atoms move along the scan.
    all_coords = np.concatenate(
        [np.array([list(a.xyz) for a in f.atoms], dtype=float) for f in frames],
        axis=0,
    )
    lo = all_coords.min(axis=0) - padding
    hi = all_coords.max(axis=0) + padding
    n = np.ceil((hi - lo) / spacing).astype(int) + 1
    grid = CubeGrid(
        origin=lo,
        spacing=np.array([spacing, spacing, spacing]),
        shape=(int(n[0]), int(n[1]), int(n[2])),
    )

    slabs = []
    for f in frames:
        basis_obj = BasisSet(f, basis)
        result = _run_single_point(
            method.lower(), f, basis_obj, functional=functional
        )
        alpha, beta = spin_densities(result)
        if alpha is not None:
            D = np.asarray(alpha, dtype=float) + np.asarray(beta, dtype=float)
        else:
            D = np.asarray(result.density, dtype=float)
        slabs.append(_density_on_grid(D, basis_obj, grid).astype(np.float32))

    vol_array = np.stack(slabs, axis=0)
    grid_descriptor = {
        "origin": [float(x) for x in grid.origin],
        "voxel_vectors": [
            [float(grid.spacing[0]), 0.0, 0.0],
            [0.0, float(grid.spacing[1]), 0.0],
            [0.0, 0.0, float(grid.spacing[2])],
        ],
        "shape": [int(s) for s in grid.shape],
    }
    return vol_array, grid_descriptor, [int(i) for i in frame_indices]


def _coordinate_label_unit(coordinate: CoordinateSpec) -> Tuple[str, str]:
    """Human-readable name + unit for a scan coordinate.

    Drives the vibe-view energy-plot x-axis label (W0/V0). Bonds are
    in bohr; angles + dihedrals in radians -- matching the ``values``
    convention of :func:`relaxed_scan`.
    """
    kind = coordinate[0]
    atoms = "-".join(str(i) for i in coordinate[1:])  # en-dash join
    if kind == "bond":
        return f"bond {atoms}", "bohr"
    if kind == "angle":
        return f"angle {atoms}", "rad"
    if kind == "dihedral":
        return f"dihedral {atoms}", "rad"
    return str(kind), ""


# ---- geometry helpers (bohr / radians) ------------------------------------


def _get_positions(system: Any) -> np.ndarray:
    """Return (n_atoms, 3) Cartesian positions in bohr."""
    atoms = (
        list(system.unit_cell)
        if isinstance(system, PeriodicSystem)
        else list(system.atoms)
    )
    return np.array([list(a.xyz) for a in atoms], dtype=float)


def _with_positions(system: Any, positions: np.ndarray) -> Any:
    """Return a new system (Molecule or PeriodicSystem) with updated positions."""
    if isinstance(system, PeriodicSystem):
        atoms = [
            Atom(int(system.unit_cell[i].Z), [float(x) for x in positions[i]])
            for i in range(len(system.unit_cell))
        ]
        return PeriodicSystem(
            system.dim,
            np.asarray(system.lattice, dtype=float),
            atoms,
            charge=system.charge,
            multiplicity=system.multiplicity,
        )
    atoms = [
        Atom(int(system.atoms[i].Z), [float(x) for x in positions[i]])
        for i in range(len(list(system.atoms)))
    ]
    return Molecule(atoms, system.charge, system.multiplicity)


def _measure_bond(positions: np.ndarray, i: int, j: int) -> float:
    return float(np.linalg.norm(positions[j] - positions[i]))


def _measure_angle(positions: np.ndarray, i: int, j: int, k: int) -> float:
    v1 = positions[i] - positions[j]
    v2 = positions[k] - positions[j]
    c = float(np.dot(v1, v2) / (np.linalg.norm(v1) * np.linalg.norm(v2)))
    return float(np.arccos(max(-1.0, min(1.0, c))))


def _measure_dihedral(
    positions: np.ndarray, i: int, j: int, k: int, l: int
) -> float:
    b1 = positions[j] - positions[i]
    b2 = positions[k] - positions[j]
    b3 = positions[l] - positions[k]
    n1 = np.cross(b1, b2)
    n2 = np.cross(b2, b3)
    m1 = np.cross(n1, b2 / np.linalg.norm(b2))
    x = float(np.dot(n1, n2))
    y = float(np.dot(m1, n2))
    return float(np.arctan2(y, x))


def _rotation_about_axis(axis: np.ndarray, theta: float) -> np.ndarray:
    """Right-hand-rule rotation matrix about ``axis`` by ``theta`` (radians)."""
    a = axis / np.linalg.norm(axis)
    c, s = float(np.cos(theta)), float(np.sin(theta))
    K = np.array(
        [
            [0.0, -a[2], a[1]],
            [a[2], 0.0, -a[0]],
            [-a[1], a[0], 0.0],
        ]
    )
    return np.eye(3) * c + (1.0 - c) * np.outer(a, a) + s * K


def _set_bond(
    positions: np.ndarray, i: int, j: int, target: float
) -> np.ndarray:
    """Move atom ``j`` along the i->j direction so |r_j - r_i| == target.

    Only atom ``j`` moves; atom ``i`` is held in place.
    """
    p = positions.copy()
    v = p[j] - p[i]
    d = float(np.linalg.norm(v))
    if d < 1e-10:
        raise ValueError(
            f"_set_bond: atoms {i} and {j} are coincident (distance "
            f"{d:.3e} bohr); cannot determine bond direction"
        )
    p[j] = p[i] + v * (target / d)
    return p


def _set_angle(
    positions: np.ndarray, i: int, j: int, k: int, target: float
) -> np.ndarray:
    """Rotate atom ``k`` about vertex ``j`` so ∠(i, j, k) == target.

    The rotation axis is perpendicular to the existing i-j-k plane. If
    the three atoms are collinear, an arbitrary perpendicular is
    chosen -- sufficient because the optimizer relaxes the rest.
    """
    p = positions.copy()
    v_ji = p[i] - p[j]
    v_jk = p[k] - p[j]
    current = _measure_angle(p, i, j, k)
    delta = target - current

    axis = np.cross(v_ji, v_jk)
    if np.linalg.norm(axis) < 1e-10:
        # Collinear case -- pick an arbitrary perpendicular.
        if abs(v_ji[0]) < 0.9:
            axis = np.cross(v_ji, np.array([1.0, 0.0, 0.0]))
        else:
            axis = np.cross(v_ji, np.array([0.0, 1.0, 0.0]))

    R = _rotation_about_axis(axis, delta)
    p[k] = p[j] + R @ v_jk
    return p


def _set_dihedral(
    positions: np.ndarray, i: int, j: int, k: int, l: int, target: float
) -> np.ndarray:
    """Rotate atom ``l`` about the j-k axis so the (i,j,k,l) dihedral == target.

    Only atom ``l`` moves; the rest are held. The optimizer relaxes
    any descendant atoms that should move with ``l``.
    """
    p = positions.copy()
    current = _measure_dihedral(p, i, j, k, l)
    # The IUPAC dihedral sign convention rotates opposite to a
    # right-hand-rule rotation about (k - j), so the rotation we apply
    # to drive the dihedral to ``target`` is ``current - target``.
    delta = current - target
    axis = p[k] - p[j]
    R = _rotation_about_axis(axis, delta)
    p[l] = p[k] + R @ (p[l] - p[k])
    return p


def _apply_constraint(
    positions: np.ndarray,
    coordinate: CoordinateSpec,
    target: float,
) -> np.ndarray:
    kind = coordinate[0]
    if kind == "bond":
        return _set_bond(positions, int(coordinate[1]), int(coordinate[2]), target)
    if kind == "angle":
        return _set_angle(
            positions,
            int(coordinate[1]),
            int(coordinate[2]),
            int(coordinate[3]),
            target,
        )
    if kind == "dihedral":
        return _set_dihedral(
            positions,
            int(coordinate[1]),
            int(coordinate[2]),
            int(coordinate[3]),
            int(coordinate[4]),
            target,
        )
    raise ValueError(
        f"relaxed_scan: unknown coordinate kind {kind!r}; "
        "expected 'bond', 'angle', or 'dihedral'"
    )


def _coordinate_atom_indices(coordinate: CoordinateSpec) -> list[int]:
    return [int(x) for x in coordinate[1:]]


def _validate_coordinate(
    coordinate: CoordinateSpec, n_atoms: int
) -> None:
    expected = {"bond": 3, "angle": 4, "dihedral": 5}
    kind = coordinate[0]
    if kind not in expected:
        raise ValueError(
            f"relaxed_scan: unknown coordinate kind {kind!r}; "
            "expected 'bond', 'angle', or 'dihedral'"
        )
    if len(coordinate) != expected[kind]:
        raise ValueError(
            f"relaxed_scan: {kind!r} coordinate needs "
            f"{expected[kind] - 1} atom indices, got {len(coordinate) - 1}"
        )
    indices = _coordinate_atom_indices(coordinate)
    if len(set(indices)) != len(indices):
        raise ValueError(
            f"relaxed_scan: duplicate atom index in {coordinate!r}"
        )
    bad = [i for i in indices if i < 0 or i >= n_atoms]
    if bad:
        raise ValueError(
            f"relaxed_scan: atom indices {bad} out of range "
            f"[0, {n_atoms - 1}]"
        )


# ---- ScanResult ------------------------------------------------------------


@dataclass
class ScanResult:
    """Container for the output of :func:`relaxed_scan`."""

    values: np.ndarray
    """Target coordinate values, shape (n_points,) -- bohr or radians."""

    energies: np.ndarray
    """Relaxed energies (Ha), shape (n_points,)."""

    geometries: list[Any]
    """Relaxed geometries -- one :class:`Molecule` or
    :class:`PeriodicSystem` per scan point."""

    converged_flags: list[bool]
    """Whether the relaxation at each point converged."""

    coordinate: CoordinateSpec
    """The constraint definition used."""

    output_path: Optional[Path] = None
    """Path stem the scan wrote, or ``None`` if no output was requested."""

    extra: dict[str, Any] = field(default_factory=dict)

    def __repr__(self) -> str:
        n_ok = sum(1 for c in self.converged_flags if c)
        return (
            f"ScanResult(n_points={len(self.values)}, "
            f"n_converged={n_ok}, "
            f"E_min={float(np.min(self.energies)):.6f} Ha, "
            f"E_max={float(np.max(self.energies)):.6f} Ha)"
        )

    # ------------------------------------------------------------------
    # QVF emission
    # ------------------------------------------------------------------

    def write_qvf(
        self,
        path: Union[str, Path],
        *,
        transition_state_frames: Optional[Sequence[int]] = None,
        emit_volumes_every: Optional[int] = None,
        volume_spacing: float = 0.25,
        volume_padding: float = 4.0,
    ) -> Path:
        """Emit a vibe-view ``reaction.path`` QVF for animation.

        Each scan point becomes one frame; the reaction coordinate
        carries the constraint values; the energy curve becomes the
        per-frame ``energies`` array. Endpoints are tagged as
        ``reactant`` / ``product``; the highest-energy intermediate
        point is tagged as a ``point`` waypoint (we don't claim it's
        a TS -- that requires a Hessian check).

        The energy-plot x-axis is labelled from the scan coordinate
        (e.g. ``"bond 0-1 (bohr)"``) via the optional
        ``reaction_coordinate_label`` / ``reaction_coordinate_unit``
        metadata.

        Periodic scans (with :class:`PeriodicSystem` geometries) ship
        as **QVF v2** archives with the per-frame lattice + dim on
        the ``reaction.path`` section. The writer detects periodic
        frames automatically; vibe-view's renderer draws the cell
        and wraps atoms across in-plane periodic boundaries.

        Parameters
        ----------
        transition_state_frames
            Optional frame indices the *caller* has independently
            verified to be transition states (e.g. via a Hessian /
            single-imaginary-mode check). Each is tagged with a
            ``transition_state`` waypoint (rendered with the red TS
            star). We never auto-promote the energy max to a TS -- a
            scan maximum is not a verified saddle (CLAUDE.md Sec.7), so
            without this argument the max stays a ``point``.
        emit_volumes_every
            If set to ``N``, attach the electron density (evaluated on a
            shared real-space grid) for every ``N``-th scan frame so the
            viewer can morph the isosurface along the scan. ``N=1`` emits
            every frame; ``N=5`` every fifth (endpoints always included).
            Each emitted frame costs one extra single-point SCF, so cubes
            are opt-in and decimated. Molecular scans only -- periodic
            raises ``NotImplementedError``. Grid box auto-sizes to the
            union of all emitted geometries with ``volume_padding`` bohr
            of headroom and ``volume_spacing`` bohr voxels.
        """
        from .output.formats.qvf import write_reaction_path_qvf

        stem = Path(path)
        if stem.suffix == ".qvf":
            stem = stem.with_suffix("")

        n = len(self.geometries)
        ts_frames = sorted({int(i) for i in (transition_state_frames or [])})
        for i in ts_frames:
            if not 0 <= i < n:
                raise ValueError(
                    f"write_qvf: transition_state_frames index {i} out "
                    f"of range [0, {n})"
                )

        waypoints: list[dict[str, Any]] = []
        if n >= 1:
            waypoints.append(
                {
                    "frame_index": 0,
                    "label": "scan start",
                    "kind": "reactant",
                    "energy_eh": float(self.energies[0]),
                }
            )
        if n >= 2:
            waypoints.append(
                {
                    "frame_index": n - 1,
                    "label": "scan end",
                    "kind": "product",
                    "energy_eh": float(self.energies[-1]),
                }
            )
        if n >= 3:
            i_max = int(np.argmax(self.energies))
            # Skip the generic "scan max" point if the caller already
            # claimed that frame as a verified TS -- avoids a duplicate
            # waypoint at one frame.
            if 0 < i_max < n - 1 and i_max not in ts_frames:
                waypoints.append(
                    {
                        "frame_index": i_max,
                        "label": "scan max",
                        "kind": "point",
                        "energy_eh": float(self.energies[i_max]),
                    }
                )
        for i in ts_frames:
            waypoints.append(
                {
                    "frame_index": i,
                    "label": "transition state",
                    "kind": "transition_state",
                    "energy_eh": float(self.energies[i]),
                }
            )

        label, unit = _coordinate_label_unit(self.coordinate)

        method = str(self.extra.get("method", "RHF"))
        basis = str(self.extra.get("basis", "sto-3g"))
        functional = self.extra.get("functional")

        frame_volumes = None
        volume_grid = None
        volume_frame_index = None
        if emit_volumes_every is not None:
            step = int(emit_volumes_every)
            if step < 1:
                raise ValueError("emit_volumes_every must be >= 1")
            emit_idx = sorted(set(list(range(0, n, step)) + [n - 1])) if n else []
            frame_volumes, volume_grid, volume_frame_index = _compute_frame_volumes(
                self.geometries,
                emit_idx,
                method=method,
                basis=basis,
                functional=functional,
                spacing=volume_spacing,
                padding=volume_padding,
            )

        return write_reaction_path_qvf(
            stem,
            frames=list(self.geometries),
            energies=[float(e) for e in self.energies],
            waypoints=waypoints,
            reaction_coordinate=[float(v) for v in self.values],
            reaction_coordinate_label=label,
            reaction_coordinate_unit=unit,
            frame_volumes=frame_volumes,
            volume_grid=volume_grid,
            volume_frame_index=volume_frame_index,
            method=method,
            basis=basis,
            functional=functional,
        )


_BOHR_TO_ANGSTROM = 0.529177210903


@dataclass
class ScanResult2D:
    """Container for the output of :func:`relaxed_scan_2d` -- a 2D
    relaxed energy surface over two driven internal coordinates."""

    values_a: np.ndarray
    """Axis-A target coordinate values, shape (nA,)."""

    values_b: np.ndarray
    """Axis-B target coordinate values, shape (nB,)."""

    energies: np.ndarray
    """Relaxed energies (Ha), shape (nA, nB)."""

    geometries: list[list[Any]]
    """Relaxed geometries -- ``geometries[ia][ib]`` is the Molecule /
    PeriodicSystem at grid node (ia, ib)."""

    converged_flags: np.ndarray
    """Per-node convergence, shape (nA, nB) bool."""

    coordinate_a: CoordinateSpec
    coordinate_b: CoordinateSpec
    output_path: Optional[Path] = None
    extra: dict[str, Any] = field(default_factory=dict)

    def __repr__(self) -> str:
        n_ok = int(np.sum(self.converged_flags))
        n = self.energies.size
        return (
            f"ScanResult2D(shape={self.energies.shape}, "
            f"n_converged={n_ok}/{n}, "
            f"E_min={float(np.min(self.energies)):.6f} Ha)"
        )

    def write_qvf(
        self, path: Union[str, Path], *, include_geometries: bool = True
    ) -> Path:
        """Emit a vibe-view ``scan.surface`` QVF -- the 2D energy grid
        plus the two coordinate axes (labelled with their names + units)
        and, optionally, the relaxed geometry at every node so a viewer
        can show the structure at a clicked point.
        """
        from .output.formats.qvf import write_scan_surface_qvf

        stem = Path(path)
        if stem.suffix == ".qvf":
            stem = stem.with_suffix("")

        la, ua = _coordinate_label_unit(self.coordinate_a)
        lb, ub = _coordinate_label_unit(self.coordinate_b)

        n_a = len(self.values_a)
        n_b = len(self.values_b)
        geometries = None
        if include_geometries:
            flat = []
            for ia in range(n_a):
                for ib in range(n_b):
                    g = self.geometries[ia][ib]
                    atoms = (
                        list(g.unit_cell)
                        if isinstance(g, PeriodicSystem)
                        else list(g.atoms)
                    )
                    flat.append(
                        [
                            [float(c) * _BOHR_TO_ANGSTROM for c in a.xyz]
                            for a in atoms
                        ]
                    )
            geometries = np.asarray(flat, dtype=np.float64)

        return write_scan_surface_qvf(
            stem,
            axis_a=[float(v) for v in self.values_a],
            axis_b=[float(v) for v in self.values_b],
            energies=np.asarray(self.energies, dtype=np.float64),
            coordinate_a_label=la,
            coordinate_a_unit=ua,
            coordinate_b_label=lb,
            coordinate_b_unit=ub,
            structure=self.geometries[0][0],
            geometries=geometries,
            method=str(self.extra.get("method", "RHF")),
            basis=str(self.extra.get("basis", "sto-3g")),
            functional=self.extra.get("functional"),
        )


# ---- main driver -----------------------------------------------------------


def _resolve_kpoints(
    kpoints: Union[None, BlochKMesh, "KPoints", Sequence[int]],
    system: PeriodicSystem,
) -> BlochKMesh:
    if kpoints is None:
        return monkhorst_pack(system, (1, 1, 1))
    if hasattr(kpoints, "to_bloch_kmesh") or (
        hasattr(kpoints, "kpoints") and hasattr(kpoints, "weights")
    ):
        from .kpoints import as_bloch_kmesh

        return as_bloch_kmesh(kpoints)
    mesh = tuple(int(x) for x in kpoints)
    return monkhorst_pack(system, mesh)


def _scan_step_molecular(
    system: Molecule,
    basis: str,
    method: str,
    functional: Optional[str],
    freeze_indices: Sequence[int],
    max_iter: int,
    conv_tol_grad: float,
    progress: bool | ProgressLogger,
    extra_kwargs: dict[str, Any],
) -> Tuple[Molecule, float, bool]:
    from .molecular_optimize import _evaluate_energy, optimize_molecule
    from ._vibeqc_core import BasisSet

    n_atoms = len(list(system.atoms))
    if len(set(freeze_indices)) >= n_atoms:
        # No free degrees of freedom: skip the optimizer and just
        # evaluate the SCF energy at the constrained geometry.
        basis_obj = BasisSet(system, basis)
        energy = _evaluate_energy(
            system,
            basis_obj,
            method.lower(),
            functional=functional,
        )
        return system, float(energy), True

    res = optimize_molecule(
        system,
        basis,
        method=method.lower(),
        functional=functional,
        max_iter=max_iter,
        conv_tol_grad=conv_tol_grad,
        freeze_indices=list(freeze_indices),
        progress=progress,
        **extra_kwargs,
    )
    return res.system, float(res.energy), bool(res.converged)


def _scan_step_periodic(
    system: PeriodicSystem,
    basis: str,
    method: str,
    functional: Optional[str],
    kmesh: BlochKMesh,
    freeze_indices: Sequence[int],
    max_iter: int,
    conv_tol_grad: float,
    extra_kwargs: dict[str, Any],
) -> Tuple[PeriodicSystem, float, bool]:
    from .bipole_optimize import _run_scf, relax_atoms
    from ._vibeqc_core import (
        BasisSet,
        PeriodicKSOptions,
        PeriodicRHFOptions,
    )

    method_upper = method.upper()
    n_atoms = len(system.unit_cell)
    if len(set(freeze_indices)) >= n_atoms:
        # All atoms frozen -- bypass the optimizer (scipy can't run
        # with zero free DOFs) and just evaluate the SCF energy.
        opts = (
            PeriodicKSOptions()
            if method_upper in ("RKS", "UKS")
            else PeriodicRHFOptions()
        )
        cutoff = float(extra_kwargs.get("cutoff_bohr", 8.0))
        opts.lattice_opts.cutoff_bohr = cutoff
        opts.lattice_opts.nuclear_cutoff_bohr = cutoff
        opts.max_iter = 50
        opts.use_diis = True
        opts.conv_tol_energy = float(extra_kwargs.get("scf_conv_tol", 1e-7))
        basis_obj = BasisSet(system.unit_cell_molecule(), basis)
        # Strip kwargs that _run_scf doesn't accept.
        bipole_kwargs = {
            k: v
            for k, v in extra_kwargs.items()
            if k not in ("cutoff_bohr", "scf_conv_tol")
        }
        energy, _ = _run_scf(
            system,
            basis_obj,
            kmesh,
            opts,
            method_upper,
            functional,
            **bipole_kwargs,
        )
        return system, float(energy), True

    res = relax_atoms(
        system,
        basis,
        kmesh,
        method=method_upper,
        functional=functional,
        max_iter=max_iter,
        conv_tol_grad=conv_tol_grad,
        freeze_indices=list(freeze_indices),
        **extra_kwargs,
    )
    return res.system, float(res.energy), bool(res.converged)


def relaxed_scan(
    system: Union[Molecule, PeriodicSystem],
    basis: str,
    coordinate: CoordinateSpec,
    values: Sequence[float],
    *,
    method: str = "RHF",
    functional: Optional[str] = None,
    kpoints: Union[None, BlochKMesh, "KPoints", Sequence[int]] = None,
    output: Optional[str] = None,
    freeze_indices: Optional[Sequence[int]] = None,
    max_iter: int = 30,
    conv_tol_grad: float = 1e-4,
    progress: bool | ProgressLogger = False,
    **kwargs: Any,
) -> ScanResult:
    """Drive one internal coordinate across a sequence of values, relaxing all
    other degrees of freedom at each point.

    Parameters
    ----------
    system
        Starting geometry -- :class:`Molecule` for a molecular scan,
        :class:`PeriodicSystem` for a periodic scan.
    basis
        Basis-set name (rebuilt at each scan point).
    coordinate
        Constraint definition. One of:

        * ``("bond", i, j)`` -- distance between atoms ``i`` and ``j`` (bohr).
        * ``("angle", i, j, k)`` -- bond angle at vertex ``j`` (radians).
        * ``("dihedral", i, j, k, l)`` -- dihedral (radians).
    values
        Target values for the constrained coordinate, in bohr (bonds)
        or radians (angles, dihedrals).
    method
        ``"RHF"``, ``"UHF"``, ``"RKS"``, or ``"UKS"``.
    functional
        XC functional for RKS / UKS.
    kpoints
        Periodic only. :class:`BlochKMesh`, :class:`KPoints`, or a 3-tuple
        ``(n1, n2, n3)`` for a Monkhorst-Pack mesh. Defaults to Γ-only
        ``(1, 1, 1)``.
    output
        If set, write a ``{output}.qvf`` reaction-path archive at the
        end of the scan (calls :meth:`ScanResult.write_qvf`).
    freeze_indices
        Additional atoms to freeze at every scan point (in addition
        to the constraint atoms, which are always frozen). Typical use
        on a slab: pass ``SlabInfo.bottom_layer_indices(n)`` so the
        slab bottom stays fixed while the adsorbate moves.
    max_iter
        Maximum optimizer iterations per scan point.
    conv_tol_grad
        Gradient convergence tolerance per scan point (Ha/bohr).
    progress
        If True, emit live progress to stdout per scan point. Pass a
        :class:`~vibeqc.progress.ProgressLogger` to select another live sink.
    **kwargs
        Forwarded to the underlying optimizer
        (:func:`vibeqc.optimize_molecule` or
        :func:`vibeqc.relax_atoms`).

    Returns
    -------
    ScanResult
    """
    is_periodic = isinstance(system, PeriodicSystem)
    plog = resolve_progress(progress, verbose=2)
    n_atoms = (
        len(system.unit_cell) if is_periodic else len(list(system.atoms))
    )
    _validate_coordinate(coordinate, n_atoms)

    constraint_atoms = _coordinate_atom_indices(coordinate)
    user_frozen = (
        [] if freeze_indices is None else [int(i) for i in freeze_indices]
    )
    step_frozen = sorted(set(constraint_atoms) | set(user_frozen))

    values_arr = np.asarray(list(values), dtype=float)
    if values_arr.size == 0:
        raise ValueError("relaxed_scan: 'values' is empty")

    kmesh = _resolve_kpoints(kpoints, system) if is_periodic else None

    geometries: list[Any] = []
    energies: list[float] = []
    converged_flags: list[bool] = []

    # Warm-start each scan point from the previous relaxed geometry --
    # cheaper and gives smoother paths.
    current = system
    for idx, target in enumerate(values_arr):
        positions = _get_positions(current)
        new_positions = _apply_constraint(positions, coordinate, float(target))
        seed = _with_positions(current, new_positions)

        kind = coordinate[0]
        unit = "bohr" if kind == "bond" else "rad"
        plog.write_raw(
            f"\n  scan point {idx + 1}/{len(values_arr)}: "
            f"{kind}{tuple(constraint_atoms)} -> "
            f"{float(target):.4f} {unit}"
        )

        if is_periodic:
            relaxed, energy, ok = _scan_step_periodic(
                seed,
                basis,
                method,
                functional,
                kmesh,  # type: ignore[arg-type]
                step_frozen,
                max_iter,
                conv_tol_grad,
                kwargs,
            )
        else:
            relaxed, energy, ok = _scan_step_molecular(
                seed,
                basis,
                method,
                functional,
                step_frozen,
                max_iter,
                conv_tol_grad,
                plog,
                kwargs,
            )

        geometries.append(relaxed)
        energies.append(energy)
        converged_flags.append(ok)
        current = relaxed

    result = ScanResult(
        values=values_arr,
        energies=np.asarray(energies, dtype=float),
        geometries=geometries,
        converged_flags=converged_flags,
        coordinate=coordinate,
        extra={
            "method": method,
            "basis": basis,
            "functional": functional,
        },
    )

    if output is not None:
        result.output_path = result.write_qvf(output)

    return result


def relaxed_scan_2d(
    system: Any,
    basis: str,
    coordinate_a: CoordinateSpec,
    values_a: Sequence[float],
    coordinate_b: CoordinateSpec,
    values_b: Sequence[float],
    *,
    method: str = "RHF",
    functional: Optional[str] = None,
    kpoints: Union[None, BlochKMesh, "KPoints", Sequence[int]] = None,
    output: Optional[str] = None,
    freeze_indices: Optional[Sequence[int]] = None,
    max_iter: int = 30,
    conv_tol_grad: float = 1e-4,
    progress: bool | ProgressLogger = False,
    **kwargs: Any,
) -> ScanResult2D:
    """Drive *two* internal coordinates over a grid, relaxing all other
    degrees of freedom at each node -- a 2D relaxed energy surface.

    The two coordinates should be independent (disjoint atom sets, or at
    least not sharing the moved atom); both are constrained (frozen)
    while the rest of the system relaxes at each ``(a, b)`` node. The
    grid is walked in row-major order over ``(a, b)`` with warm-starts
    from the previous node for speed and smoothness.

    Parameters mirror :func:`relaxed_scan` but doubled for the second
    coordinate. Returns a :class:`ScanResult2D`; if ``output`` is set,
    a ``{output}.qvf`` ``scan.surface`` archive is written.
    """
    is_periodic = isinstance(system, PeriodicSystem)
    plog = resolve_progress(progress, verbose=2)
    n_atoms = len(system.unit_cell) if is_periodic else len(list(system.atoms))
    _validate_coordinate(coordinate_a, n_atoms)
    _validate_coordinate(coordinate_b, n_atoms)

    constraint_atoms = _coordinate_atom_indices(coordinate_a) + (
        _coordinate_atom_indices(coordinate_b)
    )
    user_frozen = [] if freeze_indices is None else [int(i) for i in freeze_indices]
    step_frozen = sorted(set(constraint_atoms) | set(user_frozen))

    a_arr = np.asarray(list(values_a), dtype=float)
    b_arr = np.asarray(list(values_b), dtype=float)
    if a_arr.size == 0 or b_arr.size == 0:
        raise ValueError("relaxed_scan_2d: 'values_a' / 'values_b' is empty")

    kmesh = _resolve_kpoints(kpoints, system) if is_periodic else None
    n_a, n_b = len(a_arr), len(b_arr)

    energies = np.zeros((n_a, n_b), dtype=float)
    converged = np.zeros((n_a, n_b), dtype=bool)
    geometries: list[list[Any]] = [[None] * n_b for _ in range(n_a)]

    def _relax(seed: Any) -> Tuple[Any, float, bool]:
        if is_periodic:
            return _scan_step_periodic(
                seed, basis, method, functional, kmesh, step_frozen,
                max_iter, conv_tol_grad, kwargs,
            )
        return _scan_step_molecular(
            seed, basis, method, functional, step_frozen,
            max_iter, conv_tol_grad, plog, kwargs,
        )

    row_start = system  # warm-start seed for the first node of each row
    for ia, va in enumerate(a_arr):
        current = row_start
        for ib, vb in enumerate(b_arr):
            positions = _get_positions(current)
            positions = _apply_constraint(positions, coordinate_a, float(va))
            positions = _apply_constraint(positions, coordinate_b, float(vb))
            seed = _with_positions(current, positions)

            plog.write_raw(
                f"\n  node ({ia + 1}/{n_a}, {ib + 1}/{n_b}): "
                f"a={float(va):.4f}, b={float(vb):.4f}"
            )

            relaxed, energy, ok = _relax(seed)
            energies[ia, ib] = energy
            converged[ia, ib] = ok
            geometries[ia][ib] = relaxed
            current = relaxed
            if ib == 0:
                row_start = relaxed  # next row warm-starts from this column-0

    result = ScanResult2D(
        values_a=a_arr,
        values_b=b_arr,
        energies=energies,
        geometries=geometries,
        converged_flags=converged,
        coordinate_a=coordinate_a,
        coordinate_b=coordinate_b,
        extra={"method": method, "basis": basis, "functional": functional},
    )

    if output is not None:
        result.output_path = result.write_qvf(output)

    return result
