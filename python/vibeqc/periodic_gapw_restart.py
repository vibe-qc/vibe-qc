"""GPW SCF restart-state file -- a single ``.npz`` archive that
round-trips a converged :class:`GpwScfResult` or
:class:`GpwMultiKScfResult`.

This is vibe-qc's analogue of GPAW's ``.gpw`` file. A single
``numpy.savez_compressed`` archive carries enough information to
reconstruct:

* the :class:`vibeqc._vibeqc_core.PeriodicSystem` (lattice + atoms),
* the :class:`vibeqc.BasisSet` (rebuilt from the basis name and
  the atomic centres in the unit cell),
* the :class:`PlaneWaveGrid` the SCF ran on,
* the converged density, MO coefficients, MO energies, energy
  breakdown, SCF trace, and (for multi-k) the k-mesh
  fractional coordinates + weights + per-k MOs.

Public API
----------

``save_gpw_result(path, result, basis, system)`` -- write the
archive.

``load_gpw_result(path)`` -- read it back into a plain ``dict``
(``kind``, scalar fields, numpy arrays) **plus** reconstructed
``system`` / ``basis`` / ``grid`` keys.

``describe_gpw_result(path)`` -- return a short human-readable
summary string (mentions GPW + the energy).

Format
------

The archive layout is:

* ``kind`` -- ``"gpw_scf"`` or ``"gpw_multi_k_scf"`` (a numpy
  string).
* ``format_version`` -- integer; currently ``1``.
* ``energy`` -- total energy (Ha).
* ``converged`` -- boolean.
* ``n_iter`` -- number of SCF iterations executed.
* ``density`` -- ``(n_basis, n_basis)`` AO density matrix.
* ``mo_coeffs`` -- ``(n_basis, n_basis)`` MO coefficients (Γ
  point for the multi-k case: the first k-point).
* ``mo_energies`` -- ``(n_basis,)`` MO energies.
* ``grid_nx`` / ``grid_ny`` / ``grid_nz`` -- int grid extents.
* ``grid_lattice`` -- ``(3, 3)`` grid lattice (bohr).
* ``grid_cutoff_ha`` -- float or NaN if unknown.
* ``lattice`` -- ``(3, 3)`` system lattice (bohr).
* ``unit_cell_Z`` -- ``(n_atoms,)`` atomic numbers.
* ``unit_cell_xyz`` -- ``(n_atoms, 3)`` Cartesian positions (bohr).
* ``basis_name`` -- basis-set name (e.g. ``"sto-3g"``).
* ``breakdown_e_kinetic`` / ``breakdown_e_nuclear_attraction`` /
  ``breakdown_e_hartree`` / ``breakdown_e_hf_exchange`` /
  ``breakdown_e_xc`` / ``breakdown_e_nuclear_repulsion`` --
  per-term scalars (Ha).
* ``breakdown_functional`` -- functional name string (empty if
  none).
* ``scf_trace_iter`` / ``scf_trace_energy`` /
  ``scf_trace_delta_e`` / ``scf_trace_grad_norm`` /
  ``scf_trace_e_xc`` -- 1D arrays of per-iter SCF records (one
  entry per iteration; may be empty for legacy results).

Multi-k extras (``kind == "gpw_multi_k_scf"``):

* ``kpoints`` -- ``(n_k, 3)`` fractional k-point coordinates.
* ``kweights`` -- ``(n_k,)`` k-point weights (sum to 1).
* ``mo_coeffs_k`` -- ``(n_k, n_basis, n_basis)`` complex MO
  coefficients per k.
* ``mo_energies_k`` -- ``(n_k, n_basis)`` real MO eigenvalues
  per k.

The reconstructed ``basis`` is rebuilt by routing the
``unit_cell_Z`` / ``unit_cell_xyz`` through
:class:`vibeqc.Molecule` and :class:`vibeqc.BasisSet` with the
saved ``basis_name``. Atom ordering matches the original cell
exactly, so the density matrix indexing is preserved.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any, Union

import numpy as np

from .periodic_gapw_grid import PlaneWaveGrid


_FORMAT_VERSION = 1


def _system_arrays(system) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Pull ``(lattice, Z, xyz)`` arrays out of a ``PeriodicSystem``."""
    lattice = np.asarray(system.lattice, dtype=float)
    atoms = list(system.unit_cell)
    Z = np.asarray([int(a.Z) for a in atoms], dtype=np.int64)
    xyz = np.asarray(
        [[float(c) for c in a.xyz] for a in atoms], dtype=float
    )
    return lattice, Z, xyz


def _trace_arrays(scf_trace: tuple) -> dict[str, np.ndarray]:
    """Lift a tuple-of-dicts ``scf_trace`` into a per-field 1D bundle."""
    iters: list[int] = []
    energies: list[float] = []
    delta_e: list[float] = []
    grad_norm: list[float] = []
    e_xc: list[float] = []
    for rec in scf_trace or ():
        iters.append(int(rec.get("iter", -1)))
        energies.append(float(rec.get("energy", float("nan"))))
        delta_e.append(float(rec.get("delta_e", float("nan"))))
        grad_norm.append(float(rec.get("grad_norm", float("nan"))))
        e_xc.append(float(rec.get("e_xc", float("nan"))))
    return {
        "scf_trace_iter": np.asarray(iters, dtype=np.int64),
        "scf_trace_energy": np.asarray(energies, dtype=float),
        "scf_trace_delta_e": np.asarray(delta_e, dtype=float),
        "scf_trace_grad_norm": np.asarray(grad_norm, dtype=float),
        "scf_trace_e_xc": np.asarray(e_xc, dtype=float),
    }


def _grid_arrays(grid: PlaneWaveGrid) -> dict[str, Any]:
    return {
        "grid_nx": np.int64(grid.nx),
        "grid_ny": np.int64(grid.ny),
        "grid_nz": np.int64(grid.nz),
        "grid_lattice": np.asarray(grid.lattice_bohr, dtype=float),
        "grid_cutoff_ha": np.float64(
            grid.cutoff_ha if grid.cutoff_ha is not None else float("nan")
        ),
    }


def save_gpw_result(
    path: Union[str, Path],
    result: Any,
    basis: Any,
    system: Any,
) -> Path:
    """Serialize a converged GPW SCF result to a single ``.npz``
    restart archive.

    Parameters
    ----------
    path
        Destination path. Created (or overwritten) on the local
        filesystem. The ``.npz`` extension is appended by
        :func:`numpy.savez_compressed` if it is not already there;
        the returned path reflects what was actually written.
    result
        A :class:`vibeqc.periodic_gapw_j.GpwScfResult` (Γ-only)
        or :class:`vibeqc.periodic_gapw_j.GpwMultiKScfResult`
        (multi-k).
    basis
        :class:`vibeqc.BasisSet` the SCF was run with. Only
        ``basis.name`` is persisted -- on load the basis is rebuilt
        on the saved unit cell.
    system
        :class:`vibeqc._vibeqc_core.PeriodicSystem` -- the lattice
        + atoms.

    Returns
    -------
    pathlib.Path
        Path actually written (with ``.npz`` appended if it was
        missing).
    """
    # Lazy import to avoid the GAPWExperimentalWarning chain at
    # module-import time.
    from .periodic_gapw_j import GpwMultiKScfResult, GpwScfResult

    if not isinstance(result, (GpwScfResult, GpwMultiKScfResult)):
        raise TypeError(
            f"save_gpw_result: result must be GpwScfResult or "
            f"GpwMultiKScfResult; got {type(result).__name__}."
        )

    lattice, Z, xyz = _system_arrays(system)
    breakdown = result.breakdown
    functional_name = (
        "" if breakdown.functional is None else str(breakdown.functional)
    )

    payload: dict[str, Any] = {
        "format_version": np.int64(_FORMAT_VERSION),
        "energy": np.float64(result.energy),
        "converged": np.bool_(bool(result.converged)),
        "n_iter": np.int64(int(result.n_iter)),
        "density": np.asarray(result.density, dtype=float),
        "lattice": lattice,
        "unit_cell_Z": Z,
        "unit_cell_xyz": xyz,
        "basis_name": np.str_(str(basis.name)),
        "breakdown_e_kinetic": np.float64(breakdown.e_kinetic),
        "breakdown_e_nuclear_attraction": np.float64(
            breakdown.e_nuclear_attraction
        ),
        "breakdown_e_hartree": np.float64(breakdown.e_hartree),
        "breakdown_e_hf_exchange": np.float64(breakdown.e_hf_exchange),
        "breakdown_e_xc": np.float64(breakdown.e_xc),
        "breakdown_e_nuclear_repulsion": np.float64(
            breakdown.e_nuclear_repulsion
        ),
        "breakdown_functional": np.str_(functional_name),
    }
    payload.update(_grid_arrays(result.grid))
    payload.update(_trace_arrays(getattr(result, "scf_trace", ()) or ()))

    if isinstance(result, GpwMultiKScfResult):
        payload["kind"] = np.str_("gpw_multi_k_scf")
        kmesh = result.kmesh
        kpoints = np.asarray(kmesh.kpoints, dtype=float)
        kweights = np.asarray(kmesh.weights, dtype=float)
        mo_coeffs_k = np.asarray(
            [np.asarray(c, dtype=complex) for c in result.mo_coeffs_k],
            dtype=complex,
        )
        mo_energies_k = np.asarray(
            [np.asarray(e, dtype=float) for e in result.mo_energies_k],
            dtype=float,
        )
        payload["kpoints"] = kpoints
        payload["kweights"] = kweights
        payload["mo_coeffs_k"] = mo_coeffs_k
        payload["mo_energies_k"] = mo_energies_k
        # Γ-equivalent single-k views: pick the first k-point so
        # the unified loader has ``mo_coeffs`` / ``mo_energies``
        # without a multi-k consumer needing the per-k arrays.
        payload["mo_coeffs"] = mo_coeffs_k[0]
        payload["mo_energies"] = mo_energies_k[0]
    else:
        payload["kind"] = np.str_("gpw_scf")
        payload["mo_coeffs"] = np.asarray(result.mo_coeffs, dtype=float)
        payload["mo_energies"] = np.asarray(
            result.mo_energies, dtype=float
        )

    out_path = Path(path)
    # np.savez_compressed appends '.npz' if missing.
    np.savez_compressed(out_path, **payload)
    if out_path.suffix != ".npz":
        out_path = out_path.with_suffix(out_path.suffix + ".npz")
    return out_path


def _rebuild_system(lattice: np.ndarray, Z: np.ndarray,
                     xyz: np.ndarray):
    from . import _vibeqc_core as _core
    system = _core.PeriodicSystem()
    system.dim = 3
    system.lattice = np.asarray(lattice, dtype=float)
    system.unit_cell = [
        _core.Atom(int(z), [float(x) for x in pos])
        for z, pos in zip(Z, xyz)
    ]
    return system


def _rebuild_basis(Z: np.ndarray, xyz: np.ndarray, basis_name: str):
    # Import lazily -- keeps the module light.
    from . import Atom, BasisSet, Molecule

    atoms = [Atom(int(z), [float(c) for c in pos])
             for z, pos in zip(Z, xyz)]
    mol = Molecule(atoms, charge=0, multiplicity=1)
    return BasisSet(mol, str(basis_name))


def _rebuild_grid(arr) -> PlaneWaveGrid:
    nx = int(arr["grid_nx"])
    ny = int(arr["grid_ny"])
    nz = int(arr["grid_nz"])
    lattice = np.asarray(arr["grid_lattice"], dtype=float)
    cutoff = float(arr["grid_cutoff_ha"])
    if not np.isfinite(cutoff):
        cutoff = None
    return PlaneWaveGrid(lattice, nx, ny, nz, cutoff_ha=cutoff)


def load_gpw_result(path: Union[str, Path]) -> dict[str, Any]:
    """Read a ``.npz`` GPW restart archive into a dict.

    Parameters
    ----------
    path
        Path to the archive written by :func:`save_gpw_result`.

    Returns
    -------
    dict
        Keys include every scalar / array field from the archive
        (see module docstring) plus three reconstructed objects:

        * ``system`` -- a fresh :class:`PeriodicSystem`.
        * ``basis`` -- a fresh :class:`vibeqc.BasisSet` (rebuilt
          via :class:`vibeqc.Molecule` + the saved basis name on
          the saved unit cell).
        * ``grid`` -- a fresh :class:`PlaneWaveGrid`.

    Raises
    ------
    FileNotFoundError
        If ``path`` does not exist.
    """
    p = Path(path)
    if not p.exists():
        raise FileNotFoundError(
            f"load_gpw_result: no such file: {p}"
        )

    with np.load(p, allow_pickle=False) as arr:
        out: dict[str, Any] = {}
        out["format_version"] = int(arr["format_version"])
        out["kind"] = str(arr["kind"])
        out["energy"] = float(arr["energy"])
        out["converged"] = bool(arr["converged"])
        out["n_iter"] = int(arr["n_iter"])
        out["density"] = np.asarray(arr["density"])
        out["mo_coeffs"] = np.asarray(arr["mo_coeffs"])
        out["mo_energies"] = np.asarray(arr["mo_energies"])
        out["grid_nx"] = int(arr["grid_nx"])
        out["grid_ny"] = int(arr["grid_ny"])
        out["grid_nz"] = int(arr["grid_nz"])
        out["grid_lattice"] = np.asarray(arr["grid_lattice"])
        out["grid_cutoff_ha"] = float(arr["grid_cutoff_ha"])
        out["lattice"] = np.asarray(arr["lattice"])
        out["unit_cell_Z"] = np.asarray(arr["unit_cell_Z"])
        out["unit_cell_xyz"] = np.asarray(arr["unit_cell_xyz"])
        out["basis_name"] = str(arr["basis_name"])
        for k in (
            "breakdown_e_kinetic",
            "breakdown_e_nuclear_attraction",
            "breakdown_e_hartree",
            "breakdown_e_hf_exchange",
            "breakdown_e_xc",
            "breakdown_e_nuclear_repulsion",
        ):
            out[k] = float(arr[k])
        out["breakdown_functional"] = str(arr["breakdown_functional"])
        for k in (
            "scf_trace_iter",
            "scf_trace_energy",
            "scf_trace_delta_e",
            "scf_trace_grad_norm",
            "scf_trace_e_xc",
        ):
            out[k] = np.asarray(arr[k])
        if out["kind"] == "gpw_multi_k_scf":
            out["kpoints"] = np.asarray(arr["kpoints"])
            out["kweights"] = np.asarray(arr["kweights"])
            out["mo_coeffs_k"] = np.asarray(arr["mo_coeffs_k"])
            out["mo_energies_k"] = np.asarray(arr["mo_energies_k"])

        # Reconstruct objects.
        out["system"] = _rebuild_system(
            out["lattice"], out["unit_cell_Z"], out["unit_cell_xyz"]
        )
        out["basis"] = _rebuild_basis(
            out["unit_cell_Z"], out["unit_cell_xyz"], out["basis_name"]
        )
        out["grid"] = _rebuild_grid(arr)
    return out


def describe_gpw_result(path: Union[str, Path]) -> str:
    """Return a short human-readable summary of a GPW restart file.

    The string is guaranteed to contain the substring ``"GPW"`` and
    the total energy in Hartree, so a downstream log can grep for
    either.
    """
    data = load_gpw_result(path)
    kind = data["kind"]
    label = "GPW SCF" if kind == "gpw_scf" else "GPW multi-k SCF"
    n_atoms = int(data["unit_cell_Z"].shape[0])
    n_basis = int(data["density"].shape[0])
    nx, ny, nz = data["grid_nx"], data["grid_ny"], data["grid_nz"]
    functional = data["breakdown_functional"] or "HF"
    converged = "converged" if data["converged"] else "not converged"
    extra = ""
    if kind == "gpw_multi_k_scf":
        extra = f", {data['kpoints'].shape[0]} k-points"
    return (
        f"{label} restart file: {Path(path).name}\n"
        f"  functional = {functional}, {converged} in "
        f"{data['n_iter']} iter{extra}\n"
        f"  cell: {n_atoms} atoms, basis = {data['basis_name']} "
        f"({n_basis} functions)\n"
        f"  grid: {nx}x{ny}x{nz}\n"
        f"  energy = {data['energy']:.10f} Ha"
    )
