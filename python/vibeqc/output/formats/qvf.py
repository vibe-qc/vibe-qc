"""QVF (Quantum Visualization Format) writer for vibe-qc.

Produces ``{stem}.qvf`` -- a zip archive with a JSON manifest
(``manifest.json``) and typed binary or text payloads, one per output
section. Each section is keyed by a ``kind`` string from the QVF
registry. Consumers support whichever kinds they recognise and
silently skip the rest.

v1 scope (implemented)
----------------------
* ``structure`` -- atom positions, Z, labels, optional lattice
* ``volume.density`` -- total electron density as raw float32 .dat
* ``volume.orbital`` -- per-MO wavefunction on a grid
* ``atom_properties`` -- Mulliken / Löwdin / Hirshfeld charges, spin
  populations
* ``trajectory`` -- geometry-optimisation / IRC frames
* ``vibrations`` -- normal-mode frequencies + displacements
* ``spectra.ir`` -- IR spectrum from Hessian
* ``bands`` -- band structure (eigenvalues + k-path)
* ``provenance`` -- method, functional, basis, energy, convergence
* ``citations`` -- BibTeX references

Producer rules
--------------
* Float32 default for volumetric arrays; float64 opt-in via
  ``volume_dtype="float64"``.
* Deflate default compression; zstd if ``zipfile-zstd`` is importable.
* ``manifest.json`` always stored uncompressed.
* Every binary member carries a sha256 hex digest in the manifest.

Public API
----------

``write_qvf(stem, plan, **context) -> Path``
    Write ``{stem}.qvf`` from an :class:`OutputPlan` and result data.
    Returns the written path.

``validate_qvf(path) -> dict``
    Open a ``.qvf`` file and validate its manifest + binary payloads.
    Returns a validation report dict.
"""

from __future__ import annotations

import copy
import datetime as _dt
import hashlib
import io
import json
import os
import struct
import sys
import tempfile
import zipfile
from pathlib import Path
from typing import Any, Optional, Sequence

import numpy as np

from ..._primitive_norm import libint_primitive_norm as _primitive_norm
from ...correlation_conventions import effective_electron_count
from ...spin_channels import spin_densities
from .._text_safety import safe_json_bytes
from ..plan import OutputPlan
from .._stem_paths import stem_sibling

__all__ = [
    "write_qvf",
    "qvf_bytes",
    "validate_qvf",
    "write_reaction_path_qvf",
    "write_scan_surface_qvf",
    "QVF_FORMAT_VERSION",
    "qvf_density_data",
    "qvf_mo_data",
    "qvf_wf_data",
    "qvf_bloch_wf_data",
    "scf_history_from_result",
    "qvf_ao_data",
    "QVF_BLOCH_WF_KIND",
]

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

QVF_FORMAT_VERSION = 1
QVF_FORMAT_VERSION_V2 = 2
_SCHEMA_URI = "https://vibe-qc.org/spec/qvf/1/manifest.schema.json"
_SCHEMA_URI_V2 = "https://vibe-qc.org/spec/qvf/2/manifest.schema.json"
_PRODUCER_NAME = "vibe-qc"
QVF_BLOCH_WF_KIND = "x_vibeqc.bloch_wavefunction"

# Bohr -> Ångström (CODATA 2018).  Must match vibeqc.output.formats.xyz.
_BOHR_TO_ANGSTROM = 0.529177210903
# Hartree -> eV (CODATA 2018).
_HARTREE_TO_EV = 27.211386245988

# Single source of truth for archive size caps. The writer's voxel
# guard and the validator's per-zip-member zip-bomb guard must agree:
# a payload write_qvf will produce must not later be rejected by
# validate_qvf as "too large". 1 Gvoxel covers ~8 GiB float64, which
# is also the per-member uncompressed cap for binary blobs (large MO
# coefficient matrices and trajectory coord arrays count against the
# same limit, not just voxel grids). Compressed-on-disk bytes are
# typically much smaller; the bomb guard runs against the
# uncompressed `file_size` reported by zipfile.
_MAX_VOXELS = 1024**3
_MAX_MEMBER_UNCOMPRESSED_BYTES = _MAX_VOXELS * 8  # float64 worst case

_REAL_ARTIFACT_IMAG_ABS_TOL = 1.0e-10
_REAL_ARTIFACT_IMAG_REL_TOL = 1.0e-7


def _real_array_for_artifact(
    data: Any,
    *,
    dtype: np.dtype | type | str,
    label: str,
    abs_tol: float = _REAL_ARTIFACT_IMAG_ABS_TOL,
    rel_tol: float = _REAL_ARTIFACT_IMAG_REL_TOL,
) -> np.ndarray:
    """Return a real ndarray for real-only QVF members.

    NumPy's ``np.asarray(complex, dtype=float)`` emits ``ComplexWarning``
    while discarding the imaginary component implicitly. QVF v1 scalar
    volumes and Γ-point wavefunction payloads are real-valued by schema, so
    do the projection here intentionally: tiny imaginary residuals from
    Bloch/time-reversal folds are accepted; genuinely complex payloads raise
    and let the optional writer be skipped instead of writing misleading data.
    """
    arr = np.asarray(data)
    if np.iscomplexobj(arr):
        if arr.size:
            max_imag = float(np.max(np.abs(arr.imag)))
            max_real = float(np.max(np.abs(arr.real)))
        else:
            max_imag = 0.0
            max_real = 0.0
        limit = max(float(abs_tol), float(rel_tol) * max(max_real, 1.0))
        if max_imag > limit:
            raise ValueError(
                f"{label} is real-only but carries a non-negligible "
                f"imaginary component (max|Im|={max_imag:.2e}, "
                f"max|Re|={max_real:.2e}, tolerance={limit:.2e})"
            )
        arr = arr.real
    return np.asarray(arr, dtype=dtype)


def _mo_coefficients_for_artifact(
    data: Any,
    *,
    label: str,
    abs_tol: float = _REAL_ARTIFACT_IMAG_ABS_TOL,
    rel_tol: float = _REAL_ARTIFACT_IMAG_REL_TOL,
) -> tuple[np.ndarray, str]:
    """Return a float64 MO-coefficient payload plus its QVF encoding.

    Real coefficients are stored as the historical 2-D float64 matrix.
    Genuine complex Bloch coefficients are stored losslessly as a float64 array
    with a trailing component axis ``[..., 0] = Re`` and ``[..., 1] = Im``.
    Tiny imaginary residuals keep the old real encoding.
    """
    arr = np.asarray(data)
    if np.iscomplexobj(arr):
        if arr.size:
            max_imag = float(np.max(np.abs(arr.imag)))
            max_real = float(np.max(np.abs(arr.real)))
        else:
            max_imag = 0.0
            max_real = 0.0
        limit = max(float(abs_tol), float(rel_tol) * max(max_real, 1.0))
        if max_imag > limit:
            packed = np.empty(arr.shape + (2,), dtype=np.float64)
            packed[..., 0] = arr.real
            packed[..., 1] = arr.imag
            return np.ascontiguousarray(packed), "complex_split_last_axis"
        arr = arr.real
    return np.ascontiguousarray(np.asarray(arr, dtype=np.float64)), "real"


def _complex_split_array(data: Any) -> np.ndarray:
    """Pack a complex array as float64 ``[..., real/imag]`` components."""
    arr = np.asarray(data, dtype=np.complex128)
    packed = np.empty(arr.shape + (2,), dtype=np.float64)
    packed[..., 0] = arr.real
    packed[..., 1] = arr.imag
    return np.ascontiguousarray(packed)

# Element symbols, Z = 0..118.  Mirrors _ELEMENT_SYMBOLS in trajectory.py.
_ELEMENT_SYMBOLS = (
    "X",  # 0 -- placeholder / ghost atom
    "H",
    "He",
    "Li",
    "Be",
    "B",
    "C",
    "N",
    "O",
    "F",
    "Ne",
    "Na",
    "Mg",
    "Al",
    "Si",
    "P",
    "S",
    "Cl",
    "Ar",
    "K",
    "Ca",
    "Sc",
    "Ti",
    "V",
    "Cr",
    "Mn",
    "Fe",
    "Co",
    "Ni",
    "Cu",
    "Zn",
    "Ga",
    "Ge",
    "As",
    "Se",
    "Br",
    "Kr",
    "Rb",
    "Sr",
    "Y",
    "Zr",
    "Nb",
    "Mo",
    "Tc",
    "Ru",
    "Rh",
    "Pd",
    "Ag",
    "Cd",
    "In",
    "Sn",
    "Sb",
    "Te",
    "I",
    "Xe",
    "Cs",
    "Ba",
    "La",
    "Ce",
    "Pr",
    "Nd",
    "Pm",
    "Sm",
    "Eu",
    "Gd",
    "Tb",
    "Dy",
    "Ho",
    "Er",
    "Tm",
    "Yb",
    "Lu",
    "Hf",
    "Ta",
    "W",
    "Re",
    "Os",
    "Ir",
    "Pt",
    "Au",
    "Hg",
    "Tl",
    "Pb",
    "Bi",
    "Po",
    "At",
    "Rn",
    "Fr",
    "Ra",
    "Ac",
    "Th",
    "Pa",
    "U",
    "Np",
    "Pu",
    "Am",
    "Cm",
    "Bk",
    "Cf",
    "Es",
    "Fm",
    "Md",
    "No",
    "Lr",
    "Rf",
    "Db",
    "Sg",
    "Bh",
    "Hs",
    "Mt",
    "Ds",
    "Rg",
    "Cn",
    "Nh",
    "Fl",
    "Mc",
    "Lv",
    "Ts",
    "Og",
)


def _symbol(z: int) -> str:
    if 0 <= z < len(_ELEMENT_SYMBOLS):
        return _ELEMENT_SYMBOLS[z]
    return "X"


# Canonical section kinds that the writer can emit. This list must
# stay in lock-step with the ``Section.oneOf`` branches in
# qvf_manifest.schema.json -- tests/test_qvf_round_trip.py carries the
# schema-drift guard. ``provenance`` and ``viewer_defaults`` are root
# keys, not section kinds; they intentionally don't appear here.
_IMPLEMENTED_KINDS = frozenset(
    {
        "structure",
        "volume.density",
        "volume.orbital",
        "volume.spin",
        "volume.elf",
        "volume.difference",
        "volume.generic",
        "volume.potential",
        "volume.rdg",
        "fermi_surface",
        "phonon_bands",
        "phonon_dos",
        "equation_of_state",
        "basis.ao",
        "wavefunction.gto",
        "atom_properties",
        "trajectory",
        "reaction.path",
        "reaction.waypoints",
        "scan.surface",
        "vibrations",
        "spectra.ir",
        "spectra.raman",
        "spectra.uvvis",
        "spectra.ecd",
        "spectra.vcd",
        "spectra.nmr",
        "spectra.epr",
        "spectra.generic",
        "bands",
        "structure.symmetry",
        "bonds",
        "scf_history",
        "citations",
        "dos.total",
        "dos.projected",
        "bond_orders",
        "topology.qtaim",
        "dos.coop",
        "dos.cohp",
        "run.record",
        "job.spec",
    }
)

# Kinds reserved in the design doc but not yet implemented in the
# writer. The validator accepts them (so a vendor producer can ship
# them ahead of the canonical writer) but the writer never emits one.
#
# `basis` was reserved before `wavefunction.gto` landed -- back then
# we anticipated a separate kind that would carry just the AO basis
# shells. `wavefunction.gto` now ships basis shells + MO coefficients
# in a single section, so a standalone `basis` kind is dead weight
# and has been removed from the registry.
_RESERVED_KINDS = frozenset(
    {
        "volume.orbital_projection",
        "topology.elf_basins",
        "projections.lcao",
    }
)


# ---------------------------------------------------------------------------
# Grid-evaluation convenience helpers
# ---------------------------------------------------------------------------
#
# These produce pre-packaged data dicts that can be passed directly to
# write_qvf() as ``volume_data=`` and ``mo_data=``.  They call the
# existing grid evaluators in vibeqc.cube and repackage the results.


def qvf_density_data(
    result: Any,
    basis: Any,
    molecule: Any,
    *,
    spacing: float = 0.25,
    padding: float = 4.0,
    label: str = "Electron density",
) -> dict[str, tuple]:
    """Evaluate total electron density on a uniform grid and return a
    dict suitable for ``write_qvf(..., volume_data=...)``.

    Parameters
    ----------
    result
        Converged SCF result with a ``.density`` attribute (RHF/RKS)
        or ``.density_alpha`` + ``.density_beta`` (UHF/UKS).
    basis
        :class:`BasisSet` used in the calculation.
    molecule
        :class:`Molecule` defining the atomic positions.
    spacing
        Voxel spacing in bohr (default 0.25).
    padding
        Extra headroom around the molecular bounding box in bohr.
    label
        Human-readable label for the density section.

    Returns
    -------
    dict
        ``{label: (data_3d, origin_3, span_3x3)}`` -- pass as
        ``volume_data=`` to :func:`write_qvf`.
    """
    from vibeqc.cube import (
        CubeGrid,
        _density_on_grid,
        make_uniform_grid,
    )

    grid: CubeGrid = make_uniform_grid(
        molecule,
        spacing=spacing,
        padding=padding,
    )
    # Build density matrix.
    _qvf_a, _qvf_b = spin_densities(result)
    if _qvf_a is not None:
        D = np.asarray(_qvf_a) + np.asarray(_qvf_b)
    else:
        D = np.asarray(result.density)
    try:
        from ...properties import _real_if_hermitian
    except ImportError:
        from vibeqc.properties import _real_if_hermitian  # type: ignore[no-redef]
    D = _real_if_hermitian(D, what="QVF density matrix")

    rho = _density_on_grid(D, basis, grid)
    origin = np.asarray(grid.origin, dtype=np.float64)
    # Per-voxel step vectors (matches 'voxel_vectors' in the QVF schema).
    span = np.diag(np.asarray(grid.spacing, dtype=np.float64))
    return {label: (rho, origin, span)}


def qvf_mo_data(
    result: Any,
    basis: Any,
    molecule: Any,
    indices: list[int],
    *,
    spacing: float = 0.25,
    padding: float = 4.0,
    component: str = "real",
) -> list[dict[str, Any]]:
    """Evaluate MO wavefunctions on a uniform grid and return a list
    suitable for ``write_qvf(..., mo_data=...)``.

    Parameters
    ----------
    result
        Converged SCF result with ``.mo_coefficients`` (RHF/RKS) or
        ``.mo_coefficients_alpha`` + ``.mo_coefficients_beta`` (UHF/UKS).
    basis
        :class:`BasisSet` used in the calculation.
    molecule
        :class:`Molecule` defining the atomic positions.
    indices
        0-based MO indices to evaluate. Must be a list of plain
        ``int`` values; the tuple form returned by
        :func:`vibeqc.output.formats.cube.requested_mo_indices`
        (``[(index, name), ...]``) must be unpacked at the call site.
    spacing
        Voxel spacing in bohr (default 0.25).
    padding
        Extra headroom in bohr.
    component
        ``"real"`` (default), ``"imag"``, ``"abs"``, or ``"density"``.

    Returns
    -------
    list[dict]
        Each dict has keys ``label``, ``data``, ``origin``, ``span``,
        ``band_index``, ``energy_eh``, ``occupation``, ``spin``,
        ``component``.  Pass as ``mo_data=`` to :func:`write_qvf`.
    """
    from vibeqc.cube import CubeGrid, _mo_on_grid, make_uniform_grid

    grid: CubeGrid = make_uniform_grid(
        molecule,
        spacing=spacing,
        padding=padding,
    )
    origin = np.asarray(grid.origin, dtype=np.float64)
    # Per-voxel step vectors (matches 'voxel_vectors' in the QVF schema).
    span = np.diag(np.asarray(grid.spacing, dtype=np.float64))

    # MO coefficients.
    unrestricted_alpha = False
    if hasattr(result, "mo_coeffs"):
        C = np.asarray(result.mo_coeffs)
        spin = 0
    elif hasattr(result, "mo_coefficients"):
        C = np.asarray(result.mo_coefficients)
        spin = 0
    elif hasattr(result, "mo_coeffs_alpha"):
        C = np.asarray(result.mo_coeffs_alpha)
        spin = 0
        unrestricted_alpha = True
    else:
        raise ValueError(
            "qvf_mo_data: result has no mo_coeffs or mo_coefficients attribute"
        )

    # MO energies.
    if hasattr(result, "mo_energies"):
        energies = np.asarray(result.mo_energies)
    elif hasattr(result, "mo_energies_alpha"):
        energies = np.asarray(result.mo_energies_alpha)
    else:
        energies = np.zeros(C.shape[1])

    # Occupations.
    n_occ = getattr(
        result,
        "n_occ_a" if unrestricted_alpha else "n_occ",
        None,
    )
    if n_occ is None:
        n_elec = effective_electron_count(molecule, result)
        if unrestricted_alpha:
            multiplicity = int(getattr(molecule, "multiplicity", 1))
            spin_excess = multiplicity - 1
            if (
                multiplicity < 1
                or n_elec < spin_excess
                or (n_elec - spin_excess) % 2
            ):
                raise ValueError(
                    "qvf_mo_data: multiplicity is incompatible with the "
                    "effective electron count"
                )
            n_occ = (n_elec + spin_excess) // 2
        else:
            n_occ = int(n_elec // 2)
    occupation_value = 1.0 if unrestricted_alpha else 2.0

    out: list[dict[str, Any]] = []
    for idx in indices:
        C_col = C[:, idx]
        mo = _mo_on_grid(C_col, basis, grid)
        if component == "abs":
            mo = np.abs(mo)
        elif component == "density":
            mo = mo**2
        occ = occupation_value if idx < n_occ else 0.0
        out.append(
            {
                "label": f"MO_{idx}",
                "data": mo,
                "origin": origin,
                "span": span,
                "band_index": idx,
                "energy_eh": float(energies[idx]),
                "occupation": occ,
                "spin": spin,
                "component": component,
            }
        )
    return out


def qvf_ao_data(
    basis: Any,
    molecule: Any,
    *,
    grid: Any = None,
    spacing: float = 0.15,
    padding: float = 8.0,
    ao_indices: list[int] | None = None,
    primitive_indices: list[tuple[int, int]] | None = None,
    include_contracted: bool = True,
    include_primitives: bool = True,
    basis_label: str | None = None,
) -> list[dict[str, Any]]:
    """Build per-AO grid data for :func:`write_qvf` ``ao_data=``.

    Evaluates selected atomic orbitals (contracted shells and/or
    individual primitives) on a uniform grid and returns a list of
    dicts ready for the ``basis.ao`` section writer.

    Parameters
    ----------
    basis
        :class:`BasisSet` carrying AO shells.
    molecule
        :class:`Molecule` defining atom centers.
    grid
        Pre-built :class:`CubeGrid`.  When ``None``, one is built via
        :func:`make_uniform_grid` with the given ``spacing`` and
        ``padding``.
    spacing
        Voxel spacing in bohr (default 0.15 -- finer than the 0.25
        density default to resolve compact primitives).
    padding
        Extra headroom around the bounding box in bohr (default 8.0).
    ao_indices
        List of global AO indices for contracted shells to evaluate.
        When ``None`` and ``primitive_indices`` is also ``None``,
        evaluates ALL contracted AOs.
    primitive_indices
        List of ``(shell_idx, prim_idx)`` tuples for individual
        primitives to evaluate.  Each builds a temp single-primitive
        basis and evaluates it.
    include_contracted
        When ``True`` and ``ao_indices`` is ``None``, include every
        contracted AO.  Ignored when ``ao_indices`` is explicit.
    include_primitives
        When ``True`` and ``primitive_indices`` is ``None``, include
        every primitive in every shell.  Ignored when
        ``primitive_indices`` is explicit.
    basis_label
        Human-readable basis-set name (e.g. ``"pob-TZVP"``), written
        into each section's ``ao_metadata.basis_label``.

    Returns
    -------
    list[dict]
        One dict per AO.  Each dict has keys: ``label``, ``data``,
        ``origin``, ``span``, ``ao_metadata``, and ``viewer_hints``.
        Pass the list as ``ao_data=`` to :func:`write_qvf`.
    """
    from vibeqc.cube import (
        CubeGrid,
        _ao_ranges,
        evaluate_ao_on_grid,
        evaluate_single_primitive_on_grid,
        make_uniform_grid,
    )

    # --- grid -----------------------------------------------------------
    if grid is None:
        grid = make_uniform_grid(molecule, spacing=spacing, padding=padding)
    origin = np.asarray(grid.origin, dtype=np.float64)
    span = np.diag(np.asarray(grid.spacing, dtype=np.float64))

    # --- shell metadata --------------------------------------------------
    shells = list(basis.shells())
    ao_ranges = _ao_ranges(basis)
    name = getattr(basis, "name", None)
    blabel = basis_label or (str(name) if name and name != "<custom>" else None)

    # Atom symbol lookup.
    atoms = list(molecule.atoms)
    _Z_to_sym: dict[int, str] = {}
    for ai, a in enumerate(atoms):
        _Z_to_sym[ai] = _symbol(int(a.Z))

    results: list[dict[str, Any]] = []

    # --- contracted AOs --------------------------------------------------
    if include_contracted or ao_indices is not None:
        idxs: list[int]
        if ao_indices is not None:
            idxs = sorted(set(ao_indices))
        else:
            idxs = list(range(int(basis.nbasis)))
        grids = evaluate_ao_on_grid(basis, grid, idxs)
        for ao_idx, data_3d in zip(idxs, grids):
            # Find which shell this AO belongs to.
            shell_idx = -1
            for si, (lo, hi) in enumerate(ao_ranges):
                if lo <= ao_idx < hi:
                    shell_idx = si
                    break
            s = shells[shell_idx]
            ao_local = ao_idx - ao_ranges[shell_idx][0]
            l = int(s.l)
            atom_idx = int(s.atom_index)
            sym = _Z_to_sym.get(atom_idx, "?")

            component = _ao_component_metadata(l, bool(s.pure), ao_local)
            tag = _ao_component_tag(component)
            label = _ao_label(
                sym, shell_idx, l, component_tag=tag, prim_idx=None, contracted=True
            )
            section_id = _ao_section_id(
                sym, shell_idx, l, component_tag=tag, prim_idx=None, contracted=True
            )

            meta: dict[str, Any] = {
                "atom_index": atom_idx,
                "atom_symbol": sym,
                "shell_index": shell_idx,
                "primitive_index": 0,
                **component,
                "shell_type": _l_to_shell_type(l),
                "exponent": float(s.exponents[0]),
                "coefficient": float(s.coefficients[0]),
                "is_primitive": False,
                "is_contracted": True,
                "ao_index": ao_idx,
            }
            if blabel:
                meta["basis_label"] = blabel

            results.append(
                {
                    "label": label,
                    "data": data_3d,
                    "origin": origin,
                    "span": span,
                    "ao_metadata": meta,
                    "section_id": section_id,
                }
            )

    # --- individual primitives -------------------------------------------
    if include_primitives or primitive_indices is not None:
        pairs: list[tuple[int, int]]
        if primitive_indices is not None:
            pairs = sorted(set(primitive_indices))
        else:
            pairs = [
                (si, pi)
                for si, s in enumerate(shells)
                for pi in range(len(s.exponents))
            ]
        for shell_idx, prim_idx in pairs:
            s = shells[shell_idx]
            l = int(s.l)
            atom_idx = int(s.atom_index)
            sym = _Z_to_sym.get(atom_idx, "?")

            label = _ao_label(sym, shell_idx, l, prim_idx=prim_idx, contracted=False)
            section_id = _ao_section_id(
                sym, shell_idx, l, prim_idx=prim_idx, contracted=False
            )

            data_3d = evaluate_single_primitive_on_grid(
                basis,
                molecule,
                shell_idx,
                prim_idx,
                grid,
            )

            # evaluate_single_primitive_on_grid returns column 0 of the
            # single-primitive shell, i.e. the shell's *first* component --
            # the same AO that `ao_index` below points at.  The component
            # metadata has to name that one, not a nominal m=0.
            meta = {
                "atom_index": atom_idx,
                "atom_symbol": sym,
                "shell_index": shell_idx,
                "primitive_index": prim_idx,
                **_ao_component_metadata(l, bool(s.pure), 0),
                "shell_type": _l_to_shell_type(l),
                "exponent": float(s.exponents[prim_idx]),
                "coefficient": float(s.coefficients[prim_idx]),
                "is_primitive": True,
                "is_contracted": False,
                "ao_index": ao_ranges[shell_idx][0],
            }
            if blabel:
                meta["basis_label"] = blabel

            results.append(
                {
                    "label": label,
                    "data": data_3d,
                    "origin": origin,
                    "span": span,
                    "ao_metadata": meta,
                    "section_id": section_id,
                }
            )

    return results


# -- AO label / id helpers -------------------------------------------------

_L_LABELS: dict[int, str] = {0: "s", 1: "p", 2: "d", 3: "f", 4: "g", 5: "h"}


def _l_to_shell_type(l: int) -> str:
    return _L_LABELS.get(l, f"l{l}")


def _ao_component_metadata(l: int, pure: bool, component: int) -> dict[str, Any]:
    """Name in-shell component ``component`` for a ``basis.ao`` ``ao_metadata``.

    A **pure** shell has ``2l+1`` components -- the real spherical
    harmonics ordered ``m = -l ... +l`` -- so ``[l, m]`` names one exactly.

    A **Cartesian** shell (``ShellInfo.pure == False``) has
    ``(l+1)(l+2)/2`` monomial components and no magnetic quantum number at
    all: ``component - l`` would run past ``+l`` and denote nothing (a
    Cartesian d shell's six components would come out as ``m = -2 ... +3``).
    Such a component is named by ``cartesian_powers = [lx, ly, lz]``
    instead, per QVF spec Appendix A.3; ``angular_momentum[1]`` is then a
    placeholder ``0`` that consumers must ignore.
    """
    from vibeqc.cube import _cartesian_component_powers

    if pure:
        return {"angular_momentum": [l, component - l]}
    lx, ly, lz = _cartesian_component_powers(l)[component]
    return {"angular_momentum": [l, 0], "cartesian_powers": [lx, ly, lz]}


def _ao_component_tag(component_meta: dict[str, Any]) -> str | None:
    """Short tag naming one in-shell component -- ``None`` for an s shell.

    Derived from the very dict :func:`_ao_component_metadata` returned, so
    an AO's label and its metadata cannot disagree about which component it
    is.  ``m-1`` for a spherical component, the monomial (``xy``) for a
    Cartesian one.

    Every AO of a shell needs this in its label and section id.  Without it
    all ``2l+1`` (or ``(l+1)(l+2)/2``) components of one shell collapse onto
    a single section id and a single path inside the archive, which
    :func:`write_qvf`'s own validation gate rejects as a duplicate id.  An s
    shell has exactly one component, so it gets no tag and its ids stay
    byte-identical to what earlier versions wrote.  Section ids match
    ``^[A-Za-z0-9_.-]+$``, which admits ``-`` but not ``+`` -- hence ``m-1``
    but ``m1``, not ``m+1``.
    """
    powers = component_meta.get("cartesian_powers")
    if powers is not None:
        return "".join(axis * n for axis, n in zip("xyz", powers))
    l, m = component_meta["angular_momentum"]
    return None if l == 0 else f"m{m}"


def _ao_label(
    sym: str,
    shell_idx: int,
    l: int,
    *,
    component_tag: str | None = None,
    prim_idx: int | None = None,
    contracted: bool = False,
) -> str:
    """Human-readable AO label, e.g. 'O d xy (contracted)  sh=2'."""
    lt = _L_LABELS.get(l, f"l={l}")
    if component_tag:
        lt = f"{lt} {component_tag}"
    if contracted:
        return f"{sym} {lt} (contracted)  sh={shell_idx}"
    return f"{sym} {lt}  sh={shell_idx}  p{prim_idx}"


def _ao_section_id(
    sym: str,
    shell_idx: int,
    l: int,
    *,
    component_tag: str | None = None,
    prim_idx: int | None = None,
    contracted: bool = False,
) -> str:
    """Stable section id for an AO, e.g. 'ao_O_p_s2_p0'."""
    lt = _L_LABELS.get(l, f"l{l}")
    if component_tag:
        lt = f"{lt}_{component_tag}"
    if contracted:
        return f"ao_{sym}_{lt}_s{shell_idx}_contracted"
    return f"ao_{sym}_{lt}_s{shell_idx}_p{prim_idx}"


def _ao_isovalue_default(exponent: float) -> float:
    """Heuristic isovalue based on Gaussian exponent a.

    Diffuse functions (a < 0.1) have low peak amplitude -- a high
    isovalue shows nothing.  Tight functions (a >= 10) have high peak
    amplitude at the nucleus.
    """
    if exponent < 0.1:
        return 0.005
    if exponent < 1.0:
        return 0.02
    if exponent < 10.0:
        return 0.05
    return 0.10


# _primitive_norm (imported above from vibeqc._primitive_norm) is QVF spec
# Appendix A.1's ``N_i`` exactly: the writer divides libint's stored
# coefficients by it on the way out, so the round trip is faithful for both
# shell kinds, and consumers must not add a per-component correction on top.


def _basis_shell_payload(basis: Any) -> tuple[list[dict[str, Any]], bool, int] | None:
    """Return QVF shell JSON, top-level purity, and AO count for ``basis``.

    Both shell kinds are emitted. Coefficients are divided by
    :func:`_primitive_norm`, which is QVF spec Appendix A.1's ``N_i`` --
    one factor per shell, from the total ``l`` -- so a consumer following
    A.1 reconstructs exactly the basis functions libint's integral engine
    used, which is the basis the MO coefficients are expressed in.

    Cartesian shells were rejected here between 2026-08-05 and the same
    day's follow-up, because the two available readers
    (``vibeqc.evaluate_ao`` and vibe-view) each applied an extra
    per-component factor √((2l-1)!!/((2lx-1)!!(2ly-1)!!(2lz-1)!!)) on top
    of ``N_i``, which made an emitted d_xy evaluate 3x too large. Both
    readers now follow A.1, so the payload round-trips and the rejection
    would only be blocking a correct path. See
    ``tests/test_ao_convention_invariants.py`` for the round-trip proof.

    Cartesian shells reach this writer through the explicit-``ShellInfo``
    ``BasisSet`` constructor — which includes
    ``vibeqc.basis_toolkit.to_libint_basis``, a public API that honors an
    imported basis's ``harmonic_type`` (shipped since v0.15.0). Only the
    named-basis route forces ``set_pure(true)`` (cpp/src/basis.cpp), so
    archives from named bases are spherical, but a Cartesian payload is a
    real production path, not a hypothetical (see
    handovers/HANDOVER_AO_CONVENTION.md).
    """
    try:
        shells_native = list(basis.shells())
    except AttributeError:
        return None

    # Per-shell `pure` is emitted (and echoed at the top level) because the
    # QVF schema requires the flag and a reader must not have to infer it.
    pure_top = all(bool(sh.pure) for sh in shells_native) if shells_native else True

    shell_list: list[dict[str, Any]] = []
    n_ao = 0
    for sh in shells_native:
        l = int(sh.l)
        shell_pure = bool(sh.pure)
        shell_list.append(
            {
                "center": int(sh.atom_index),
                "l": l,
                "exponents": [float(x) for x in sh.exponents],
                "coefficients": [
                    float(c) / _primitive_norm(float(a), l)
                    for a, c in zip(sh.exponents, sh.coefficients)
                ],
                "pure": shell_pure,
            }
        )
        n_ao += (2 * l + 1) if shell_pure else ((l + 1) * (l + 2) // 2)
    return shell_list, pure_top, n_ao


def qvf_natural_wf_data(
    orbitals: Any,
    occupations: Any,
    basis: Any,
    molecule: Any,
    *,
    structure_ref: str = "structure",
) -> dict[str, Any] | None:
    """Package explicit natural orbitals + occupations for the QVF archive.

    The sibling :func:`qvf_wf_data` derives occupations from the electron
    count, filling 2.0 for the first ``n_elec/2`` orbitals and 0.0 after.
    That is right for a single-determinant reference and wrong for anything
    correlated: a natural orbital's occupancy is an eigenvalue of the
    density matrix, fractional across the active space, and it is exactly
    that departure from {0, 2} that carries the multireference character. So
    this takes the occupations as given rather than reconstructing them.

    ``orbitals`` is AO-by-MO with columns as orbitals (vibe-qc's
    ``mo_coeffs`` convention); the QVF stores rows as orbitals, so it is
    transposed here. ``occupations`` must line up column for column.

    Returns None when the basis payload cannot be built or the shapes
    disagree, so a caller can degrade to omitting the section rather than
    writing an archive whose orbitals and occupations describe different
    things.
    """
    basis_payload = _basis_shell_payload(basis)
    if basis_payload is None:
        return None
    shell_list, pure_top, n_ao = basis_payload

    C = np.ascontiguousarray(np.asarray(orbitals, dtype=np.float64).T)
    occ = np.asarray(occupations, dtype=np.float64).ravel()
    if C.ndim != 2 or C.shape[0] != occ.size or C.shape[1] != n_ao:
        return None
    n_mo = int(C.shape[0])

    return {
        "basis": shell_list,
        "structure_ref": structure_ref,
        "pure": pure_top,
        "n_ao": n_ao,
        "mo_metadata": {
            "n_mo": n_mo,
            "n_ao": n_ao,
            "spin": "restricted",
            "orbital_kind": "natural",
            "occupation_semantics": "electron_occupation",
            # A natural orbital has no orbital energy: its eigenvalue *is*
            # the occupation number. Zeros rather than a fabricated
            # spectrum, matching what the NTO emitter writes.
            "energies": [0.0] * n_mo,
            "occupations": occ.tolist(),
        },
        "mo_coefficients": C,
    }


def qvf_localized_wf_data(
    orbitals: Any,
    basis: Any,
    *,
    structure_ref: str = "structure",
    method: str = "ibo",
    charges: Any = None,
    atom_populations: Any = None,
    centroids: Any = None,
    n_centres: Any = None,
    reference_basis: str | None = None,
) -> dict[str, Any] | None:
    """Package localized occupied orbitals for the ``wf_localized`` section.

    Shaped like :func:`qvf_natural_wf_data`, and emitted by
    :func:`write_qvf` under the ``wf_localized_data`` context key as a second
    ``wavefunction.gto`` section with ``orbital_kind="localized"``.

    A localized orbital has **no orbital energy** -- it is a unitary mixture of
    canonical orbitals spanning a range of eigenvalues -- so ``energies`` is
    written as explicit zeros. It must not be omitted or written as null:
    vibe-view reads it with ``np.asarray(meta.get("energies"), dtype=float64)``,
    and ``np.asarray(None, dtype=np.float64)`` yields a 0-d ``array(nan)`` that
    passes the reader's ``is not None`` guard and crashes downstream.

    All orbitals in this section are occupied, so occupations are 2.0
    throughout (closed shell). The chemistry the viewer actually needs -- which
    atoms each orbital sits on, and where it sits in space -- rides in
    ``mo_metadata`` as ``atom_populations`` / ``centroids`` / ``n_centres``.

    Returns None when the basis payload cannot be built or shapes disagree,
    so the caller can omit the section rather than write an inconsistent one.
    """
    basis_payload = _basis_shell_payload(basis)
    if basis_payload is None:
        return None
    shell_list, pure_top, n_ao = basis_payload

    C = np.ascontiguousarray(np.asarray(orbitals, dtype=np.float64).T)
    if C.ndim != 2 or C.shape[1] != n_ao:
        return None
    n_mo = int(C.shape[0])

    metadata: dict[str, Any] = {
        "n_mo": n_mo,
        "n_ao": n_ao,
        "spin": "restricted",
        "orbital_kind": "localized",
        "occupation_semantics": "electron_occupation",
        "energies": [0.0] * n_mo,
        "occupations": [2.0] * n_mo,
        "localization_method": method,
    }
    if reference_basis is not None:
        metadata["localization_reference_basis"] = reference_basis
    if charges is not None:
        metadata["iao_charges"] = np.asarray(charges, dtype=np.float64).tolist()
    if atom_populations is not None:
        populations = np.asarray(atom_populations, dtype=np.float64)
        if populations.shape[0] != n_mo:
            return None
        metadata["atom_populations"] = populations.tolist()
    if centroids is not None:
        centre = np.asarray(centroids, dtype=np.float64)
        if centre.shape[0] != n_mo:
            return None
        metadata["centroids_bohr"] = centre.tolist()
    if n_centres is not None:
        counts = np.asarray(n_centres, dtype=np.int64)
        if counts.shape[0] != n_mo:
            return None
        metadata["n_centres"] = counts.tolist()

    return {
        "basis": shell_list,
        "structure_ref": structure_ref,
        "pure": pure_top,
        "n_ao": n_ao,
        "mo_metadata": metadata,
        "mo_coefficients": C,
    }


def qvf_wf_data(
    result: Any,
    basis: Any,
    molecule: Any,
    *,
    structure_ref: str = "structure",
    orbital_kind: str = "canonical",
    k_point: "list[float] | None" = None,
) -> dict[str, Any] | None:
    """Package basis shells + MO coefficients + per-orbital metadata into
    a dict suitable for ``write_qvf(..., wf_data=...)``.

    Lets a re-sampling viewer (vibe-view, moltui) evaluate any orbital
    on a grid of its own choosing -- the Molden-style separation of
    concerns described in design Sec. 1.5.

    Parameters
    ----------
    result
        Converged SCF result. RHF/RKS: ``.mo_coeffs`` + ``.mo_energies``.
        UHF/UKS: ``.mo_coeffs_alpha`` + ``_beta`` + matching energies.
    basis
        :class:`BasisSet` carrying the AO shells used in the calculation.
    molecule
        :class:`Molecule` defining atom centers (referenced by ``center``
        indices in the basis shells).
    structure_ref
        ``id`` of the structure section the shell centers refer to
        (default ``"structure"`` -- matches what
        :func:`_write_structure_section` emits).
    orbital_kind
        ``"canonical"`` | ``"natural"`` | ``"localized"`` -- written
        verbatim into the mo_metadata.
    k_point
        Optional fractional reciprocal-space coordinate the MO coefficients
        describe. Pass ``[0.0, 0.0, 0.0]`` for the Γ-point of a periodic
        system; leave ``None`` for molecular wavefunctions. Recorded in
        ``mo_metadata["k_point"]`` so the archive is self-describing.
        Non-Gamma Bloch coefficients are stored as real/imag component pairs.

    Returns
    -------
    dict | None
        A wf_data dict, or ``None`` if neither restricted nor
        unrestricted MO coefficients can be found on ``result`` (e.g.
        a DFTB result that exposes a different attribute layout).
        The shape matches what :func:`_write_wavefunction_gto_section`
        consumes.
    """
    # --- shells -------------------------------------------------------
    basis_payload = _basis_shell_payload(basis)
    if basis_payload is None:
        return None
    shell_list, pure_top, n_ao = basis_payload

    # --- MO coefficients + metadata -----------------------------------
    # vibe-qc convention: mo_coeffs has shape [n_ao, n_mo] (columns are
    # MOs). The QVF format wants [n_mo, n_ao] (rows are MOs) so we
    # transpose on the way out.
    def _as_rowmo(arr: np.ndarray) -> tuple[np.ndarray, str]:
        row = np.asarray(arr).T
        coeff, encoding = _mo_coefficients_for_artifact(
            row,
            label="wavefunction.gto MO coefficients",
        )
        return coeff, encoding

    def _occupations_or_default(
        attr: str,
        n_mo: int,
        default: list[float],
    ) -> list[float]:
        value = getattr(result, attr, None)
        if value is None:
            return default
        occupations = np.asarray(value, dtype=np.float64).reshape(-1)
        # Several legacy SCF result dataclasses expose an empty array to mean
        # that explicit occupations were not recorded (not that the orbital
        # set has zero occupation).  Retain the historical Aufbau fallback
        # for that sentinel while preserving every non-empty occupation
        # vector, including fractional occupations.  A non-empty vector with
        # the wrong size remains an error below.
        if occupations.size == 0:
            return default
        if occupations.size != n_mo:
            raise ValueError(
                f"wavefunction.gto {attr} has {occupations.size} entries; "
                f"expected {n_mo}"
            )
        return occupations.tolist()

    # Restricted (RHF / RKS).
    if hasattr(result, "mo_coeffs") and not hasattr(result, "mo_coeffs_alpha"):
        C, encoding = _as_rowmo(result.mo_coeffs)
        n_mo = int(C.shape[0])
        energies = (
            np.asarray(result.mo_energies, dtype=np.float64).tolist()
            if hasattr(result, "mo_energies")
            else [0.0] * n_mo
        )
        # Restricted occupations: 2.0 for the first n_elec/2 MOs, where an
        # ECP result's authoritative replaced-core count defines n_elec.
        n_elec = effective_electron_count(molecule, result)
        n_doubly = int(n_elec // 2)
        occupations = _occupations_or_default(
            "occupations",
            n_mo,
            [2.0 if i < n_doubly else 0.0 for i in range(n_mo)],
        )
        mo_metadata: dict[str, Any] = {
            "n_mo": n_mo,
            "n_ao": n_ao,
            "spin": "restricted",
            "orbital_kind": orbital_kind,
            "occupation_semantics": "electron_occupation",
            "energies": energies,
            "occupations": occupations,
        }
        if k_point is not None:
            mo_metadata["k_point"] = [float(x) for x in k_point]
        if encoding != "real":
            mo_metadata["coefficient_encoding"] = encoding
            mo_metadata["coefficient_components"] = ["real", "imag"]
        return {
            "basis": shell_list,
            "structure_ref": structure_ref,
            "pure": pure_top,
            "n_ao": n_ao,
            "mo_metadata": mo_metadata,
            "mo_coefficients": C,
        }

    # Unrestricted (UHF / UKS).
    if hasattr(result, "mo_coeffs_alpha") and hasattr(result, "mo_coeffs_beta"):
        Ca, enc_alpha = _as_rowmo(result.mo_coeffs_alpha)
        Cb, enc_beta = _as_rowmo(result.mo_coeffs_beta)
        n_elec = effective_electron_count(molecule, result)
        mult = int(getattr(molecule, "multiplicity", 1))
        n_alpha = (n_elec + mult - 1) // 2
        n_beta = (n_elec - mult + 1) // 2
        e_alpha = (
            np.asarray(result.mo_energies_alpha, dtype=np.float64).tolist()
            if hasattr(result, "mo_energies_alpha")
            else [0.0] * int(Ca.shape[0])
        )
        e_beta = (
            np.asarray(result.mo_energies_beta, dtype=np.float64).tolist()
            if hasattr(result, "mo_energies_beta")
            else [0.0] * int(Cb.shape[0])
        )
        mo_metadata = {
            "n_ao": n_ao,
            "spin": "unrestricted",
            "orbital_kind": orbital_kind,
            "occupation_semantics": "electron_occupation",
            "alpha": {
                "n_mo": int(Ca.shape[0]),
                "energies": e_alpha,
                "occupations": _occupations_or_default(
                    "occupations_alpha",
                    int(Ca.shape[0]),
                    [
                        1.0 if i < n_alpha else 0.0
                        for i in range(int(Ca.shape[0]))
                    ],
                ),
            },
            "beta": {
                "n_mo": int(Cb.shape[0]),
                "energies": e_beta,
                "occupations": _occupations_or_default(
                    "occupations_beta",
                    int(Cb.shape[0]),
                    [
                        1.0 if i < n_beta else 0.0
                        for i in range(int(Cb.shape[0]))
                    ],
                ),
            },
        }
        if k_point is not None:
            mo_metadata["k_point"] = [float(x) for x in k_point]
        if enc_alpha != "real":
            mo_metadata["alpha"]["coefficient_encoding"] = enc_alpha
            mo_metadata["alpha"]["coefficient_components"] = ["real", "imag"]
        if enc_beta != "real":
            mo_metadata["beta"]["coefficient_encoding"] = enc_beta
            mo_metadata["beta"]["coefficient_components"] = ["real", "imag"]
        return {
            "basis": shell_list,
            "structure_ref": structure_ref,
            "pure": pure_top,
            "n_ao": n_ao,
            "mo_metadata": mo_metadata,
            "mo_coefficients_alpha": Ca,
            "mo_coefficients_beta": Cb,
        }

    return None


def _multik_sequence(obj: Any, names: Sequence[str]) -> list[Any] | None:
    for name in names:
        value = getattr(obj, name, None)
        if isinstance(value, (list, tuple)) and len(value) > 0:
            return list(value)
    return None


def _single_or_multik_coefficients(obj: Any) -> list[Any] | None:
    coeffs = _multik_sequence(obj, ("mo_coeffs_k", "mo_coeffs"))
    if coeffs is not None:
        return coeffs
    value = getattr(obj, "mo_coeffs", None)
    if value is not None and not hasattr(obj, "mo_coeffs_alpha"):
        return [value]
    return None


def _single_or_multik_energies(obj: Any) -> list[Any] | None:
    energies = _multik_sequence(obj, ("mo_energies_k", "mo_energies"))
    if energies is not None:
        return energies
    value = getattr(obj, "mo_energies", None)
    if value is not None:
        return [value]
    return None


def _single_or_multik_occupations(
    obj: Any,
    coeffs: Sequence[Any],
    molecule: Any | None,
) -> list[Any] | None:
    occs = _multik_sequence(obj, ("occupations_k", "occupations"))
    if occs is not None:
        # Γ-only periodic drivers may wrap the legacy empty-array sentinel
        # in their one-element per-k list.  An entirely empty sequence means
        # occupations were unavailable; a partially empty sequence remains
        # malformed and is deliberately returned for strict validation.
        if any(np.asarray(occ).size > 0 for occ in occs):
            return occs
    value = getattr(obj, "occupations", None)
    if value is not None:
        # A zero-length array/list is the legacy "not recorded" sentinel,
        # matching ``qvf_wf_data`` above.  Let the single-k compatibility
        # fallback derive Aufbau occupations; non-empty data continues into
        # the strict block-count/shape checks in ``qvf_bloch_wf_data``.
        if np.asarray(value).size > 0:
            return [value]
    if len(coeffs) == 1 and molecule is not None and hasattr(molecule, "n_electrons"):
        C = np.asarray(coeffs[0])
        if C.ndim != 2:
            return None
        n_mo = int(C.shape[1])
        n_doubly = effective_electron_count(molecule, obj) // 2
        occ = np.zeros(n_mo, dtype=np.float64)
        occ[: min(n_doubly, n_mo)] = 2.0
        return [occ]
    return None


def _qvf_spin_bloch_density_data(result, basis_payload, k_points, k_weights, structure_ref):
    """Lossless SCF spin restart; later canonical orbitals are not the state."""
    from ...guess_read import _periodic_spin_density_blocks

    shells, pure, n_ao = basis_payload
    pair = _periodic_spin_density_blocks(result, n_basis=n_ao)
    if pair is None:
        return None
    points = np.asarray(k_points, dtype=float)
    count = len(pair[0])
    if points.shape != (count, 3) or not np.all(np.isfinite(points)):
        raise ValueError("QVF spin restart requires one finite k point per density block")
    if k_weights is None:
        k_weights = getattr(result, "kpoint_weights", None)
    if k_weights is None:
        k_weights = getattr(result, "restart_weights", None)
    if k_weights is None:
        k_weights = getattr(getattr(result, "kmesh", None), "weights", None)
    if k_weights is None and count == 1:
        k_weights = [1.]
    weights = np.asarray(k_weights, dtype=float)
    if (weights.shape != (count,) or not np.all(np.isfinite(weights))
        or np.any(weights < 0) or not np.isclose(weights.sum(), 1, rtol=0, atol=1e-12)):
        raise ValueError("QVF spin restart requires complete normalized k-point weights")
    blocks = [{"k_index": i, "k_point": point.tolist(), "k_weight": float(weight),
               "density_alpha": f"density_alpha_k{i}", "density_beta": f"density_beta_k{i}"}
              for i, (point, weight) in enumerate(zip(points, weights))]
    return {
        "basis": shells, "pure": pure, "n_ao": n_ao, "structure_ref": structure_ref,
        "kpoints": np.ascontiguousarray(points),
        "density_alpha": [_complex_split_array(d) for d in pair[0]],
        "density_beta": [_complex_split_array(d) for d in pair[1]],
        "mo_metadata": {
            "schema_version": "1.1", "restart_kind": "spin_density_bloch_kpoints",
            "spin": "unrestricted", "n_kpoints": count, "n_ao": n_ao,
            "k_point_units": "fractional_reciprocal",
            "density_encoding": "complex_split_last_axis",
            "density_layout": "ao_row_ao_column", "density_components": ["real", "imag"],
            "blocks": blocks,
        },
    }


def qvf_bloch_wf_data(
    result: Any,
    basis: Any,
    molecule: Any | None = None,
    *,
    k_points: Sequence[Sequence[float]],
    k_weights: Sequence[float] | None = None,
    structure_ref: str = "structure",
    orbital_kind: str = "canonical",
) -> dict[str, Any] | None:
    """Package all-k Bloch states for QVF READ restarts.

    Spin-resolved sources store both physical AO density matrices (schema
    1.1), retaining the SCF state independently of later orbital rotations.
    Restricted sources retain the schema 1.0 coefficient/occupation payload.

    This is a vibe-qc vendor extension, not the visualization-oriented
    ``wavefunction.gto`` contract.  Coefficients are stored per k-point as
    MO-major arrays ``[n_mo, n_ao, 2]`` with the last axis equal to
    ``[real, imag]``.  Per-k members are deliberately ragged so restart
    archives remain valid when canonical orthogonalisation prunes a different
    number of near-null directions at different k-points.
    """
    basis_payload = _basis_shell_payload(basis)
    if basis_payload is None:
        return None
    shell_list, pure_top, n_ao = basis_payload
    spin_payload = _qvf_spin_bloch_density_data(
        result, basis_payload, k_points, k_weights, structure_ref)
    if spin_payload is not None:
        return spin_payload

    coeffs = _single_or_multik_coefficients(result)
    occs = (
        _single_or_multik_occupations(result, coeffs, molecule)
        if coeffs is not None
        else None
    )
    if coeffs is None or occs is None:
        return None
    if len(coeffs) != len(occs):
        raise ValueError(
            "qvf_bloch_wf_data: coefficient and occupation k-block counts "
            f"differ ({len(coeffs)} vs {len(occs)})."
        )

    kpts = np.asarray(k_points, dtype=np.float64).reshape(-1, 3)
    if kpts.shape[0] != len(coeffs):
        raise ValueError(
            "qvf_bloch_wf_data: k-point metadata count does not match "
            f"coefficient blocks ({kpts.shape[0]} vs {len(coeffs)})."
        )

    if k_weights is None:
        k_weights = getattr(result, "kpoint_weights", None)
    if k_weights is None and getattr(result, "kmesh", None) is not None:
        k_weights = result.kmesh.weights
    if k_weights is None and len(coeffs) == 1:
        k_weights = [1.0]
    if k_weights is not None:
        k_weights = np.asarray(k_weights, dtype=float)
        if (k_weights.shape != (len(coeffs),) or not np.all(np.isfinite(k_weights))
            or np.any(k_weights < 0) or not np.isclose(k_weights.sum(), 1, rtol=0, atol=1e-12)):
            raise ValueError("qvf_bloch_wf_data: invalid k-point integration weights")

    energies = _single_or_multik_energies(result)
    if energies is not None and len(energies) != len(coeffs):
        raise ValueError(
            "qvf_bloch_wf_data: energy and coefficient k-block counts "
            f"differ ({len(energies)} vs {len(coeffs)})."
        )

    coeff_blocks: list[np.ndarray] = []
    occ_blocks: list[np.ndarray] = []
    energy_blocks: list[np.ndarray] = []
    block_meta: list[dict[str, Any]] = []
    for ik, (C_raw, occ_raw) in enumerate(zip(coeffs, occs)):
        C = np.asarray(C_raw, dtype=np.complex128)
        occ = np.asarray(occ_raw, dtype=np.float64)
        if C.ndim != 2:
            raise ValueError(
                f"qvf_bloch_wf_data: coefficient block {ik} must be 2-D; "
                f"got shape {C.shape}."
            )
        if C.shape[0] != n_ao:
            raise ValueError(
                f"qvf_bloch_wf_data: coefficient block {ik} has {C.shape[0]} "
                f"AO rows but the basis has {n_ao} functions."
            )
        if occ.ndim != 1:
            raise ValueError(
                f"qvf_bloch_wf_data: occupation block {ik} must be 1-D; "
                f"got shape {occ.shape}."
            )
        if C.shape[1] != occ.shape[0]:
            raise ValueError(
                f"qvf_bloch_wf_data: block {ik} has {C.shape[1]} orbitals "
                f"but {occ.shape[0]} occupations."
            )
        if energies is None:
            eps = np.zeros(C.shape[1], dtype=np.float64)
        else:
            eps = np.asarray(energies[ik], dtype=np.float64)
        if eps.ndim != 1 or eps.shape[0] != C.shape[1]:
            raise ValueError(
                f"qvf_bloch_wf_data: energy block {ik} must have shape "
                f"({C.shape[1]},); got {eps.shape}."
            )

        coeff_name = f"mo_coefficients_k{ik}"
        occ_name = f"occupations_k{ik}"
        energy_name = f"mo_energies_k{ik}"
        coeff_blocks.append(_complex_split_array(C.T))
        occ_blocks.append(np.ascontiguousarray(occ))
        energy_blocks.append(np.ascontiguousarray(eps))
        block_meta.append(
            {
                "k_index": ik,
                "k_point": [float(x) for x in kpts[ik]],
                "n_mo": int(C.shape[1]),
                "n_ao": int(n_ao),
                "mo_coefficients": coeff_name,
                "occupations": occ_name,
                "mo_energies": energy_name,
            }
        )

    if k_weights is not None:
        for block, weight in zip(block_meta, k_weights):
            block["k_weight"] = float(weight)

    mo_metadata: dict[str, Any] = {
        "schema_version": "1.0",
        "restart_kind": "closed_shell_bloch_kpoints",
        "spin": "restricted",
        "orbital_kind": orbital_kind,
        "n_kpoints": len(coeff_blocks),
        "n_ao": int(n_ao),
        "k_point_units": "fractional_reciprocal",
        "coefficient_encoding": "complex_split_last_axis",
        "coefficient_components": ["real", "imag"],
        "blocks": block_meta,
    }
    return {
        "basis": shell_list,
        "structure_ref": structure_ref,
        "pure": pure_top,
        "n_ao": n_ao,
        "kpoints": np.ascontiguousarray(kpts),
        "mo_metadata": mo_metadata,
        "mo_coefficients": coeff_blocks,
        "occupations": occ_blocks,
        "mo_energies": energy_blocks,
    }


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _sha256_hex(data: bytes) -> str:
    """Return the sha256 hex digest of ``data``."""
    return hashlib.sha256(data).hexdigest()


def _slug(label: str, *, fallback: str = "section") -> str:
    """Reduce a user-supplied label to a zip-path-safe slug.

    Keeps ASCII ``[A-Za-z0-9._-]``, replaces every other character with
    ``_``, collapses runs of ``_``, and trims leading/trailing ``_./-``.
    Returns ``fallback`` if the result is empty after trimming. Used
    for the label-derived components of zip paths (volume / orbital /
    spin / elf / difference) so that a label like
    ``"r(product) - r(reactant)"`` or a path-traversal attempt like
    ``"../etc/passwd"`` cannot reach the zip writer as-is.
    """
    if not label:
        return fallback
    _SAFE = set("abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789._-")
    out_chars: list[str] = []
    prev_us = False
    for ch in label:
        if ch in _SAFE:
            out_chars.append(ch)
            prev_us = False
        elif not prev_us:
            out_chars.append("_")
            prev_us = True
    slug = "".join(out_chars).strip("._-")
    return slug or fallback


def _require_3d_volume(data: np.ndarray, kind: str, label: str) -> None:
    """Raise ValueError if ``data`` is not a 3-D array.

    The volume writers previously skipped malformed sections silently
    (``if vol.ndim != 3: continue``), which let bad caller-supplied
    data ship as a QVF missing the section. We'd rather fail the write
    than produce an archive that quietly drops user data.
    """
    if data.ndim != 3:
        raise ValueError(
            f"{kind}[{label!r}]: data must be 3-D (got ndim={data.ndim}, "
            f"shape={tuple(data.shape)})"
        )


def _binary_array_entry(
    path_in_zip: str,
    data: np.ndarray,
) -> tuple[bytes, dict[str, Any]]:
    """Serialize ``data`` as raw little-endian bytes and build the
    matching manifest member-entry dict.

    Returns ``(raw_bytes, member_spec)``.  The caller writes
    ``raw_bytes`` into the zip and embeds ``member_spec`` in the
    appropriate section's ``members`` dict.

    The member spec has keys ``path``, ``format`` ("binary"),
    ``sha256``, and optionally ``dtype``, ``shape``.
    """
    if not data.flags["C_CONTIGUOUS"]:
        data = np.ascontiguousarray(data)
    raw = data.tobytes()
    dtype_name = np.dtype(data.dtype).name
    member = {
        "path": path_in_zip,
        "format": "binary",
        "dtype": dtype_name,
        "shape": list(data.shape),
        "sha256": _sha256_hex(raw),
    }
    return raw, member


def _write_binary_to_zip(
    zf: zipfile.ZipFile,
    path_in_zip: str,
    data: np.ndarray,
) -> dict[str, Any]:
    """Convenience: serialise ``data``, write it into ``zf``, and return
    the manifest member-spec dict."""
    raw, member = _binary_array_entry(path_in_zip, data)
    zf.writestr(path_in_zip, raw)
    return member


def _now_iso() -> str:
    return _dt.datetime.now().astimezone().isoformat(timespec="seconds")


def _is_vendor_kind(kind: str) -> bool:
    return kind.startswith("x_")


# ---------------------------------------------------------------------------
# Main writer
# ---------------------------------------------------------------------------


def write_qvf(
    stem: os.PathLike | str,
    plan: OutputPlan,
    *,
    compression: int = zipfile.ZIP_DEFLATED,
    volume_dtype: str = "float32",
    atomic: bool = False,
    **context: Any,
) -> Path:
    """Write ``{stem}.qvf``.

    Parameters
    ----------
    stem
        Path stem; ``.qvf`` suffix is appended.
    plan
        :class:`OutputPlan` declaring what artefacts are expected.
    compression
        ``zipfile.ZIP_DEFLATED`` (default), ``zipfile.ZIP_STORED``, or
        the zstd constant if ``zipfile-zstd`` is importable.
    volume_dtype
        ``"float32"`` (default) or ``"float64"`` for volumetric grids.
    atomic
        When ``True``, write the archive to a temporary sibling file and
        ``os.replace`` it onto ``{stem}.qvf`` only after write-time
        validation passes. A concurrent reader then sees either the
        previous complete archive or the new complete archive -- never a
        half-written zip. Used by the live-checkpoint path so vibe-view
        can hot-reload a running job's QVF safely. Default ``False``
        preserves the historical in-place write.
    **context
        Data objects the section writers need.  Typical keys:

        * ``run_status`` -- ``"pending"`` | ``"running"`` | ``"converged"``
          | ``"failed"``; emitted at ``provenance.run_status``. Lets a
          consumer tell a not-yet-run job container (``pending``) or a
          mid-run checkpoint (``running``) from the settled final archive.
        * ``checkpoint`` -- dict with ``seq`` (monotonic int),
          ``wall_time_s`` (float), ``written_at`` (ISO-8601 str), and
          optional ``scf_iteration`` / ``energy_eh`` running hints;
          emitted at ``provenance.checkpoint``.
        * ``partial_sections`` -- ``True`` to mark every section
          ``partial: true``, or an iterable of section ids / ``kind``
          strings (e.g. ``{"trajectory"}``) to flag only the still-growing
          ones in a checkpoint snapshot.
        * ``biomolecule_data`` -- optional dict adding biomolecule
          metadata to the ``structure`` section for ribbon/cartoon
          rendering: ``chains`` (list[str]), ``residues`` (list of
          ``{name, seq, chain, atom_indices}``, 0-based atom indices),
          ``secondary_structure`` (list of ``{type, chain, start_seq,
          end_seq}``) and ``b_factors`` (list[float], one per atom in
          payload order). Additive peer keys on the section object; all
          four independently optional. See
          :func:`_normalize_biomolecule`.

        * ``molecule`` / ``system`` -- :class:`Molecule` or
          :class:`PeriodicSystem`
        * ``structure_lattice_bohr`` -- optional 3x3 lattice override for
          the ``structure`` section, using the same bohr column-vector
          convention as ``PeriodicSystem.lattice``. Periodic CCM callers use
          this to emit the full BvK torus cell while keeping the primitive
          input ``system`` intact.
        * ``result`` -- converged SCF result object
        * ``basis`` -- :class:`BasisSet`
        * ``population_summary`` -- :class:`PopulationSummary`
        * ``hessian_result`` -- :class:`HessianResult`
        * ``band_structure`` -- :class:`BandStructure`
        * ``trajectory_frames`` -- list of :class:`Molecule`
        * ``trajectory_energies`` -- list of float (Hartree)
        * ``trajectory_rms_grad`` -- list of float (optional)
        * ``bibtex_content`` -- str, the full BibTeX file body
        * ``run_record`` -- dict describing the producing invocation,
          emitted as a ``run.record`` section (spec Sec. 5.8). Keys:
          ``program`` (required), ``input_text``/``log_text`` (at least
          one), ``input_filename``, ``log_filename``, ``log_truncated``,
          ``program_version``, ``command``, ``exit_status``,
          ``started_utc``, ``finished_utc``, ``sequence``. The runners
          assemble this after the ``.out`` channel closes, so the
          embedded log is the complete on-disk log at archive time.
        * ``job_spec`` -- declarative JobSpecPayload dict emitted as a
          ``job.spec`` section (spec Sec. 5.9): ``job_type`` (required,
          ``"molecular"`` | ``"periodic"``) plus optional ``method``,
          ``basis``, ``functional``, ``charge``, ``multiplicity``,
          ``kpoints``, ``tasks``, ``options``. Pair with
          ``run_status="pending"`` to describe a job that has not yet
          run; a runner executing the container preserves this section
          verbatim through the in-place update.
        * ``bond_orders_data`` -- dict with keys ``method`` (str, e.g.
          ``"mayer"``) and ``pairs`` (list of dicts each with ``i``, ``j``,
          ``order`` and optional ``distance_ang``, ``symbol_i``, ``symbol_j``).
          Emitted as a ``bond_orders`` section.
        * ``volume_data`` -- dict of ``{label: (data_3d, origin, span)}``
        * ``mo_data`` -- list of dicts with keys ``label``, ``data``,
          ``origin``, ``span``, ``band_index``, ``energy_eh``, ``occupation``,
          ``spin``, ``component``
        * ``spin_data`` -- dict of ``{label: (data_3d, origin, span)}``
        * ``elf_data`` -- dict of ``{label: (data_3d, origin, span)}``
        * ``generic_volume_data`` -- dict of ``{label: (data_3d, origin,
          span)}`` for ``volume.generic`` (escape hatch for any scalar
          field that doesn't fit density/orbital/spin/elf/difference)
        * ``potential_data`` -- dict of ``{label: (data_3d, origin, span)}``
          for ``volume.potential`` (electrostatic potential grid; same
          member structure as ``volume.density``, QVF spec Sec. 4.10)
        * ``rdg_data`` -- dict of ``{label: (data_3d, origin, span)}``
          for ``volume.rdg`` (reduced density gradient for NCI analysis;
          same member structure, QVF spec Sec. 4.11)
        * ``diff_data`` -- dict of ``{label: spec}`` for difference
          density (e.g. r(product) - r(reactant)). ``spec`` is either
          a 3-tuple ``(data_3d, origin, span)`` for an unannotated
          difference, or a dict with keys ``data``, ``origin``, ``span``,
          and optionally ``operand_a`` (str, section id of minuend),
          ``operand_b`` (str, section id of subtrahend), ``description``.
        * ``reaction_path`` -- dict ``{frames, waypoints, energies?,
          reaction_coordinate?}`` for a self-contained ``reaction.path``
          section. ``waypoints`` is a list of
          ``{frame_index, label, kind, energy_eh?}`` records where
          ``kind`` is one of ``"reactant" | "transition_state" |
          "intermediate" | "product" | "point"``.
        * ``reaction_waypoints`` -- dict ``{trajectory_ref, waypoints,
          reaction_coordinate?}`` for a lightweight
          ``reaction.waypoints`` annotation over an already-emitted
          ``trajectory`` section. ``trajectory_ref`` must name a
          trajectory section emitted in the same archive; the writer
          raises if it doesn't resolve.
        * ``viewer_defaults`` -- dict written verbatim to the manifest
          root. Recognised keys: ``auto_open`` (list of section ids),
          per-section render hints, and ``bookmarks`` (ordered list of
          ``{name, camera}`` records using the VTK camera model).
        * ``thermochemistry_data`` -- dict with keys ``zpve_eh``,
          ``enthalpy_eh``, ``entropy_cal_mol_k``, ``gibbs_free_energy_eh``,
          ``temperature_k``, ``pressure_atm`` for a root
          ``thermochemistry`` field (QVF spec Sec. 4.7).
        * ``dipole_moment_data`` -- dict with keys ``total_debye``,
          ``vector_debye`` (3-vector), and ``origin`` (a named reference
          point or an explicit 3-vector in bohr) for a root
          ``dipole_moment`` field (QVF spec Sec. 4.7).
        * ``spin_diagnostic_data`` -- dict with ``s_squared``,
          ``s_squared_ideal``, and ``s_squared_deviation``. Emitted in the
          open ``provenance`` block for unrestricted wavefunctions.
        * ``constraints_data`` -- dict with keys ``frozen_atoms`` (list
          of int), ``distance_constraints`` (list of ``{atoms, target_angstrom}``)
          for a root ``constraints`` field (QVF spec Sec. 4.7).
        * ``extensions`` -- dict of ``{vendor_ns: {version, schema_uri?,
          critical?}}`` for the root ``extensions`` governance block
          (QVF spec Sec. 5.4).
        * ``vendor_json_sections`` -- list of first- or third-party vendor
          JSON sections. Each entry is ``{id, kind, payload}`` with optional
          ``member`` (default ``"data"``), ``label``, and ``critical``. The
          kind must live in the ``x_<vendor>.*`` namespace.
        * ``eos_data`` -- dict with keys ``volumes`` (float64 [n_points]),
          ``energies`` (float64 [n_points]), ``fit`` (dict with ``model``,
          ``V0``, ``E0``, ``B0``, ``B0_prime``, etc.) for an
          ``equation_of_state`` section (QVF spec Sec. 4.14).
        * ``fermi_surface_data`` -- dict with keys ``nk1``, ``nk2``, ``nk3``
          (int), ``energies`` (float64 [nk1, nk2, nk3, n_bands]),
          ``band_indices`` (list of int), ``lattice_vectors`` (3x3),
          ``fermi_energy_ev`` (float), and optional ``n_spin`` (int,
          default 1) for a ``fermi_surface`` section (QVF spec Sec. 4.12).
        * ``wf_data`` -- dict with keys ``basis`` (list of shell dicts),
          ``mo_metadata`` (dict), ``mo_coefficients`` (2D || `[n_mo, n_ao]`),
          and optionally ``mo_coefficients_alpha`` / ``mo_coefficients_beta``
          for unrestricted. Emitted as ``wavefunction.gto`` with id ``"wf"``.
        * ``bloch_wf_data`` -- dict from :func:`qvf_bloch_wf_data` carrying
          all-k restricted Bloch orbitals or a physical spin-density pair,
          with source k-point metadata for QVF-backed periodic READ restarts. Emitted as the
          first-party vendor section ``x_vibeqc.bloch_wavefunction``.
        * ``wf_localized_data`` -- same shape as ``wf_data`` but emitted as a
          second ``wavefunction.gto`` section with id ``"wf_localized"``.
          ``mo_metadata["orbital_kind"]`` should be ``"localized"``.
        * ``wf_nto_hole_data`` -- same shape as ``wf_data``, emitted as
          ``wavefunction.gto`` with id ``"wf_nto_hole"`` and
          ``orbital_kind="natural"``.  The "hole" side of a single
          NTO pair (simplest post-TD-DFT path).
        * ``wf_nto_electron_data`` -- same shape as ``wf_data``, emitted as
          ``wavefunction.gto`` with id ``"wf_nto_electron"`` and
          ``orbital_kind="natural"``.  The "electron" side of a single
          NTO pair.
        * ``nto_data`` -- list of ``{"hole": wf_dict, "electron": wf_dict,
          "state_index": int, "excitation_energy_ev": float}`` records, one per
          excited state. Each emits two ``wavefunction.gto`` sections with
          ``orbital_kind="natural"`` and ids ``"wf_nto_S{n}_hole"`` /
          ``"wf_nto_S{n}_electron"``. Intended for Natural Transition Orbitals
          (post-TD-DFT).
        * ``ao_data`` -- list of dicts from :func:`qvf_ao_data`, each with
          keys ``label``, ``data`` (3-D array), ``origin``, ``span``,
          ``ao_metadata``, ``section_id``.  Emitted as ``basis.ao`` sections.
        * ``coop_data`` -- dict with keys ``energies`` (float64 [n_points]
          in eV), ``projections`` (float64 [n_pairs, n_points] or
          [n_spin, n_pairs, n_points]), ``integrated`` (float64 [n_pairs]),
          ``energies_units``, ``n_spin``, ``fermi_energy_ev``, ``sigma_ev``,
          ``pairs`` for a ``dos.coop`` section (QVF spec Sec. 4.8b).
        * ``cohp_data`` -- dict with keys ``energies``, ``projections``,
          ``integrated``, ``energies_units``, ``n_spin``, ``fermi_energy_ev``,
          ``sigma_ev``, ``pairs`` for a ``dos.cohp`` section
          (QVF spec Sec. 4.8c).

    Returns
    -------
    pathlib.Path
        The on-disk ``{stem}.qvf`` path.
    """
    stem = Path(os.fspath(stem))
    target = stem_sibling(stem, ".qvf")
    target.parent.mkdir(parents=True, exist_ok=True)

    # --- guardrails -----------------------------------------------------
    if not isinstance(plan, OutputPlan):
        raise TypeError(
            f"write_qvf: 'plan' must be an OutputPlan, got {type(plan).__name__}"
        )

    mol_or_sys = context.get("molecule") or context.get("system")

    # Volume size check.
    vol_data = context.get("volume_data")
    if vol_data:
        for label, (data, _o, _s) in vol_data.items():
            nv = int(np.prod(data.shape)) if hasattr(data, "shape") else 0
            if nv > _MAX_VOXELS:
                raise ValueError(
                    f"volume_data[{label!r}]: {nv:_d} voxels exceeds "
                    f"max {_MAX_VOXELS:_d}. Reduce grid or use a separate "
                    f".cube/.xsf file."
                )
    mo_data = context.get("mo_data")
    if mo_data:
        for mo in mo_data:
            data = mo.get("data")
            if data is not None and hasattr(data, "shape"):
                nv = int(np.prod(data.shape))
                if nv > _MAX_VOXELS:
                    raise ValueError(
                        f"mo_data[{mo.get('label', '?')!r}]: {nv:_d} voxels "
                        f"exceeds max {_MAX_VOXELS:_d}."
                    )

    # Warn (don't crash) on missing structure -- a QVF with no structure
    # section is valid but unusual.
    if mol_or_sys is None:
        import warnings

        warnings.warn(
            "write_qvf: no 'molecule' or 'system' in context -- the QVF "
            "will have no structure section. Pass molecule=<Mol> or "
            "system=<PeriodicSystem> for a complete archive.",
            UserWarning,
            stacklevel=2,
        )

    # Resolve compression.  Default deflate; try zstd if available.
    _compression = compression
    if compression == zipfile.ZIP_DEFLATED:
        try:
            from zipfile_zstd import ZIP_ZSTANDARD  # noqa: F811

            _compression = ZIP_ZSTANDARD
        except ImportError:
            pass

    volume_dt = np.dtype(volume_dtype)

    # --- build manifest skeleton ----------------------------------------
    _version = _resolve_version()
    manifest: dict[str, Any] = {
        "qvf_version": QVF_FORMAT_VERSION,
        "schema_uri": _SCHEMA_URI,
        "source": {
            "program": _PRODUCER_NAME,
            "version": _version,
            "calculation": (
                f"{context.get('method', '?')}/{context.get('basis', '?')}"
            ),
        },
        "sections": [],
    }
    sections: list[dict[str, Any]] = manifest["sections"]

    # --- provenance (manifest root) --------------------------------------
    manifest["provenance"] = _build_provenance(context)
    spin_diagnostic = context.get("spin_diagnostic_data")
    if spin_diagnostic is not None:
        manifest["provenance"]["spin_diagnostic"] = {
            "s_squared": float(spin_diagnostic["s_squared"]),
            "s_squared_ideal": float(spin_diagnostic["s_squared_ideal"]),
            "s_squared_deviation": float(
                spin_diagnostic["s_squared_deviation"]
            ),
        }

    # --- viewer_defaults (manifest root, optional) -----------------------
    vd = context.get("viewer_defaults")
    if vd is not None:
        manifest["viewer_defaults"] = dict(vd)

    # --- thermochemistry (manifest root, optional) -----------------------
    thermo = context.get("thermochemistry_data")
    if thermo is not None:
        manifest["thermochemistry"] = dict(thermo)

    # --- dipole_moment (manifest root, optional) -------------------------
    dipole = context.get("dipole_moment_data")
    if dipole is not None:
        manifest["dipole_moment"] = dict(dipole)

    # --- constraints (manifest root, optional) ---------------------------
    constraints = context.get("constraints_data")
    if constraints is not None:
        manifest["constraints"] = dict(constraints)

    # --- extensions (manifest root, optional) ----------------------------
    ext_block = context.get("extensions")
    if ext_block is not None:
        manifest["extensions"] = dict(ext_block)

    # Atomic mode writes to a temp sibling on the same filesystem and
    # ``os.replace``s it onto ``target`` only after validation passes, so
    # a concurrent reader never observes a partially-written zip. The
    # historical (default) path writes ``target`` in place.
    if atomic:
        fd, _tmp_name = tempfile.mkstemp(
            dir=str(target.parent),
            prefix=f".{target.name}.",
            suffix=".tmp",
        )
        os.close(fd)
        write_target = Path(_tmp_name)
    else:
        write_target = target

    try:
        with zipfile.ZipFile(write_target, "w", _compression) as zf:
            _emit_qvf_into_zip(
                zf,
                manifest=manifest,
                sections=sections,
                context=context,
                mol_or_sys=mol_or_sys,
                volume_dt=volume_dt,
            )

        # --- write-time validation gate ---------------------------------
        # Validate the freshly-written archive against the canonical
        # schema before returning. This is the producer-side enforcement
        # of the SSOT: write_qvf() never returns a path to an invalid
        # archive. Skipping the gate would let writer regressions ship
        # to consumers -- the exact failure mode the SSOT work fixes.
        report = validate_qvf(write_target)
        if not report["valid"]:
            raise ValueError(
                f"write_qvf produced an archive that fails canonical "
                f"validation -- this is a writer bug. Errors:\n  - "
                + "\n  - ".join(report["errors"][:8])
            )
    except BaseException:
        # Wipe the bad/partial output so a caller cannot mistake a stale
        # invalid file for a successful write. In atomic mode this only
        # touches the temp file, leaving any prior ``target`` intact.
        try:
            write_target.unlink()
        except OSError:
            pass
        raise

    if atomic:
        # POSIX ``os.replace`` is atomic within a filesystem; the temp
        # file and target share ``target.parent`` so the rename cannot
        # cross a filesystem boundary.
        os.replace(write_target, target)
    return target


def _emit_qvf_into_zip(
    zf: zipfile.ZipFile,
    *,
    manifest: dict[str, Any],
    sections: list[dict[str, Any]],
    context: dict[str, Any],
    mol_or_sys: Any,
    volume_dt: np.dtype,
) -> None:
    """Write every QVF section into an open zipfile + finalize the manifest.

    Factored out of :func:`write_qvf` so the in-memory helper
    :func:`qvf_bytes` shares the same emission pipeline.
    """
    # --- structure ----------------------------------------------
    if mol_or_sys is not None:
        is_periodic = bool(
            context.get("system") is not None and context.get("molecule") is None
        )
        _write_structure_section(
            zf,
            mol_or_sys,
            sections,
            periodic=is_periodic,
            biomolecule_data=context.get("biomolecule_data"),
            structure_lattice_bohr=context.get("structure_lattice_bohr"),
        )

    # --- volume.density -----------------------------------------
    vol_data = context.get("volume_data")
    if vol_data:
        _write_volume_density_section(zf, vol_data, sections, volume_dtype=volume_dt)

    # --- volume.orbital -----------------------------------------
    mo_data = context.get("mo_data")
    if mo_data:
        _write_volume_orbital_section(zf, mo_data, sections, volume_dtype=volume_dt)

    # --- volume.spin --------------------------------------------
    spin_data = context.get("spin_data")
    if spin_data:
        _write_volume_spin_section(zf, spin_data, sections, volume_dtype=volume_dt)

    # --- volume.elf ---------------------------------------------
    elf_data = context.get("elf_data")
    if elf_data:
        _write_volume_elf_section(zf, elf_data, sections, volume_dtype=volume_dt)

    # --- volume.difference --------------------------------------
    diff_data = context.get("diff_data")
    if diff_data:
        _write_volume_difference_section(
            zf, diff_data, sections, volume_dtype=volume_dt
        )

    # --- volume.generic -----------------------------------------
    gen_vol_data = context.get("generic_volume_data")
    if gen_vol_data:
        _write_volume_generic_section(
            zf, gen_vol_data, sections, volume_dtype=volume_dt
        )

    # --- volume.potential --------------------------------------
    potential_data = context.get("potential_data")
    if potential_data:
        _write_volume_potential_section(
            zf, potential_data, sections, volume_dtype=volume_dt
        )

    # --- volume.rdg -------------------------------------------
    rdg_data = context.get("rdg_data")
    if rdg_data:
        _write_volume_rdg_section(zf, rdg_data, sections, volume_dtype=volume_dt)

    # --- wavefunction.gto (canonical) ---------------------------
    wf_data = context.get("wf_data")
    if wf_data:
        _write_wavefunction_gto_section(zf, wf_data, sections)

    # --- x_vibeqc.bloch_wavefunction (all-k READ restart) ------
    bloch_wf_data = context.get("bloch_wf_data")
    if bloch_wf_data:
        _write_bloch_wavefunction_section(zf, bloch_wf_data, sections)

    # --- wavefunction.gto (localized) ----------------------------
    wf_localized_data = context.get("wf_localized_data")
    if isinstance(wf_localized_data, (list, tuple)):
        # Several localization criteria on the same wave function, each its
        # own section (``wf_localized_ibo``, ``..._boys``, ``..._pipek_mezey``)
        # so the viewer can switch between them.
        for _entry in wf_localized_data:
            if not _entry:
                continue
            _section_id, _payload = _entry
            if _payload:
                _write_wavefunction_gto_section(
                    zf, _payload, sections, section_id=_section_id
                )
        wf_localized_data = None
    if wf_localized_data:
        _write_wavefunction_gto_section(
            zf, wf_localized_data, sections, section_id="wf_localized"
        )

    # --- wavefunction.gto (NTO hole / electron pairs) --------------
    # Standalone keys: a single hole/electron pair (simplest path).
    # Emitted as sections "wf_nto_hole" and "wf_nto_electron".
    # The mo_metadata["orbital_kind"] should be "natural".
    wf_nto_hole_data = context.get("wf_nto_hole_data")
    if wf_nto_hole_data:
        wf_nto_hole_data.setdefault("mo_metadata", {})
        wf_nto_hole_data["mo_metadata"]["orbital_kind"] = "natural"
        wf_nto_hole_data["mo_metadata"]["occupation_semantics"] = (
            "transition_weight"
        )
        _write_wavefunction_gto_section(
            zf, wf_nto_hole_data, sections, section_id="wf_nto_hole"
        )
    wf_nto_electron_data = context.get("wf_nto_electron_data")
    if wf_nto_electron_data:
        wf_nto_electron_data.setdefault("mo_metadata", {})
        wf_nto_electron_data["mo_metadata"]["orbital_kind"] = "natural"
        wf_nto_electron_data["mo_metadata"]["occupation_semantics"] = (
            "transition_weight"
        )
        _write_wavefunction_gto_section(
            zf, wf_nto_electron_data, sections, section_id="wf_nto_electron"
        )

    # --- wavefunction.gto (NTO hole / electron pairs, multi-state) --
    # Each entry is {"hole": wf_data_dict, "electron": wf_data_dict,
    # "state_index": int, "excitation_energy_ev": float,
    # optional "spin": "alpha"|"beta"}.
    # Emitted as two sections per state: wf_nto_S{n}_hole and
    # wf_nto_S{n}_electron (or wf_nto_S{n}_{spin}_hole etc. for UHF).
    # The mo_metadata["orbital_kind"] should be "natural".
    nto_states = context.get("nto_data")
    if nto_states:
        for state in nto_states:
            si = state.get("state_index", 0)
            e_ev = state.get("excitation_energy_ev")
            _spin = state.get("spin", "")
            _spin_tag = f"_{_spin}" if _spin else ""
            for role in ("hole", "electron"):
                wf = state.get(role)
                if wf is None:
                    continue
                # Ensure orbital_kind is set.
                wf.setdefault("mo_metadata", {})
                wf["mo_metadata"]["orbital_kind"] = "natural"
                wf["mo_metadata"]["occupation_semantics"] = "transition_weight"
                if e_ev is not None:
                    wf["mo_metadata"].setdefault("excitation_energy_ev", float(e_ev))
                    wf["mo_metadata"].setdefault("excitation_index", int(si))
                sec_id = f"wf_nto_S{si}{_spin_tag}_{role}"
                _write_wavefunction_gto_section(zf, wf, sections, section_id=sec_id)

    # --- basis.ao -----------------------------------------------
    ao_data = context.get("ao_data")
    if ao_data:
        _write_basis_ao_section(zf, ao_data, sections, volume_dtype=volume_dt)

    # --- atom_properties ----------------------------------------
    pop = context.get("population_summary")
    iao_charges = context.get("iao_charges")
    iao_analysis = getattr(pop, "iao_analysis", None)
    if iao_analysis is not None:
        if iao_analysis.available:
            iao_charges = iao_analysis.charges
        _write_vendor_json_sections(zf, [{
            "id": "iao_analysis", "kind": "x_vibeqc.iao_analysis",
            "path": "analysis/iao.json", "payload": iao_analysis.to_dict(),
        }], sections)
    if pop is not None or iao_charges is not None:
        _write_atom_properties_section(
            zf, pop, sections, iao_charges=iao_charges
        )

    # --- trajectory ---------------------------------------------
    traj_frames = context.get("trajectory_frames")
    if traj_frames:
        _write_trajectory_section(
            zf,
            traj_frames,
            sections,
            energies=context.get("trajectory_energies"),
            rms_grad=context.get("trajectory_rms_grad"),
            trajectory_type=context.get(
                "trajectory_type",
                "geometry_optimization",
            ),
        )

    # --- reaction.path (self-contained) -------------------------
    rxn_path = context.get("reaction_path")
    if rxn_path:
        # A reaction path whose frames are PeriodicSystem instances carries the
        # per-frame `lattice` member + `dim` metadata that vibe-view needs to
        # render the cell and wrap atoms across periodic boundaries. Both are
        # *optional* additions, so the archive stays qvf_version=1: a consumer
        # detects a periodic path from the presence of the `lattice` member,
        # never from the manifest version.
        #
        # vibe-qc v0.10.0-v0.15.x bumped to qvf_version=2 here. That bump was
        # withdrawn by the governance ruling of 2026-07-10 (an optional member
        # is the additive case and must not bump the version, and the bump
        # obliged conforming v1-only consumers to refuse archives we routinely
        # wrote). See qvf-writer/GOVERNANCE.md, "Version history".
        _write_reaction_path_section(
            zf,
            rxn_path["frames"],
            rxn_path["waypoints"],
            sections,
            energies=rxn_path.get("energies"),
            reaction_coordinate=rxn_path.get("reaction_coordinate"),
            reaction_coordinate_label=rxn_path.get("reaction_coordinate_label"),
            reaction_coordinate_unit=rxn_path.get("reaction_coordinate_unit"),
            frame_volumes=rxn_path.get("frame_volumes"),
            volume_grid=rxn_path.get("volume_grid"),
            volume_frame_index=rxn_path.get("volume_frame_index"),
            volume_label=rxn_path.get("volume_label", "Electron density"),
            volume_isovalue=rxn_path.get("volume_isovalue"),
        )

    # --- reaction.waypoints (annotation over a trajectory) ------
    rxn_wps = context.get("reaction_waypoints")
    if rxn_wps:
        traj_ref = rxn_wps["trajectory_ref"]
        traj_section = next(
            (s for s in sections if s.get("id") == traj_ref),
            None,
        )
        if traj_section is None or traj_section.get("kind") != "trajectory":
            raise ValueError(
                f"reaction.waypoints: trajectory_ref={traj_ref!r} does "
                "not name a trajectory section emitted in this archive. "
                "Producers MUST emit the referenced trajectory first."
            )
        n_traj_frames = int(traj_section["members"]["coords"]["shape"][0])
        _write_reaction_waypoints_section(
            zf,
            traj_ref,
            rxn_wps["waypoints"],
            n_traj_frames,
            sections,
            reaction_coordinate=rxn_wps.get("reaction_coordinate"),
        )

    # --- scan.surface (2D relaxed-scan energy grid) -------------
    scan_surf = context.get("scan_surface")
    if scan_surf:
        _write_scan_surface_section(
            zf,
            sections,
            axis_a=scan_surf["axis_a"],
            axis_b=scan_surf["axis_b"],
            energies=scan_surf["energies"],
            coordinate_a_label=scan_surf["coordinate_a_label"],
            coordinate_a_unit=scan_surf["coordinate_a_unit"],
            coordinate_b_label=scan_surf["coordinate_b_label"],
            coordinate_b_unit=scan_surf["coordinate_b_unit"],
            geometries=scan_surf.get("geometries"),
            atoms=scan_surf.get("atoms"),
        )

    # --- vibrations ---------------------------------------------
    hess = context.get("hessian_result")
    if hess is not None:
        mol = context.get("molecule")
        syms = None
        if mol is not None:
            syms = [_symbol(int(a.Z)) for a in mol.atoms]
        _write_vibrations_section(
            zf,
            hess,
            sections,
            atom_symbols=syms,
            molecule=mol,
            surface=context.get("hessian_surface"),
        )

    # --- spectra.ir ---------------------------------------------
    if hess is not None:
        _write_spectra_ir_section(zf, hess, sections)

    # --- bands --------------------------------------------------
    bs = context.get("band_structure")
    if bs is not None:
        _write_bands_section(zf, bs, sections)

    # --- citations ----------------------------------------------
    bib = context.get("bibtex_content")
    if bib:
        _write_citations_section(zf, bib, sections)

    # --- run.record ---------------------------------------------
    run_record = context.get("run_record")
    if run_record:
        _write_run_record_section(zf, run_record, sections)

    # --- job.spec -----------------------------------------------
    job_spec = context.get("job_spec")
    if job_spec:
        _write_job_spec_section(zf, job_spec, sections)

    # --- dos.total ----------------------------------------------
    dos_data = context.get("dos_data")
    if dos_data:
        _write_dos_total_section(zf, dos_data, sections)

    # --- dos.projected ------------------------------------------
    pdos_data = context.get("pdos_data")
    if pdos_data:
        _write_dos_projected_section(zf, pdos_data, sections)

    # --- dos.coop ----------------------------------------------
    coop_data = context.get("coop_data")
    if coop_data:
        _write_dos_coop_section(zf, coop_data, sections)

    # --- dos.cohp ----------------------------------------------
    cohp_data = context.get("cohp_data")
    if cohp_data:
        _write_dos_cohp_section(zf, cohp_data, sections)

    # --- spectra.raman ------------------------------------------
    raman = context.get("raman_data")
    if raman:
        _write_spectra_raman_section(zf, raman, sections)

    # --- spectra.uvvis ------------------------------------------
    uvvis = context.get("uvvis_data")
    if uvvis:
        _write_spectra_uvvis_section(zf, uvvis, sections)

    # --- spectra.ecd --------------------------------------------
    ecd = context.get("ecd_data")
    if ecd:
        _write_spectra_ecd_section(zf, ecd, sections)

    # --- spectra.vcd --------------------------------------------
    vcd = context.get("vcd_data")
    if vcd:
        _write_spectra_vcd_section(zf, vcd, sections)

    # --- spectra.nmr --------------------------------------------
    nmr = context.get("nmr_data")
    if nmr:
        _write_spectra_nmr_section(zf, nmr, sections)

    # --- spectra.epr --------------------------------------------
    epr = context.get("epr_data")
    if epr:
        _write_spectra_epr_section(zf, epr, sections)

    # --- spectra.generic ----------------------------------------
    generic_spec = context.get("generic_spectrum_data")
    if generic_spec:
        _write_spectra_generic_section(zf, generic_spec, sections)

    # --- structure.symmetry -------------------------------------
    sym = context.get("symmetry_data")
    if sym:
        _write_symmetry_section(zf, sym, sections)

    # --- bonds --------------------------------------------------
    bonds = context.get("bonds_data")
    if bonds:
        _write_bonds_section(zf, bonds, sections)

    # --- bond_orders ---------------------------------------------
    bo_data = context.get("bond_orders_data")
    if bo_data:
        _write_bond_orders_section(zf, bo_data["method"], bo_data["pairs"], sections)

    # --- scf_history --------------------------------------------
    scf_hist = context.get("scf_history_data")
    if scf_hist:
        _write_scf_history_section(zf, scf_hist, sections)

    # --- equation_of_state --------------------------------------
    eos_data = context.get("eos_data")
    if eos_data:
        _write_equation_of_state_section(zf, eos_data, sections)

    # --- fermi_surface -----------------------------------------
    fermi_data = context.get("fermi_surface_data")
    if fermi_data:
        _write_fermi_surface_section(zf, fermi_data, sections)

    # --- phonon_bands ------------------------------------------
    phonon_bands_data = context.get("phonon_bands_data")
    if phonon_bands_data:
        _write_phonon_bands_section(zf, phonon_bands_data, sections)

    # --- phonon_dos --------------------------------------------
    phonon_dos_data = context.get("phonon_dos_data")
    if phonon_dos_data:
        _write_phonon_dos_section(zf, phonon_dos_data, sections)

    # --- topology.qtaim -----------------------------------------
    qtaim = context.get("qtaim_data")
    if qtaim:
        _write_topology_qtaim_section(
            zf,
            qtaim["critical_points"],
            bond_paths=qtaim.get("bond_paths"),
            sections=sections,
        )

    # --- x_<vendor> JSON sections -----------------------------------
    vendor_json_sections = context.get("vendor_json_sections")
    if vendor_json_sections:
        _write_vendor_json_sections(zf, vendor_json_sections, sections)

    # --- partial-section marking (streaming checkpoints) --------
    # A checkpoint snapshot flags every still-growing section (e.g. an
    # optimization ``trajectory`` or an SCF-history section not yet at
    # its final point) with ``partial: true`` so a live consumer knows
    # not to treat it as final. Sections carry no ``additionalProperties:
    # false`` in the schema, so the extra key validates under v1.
    _mark_partial_sections(sections, context.get("partial_sections"))

    # --- manifest.json (ZIP_STORED, always last by convention) ---
    manifest_bytes = safe_json_bytes(manifest, indent=2)
    zf.writestr(
        zipfile.ZipInfo("manifest.json"),
        manifest_bytes,
        compress_type=zipfile.ZIP_STORED,
    )


def _mark_partial_sections(
    sections: list[dict[str, Any]],
    partial_sections: Any,
) -> None:
    """Set ``partial=True`` on emitted sections named by ``partial_sections``.

    ``partial_sections`` may be ``None`` (no-op), ``True`` (mark every
    section partial -- the whole snapshot is mid-flight), or an iterable of
    section ids and/or ``kind`` strings. Matching is by section ``id`` first,
    then ``kind``, so a caller can say ``{"trajectory"}`` to flag every
    trajectory section without knowing the generated ids.
    """
    if not partial_sections:
        return
    if partial_sections is True:
        for sec in sections:
            sec["partial"] = True
        return
    wanted = {str(x) for x in partial_sections}
    for sec in sections:
        if sec.get("id") in wanted or sec.get("kind") in wanted:
            sec["partial"] = True


def _write_vendor_json_sections(
    zf: zipfile.ZipFile,
    vendor_sections: Sequence[dict[str, Any]],
    sections: list[dict[str, Any]],
) -> None:
    """Write caller-supplied JSON sections in an ``x_<vendor>`` namespace."""
    for idx, spec in enumerate(vendor_sections):
        kind = str(spec.get("kind", ""))
        if not _is_vendor_kind(kind):
            raise ValueError(
                "vendor_json_sections entries must use an x_<vendor> kind; "
                f"got {kind!r}"
            )
        if "payload" not in spec:
            raise ValueError(
                "vendor_json_sections entries require a JSON 'payload' value"
            )

        section_id = str(spec.get("id") or f"vendor_json_{idx}")
        member_name = str(spec.get("member", "data"))
        section_slug = _slug(section_id, fallback=f"vendor_json_{idx}")
        member_slug = _slug(member_name, fallback="data")
        path_in_zip = str(
            spec.get("path") or f"vendor/{section_slug}/{member_slug}.json"
        )
        payload_json = safe_json_bytes(spec["payload"], indent=2)
        zf.writestr(path_in_zip, payload_json)

        section: dict[str, Any] = {
            "id": section_id,
            "kind": kind,
            "members": {
                member_name: {
                    "path": path_in_zip,
                    "format": "json",
                    "sha256": _sha256_hex(payload_json),
                },
            },
        }
        if "label" in spec:
            section["label"] = str(spec["label"])
        if "critical" in spec:
            section["critical"] = bool(spec["critical"])
        sections.append(section)


def qvf_bytes(
    plan: "OutputPlan",  # noqa: F821 -- same forward ref as write_qvf
    *,
    compression: int = zipfile.ZIP_DEFLATED,
    volume_dtype: str = "float32",
    **context: Any,
) -> bytes:
    """Build a QVF archive in memory and return the raw zip bytes.

    Mirrors :func:`write_qvf` -- same ``plan`` and ``**context`` surface,
    same emission pipeline, same canonical post-build validation gate
    -- but never touches the filesystem. Use this when vibe-qc wants to
    hand a freshly built QVF directly to ``vibe-view`` (see
    :func:`vibeview.launch_qvf`) without a temporary file.

    Returns
    -------
    bytes
        The complete .qvf archive bytes. ``QVFReader(<bytes>)`` opens
        them directly.

    Raises
    ------
    ValueError
        If the in-memory archive fails the SSOT validation gate. The
        bytes are *not* returned in that case -- same behaviour as
        :func:`write_qvf` unlinking the on-disk artefact.
    """
    if not isinstance(plan, OutputPlan):
        raise TypeError(
            f"qvf_bytes: 'plan' must be an OutputPlan, got {type(plan).__name__}"
        )

    mol_or_sys = context.get("molecule") or context.get("system")

    # Shared voxel size guard (single source = _MAX_VOXELS module const).
    vol_data = context.get("volume_data")
    if vol_data:
        for label, (data, _o, _s) in vol_data.items():
            nv = int(np.prod(data.shape)) if hasattr(data, "shape") else 0
            if nv > _MAX_VOXELS:
                raise ValueError(
                    f"volume_data[{label!r}]: {nv:_d} voxels exceeds "
                    f"max {_MAX_VOXELS:_d}."
                )
    mo_data = context.get("mo_data")
    if mo_data:
        for mo in mo_data:
            data = mo.get("data")
            if data is not None and hasattr(data, "shape"):
                nv = int(np.prod(data.shape))
                if nv > _MAX_VOXELS:
                    raise ValueError(
                        f"mo_data[{mo.get('label', '?')!r}]: {nv:_d} voxels "
                        f"exceeds max {_MAX_VOXELS:_d}."
                    )

    if mol_or_sys is None:
        import warnings

        warnings.warn(
            "qvf_bytes: no 'molecule' or 'system' in context -- the QVF "
            "will have no structure section.",
            UserWarning,
            stacklevel=2,
        )

    volume_dt = np.dtype(volume_dtype)
    _version = _resolve_version()
    manifest: dict[str, Any] = {
        "qvf_version": QVF_FORMAT_VERSION,
        "schema_uri": _SCHEMA_URI,
        "source": {
            "program": _PRODUCER_NAME,
            "version": _version,
            "calculation": (
                f"{context.get('method', '?')}/{context.get('basis', '?')}"
            ),
        },
        "sections": [],
    }
    sections: list[dict[str, Any]] = manifest["sections"]
    manifest["provenance"] = _build_provenance(context)
    vd = context.get("viewer_defaults")
    if vd is not None:
        manifest["viewer_defaults"] = dict(vd)

    buf = io.BytesIO()
    with zipfile.ZipFile(buf, "w", compression) as zf:
        _emit_qvf_into_zip(
            zf,
            manifest=manifest,
            sections=sections,
            context=context,
            mol_or_sys=mol_or_sys,
            volume_dt=volume_dt,
        )

    # SSOT validation gate -- re-open the in-memory archive read-only
    # and run the canonical validator. Same blast radius as write_qvf:
    # raise rather than return bad bytes.
    buf.seek(0)
    with zipfile.ZipFile(buf, "r") as zf_ro:
        report = validate_qvf(zf_ro)
    if not report["valid"]:
        raise ValueError(
            "qvf_bytes produced an archive that fails canonical "
            "validation -- this is a writer bug. Errors:\n  - "
            + "\n  - ".join(report["errors"][:8])
        )
    return buf.getvalue()


def _resolve_version() -> str:
    """Try to import vibe-qc's version string; fall back gracefully."""
    try:
        from ...banner import VIBEQC_VERSION

        return str(VIBEQC_VERSION)
    except Exception:
        return "0.0.0"


# Whole-or-nothing embedding caps for run.record payloads. A driving
# script larger than the input cap is almost certainly not the job input;
# a log beyond the log cap would hold the archive hostage in RAM.
_RUN_RECORD_INPUT_CAP = 4 * 1024 * 1024
_RUN_RECORD_LOG_CAP = 256 * 1024 * 1024


def terminal_run_status(result: Any) -> str:
    """``"converged"`` or ``"failed"`` for a settled terminal archive.

    Spec § 3.2 gives ``provenance.run_status`` normative lifecycle
    semantics, and an *absent* status means "not stated" -- a consumer
    MUST NOT infer one. So a finished run has to say so explicitly, or
    the archive it wrote never declares that it finished.

    The rule matches the one the QVF job-container finalizer applies
    (``qvf_job.py``), so a calculation settles to the same status whether
    it was driven by the ordinary API or by ``vibeqc run job.qvf``: a run
    is ``failed`` only when it reports a falsy ``converged``; a result
    that does not report convergence at all settles as ``converged``.
    """
    converged = getattr(result, "converged", None)
    if converged is not None and not bool(converged):
        return "failed"
    return "converged"


def assemble_run_record(
    plan: OutputPlan | None,
    *,
    wall_seconds: float | None = None,
    log_truncated: bool = False,
) -> dict[str, Any] | None:
    """Best-effort ``run_record`` context dict for vibe-qc's own runs.

    Captures two things, each optional:

    * the **driving user script** -- ``sys.modules["__main__"].__file__``
      when it is a readable ``.py`` file. This covers ``python script.py``
      and ``vibeqc run script.py`` (runpy installs the script as
      ``__main__`` for the duration); notebooks and the REPL have no
      script file and are skipped.
    * the **on-disk `.out` log** from the plan's ``log`` role. The
      runners call this after the output channel has closed, so the text
      is the complete log at archive time (pass ``log_truncated=True``
      when content is still appended afterwards, e.g. the periodic
      geometry-optimization epilogue).

    Only basenames are recorded -- absolute paths never enter the
    archive. Returns ``None`` when neither payload is available (e.g.
    stdout-only channel driven from a REPL).
    """
    input_text: str | None = None
    input_filename: str | None = None
    script = getattr(sys.modules.get("__main__"), "__file__", None)
    if script:
        script_path = Path(script)
        try:
            if (
                script_path.suffix == ".py"
                and script_path.is_file()
                and script_path.stat().st_size <= _RUN_RECORD_INPUT_CAP
            ):
                input_text = script_path.read_text(
                    encoding="utf-8", errors="replace"
                )
                input_filename = script_path.name
        except OSError:
            pass

    log_text: str | None = None
    log_filename: str | None = None
    if plan is not None:
        for planned in plan.files_by_role("log"):
            log_path = Path(planned.path)
            try:
                if (
                    log_path.is_file()
                    and log_path.stat().st_size <= _RUN_RECORD_LOG_CAP
                ):
                    log_text = log_path.read_text(
                        encoding="utf-8", errors="replace"
                    )
                    log_filename = log_path.name
            except OSError:
                pass
            break

    if input_text is None and log_text is None:
        return None

    finished = _dt.datetime.now(_dt.timezone.utc)
    record: dict[str, Any] = {
        "program": "vibe-qc",
        "program_version": _resolve_version(),
        "finished_utc": finished.strftime("%Y-%m-%dT%H:%M:%SZ"),
    }
    if wall_seconds is not None and wall_seconds >= 0:
        started = finished - _dt.timedelta(seconds=float(wall_seconds))
        record["started_utc"] = started.strftime("%Y-%m-%dT%H:%M:%SZ")
    if input_text is not None:
        record["input_text"] = input_text
        record["input_filename"] = input_filename
    if log_text is not None:
        record["log_text"] = log_text
        record["log_filename"] = log_filename
        if log_truncated:
            record["log_truncated"] = True
    return record


# ---------------------------------------------------------------------------
# Section writers -- each adds its entry to the ``sections`` list
# ---------------------------------------------------------------------------

# -- structure ------------------------------------------------------------


# Pinned by ``$defs/BiomoleculeSecondaryStructure.type`` in the manifest
# schema; kept here so the writer drops what the schema would reject.
_SECONDARY_STRUCTURE_TYPES = frozenset({"helix", "sheet", "coil"})


def _normalize_biomolecule(bio: Any) -> dict[str, Any]:
    """Coerce a ``biomolecule_data`` dict to plain JSON-safe values.

    Returns a dict with only the fields that were supplied — all four
    are independently optional. numpy ints/arrays are coerced to plain
    ``int``/``list`` so :func:`safe_json_bytes` and the write-time
    :func:`validate_qvf` gate stay clean. Malformed entries are dropped
    rather than raising, mirroring the fail-soft writer policy.

    Fields (peer keys on the ``structure`` section object, additive —
    no ``qvf_version`` bump; schema'd as optional properties of
    ``$defs/SectionStructure``):

      * ``chains`` -- list[str] of chain ids, e.g. ``["A", "B"]``.
      * ``residues`` -- list of ``{name, seq, chain, atom_indices}``;
        ``atom_indices`` are 0-based into the structure payload atoms.
      * ``secondary_structure`` -- list of
        ``{type, chain, start_seq, end_seq}`` with ``type`` in
        ``{"helix", "sheet", "coil"}``. Anything else is dropped rather
        than written, because the schema pins the enum and an unknown
        descriptor would fail the write-time validate gate for the whole
        archive.
      * ``b_factors`` -- list[float], one per atom, parallel to the
        structure payload's atom order.
    """
    if not isinstance(bio, dict):
        return {}
    out: dict[str, Any] = {}

    chains = bio.get("chains")
    if chains is not None:
        out["chains"] = [str(c) for c in chains]

    residues = bio.get("residues")
    if residues is not None:
        norm_res = []
        for r in residues:
            if not isinstance(r, dict):
                continue
            # Explicit None check, not ``or []`` — a numpy-array
            # atom_indices would raise "truth value ambiguous".
            ai = r.get("atom_indices")
            ai = [] if ai is None else list(ai)
            norm_res.append(
                {
                    "name": str(r.get("name", "")),
                    "seq": int(r.get("seq", 0)),
                    "chain": str(r.get("chain", "")),
                    "atom_indices": [int(i) for i in ai],
                }
            )
        out["residues"] = norm_res

    ss = bio.get("secondary_structure")
    if ss is not None:
        norm_ss = []
        for s in ss:
            if not isinstance(s, dict):
                continue
            # The schema pins ``type`` to helix/sheet/coil. An unrecognized
            # descriptor (a DSSP "turn", say) is dropped rather than written:
            # writing it would fail the write-time validate_qvf gate and take
            # the whole archive down over one range.
            kind = str(s.get("type", "coil"))
            if kind not in _SECONDARY_STRUCTURE_TYPES:
                continue
            norm_ss.append(
                {
                    "type": kind,
                    "chain": str(s.get("chain", "")),
                    "start_seq": int(s.get("start_seq", 0)),
                    "end_seq": int(s.get("end_seq", 0)),
                }
            )
        out["secondary_structure"] = norm_ss

    b_factors = bio.get("b_factors")
    if b_factors is not None:
        norm_b = []
        for value in b_factors:
            try:
                norm_b.append(float(value))
            except (TypeError, ValueError):
                # A non-numeric entry would shift every later atom's
                # b-factor by one if it were skipped, so neutralize it in
                # place instead of dropping it.
                norm_b.append(0.0)
        out["b_factors"] = norm_b

    return out


def _write_structure_section(
    zf: zipfile.ZipFile,
    mol_or_sys: Any,
    sections: list[dict[str, Any]],
    *,
    periodic: bool = False,
    biomolecule_data: Any = None,
    structure_lattice_bohr: Any = None,
) -> None:
    """Write the ``structure`` section.

    Molecules: ``Molecule.atoms``. Periodic systems: the bound
    ``PeriodicSystem`` exposes ``unit_cell`` (not ``atoms``) plus
    ``dim`` in {1,2,3} and ``lattice`` (Cartesian column vectors in
    bohr). Tries ``.atoms`` first for back-compat with molecular
    callers and any test stubs that still use the old name.
    """
    # PeriodicSystem (the bound C++ type) carries atoms under
    # ``unit_cell``; Molecule carries them under ``atoms``. Pick the
    # populated one. Some stubs and the molecular path use ``atoms``;
    # the periodic path uses ``unit_cell``.
    atoms = list(getattr(mol_or_sys, "atoms", []) or [])
    if not atoms:
        atoms = list(getattr(mol_or_sys, "unit_cell", []) or [])
    if not atoms:
        import warnings

        warnings.warn(
            "_write_structure_section: molecule/system has no atoms "
            "(.atoms / .unit_cell both empty) -- skipping structure "
            "section.",
            UserWarning,
            stacklevel=2,
        )
        return

    # Build atoms array for JSON.
    atom_list = []
    for a in atoms:
        x, y, z = a.xyz
        atom_list.append(
            {
                "symbol": _symbol(int(a.Z)),
                "position": [
                    float(x) * _BOHR_TO_ANGSTROM,
                    float(y) * _BOHR_TO_ANGSTROM,
                    float(z) * _BOHR_TO_ANGSTROM,
                ],
                "atomic_number": int(a.Z),
            }
        )

    payload: dict[str, Any] = {
        "atoms": atom_list,
        "pbc": [False, False, False],
    }

    if periodic:
        lattice_source = (
            structure_lattice_bohr
            if structure_lattice_bohr is not None
            else mol_or_sys.lattice
        )
        L_bohr = np.asarray(lattice_source, dtype=np.float64)
        if L_bohr.shape != (3, 3):
            raise ValueError(
                "structure_lattice_bohr must be a 3x3 bohr column-vector lattice"
            )
        L_ang = (L_bohr * _BOHR_TO_ANGSTROM).T.tolist()  # row vectors for JSON
        # PeriodicSystem exposes ``dim`` in {1,2,3}; molecular stubs may
        # use ``dimensionality``. Default to fully periodic.
        dim = int(
            getattr(mol_or_sys, "dim", None)
            or getattr(mol_or_sys, "dimensionality", 3)
            or 3
        )
        dim = max(1, min(3, dim))
        payload["pbc"] = [i < dim for i in range(3)]
        payload["dimensionality"] = dim
        payload["lattice_vectors"] = [
            [float(L_ang[0][0]), float(L_ang[0][1]), float(L_ang[0][2])],
            [float(L_ang[1][0]), float(L_ang[1][1]), float(L_ang[1][2])],
            [float(L_ang[2][0]), float(L_ang[2][1]), float(L_ang[2][2])],
        ]

    struct_json = safe_json_bytes(payload, indent=2)
    path_in_zip = "structure/structure.json"
    zf.writestr(path_in_zip, struct_json)

    section: dict[str, Any] = {
        "id": "structure",
        "kind": "structure",
        "members": {
            "structure": {
                "path": path_in_zip,
                "format": "json",
                "sha256": _sha256_hex(struct_json),
            },
        },
    }
    # Biomolecule metadata (Ask B) — optional, additive peer keys on the
    # structure Section object (NOT inside structure.json), schema'd as
    # optional properties of $defs/SectionStructure; no qvf_version bump
    # (qvf-writer/GOVERNANCE.md: an optional member is the additive case).
    if biomolecule_data is not None:
        section.update(_normalize_biomolecule(biomolecule_data))
    sections.append(section)


# -- volume.density / volume.orbital helpers ------------------------------


def _grid_descriptor(
    data: np.ndarray,
    origin: np.ndarray,
    span: np.ndarray,
) -> dict[str, Any]:
    """Build the ``grid`` member dict for a volumetric section."""
    ox, oy, oz = float(origin[0]), float(origin[1]), float(origin[2])
    nx, ny, nz = data.shape
    grid: dict[str, Any] = {
        "origin": [ox, oy, oz],
        "voxel_vectors": [
            [float(span[0, 0]), float(span[0, 1]), float(span[0, 2])],
            [float(span[1, 0]), float(span[1, 1]), float(span[1, 2])],
            [float(span[2, 0]), float(span[2, 1]), float(span[2, 2])],
        ],
        "shape": [int(nx), int(ny), int(nz)],
    }
    return grid


def _write_volume_density_section(
    zf: zipfile.ZipFile,
    vol_data: dict[str, tuple],
    sections: list[dict[str, Any]],
    *,
    volume_dtype: np.dtype = np.dtype("float32"),
) -> None:
    """Write ``volume.density`` sections.

    ``vol_data`` is ``{label: (data_3d, origin_bohr, span_bohr)}``.
    Origin and span are in bohr; we write the grid descriptor in bohr.
    """
    for idx, (label, (data, origin, span)) in enumerate(vol_data.items()):
        vol = _real_array_for_artifact(
            data,
            dtype=volume_dtype,
            label=f"volume.density[{label!r}]",
        )
        _require_3d_volume(vol, "volume.density", label)
        section_id = f"vol_dens_{idx}"
        slug = _slug(label, fallback=section_id)
        path_in_zip = f"volumes/{slug}.dat"
        file_member = _write_binary_to_zip(zf, path_in_zip, vol)
        origin_arr = np.asarray(origin, dtype=np.float64)
        span_arr = np.asarray(span, dtype=np.float64).reshape(3, 3)
        grid = _grid_descriptor(vol, origin_arr, span_arr)
        grid_json = safe_json_bytes(grid)
        grid_path = f"volumes/{slug}_grid.json"
        zf.writestr(grid_path, grid_json)
        section: dict[str, Any] = {
            "id": section_id,
            "kind": "volume.density",
            "label": label,
            "members": {
                "data": file_member,
                "grid": {
                    "path": grid_path,
                    "format": "json",
                    "sha256": _sha256_hex(grid_json),
                },
            },
        }
        sections.append(section)


def _write_volume_orbital_section(
    zf: zipfile.ZipFile,
    mo_data: list[dict[str, Any]],
    sections: list[dict[str, Any]],
    *,
    volume_dtype: np.dtype = np.dtype("float32"),
) -> None:
    """Write ``volume.orbital`` sections.

    Each item in ``mo_data`` is a dict with keys: ``label``, ``data``
    (3D array), ``origin`` (3-vector, bohr), ``span`` (3x3, bohr),
    ``band_index`` (int), ``energy_eh`` (float), optional
    ``occupation`` (float, default 2.0 for restricted), ``spin``
    (int, default 0), ``component`` (str, "real" default).
    """
    for idx, mo in enumerate(mo_data):
        label = mo.get("label", f"MO_{idx}")
        vol = _real_array_for_artifact(
            mo["data"],
            dtype=volume_dtype,
            label=f"volume.orbital[{label!r}]",
        )
        _require_3d_volume(vol, "volume.orbital", label)
        section_id = f"vol_mo_{idx}"
        slug = _slug(label, fallback=section_id)
        path_in_zip = f"volumes/{slug}.dat"
        file_member = _write_binary_to_zip(zf, path_in_zip, vol)
        comp = mo.get("component", "real")
        origin_arr = np.asarray(mo["origin"], dtype=np.float64)
        span_arr = np.asarray(mo["span"], dtype=np.float64).reshape(3, 3)
        grid = _grid_descriptor(vol, origin_arr, span_arr)
        grid_json = safe_json_bytes(grid)
        grid_path = f"volumes/{slug}_grid.json"
        zf.writestr(grid_path, grid_json)
        section: dict[str, Any] = {
            "id": section_id,
            "kind": "volume.orbital",
            "label": label,
            "component": comp,
            "members": {
                "data": file_member,
                "grid": {
                    "path": grid_path,
                    "format": "json",
                    "sha256": _sha256_hex(grid_json),
                },
            },
        }
        sections.append(section)


# -- volume.spin -----------------------------------------------------------


def _write_volume_spin_section(
    zf: zipfile.ZipFile,
    spin_data: dict[str, tuple],
    sections: list[dict[str, Any]],
    *,
    volume_dtype: np.dtype = np.dtype("float32"),
) -> None:
    """Write ``volume.spin`` sections.

    ``spin_data`` is ``{label: (data_3d, origin_bohr, span_bohr)}``.
    Same shape as ``volume.density``, different kind string.
    """
    for idx, (label, (data, origin, span)) in enumerate(spin_data.items()):
        vol = _real_array_for_artifact(
            data,
            dtype=volume_dtype,
            label=f"volume.spin[{label!r}]",
        )
        _require_3d_volume(vol, "volume.spin", label)
        section_id = f"vol_spin_{idx}"
        slug = _slug(label, fallback=section_id)
        path_in_zip = f"volumes/{slug}_spin.dat"
        file_member = _write_binary_to_zip(zf, path_in_zip, vol)
        origin_arr = np.asarray(origin, dtype=np.float64)
        span_arr = np.asarray(span, dtype=np.float64).reshape(3, 3)
        grid = _grid_descriptor(vol, origin_arr, span_arr)
        grid_json = safe_json_bytes(grid)
        grid_path = f"volumes/{slug}_spin_grid.json"
        zf.writestr(grid_path, grid_json)
        section: dict[str, Any] = {
            "id": section_id,
            "kind": "volume.spin",
            "label": label,
            "members": {
                "data": file_member,
                "grid": {
                    "path": grid_path,
                    "format": "json",
                    "sha256": _sha256_hex(grid_json),
                },
            },
        }
        sections.append(section)


# -- volume.elf -------------------------------------------------------------


def _write_volume_elf_section(
    zf: zipfile.ZipFile,
    elf_data: dict[str, tuple],
    sections: list[dict[str, Any]],
    *,
    volume_dtype: np.dtype = np.dtype("float32"),
) -> None:
    """Write ``volume.elf`` sections (electron localisation function)."""
    for idx, (label, (data, origin, span)) in enumerate(elf_data.items()):
        vol = _real_array_for_artifact(
            data,
            dtype=volume_dtype,
            label=f"volume.elf[{label!r}]",
        )
        _require_3d_volume(vol, "volume.elf", label)
        section_id = f"vol_elf_{idx}"
        slug = _slug(label, fallback=section_id)
        path_in_zip = f"volumes/{slug}_elf.dat"
        file_member = _write_binary_to_zip(zf, path_in_zip, vol)
        origin_arr = np.asarray(origin, dtype=np.float64)
        span_arr = np.asarray(span, dtype=np.float64).reshape(3, 3)
        grid = _grid_descriptor(vol, origin_arr, span_arr)
        grid_json = safe_json_bytes(grid)
        grid_path = f"volumes/{slug}_elf_grid.json"
        zf.writestr(grid_path, grid_json)
        section: dict[str, Any] = {
            "id": section_id,
            "kind": "volume.elf",
            "label": label,
            "members": {
                "data": file_member,
                "grid": {
                    "path": grid_path,
                    "format": "json",
                    "sha256": _sha256_hex(grid_json),
                },
            },
        }
        sections.append(section)


# -- volume.generic -------------------------------------------------------


def _write_volume_generic_section(
    zf: zipfile.ZipFile,
    gen_data: dict[str, tuple],
    sections: list[dict[str, Any]],
    *,
    volume_dtype: np.dtype = np.dtype("float32"),
) -> None:
    """Write ``volume.generic`` sections.

    Escape hatch for any scalar field that doesn't fit one of the
    purpose-built kinds (density / orbital / spin / elf / difference).
    Producers should prefer a more specific kind when one applies; the
    viewer renders this with the same isosurface machinery but cannot
    apply kind-specific defaults (colormap, sign convention, etc.).

    ``gen_data`` is ``{label: (data_3d, origin_bohr, span_bohr)}`` --
    same shape as ``volume.density``.
    """
    for idx, (label, (data, origin, span)) in enumerate(gen_data.items()):
        vol = _real_array_for_artifact(
            data,
            dtype=volume_dtype,
            label=f"volume.generic[{label!r}]",
        )
        _require_3d_volume(vol, "volume.generic", label)
        section_id = f"vol_gen_{idx}"
        slug = _slug(label, fallback=section_id)
        path_in_zip = f"volumes/{slug}_generic.dat"
        file_member = _write_binary_to_zip(zf, path_in_zip, vol)
        origin_arr = np.asarray(origin, dtype=np.float64)
        span_arr = np.asarray(span, dtype=np.float64).reshape(3, 3)
        grid = _grid_descriptor(vol, origin_arr, span_arr)
        grid_json = safe_json_bytes(grid)
        grid_path = f"volumes/{slug}_generic_grid.json"
        zf.writestr(grid_path, grid_json)
        section: dict[str, Any] = {
            "id": section_id,
            "kind": "volume.generic",
            "label": label,
            "members": {
                "data": file_member,
                "grid": {
                    "path": grid_path,
                    "format": "json",
                    "sha256": _sha256_hex(grid_json),
                },
            },
        }
        sections.append(section)


# -- volume.difference ----------------------------------------------------


def _write_volume_difference_section(
    zf: zipfile.ZipFile,
    diff_data: dict[str, Any],
    sections: list[dict[str, Any]],
    *,
    volume_dtype: np.dtype = np.dtype("float32"),
) -> None:
    """Write ``volume.difference`` sections.

    ``diff_data`` is ``{label: spec}``, where ``spec`` is either:

    * ``(data_3d, origin_bohr, span_bohr)`` -- no operand metadata.
    * a dict with keys ``data``, ``origin``, ``span``, and optionally
      ``operand_a`` (str, section id of the minuend), ``operand_b``
      (str, section id of the subtrahend), ``description`` (str). If
      one operand is given the other is required (schema's
      ``dependentRequired``).

    Sign convention: ``data = r(operand_a) - r(operand_b)``.
    """
    for idx, (label, spec) in enumerate(diff_data.items()):
        if isinstance(spec, dict):
            data = spec["data"]
            origin = spec["origin"]
            span = spec["span"]
            operand_a = spec.get("operand_a")
            operand_b = spec.get("operand_b")
            description = spec.get("description")
        else:
            data, origin, span = spec
            operand_a = operand_b = description = None

        vol = _real_array_for_artifact(
            data,
            dtype=volume_dtype,
            label=f"volume.difference[{label!r}]",
        )
        _require_3d_volume(vol, "volume.difference", label)
        section_id = f"vol_diff_{idx}"
        slug = _slug(label, fallback=section_id)
        path_in_zip = f"volumes/{slug}_diff.dat"
        file_member = _write_binary_to_zip(zf, path_in_zip, vol)
        origin_arr = np.asarray(origin, dtype=np.float64)
        span_arr = np.asarray(span, dtype=np.float64).reshape(3, 3)
        grid = _grid_descriptor(vol, origin_arr, span_arr)
        grid_json = safe_json_bytes(grid)
        grid_path = f"volumes/{slug}_diff_grid.json"
        zf.writestr(grid_path, grid_json)
        section: dict[str, Any] = {
            "id": section_id,
            "kind": "volume.difference",
            "label": label,
            "members": {
                "data": file_member,
                "grid": {
                    "path": grid_path,
                    "format": "json",
                    "sha256": _sha256_hex(grid_json),
                },
            },
        }
        if operand_a is not None:
            section["operand_a"] = str(operand_a)
        if operand_b is not None:
            section["operand_b"] = str(operand_b)
        if description is not None:
            section["description"] = str(description)
        sections.append(section)


# -- volume.potential ----------------------------------------------------


def _write_volume_potential_section(
    zf: zipfile.ZipFile,
    potential_data: dict[str, tuple],
    sections: list[dict[str, Any]],
    *,
    volume_dtype: np.dtype = np.dtype("float32"),
) -> None:
    """Write ``volume.potential`` sections (QVF spec 4.10).

    Member structure is identical to ``volume.density``:
    ``{label: (data_3d, origin_bohr, span_bohr)}``.
    Origin and span are in bohr; grid descriptor in bohr.
    """
    for idx, (label, (data, origin, span)) in enumerate(potential_data.items()):
        vol = _real_array_for_artifact(
            data,
            dtype=volume_dtype,
            label=f"volume.potential[{label!r}]",
        )
        _require_3d_volume(vol, "volume.potential", label)
        section_id = f"vol_esp_{idx}"
        slug = _slug(label, fallback=section_id)
        path_in_zip = f"volumes/{slug}.dat"
        file_member = _write_binary_to_zip(zf, path_in_zip, vol)
        origin_arr = np.asarray(origin, dtype=np.float64)
        span_arr = np.asarray(span, dtype=np.float64).reshape(3, 3)
        grid = _grid_descriptor(vol, origin_arr, span_arr)
        grid_json = safe_json_bytes(grid)
        grid_path = f"volumes/{slug}_grid.json"
        zf.writestr(grid_path, grid_json)
        section: dict[str, Any] = {
            "id": section_id,
            "kind": "volume.potential",
            "label": label,
            "members": {
                "data": file_member,
                "grid": {
                    "path": grid_path,
                    "format": "json",
                    "sha256": _sha256_hex(grid_json),
                },
            },
        }
        sections.append(section)


# -- volume.rdg ----------------------------------------------------------


def _write_volume_rdg_section(
    zf: zipfile.ZipFile,
    rdg_data: dict[str, tuple],
    sections: list[dict[str, Any]],
    *,
    volume_dtype: np.dtype = np.dtype("float32"),
) -> None:
    """Write ``volume.rdg`` sections (QVF spec 4.11).

    Member structure is identical to ``volume.density``:
    ``{label: (data_3d, origin_bohr, span_bohr)}``.
    Data is dimensionless RDG = |grad rho| / (2*(3*pi^2)^(1/3) * rho^(4/3)).
    """
    for idx, (label, (data, origin, span)) in enumerate(rdg_data.items()):
        vol = _real_array_for_artifact(
            data,
            dtype=volume_dtype,
            label=f"volume.rdg[{label!r}]",
        )
        _require_3d_volume(vol, "volume.rdg", label)
        section_id = f"vol_rdg_{idx}"
        slug = _slug(label, fallback=section_id)
        path_in_zip = f"volumes/{slug}.dat"
        file_member = _write_binary_to_zip(zf, path_in_zip, vol)
        origin_arr = np.asarray(origin, dtype=np.float64)
        span_arr = np.asarray(span, dtype=np.float64).reshape(3, 3)
        grid = _grid_descriptor(vol, origin_arr, span_arr)
        grid_json = safe_json_bytes(grid)
        grid_path = f"volumes/{slug}_grid.json"
        zf.writestr(grid_path, grid_json)
        section: dict[str, Any] = {
            "id": section_id,
            "kind": "volume.rdg",
            "label": label,
            "members": {
                "data": file_member,
                "grid": {
                    "path": grid_path,
                    "format": "json",
                    "sha256": _sha256_hex(grid_json),
                },
            },
        }
        sections.append(section)


# -- wavefunction.gto -----------------------------------------------------


def _write_wavefunction_gto_section(
    zf: zipfile.ZipFile,
    wf_data: dict[str, Any],
    sections: list[dict[str, Any]],
    *,
    section_id: str = "wf",
) -> None:
    """Write the ``wavefunction.gto`` section.

    ``wf_data`` is a dict with keys:

    * ``basis`` -- list of shell dicts, each with ``center`` (int, 0-based),
      ``l`` (int), ``exponents`` (list of float), ``coefficients``
      (list of list of float, [n_prim, n_gen] for general contraction).
    * ``mo_metadata`` -- dict with ``spin`` ("restricted" or "unrestricted"),
      ``orbital_kind``, ``occupation_semantics``, ``n_ao``, and either
      top-level ``energies``, ``occupations`` (restricted) or
      ``alpha``/``beta`` sub-dicts (unrestricted).
    * ``mo_coefficients`` -- 2D ||(`[n_mo, n_ao]`, float64) for restricted,
      or 3D ||(`[n_mo, n_ao, 2]`, float64) with last axis `[real, imag]`.
    * ``mo_coefficients_alpha``, ``mo_coefficients_beta`` -- for unrestricted.
    * ``structure_ref`` -- str, section id of the structure (default
      ``"structure"``).
    * ``pure`` -- bool, whether spherical harmonics are used (default True).

    ``section_id`` overrides the section id in the manifest (default ``"wf"``).
    Use ``"wf_localized"`` for a second wavefunction section carrying localized
    orbitals beside the canonical set.

    Covers molecular wavefunctions, Γ crystalline orbitals, and one selected
    complex Bloch k-point. The optional ``mo_metadata["k_point"]`` records the
    reciprocal fractional coordinate; complex coefficients are stored as
    real/imag component pairs.
    """
    # Derive a subdirectory from the section id so two wavefunction
    # sections (canonical + localized) don't collide on zip paths.
    subdir = "wavefunction" if section_id == "wf" else f"wavefunction_{section_id}"

    # --- basis JSON ----------------------------------------------------
    basis_shells = wf_data.get("basis", [])
    structure_ref = wf_data.get("structure_ref", "structure")
    pure = wf_data.get("pure", True)
    if "n_ao" in wf_data:
        n_ao = int(wf_data["n_ao"])
    else:
        n_ao = sum(
            (2 * sh["l"] + 1) if pure else ((sh["l"] + 1) * (sh["l"] + 2) // 2)
            for sh in basis_shells
        )

    basis_dict: dict[str, Any] = {
        "structure_ref": structure_ref,
        "pure": pure,
        "n_ao": n_ao,
        "shells": basis_shells,
    }
    basis_json = safe_json_bytes(basis_dict, indent=2)
    basis_path = f"{subdir}/basis.json"
    zf.writestr(basis_path, basis_json)

    # --- mo_metadata JSON -----------------------------------------------
    mo_meta = dict(wf_data.get("mo_metadata", {}))
    mo_meta.setdefault("n_ao", n_ao)
    mo_meta_json = safe_json_bytes(mo_meta, indent=2)
    mo_meta_path = f"{subdir}/mo_metadata.json"
    zf.writestr(mo_meta_path, mo_meta_json)

    # --- mo_coefficients binary -----------------------------------------
    members: dict[str, Any] = {
        "basis": {
            "path": basis_path,
            "format": "json",
            "sha256": _sha256_hex(basis_json),
        },
        "mo_metadata": {
            "path": mo_meta_path,
            "format": "json",
            "sha256": _sha256_hex(mo_meta_json),
        },
    }

    spin = mo_meta.get("spin", "restricted")
    if spin == "unrestricted":
        for tag in ("mo_coefficients_alpha", "mo_coefficients_beta"):
            coeff = wf_data.get(tag)
            if coeff is not None:
                arr, _encoding = _mo_coefficients_for_artifact(
                    coeff,
                    label=f"wavefunction.gto {tag}",
                )
                if arr.ndim in (2, 3):
                    path = f"{subdir}/{tag}.dat"
                    members[tag] = _write_binary_to_zip(zf, path, arr)
    else:
        coeff = wf_data.get("mo_coefficients")
        if coeff is not None:
            arr, _encoding = _mo_coefficients_for_artifact(
                coeff,
                label="wavefunction.gto mo_coefficients",
            )
            if arr.ndim in (2, 3):
                path = f"{subdir}/mo_coefficients.dat"
                members["mo_coefficients"] = _write_binary_to_zip(zf, path, arr)

    section: dict[str, Any] = {
        "id": section_id,
        "kind": "wavefunction.gto",
        "members": members,
    }
    sections.append(section)


def _write_bloch_wavefunction_section(
    zf: zipfile.ZipFile,
    wf_data: dict[str, Any],
    sections: list[dict[str, Any]],
    *,
    section_id: str = "wf_bloch_kpoints",
) -> None:
    """Write the vibe-qc all-k Bloch wavefunction restart section."""
    subdir = "wavefunction_bloch"

    basis_shells = wf_data.get("basis", [])
    structure_ref = wf_data.get("structure_ref", "structure")
    pure = wf_data.get("pure", True)
    n_ao = int(
        wf_data.get(
            "n_ao",
            sum(
                (2 * sh["l"] + 1)
                if bool(sh.get("pure", pure))
                else ((sh["l"] + 1) * (sh["l"] + 2) // 2)
                for sh in basis_shells
            ),
        )
    )
    basis_dict: dict[str, Any] = {
        "structure_ref": structure_ref,
        "pure": pure,
        "n_ao": n_ao,
        "shells": basis_shells,
    }
    basis_json = safe_json_bytes(basis_dict, indent=2)
    basis_path = f"{subdir}/basis.json"
    zf.writestr(basis_path, basis_json)

    mo_meta = dict(wf_data.get("mo_metadata", {}))
    mo_meta.setdefault("n_ao", n_ao)
    mo_meta_json = safe_json_bytes(mo_meta, indent=2)
    mo_meta_path = f"{subdir}/mo_metadata.json"
    zf.writestr(mo_meta_path, mo_meta_json)

    members: dict[str, Any] = {
        "basis": {
            "path": basis_path,
            "format": "json",
            "sha256": _sha256_hex(basis_json),
        },
        "mo_metadata": {
            "path": mo_meta_path,
            "format": "json",
            "sha256": _sha256_hex(mo_meta_json),
        },
    }

    if "kpoints" in wf_data:
        kpts = np.ascontiguousarray(np.asarray(wf_data["kpoints"], dtype=np.float64))
        members["kpoints"] = _write_binary_to_zip(
            zf,
            f"{subdir}/kpoints.dat",
            kpts,
        )

    if mo_meta.get("restart_kind") == "spin_density_bloch_kpoints":
        blocks = mo_meta["blocks"]
        if not blocks:
            raise ValueError("QVF spin restart requires nonempty density blocks")
        for spin in ("alpha", "beta"):
            payload = wf_data[f"density_{spin}"]
            if len(payload) != len(blocks):
                raise ValueError("QVF spin restart density block counts differ")
            for i, block in enumerate(blocks):
                density = np.asarray(payload[i], dtype=float)
                if density.shape != (n_ao, n_ao, 2):
                    raise ValueError("QVF spin density must have shape [n_ao, n_ao, 2]")
                name = block[f"density_{spin}"]
                members[name] = _write_binary_to_zip(zf, f"{subdir}/{name}.dat", density)
        sections.append({"id": section_id, "kind": QVF_BLOCH_WF_KIND, "members": members})
        return

    coeff_blocks = list(wf_data.get("mo_coefficients", []))
    occ_blocks = list(wf_data.get("occupations", []))
    energy_blocks = list(wf_data.get("mo_energies", []))
    if not (len(coeff_blocks) == len(occ_blocks) == len(energy_blocks)):
        raise ValueError(
            "_write_bloch_wavefunction_section: coefficient, occupation, and "
            "energy block counts must match."
        )
    blocks = mo_meta.get("blocks", [])
    if len(blocks) != len(coeff_blocks):
        raise ValueError(
            "_write_bloch_wavefunction_section: metadata block count does not "
            f"match payload blocks ({len(blocks)} vs {len(coeff_blocks)})."
        )

    for ik, block in enumerate(blocks):
        coeff_name = str(block.get("mo_coefficients", f"mo_coefficients_k{ik}"))
        occ_name = str(block.get("occupations", f"occupations_k{ik}"))
        energy_name = str(block.get("mo_energies", f"mo_energies_k{ik}"))

        coeff = np.ascontiguousarray(
            np.asarray(coeff_blocks[ik], dtype=np.float64)
        )
        if coeff.ndim != 3 or coeff.shape[-1] != 2:
            raise ValueError(
                "_write_bloch_wavefunction_section: coefficient blocks must "
                f"have shape [n_mo, n_ao, 2]; block {ik} has {coeff.shape}."
            )
        members[coeff_name] = _write_binary_to_zip(
            zf,
            f"{subdir}/{coeff_name}.dat",
            coeff,
        )
        members[occ_name] = _write_binary_to_zip(
            zf,
            f"{subdir}/{occ_name}.dat",
            np.ascontiguousarray(np.asarray(occ_blocks[ik], dtype=np.float64)),
        )
        members[energy_name] = _write_binary_to_zip(
            zf,
            f"{subdir}/{energy_name}.dat",
            np.ascontiguousarray(np.asarray(energy_blocks[ik], dtype=np.float64)),
        )

    sections.append(
        {
            "id": section_id,
            "kind": QVF_BLOCH_WF_KIND,
            "members": members,
        }
    )


# -- spectra.raman ---------------------------------------------------------


def _write_spectra_raman_section(
    zf: zipfile.ZipFile,
    raman: dict[str, Any],
    sections: list[dict[str, Any]],
) -> None:
    """Write the ``spectra.raman`` section.

    ``raman`` is a dict with keys like ``frequencies_cm1``,
    ``intensities``, ``broadening``, ``units_x``, ``units_y``, etc.
    Mirrors the IR spectrum JSON format.
    """
    spec = dict(raman)
    spec.setdefault("kind", "spectra.raman")
    spec.setdefault("version", "1.0")
    # Consumer expects {frequencies, intensities}.
    if "frequencies" not in spec:
        spec["frequencies"] = spec.pop("frequencies_cm1", [])
    spec.pop("kind", None)
    spec.pop("version", None)
    spec.pop("units_x", None)
    spec.pop("units_y", None)
    spec.pop("label_x", None)
    spec.pop("label_y", None)
    spec.pop("broadening", None)
    spec_json = safe_json_bytes(spec, indent=2)
    path_in_zip = "spectra/raman.json"
    zf.writestr(path_in_zip, spec_json)
    section: dict[str, Any] = {
        "id": "raman_spec",
        "kind": "spectra.raman",
        "label": "Raman spectrum",
        "members": {
            "spectrum": {
                "path": path_in_zip,
                "format": "json",
                "sha256": _sha256_hex(spec_json),
            },
        },
    }
    sections.append(section)


# -- spectra.uvvis ---------------------------------------------------------


def _write_spectra_uvvis_section(
    zf: zipfile.ZipFile,
    uvvis_data: dict[str, Any],
    sections: list[dict[str, Any]],
) -> None:
    """Write the ``spectra.uvvis`` section.

    ``uvvis_data`` is a dict with keys:

    * ``energies_ev`` -- list of float, excitation energies (eV)
    * ``intensities`` -- list of float, oscillator strengths
    * optionally ``wavelength_nm``, ``broadening``, ``units_x``, ``units_y``
    """
    spec = dict(uvvis_data)
    spec.setdefault("kind", "spectra.uvvis")
    spec.setdefault("version", "1.0")
    # Consumer expects {frequencies, intensities} -- we store energies as
    # "frequencies" in eV (the consumer can convert to nm).
    if "frequencies" not in spec and "energies_ev" in spec:
        spec["frequencies"] = spec.pop("energies_ev")
    if "frequencies" not in spec and "wavelength_nm" in spec:
        import numpy as _np

        wl = _np.asarray(spec.pop("wavelength_nm"), dtype=float)
        # eV ≈ 1240 / l(nm)
        spec["frequencies"] = (1240.0 / wl).tolist()
    spec.pop("kind", None)
    spec.pop("version", None)
    spec.pop("units_x", None)
    spec.pop("units_y", None)
    spec.pop("broadening", None)
    spec_json = safe_json_bytes(spec, indent=2)
    path_in_zip = "spectra/uvvis.json"
    zf.writestr(path_in_zip, spec_json)
    section: dict[str, Any] = {
        "id": "uvvis_spec",
        "kind": "spectra.uvvis",
        "label": "UV-Vis spectrum",
        "members": {
            "spectrum": {
                "path": path_in_zip,
                "format": "json",
                "sha256": _sha256_hex(spec_json),
            },
        },
    }
    sections.append(section)


# -- spectra.ecd -----------------------------------------------------------


def _write_spectra_ecd_section(
    zf: zipfile.ZipFile,
    ecd: dict[str, Any],
    sections: list[dict[str, Any]],
) -> None:
    """Write the ``spectra.ecd`` section (electronic circular dichroism)."""
    spec = dict(ecd)
    spec.setdefault("kind", "spectra.ecd")
    spec.setdefault("version", "1.0")
    if "frequencies" not in spec and "energies_ev" in spec:
        spec["frequencies"] = spec.pop("energies_ev")
    spec.pop("kind", None)
    spec.pop("version", None)
    spec.pop("units_x", None)
    spec.pop("units_y", None)
    spec.pop("broadening", None)
    spec_json = safe_json_bytes(spec, indent=2)
    path_in_zip = "spectra/ecd.json"
    zf.writestr(path_in_zip, spec_json)
    section: dict[str, Any] = {
        "id": "ecd_spec",
        "kind": "spectra.ecd",
        "label": "ECD spectrum",
        "members": {
            "spectrum": {
                "path": path_in_zip,
                "format": "json",
                "sha256": _sha256_hex(spec_json),
            },
        },
    }
    sections.append(section)


# -- spectra.vcd -----------------------------------------------------------


def _write_spectra_vcd_section(
    zf: zipfile.ZipFile,
    vcd: dict[str, Any],
    sections: list[dict[str, Any]],
) -> None:
    """Write the ``spectra.vcd`` section (vibrational circular dichroism)."""
    spec = dict(vcd)
    spec.setdefault("kind", "spectra.vcd")
    spec.setdefault("version", "1.0")
    if "frequencies" not in spec and "frequencies_cm1" in spec:
        spec["frequencies"] = spec.pop("frequencies_cm1")
    spec.pop("kind", None)
    spec.pop("version", None)
    spec.pop("units_x", None)
    spec.pop("units_y", None)
    spec.pop("broadening", None)
    spec_json = safe_json_bytes(spec, indent=2)
    path_in_zip = "spectra/vcd.json"
    zf.writestr(path_in_zip, spec_json)
    section: dict[str, Any] = {
        "id": "vcd_spec",
        "kind": "spectra.vcd",
        "label": "VCD spectrum",
        "members": {
            "spectrum": {
                "path": path_in_zip,
                "format": "json",
                "sha256": _sha256_hex(spec_json),
            },
        },
    }
    sections.append(section)


# -- spectra.nmr -----------------------------------------------------------


def _write_spectra_nmr_section(
    zf: zipfile.ZipFile,
    nmr: dict[str, Any],
    sections: list[dict[str, Any]],
) -> None:
    """Write the ``spectra.nmr`` section.

    ``nmr`` is a dict with keys like ``chemical_shifts``, ``shielding_tensors``,
    ``j_couplings``, ``isotope``, ``reference``, ``solvent``.
    """
    spec = dict(nmr)
    spec.setdefault("kind", "spectra.nmr")
    spec.setdefault("version", "1.0")
    spec.pop("kind", None)
    spec.pop("version", None)
    spec_json = safe_json_bytes(spec, indent=2)
    path_in_zip = "spectra/nmr.json"
    zf.writestr(path_in_zip, spec_json)
    section: dict[str, Any] = {
        "id": "nmr_spec",
        "kind": "spectra.nmr",
        "label": "NMR spectrum",
        "members": {
            "spectrum": {
                "path": path_in_zip,
                "format": "json",
                "sha256": _sha256_hex(spec_json),
            },
        },
    }
    sections.append(section)


# -- spectra.epr -----------------------------------------------------------


def _write_spectra_epr_section(
    zf: zipfile.ZipFile,
    epr: dict[str, Any],
    sections: list[dict[str, Any]],
) -> None:
    """Write the ``spectra.epr`` section (QVF spec § 4.4).

    ``epr`` is a dict carrying EPR parameters, e.g. ``g_tensor`` (with
    ``matrix`` / ``isotropic`` / ``principal``), ``hyperfine`` (list of
    per-nucleus ``a_tensor_mhz`` / ``a_iso_mhz``), and
    ``zero_field_splitting`` (``d_mhz`` / ``e_mhz``). Payload shape is
    intentionally loose in v1, mirroring ``spectra.nmr``. vibe-qc does not
    yet compute EPR; this is a dormant writer (like the phonon / EOS writers)
    driven by the ``epr_data`` context key, present so an external producer's
    format and vibe-view's renderer share one contract.
    """
    spec = dict(epr)
    spec.pop("kind", None)
    spec.pop("version", None)
    spec_json = safe_json_bytes(spec, indent=2)
    path_in_zip = "spectra/epr.json"
    zf.writestr(path_in_zip, spec_json)
    section: dict[str, Any] = {
        "id": "epr_spec",
        "kind": "spectra.epr",
        "label": "EPR spectrum",
        "members": {
            "spectrum": {
                "path": path_in_zip,
                "format": "json",
                "sha256": _sha256_hex(spec_json),
            },
        },
    }
    sections.append(section)


# -- spectra.generic -------------------------------------------------------


def _write_spectra_generic_section(
    zf: zipfile.ZipFile,
    generic: dict[str, Any],
    sections: list[dict[str, Any]],
) -> None:
    """Write a ``spectra.generic`` section.

    ``generic`` is a dict with at least ``frequencies`` and
    ``intensities``, plus any user-defined metadata keys.
    """
    spec = dict(generic)
    spec.setdefault("kind", "spectra.generic")
    spec.setdefault("version", "1.0")
    label = spec.pop("label", "Generic spectrum")
    spec_id = spec.pop("section_id", "gen_spec")
    spec.pop("kind", None)
    spec.pop("version", None)
    spec_json = safe_json_bytes(spec, indent=2)
    # Slug the caller-supplied section_id before it becomes a zip member
    # name -- like the volume labels above -- so a bidi/control char or a
    # path-traversal attempt cannot reach the archive path. See _slug.
    path_in_zip = f"spectra/{_slug(spec_id, fallback='gen_spec')}.json"
    zf.writestr(path_in_zip, spec_json)
    section: dict[str, Any] = {
        "id": spec_id,
        "kind": "spectra.generic",
        "label": label,
        "members": {
            "spectrum": {
                "path": path_in_zip,
                "format": "json",
                "sha256": _sha256_hex(spec_json),
            },
        },
    }
    sections.append(section)


# -- structure.symmetry ----------------------------------------------------


def _write_symmetry_section(
    zf: zipfile.ZipFile,
    sym: dict[str, Any],
    sections: list[dict[str, Any]],
) -> None:
    """Write the ``structure.symmetry`` section.

    ``sym`` is a dict with keys like ``space_group_number``,
    ``space_group_symbol``, ``point_group``, etc. (spglib output).
    """
    payload = dict(sym)
    payload.setdefault("kind", "structure.symmetry")
    payload.setdefault("version", "1.0")
    sym_json = safe_json_bytes(payload, indent=2)
    path_in_zip = "structure/symmetry.json"
    zf.writestr(path_in_zip, sym_json)
    section: dict[str, Any] = {
        "id": "sym0",
        "kind": "structure.symmetry",
        "members": {
            "data": {
                "path": path_in_zip,
                "format": "json",
                "sha256": _sha256_hex(sym_json),
            },
        },
    }
    sections.append(section)


# -- bonds ------------------------------------------------------------------


def _write_bonds_section(
    zf: zipfile.ZipFile,
    bonds: list[tuple[int, int, float]],
    sections: list[dict[str, Any]],
) -> None:
    """Write the ``bonds`` section.

    ``bonds`` is a list of ``(i, j, order)`` tuples. Emitted as one
    JSON member ``bonds`` carrying ``{"pairs": [{"i", "j", "order"}, ...]}``
    -- a small table that doesn't justify a binary blob, and that
    round-trips through every consumer without a custom struct format.
    """
    payload = {
        "pairs": [
            {"i": int(i), "j": int(j), "order": float(order)} for (i, j, order) in bonds
        ],
    }
    bonds_json = safe_json_bytes(payload)
    path_in_zip = "bonds/connectivity.json"
    zf.writestr(path_in_zip, bonds_json)
    section: dict[str, Any] = {
        "id": "bonds0",
        "kind": "bonds",
        "label": "Bond connectivity",
        "members": {
            "bonds": {
                "path": path_in_zip,
                "format": "json",
                "sha256": _sha256_hex(bonds_json),
            },
        },
    }
    sections.append(section)


# -- bond_orders ------------------------------------------------------------


def _write_bond_orders_section(
    zf: zipfile.ZipFile,
    method: str,
    pairs: list[dict[str, Any]],
    sections: list[dict[str, Any]],
) -> None:
    """Write the ``bond_orders`` section.

    ``method`` is the bond-order definition (``"mayer"``, ``"wiberg"``, etc.).
    ``pairs`` is a list of dicts each with ``i``, ``j``, ``order`` and
    optional ``distance_ang``, ``symbol_i``, ``symbol_j``.
    """
    payload: dict[str, Any] = {
        "method": method,
        "pairs": [{k: v for k, v in p.items()} for p in pairs],
    }
    bo_json = safe_json_bytes(payload, indent=2)
    path_in_zip = "bonds/orders.json"
    zf.writestr(path_in_zip, bo_json)
    section: dict[str, Any] = {
        "id": "bond_orders",
        "kind": "bond_orders",
        "members": {
            "bond_orders": {
                "path": path_in_zip,
                "format": "json",
                "sha256": _sha256_hex(bo_json),
            },
        },
    }
    sections.append(section)


# -- topology.qtaim ---------------------------------------------------------


def _write_topology_qtaim_section(
    zf: zipfile.ZipFile,
    critical_points: list[dict[str, Any]],
    *,
    bond_paths: list[dict[str, Any]] | None = None,
    sections: list[dict[str, Any]],
) -> None:
    """Write the ``topology.qtaim`` section.

    ``critical_points`` is a list of dicts each with at minimum ``type``
    (``"bcp"``, ``"rcp"``, ``"ccp"``), ``position`` (3-vector in Angstrom),
    ``rho`` (e/Å³), and ``laplacian`` (e/Å⁵). Optional fields:
    ``ellipticity``, ``atom_pair`` (for BCPs).

    ``bond_paths`` is an optional list of ``{atoms: [i, j], path: [[x,y,z],...]}``
    records for gradient-path polylines.

    The compute module (critical-point search, Hessian evaluation,
    gradient-path tracing) is not yet in vibe-qc; the writer accepts
    pre-computed data so it is usable as soon as the compute side lands.
    """
    payload: dict[str, Any] = {"points": list(critical_points)}
    if bond_paths:
        payload["bond_paths"] = list(bond_paths)

    cp_json = safe_json_bytes(payload, indent=2)
    path_in_zip = "topology/critical_points.json"
    zf.writestr(path_in_zip, cp_json)
    section: dict[str, Any] = {
        "id": "qtaim",
        "kind": "topology.qtaim",
        "members": {
            "critical_points": {
                "path": path_in_zip,
                "format": "json",
                "sha256": _sha256_hex(cp_json),
            },
        },
    }
    sections.append(section)


# -- scf_history ------------------------------------------------------------


def _write_scf_history_section(
    zf: zipfile.ZipFile,
    history: list[dict[str, Any]],
    sections: list[dict[str, Any]],
) -> None:
    """Write the ``scf_history`` section.

    ``history`` is a list of per-iteration records with keys like
    ``iter``, ``energy_eh``, ``delta_e``, ``diis_error``. Emitted as a
    JSON document carrying ``{"iterations": [...]}`` (not JSONL -- the
    canonical schema declares this member's format as ``json`` and a
    JSONL payload isn't a valid JSON document).
    """
    payload = {"iterations": list(history)}
    hist_json = safe_json_bytes(payload)
    path_in_zip = "scf_history/iterations.json"
    zf.writestr(path_in_zip, hist_json)
    section: dict[str, Any] = {
        "id": "scf_hist0",
        "kind": "scf_history",
        "members": {
            "iterations": {
                "path": path_in_zip,
                "format": "json",
                "sha256": _sha256_hex(hist_json),
            },
        },
    }
    sections.append(section)


# -- atom_properties ------------------------------------------------------


def _write_atom_properties_section(
    zf: zipfile.ZipFile,
    pop: Any,
    sections: list[dict[str, Any]],
    *,
    iao_charges: Any = None,
) -> None:
    """Write the ``atom_properties`` section.

    ``pop`` is a :class:`PopulationSummary` with fields
    ``mulliken_atoms``, ``loewdin_atoms``, ``mayer_bonds``,
    ``dipole``, ``errors``. It may be None when the only per-atom
    property available is ``iao_charges``.

    ``iao_charges`` preserves the legacy localization-only charge payload.
    An independently requested analysis additionally supplies a complete
    JSON payload in the separate ``x_vibeqc.iao_analysis`` vendor section.
    """
    section: dict[str, Any] = {
        "id": "props0",
        "kind": "atom_properties",
        "members": {},
    }

    if iao_charges is not None:
        charges = np.asarray(iao_charges, dtype=np.float64).ravel()
        if charges.size:
            member = _write_binary_to_zip(
                zf,
                "atom_properties/iao_charge.bin",
                charges,
            )
            section["members"]["iao_charge"] = member

    if pop is None:
        if section["members"]:
            sections.append(section)
        return

    if pop.mulliken_atoms:
        charges = np.array([row[3] for row in pop.mulliken_atoms], dtype=np.float64)
        member = _write_binary_to_zip(
            zf,
            "atom_properties/mulliken_charge.bin",
            charges,
        )
        section["members"]["mulliken_charge"] = member

    if pop.loewdin_atoms:
        charges = np.array([row[3] for row in pop.loewdin_atoms], dtype=np.float64)
        member = _write_binary_to_zip(
            zf,
            "atom_properties/loewdin_charge.bin",
            charges,
        )
        section["members"]["loewdin_charge"] = member

        hirshfeld_atoms = getattr(pop, "hirshfeld_atoms", None)
        if hirshfeld_atoms:
            charges = np.array([row[3] for row in hirshfeld_atoms], dtype=np.float64)
            member = _write_binary_to_zip(
                zf,
                "atom_properties/hirshfeld_charge.bin",
                charges,
            )
            section["members"]["hirshfeld_charge"] = member

        mulliken_spin_atoms = getattr(pop, "mulliken_spin_atoms", None)
        if mulliken_spin_atoms:
            spin_pop = np.array([row[3] for row in mulliken_spin_atoms], dtype=np.float64)
            member = _write_binary_to_zip(
                zf,
                "atom_properties/spin_population.bin",
                spin_pop,
            )
            section["members"]["spin_population"] = member

    if section["members"]:
        sections.append(section)


# -- trajectory -----------------------------------------------------------


def _write_trajectory_section(
    zf: zipfile.ZipFile,
    frames: Sequence[Any],
    sections: list[dict[str, Any]],
    *,
    energies: Optional[Sequence[float]] = None,
    rms_grad: Optional[Sequence[float]] = None,
    trajectory_type: str = "geometry_optimization",
) -> None:
    """Write the ``trajectory`` section."""
    n_steps = len(frames)
    # Metadata JSON
    first_atoms = list(frames[0].atoms)
    meta_atoms = [
        {
            "symbol": _symbol(int(a.Z)),
            "position": [0.0, 0.0, 0.0],
            "atomic_number": int(a.Z),
        }
        for a in first_atoms
    ]
    meta = {"atoms": meta_atoms}
    if energies is not None:
        meta["energies"] = [float(e) for e in energies]
    meta_json = safe_json_bytes(meta)
    meta_path = "trajectory/metadata.json"
    zf.writestr(meta_path, meta_json)

    # Coords binary: (n_steps, n_atoms, 3) float64 in Å.
    n_atoms_f = len(first_atoms)
    coords = np.zeros((n_steps, n_atoms_f, 3), dtype=np.float64)
    for i, mol in enumerate(frames):
        for j, a in enumerate(mol.atoms):
            coords[i, j, 0] = float(a.xyz[0]) * _BOHR_TO_ANGSTROM
            coords[i, j, 1] = float(a.xyz[1]) * _BOHR_TO_ANGSTROM
            coords[i, j, 2] = float(a.xyz[2]) * _BOHR_TO_ANGSTROM
    coords_member = _write_binary_to_zip(
        zf,
        "trajectory/coords.bin",
        coords,
    )

    section: dict[str, Any] = {
        "id": "traj0",
        "kind": "trajectory",
        "members": {
            "metadata": {
                "path": meta_path,
                "format": "json",
                "sha256": _sha256_hex(meta_json),
            },
            "coords": coords_member,
        },
    }
    sections.append(section)


# -- reaction.path / reaction.waypoints -----------------------------------


_WAYPOINT_KINDS = frozenset(
    {"reactant", "transition_state", "intermediate", "product", "point"}
)


def _validate_waypoints(
    waypoints: Sequence[dict[str, Any]],
    n_frames: int,
    *,
    context_label: str,
) -> list[dict[str, Any]]:
    """Sanity-check a waypoint list and return a normalised copy.

    Raises ``ValueError`` if a waypoint is missing required fields, the
    ``kind`` is outside the registry, or ``frame_index`` is out of
    ``[0, n_frames)``.
    """
    if not waypoints:
        raise ValueError(
            f"{context_label}: at least one waypoint is required "
            "(reaction.path / reaction.waypoints both need them)"
        )
    out: list[dict[str, Any]] = []
    for i, wp in enumerate(waypoints):
        if "frame_index" not in wp or "label" not in wp or "kind" not in wp:
            raise ValueError(
                f"{context_label}: waypoint #{i} must carry "
                "'frame_index', 'label', and 'kind'"
            )
        fi = int(wp["frame_index"])
        if not 0 <= fi < n_frames:
            raise ValueError(
                f"{context_label}: waypoint #{i} frame_index={fi} "
                f"is outside [0, {n_frames})"
            )
        kind = str(wp["kind"])
        if kind not in _WAYPOINT_KINDS:
            raise ValueError(
                f"{context_label}: waypoint #{i} kind={kind!r} is not "
                f"one of {sorted(_WAYPOINT_KINDS)}"
            )
        rec: dict[str, Any] = {
            "frame_index": fi,
            "label": str(wp["label"]),
            "kind": kind,
        }
        if "energy_eh" in wp:
            rec["energy_eh"] = float(wp["energy_eh"])
        out.append(rec)
    return out


def _frame_atoms(frame: Any) -> list[Any]:
    """Pick the populated atom collection on ``frame``.

    Molecule exposes atoms under ``.atoms``; PeriodicSystem under
    ``.unit_cell``. Mirrors the dispatch in the structure-section
    writer so reaction.path frames can be either type.
    """
    atoms = list(getattr(frame, "atoms", []) or [])
    if not atoms:
        atoms = list(getattr(frame, "unit_cell", []) or [])
    return atoms


def _frame_is_periodic(frame: Any) -> bool:
    """A reaction.path frame is periodic iff it carries a lattice."""
    return getattr(frame, "lattice", None) is not None and bool(
        list(getattr(frame, "unit_cell", []) or [])
    )


def _reaction_path_is_periodic(frames: Sequence[Any]) -> bool:
    """True iff every frame in ``frames`` is a periodic system.

    Raises ValueError on a mixed reaction path (molecular + periodic
    interleaved) -- that's malformed input, not a silent fall-through.
    """
    flags = [_frame_is_periodic(f) for f in frames]
    if all(flags):
        return True
    if not any(flags):
        return False
    raise ValueError(
        "reaction.path: mixed molecular/periodic frames are not "
        "supported -- every frame must be the same system type"
    )


def _write_reaction_path_section(
    zf: zipfile.ZipFile,
    frames: Sequence[Any],
    waypoints: Sequence[dict[str, Any]],
    sections: list[dict[str, Any]],
    *,
    energies: Optional[Sequence[float]] = None,
    reaction_coordinate: Optional[Sequence[float]] = None,
    reaction_coordinate_label: Optional[str] = None,
    reaction_coordinate_unit: Optional[str] = None,
    frame_volumes: Optional[np.ndarray] = None,
    volume_grid: Optional[dict[str, Any]] = None,
    volume_frame_index: Optional[Sequence[int]] = None,
    volume_label: str = "Electron density",
    volume_isovalue: Optional[float] = None,
    volume_dtype: np.dtype = np.dtype("float32"),
) -> None:
    """Write a self-contained ``reaction.path`` section.

    Binary layout matches ``trajectory`` (coords float64 [n_frames,
    n_atoms, 3] in Å) so the same readers decode coords. The waypoint
    annotations live in the JSON metadata member.

    Frames may be ``Molecule`` (atoms under ``.atoms``) or
    ``PeriodicSystem`` (atoms under ``.unit_cell``; lattice + dim
    available). Periodic reaction paths add a binary ``lattice``
    member (float64, columns = a, b, c, in bohr -- matching
    ``PeriodicSystem.lattice``) and a ``dim`` integer in the metadata
    JSON. A shared lattice across all frames is stored once as shape
    [3, 3]; per-frame lattices (forward-compat with variable-cell)
    are stored as shape [n_frames, 3, 3]. Mixed molecular/periodic
    reaction paths are rejected at write time.
    """
    n_frames = len(frames)
    if n_frames == 0:
        raise ValueError("reaction.path: frames is empty")

    is_periodic = _reaction_path_is_periodic(frames)

    norm_wps = _validate_waypoints(waypoints, n_frames, context_label="reaction.path")

    first_atoms = _frame_atoms(frames[0])
    if not first_atoms:
        raise ValueError("reaction.path: first frame has no atoms")
    meta_atoms = [
        {
            "symbol": _symbol(int(a.Z)),
            "position": [0.0, 0.0, 0.0],
            "atomic_number": int(a.Z),
        }
        for a in first_atoms
    ]
    meta: dict[str, Any] = {"atoms": meta_atoms, "waypoints": norm_wps}
    if energies is not None:
        meta["energies"] = [float(e) for e in energies]
    if reaction_coordinate is not None:
        meta["reaction_coordinate"] = [float(x) for x in reaction_coordinate]
    # Optional human-readable axis annotations for the energy plot.
    # Additive + optional: v1/v2 readers that don't know them ignore
    # them, so this needs no qvf_version bump.
    if reaction_coordinate_label is not None:
        meta["reaction_coordinate_label"] = str(reaction_coordinate_label)
    if reaction_coordinate_unit is not None:
        meta["reaction_coordinate_unit"] = str(reaction_coordinate_unit)

    # v2: pull lattice + dim off periodic frames before writing the
    # metadata JSON so its sha256 includes them.
    lattice_array: Optional[np.ndarray] = None
    if is_periodic:
        lattices = [np.asarray(f.lattice, dtype=np.float64) for f in frames]
        for L in lattices:
            if L.shape != (3, 3):
                raise ValueError(
                    "reaction.path: PeriodicSystem.lattice must be "
                    f"3x3; got shape {L.shape}"
                )
        all_equal = all(np.allclose(L, lattices[0]) for L in lattices[1:])
        if all_equal:
            lattice_array = lattices[0]
        else:
            lattice_array = np.stack(lattices, axis=0)
        dims = [int(getattr(f, "dim", 3) or 3) for f in frames]
        if len(set(dims)) == 1:
            meta["dim"] = dims[0]
        else:
            meta["dim_per_frame"] = dims

    # Optional per-frame volumetric data (W1). A 4D array
    # [n_emitted, nx, ny, nz] of scalar field values morphing along the
    # path, plus a shared grid descriptor and an index map saying which
    # path frame each emitted slab corresponds to (so callers can
    # decimate -- emit every Nth frame's cube without claiming a cube for
    # every frame). Validated here so the sha256 includes the metadata.
    vol_array: Optional[np.ndarray] = None
    if frame_volumes is not None:
        vol_array = _real_array_for_artifact(
            frame_volumes,
            dtype=volume_dtype,
            label="reaction.path frame_volumes",
        )
        if vol_array.ndim != 4:
            raise ValueError(
                "reaction.path: frame_volumes must be 4D "
                f"[n_emitted, nx, ny, nz]; got shape {vol_array.shape}"
            )
        if volume_grid is None:
            raise ValueError(
                "reaction.path: frame_volumes requires volume_grid "
                "(an {origin, voxel_vectors, shape} descriptor)"
            )
        n_emit = vol_array.shape[0]
        if volume_frame_index is None:
            vfi = list(range(n_emit))
        else:
            vfi = [int(i) for i in volume_frame_index]
        if len(vfi) != n_emit:
            raise ValueError(
                "reaction.path: volume_frame_index length "
                f"({len(vfi)}) must equal n_emitted ({n_emit})"
            )
        for i in vfi:
            if not 0 <= i < n_frames:
                raise ValueError(
                    f"reaction.path: volume_frame_index entry {i} out of "
                    f"range [0, {n_frames})"
                )
        grid_shape = list(int(x) for x in volume_grid["shape"])
        if list(vol_array.shape[1:]) != grid_shape:
            raise ValueError(
                "reaction.path: frame_volumes grid dims "
                f"{list(vol_array.shape[1:])} disagree with volume_grid "
                f"shape {grid_shape}"
            )
        meta["volume_frame_index"] = vfi
        meta["volume_label"] = str(volume_label)
        if volume_isovalue is not None:
            meta["volume_isovalue"] = float(volume_isovalue)

    meta_json = safe_json_bytes(meta)
    meta_path = "reaction/metadata.json"
    zf.writestr(meta_path, meta_json)

    n_atoms_f = len(first_atoms)
    coords = np.zeros((n_frames, n_atoms_f, 3), dtype=np.float64)
    for i, frame in enumerate(frames):
        frame_atoms = _frame_atoms(frame)
        if len(frame_atoms) != n_atoms_f:
            raise ValueError(
                f"reaction.path: frame {i} has {len(frame_atoms)} "
                f"atoms; expected {n_atoms_f} (matching frame 0)"
            )
        for j, a in enumerate(frame_atoms):
            coords[i, j, 0] = float(a.xyz[0]) * _BOHR_TO_ANGSTROM
            coords[i, j, 1] = float(a.xyz[1]) * _BOHR_TO_ANGSTROM
            coords[i, j, 2] = float(a.xyz[2]) * _BOHR_TO_ANGSTROM
    coords_member = _write_binary_to_zip(zf, "reaction/coords.bin", coords)

    members: dict[str, Any] = {
        "metadata": {
            "path": meta_path,
            "format": "json",
            "sha256": _sha256_hex(meta_json),
        },
        "coords": coords_member,
    }
    if lattice_array is not None:
        members["lattice"] = _write_binary_to_zip(
            zf, "reaction/lattice.bin", lattice_array
        )
    if vol_array is not None:
        assert volume_grid is not None  # guarded above
        members["frame_volumes"] = _write_binary_to_zip(
            zf, "reaction/frame_volumes.bin", vol_array
        )
        grid_json = safe_json_bytes(volume_grid)
        grid_path = "reaction/volume_grid.json"
        zf.writestr(grid_path, grid_json)
        members["volume_grid"] = {
            "path": grid_path,
            "format": "json",
            "sha256": _sha256_hex(grid_json),
        }

    section: dict[str, Any] = {
        "id": "rxn0",
        "kind": "reaction.path",
        "members": members,
    }
    sections.append(section)


def write_reaction_path_qvf(
    stem: "os.PathLike | str",
    *,
    frames: Sequence[Any],
    energies: Sequence[float],
    waypoints: Sequence[dict[str, Any]],
    reaction_coordinate: Optional[Sequence[float]] = None,
    reaction_coordinate_label: Optional[str] = None,
    reaction_coordinate_unit: Optional[str] = None,
    frame_volumes: Optional[np.ndarray] = None,
    volume_grid: Optional[dict[str, Any]] = None,
    volume_frame_index: Optional[Sequence[int]] = None,
    volume_label: str = "Electron density",
    volume_isovalue: Optional[float] = None,
    method: str,
    basis: str,
    functional: Optional[str] = None,
    extra_assemble_kwargs: Optional[dict[str, Any]] = None,
    compression: Optional[int] = None,
) -> Path:
    """High-level helper: emit a vibe-view reaction.path archive.

    Wraps :func:`write_qvf` for the common pattern shared by
    :meth:`vibeqc.NEBResult.write_qvf` and
    :meth:`vibeqc.ScanResult.write_qvf`: a structure section
    (reactant geometry -- first frame), a reaction.path section
    (frames + waypoints + energies + reaction coordinate), and a
    citations section (BibTeX assembled with the caller's flags).

    Periodic vs molecular dispatch is automatic: if
    ``frames[0]`` is a :class:`PeriodicSystem` the writer detects
    it (via ``_reaction_path_is_periodic``) and bumps the manifest
    to QVF v2 with the per-frame lattice + dim -- no extra knobs
    needed.

    Parameters
    ----------
    stem
        Path stem; ``.qvf`` is appended.
    frames
        Sequence of :class:`Molecule` or :class:`PeriodicSystem`,
        one per image. All frames must be the same type; mixed
        molecular/periodic raises in the lower-level writer.
    energies
        Per-frame energies in Hartree, ``len == len(frames)``.
    waypoints
        Iterable of
        ``{frame_index, label, kind, energy_eh?}`` dicts.
        ``kind`` is one of ``"reactant" | "transition_state" |
        "intermediate" | "product" | "point"``.
    reaction_coordinate
        Per-frame coordinate values; whatever the caller wants
        the x-axis of the energy plot to show -- arc length for
        NEB (normalised 0-1), bond length / angle for a relaxed
        scan, etc. ``None`` => the plot uses frame indices.
    reaction_coordinate_label, reaction_coordinate_unit
        Optional human-readable name + unit for the reaction
        coordinate (e.g. ``"O-H distance"`` / ``"bohr"``). The
        viewer labels the energy-plot x-axis ``"{label} ({unit})"``
        when present, else falls back to ``"Reaction coordinate"``.
        Additive optional metadata -- no ``qvf_version`` bump.
    method, basis, functional
        SCF flavour for the OutputPlan + citation routing.
        ``functional`` is None for HF methods.
    extra_assemble_kwargs
        Forwarded to ``CitationDatabase.assemble`` --
        e.g. ``{"uses_neb": True, "uses_ci_neb": True}`` for a
        climbing-image NEB run. Use this to fire driver-specific
        citation routes; the per-image SCF citations (method /
        basis / functional) fire automatically from the args
        above.
    compression
        Optional ``zipfile`` compression constant; ``None`` =>
        the writer's default.

    Returns
    -------
    pathlib.Path
        The on-disk ``{stem}.qvf`` path.
    """
    from ..citations.bibtex import format_bibtex
    from ..citations.registry import load_default_database
    from ..plan import OutputPlan as _OutputPlan

    frames_list = list(frames)
    if not frames_list:
        raise ValueError("write_reaction_path_qvf: frames is empty")

    is_periodic = _reaction_path_is_periodic(frames_list)
    first = frames_list[0]

    rc: Optional[list[float]] = None
    if reaction_coordinate is not None:
        rc = [float(v) for v in reaction_coordinate]
        if len(rc) != len(frames_list):
            raise ValueError(
                f"write_reaction_path_qvf: reaction_coordinate length "
                f"({len(rc)}) must equal n_frames ({len(frames_list)})"
            )

    energy_list = [float(e) for e in energies]
    if len(energy_list) != len(frames_list):
        raise ValueError(
            f"write_reaction_path_qvf: energies length "
            f"({len(energy_list)}) must equal n_frames "
            f"({len(frames_list)})"
        )

    # OutputPlan needs a string method/basis. Caller is responsible
    # for not passing None there -- citations + provenance get
    # garbled otherwise.
    plan = _OutputPlan.from_run_job_kwargs(
        output=stem,
        method=method,
        basis=basis,
        functional=functional,
        job_kind="periodic_scf" if is_periodic else "molecular_scf",
        output_qvf=True,
        citations=True,
        write_xyz=False,
        write_molden_file=False,
        write_population=False,
    )

    # Stub SCF result carrying the first frame's energy -- enough
    # for the structure-section writer; the reaction.path section
    # carries its own per-image energies.
    stub_result = type(
        "_ReactionPathStubResult",
        (),
        {
            "converged": True,
            "energy": energy_list[0],
        },
    )()

    # Assemble citations. The method / basis / functional routes
    # always fire; extra_assemble_kwargs fires driver-level routes
    # (e.g. uses_neb / uses_ci_neb).
    db = load_default_database()
    asm_kwargs: dict[str, Any] = dict(extra_assemble_kwargs or {})
    asm_kwargs.setdefault("method", method.lower())
    asm_kwargs.setdefault("basis", basis)
    if functional is not None:
        asm_kwargs.setdefault("functional", functional)
    asm_kwargs.setdefault("periodic", is_periodic)
    assembled = db.assemble(**asm_kwargs)
    bibtex_content = format_bibtex(assembled)

    context: dict[str, Any] = {
        "method": method,
        "basis": basis,
        "functional": functional,
        "result": stub_result,
        "reaction_path": {
            "frames": frames_list,
            "waypoints": list(waypoints),
            "energies": energy_list,
        },
        "bibtex_content": bibtex_content,
    }
    if rc is not None:
        context["reaction_path"]["reaction_coordinate"] = rc
    if reaction_coordinate_label is not None:
        context["reaction_path"]["reaction_coordinate_label"] = str(
            reaction_coordinate_label
        )
    if reaction_coordinate_unit is not None:
        context["reaction_path"]["reaction_coordinate_unit"] = str(
            reaction_coordinate_unit
        )
    if frame_volumes is not None:
        context["reaction_path"]["frame_volumes"] = frame_volumes
        context["reaction_path"]["volume_grid"] = volume_grid
        context["reaction_path"]["volume_frame_index"] = volume_frame_index
        context["reaction_path"]["volume_label"] = volume_label
        if volume_isovalue is not None:
            context["reaction_path"]["volume_isovalue"] = volume_isovalue
    if is_periodic:
        context["system"] = first
    else:
        context["molecule"] = first

    kwargs: dict[str, Any] = {}
    if compression is not None:
        kwargs["compression"] = compression
    return write_qvf(stem, plan, **context, **kwargs)


def _write_scan_surface_section(
    zf: zipfile.ZipFile,
    sections: list[dict[str, Any]],
    *,
    axis_a: Sequence[float],
    axis_b: Sequence[float],
    energies: np.ndarray,
    coordinate_a_label: str,
    coordinate_a_unit: str,
    coordinate_b_label: str,
    coordinate_b_unit: str,
    geometries: Optional[np.ndarray] = None,
    atoms: Optional[Sequence[dict[str, Any]]] = None,
) -> None:
    """Write a self-contained ``scan.surface`` section (2D energy grid).

    ``energies`` is ``[nA, nB]`` (Hartree); ``axis_a`` / ``axis_b`` are
    the 1D driven-coordinate values. Optional ``geometries`` is the
    relaxed structure at each node, flattened ``[nA*nB, n_atoms, 3]``
    in Å, row-major over ``(a, b)``.
    """
    e = np.asarray(energies, dtype=np.float64)
    if e.ndim != 2:
        raise ValueError(f"scan.surface: energies must be 2D; got {e.shape}")
    n_a, n_b = e.shape
    a_arr = np.asarray(list(axis_a), dtype=np.float64)
    b_arr = np.asarray(list(axis_b), dtype=np.float64)
    if a_arr.shape != (n_a,):
        raise ValueError(f"scan.surface: axis_a length {a_arr.shape} != nA={n_a}")
    if b_arr.shape != (n_b,):
        raise ValueError(f"scan.surface: axis_b length {b_arr.shape} != nB={n_b}")

    meta: dict[str, Any] = {
        "shape": [int(n_a), int(n_b)],
        "coordinate_a_label": str(coordinate_a_label),
        "coordinate_a_unit": str(coordinate_a_unit),
        "coordinate_b_label": str(coordinate_b_label),
        "coordinate_b_unit": str(coordinate_b_unit),
    }
    if geometries is not None and atoms is not None:
        meta["atoms"] = list(atoms)

    meta_json = safe_json_bytes(meta)
    meta_path = "scan_surface/metadata.json"
    zf.writestr(meta_path, meta_json)

    members: dict[str, Any] = {
        "metadata": {
            "path": meta_path,
            "format": "json",
            "sha256": _sha256_hex(meta_json),
        },
        "axis_a": _write_binary_to_zip(zf, "scan_surface/axis_a.bin", a_arr),
        "axis_b": _write_binary_to_zip(zf, "scan_surface/axis_b.bin", b_arr),
        "energies": _write_binary_to_zip(zf, "scan_surface/energies.bin", e),
    }
    if geometries is not None and atoms is not None:
        g = np.asarray(geometries, dtype=np.float64)
        if g.ndim != 3 or g.shape[0] != n_a * n_b:
            raise ValueError(
                "scan.surface: geometries must be "
                f"[nA*nB, n_atoms, 3]; got {g.shape} (nA*nB={n_a * n_b})"
            )
        members["geometries"] = _write_binary_to_zip(
            zf, "scan_surface/geometries.bin", g
        )

    sections.append({"id": "scan0", "kind": "scan.surface", "members": members})


def write_scan_surface_qvf(
    stem: "os.PathLike | str",
    *,
    axis_a: Sequence[float],
    axis_b: Sequence[float],
    energies: np.ndarray,
    coordinate_a_label: str,
    coordinate_a_unit: str,
    coordinate_b_label: str,
    coordinate_b_unit: str,
    structure: Any,
    geometries: Optional[np.ndarray] = None,
    method: str,
    basis: str,
    functional: Optional[str] = None,
    compression: Optional[int] = None,
) -> Path:
    """High-level helper: emit a vibe-view ``scan.surface`` archive for
    a 2D relaxed scan -- a structure section (the reference geometry), a
    scan.surface section (the energy grid + axes), and a citations
    section. Mirrors :func:`write_reaction_path_qvf`.
    """
    from ..citations.bibtex import format_bibtex
    from ..citations.registry import load_default_database
    from ..plan import OutputPlan as _OutputPlan

    is_periodic = _frame_is_periodic(structure)
    ref_mol = structure.unit_cell_molecule() if is_periodic else structure

    atoms_meta = None
    if geometries is not None:
        atoms_meta = [
            {"symbol": _symbol(int(a.Z)), "atomic_number": int(a.Z)}
            for a in _frame_atoms(structure)
        ]

    plan = _OutputPlan.from_run_job_kwargs(
        output=stem,
        method=method,
        basis=basis,
        functional=functional,
        job_kind="periodic_scf" if is_periodic else "molecular_scf",
        output_qvf=True,
        citations=True,
        write_xyz=False,
        write_molden_file=False,
        write_population=False,
    )
    e0 = float(np.asarray(energies, dtype=np.float64).flat[0])
    stub_result = type(
        "_ScanSurfaceStubResult",
        (),
        {"converged": True, "energy": e0},
    )()

    db = load_default_database()
    asm_kwargs: dict[str, Any] = {
        "method": method.lower(),
        "basis": basis,
        "periodic": is_periodic,
    }
    if functional is not None:
        asm_kwargs["functional"] = functional
    bibtex_content = format_bibtex(db.assemble(**asm_kwargs))

    context: dict[str, Any] = {
        "method": method,
        "basis": basis,
        "functional": functional,
        "result": stub_result,
        "bibtex_content": bibtex_content,
        "scan_surface": {
            "axis_a": list(axis_a),
            "axis_b": list(axis_b),
            "energies": energies,
            "coordinate_a_label": coordinate_a_label,
            "coordinate_a_unit": coordinate_a_unit,
            "coordinate_b_label": coordinate_b_label,
            "coordinate_b_unit": coordinate_b_unit,
            "geometries": geometries,
            "atoms": atoms_meta,
        },
    }
    if is_periodic:
        context["system"] = structure
    else:
        context["molecule"] = ref_mol

    kwargs: dict[str, Any] = {}
    if compression is not None:
        kwargs["compression"] = compression
    return write_qvf(stem, plan, **context, **kwargs)


def _write_reaction_waypoints_section(
    zf: zipfile.ZipFile,
    trajectory_ref: str,
    waypoints: Sequence[dict[str, Any]],
    n_trajectory_frames: int,
    sections: list[dict[str, Any]],
    *,
    reaction_coordinate: Optional[Sequence[float]] = None,
) -> None:
    """Write a ``reaction.waypoints`` section pointing at an existing
    ``trajectory`` section.

    No coords are emitted -- they live in the referenced trajectory.
    The producer is responsible for ensuring ``trajectory_ref`` names
    a section actually present in this archive; the validator checks
    that cross-reference at write time.
    """
    norm_wps = _validate_waypoints(
        waypoints,
        n_trajectory_frames,
        context_label=f"reaction.waypoints(trajectory_ref={trajectory_ref!r})",
    )
    payload: dict[str, Any] = {"waypoints": norm_wps}
    if reaction_coordinate is not None:
        if len(reaction_coordinate) != n_trajectory_frames:
            raise ValueError(
                "reaction.waypoints: reaction_coordinate length "
                f"({len(reaction_coordinate)}) must equal "
                f"n_trajectory_frames ({n_trajectory_frames})"
            )
        payload["reaction_coordinate"] = [float(x) for x in reaction_coordinate]
    wps_json = safe_json_bytes(payload)
    path_in_zip = "reaction/waypoints.json"
    zf.writestr(path_in_zip, wps_json)
    section: dict[str, Any] = {
        "id": "rxn_wp0",
        "kind": "reaction.waypoints",
        "trajectory_ref": str(trajectory_ref),
        "members": {
            "waypoints": {
                "path": path_in_zip,
                "format": "json",
                "sha256": _sha256_hex(wps_json),
            },
        },
    }
    sections.append(section)


# -- vibrations -----------------------------------------------------------


def _write_vibrations_section(
    zf: zipfile.ZipFile,
    hess: Any,
    sections: list[dict[str, Any]],
    *,
    atom_symbols: list[str] | None = None,
    molecule: Any | None = None,
    surface: dict[str, Any] | None = None,
) -> None:
    """Write the ``vibrations`` section.

    ``hess`` is a :class:`HessianResult` with ``frequencies_cm1``,
    ``normal_modes`` (and optionally ``masses_amu``).
    ``atom_symbols`` provides element symbols; ``molecule`` provides the
    equilibrium geometry + atomic numbers for the metadata atoms array.
    ``surface`` names the potential-energy surface the Hessian was built
    on (see :func:`vibeqc.output.hessian_surface_manifest_fields`). It
    matters because run_job resolves a correlated request down to its
    mean-field reference before differentiating it, so an MP2 job's
    frequencies are RHF frequencies -- without this key a consumer
    reading the section cannot tell the two apart.

    Two correctness fixes vs. the original implementation (audit findings
    A5-01/A5-02/A5-05):

    * The metadata atoms array now carries the **real equilibrium
      positions** (Å) and atomic numbers, not ``[0,0,0]`` / ``Z=0``.
      Without this the viewer animated every atom collapsed at the origin.
    * The displacements are now **un-mass-weighted Cartesian** patterns.
      ``HessianResult.normal_modes`` are orthonormal *mass-weighted*
      eigenvectors; the Cartesian motion of atom ``i`` is
      ``modes[3i:3i+3, k] / sqrt(M_i)``. Shipping the raw mass-weighted
      vector made heavy atoms over-move and light atoms under-move
      (a C-H stretch showed the C moving as much as the H). Each mode is
      then normalized to unit maximum atomic displacement so the viewer's
      amplitude slider gives a consistent, visible scale while preserving
      the (now correct) relative per-atom amplitudes.
    """
    freqs = np.asarray(hess.frequencies_cm1, dtype=np.float64)
    n_modes = freqs.shape[0]
    n_atoms = n_modes // 3
    modes = np.asarray(hess.normal_modes, dtype=np.float64)

    # Equilibrium geometry (bohr -> Å) + atomic numbers from the molecule.
    positions_ang: list[list[float]] | None = None
    atomic_numbers: list[int] | None = None
    if molecule is not None:
        try:
            positions_ang = [
                [
                    float(a.xyz[0]) * _BOHR_TO_ANGSTROM,
                    float(a.xyz[1]) * _BOHR_TO_ANGSTROM,
                    float(a.xyz[2]) * _BOHR_TO_ANGSTROM,
                ]
                for a in molecule.atoms
            ]
            atomic_numbers = [int(a.Z) for a in molecule.atoms]
        except Exception:  # noqa: BLE001 -- degrade to zeros if molecule API differs
            positions_ang = None
            atomic_numbers = None

    syms = atom_symbols or ["?"] * n_atoms
    atoms_list = []
    for a_idx in range(n_atoms):
        sym = syms[a_idx] if a_idx < len(syms) else "?"
        pos = (
            positions_ang[a_idx]
            if positions_ang is not None and a_idx < len(positions_ang)
            else [0.0, 0.0, 0.0]
        )
        z = (
            atomic_numbers[a_idx]
            if atomic_numbers is not None and a_idx < len(atomic_numbers)
            else 0
        )
        atoms_list.append({"symbol": sym, "position": pos, "atomic_number": z})
    meta: dict[str, Any] = {
        "frequencies": [float(freqs[p]) for p in range(n_modes)],
        "atoms": atoms_list,
    }
    if surface:
        meta["surface"] = dict(surface)
    meta_json = safe_json_bytes(meta)
    meta_path = "vibrations/metadata.json"
    zf.writestr(meta_path, meta_json)

    # Un-mass-weighting factor 1/sqrt(M_i) per atom (electron-mass units, to
    # match HessianResult.normal_modes). Falls back to 1 when masses are
    # unavailable (e.g. minimal test stubs) so we never crash.
    masses_amu = getattr(hess, "masses_amu", None)
    inv_sqrt_m: np.ndarray | None = None
    if masses_amu is not None:
        m = np.asarray(masses_amu, dtype=np.float64)
        if m.shape == (n_atoms,) and np.all(m > 0):
            _AMU_TO_E = 1822.8884862
            inv_sqrt_m = 1.0 / np.sqrt(m * _AMU_TO_E)

    # Displacements (n_modes, n_atoms, 3): un-mass-weighted Cartesian
    # patterns, each mode normalized to unit max atomic displacement.
    disp = np.zeros((n_modes, n_atoms, 3), dtype=np.float64)
    for p in range(n_modes):
        block = modes[:, p].reshape(n_atoms, 3)
        if inv_sqrt_m is not None:
            block = block * inv_sqrt_m[:, None]
        max_norm = float(np.max(np.linalg.norm(block, axis=1))) if n_atoms else 0.0
        if max_norm > 1e-12:
            block = block / max_norm
        disp[p] = block

    disp_member = _write_binary_to_zip(
        zf,
        "vibrations/displacements.bin",
        disp,
    )

    section: dict[str, Any] = {
        "id": "vib0",
        "kind": "vibrations",
        "members": {
            "metadata": {
                "path": meta_path,
                "format": "json",
                "sha256": _sha256_hex(meta_json),
            },
            "displacements": disp_member,
        },
    }
    sections.append(section)


# -- spectra.ir -----------------------------------------------------------


def _write_spectra_ir_section(
    zf: zipfile.ZipFile,
    hess: Any,
    sections: list[dict[str, Any]],
) -> None:
    """Write the ``spectra.ir`` section."""
    try:
        from ...hessian import ir_intensities

        intensities = np.asarray(ir_intensities(hess), dtype=np.float64)
    except Exception:
        return

    freqs = np.asarray(hess.frequencies_cm1, dtype=np.float64)
    # Only positive (real) frequencies -- skip imaginary and trans/rot.
    mask = freqs > 1.0
    spec = {
        "frequencies": freqs[mask].tolist(),
        "intensities": intensities[mask].tolist(),
    }
    spec_json = safe_json_bytes(spec, indent=2)
    path_in_zip = "spectra/ir.json"
    zf.writestr(path_in_zip, spec_json)
    section: dict[str, Any] = {
        "id": "ir_spec",
        "kind": "spectra.ir",
        "label": "IR spectrum",
        "members": {
            "spectrum": {
                "path": path_in_zip,
                "format": "json",
                "sha256": _sha256_hex(spec_json),
            },
        },
    }
    sections.append(section)


# -- basis.ao -------------------------------------------------------------


def _write_basis_ao_section(
    zf: zipfile.ZipFile,
    ao_data: list[dict[str, Any]],
    sections: list[dict[str, Any]],
    *,
    volume_dtype: np.dtype = np.dtype("float32"),
) -> None:
    """Write ``basis.ao`` sections -- one per atomic orbital.

    ``ao_data`` is a list of dicts from :func:`qvf_ao_data`.  Each
    dict has keys: ``label``, ``data`` (3-D array), ``origin`` (bohr),
    ``span`` (3x3 bohr), ``ao_metadata``, ``section_id``.

    The grid+data member layout is identical to ``volume.orbital``;
    the ``ao_metadata`` is embedded at the section root.
    """
    for idx, ao in enumerate(ao_data):
        label = ao.get("label", f"AO_{idx}")
        vol = _real_array_for_artifact(
            ao["data"],
            dtype=volume_dtype,
            label=f"basis.ao[{label!r}]",
        )
        _require_3d_volume(vol, "basis.ao", label)
        section_id = ao.get("section_id", f"ao_{idx}")
        slug = _slug(label, fallback=section_id)
        path_in_zip = f"basis_ao/{slug}.dat"
        file_member = _write_binary_to_zip(zf, path_in_zip, vol)

        origin_arr = np.asarray(ao["origin"], dtype=np.float64)
        span_arr = np.asarray(ao["span"], dtype=np.float64).reshape(3, 3)
        grid = _grid_descriptor(vol, origin_arr, span_arr)
        grid_json = safe_json_bytes(grid)
        grid_path = f"basis_ao/{slug}_grid.json"
        zf.writestr(grid_path, grid_json)

        meta = ao.get("ao_metadata", {})

        section: dict[str, Any] = {
            "id": section_id,
            "kind": "basis.ao",
            "label": label,
            "ao_metadata": {
                "atom_index": int(meta.get("atom_index", 0)),
                "atom_symbol": str(meta.get("atom_symbol", "?")),
                "shell_index": int(meta.get("shell_index", 0)),
                "primitive_index": int(meta.get("primitive_index", 0)),
                "angular_momentum": [
                    int(meta["angular_momentum"][0]),
                    int(meta["angular_momentum"][1]),
                ],
                "shell_type": str(meta.get("shell_type", "s")),
                "exponent": float(meta.get("exponent", 0.0)),
                "coefficient": float(meta.get("coefficient", 0.0)),
                "is_primitive": bool(meta.get("is_primitive", False)),
                "is_contracted": bool(meta.get("is_contracted", True)),
                "ao_index": int(meta.get("ao_index", 0)),
            },
            "members": {
                "data": file_member,
                "grid": {
                    "path": grid_path,
                    "format": "json",
                    "sha256": _sha256_hex(grid_json),
                },
            },
        }
        # Cartesian (`pure: false`) shells name their component by its
        # (lx, ly, lz) powers -- there is no magnetic quantum number to put
        # in `angular_momentum`. Optional, so a spherical AO omits it.
        cart = meta.get("cartesian_powers")
        if cart is not None:
            section["ao_metadata"]["cartesian_powers"] = [int(p) for p in cart]

        bl = meta.get("basis_label")
        if bl:
            section["ao_metadata"]["basis_label"] = str(bl)

        sections.append(section)


# -- bands ----------------------------------------------------------------


def _write_bands_section(
    zf: zipfile.ZipFile,
    bs: Any,
    sections: list[dict[str, Any]],
) -> None:
    """Write the ``bands`` section.

    ``bs`` is a :class:`BandStructure` with ``kpath``, ``energies``
    (n_points, n_bands, Hartree), and optional ``e_fermi``,
    ``projections``, ``channels``.
    """
    kp = bs.kpath
    energies = np.asarray(bs.energies, dtype=np.float64)
    n_points, n_bands = energies.shape
    e_fermi_eh = bs.e_fermi if bs.e_fermi is not None else 0.0
    e_fermi_ev = float(e_fermi_eh) * _HARTREE_TO_EV

    # Eigenvalues as (1, n_k, n_bands) float64, eV.
    energies_ev = energies * _HARTREE_TO_EV
    data = energies_ev.reshape(1, n_points, n_bands)
    data_member = _write_binary_to_zip(
        zf,
        "bands/eigenvalues.bin",
        data,
    )

    # k-path JSON
    segs: list[dict[str, Any]] = []
    kpath_json_data: dict[str, Any] = {
        "kind": "bands",
        "version": "1.0",
        "n_kpoints": int(n_points),
        "n_bands": int(n_bands),
        "n_spin": 1,
        "fermi": float(e_fermi_ev),
        "fermi_energy_ev": float(e_fermi_ev),
        "reciprocal_space": True,
        "segments": [],
    }
    # Build segments from label pairs.
    labels = kp.labels if hasattr(kp, "labels") else []
    if len(labels) >= 2:
        for i in range(len(labels) - 1):
            d_start, name_start = labels[i]
            d_end, name_end = labels[i + 1]
            seg_mask = (kp.distances >= d_start) & (kp.distances <= d_end)
            n_pts = int(np.sum(seg_mask))
            n_pts = max(n_pts, 2)
            segs.append(
                {
                    "label_start": name_start,
                    "label_end": name_end,
                    "k_start": kp.kpoints_frac[
                        int(np.argmin(np.abs(kp.distances - d_start)))
                    ].tolist(),
                    "k_end": kp.kpoints_frac[
                        int(np.argmin(np.abs(kp.distances - d_end)))
                    ].tolist(),
                    "n_points": n_pts,
                }
            )
    kpath_json_data["segments"] = segs

    # --- band-character channels (fat bands) ------------------------------
    projections = getattr(bs, "projections", None)
    channels = getattr(bs, "channels", None)
    if projections is not None:
        proj = np.asarray(projections, dtype=np.float64)
        if proj.ndim != 3:
            raise ValueError(
                f"_write_bands_section: projections must be rank-3, got "
                f"ndim={proj.ndim}"
            )
        if proj.shape[0] != n_points or proj.shape[1] != n_bands:
            raise ValueError(
                f"_write_bands_section: projections shape {proj.shape} "
                f"mismatches eigenvalues ({n_points}, {n_bands})"
            )
        n_channels = proj.shape[2]
        if channels is not None and len(channels) != n_channels:
            raise ValueError(
                f"_write_bands_section: channels length {len(channels)} "
                f"mismatches projections n_channels={n_channels}"
            )
        # Write channels list into the kpath JSON.
        kpath_json_data["channels"] = [str(c) for c in channels] if channels else []

    kpath_json = safe_json_bytes(kpath_json_data, indent=2)
    kpath_path = "bands/kpath.json"
    zf.writestr(kpath_path, kpath_json)

    members: dict[str, Any] = {
        "kpath": {
            "path": kpath_path,
            "format": "json",
            "sha256": _sha256_hex(kpath_json),
        },
        "eigenvalues": data_member,
    }

    if projections is not None:
        proj_member = _write_binary_to_zip(zf, "bands/projections.bin", proj)
        members["projections"] = proj_member

    section: dict[str, Any] = {
        "id": "bands0",
        "kind": "bands",
        "members": members,
    }
    sections.append(section)


# -- citations ------------------------------------------------------------


def _write_citations_section(
    zf: zipfile.ZipFile,
    bibtex_content: str,
    sections: list[dict[str, Any]],
) -> None:
    """Write the ``citations`` section (embedded BibTeX).

    BibTeX is utf-8 bytes; the manifest format is ``binary`` (not
    ``json``) and carries a sha256 like every other binary member.
    Consumers decode the bytes as utf-8 when they want the text.
    """
    bib_bytes = bibtex_content.encode("utf-8")
    path_in_zip = "citations/references.bib"
    zf.writestr(path_in_zip, bib_bytes)
    section: dict[str, Any] = {
        "id": "citations0",
        "kind": "citations",
        "members": {
            "references": {
                "path": path_in_zip,
                "format": "binary",
                "sha256": _sha256_hex(bib_bytes),
            },
        },
    }
    sections.append(section)


def _write_run_record_section(
    zf: zipfile.ZipFile,
    run_record: dict[str, Any],
    sections: list[dict[str, Any]],
) -> None:
    """Write the ``run.record`` section (QVF spec Sec. 5.8).

    ``run_record`` is a dict assembled by the runner:

    * ``program`` (str, required) + optional ``program_version``,
      ``command``, ``exit_status``, ``started_utc``, ``finished_utc``,
      ``sequence`` -- section-level identity fields.
    * ``input_text`` (str | None) -- the verbatim job input (e.g. the
      driving Python script). Stored as opaque UTF-8 bytes.
    * ``log_text`` (str | None) -- the full ``.out`` log. Stored as
      opaque UTF-8 bytes. At least one of the two must be present.
    * ``input_filename`` / ``log_filename`` (str | None) -- original
      basenames, recorded in the ``files`` index.
    * ``log_truncated`` (bool) -- set when the archive is finalized
      before the log is complete (e.g. periodic geometry optimization
      appends after the QVF is written); recorded in the ``files`` index.
    """
    program = run_record.get("program")
    input_text = run_record.get("input_text")
    log_text = run_record.get("log_text")
    if not program:
        raise ValueError("run.record requires a non-empty 'program'")
    if input_text is None and log_text is None:
        raise ValueError(
            "run.record requires 'input_text' and/or 'log_text'"
        )
    members: dict[str, Any] = {}
    files_index: dict[str, Any] = {}

    def _opaque(role: str, path_in_zip: str, text: str) -> None:
        data = text.encode("utf-8")
        zf.writestr(path_in_zip, data)
        members[role] = {
            "path": path_in_zip,
            "format": "binary",
            "sha256": _sha256_hex(data),
        }

    if input_text is not None:
        _opaque("input", "run_record/input.txt", str(input_text))
        if run_record.get("input_filename"):
            files_index["input"] = {
                "filename": str(run_record["input_filename"]),
            }
    if log_text is not None:
        _opaque("log", "run_record/log.txt", str(log_text))
        log_entry: dict[str, Any] = {}
        if run_record.get("log_filename"):
            log_entry["filename"] = str(run_record["log_filename"])
        if run_record.get("log_truncated"):
            log_entry["truncated"] = True
        if log_entry:
            files_index["log"] = log_entry
    if files_index:
        files_bytes = safe_json_bytes(files_index, indent=2)
        files_path = "run_record/files.json"
        zf.writestr(files_path, files_bytes)
        members["files"] = {
            "path": files_path,
            "format": "json",
            "sha256": _sha256_hex(files_bytes),
        }
    section: dict[str, Any] = {
        "id": "run_record0",
        "kind": "run.record",
        "program": str(program),
        "members": members,
    }
    for key in ("program_version", "command", "started_utc", "finished_utc"):
        if run_record.get(key):
            section[key] = str(run_record[key])
    if run_record.get("exit_status") is not None:
        section["exit_status"] = int(run_record["exit_status"])
    if run_record.get("sequence") is not None:
        section["sequence"] = int(run_record["sequence"])
    sections.append(section)


# -- job.spec ---------------------------------------------------------------


def _write_job_spec_section(
    zf: zipfile.ZipFile,
    job_spec: dict[str, Any],
    sections: list[dict[str, Any]],
) -> None:
    """Write the ``job.spec`` section (QVF spec Sec. 5.9).

    ``job_spec`` is the declarative JobSpecPayload dict: ``job_type``
    (required, ``"molecular"`` | ``"periodic"``) plus the optional typed
    fields ``method``, ``basis``, ``functional``, ``charge``,
    ``multiplicity``, ``kpoints``, ``tasks``, and the open ``options``
    object of extra engine keywords. The payload is data, never code —
    it is stored verbatim as the section's ``spec`` JSON member, so a
    settled archive still records exactly what was asked.
    """
    job_type = job_spec.get("job_type")
    if job_type not in ("molecular", "periodic"):
        raise ValueError(
            "job.spec requires job_type 'molecular' or 'periodic', "
            f"got {job_type!r}"
        )
    kpoints = job_spec.get("kpoints")
    if kpoints is not None:
        mesh = [int(n) for n in kpoints]
        if len(mesh) != 3 or any(n < 1 for n in mesh):
            raise ValueError(
                "job.spec kpoints must be three integers >= 1, "
                f"got {kpoints!r}"
            )
    spec_bytes = safe_json_bytes(dict(job_spec), indent=2)
    path_in_zip = "job_spec/spec.json"
    zf.writestr(path_in_zip, spec_bytes)
    sections.append(
        {
            "id": "job_spec",
            "kind": "job.spec",
            "members": {
                "spec": {
                    "path": path_in_zip,
                    "format": "json",
                    "sha256": _sha256_hex(spec_bytes),
                },
            },
        }
    )


# -- dos.total --------------------------------------------------------------


def _spectral_operator_metadata(data: dict[str, Any]) -> dict[str, str]:
    """Preserve an explicit spectral operator contract, or no contract.

    ROHF diagnostic energies and Hamiltonian-weighted projections can use
    different operators. Dropping part of their provenance would make the
    artifact ambiguous. Ordinary payloads without these labels retain their
    existing representation. Validate before writing any binary members.
    """
    fields = (
        'energy_operator', 'projection_operator', 'orbital_basis',
        'population_density', 'validation_status',
    )
    if not any(field in data for field in fields):
        return {}
    if any(not isinstance(data.get(field), str) or not data[field].strip()
           for field in fields):
        raise ValueError(
            'spectral operator provenance requires nonempty string values for '
            + ', '.join(fields)
        )
    return {field: data[field] for field in fields}


def _write_dos_total_section(
    zf: zipfile.ZipFile,
    dos_data: dict[str, Any],
    sections: list[dict[str, Any]],
) -> None:
    """Write the ``dos.total`` section (QVF spec Sec.4.8).

    ``dos_data`` is a dict with keys:

    * ``energies`` -- float64 `[n_points]` in eV, Fermi = 0
    * ``dos`` -- float64 `[n_points]` (restricted) or `[2, n_points]`
      (spin-polarized), states / eV / cell
    * ``smearing`` -- float, broadening width in eV
    * ``smearing_type`` -- str, e.g. ``"gaussian"``
    * ``fermi_energy_ev`` -- float, absolute Fermi level in eV
    * ``n_electrons`` -- float, integrated electron count
    * ``n_spin`` -- int, 1 (restricted) or 2 (spin-polarized)
    """
    operator_metadata = _spectral_operator_metadata(dos_data)
    energies = np.asarray(dos_data["energies"], dtype=np.float64)
    dos_arr = np.asarray(dos_data["dos"], dtype=np.float64)
    n_spin = int(dos_data.get("n_spin", 1))

    if n_spin == 1:
        if dos_arr.ndim != 1:
            raise ValueError(
                f"dos.total: for n_spin=1, dos must be 1-D; got shape {dos_arr.shape}"
            )
        dos_shape = list(dos_arr.shape)
    else:
        if dos_arr.ndim != 2 or dos_arr.shape[0] != 2:
            raise ValueError(
                "dos.total: for n_spin=2, dos must be shape [2, n_points]; "
                f"got shape {dos_arr.shape}"
            )
        dos_shape = list(dos_arr.shape)

    # Binary payloads.
    energies_member = _write_binary_to_zip(
        zf,
        "dos/energies.bin",
        energies,
    )
    dos_member = _write_binary_to_zip(
        zf,
        "dos/total.bin",
        dos_arr,
    )

    # Metadata JSON (per-spec "role meta").
    smearing_ev = float(dos_data.get("smearing", 0.05))
    smearing_type = str(dos_data.get("smearing_type", "gaussian"))
    fermi_ev = float(dos_data.get("fermi_energy_ev", 0.0))
    n_elec = float(dos_data.get("n_electrons", 0.0))

    section: dict[str, Any] = {
        "id": "dos_total",
        "kind": "dos.total",
        "members": {
            "energies": energies_member,
            "dos": dos_member,
        },
        # Per the spec, metadata is embedded at the section level as
        # optional JSON keys (not as a separate member JSON).
        "smearing": smearing_ev,
        "smearing_type": smearing_type,
        "fermi_energy_ev": fermi_ev,
        "n_electrons": n_elec,
        "n_spin": n_spin,
        **operator_metadata,
    }
    sections.append(section)


# -- dos.projected ----------------------------------------------------------


def _write_dos_projected_section(
    zf: zipfile.ZipFile,
    pdos_data: dict[str, Any],
    sections: list[dict[str, Any]],
) -> None:
    """Write the ``dos.projected`` section (QVF spec Sec.4.9).

    ``pdos_data`` is a dict with keys:

    * ``energies`` -- float64 `[n_points]` in eV, Fermi = 0
    * ``projections`` -- float64 `[n_channels, n_points]` (restricted) or
      `[n_spin, n_channels, n_points]` (spin-polarized), states / eV / cell
    * ``energies_units`` -- str, ``"eV"``
    * ``n_spin`` -- int, 1 or 2
    * ``fermi_energy_ev`` -- float, absolute Fermi level in eV
    * ``channels`` -- list of dicts, each with ``atom_index``, ``symbol``,
      ``l``, ``label``
    """
    operator_metadata = _spectral_operator_metadata(pdos_data)
    energies = np.asarray(pdos_data["energies"], dtype=np.float64)
    projections = np.asarray(pdos_data["projections"], dtype=np.float64)
    n_spin = int(pdos_data.get("n_spin", 1))

    if n_spin == 1:
        if projections.ndim != 2:
            raise ValueError(
                "dos.projected: for n_spin=1, projections must be "
                f"[n_channels, n_points]; got shape {projections.shape}"
            )
    else:
        if projections.ndim != 3 or projections.shape[0] != 2:
            raise ValueError(
                "dos.projected: for n_spin=2, projections must be "
                f"[2, n_channels, n_points]; got shape {projections.shape}"
            )

    # Binary payloads. Use a projected-specific energies path so a
    # dos.projected section can coexist with a dos.total section in the
    # same archive -- both used to write "dos/energies.bin", which
    # produced a duplicate zip member (the energy grid member path is
    # per-section, so the consumer resolves each independently).
    energies_member = _write_binary_to_zip(
        zf,
        "dos/projected_energies.bin",
        energies,
    )
    proj_member = _write_binary_to_zip(
        zf,
        "dos/projections.bin",
        projections,
    )

    # Channel metadata.
    channels: list[dict[str, Any]] = list(pdos_data.get("channels", []))
    energies_units = str(pdos_data.get("energies_units", "eV"))
    fermi_ev = float(pdos_data.get("fermi_energy_ev", 0.0))

    section: dict[str, Any] = {
        "id": "dos_pdos",
        "kind": "dos.projected",
        "members": {
            "energies": energies_member,
            "projections": proj_member,
        },
        # Metadata at section level per spec.  Note: energies_units is
        # a separate meta key to distinguish from dos.total's units layout.
        "energies_units": energies_units,
        "n_spin": n_spin,
        "fermi_energy_ev": fermi_ev,
        "channels": channels,
        **operator_metadata,
    }
    sections.append(section)


# -- dos.coop --------------------------------------------------------------


def _write_dos_coop_section(
    zf: zipfile.ZipFile,
    coop_data: dict[str, Any],
    sections: list[dict[str, Any]],
) -> None:
    """Write the ``dos.coop`` section (QVF spec Sec.4.8b).

    ``coop_data`` is a dict with keys:

    * ``energies`` -- float64 ``[n_points]`` in eV, Fermi = 0
    * ``projections`` -- float64 ``[n_pairs, n_points]`` (restricted) or
      ``[n_spin, n_pairs, n_points]`` (spin-polarized), COOP(E)
    * ``integrated`` -- float64 ``[n_pairs]``, ICOOP integrated to E_F
    * ``energies_units`` -- str, ``"eV"``
    * ``n_spin`` -- int, 1 or 2
    * ``fermi_energy_ev`` -- float, absolute Fermi level in eV
    * ``sigma_ev`` -- float, Gaussian broadening in eV
    * ``pairs`` -- list of dicts, each with ``i``, ``j``, ``symbol_i``,
      ``symbol_j``, ``distance_ang``
    """
    operator_metadata = _spectral_operator_metadata(coop_data)
    energies = np.asarray(coop_data["energies"], dtype=np.float64)
    projections = np.asarray(coop_data["projections"], dtype=np.float64)
    integrated = np.asarray(coop_data["integrated"], dtype=np.float64)
    n_spin = int(coop_data.get("n_spin", 1))

    if n_spin == 1:
        if projections.ndim != 2:
            raise ValueError(
                "dos.coop: for n_spin=1, projections must be "
                f"[n_pairs, n_points]; got shape {projections.shape}"
            )
        if integrated.ndim != 1:
            raise ValueError(
                "dos.coop: for n_spin=1, integrated must be "
                f"[n_pairs]; got shape {integrated.shape}"
            )
    else:
        if projections.ndim != 3 or projections.shape[0] != 2:
            raise ValueError(
                "dos.coop: for n_spin=2, projections must be "
                f"[2, n_pairs, n_points]; got shape {projections.shape}"
            )
        if integrated.ndim != 2 or integrated.shape[0] != 2:
            raise ValueError(
                "dos.coop: for n_spin=2, integrated must be "
                f"[2, n_pairs]; got shape {integrated.shape}"
            )

    energies_member = _write_binary_to_zip(zf, "dos/coop_energies.bin", energies)
    proj_member = _write_binary_to_zip(zf, "dos/coop_projections.bin", projections)
    integ_member = _write_binary_to_zip(zf, "dos/coop_integrated.bin", integrated)

    # Pair metadata as JSON
    pairs: list[dict[str, Any]] = list(coop_data.get("pairs", []))
    meta = {
        "method": "coop",
        **operator_metadata,
        "fermi_energy_ev": float(coop_data.get("fermi_energy_ev", 0.0)),
        "sigma_ev": float(coop_data.get("sigma_ev", 0.27)),
        "n_spin": n_spin,
        "pairs": pairs,
    }
    meta_json = safe_json_bytes(meta)
    meta_path = "dos/coop_meta.json"
    zf.writestr(meta_path, meta_json)

    section: dict[str, Any] = {
        "id": "coop0",
        "kind": "dos.coop",
        "members": {
            "energies": energies_member,
            "projections": proj_member,
            "integrated": integ_member,
            "meta": {
                "path": meta_path,
                "format": "json",
                "sha256": _sha256_hex(meta_json),
            },
        },
        "energies_units": str(coop_data.get("energies_units", "eV")),
        **operator_metadata,
        "n_spin": n_spin,
        "fermi_energy_ev": float(coop_data.get("fermi_energy_ev", 0.0)),
        "sigma_ev": float(coop_data.get("sigma_ev", 0.27)),
        "pairs": pairs,
    }
    sections.append(section)


# -- dos.cohp --------------------------------------------------------------


def _write_dos_cohp_section(
    zf: zipfile.ZipFile,
    cohp_data: dict[str, Any],
    sections: list[dict[str, Any]],
) -> None:
    """Write the ``dos.cohp`` section (QVF spec Sec.4.8c).

    ``cohp_data`` is a dict with keys:

    * ``energies`` -- float64 ``[n_points]`` in eV, Fermi = 0
    * ``projections`` -- float64 ``[n_pairs, n_points]`` (restricted) or
      ``[n_spin, n_pairs, n_points]`` (spin-polarized), -COHP(E) with
      bonding positive
    * ``integrated`` -- float64 ``[n_pairs]``, -ICOHP
    * ``energies_units`` -- str, ``"eV"``
    * ``n_spin`` -- int, 1 or 2
    * ``fermi_energy_ev`` -- float
    * ``sigma_ev`` -- float
    * ``pairs`` -- list of dicts
    """
    operator_metadata = _spectral_operator_metadata(cohp_data)
    energies = np.asarray(cohp_data["energies"], dtype=np.float64)
    projections = np.asarray(cohp_data["projections"], dtype=np.float64)
    integrated = np.asarray(cohp_data["integrated"], dtype=np.float64)
    n_spin = int(cohp_data.get("n_spin", 1))

    if n_spin == 1:
        if projections.ndim != 2:
            raise ValueError(
                "dos.cohp: for n_spin=1, projections must be "
                f"[n_pairs, n_points]; got shape {projections.shape}"
            )
        if integrated.ndim != 1:
            raise ValueError(
                "dos.cohp: for n_spin=1, integrated must be "
                f"[n_pairs]; got shape {integrated.shape}"
            )
    else:
        if projections.ndim != 3 or projections.shape[0] != 2:
            raise ValueError(
                "dos.cohp: for n_spin=2, projections must be "
                f"[2, n_pairs, n_points]; got shape {projections.shape}"
            )
        if integrated.ndim != 2 or integrated.shape[0] != 2:
            raise ValueError(
                "dos.cohp: for n_spin=2, integrated must be "
                f"[2, n_pairs]; got shape {integrated.shape}"
            )

    energies_member = _write_binary_to_zip(zf, "dos/cohp_energies.bin", energies)
    proj_member = _write_binary_to_zip(zf, "dos/cohp_projections.bin", projections)
    integ_member = _write_binary_to_zip(zf, "dos/cohp_integrated.bin", integrated)

    pairs: list[dict[str, Any]] = list(cohp_data.get("pairs", []))
    meta = {
        "method": "cohp",
        **operator_metadata,
        "fermi_energy_ev": float(cohp_data.get("fermi_energy_ev", 0.0)),
        "sigma_ev": float(cohp_data.get("sigma_ev", 0.27)),
        "n_spin": n_spin,
        "pairs": pairs,
    }
    meta_json = safe_json_bytes(meta)
    meta_path = "dos/cohp_meta.json"
    zf.writestr(meta_path, meta_json)

    section: dict[str, Any] = {
        "id": "cohp0",
        "kind": "dos.cohp",
        "members": {
            "energies": energies_member,
            "projections": proj_member,
            "integrated": integ_member,
            "meta": {
                "path": meta_path,
                "format": "json",
                "sha256": _sha256_hex(meta_json),
            },
        },
        "energies_units": str(cohp_data.get("energies_units", "eV")),
        **operator_metadata,
        "n_spin": n_spin,
        "fermi_energy_ev": float(cohp_data.get("fermi_energy_ev", 0.0)),
        "sigma_ev": float(cohp_data.get("sigma_ev", 0.27)),
        "pairs": pairs,
    }
    sections.append(section)


# -- equation_of_state -----------------------------------------------------


def _write_equation_of_state_section(
    zf: zipfile.ZipFile,
    eos_data: dict[str, Any],
    sections: list[dict[str, Any]],
) -> None:
    """Write the ``equation_of_state`` section (QVF spec Sec. 4.14).

    ``eos_data`` is a dict with keys:

    * ``volumes`` -- float64 `[n_points]`, unit-cell volumes
    * ``energies`` -- float64 `[n_points]`, total energies
    * ``fit`` -- dict with ``model``, ``V0``, ``E0``, ``B0``, ``B0_prime``,
      ``energy_unit``, ``volume_unit``, ``pressure_unit``, ``residual_rms``,
      and optional ``pressures_gpa``.
    """
    volumes = np.asarray(eos_data["volumes"], dtype=np.float64)
    energies = np.asarray(eos_data["energies"], dtype=np.float64)
    if volumes.shape != energies.shape:
        raise ValueError(
            "equation_of_state: volumes and energies must have the same shape; "
            f"got {volumes.shape} vs {energies.shape}"
        )

    vols_member = _write_binary_to_zip(zf, "eos/volumes.bin", volumes)
    energies_member = _write_binary_to_zip(zf, "eos/energies.bin", energies)

    fit = dict(eos_data.get("fit", {}))
    fit_json = safe_json_bytes(fit)
    fit_path = "eos/fit.json"
    zf.writestr(fit_path, fit_json)

    section: dict[str, Any] = {
        "id": "eos",
        "kind": "equation_of_state",
        "members": {
            "volumes": vols_member,
            "energies": energies_member,
            "fit": {
                "path": fit_path,
                "format": "json",
                "sha256": _sha256_hex(fit_json),
            },
        },
    }
    sections.append(section)


# -- fermi_surface --------------------------------------------------------


def _write_fermi_surface_section(
    zf: zipfile.ZipFile,
    fermi_data: dict[str, Any],
    sections: list[dict[str, Any]],
) -> None:
    """Write the ``fermi_surface`` section (QVF spec Sec. 4.12).

    ``fermi_data`` is a dict with keys:

    * ``nk1``, ``nk2``, ``nk3`` -- int, Monkhorst-Pack mesh dimensions
    * ``energies`` -- float64 `[nk1, nk2, nk3, n_bands]`, signed distance
      from E_F in eV for bands near the Fermi level
    * ``fermi_energy_ev`` -- float, absolute Fermi level in eV
    * ``band_indices`` -- list of int, which bands are included
    * ``lattice_vectors`` -- 3x3 array, real-space lattice in Angstrom
    * ``n_spin`` -- int, 1 (default) or 2
    """
    nk1 = int(fermi_data["nk1"])
    nk2 = int(fermi_data["nk2"])
    nk3 = int(fermi_data["nk3"])
    energies = np.asarray(fermi_data["energies"], dtype=np.float64)
    if energies.ndim != 4:
        raise ValueError(
            "fermi_surface: energies must be 4-D [nk1, nk2, nk3, n_bands]; "
            f"got shape {energies.shape}"
        )
    fermi_ev = float(fermi_data.get("fermi_energy_ev", 0.0))
    band_indices = [int(b) for b in fermi_data.get("band_indices", [])]
    lattice = np.asarray(fermi_data["lattice_vectors"], dtype=np.float64)
    n_spin = int(fermi_data.get("n_spin", 1))

    # Mesh descriptor (JSON).
    mesh = {
        "nk1": nk1,
        "nk2": nk2,
        "nk3": nk3,
        "n_spin": n_spin,
        "fermi_energy_ev": fermi_ev,
        "band_indices": band_indices,
        "lattice_vectors": lattice.tolist(),
    }
    mesh_json = safe_json_bytes(mesh)
    mesh_path = "fermi/mesh.json"
    zf.writestr(mesh_path, mesh_json)

    # Binary payload.
    energies_member = _write_binary_to_zip(zf, "fermi/energies.bin", energies)

    section: dict[str, Any] = {
        "id": "fermi0",
        "kind": "fermi_surface",
        "members": {
            "mesh": {
                "path": mesh_path,
                "format": "json",
                "sha256": _sha256_hex(mesh_json),
            },
            "energies": energies_member,
        },
    }
    sections.append(section)


# -- phonon_bands ---------------------------------------------------------


def _write_phonon_bands_section(
    zf: zipfile.ZipFile,
    bands_data: dict[str, Any],
    sections: list[dict[str, Any]],
) -> None:
    """Write the ``phonon_bands`` section (QVF spec Sec. 4.13).

    ``bands_data`` is a dict with keys:

    * ``qpath`` -- dict with ``n_atoms``, ``n_modes``,
      ``has_eigenvectors`` (bool), ``segments`` (list of dicts with
      ``label_start``, ``label_end``, ``k_start``, ``k_end``, ``n_points``)
    * ``frequencies`` -- float64 `[n_qpts, n_modes]`, in cm^-1
    * ``eigenvectors`` -- optional float64 `[n_qpts, n_modes, n_atoms, 3]`
    """
    qpath = dict(bands_data["qpath"])
    qpath_json = safe_json_bytes(qpath)
    qpath_path = "phonons/qpath.json"
    zf.writestr(qpath_path, qpath_json)

    freq = np.asarray(bands_data["frequencies"], dtype=np.float64)
    freq_member = _write_binary_to_zip(zf, "phonons/frequencies.bin", freq)

    members: dict[str, Any] = {
        "qpath": {
            "path": qpath_path,
            "format": "json",
            "sha256": _sha256_hex(qpath_json),
        },
        "frequencies": freq_member,
    }

    eig = bands_data.get("eigenvectors")
    if eig is not None:
        eig_arr = np.asarray(eig, dtype=np.float64)
        members["eigenvectors"] = _write_binary_to_zip(
            zf, "phonons/eigenvectors.bin", eig_arr
        )

    section: dict[str, Any] = {
        "id": "phonon_bands",
        "kind": "phonon_bands",
        "members": members,
    }
    sections.append(section)


# -- phonon_dos -----------------------------------------------------------


def _write_phonon_dos_section(
    zf: zipfile.ZipFile,
    dos_data: dict[str, Any],
    sections: list[dict[str, Any]],
) -> None:
    """Write the ``phonon_dos`` section (QVF spec Sec. 4.13).

    ``dos_data`` is a dict with keys:

    * ``frequencies`` -- float64 `[n_points]`, frequency grid in cm^-1
    * ``dos`` -- float64 `[n_points]`, phonon DOS in states / cm^-1
    * ``meta`` -- optional dict with ``smearing``, ``smearing_type``,
      ``n_atoms``, ``n_modes``
    * ``projected`` -- optional float64 `[n_atoms, n_points]`
    """
    freq = np.asarray(dos_data["frequencies"], dtype=np.float64)
    dos_arr = np.asarray(dos_data["dos"], dtype=np.float64)
    if freq.shape != dos_arr.shape:
        raise ValueError(
            "phonon_dos: frequencies and dos must have the same shape; "
            f"got {freq.shape} vs {dos_arr.shape}"
        )

    meta = dict(dos_data.get("meta", {}))
    meta_json = safe_json_bytes(meta)
    meta_path = "phonons/dos_meta.json"
    zf.writestr(meta_path, meta_json)

    freq_member = _write_binary_to_zip(zf, "phonons/dos_freq.bin", freq)
    dos_member = _write_binary_to_zip(zf, "phonons/dos_total.bin", dos_arr)

    members: dict[str, Any] = {
        "meta": {
            "path": meta_path,
            "format": "json",
            "sha256": _sha256_hex(meta_json),
        },
        "frequencies": freq_member,
        "dos": dos_member,
    }

    proj = dos_data.get("projected")
    if proj is not None:
        proj_arr = np.asarray(proj, dtype=np.float64)
        members["projected"] = _write_binary_to_zip(
            zf, "phonons/dos_projected.bin", proj_arr
        )

    section: dict[str, Any] = {
        "id": "phonon_dos",
        "kind": "phonon_dos",
        "members": members,
    }
    sections.append(section)


# -- scf history extraction -----------------------------------------------


def scf_history_from_result(result: Any) -> list[dict[str, Any]] | None:
    """Extract a QVF ``scf_history`` payload from an SCF result object.

    Reads ``result.scf_trace`` -- the canonical per-iteration record both
    the molecular (C++ ``SolverResult``) and periodic (e.g.
    ``GpwScfResult``) solvers carry. Molecular trace steps are objects
    with ``.iter`` / ``.energy`` / ``.delta_e`` attributes; periodic
    steps are dicts with the same keys. Returns a list of records shaped
    for :func:`_write_scf_history_section` (``iter`` / ``energy_eh`` /
    ``delta_e``), or ``None`` when no per-iteration trace is available.

    ``delta_e`` is omitted for the first iteration, mirroring the SCF
    log / perf-tracker convention (no meaningful energy delta exists
    before the second cycle).
    """
    trace = getattr(result, "scf_trace", None)
    if not trace:
        return None
    history: list[dict[str, Any]] = []
    for step in trace:
        if isinstance(step, dict):
            it = step.get("iter")
            energy = step.get("energy")
            delta_e = step.get("delta_e")
        else:
            it = getattr(step, "iter", None)
            energy = getattr(step, "energy", None)
            delta_e = getattr(step, "delta_e", None)
        if it is None or energy is None:
            continue
        it = int(it)
        record: dict[str, Any] = {"iter": it, "energy_eh": float(energy)}
        if delta_e is not None and it > 1:
            record["delta_e"] = float(delta_e)
        history.append(record)
    return history or None


# -- provenance -----------------------------------------------------------


def _build_provenance(context: dict[str, Any]) -> dict[str, Any]:
    """Build the ``provenance`` block for the manifest root."""
    result = context.get("result")
    provenance: dict[str, Any] = {}
    if context.get("method"):
        provenance["method"] = str(context["method"])
    if context.get("functional"):
        provenance["functional"] = str(context["functional"])
    if context.get("basis"):
        provenance["basis"] = str(context["basis"])
    if context.get("jk_method"):
        provenance["jk_method"] = str(context["jk_method"])
    if context.get("jk_method_resolved"):
        provenance["jk_method_resolved"] = str(context["jk_method_resolved"])
    if context.get("jk_method_executed"):
        provenance["jk_method_executed"] = str(context["jk_method_executed"])
    if "dft_plus_u" in context:
        provenance["dft_plus_u"] = bool(context["dft_plus_u"])
    if context.get("dft_plus_u_route"):
        provenance["dft_plus_u_route"] = str(context["dft_plus_u_route"])
    mol_or_sys = context.get("molecule") or context.get("system")
    if mol_or_sys is not None:
        provenance["charge"] = int(getattr(mol_or_sys, "charge", 0))
        provenance["multiplicity"] = int(
            getattr(mol_or_sys, "multiplicity", 1),
        )
        n_elec = getattr(mol_or_sys, "n_electrons", None)
        if n_elec is not None:
            if callable(n_elec):
                n_elec = n_elec()
            provenance["n_electrons"] = int(n_elec)

    if result is not None:
        provenance["scf_converged"] = bool(
            getattr(result, "converged", False),
        )
        # SCF iteration count. Both the molecular ``SolverResult`` and
        # the periodic solver results carry ``n_iter``; fall back to the
        # length of the per-iteration trace when the count isn't a field.
        n_iter = getattr(result, "n_iter", None)
        if n_iter is None:
            _trace = getattr(result, "scf_trace", None)
            if _trace:
                n_iter = len(_trace)
        if n_iter is not None:
            provenance["n_scf_iterations"] = int(n_iter)
        energy = getattr(result, "energy", None)
        if energy is not None:
            provenance["scf_energy"] = {
                "value": float(energy),
                "units": "Eh",
            }
        fermi = getattr(result, "fermi_energy", None)
        if fermi is not None:
            provenance["fermi_energy"] = {
                "value": float(fermi),
                "units": "Eh",
            }

    if context.get("wall_seconds") is not None:
        provenance["wall_seconds"] = float(context["wall_seconds"])

    # Honour the same hostname opt-out the ``.system`` manifest honours.
    # Both levers (the ``record_hostname=False`` kwarg, threaded in through
    # the dispatch context, and the global ``VIBEQC_NO_HOSTNAME`` env var)
    # used to redact only the manifest, so an archive written by a run that
    # had explicitly opted out still carried the machine name. The `.qvf` is
    # the artefact the docs push hardest for sharing (hand to a colleague,
    # attach to a paper's SI), which made it the worst place to leak it.
    # The field stays present and becomes "<redacted>", matching the
    # manifest's placeholder, so consumer shape stays stable.
    from .system_info import _hostname_redacted

    if not context.get("record_hostname", True) or _hostname_redacted():
        provenance["hostname"] = "<redacted>"
    else:
        try:
            import socket

            provenance["hostname"] = socket.gethostname()
        except Exception:
            provenance["hostname"] = "unknown"

    if mol_or_sys is not None:
        # The bound PeriodicSystem exposes ``dim`` in {1,2,3}; molecular
        # stubs may carry ``dimensionality``; a bare Molecule has neither
        # (0-D). Reading only ``dimensionality`` used to record 0 for
        # every real periodic system.
        dim = getattr(mol_or_sys, "dim", None)
        if dim is None:
            dim = getattr(mol_or_sys, "dimensionality", None)
        if dim is None:
            dim = 3 if context.get("system") is not None else 0
        provenance["dimensionality"] = int(dim)

    # --- streaming / checkpoint fields (Ask-1 / Ask-2) -------------------
    # ``run_status`` and ``checkpoint`` let a live consumer (vibe-view)
    # tell a running snapshot from the settled final one, and order a
    # sequence of snapshots by ``checkpoint.seq`` without diffing bytes.
    # Both live under the ``provenance`` block, which the schema declares
    # ``additionalProperties: true`` -- so these validate against the
    # current QVF v1 manifest schema with no v3 bump required.
    run_status = context.get("run_status")
    if run_status is not None:
        rs = str(run_status)
        if rs not in ("pending", "running", "converged", "failed"):
            raise ValueError(
                "write_qvf: run_status must be one of "
                f"'pending' | 'running' | 'converged' | 'failed', got {rs!r}."
            )
        provenance["run_status"] = rs
    checkpoint = context.get("checkpoint")
    if checkpoint is not None:
        # Copy defensively; the caller's dict must not be mutated and any
        # non-JSON-native values (e.g. numpy floats) are coerced here.
        ckpt: dict[str, Any] = {}
        if "seq" in checkpoint:
            ckpt["seq"] = int(checkpoint["seq"])
        if "wall_time_s" in checkpoint:
            ckpt["wall_time_s"] = float(checkpoint["wall_time_s"])
        if checkpoint.get("written_at") is not None:
            ckpt["written_at"] = str(checkpoint["written_at"])
        # Optional running-state hints (additive; provenance is open).
        if checkpoint.get("scf_iteration") is not None:
            ckpt["scf_iteration"] = int(checkpoint["scf_iteration"])
        if checkpoint.get("energy_eh") is not None:
            ckpt["energy_eh"] = float(checkpoint["energy_eh"])
        provenance["checkpoint"] = ckpt

    return provenance


# ---------------------------------------------------------------------------
# Validation tool
# ---------------------------------------------------------------------------
#
# validate_qvf() drives off the canonical schema at
# qvf_manifest.schema.json -- the SSOT. The hand-rolled per-kind checks
# that used to live in this function are gone; what remains is the
# cross-cutting work that JSON Schema can't express: sha256 of every
# member matches the bytes on disk, every referenced zip path exists,
# binary dtype/shape add up to the byte count on disk, operand_a /
# operand_b / trajectory_ref cross-references resolve to existing
# sections.

_SCHEMA_PATH_V1 = Path(__file__).parent / "qvf_manifest.schema.json"
_SCHEMA_PATH_V2 = Path(__file__).parent / "qvf_manifest_v2.schema.json"
# Back-compat alias for any caller still reaching for ``_SCHEMA_PATH``;
# defaults to the v1 path (the only version that existed before v2).
_SCHEMA_PATH = _SCHEMA_PATH_V1
_SCHEMA_CACHE: dict[int, dict[str, Any]] = {}

# ---------------------------------------------------------------------------
# v2 = v1 + the periodic-reaction-path delta, derived — never forked.
#
# v2 began life as a hand-copied snapshot of v1 and then fell behind it:
# every section kind added to v1 afterwards (bond_orders, spectra.epr,
# volume.rdg, volume.potential, basis.ao, fermi_surface, phonon.bands,
# phonon.dos, equation_of_state, topology.qtaim), the root
# thermochemistry / dipole_moment / constraints / extensions blocks, the
# Section ``critical`` flag, and the ``bands`` fat-band ``projections``
# member were all absent from the fork. Because the manifest root sets
# ``additionalProperties: false`` and ``Section`` is a closed ``oneOf``,
# a periodic reaction.path -- which forces qvf_version=2 -- carrying any
# of them tripped the write-time validation gate in :func:`write_qvf`.
#
# Deriving v2 from v1 makes the delta below the *only* way the two can
# differ, so a section added to v1 tomorrow lands in v2 for free.
# ``scripts/gen_qvf_v2_schema.py`` renders this derivation to
# ``qvf_manifest_v2.schema.json`` for external validators;
# ``tests/test_qvf_v2_schema_drift.py`` pins that file to the derivation.
# ---------------------------------------------------------------------------

_V2_TITLE = "QVF Manifest (v2)"

_V2_DESCRIPTION = (
    "DEPRECATED. Canonical JSON Schema for manifest.json inside a .qvf "
    "archive that declares qvf_version=2. GENERATED FILE -- do not "
    "hand-edit; regenerate with scripts/gen_qvf_v2_schema.py. "
    "qvf_version=2 was WITHDRAWN by the governance ruling of 2026-07-10 "
    "(qvf-writer/GOVERNANCE.md, Version history): its only delta over v1 "
    "was the optional SectionReactionPath `lattice` member, which is the "
    "additive case and must not bump the version. `lattice` is now an "
    "optional member of v1, and a periodic reaction path is detected by "
    "its presence. This schema is therefore v1 with a different "
    "qvf_version const, retained only so that archives written by "
    "vibe-qc v0.10.0-v0.15.x (which are v1-compatible) still validate and "
    "so anything resolving its $id keeps working. Producers MUST NOT emit "
    "qvf_version=2; consumers SHOULD accept it and treat it as 1."
)

_V2_QVF_VERSION_DESCRIPTION = (
    "QVF format version. DEPRECATED: 2 was withdrawn (see the schema "
    "description); it is v1 in every respect but this const. Producers "
    "must emit 1."
)

def _derive_v2_schema(v1_schema: dict[str, Any]) -> dict[str, Any]:
    """Return the (deprecated) v2 manifest schema derived from ``v1_schema``.

    ``qvf_version: 2`` was withdrawn by the governance ruling of 2026-07-10
    (``qvf-writer/GOVERNANCE.md``, "Version history"): its only delta over v1
    was the optional ``SectionReactionPath.lattice`` member, which is the
    *additive* case and must not bump the version. ``lattice`` now lives in the
    v1 schema, so the v2 delta has collapsed to pure identity metadata:
    ``$id``, ``title``, ``description``, and ``properties.qvf_version``
    (const + description). Everything else is v1 verbatim.

    v2 is therefore v1 with a different ``qvf_version`` const. It is retained
    only so archives written by vibe-qc v0.10.0-v0.15.x still validate, and so
    anything resolving the v2 ``$id`` keeps working. Producers must emit 1.
    """
    schema = copy.deepcopy(v1_schema)
    schema["$id"] = _SCHEMA_URI_V2
    schema["title"] = _V2_TITLE
    schema["description"] = _V2_DESCRIPTION

    version_prop = schema["properties"]["qvf_version"]
    version_prop["const"] = QVF_FORMAT_VERSION_V2
    version_prop["description"] = _V2_QVF_VERSION_DESCRIPTION
    return schema

def _load_canonical_schema(qvf_version: int = 1) -> dict[str, Any]:
    """Return the canonical QVF manifest schema for ``qvf_version``.

    v1 is loaded from ``qvf_manifest.schema.json``, the single source of
    truth. v2 is *derived* from it by :func:`_derive_v2_schema` rather
    than read from disk, so the on-disk v2 copy can never drift out from
    under the validator. v2 extends only ``SectionReactionPath``, with an
    optional ``lattice`` binary member + ``dim`` carried in metadata, so
    that periodic reaction paths (slabs, surfaces, NEB) round-trip
    cleanly through vibe-view. Every other section is v1's.
    """
    cached = _SCHEMA_CACHE.get(qvf_version)
    if cached is not None:
        return cached
    if qvf_version not in (1, 2):
        raise ValueError(f"unknown qvf_version {qvf_version!r}; supported: 1, 2")

    with open(_SCHEMA_PATH_V1, encoding="utf-8") as f:
        schema = json.load(f)
    _SCHEMA_CACHE[1] = schema
    if qvf_version == 2:
        schema = _derive_v2_schema(schema)
        _SCHEMA_CACHE[2] = schema
    return schema


def validate_qvf(
    source: "os.PathLike | str | zipfile.ZipFile",
) -> dict[str, Any]:
    """Validate a QVF against the canonical SSOT schema.

    ``source`` may be either a filesystem path to a ``.qvf`` file or an
    already-open :class:`zipfile.ZipFile`. The latter form lets
    :func:`qvf_bytes` validate an in-memory archive without
    round-tripping through disk.

    Returns a dict with keys ``valid`` (bool), ``summary`` (list of
    per-section result strs), and ``errors`` (list of error strs).

    Checks performed:

    * The archive is a valid zip, no member exceeds the zip-bomb cap.
    * ``manifest.json`` exists and parses as JSON.
    * The manifest validates against
      :data:`_SCHEMA_PATH` (the canonical schema) -- this catches per-kind
      member shape, dtype, format, and unknown kinds.
    * Every member's declared zip path exists in the archive.
    * Every member's declared sha256 matches the bytes on disk.
    * Every binary member's ``len(bytes) == np.dtype(dtype).itemsize *
      product(shape)`` (no silent under/over-sized buffer).
    * On ``volume.difference``: both ``operand_a`` and ``operand_b``
      (if present) resolve to section ids that exist in the archive.
    * On ``reaction.waypoints``: ``trajectory_ref`` resolves to a
      section in the archive whose kind is ``trajectory``.

    Sections whose kind is in :data:`_RESERVED_KINDS` are not
    shape-validated (no schema branch yet); their file refs are still
    checked. Vendor (``x_*``) sections must conform to the
    ``SectionVendor`` schema branch (members must be valid Json/Binary
    members) but the shape of those members is unconstrained.
    """
    report: dict[str, Any] = {"valid": True, "summary": [], "errors": []}
    summary: list[str] = []
    errors: list[str] = []

    # Accept either a path-like (open + close here) or an already-open
    # ZipFile (do not close -- caller owns the handle).
    own_zf = True
    if isinstance(source, zipfile.ZipFile):
        zf = source
        own_zf = False
    else:
        path = Path(os.fspath(source))
        if not path.is_file():
            report.update(valid=False, errors=[f"file not found: {path}"])
            return report
        try:
            zf = zipfile.ZipFile(path, "r")
        except zipfile.BadZipFile as exc:
            report.update(valid=False, errors=[f"not a valid zip file: {exc}"])
            return report

    # --- zip-bomb guard: per-member uncompressed-size ceiling ----------
    # Aligned with the writer's _MAX_VOXELS payload guard (see module
    # constants) so a valid write_qvf output cannot subsequently fail
    # validation as "too large". Compressed bytes on disk are typically
    # far smaller; this guard reads `file_size` which is the
    # uncompressed length encoded in the zip central directory.
    try:
        for _info in zf.infolist():
            if _info.file_size > _MAX_MEMBER_UNCOMPRESSED_BYTES:
                errors.append(
                    f"member {_info.filename!r}: uncompressed size "
                    f"{_info.file_size:_d} bytes exceeds max "
                    f"{_MAX_MEMBER_UNCOMPRESSED_BYTES:_d}; possible zip bomb"
                )
                report["valid"] = False
    except Exception as exc:
        if own_zf:
            zf.close()
        report.update(valid=False, errors=[f"cannot read zip directory: {exc}"])
        return report

    try:
        # --- manifest.json -------------------------------------------------
        try:
            manifest_bytes = zf.read("manifest.json")
        except KeyError:
            report.update(
                valid=False, errors=errors + ["manifest.json missing from archive"]
            )
            return report

        try:
            manifest = json.loads(manifest_bytes.decode("utf-8"))
        except (json.JSONDecodeError, UnicodeDecodeError) as exc:
            report.update(
                valid=False, errors=errors + [f"manifest.json is not valid JSON: {exc}"]
            )
            return report

        # --- jsonschema validation against the canonical SSOT --------------
        #
        # Sections with reserved kinds have no schema branch yet -- temporarily
        # remove them from the to-be-validated manifest and shape-skip them.
        # Their file refs are still verified below.
        try:
            import jsonschema  # noqa: F401  (soft import; required dep)
        except ImportError:
            errors.append(
                "validate_qvf requires the 'jsonschema' package "
                "(`pip install jsonschema`) -- the canonical schema "
                "cannot be enforced without it."
            )
            report.update(valid=False, errors=errors)
            return report

        # Dispatch by the manifest's qvf_version: v1 archives validate
        # against the v1 schema; v2 archives (periodic reaction.path)
        # against the v2 schema. An unknown version falls through to
        # the v1 schema's `const: 1` rule, which produces the natural
        # "qvf_version was X, expected 1" validation error.
        try:
            manifest_qvf_version = int(manifest.get("qvf_version", 1))
        except (TypeError, ValueError):
            manifest_qvf_version = 1
        schema_version = manifest_qvf_version if manifest_qvf_version in (1, 2) else 1
        schema = _load_canonical_schema(schema_version)

        # Build a "check manifest" that mirrors the input but with
        # reserved-kind sections pulled out (they have no schema branch
        # yet; the validator still file-ref-checks them below). If
        # `sections` was absent from the input we leave it absent so
        # the schema's required-field rule fires.
        check_manifest = dict(manifest)
        raw_sections = manifest.get("sections")
        if isinstance(raw_sections, list):
            shape_check_secs = [
                s
                for s in raw_sections
                if not (isinstance(s, dict) and s.get("kind") in _RESERVED_KINDS)
            ]
            check_manifest["sections"] = shape_check_secs
        # else: `sections` is missing or not a list -- leave the manifest
        # as-is so the schema validator reports it.

        validator = jsonschema.Draft202012Validator(schema)
        schema_errors = sorted(
            validator.iter_errors(check_manifest), key=lambda e: list(e.absolute_path)
        )
        for e in schema_errors:
            loc = "/".join(str(p) for p in e.absolute_path) or "<root>"
            errors.append(f"schema: {loc}: {e.message}")
            report["valid"] = False

        fmt_ver = manifest.get("qvf_version", "?")
        n_secs = len(raw_sections) if isinstance(raw_sections, list) else 0
        summary.append(
            f"manifest.json: qvf_version={fmt_ver}, "
            f"{n_secs} section(s); "
            f"{'schema OK' if not schema_errors else f'{len(schema_errors)} schema error(s)'}"
        )

        # If sections is missing/malformed, the per-section loop is a
        # no-op -- the schema error already covered it.
        if not isinstance(raw_sections, list):
            return report

        # --- critical-flag enforcement (QVF spec 5.4-5.5) -----------
        # Per-section critical: warn if a section is marked critical
        # but its kind is reserved (no schema branch) or an unknown
        # non-vendor kind -- no consumer can render it, so a critical
        # flag is a contract violation.
        if isinstance(raw_sections, list):
            for sec in raw_sections:
                if not isinstance(sec, dict):
                    continue
                if sec.get("critical") is True:
                    kind = sec.get("kind", "")
                    sec_id = sec.get("id", "?")
                    if kind in _RESERVED_KINDS:
                        errors.append(
                            f"section {sec_id} ({kind}): critical=true but "
                            f"kind is reserved (no consumer renders it)"
                        )
                        report["valid"] = False
                    elif not _is_vendor_kind(kind) and kind not in _IMPLEMENTED_KINDS:
                        errors.append(
                            f"section {sec_id} ({kind}): critical=true but "
                            f"kind is not a known canonical or vendor kind"
                        )
                        report["valid"] = False

        # Root extensions block: warn if a critical extension is declared
        # but no sections use its vendor namespace.
        ext_block = manifest.get("extensions")
        if isinstance(ext_block, dict) and isinstance(raw_sections, list):
            section_kinds = {
                s["kind"]
                for s in raw_sections
                if isinstance(s, dict) and isinstance(s.get("kind"), str)
            }
            for ns, ext_info in ext_block.items():
                if not isinstance(ext_info, dict):
                    continue
                if ext_info.get("critical") is True:
                    prefix = ns if ns.endswith(".") else ns + "."
                    used = any(k.startswith(prefix) for k in section_kinds)
                    if not used:
                        summary.append(
                            f"extensions: {ns} declares critical=true but "
                            f"no section uses its namespace"
                        )

        # --- per-section cross-checks --------------------------------------
        zip_names = set(zf.namelist())
        section_ids = {s.get("id") for s in raw_sections if isinstance(s, dict)}
        seen_section_ids: set[str] = set()
        for sec in raw_sections:
            if not isinstance(sec, dict):
                continue
            sec_id = sec.get("id")
            if not isinstance(sec_id, str):
                continue
            if sec_id in seen_section_ids:
                errors.append(f"manifest: duplicate section id {sec_id!r}")
                report["valid"] = False
            seen_section_ids.add(sec_id)

        for sec in raw_sections:
            if not isinstance(sec, dict):
                continue
            sec_id = sec.get("id", "?")
            kind = sec.get("kind", "")

            if _is_vendor_kind(kind):
                file_refs = _collect_file_refs(sec)
                payload_ok = _validate_file_refs(
                    zf, file_refs, zip_names, sec_id, errors
                )
                summary.append(
                    f"section {sec_id} ({kind}): vendor "
                    f"({'refs OK' if payload_ok else 'refs FAILED'}); "
                    "member shape not validated"
                )
                continue

            if kind in _RESERVED_KINDS:
                file_refs = _collect_file_refs(sec)
                payload_ok = _validate_file_refs(
                    zf, file_refs, zip_names, sec_id, errors
                )
                summary.append(
                    f"section {sec_id} ({kind}): reserved, not yet "
                    f"implemented; refs {'OK' if payload_ok else 'FAILED'}"
                )
                continue

            # Implemented (canonical) kind: file refs + dtype/shape match.
            file_refs = _collect_file_refs(sec)
            refs_ok = _validate_file_refs(zf, file_refs, zip_names, sec_id, errors)
            sizes_ok = _validate_binary_shapes(zf, file_refs, sec_id, errors)

            # Cross-reference checks for the kinds that carry them.
            xref_ok = True
            if kind == "volume.difference":
                for key in ("operand_a", "operand_b"):
                    ref = sec.get(key)
                    if ref is not None and ref not in section_ids:
                        errors.append(
                            f"section {sec_id} (volume.difference): {key}="
                            f"{ref!r} does not name a section in this archive"
                        )
                        xref_ok = False
            if kind == "reaction.waypoints":
                ref = sec.get("trajectory_ref")
                target_sec = next(
                    (
                        s
                        for s in raw_sections
                        if isinstance(s, dict) and s.get("id") == ref
                    ),
                    None,
                )
                if target_sec is None:
                    errors.append(
                        f"section {sec_id} (reaction.waypoints): "
                        f"trajectory_ref={ref!r} does not name a section "
                        "in this archive"
                    )
                    xref_ok = False
                elif target_sec.get("kind") != "trajectory":
                    errors.append(
                        f"section {sec_id} (reaction.waypoints): "
                        f"trajectory_ref={ref!r} resolves to a section "
                        f"of kind {target_sec.get('kind')!r}, expected "
                        "'trajectory'"
                    )
                    xref_ok = False

            status_bits = []
            if refs_ok and sizes_ok and xref_ok:
                status_bits.append("OK")
            else:
                if not refs_ok:
                    status_bits.append("refs FAILED")
                if not sizes_ok:
                    status_bits.append("sizes FAILED")
                if not xref_ok:
                    status_bits.append("xref FAILED")
            summary.append(f"section {sec_id} ({kind}): {', '.join(status_bits)}")
            if not (refs_ok and sizes_ok and xref_ok):
                report["valid"] = False

    finally:
        if own_zf:
            zf.close()

    report["summary"] = summary
    report["errors"] = errors
    if errors:
        report["valid"] = False
    return report


def _collect_file_refs(section: dict[str, Any]) -> list[dict[str, Any]]:
    """Walk a section dict and return every file-like sub-dict that has
    a ``path`` key and (optionally) a ``sha256`` key."""
    refs: list[dict[str, Any]] = []

    def _recurse(obj: Any) -> None:
        if isinstance(obj, dict):
            if "path" in obj and isinstance(obj["path"], str):
                refs.append(obj)
            for v in obj.values():
                _recurse(v)
        elif isinstance(obj, list):
            for item in obj:
                _recurse(item)

    _recurse(section)
    return refs


def _validate_binary_shapes(
    zf: zipfile.ZipFile,
    file_refs: list[dict[str, Any]],
    sec_id: str,
    errors: list[str],
) -> bool:
    """Check that every binary member's dtype x shape matches the byte
    length of the member on disk. Catches dtype-shape lies the schema
    cannot express. Returns True if all pass, False otherwise.
    """
    ok = True
    for ref in file_refs:
        if ref.get("format") != "binary":
            continue
        dtype = ref.get("dtype")
        shape = ref.get("shape")
        if dtype is None or shape is None:
            # schema-required fields were missing; the schema error
            # already covers this.
            continue
        try:
            itemsize = int(np.dtype(dtype).itemsize)
        except TypeError:
            errors.append(
                f"section {sec_id}: member {ref['path']!r} declares "
                f"dtype={dtype!r} which numpy cannot resolve"
            )
            ok = False
            continue
        expected = itemsize
        for d in shape:
            expected *= int(d)
        try:
            actual = zf.getinfo(ref["path"]).file_size
        except KeyError:
            # missing-file error already reported by _validate_file_refs.
            continue
        if actual != expected:
            errors.append(
                f"section {sec_id}: member {ref['path']!r} byte length "
                f"{actual} != dtype({dtype}).itemsize x prod({shape}) "
                f"= {expected}"
            )
            ok = False
    return ok


def _validate_file_refs(
    zf: zipfile.ZipFile,
    file_refs: list[dict[str, Any]],
    zip_names: set[str],
    sec_id: str,
    errors: list[str],
) -> bool:
    """Validate sha256 and existence for a list of file reference
    dicts.  Returns True if all pass, False otherwise."""
    ok = True
    for ref in file_refs:
        path_in_zip = ref["path"]
        fmt = ref.get("format", "binary")
        if fmt not in ("json", "binary"):
            errors.append(
                f"section {sec_id}: member {path_in_zip!r} has invalid "
                f"format {fmt!r} (expected 'json' or 'binary')",
            )
            ok = False
            continue

        if path_in_zip not in zip_names:
            errors.append(
                f"section {sec_id}: file {path_in_zip!r} "
                "declared in manifest but missing from zip",
            )
            ok = False
            continue

        expected_sha = ref.get("sha256")
        data: bytes | None = None
        if expected_sha is not None or fmt == "json":
            try:
                data = zf.read(path_in_zip)
            except Exception as exc:
                errors.append(
                    f"section {sec_id}: cannot read {path_in_zip!r}: {exc}",
                )
                ok = False
                continue
            actual = _sha256_hex(data)
            if actual != expected_sha:
                errors.append(
                    f"section {sec_id}: sha256 mismatch for {path_in_zip!r} "
                    f"(expected {expected_sha[:12]}..., got {actual[:12]}...)",
                )
                ok = False

        if fmt == "json" and data is not None:
            try:
                json.loads(data.decode("utf-8"))
            except (UnicodeDecodeError, json.JSONDecodeError) as exc:
                errors.append(
                    f"section {sec_id}: JSON member {path_in_zip!r} "
                    f"does not parse: {exc}"
                )
                ok = False
    return ok


def _print_validate_report(report: dict[str, Any]) -> str:
    """Print-friendly summary of a :func:`validate_qvf` report."""
    lines: list[str] = []
    for s in report["summary"]:
        if "vendor" in s.lower():
            prefix = "⚠"
        elif "OK" in s or "format " in s:
            prefix = "✓"
        else:
            prefix = "✗"
        lines.append(f"{prefix} {s}")
    for e in report["errors"]:
        lines.append(f"✗ ERROR: {e}")
    if report["valid"]:
        lines.append("\n✓ QVF file is valid.")
    else:
        lines.append("\n✗ QVF file has validation errors.")
    return "\n".join(lines)


# ---------------------------------------------------------------------------
# CLI entry point for qvf-validate
# ---------------------------------------------------------------------------


def _qvf_validate_cli() -> None:
    """CLI: ``python -m vibeqc.output.formats.qvf <path.qvf>``."""
    import sys

    if len(sys.argv) < 2:
        print("usage: python -m vibeqc.output.formats.qvf <path.qvf>", file=sys.stderr)
        sys.exit(2)
    report = validate_qvf(sys.argv[1])
    print(_print_validate_report(report))
    sys.exit(0 if report["valid"] else 1)


if __name__ == "__main__":
    _qvf_validate_cli()
