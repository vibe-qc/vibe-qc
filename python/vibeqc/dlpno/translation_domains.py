"""Translated Mulliken atom domains for the bounded finite-torus MP2 route.

The absolute population test and union of orbital domains are the existing
molecular Mulliken convention in :mod:`vibeqc.dlpno.pao`. Only home occupied
columns are thresholded; translated masks follow by permutation, including
at a cutoff boundary. This is not a periodic differential-overlap grid or
the OSV domains of Nejad et al., doi:10.1063/5.0290816, Sec. IV C.
"""

from dataclasses import dataclass
from numbers import Integral

import numpy as np

from ..periodic.chi.pno import _validated_pair_mesh, build_translation_pair_plan
from .source import source_identity
from .translation import _error, _real_matrix
from .translation_builder import _array_record, _freeze, _nonnegative
from .translation_operators import (
    matrix_apply, matrix_record, matrix_snapshot, validate_translation_matrix,
)


def _indices(value):
    array = np.ascontiguousarray(value, dtype=np.int64)
    return np.frombuffer(array.tobytes(), dtype=np.int64).reshape(array.shape)


def _groups(supplied, nbf_cell):
    groups = []
    for value in supplied:
        array = np.asarray(value)
        if (array.ndim != 1 or array.dtype.kind not in "iu" or not array.size
                or np.min(array) < 0 or np.max(array) >= nbf_cell):
            raise ValueError("home atom groups must be nonempty integer AO indices within one cell")
        groups.append(np.sort(array).astype(np.int64, copy=True))
    if not groups or not np.array_equal(np.sort(np.concatenate(groups)), np.arange(nbf_cell)):
        raise ValueError("home atom groups must partition every home-cell AO exactly once")
    return tuple(_indices(group) for group in groups)


def derive_translation_atom_groups(source, mesh):
    """Read an atom partition from the source's repeated cell-major shells.

    Both spherical and Cartesian shell sizes are counted. Every AO must map
    to its own cell's atom and the within-cell map must repeat exactly. A
    source without shell metadata can instead supply explicit home groups
    to the experimental driver; no atom map is inferred from AO positions.
    """
    mesh = _validated_pair_mesh(mesh)
    n_cells = int(np.prod(mesh))
    nbf = source.basis.nbasis
    if not hasattr(source.molecule, "atoms") or not callable(getattr(source.basis, "shells", None)):
        raise ValueError("source lacks atom/shell metadata; supply explicit home_atom_aos")
    atoms = tuple(source.molecule.atoms)
    if (not isinstance(nbf, Integral) or isinstance(nbf, (bool, np.bool_))
            or nbf <= 0 or nbf % n_cells or not atoms or len(atoms) % n_cells):
        raise ValueError("atom domain selection requires complete cell-major AO and atom groups")
    nbf_cell, natom_cell = int(nbf) // n_cells, len(atoms) // n_cells
    labels = []
    for shell in source.basis.shells():
        atom, angular = shell.atom_index, shell.l
        if (any(isinstance(x, (bool, np.bool_)) or not isinstance(x, Integral)
                for x in (atom, angular)) or atom < 0 or atom >= len(atoms) or angular < 0
                or not isinstance(shell.pure, (bool, np.bool_))):
            raise ValueError("invalid shell atom or angular-momentum metadata")
        size = 2 * angular + 1 if shell.pure else (angular + 1) * (angular + 2) // 2
        labels.extend([int(atom)] * size)
    if len(labels) != nbf:
        raise ValueError("shell sizes must cover exactly the source AO basis")
    labels = np.asarray(labels).reshape(n_cells, nbf_cell)
    if (np.any(labels[0] >= natom_cell)
            or not np.array_equal(labels, labels[0] + np.arange(n_cells)[:, None] * natom_cell)):
        raise ValueError("shell atom map must repeat in the explicit cell-major AO order")
    return _groups([np.flatnonzero(labels[0] == atom) for atom in range(natom_cell)], nbf_cell)


@dataclass(frozen=True)
class TranslationDomainSelection:
    """Immutable representative AO domains and their consumed-input record."""

    pair_ao_domains: tuple
    home_atom_aos: tuple
    home_populations: np.ndarray
    home_atom_masks: np.ndarray
    identity: str
    provenance: object


def select_translation_pair_domains(
    mesh, overlap, occupied, home_atom_aos, *, tcut_mkn=0.0, tolerance=1e-10,
):
    """Select home orbital atom domains and translate their pairwise unions.

    ``home_atom_aos`` partitions AO indices within a primitive cell. Home
    populations include atoms throughout the finite torus. A strict
    ``abs(population) > tcut_mkn`` selects atoms, while zero selects all.
    Empty domains are errors; there is no largest-population fallback,
    distance growth, pair dropping or reinterpretation of the threshold.
    """
    mesh = _validated_pair_mesh(mesh)
    cutoff = _nonnegative(tcut_mkn, "tcut_mkn")
    tol = _nonnegative(tolerance, "tolerance")
    if tol == 0:
        raise ValueError("tolerance must be positive")
    S = matrix_snapshot(overlap, mesh, name="domain overlap")
    from .translation_storage import (occupied_input, occupied_orthogonality_error,
        occupied_gauge_error, occupied_columns)
    C = occupied_input(occupied, mesh, "domain occupied")
    n_cells = int(np.prod(mesh))
    nbf, nocc = C.shape
    if (not nbf or not nocc or nbf % n_cells or nocc % n_cells or S.shape != (nbf, nbf)
            or occupied_orthogonality_error(C, S) > tol):
        raise ValueError("domain tensors need a symmetric metric and complete orthonormal occupied bands")
    validate_translation_matrix(S, mesh, tol, positive=True, name="domain overlap")
    nbf_cell, n_bands = nbf // n_cells, nocc // n_cells
    groups = _groups(home_atom_aos, nbf_cell)
    n_atoms = len(groups)
    if occupied_gauge_error(C, mesh) > tol:
        raise ValueError("domain tensors must have a phase-free cell-major translation gauge")
    home = occupied_columns(C, range(n_bands))
    products = home * matrix_apply(S, home)
    populations = np.empty((n_bands, n_cells, n_atoms))
    for cell in range(n_cells):
        for atom, group in enumerate(groups):
            populations[:, cell, atom] = np.sum(products[cell * nbf_cell + group], axis=0)
    if not np.isfinite(populations).all():
        raise FloatingPointError("non-finite Mulliken domain populations")
    masks = np.ones_like(populations, dtype=bool) if cutoff == 0 else np.abs(populations) > cutoff
    if np.any(~np.any(masks, axis=(1, 2))):
        raise ValueError("Mulliken threshold leaves an empty home orbital domain")
    plan = build_translation_pair_plan(n_bands, mesh)
    domains = []
    for (i, j), displacement in zip(plan.band_pairs, plan.displacements):
        translated_j = np.roll(masks[j].reshape(*mesh, n_atoms), tuple(displacement),
                               axis=(0, 1, 2)).reshape(n_cells, n_atoms)
        union = masks[i] | translated_j
        domain = np.concatenate([cell * nbf_cell + groups[atom]
                                 for cell, atom in zip(*np.nonzero(union))])
        domains.append(_indices(np.sort(domain)))
    record = {
        "kind": "translation-mulliken-domains-v2", "mesh": list(mesh),
        "overlap": matrix_record(S), "occupied": _array_record(C),
        "home_atom_aos": [group.tolist() for group in groups],
        "tcut_mkn": cutoff, "tolerance": tol,
        "home_populations": _array_record(populations),
        "home_atom_masks": masks.tolist(),
        "pair_ao_domains": [domain.tolist() for domain in domains],
        "plan_identity": plan.fingerprint,
    }
    population_copy = np.frombuffer(populations.tobytes(), dtype=float).reshape(populations.shape)
    mask_copy = np.frombuffer(masks.tobytes(), dtype=bool).reshape(masks.shape)
    return TranslationDomainSelection(tuple(domains), groups, population_copy, mask_copy,
                                      source_identity(record), _freeze(record))
