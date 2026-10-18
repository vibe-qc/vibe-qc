"""Validated translation transport for real finite-torus pair spaces.

This first contract requires explicit cell-major AOs and occupied columns:
``index = cell * n_bands + band``. Primitive translations must carry occupied
columns into one another without phases. It does not infer a Wannier gauge
from centroids or cached localization diagnostics, and does not dispatch a
correlation calculation automatically.

Pair representatives and stabilizer multiplicities follow Nejad et al.,
J. Chem. Phys. 163, 214107 (2025), doi:10.1063/5.0290816, Sec. II,
Eqs. (11)-(16). Cross-pair overlaps retain the reference AO metric as in
Riplinger and Neese, J. Chem. Phys. 138, 034106 (2013),
doi:10.1063/1.4773581, Sec. II C, Eq. (30).
"""

from __future__ import annotations

from dataclasses import dataclass
from itertools import product
from numbers import Integral, Real
from types import MappingProxyType

import numpy as np

from .translation_operators import (
    matrix_apply, matrix_snapshot, validate_translation_matrix,
)

from ..periodic.chi.pno import build_translation_pair_plan

__all__ = ["PlacedPair", "TranslationPairSpaces"]


@dataclass(frozen=True)
class PlacedPair:
    """An ordered placed pair in its representative's translated AO basis."""

    row: int
    translation: tuple[int, int, int]
    transposed: bool


def _real_matrix(value, name):
    array = np.asarray(value)
    if array.ndim != 2 or array.dtype.kind not in "iuf":
        raise ValueError(f"{name} must be a real matrix")
    array = np.asarray(array, dtype=float)
    if not np.all(np.isfinite(array)):
        raise ValueError(f"{name} must contain only finite values")
    return array


def _readonly(array):
    """Copy into an immutable buffer, preserving caller arrays and flags."""
    array = np.ascontiguousarray(array, dtype=float)
    return np.frombuffer(array.tobytes(), dtype=float).reshape(array.shape)


def _error(array):
    if not np.all(np.isfinite(array)):
        return float("inf")
    return float(np.max(np.abs(array))) if array.size else 0.0


class TranslationPairSpaces:
    """One metric-orthonormal pair basis per unique home-cell pair.

    ``occupied`` contains the active occupied space. An optional
    ``frozen_occupied`` supplies complete frozen bands in the same explicit
    cell-major gauge; pair bases exclude both spaces. The caller owns this
    partition. Every retained pair basis must be orthogonal to all supplied
    occupieds, and a self-inverse pair's basis span must be closed under its
    half-cell translation. Internal pair-basis rotations are unrestricted.

    ``pair_coefficients`` follows ``build_translation_pair_plan`` row order.
    Transport moves its AO rows without re-diagonalizing a translated PNO
    density. Stored arrays are immutable; compact storage shares occupied
    home columns and reconstructs individual bases from projected recipes.
    """

    def __init__(
        self, mesh, overlap, fock, occupied, pair_coefficients, *,
        tolerance=1e-10, frozen_occupied=None, compact_storage=False,
    ):
        if not isinstance(compact_storage, bool):
            raise ValueError("compact_storage must be boolean")
        if (isinstance(tolerance, (bool, np.bool_))
                or not isinstance(tolerance, Real)
                or not np.isfinite(tolerance) or tolerance <= 0):
            raise ValueError("translation tolerance must be a positive finite real number")
        self.tolerance = float(tolerance)
        # Let the existing compact planner own finite-group mesh admission.
        checked_mesh = build_translation_pair_plan(1, mesh).mesh
        self.mesh = checked_mesh
        self._cells = tuple(product(*(range(extent) for extent in self.mesh)))
        self._cell_lookup = {cell: index for index, cell in enumerate(self._cells)}
        n_cells = len(self._cells)
        S = matrix_snapshot(overlap, self.mesh, "overlap")
        F = matrix_snapshot(fock, self.mesh, "Fock")
        from .translation_storage import (occupied_input, occupied_orthogonality_error,
            occupied_transpose_apply, occupied_contraction)
        C = occupied_input(occupied, self.mesh, "active occupied coefficients")
        if (S.shape[0] == 0 or S.shape[0] != S.shape[1]
                or S.shape[0] % n_cells or F.shape != S.shape
                or C.shape[0] != S.shape[0] or C.shape[1] == 0
                or C.shape[1] % n_cells):
            raise ValueError("translation tensors require complete cell-major AO and occupied bands")
        validate_translation_matrix(S, self.mesh, self.tolerance, positive=True, name="translation overlap")
        validate_translation_matrix(F, self.mesh, self.tolerance, name="translation Fock")
        frozen = (np.empty((S.shape[0], 0)) if frozen_occupied is None
                  else occupied_input(frozen_occupied, self.mesh, "frozen occupied coefficients"))
        if frozen.shape[0] != S.shape[0] or frozen.shape[1] % n_cells:
            raise ValueError("frozen occupied coefficients require complete cell-major bands")
        if C.shape[1] + frozen.shape[1] >= S.shape[0]:
            raise ValueError("translation pair spaces require a nonempty virtual space")
        self._nbf_cell = S.shape[0] // n_cells
        self._n_occupied = C.shape[1]
        self._n_bands = C.shape[1] // n_cells
        self._overlap, self._fock = S, F
        from .translation_storage import occupied_storage, ProjectedPairCoefficients
        self.compact_storage = bool(compact_storage)
        if compact_storage:
            self._occupied = occupied_storage(occupied, self.mesh, tolerance)
            self._frozen_occupied = occupied_storage(
                frozen if frozen_occupied is None else frozen_occupied, self.mesh, tolerance)
        else:
            self._occupied, self._frozen_occupied = _readonly(C), _readonly(frozen)
        self._validate_occupied(C, "active occupied")
        self._validate_occupied(frozen, "frozen occupied")
        if occupied_orthogonality_error(C, S, frozen) > self.tolerance:
            raise ValueError("active and frozen occupied spaces must be orthogonal")
        self.plan = build_translation_pair_plan(self._n_bands, self.mesh)
        self.representative_pairs = tuple(
            (int(bands[0]), self._cell_lookup[tuple(int(x) for x in displacement)]
             * self._n_bands + int(bands[1]))
            for bands, displacement in zip(self.plan.band_pairs, self.plan.displacements)
        )
        self._row_lookup = {
            (int(bands[0]), int(bands[1]), *(int(x) for x in displacement)): row
            for row, (bands, displacement) in enumerate(
                zip(self.plan.band_pairs, self.plan.displacements)
            )
        }
        lazy = type(pair_coefficients) is ProjectedPairCoefficients
        if lazy and (not compact_storage or pair_coefficients.active is not self._occupied
                     or pair_coefficients.frozen is not self._frozen_occupied):
            raise ValueError("pair recipes must share the admitted occupied storage")
        supplied = pair_coefficients if lazy else tuple(pair_coefficients)
        if len(supplied) != self.plan.n_representatives:
            raise ValueError("pair coefficients must supply exactly one basis per representative")
        bases = []
        stabilizers = {}
        for row, value in enumerate(supplied):
            V = _real_matrix(value, f"pair coefficients row {row}")
            if V.shape[0] != S.shape[0] or V.shape[1] == 0:
                raise ValueError("pair coefficients must have one row per AO and nonzero rank")
            metric_basis = matrix_apply(S, V)
            if _error(V.T @ metric_basis - np.eye(V.shape[1])) > self.tolerance:
                raise ValueError("pair coefficients must be S-orthonormal")
            if max(_error(occupied_transpose_apply(part, metric_basis)) for part in (C, frozen)) > self.tolerance:
                raise ValueError("pair coefficients must exclude active and frozen occupied spaces")
            displacement = tuple(int(x) for x in self.plan.displacements[row])
            bands = self.plan.band_pairs[row]
            if (bands[0] == bands[1] and displacement != (0, 0, 0)
                    and all((2 * displacement[axis]) % self.mesh[axis] == 0
                            for axis in range(3))):
                translated = V[self._translated_rows(displacement)]
                U = V.T @ matrix_apply(S, translated)
                if (_error(translated - V @ U) > self.tolerance
                        or _error(U.T @ U - np.eye(V.shape[1])) > self.tolerance):
                    raise ValueError("self-inverse pair basis is not closed under its translation stabilizer")
                stabilizers[row] = _readonly(U)
            if not lazy:
                bases.append(_readonly(V))
        self.pair_coefficients = supplied if lazy else tuple(bases)
        self.pair_ranks = supplied.ranks if lazy else tuple(value.shape[1] for value in bases)
        self.stabilizers = MappingProxyType(stabilizers)
        self.occupied_fock = _readonly(_real_matrix(occupied_contraction(C, F, C), "active occupied Fock"))
        self.energy_weights = _readonly(np.asarray([
            int(multiplicity) / n_cells * (1. if i == j else 2.)
            for multiplicity, (i, j) in zip(self.plan.multiplicities, self.representative_pairs)
        ]))

    def _translated_rows(self, translation):
        """AO rows of P_translation C, with P shifting cells forward."""
        cells = [self._cell_lookup[tuple((cell[axis] - translation[axis]) % self.mesh[axis]
                                        for axis in range(3))]
                 for cell in self._cells]
        return (np.asarray(cells)[:, None] * self._nbf_cell
                + np.arange(self._nbf_cell)[None, :]).ravel()

    def _validate_occupied(self, coefficients, label):
        from .translation_storage import occupied_orthogonality_error, occupied_gauge_error
        if occupied_orthogonality_error(coefficients, self._overlap) > self.tolerance:
            raise ValueError(f"{label} coefficients must be S-orthonormal")
        if occupied_gauge_error(coefficients, self.mesh) > self.tolerance:
            raise ValueError(f"{label} columns must have the explicit phase-free cell-major translation gauge")

    def resolve(self, i, j):
        """Map an ordered occupied pair to a translated home representative.

        For equal-band self-inverse displacements the tie favors no swap.
        Reversing that pair therefore uses the half-translated basis; the
        stabilizer condition in ``validate_tensors`` makes both routes agree.
        """
        if any(isinstance(index, (bool, np.bool_)) or not isinstance(index, Integral)
               or index < 0 or index >= self._n_occupied for index in (i, j)):
            raise ValueError("occupied pair indices must be integers within the active space")
        cell_i, band_i = divmod(int(i), self._n_bands)
        cell_j, band_j = divmod(int(j), self._n_bands)
        origin_i, origin_j = self._cells[cell_i], self._cells[cell_j]
        displacement = tuple((origin_j[axis] - origin_i[axis]) % self.mesh[axis]
                             for axis in range(3))
        inverse = tuple((-displacement[axis]) % self.mesh[axis] for axis in range(3))
        transpose = band_i > band_j or (band_i == band_j and displacement > inverse)
        if transpose:
            band_i, band_j = band_j, band_i
            displacement = inverse
        row = self._row_lookup[(band_i, band_j, *displacement)]
        return PlacedPair(row, origin_j if transpose else origin_i, transpose)

    def coefficients(self, i, j):
        """Return the representative basis translated to this placed pair."""
        placement = self.resolve(i, j)
        return self.pair_coefficients[placement.row][self._translated_rows(placement.translation)]

    def overlap_between(self, target_pair, source_pair):
        """Cross-pair overlap in the actual calculation's AO metric."""
        return _real_matrix(
            self.coefficients(*target_pair).T @ matrix_apply(self._overlap, self.coefficients(*source_pair)),
            "cross-pair overlap",
        )

    def fock_between(self, target_pair, source_pair):
        """AO Fock transported between two placed pair bases."""
        return _real_matrix(
            self.coefficients(*target_pair).T @ matrix_apply(self._fock, self.coefficients(*source_pair)),
            "cross-pair Fock",
        )

    def validate_tensors(self, tensors, name="pair tensors"):
        """Validate exchange/stabilizer sewing and return immutable copies.

        For a same-band half-cell pair, translation swaps its two occupied
        labels and rotates its basis by U. Thus T = U T.T U.T. Violations
        are refused; amplitudes and interaction tensors are never repaired.
        """
        supplied = tuple(tensors)
        if len(supplied) != len(self.pair_coefficients):
            raise ValueError(f"{name} must supply exactly one matrix per representative")
        validated = []
        for row, (value, rank, pair) in enumerate(zip(supplied, self.pair_ranks, self.representative_pairs)):
            tensor = _real_matrix(value, f"{name} row {row}")
            if tensor.shape != (rank, rank):
                raise ValueError(f"{name} row {row} must match its pair basis rank")
            if pair[0] == pair[1] and _error(tensor - tensor.T) > self.tolerance:
                raise ValueError(f"{name} diagonal pair must be symmetric")
            U = self.stabilizers.get(row)
            if U is not None and _error(tensor - U @ tensor.T @ U.T) > self.tolerance:
                raise ValueError(f"{name} must respect the self-inverse pair translation stabilizer")
            validated.append(_readonly(tensor))
        return tuple(validated)

    def orient_tensor(self, i, j, tensors):
        """Orient validated representative tensors in a placed pair's basis."""
        placement = self.resolve(i, j)
        tensor = tensors[placement.row]
        return tensor.T if placement.transposed else tensor
