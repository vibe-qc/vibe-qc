"""Exact retained coefficient recipes for real finite-torus pair spaces.

C[R mu,T i] = C[(R-T) mu,0 i] follows the phase-free translation gauge
of Nejad et al., doi:10.1063/5.0290816, Sec. II. Pair columns use
V = E_D A - C_active B_active - C_frozen B_frozen (Riplinger and Neese,
doi:10.1063/1.4773581, Sec. II B1). No tails or entries are screened.
Full arrays are temporary audit/compatibility outputs, never cached here.
"""

from collections.abc import Sequence
from dataclasses import dataclass
from hashlib import sha256
from math import prod

import numpy as np


@dataclass(frozen=True, init=False)
class TranslationOccupiedColumns:
    """Immutable home columns; complete column reconstruction is explicit.

    Explicit NumPy conversion remains a compatibility output. Preparation
    and admission stream columns instead; conversion is never cached.
    """

    home: np.ndarray
    mesh: tuple
    reconstruction_error: float

    @classmethod
    def from_home(cls, home, mesh):
        """Construct the exact translation recipe without an assembled frame.

        Consumers still audit every metric/projector entry. This constructor
        asserts no source identity or orthonormality from covariance alone.
        """
        from .translation import _real_matrix, _readonly
        from ..periodic.chi.pno import _validated_pair_mesh
        mesh = _validated_pair_mesh(mesh)
        home = _real_matrix(home, "occupied home columns")
        if not home.shape[0] or home.shape[0] % prod(mesh):
            raise ValueError("occupied home columns require complete AO cells")
        result = object.__new__(cls)
        object.__setattr__(result, "mesh", mesh)
        object.__setattr__(result, "home", _readonly(home))
        object.__setattr__(result, "reconstruction_error", 0.)
        return result

    def __init__(self, coefficients, mesh, tolerance=1e-10):
        if hasattr(self, 'home'):
            raise AttributeError('occupied columns cannot be reinitialized')
        from .translation import _real_matrix, _readonly, _error
        from ..periodic.chi.pno import _validated_pair_mesh
        mesh = _validated_pair_mesh(mesh)
        value = _real_matrix(coefficients, "selected occupied coefficients")
        if (value.shape[0] == 0 or value.shape[0] % prod(mesh)
                or value.shape[1] % prod(mesh)):
            raise ValueError("selected occupied coefficients require complete translated bands")
        if not np.isfinite(tolerance) or tolerance <= 0:
            raise ValueError("occupied reconstruction tolerance must be positive and finite")
        object.__setattr__(self, "mesh", mesh)
        object.__setattr__(self, "home", _readonly(value[:, :value.shape[1] // prod(mesh)]))
        error = 0.
        for cell in range(prod(mesh)):
            start = cell * self.home.shape[1]
            error = max(error, _error(self.cell(cell) - value[:, start:start+self.home.shape[1]]))
        if error > tolerance:
            raise ValueError("selected occupied home columns do not recover the full translation gauge")
        object.__setattr__(self, "reconstruction_error", error)

    @property
    def shape(self):
        return self.home.shape[0], self.home.shape[1] * prod(self.mesh)

    @property
    def nbytes(self):
        return self.home.nbytes

    def cell(self, cell):
        shift = np.unravel_index(cell, self.mesh)
        return np.roll(self.home.reshape(*self.mesh, -1, self.home.shape[1]),
                       shift, axis=(0, 1, 2)).reshape(self.home.shape) if self.home.shape[1] else self.home

    def __array__(self, dtype=None, copy=None):
        if copy is False:
            raise ValueError("translated occupied columns require reconstruction")
        result = np.empty(self.shape, dtype=float)
        bands = self.home.shape[1]
        for cell in range(prod(self.mesh)):
            result[:, cell*bands:(cell+1)*bands] = self.cell(cell)
        return result.astype(dtype, copy=False) if dtype is not None else result

    def apply(self, coefficients):
        """C @ coefficients with one home-family AO tile at a time."""
        if coefficients.ndim != 2 or coefficients.shape[0] != self.shape[1]:
            raise ValueError("occupied tail coefficients have incompatible dimensions")
        result = np.zeros((self.shape[0], coefficients.shape[1]))
        bands = self.home.shape[1]
        for cell in range(prod(self.mesh)):
            result += self.cell(cell) @ coefficients[cell*bands:(cell+1)*bands]
        return result

    def record(self):
        canonical = self.home.astype("<f8", copy=False)
        return dict(kind="translation-occupied-home-columns-v1", mesh=list(self.mesh),
                    shape=list(self.shape), home_shape=list(self.home.shape),
                    home_sha256=sha256(memoryview(canonical).cast('B') if canonical.size else b'').hexdigest(),
                    retained_bytes=self.nbytes, dense_bytes=8*prod(self.shape),
                    reconstruction_error=self.reconstruction_error)


def occupied_storage(value, mesh, tolerance):
    if type(value) is TranslationOccupiedColumns:
        if value.mesh != tuple(mesh):
            raise ValueError("selected occupied storage mesh mismatch")
        return value
    return TranslationOccupiedColumns(value, mesh, tolerance)


def occupied_input(value, mesh, name):
    from .translation import _real_matrix
    if type(value) is TranslationOccupiedColumns:
        if value.mesh != tuple(mesh):
            raise ValueError("selected occupied storage mesh mismatch")
        return value
    return _real_matrix(value, name)


def occupied_tiles(value):
    """Yield every column exactly once, with at most one cell family live."""
    if type(value) is TranslationOccupiedColumns:
        width = value.home.shape[1]
        if width:
            for cell in range(prod(value.mesh)):
                yield slice(cell*width, (cell+1)*width), value.cell(cell)
    elif value.shape[1]:
        yield slice(0, value.shape[1]), value


def occupied_columns(value, indices):
    if type(value) is not TranslationOccupiedColumns:
        return value[:, indices]
    result = np.empty((value.shape[0], len(indices)))
    width = value.home.shape[1]
    for column, index in enumerate(indices):
        result[:, column] = value.cell(index//width)[:, index % width]
    return result


def occupied_transpose_apply(value, other):
    result = np.empty((value.shape[1], other.shape[1]))
    for columns, tile in occupied_tiles(value):
        result[columns] = tile.T @ other
    return result


def occupied_contraction(left, operator, right):
    from .translation_operators import matrix_apply
    result = np.empty((left.shape[1], right.shape[1]))
    for columns, tile in occupied_tiles(right):
        result[:, columns] = occupied_transpose_apply(left, matrix_apply(operator, tile))
    return result


def occupied_orthogonality_error(left, operator, right=None, *, tolerance=None):
    """Exhaustive maximum entry, never a full occupied Gram matrix."""
    from .translation_audit import cyclic_inputs, metric_defect
    if cyclic_inputs(operator, left, left if right is None else right):
        return metric_defect(left, operator, left if right is None else right, right is None, tolerance)
    return _streamed_orthogonality_error(left, operator, right)


def _streamed_orthogonality_error(left, operator, right=None):
    """Generic oracle, including inputs without established cyclic structure."""
    from .translation import _error
    from .translation_operators import matrix_apply
    diagonal = right is None
    right = left if diagonal else right
    error = 0.
    for columns, tile in occupied_tiles(right):
        metric = matrix_apply(operator, tile)
        for rows, other in occupied_tiles(left):
            block = other.T @ metric
            if diagonal and rows.start == columns.start:
                block -= np.eye(block.shape[0])
            error = max(error, _error(block))
    return error


def occupied_gauge_error(value, mesh):
    """Exhaustively check translated columns, including all generators."""
    from .translation import _error
    n = prod(mesh)
    bands, local = value.shape[1]//n, value.shape[0]//n
    error = 0.
    if not bands:
        return error
    grid = np.arange(value.shape[0]).reshape(*mesh, local)
    cells = np.arange(n).reshape(mesh)
    for axis, extent in enumerate(mesh):
        if extent == 1:
            continue
        rows = np.roll(grid, 1, axis=axis).ravel()
        images = np.roll(cells, -1, axis=axis).ravel()
        # In the exact immutable recipe, every origin is a permutation of
        # this entire home-column comparison, including all displacements.
        # Dense/custom inputs retain the complete origin loop.
        verified = type(value) is TranslationOccupiedColumns and value.mesh == tuple(mesh)
        for cell in (range(1) if verified else range(n)):
            left = occupied_columns(value, range(cell*bands, (cell+1)*bands))
            right = occupied_columns(value, range(images[cell]*bands, (images[cell]+1)*bands))
            error = max(error, _error(left[rows]-right))
    return error


def occupied_projector_rows(value, overlap, rows):
    """Every AO projector column, accumulated over all occupied families."""
    from .translation_operators import matrix_apply
    result = np.zeros((len(rows), value.shape[0]))
    for _, tile in occupied_tiles(value):
        right = matrix_apply(overlap, tile, transpose=True).T
        result += tile[rows] @ right
    return result


@dataclass(frozen=True)
class ProjectedPairCoefficients(Sequence):
    """Lazy immutable pair AO sequence sharing selected occupied home columns.

    Each index returns one independent reconstructed AO basis. No sequence
    of AO bases is retained or passed to the native binding in this mode.
    """

    active: TranslationOccupiedColumns
    frozen: TranslationOccupiedColumns
    recipes: tuple

    def __post_init__(self):
        from .translation_pao import ProjectedPairBasis
        if (type(self.active) is not TranslationOccupiedColumns
                or type(self.frozen) is not TranslationOccupiedColumns
                or self.active.mesh != self.frozen.mesh
                or self.active.shape[0] != self.frozen.shape[0]):
            raise ValueError("projected pair storage requires matching occupied home columns")
        object.__setattr__(self, "recipes", tuple(self.recipes))
        for recipe in self.recipes:
            if type(recipe) is not ProjectedPairBasis:
                raise TypeError("projected pair storage requires immutable projected basis recipes")
            domain, a, b = recipe.ao_indices, recipe.local_coefficients, recipe.occupied_coefficients
            if (domain.ndim != 1 or not domain.size or min(domain) < 0
                    or max(domain) >= self.active.shape[0] or np.unique(domain).size != domain.size
                    or a.ndim != 2 or a.shape[0] != domain.size or not a.shape[1]
                    or b.shape != (self.active.shape[1]+self.frozen.shape[1], a.shape[1])
                    or not np.isfinite(a).all() or not np.isfinite(b).all()):
                raise ValueError("invalid projected pair reconstruction recipe")

    def __len__(self):
        return len(self.recipes)

    @property
    def ranks(self):
        return tuple(recipe.local_coefficients.shape[1] for recipe in self.recipes)

    @property
    def nbytes(self):
        return sum(v.nbytes for recipe in self.recipes for v in
                   (recipe.ao_indices, recipe.local_coefficients, recipe.occupied_coefficients))

    def __getitem__(self, index):
        if isinstance(index, slice):
            return tuple(self[i] for i in range(*index.indices(len(self))))
        recipe = self.recipes[index]
        split = self.active.shape[1]
        result = -self.active.apply(recipe.occupied_coefficients[:split])
        result -= self.frozen.apply(recipe.occupied_coefficients[split:])
        result[recipe.ao_indices] += recipe.local_coefficients
        return result
