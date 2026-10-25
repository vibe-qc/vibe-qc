"""Exact cyclic overlap/Fock operators without square supercell AO storage.

Home-row blocks use A[(R,a),(T,b)] = blocks[T-R,a,b] on the complete
finite translation group. This is the block/Fourier representation of
Nejad et al., doi:10.1063/5.0290816, Sec. II, Eqs. (2)-(6), (15)-(16).
All displacements remain present; neither matrix actions nor reciprocal
symbols introduce a cell normalization, screening or zero-mode correction.
"""

from hashlib import sha256
from itertools import product
from math import prod
from types import MappingProxyType

import numpy as np

from ..periodic.chi.pno import _validated_pair_mesh


def _real(value, name):
    array = np.asarray(value)
    if array.dtype.kind not in 'iuf' or not np.isfinite(array).all():
        raise ValueError(f'{name} must contain finite real numbers')
    with np.errstate(over='ignore', invalid='ignore'):
        array = np.ascontiguousarray(array, dtype=float)
    if not np.isfinite(array).all():
        raise ValueError(f'{name} exceeds the real floating-point range')
    return array


def _readonly(value):
    value = np.ascontiguousarray(value, dtype=float)
    return np.frombuffer(value.tobytes(), dtype=float).reshape(value.shape)


def _array_record(value):
    value = np.ascontiguousarray(value, dtype='<f8')
    return {'shape': list(value.shape), 'sha256': sha256(value.tobytes()).hexdigest()}


class _SealedOperatorType(type):
    def __setattr__(cls, name, value):
        raise TypeError('cyclic operator implementation is sealed')

    def __delattr__(cls, name):
        raise TypeError('cyclic operator implementation is sealed')


class CyclicAOOperator(metaclass=_SealedOperatorType):
    """Immutable real home-row blocks, with no implicit dense conversion.

    ``home_row_blocks`` has shape (prod(mesh), n_ao_cell, n_ao_cell), in
    lexicographic displacement order. Symmetry and positive definiteness
    are checked at the consuming calculation's existing tolerance. An
    operator identity records its values, not retained-SCF authentication.
    """

    __slots__ = ('_blocks', '_mesh', '_cells', '_lookup', '_shape', '_identity', '_provenance')

    def __init_subclass__(cls, **kwargs):
        raise TypeError('cyclic operator implementation cannot be subclassed')

    def __setattr__(self, name, value):
        raise AttributeError('cyclic operators are immutable')

    def __delattr__(self, name):
        raise AttributeError('cyclic operators are immutable')

    def __init__(self, home_row_blocks, mesh):
        if hasattr(self, '_blocks'):
            raise AttributeError('cyclic operators cannot be reinitialized')
        mesh = _validated_pair_mesh(mesh)
        blocks = _real(home_row_blocks, 'home-row blocks')
        if (blocks.ndim != 3 or blocks.shape[0] != prod(mesh)
                or blocks.shape[1] == 0 or blocks.shape[1] != blocks.shape[2]):
            raise ValueError('home-row blocks need shape (n_cells,n_ao_cell,n_ao_cell)')
        cells = tuple(product(*(range(n) for n in mesh)))
        shape = (len(cells)*blocks.shape[1],)*2
        record = {'kind': 'cyclic-ao-home-row-v1', 'mesh': mesh, 'shape': shape,
                  'convention': 'A[R,T]=blocks[T-R]', 'blocks': _array_record(blocks)}
        from .source import source_identity
        for key, value in {'_blocks': _readonly(blocks), '_mesh': mesh, '_cells': cells,
                           '_lookup': MappingProxyType({cell: i for i, cell in enumerate(cells)}),
                           '_shape': shape, '_identity': source_identity(record),
                           '_provenance': MappingProxyType({**record, 'blocks': MappingProxyType({
                               'shape': tuple(blocks.shape), 'sha256': record['blocks']['sha256']})})}.items():
            object.__setattr__(self, key, value)

    @property
    def blocks(self):
        return self._blocks

    @property
    def mesh(self):
        return self._mesh

    @property
    def shape(self):
        return self._shape

    @property
    def identity(self):
        return self._identity

    @property
    def provenance(self):
        return self._provenance

    def __array__(self, dtype=None, copy=None):
        raise TypeError('cyclic AO operators require blockwise access; implicit dense conversion is forbidden')

    def _difference(self, target, origin):
        return self._lookup[tuple((target[a]-origin[a]) % self.mesh[a] for a in range(3))]

    def apply(self, coefficients, *, transpose=False):
        coefficients = _real(coefficients, 'operator coefficients')
        if coefficients.ndim != 2 or coefficients.shape[0] != self.shape[0]:
            raise ValueError('operator coefficients need one row per AO')
        local = self.blocks.shape[1]
        grid = coefficients.reshape(len(self._cells), local, coefficients.shape[1])
        result = np.zeros_like(grid)
        with np.errstate(over='raise', invalid='raise'):
            for r, origin in enumerate(self._cells):
                for t, target in enumerate(self._cells):
                    block = (self.blocks[self._difference(origin, target)].T if transpose
                             else self.blocks[self._difference(target, origin)])
                    result[r] += block @ grid[t]
        if not np.isfinite(result).all():
            raise FloatingPointError('non-finite cyclic AO action')
        return result.reshape(coefficients.shape)

    def block(self, rows, columns):
        rows, columns = _indices(rows, self.shape[0]), _indices(columns, self.shape[0])
        local = self.blocks.shape[1]
        rc, tc = rows // local, columns // local
        displacements = np.empty((len(rows), len(columns)), dtype=np.int64)
        for r in np.unique(rc):
            for t in np.unique(tc):
                displacements[np.ix_(rc == r, tc == t)] = self._difference(self._cells[t], self._cells[r])
        return self.blocks[displacements, rows[:, None] % local, columns[None, :] % local]

    def diagonal(self):
        return np.tile(np.diag(self.blocks[0]), len(self._cells))


def _indices(value, size):
    array = np.asarray(value)
    if (array.ndim != 1 or array.dtype.kind not in 'iu'
            or (array.size and (array.min() < 0 or array.max() >= size))):
        raise ValueError('selected AO indices must be integers in range')
    return array.astype(np.int64, copy=False)


def matrix_snapshot(value, mesh, name='matrix'):
    mesh = _validated_pair_mesh(mesh)
    if type(value) is CyclicAOOperator:
        if value.mesh != mesh:
            raise ValueError(f'{name} operator mesh must match the calculation')
        return value
    array = _real(value, name)
    if (array.ndim != 2 or not array.shape[0] or array.shape[0] != array.shape[1]
            or array.shape[0] % prod(mesh)):
        raise ValueError(f'{name} requires square complete cell-major AO spaces')
    return _readonly(array)


def matrix_apply(value, coefficients, *, transpose=False):
    if type(value) is CyclicAOOperator:
        return value.apply(coefficients, transpose=transpose)
    return (value.T if transpose else value) @ coefficients


def matrix_block(value, rows, columns):
    if type(value) is CyclicAOOperator:
        return value.block(rows, columns)
    return value[np.ix_(rows, columns)]


def matrix_columns_apply(value, indices, coefficients, *, transpose=False):
    """Apply selected AO columns without retaining a full column rectangle."""
    if type(value) is not CyclicAOOperator:
        return (value.T if transpose else value)[:, indices] @ coefficients
    indices = _indices(indices, value.shape[0])
    coefficients = _real(coefficients, 'selected-column coefficients')
    if coefficients.ndim != 2 or coefficients.shape[0] != len(indices):
        raise ValueError('selected-column coefficients must match the AO selection')
    result = np.empty((value.shape[0], coefficients.shape[1]))
    local = value.blocks.shape[1]
    for start in range(0, value.shape[0], local):
        rows = np.arange(start, start+local)
        block = value.block(indices, rows).T if transpose else value.block(rows, indices)
        result[start:start+local] = block @ coefficients
    if not np.isfinite(result).all():
        raise FloatingPointError('non-finite selected-column AO action')
    return result


def matrix_diagonal(value):
    return value.diagonal() if type(value) is CyclicAOOperator else np.diag(value)


def matrix_record(value):
    if type(value) is not CyclicAOOperator:
        return _array_record(value)
    return {'kind': 'cyclic-ao-home-row-v1', 'identity': value.identity,
            'shape': list(value.shape), 'mesh': list(value.mesh),
            'convention': 'A[R,T]=blocks[T-R]', 'blocks': _array_record(value.blocks)}


def validate_translation_matrix(value, mesh, tolerance, *, positive=False, name='matrix'):
    """Check the same finite operator, using reciprocal symbols for block SPD."""
    mesh = _validated_pair_mesh(mesh)
    if type(value) is CyclicAOOperator:
        if value.mesh != mesh:
            raise ValueError(f'{name} operator mesh must match the calculation')
        for i, cell in enumerate(value._cells):
            inverse = value._difference((0, 0, 0), cell)
            if np.max(np.abs(value.blocks[i]-value.blocks[inverse].T)) > tolerance:
                raise ValueError(f'{name} must be symmetric')
        if positive:
            # Home-row convention has the positive Fourier sign. The inverse
            # FFT supplies it; multiply away its 1/N, preserving the operator.
            symbols = np.fft.ifftn(value.blocks.reshape(*mesh, *value.blocks.shape[1:]),
                                  axes=(0, 1, 2))*prod(mesh)
            try:
                for symbol in symbols.reshape(value.blocks.shape):
                    # Block transpose symmetry above owns the AO-entry
                    # tolerance. A second absolute Fourier symmetry cutoff
                    # would amplify admitted AO roundoff by the cell count.
                    if not np.isfinite(symbol).all():
                        raise ValueError(f'{name} reciprocal symbols must be finite')
                    np.linalg.cholesky(symbol)
            except np.linalg.LinAlgError as exc:
                raise ValueError(f'{name} must be positive definite at every reciprocal symbol') from exc
        return
    if np.max(np.abs(value-value.T)) > tolerance:
        raise ValueError(f'{name} must be symmetric')
    if positive:
        try:
            np.linalg.cholesky(value)
        except np.linalg.LinAlgError as exc:
            raise ValueError(f'{name} must be positive definite') from exc
    grid = np.arange(value.shape[0]).reshape(*mesh, value.shape[0]//prod(mesh))
    for axis, extent in enumerate(mesh):
        if extent > 1:
            rows = np.roll(grid, 1, axis=axis).ravel()
            if np.max(np.abs(value[np.ix_(rows, rows)]-value)) > tolerance:
                raise ValueError(f'{name} must be translation covariant')
