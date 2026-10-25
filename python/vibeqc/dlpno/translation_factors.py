"""Constructively translation-covariant factors on a real finite torus.

Only the home auxiliary block is stored. AO and auxiliary cells use the
same lexicographic mesh order, and the complete factor is defined by
``B[t P, mu, nu] = B0[P, mu-t, nu-t]``. Translating both orbital inputs
therefore permutes the auxiliary cells. Summing over those cells proves
translation covariance without sampling translated pair integrals.

The class is sealed and its numerical storage is immutable. A builder may
admit this exact implementation and matching mesh structurally; arbitrary
factor providers still need an independent covariance audit. This is an
input-construction guarantee, not a certification of the fitted Hamiltonian
or of a source's ``same_source`` declaration. The ordinary MO transform is
retained for existing molecular and periodic engine contracts. The narrow
real pair contraction instead streams auxiliary cells, never materializing
the complete ``B[n_cells*n_aux_home, 2, n_pair]`` transform.
"""

from __future__ import annotations

from itertools import product
from types import MappingProxyType

import numpy as np

from .._vibeqc_core import (
    aiccm2026dev_b_home_auxiliary_3index_mo_transform,
    aiccm2026dev_b_home_auxiliary_3index_mo_transform_complex,
)
from ..periodic.chi._transform_inputs import checked_mesh_size, integer_array, real_array
from .source import _array_identity, source_identity

__all__ = ["TranslationHomeFactors"]


class _SealedFactorType(type):
    def __setattr__(cls, name, value):
        raise TypeError("translation factor implementation is sealed")

    def __delattr__(cls, name):
        raise TypeError("translation factor implementation is sealed")


def _coefficients(value, n_orb, *, real_only=False):
    array = np.asarray(value)
    kinds = "iuf" if real_only else "iufc"
    if (array.ndim != 2 or array.shape[0] != n_orb or array.dtype.kind not in kinds
            or not np.isfinite(array).all()):
        qualifier = "real " if real_only else ""
        raise ValueError(f"factor MO coefficients must be finite {qualifier}matrices with one row per AO")
    dtype = complex if np.iscomplexobj(array) else float
    with np.errstate(over="ignore", invalid="ignore"):
        array = np.ascontiguousarray(array, dtype=dtype)
    if not np.isfinite(array).all():
        raise ValueError("factor MO coefficients exceed the floating-point range")
    return array


class TranslationHomeFactors(metaclass=_SealedFactorType):
    """Immutable home-auxiliary factors with exact cyclic translation action.

    ``home_factors`` has shape ``(n_aux_home, n_ao, n_ao)``. The AO count
    must contain complete cell-major copies for ``mesh``. Construction
    snapshots its values; changing the original array has no effect. The
    optional auxiliary-basis object is descriptive and never participates
    in numerical contractions or the factor identity.
    """

    __slots__ = ("_home_factors", "_mesh", "_aux_basis", "_identity", "_provenance")

    def __init_subclass__(cls, **kwargs):
        raise TypeError("translation factor implementation cannot be subclassed")

    def __setattr__(self, name, value):
        raise AttributeError("translation factors are immutable")

    def __delattr__(self, name):
        raise AttributeError("translation factors are immutable")

    def __init__(self, home_factors, mesh, *, auxiliary_basis=None):
        if hasattr(self, "_home_factors"):
            raise AttributeError("translation factors are immutable and cannot be reinitialized")
        factors = np.ascontiguousarray(real_array(home_factors, name="home_factors"))
        mesh_tuple = tuple(integer_array(mesh, name="mesh", shape=(3,)).tolist())
        n_cells = checked_mesh_size(mesh_tuple)
        if (factors.ndim != 3 or factors.shape[0] == 0 or factors.shape[1] == 0
                or factors.shape[1] != factors.shape[2]):
            raise ValueError("home-auxiliary factors must have shape (n_aux_home, n_ao, n_ao)")
        if factors.shape[1] % n_cells:
            raise ValueError("home-auxiliary AO rank does not match the cyclic mesh")
        # A read-only flag on caller-backed storage could be reversed through
        # its owner. A bytes-backed snapshot has no writable array owner.
        factors = np.frombuffer(factors.tobytes(), dtype=float).reshape(factors.shape)
        tensor = _array_identity(factors)
        record = {
            "kind": "translation-home-factors-v1",
            "construction": "B[tP,mu,nu]=B0[P,mu-t,nu-t]",
            "mesh": mesh_tuple,
            "home_factors": {"shape": tuple(tensor["shape"]), "sha256": tensor["sha256"]},
            "n_aux": n_cells * factors.shape[0], "n_orb": factors.shape[1],
        }
        object.__setattr__(self, "_home_factors", factors)
        object.__setattr__(self, "_mesh", mesh_tuple)
        object.__setattr__(self, "_aux_basis", auxiliary_basis)
        object.__setattr__(self, "_identity", source_identity(record))
        object.__setattr__(self, "_provenance", MappingProxyType({
            **record, "home_factors": MappingProxyType(record["home_factors"]),
        }))

    @property
    def home_factors(self):
        return self._home_factors

    @property
    def mesh(self):
        return self._mesh

    @property
    def aux_basis(self):
        return self._aux_basis

    @property
    def n_aux(self):
        return checked_mesh_size(self._mesh) * self._home_factors.shape[0]

    @property
    def n_orb(self):
        return self._home_factors.shape[1]

    @property
    def identity(self):
        return self._identity

    @property
    def provenance(self):
        return self._provenance

    def mo_transform(self, coefficients_left, coefficients_right=None):
        """Return the complete logical auxiliary transform for legacy consumers."""
        right = coefficients_left if coefficients_right is None else coefficients_right
        left_array = _coefficients(coefficients_left, self.n_orb)
        right_array = _coefficients(right, self.n_orb)
        mesh = np.asarray(self._mesh, dtype=np.int64)
        if np.iscomplexobj(left_array) or np.iscomplexobj(right_array):
            transformed = aiccm2026dev_b_home_auxiliary_3index_mo_transform_complex(
                np.ascontiguousarray(self._home_factors, dtype=complex),
                np.ascontiguousarray(left_array, dtype=complex),
                np.ascontiguousarray(right_array, dtype=complex), mesh,
            )
        else:
            transformed = aiccm2026dev_b_home_auxiliary_3index_mo_transform(
                self._home_factors, left_array, right_array, mesh,
            )
        if not np.isfinite(transformed).all():
            raise FloatingPointError("non-finite home-auxiliary MO transform")
        return transformed

    def pair_exchange(self, occupied, virtual):
        """Contract real ``(i a|j b)`` for two occupied columns in bounded storage.

        Each auxiliary cell contributes ``B_t[:,0,:].T @ B_t[:,1,:]``.
        Shifting the coefficient rows by minus the auxiliary translation
        applies the defining AO translation without copying translated AO
        factors. Only one cell's two occupied transforms is live at a time.
        There is no cell normalization here; the energy contraction owns it.
        """
        occupied = _coefficients(occupied, self.n_orb, real_only=True)
        virtual = _coefficients(virtual, self.n_orb, real_only=True)
        if occupied.shape[1] != 2 or virtual.shape[1] == 0:
            raise ValueError("pair exchange needs two occupied columns and a nonempty pair basis")
        n_cells = checked_mesh_size(self._mesh)
        n_local = self.n_orb // n_cells
        left_grid = occupied.reshape(*self._mesh, n_local, 2)
        right_grid = virtual.reshape(*self._mesh, n_local, virtual.shape[1])
        exchange = np.zeros((virtual.shape[1], virtual.shape[1]))
        with np.errstate(over="raise", invalid="raise"):
            for cell in product(*(range(value) for value in self._mesh)):
                shift = tuple(-value for value in cell)
                left = np.roll(left_grid, shift, axis=(0, 1, 2)).reshape(occupied.shape)
                right = np.roll(right_grid, shift, axis=(0, 1, 2)).reshape(virtual.shape)
                # Contract the two-column side first, keeping intermediates
                # bounded by n_aux_home*2*n_ao and n_aux_home*2*n_pair.
                left_home = np.matmul(left.T, self._home_factors)
                transformed = np.matmul(left_home, right)
                exchange += transformed[:, 0, :].T @ transformed[:, 1, :]
        if not np.isfinite(exchange).all():
            raise FloatingPointError("non-finite home-auxiliary pair exchange")
        return exchange
