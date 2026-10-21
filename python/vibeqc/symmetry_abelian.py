"""Experimental complex Abelian selection and finite-ERI J/K execution.

Kitsaras and Stopkowicz, arXiv:2602.12457v1, Eqs. (16)-(25)
(journal DOI 10.1021/acs.jpca.6c00998); Stanton et al.,
J. Chem. Phys. 94, 4334 (1991), DOI 10.1063/1.460620.
This adapter consumes an already computed, symmetry-adapted finite tensor.
It neither evaluates fewer integrals nor authorizes a production source.
"""
from dataclasses import dataclass
from hashlib import sha256
from itertools import product
import math

import numpy as np

from .symmetry_shared import (
    Budget, FiniteGroup, OperatorContract, SpaceIdentity,
    _array, _component_scale, _freeze, _integer, _scaled_frobenius, _tolerance,
)


@dataclass(frozen=True, slots=True, init=False)
class AbelianCharacters:
    """Complete unitary one-dimensional irreps, in supplied operation order.

    Ordinary finite groups only: no antiunitary grading or lattice cocycle.
    Molecular subgroups are allowed, but merged real conjugate-pair rows are
    not one-dimensional irreps. Labels need not put the trivial irrep first.
    """
    group: FiniteGroup
    characters: np.ndarray
    products: np.ndarray
    conjugates: np.ndarray
    trivial: int
    required_bytes: int
    required_work: int

    def __init__(self, group, characters, *, budget: Budget, tolerance=1e-10):
        if not isinstance(group, FiniteGroup):
            raise TypeError('characters require an admitted FiniteGroup')
        _array(characters, 'complex128', 2)
        n = group.order
        if characters.shape != (n, n):
            raise ValueError('complete Abelian character table must be order by order')
        required_bytes = 4096 + group._native.memory.bytes + 256*n*n
        required_work = 128*n**4 + 4096
        budget.admit(required_bytes, required_work)
        tolerance = _tolerance(tolerance)
        if tolerance > 1e-8:
            raise ValueError('character tolerance must not exceed 1e-8')
        if any(group.antiunitary(g) for g in range(n)):
            raise ValueError('antiunitary time reversal requires a separate action')
        for g, h in product(range(n), repeat=2):
            if group.product(g, h) != group.product(h, g):
                raise ValueError('non-Abelian products require partner resolution')
            if group.cocycle(g, h) != (0, 0, 0):
                raise ValueError('lattice cocycles require Bloch/projective representations')
        chars = characters.copy()
        if not np.isfinite(chars).all():
            raise ValueError('nonfinite characters')
        if (np.max(np.abs(chars[:, group.identity] - 1)) > tolerance
                or np.max(np.abs(np.abs(chars) - 1)) > tolerance
                or np.max(np.abs(chars @ chars.conj().T / n - np.eye(n))) > tolerance):
            raise ValueError('characters must be distinct unitary one-dimensional irreps')
        for g, h in product(range(n), repeat=2):
            if np.max(np.abs(chars[:, g]*chars[:, h] - chars[:, group.product(g, h)])) > tolerance:
                raise ValueError('characters violate the supplied group law')
        def identify(row):
            distances = np.max(np.abs(chars - row), axis=1)
            matches = np.flatnonzero(distances <= tolerance)
            if len(matches) != 1:
                raise ValueError('character products or conjugates do not close uniquely')
            return int(matches[0])
        products = np.array([[identify(a*b) for b in chars] for a in chars], np.int64)
        conjugates = np.array([identify(a.conj()) for a in chars], np.int64)
        for key, value in dict(group=group, characters=_freeze(chars),
                products=_freeze(products), conjugates=_freeze(conjugates),
                trivial=identify(np.ones(n)), required_bytes=required_bytes,
                required_work=required_work).items():
            object.__setattr__(self, key, value)

    def allows(self, labels, variance, *, target=None):
        """Whether product chi[label]**variance equals the target irrep.

        +1 is an unconjugated character and -1 its conjugate. For chemist
        ERIs (p* q | r* s) the signature is (-1,+1,-1,+1).
        This selection rule does not assert that a source obeys it.
        """
        labels, variance = tuple(labels), tuple(variance)
        if len(labels) != len(variance):
            raise ValueError('one variance is required for every index')
        target = self.trivial if target is None else _integer(target, 'target')
        if not 0 <= target < self.group.order:
            raise ValueError('invalid target irrep')
        result = self.trivial
        for label, sign in zip(labels, variance):
            label = _integer(label, 'irrep label')
            if not 0 <= label < self.group.order or type(sign) is not int or sign not in (-1, 1):
                raise ValueError('invalid irrep label or index variance')
            result = int(self.products[result, label if sign == 1 else self.conjugates[label]])
        return result == target


def _norm(array):
    """Scaled Frobenius norm, rejecting a nonrepresentable error bound."""
    scale = _component_scale(array)
    result = scale * _scaled_frobenius(array, scale)
    if not math.isfinite(result):
        raise ValueError('nonfinite tensor or contraction error bound')
    return result


@dataclass(frozen=True, slots=True)
class AbelianJKResult:
    j: np.ndarray
    k: np.ndarray
    scalar_products: int
    block_contractions: int
    error_bound: float


@dataclass(frozen=True, slots=True, init=False)
class AbelianERIPlan:
    """Snapshot of allowed blocks in one complete finite chemist ERI tensor.

    Construction checks the aggregate forbidden-block Frobenius norm against
    the explicit source tolerance; it never averages the source or density.
    If rejected, use the original unreduced builder. The numerical omission
    bounds each omitted-block J/K Frobenius contribution in exact arithmetic
    by omitted_norm * ||D||_F; contraction roundoff is separate. This certifies
    only the supplied array, not its producer, metric, spin or retained space.
    """
    irreps: AbelianCharacters
    labels: np.ndarray
    space: SpaceIdentity
    contract: OperatorContract
    source_digest: str
    blocks: tuple
    indices: tuple
    omitted_norm: float
    source_tolerance: float
    stored_elements: int
    dense_elements: int
    retained_bytes: int
    required_bytes: int
    required_work: int

    def __init__(self, eri, labels, irreps, *, space, contract,
                 source_tolerance, budget: Budget):
        if not isinstance(irreps, AbelianCharacters):
            raise TypeError('an admitted Abelian character table is required')
        if not isinstance(space, SpaceIdentity) or not isinstance(contract, OperatorContract):
            raise TypeError('named space and operator contract are required')
        _array(eri, 'complex128', 4)
        _array(labels, 'int64', 1)
        n, g = len(labels), irreps.group.order
        if n < 1 or eri.shape != (n,)*4:
            raise ValueError('ERI shape and orbital labels disagree')
        # Conservative logical admission includes borrowed input, block scan,
        # immutable byte snapshots and metadata. No dense n^4 mask is built.
        required_bytes = irreps.required_bytes + 16384 + 4*eri.nbytes + 128*n*n + 256*g**4
        required_work = irreps.required_work + 64*n**4 + 128*g**4
        budget.admit(required_bytes, required_work)
        source_tolerance = _tolerance(source_tolerance, 'source tolerance')
        if np.any(labels < 0) or np.any(labels >= g):
            raise ValueError('orbital irrep outside character table')
        if not np.isfinite(eri).all():
            raise ValueError('nonfinite ERI source')
        labels = _freeze(labels)
        indices = tuple(_freeze(np.flatnonzero(labels == a)) for a in range(g))
        occupied = tuple(a for a in range(g) if len(indices[a]))
        blocks, omitted, stored = [], 0., 0
        digest = sha256(b'vibeqc-abelian-chemist-eri-v1\0')
        digest.update(repr(space).encode())
        digest.update(repr(contract).encode())
        digest.update(irreps.characters.tobytes())
        digest.update(labels.tobytes())
        # Stream the input buffer into the digest without another n^4 copy.
        digest.update(memoryview(eri).cast('B'))
        for key in product(occupied, repeat=4):
            block = eri[np.ix_(*(indices[a] for a in key))]
            if irreps.allows(key, (-1, 1, -1, 1)):
                blocks.append((key, _freeze(block)))
                stored += block.size
            else:
                omitted = math.hypot(omitted, _norm(block))
        if not math.isfinite(omitted) or omitted > source_tolerance:
            raise ValueError('finite ERI source violates Abelian selection rules; use unreduced source')
        retained_bytes = (irreps.characters.nbytes + irreps.products.nbytes
                          + irreps.conjugates.nbytes + labels.nbytes
                          + sum(x.nbytes for x in indices) + 16*stored)
        for key, value in dict(irreps=irreps, labels=labels, space=space, contract=contract,
                source_digest=digest.hexdigest(), blocks=tuple(blocks), indices=indices,
                omitted_norm=omitted, source_tolerance=source_tolerance,
                stored_elements=stored, dense_elements=n**4, retained_bytes=retained_bytes,
                required_bytes=required_bytes, required_work=required_work).items():
            object.__setattr__(self, key, value)

    @property
    def production_reduction_authorized(self):
        return False

    @property
    def citation_numerics(self):
        return ('complex_abelian_tensor_blocks',)

    def jk(self, density, *, budget: Budget):
        """Contract spin-summed or per-spin D without assuming state symmetry.

        J[p,q] = sum E[p,q,r,s] D[s,r]
        K[p,s] = sum E[p,q,r,s] D[q,r]

        Density is contravariant. In these symmetry-adapted coordinates,
        D -> U D U^dagger; F -> U^-dagger F U^-1. No real-only eightfold
        ERI permutation, conjugation of D, or final Hermitization is used.
        """
        _array(density, 'complex128', 2)
        n = len(self.labels)
        if density.shape != (n, n):
            raise ValueError('density shape disagrees with ERI space')
        budget.admit(self.required_bytes + 256*n*n,
                     self.required_work + 32*self.stored_elements + 64*n*n)
        if not np.isfinite(density).all():
            raise ValueError('nonfinite density')
        density_norm = _norm(density)
        error_bound = self.omitted_norm * density_norm
        if (not math.isfinite(error_bound)
                or (error_bound == 0 and self.omitted_norm > 0 and density_norm > 0)):
            raise ValueError('nonrepresentable contraction error bound')
        j, k = np.zeros((n, n), complex), np.zeros((n, n), complex)
        for (a, b, c, d), block in self.blocks:
            p, q, r, s = (self.indices[x] for x in (a, b, c, d))
            j[np.ix_(p, q)] += np.einsum('pqrs,sr->pq', block, density[np.ix_(s, r)], optimize=False)
            k[np.ix_(p, s)] += np.einsum('pqrs,qr->ps', block, density[np.ix_(q, r)], optimize=False)
        if not np.isfinite(j).all() or not np.isfinite(k).all():
            raise ValueError('nonfinite J/K contraction')
        return AbelianJKResult(j, k, 2*self.stored_elements, 2*len(self.blocks), error_bound)
