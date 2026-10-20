"""Ordinary matrix irreps, partner resolution and reduced invariant maps.

Goodlett et al., DOI 10.1063/5.0216738, Eq. (3): complete matrix
projectors align partners across repeated irreps. Hellmann and Gauss,
DOI 10.1021/acs.jpca.6c00311, Eqs. (4)-(12), motivates product reduction,
full reconstruction and recoupling before tensor re-sorts. Their CC pilot
is C3v; this module supplies algebra, not a general CC implementation.

Inputs use orthonormal coordinates and ordinary unitary spatial actions.
AO metrics, producer qualification, finite supports, states and retained
spaces remain caller-owned. A finite matrix audit is not a producer seal.
"""
from dataclasses import dataclass
from hashlib import sha256
import math

import numpy as np

from .symmetry_shared import Budget, FiniteGroup, _array, _freeze, _integer, _tolerance
from .symmetry_abelian import _norm


def _tol(value):
    value = _tolerance(value)
    if value > 1e-8:
        raise ValueError('representation/source tolerance must not exceed 1e-8')
    return value


def _digest(label, identities=(), arrays=()):
    h = sha256(label.encode())
    for value in identities:
        h.update(value.encode()); h.update(b'\0')
    for a in arrays:
        h.update(str(a.shape).encode()); h.update(a.tobytes())
    return h.hexdigest()


def _product_bound(a, b):
    value = a*b
    if not math.isfinite(value) or (value == 0 and a > 0 and b > 0):
        raise ValueError('nonrepresentable tensor map error bound')
    return value


def _representation(group, matrices, tolerance):
    n = matrices.shape[1]
    if not np.isfinite(matrices).all():
        raise ValueError('nonfinite representation')
    if _norm(matrices[group.identity]-np.eye(n)) > tolerance:
        raise ValueError('representation identity mismatch')
    for g, u in enumerate(matrices):
        if _norm(u.conj().T@u-np.eye(n)) > tolerance:
            raise ValueError('representation requires orthonormal coordinates')
        for h, v in enumerate(matrices):
            if _norm(u@v-matrices[group.product(g, h)]) > tolerance:
                raise ValueError('representation violates the group law')


@dataclass(frozen=True, slots=True, init=False, eq=False)
class IrrepMatrices:
    """Complete inequivalent unitary matrix irreps in exact operation order."""
    group: FiniteGroup
    matrices: tuple
    dimensions: tuple
    characters: np.ndarray
    products: np.ndarray
    conjugates: tuple
    identity: str
    tolerance: float
    required_bytes: int

    def __init__(self, group, matrices, *, budget: Budget, tolerance=1e-10):
        if not isinstance(group, FiniteGroup) or type(matrices) is not tuple or not matrices:
            raise TypeError('an admitted group and tuple of matrix irreps are required')
        g, count = group.order, len(matrices)
        dimensions = []
        for a in matrices:
            _array(a, 'complex128', 3)
            if a.shape[0] != g or not a.shape[1] or a.shape[1] != a.shape[2]:
                raise ValueError('irrep descriptor shape mismatch')
            dimensions.append(a.shape[1])
        size = group._native.memory.bytes+8192+64*sum(a.size for a in matrices)+64*count**3
        work = 128*g*g*sum(d**3 for d in dimensions)+128*g*count**3
        budget.admit(size, work)
        tolerance = _tol(tolerance)
        if sum(d*d for d in dimensions) != g:
            raise ValueError('irrep dimensions do not resolve the group order')
        for i in range(g):
            if group.antiunitary(i):
                raise ValueError('antiunitary actions need separate representation admission')
            for j in range(g):
                if group.cocycle(i, j) != (0, 0, 0):
                    raise ValueError('lattice cocycles need projective representation admission')
        for a in matrices:
            _representation(group, a, tolerance)
        characters = np.array([np.trace(a, axis1=1, axis2=2) for a in matrices])
        if _norm(characters@characters.conj().T/g-np.eye(count)) > tolerance:
            raise ValueError('matrix irreps are reducible or equivalent')
        products = np.einsum('ag,bg,cg->abc', characters, characters,
                            characters.conj(), optimize=False)/g
        integers = np.rint(products.real).astype(np.int64)
        if np.any(integers < 0) or _norm(products-integers) > tolerance:
            raise ValueError('irreducible products do not resolve with integer multiplicities')
        conjugates = []
        for row in characters:
            matches = [i for i, other in enumerate(characters) if _norm(row.conj()-other) <= tolerance]
            if len(matches) != 1:
                raise ValueError('conjugate irreps do not close uniquely')
            conjugates.append(matches[0])
        table = np.array([[group.product(i,j) for j in range(g)] for i in range(g)], np.int64)
        fields = dict(group=group, matrices=tuple(_freeze(a) for a in matrices),
            dimensions=tuple(dimensions), characters=_freeze(characters), products=_freeze(integers),
            conjugates=tuple(conjugates), tolerance=tolerance, required_bytes=size,
            identity=_digest('ordinary-matrix-irreps-v1', arrays=(table, *matrices)))
        for key, value in fields.items():
            object.__setattr__(self, key, value)

    def resolve(self, matrices, *, budget):
        return ResolvedRepresentation(self, matrices, budget=budget)

    @property
    def citation_numerics(self):
        return ('symmetry_irreducible_products',)

    def product(self, left, right, *, variance=(1, 1), budget):
        labels = (_integer(left, 'left irrep'), _integer(right, 'right irrep'))
        if any(i >= len(self.dimensions) for i in labels) or type(variance) is not tuple or len(variance) != 2:
            raise ValueError('invalid direct-product labels or variance')
        if any(type(s) is not int or s not in (-1, 1) for s in variance):
            raise ValueError('index variance must be +1 or -1')
        n = self.dimensions[left]*self.dimensions[right]
        _resolution_budget(self, n, budget)
        a, b = (self.matrices[i] if sign == 1 else self.matrices[i].conj()
                for i, sign in zip(labels, variance))
        return self.resolve(np.array([np.kron(x, y) for x, y in zip(a, b)]), budget=budget)


def _resolution_budget(irreps, n, budget):
    g = irreps.group.order
    size = irreps.required_bytes+16384+32*(g+16)*n*n
    work = 128*(g*g+g*sum(irreps.dimensions)+16)*n**3
    budget.admit(size, work)
    return size


@dataclass(frozen=True, slots=True, init=False, eq=False)
class ResolvedRepresentation:
    """Columns ordered as (irrep, multiplicity, partner), never partner signs.

    The chosen multiplicity gauge is owned and hashed. Distinct resolutions
    cannot silently share reduced arrays even if their characters agree.
    """
    irreps: IrrepMatrices
    basis: np.ndarray
    multiplicities: tuple
    offsets: tuple
    identity: str
    representation_identity: str
    maximum_residual: float
    required_bytes: int

    def __init__(self, irreps, matrices, *, budget):
        if type(irreps) is not IrrepMatrices:
            raise TypeError('admitted matrix irreps are required')
        _array(matrices, 'complex128', 3)
        g, n, n2 = matrices.shape
        if g != irreps.group.order or not n or n != n2:
            raise ValueError('representation descriptor shape mismatch')
        size = _resolution_budget(irreps, n, budget)
        tolerance = irreps.tolerance
        _representation(irreps.group, matrices, tolerance)
        columns, multiplicities, offsets = [], [], [0]
        residual = 0.
        for gamma in irreps.matrices:
            d = gamma.shape[1]
            # Complete matrix units, Goodlett Eq. (3). The single P00 range
            # chooses multiplicity vectors; Pi0 supplies aligned partners.
            p00 = np.einsum('g,gab->ab', gamma[:,0,0].conj(), matrices)*d/g
            if _norm(p00-p00.conj().T) > tolerance:
                raise ValueError('non-Hermitian partner projector')
            values, vectors = np.linalg.eigh(p00)
            if np.max(np.minimum(abs(values),abs(values-1)), initial=0.) > tolerance:
                raise ValueError('partner projector is not idempotent')
            seeds = vectors[:,values > .5]
            multiplicities.append(seeds.shape[1])
            partner = np.empty((n, seeds.shape[1], d), complex)
            for i in range(d):
                pi0 = np.einsum('g,gab->ab', gamma[:,i,0].conj(), matrices)*d/g
                partner[:,:,i] = pi0@seeds
            panel = partner.reshape(n,-1)
            for op, u in enumerate(matrices):
                moved = np.einsum('ami,ij->amj', partner, gamma[op], optimize=False).reshape(n,-1)
                residual = max(residual, _norm(u@panel-moved))
            columns.append(panel); offsets.append(offsets[-1]+panel.shape[1])
        basis = np.column_stack(columns)
        if basis.shape != (n,n):
            raise ValueError('partner spaces do not resolve the full retained dimension')
        residual = max(residual, _norm(basis.conj().T@basis-np.eye(n)))
        if residual > tolerance:
            raise ValueError('partner intertwining or orthogonality failed')
        representation_id = _digest('ordinary-representation-v1', (irreps.identity,), (matrices,))
        fields = dict(irreps=irreps, basis=_freeze(basis), multiplicities=tuple(multiplicities),
            offsets=tuple(offsets), maximum_residual=residual, required_bytes=size,
            representation_identity=representation_id,
            identity=_digest('resolved-partner-basis-v1', (representation_id,), (basis,)))
        for key, value in fields.items():
            object.__setattr__(self, key, value)

    @property
    def dimension(self):
        return len(self.basis)

    @property
    def retained_bytes(self):
        # Resolution scratch is gone; only the irrep table, basis and small
        # multiplicity/offset metadata survive for subsequent operations.
        return self.irreps.required_bytes+self.basis.nbytes+4096+64*len(self.offsets)


@dataclass(frozen=True, slots=True)
class ReducedMapApplication:
    values: np.ndarray
    scalar_products: int
    transport_scalar_products: int
    error_bound: float


def _compatible(target, source):
    if type(target) is not ResolvedRepresentation or type(source) is not ResolvedRepresentation:
        raise TypeError('resolved domain and codomain representations are required')
    if target.irreps.identity != source.irreps.identity:
        raise ValueError('domain and codomain use different matrix irrep gauges')


def _map_budget(target, source, budget):
    _compatible(target, source)
    a, b = target.dimension, source.dimension
    retained = target.retained_bytes+(0 if source is target else source.retained_bytes)
    size = retained+16384+192*a*b
    budget.admit(size, 128*(a*a*b+a*b*b+1))
    return size


@dataclass(frozen=True, slots=True, init=False, eq=False)
class InvariantTensorMap:
    """Qualified finite intertwiner: one matrix per multiplicity space.

    Full tensor = U_target [direct-sum (A_gamma tensor I_d)] U_source^H.
    Complete reconstruction is audited; no state projection or Hermitization
    occurs. Arithmetic roundoff is separate from the measured source error.
    This bounded dense audit is not a scalable integral-generation algorithm.
    """
    target: ResolvedRepresentation
    source: ResolvedRepresentation
    blocks: tuple
    identity: str
    source_identity: str
    error_bound: float
    required_bytes: int
    stored_elements: int
    composition_products: int

    @property
    def citation_numerics(self):
        return self.target.irreps.citation_numerics

    @property
    def retained_bytes(self):
        return (self.target.retained_bytes+(0 if self.source is self.target else self.source.retained_bytes)
                +4096+sum(b.nbytes for b in self.blocks)+128*len(self.blocks))

    def __init__(self, matrix, *, target, source, source_tolerance, budget):
        size = _map_budget(target, source, budget)
        _array(matrix, 'complex128', 2)
        if matrix.shape != (target.dimension, source.dimension) or not np.isfinite(matrix).all():
            raise ValueError('finite tensor map descriptor or values are invalid')
        tolerance = _tol(source_tolerance)
        coupled = target.basis.conj().T@matrix@source.basis
        selected = np.zeros_like(coupled)
        blocks = []
        for label, d in enumerate(target.irreps.dimensions):
            a, b = target.multiplicities[label], source.multiplicities[label]
            ts = slice(target.offsets[label],target.offsets[label+1])
            ss = slice(source.offsets[label],source.offsets[label+1])
            full = coupled[ts,ss].reshape(a,d,b,d)
            block = full[:,0,:,0].copy()
            # Keep a specified partner, not an averaged/repaired matrix.
            # Reconstruct all others, including missing re-sort elements.
            selected[ts,ss] = np.kron(block,np.eye(d))
            blocks.append(_freeze(block))
        restored = target.basis@selected@source.basis.conj().T
        error = _norm(matrix-restored)
        if error > tolerance:
            raise ValueError('finite tensor fails full partner reconstruction; use unreduced source')
        source_id = _digest('finite-invariant-map-source-v1',
            (target.identity,source.identity), (matrix,))
        self._seal(target, source, blocks, error, source_id, size, 0)

    def _seal(self, target, source, blocks, error, source_id, size, products):
        fields = dict(target=target, source=source, blocks=tuple(blocks), error_bound=error,
            source_identity=source_id, required_bytes=size, stored_elements=sum(b.size for b in blocks),
            identity=_digest('reduced-invariant-map-v1', (source_id,), blocks), composition_products=products)
        for key, value in fields.items():
            object.__setattr__(self, key, value)

    def to_full(self, *, budget):
        _map_budget(self.target,self.source,budget)
        coupled = np.zeros((self.target.dimension,self.source.dimension),complex)
        for i,(block,d) in enumerate(zip(self.blocks,self.target.irreps.dimensions)):
            coupled[self.target.offsets[i]:self.target.offsets[i+1],
                    self.source.offsets[i]:self.source.offsets[i+1]] = np.kron(block,np.eye(d))
        return _freeze(self.target.basis@coupled@self.source.basis.conj().T)

    def apply(self, values, *, budget):
        _array(values,'complex128',2)
        a,b = self.target.dimension,self.source.dimension
        if values.shape[0] != b or not values.shape[1]:
            raise ValueError('tensor map input dimensions disagree')
        batch = values.shape[1]
        budget.admit(self.retained_bytes+128*(a+b)*batch, 128*(a*a+b*b+a*b)*batch)
        if not np.isfinite(values).all():
            raise ValueError('nonfinite tensor map input')
        x = self.source.basis.conj().T@values
        y = np.zeros((a,batch),complex); products=0
        for i,(block,d) in enumerate(zip(self.blocks,self.target.irreps.dimensions)):
            part = x[self.source.offsets[i]:self.source.offsets[i+1]].reshape(block.shape[1],d,batch)
            y[self.target.offsets[i]:self.target.offsets[i+1]] = np.einsum('ab,bik->aik',block,part).reshape(-1,batch)
            products += block.size*d*batch
        result = self.target.basis@y
        error = _product_bound(self.error_bound,_norm(values))
        if not np.isfinite(result).all():
            raise ValueError('nonrepresentable tensor map result or error bound')
        return ReducedMapApplication(_freeze(result),products,(a*a+b*b)*batch,error)

    def compose(self, right, *, budget):
        if type(right) is not InvariantTensorMap or self.source.identity != right.target.identity:
            raise ValueError('composition needs the same resolved intermediate gauge; recouple first')
        size = _map_budget(self.target,right.source,budget)
        products = sum(a.shape[0]*a.shape[1]*b.shape[1] for a,b in zip(self.blocks,right.blocks))
        budget.admit(size+self.retained_bytes+right.retained_bytes,128*(products+1))
        blocks = tuple(_freeze(a@b) for a,b in zip(self.blocks,right.blocks))
        norms = [math.hypot(*(math.sqrt(d)*_norm(b) for d,b in zip(m.target.irreps.dimensions,m.blocks)))
                 for m in (self,right)]
        error = (_product_bound(self.error_bound,norms[1])+_product_bound(right.error_bound,norms[0])
                 +_product_bound(self.error_bound,right.error_bound))
        if not math.isfinite(error) or any(not np.isfinite(b).all() for b in blocks):
            raise ValueError('nonrepresentable reduced composition')
        result = object.__new__(InvariantTensorMap)
        result._seal(self.target,right.source,blocks,error,
            _digest('invariant-map-composition-v1',(self.identity,right.identity)),size,products)
        return result


def recoupling(target, source, *, tensor_shape, permutation, budget):
    """Resolve an explicit tensor-leg permutation, including multiplicity mix.

    Return the intertwiner between two coupled bases. A partner permutation
    alone is insufficient in general. Variance is carried by the two supplied
    representations; a permutation that does not intertwine them refuses.
    """
    _map_budget(target,source,budget)
    if type(tensor_shape) is not tuple or type(permutation) is not tuple:
        raise TypeError('tensor shape and permutation must be tuples')
    shape = tuple(_integer(n,'tensor dimension',positive=True) for n in tensor_shape)
    if (not shape or len(permutation)!=len(shape) or any(type(i) is not int for i in permutation)
            or sorted(permutation)!=list(range(len(shape)))
            or math.prod(shape)!=source.dimension or target.dimension!=source.dimension):
        raise ValueError('invalid tensor recoupling shape or permutation')
    n = source.dimension
    order = np.arange(n).reshape(shape).transpose(permutation).ravel()
    transform = np.eye(n,dtype=complex)[order]
    return InvariantTensorMap(transform,target=target,source=source,
        source_tolerance=min(target.irreps.tolerance,source.irreps.tolerance),budget=budget)
