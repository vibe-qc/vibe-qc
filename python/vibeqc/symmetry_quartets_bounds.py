"""Bind MANTA's existing analytic product envelopes to an immutable P0 source.

No envelope, integral kernel or BUPO hierarchy is implemented here. MANTA
supplies Q >= integral |phi_a phi_b| and V >= sup Coulomb(|phi_a phi_b|).
Then |(ab|cd)| <= min(Q_ab V_cd, Q_cd V_ab); a tile Frobenius bound includes
the number of AO entries. Bounds concern the exact Coulomb integrals in the
declared shell convention, not libint arithmetic error or equivariance.
See MANTA's absolute Cartesian polynomial envelopes and Neese et al.,
doi:10.1021/acs.jpca.4c07415, Sec. 2.5, for the distinct extent/BUPO role.
"""
from dataclasses import dataclass, field
from hashlib import sha256
import math

import numpy as np

from . import _vibeqc_core as core
from .symmetry_shared import Budget, _freeze, _integer
from .symmetry_quartets import _shell_signature


def _upper_product(a, b):
    if a == 0. or b == 0.:
        return 0.
    am, ae = math.frexp(a)
    bm, be = math.frexp(b)
    try:
        value = math.ldexp(math.nextafter(am*bm, math.inf), ae+be)
    except OverflowError as exc:
        raise ValueError('nonrepresentable producer bound; use full audit') from exc
    value = math.nextafter(value, math.inf)
    if not math.isfinite(value):
        raise ValueError('nonrepresentable producer bound; use full audit')
    return value


def _upper_hypot(a, b):
    if a == 0.:
        return b
    if b == 0.:
        return a
    value = math.nextafter(math.hypot(a, b), math.inf)
    if not math.isfinite(value):
        raise ValueError('nonrepresentable producer bound; use full audit')
    return value


def _options():
    options = core._manta.ExtentOptions()
    # Only full-space Q/V are consumed. Fixed equal radii avoid an unused
    # extent search; this radius is NOT a finite-support cutoff.
    options.minimum_radius = options.maximum_radius = 1.
    options.moment_order = 0
    return options


def _requirements(basis, provider, budget):
    if type(provider) is not core._manta.ReferenceJKBuilder:
        raise TypeError('producer bounds require the immutable molecular P0 source')
    n, ns = int(basis.nbasis), int(basis.nshells)
    if (n, ns) != (provider.plan.n_basis, provider.plan.n_shells):
        raise ValueError('producer-bound basis dimensions disagree')
    native = core._manta.ResourceBudget()
    native.memory_bytes = budget.maximum_bytes
    native.shell_calls = native.primitive_quartets = budget.maximum_work
    native.scalar_terms = native.kernel_terms = budget.maximum_work
    plan = core._manta.plan_pair_extents(basis, _options(), native)
    # Include native result, exported Python objects, both shell snapshots,
    # immutable arrays and the already-live integral producer. Conservative
    # lifetime accounting, not an RSS guarantee.
    size = (plan.modeled_peak_bytes + provider.plan.modeled_peak_bytes
            + 4096 + 2048*plan.shell_pairs + 8*provider.plan.basis_payload_bytes)
    work = plan.kernel_terms + plan.scalar_terms + 256*plan.shell_pairs
    budget.admit(size, work)
    return Budget(size, work), native


def molecular_coulomb_bounds_requirements(basis, provider, *, budget: Budget):
    """Count-only reservation before bound computation or shell snapshots."""
    return _requirements(basis, provider, budget)[0]


@dataclass(frozen=True, slots=True, init=False, eq=False)
class MolecularCoulombBounds:
    """Owned adapter to native MANTA Q/V envelopes for one exact P0 basis.

    This is not a general producer attestation or a symmetry certificate.
    Caller-declared arrays/callbacks cannot enter. Overflow is a refusal;
    positive underflow remains positive. Native envelope roundoff is not an
    interval proof and the receipt keeps it separate from approximation.
    """
    _provider: object = field(repr=False)
    _q: np.ndarray = field(repr=False)
    _v: np.ndarray = field(repr=False)
    shell_sizes: np.ndarray
    source_identity: str
    envelope_identity: str
    identity: str
    required_bytes: int
    required_work: int
    shell_pairs: int
    primitive_pairs: int
    envelope_evaluations: int

    def __init__(self, basis, provider, *, budget: Budget):
        reservation, native_budget = _requirements(basis, provider, budget)
        signature = _shell_signature(provider.shells)
        if signature != _shell_signature(basis.shells()):
            raise ValueError('producer-bound basis snapshot disagrees with source')
        extents = core._manta.compute_pair_extents(basis, _options(), native_budget)
        if signature != _shell_signature(basis.shells()):
            raise ValueError('producer-bound basis changed during construction')
        ns = len(signature)
        q, v = np.empty(ns*(ns+1)//2), np.empty(ns*(ns+1)//2)
        for i, pair in enumerate(extents.pairs):
            a,b = map(int, pair.shells)
            if a*(a+1)//2+b != i or not 0 <= b <= a < ns:
                raise ValueError('unexpected native product-envelope ordering')
            q[i],v[i] = pair.l1_norm_bound,pair.potential_bound
        if (len(extents.pairs) != len(q) or not np.isfinite(q).all()
                or not np.isfinite(v).all() or np.any(q < 0.) or np.any(v < 0.)):
            raise ValueError('invalid native product envelopes')
        sizes = np.array([2*s[1]+1 if s[2] else (s[1]+1)*(s[1]+2)//2
                          for s in signature], np.int64)
        source = provider.source_identity
        receipt = extents.receipt
        digest = sha256(b'vibeqc-manta-P0-QV-envelope-v1\0')
        digest.update(source.encode())
        digest.update(receipt.source_identity.encode())
        digest.update(sizes.tobytes()+q.tobytes()+v.tobytes())
        for key,value in dict(_provider=provider, _q=_freeze(q), _v=_freeze(v),
                shell_sizes=_freeze(sizes), source_identity=source,
                envelope_identity=receipt.source_identity, identity=digest.hexdigest(),
                required_bytes=reservation.maximum_bytes, required_work=reservation.maximum_work,
                shell_pairs=int(receipt.shell_pairs), primitive_pairs=int(receipt.primitive_pairs),
                envelope_evaluations=int(receipt.envelope_evaluations)).items():
            object.__setattr__(self,key,value)

    def _validate(self, provider):
        if (type(provider) is not core._manta.ReferenceJKBuilder
                or provider.source_identity != self.source_identity
                or self._provider.source_identity != self.source_identity):
            raise ValueError('producer bounds belong to a different numerical source')

    def tile_frobenius_bound(self, quartet):
        if not isinstance(quartet, tuple) or len(quartet) != 4:
            raise TypeError('four shell indices are required')
        indices = tuple(_integer(x, 'bound shell index') for x in quartet)
        if max(indices) >= len(self.shell_sizes):
            raise ValueError('bound shell index outside source')
        a,b,c,d = indices
        def pair(i,j):
            i,j = max(i,j),min(i,j)
            return i*(i+1)//2+j
        ab,cd = pair(a,b),pair(c,d)
        entry = min(_upper_product(float(self._q[ab]),float(self._v[cd])),
                    _upper_product(float(self._q[cd]),float(self._v[ab])))
        entries = math.prod(int(self.shell_sizes[s]) for s in indices)
        factor = math.nextafter(math.sqrt(entries),math.inf)
        return _upper_product(entry,factor)

    @property
    def receipt(self):
        return dict(schema='vibeqc-manta-P0-QV-envelope-v1', identity=self.identity,
            source_identity=self.source_identity, envelope_identity=self.envelope_identity,
            kernel='full-coulomb', primitive_precision=0., finite_domain='all molecular quartets',
            bound_kind='absolute-product-L1-potential', normalization='libint-shell-axial-v1',
            arithmetic_roundoff_included=False, operator_equivariance_certified=False,
            integral_shell_calls=0, shell_pairs=self.shell_pairs, primitive_pairs=self.primitive_pairs,
            envelope_evaluations=self.envelope_evaluations,
            array_bytes=self._q.nbytes+self._v.nbytes+self.shell_sizes.nbytes,
            required_bytes=self.required_bytes, required_work=self.required_work)


def _bounds_overhead(bounds, provider, ns):
    if bounds is None:
        return 0, 0
    from .symmetry_quartets_hierarchy import MolecularHierarchicalBounds, _COVERAGE_CACHE_BYTES
    if type(bounds) not in (MolecularCoulombBounds, MolecularHierarchicalBounds):
        raise TypeError('owned native product-envelope bounds are required')
    bounds._validate(provider)
    if len(bounds.shell_sizes) != ns:
        raise ValueError('bound shell inventory disagrees with source')
    depth = bounds.receipt.get("depth", 0)
    cache_bytes = _COVERAGE_CACHE_BYTES if type(bounds) is MolecularHierarchicalBounds else 0
    return bounds.required_bytes+cache_bytes, bounds.required_work+256*ns**4*(depth+1)**2


from .symmetry_quartets_screening import _StreamingScreenAudit


class _ProducerBoundAudit(_StreamingScreenAudit):
    """Bound-only omission, followed by full audit of every retained orbit."""
    policy = 'producer-QV-schedule-first-fit-v1'

    def __init__(self, source, count, cap, bounds):
        super().__init__(source, count, cap)
        self.bounds = bounds
        self.queries = 0

    def try_omit(self, orbit, offsets):
        norm = 0.
        # Complete member closure matters for nonunitary Cartesian actions.
        # Scalar bounds need not be equal in rotated AO coordinates.
        for member in orbit.members:
            norm = _upper_hypot(norm, self.bounds.tile_frobenius_bound(member.shells))
            self.queries += 1
        candidate = _upper_hypot(self.omitted, norm)
        if candidate > self.cap:
            return False
        self.omitted = candidate
        self.visited += 1
        self.reps += 1
        self.tiles += len(orbit.members)
        self.values += sum(math.prod(int(offsets[s+1]-offsets[s])
            for s in member.shells) for member in orbit.members)
        return True

    def observe(self, orbit, original, norm, residual, offsets):
        # This orbit failed bound-only omission. Keep it even if an exact
        # a posteriori norm would fit: no unreported audit reads are skipped.
        from .symmetry_quartets_streaming import _digest_tile
        if not math.isfinite(residual):
            raise ValueError('nonrepresentable retained-source residual')
        index = self.visited
        self.visited += 1
        self.mask[index//8] |= 1 << (index%8)
        self.retained = _upper_hypot(self.retained, residual)
        _digest_tile(self.digest, orbit.representative, original)
