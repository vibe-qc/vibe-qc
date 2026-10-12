"""Streaming real molecular quartet orbits over a source-audited provider.

Davidson, J. Chem. Phys. 62, 400 (1975), doi:10.1063/1.430484:
pair stabilizers resolve double cosets; their sizes depend on the orbit.
Here H_a orbits on G/H_b enumerate pair products, with pair exchange
resolved separately. Full AO blocks retain all angular partner information.
This is not a non-Abelian reduced-irrep tensor. Optional fixed screening
is certified by the complete numerical audit, never by this topology.
"""
from dataclasses import dataclass
from hashlib import sha256
from itertools import product
import math

import numpy as np

from .symmetry_shared import BlockSpaceAction, Budget, FiniteGroup, _array, _freeze, _integer, _text, _tolerance
from .symmetry_quartets import (
    QuartetImage, QuartetJKResult, QuartetWork, _MantaSource, _PERMUTATIONS,
    _QuartetStabilizer, _norm, _quartet_requirements, _read, _real_shell_actions, _transform,
)


def _pair_requirements(ns, ng):
    pairs = ns*(ns+1)//2
    return 262144 + 64*ng*pairs + 256*pairs + 8192*ng, 128*ng*ng*max(ns, pairs)


def _streaming_requirements(n, ns, ng, maximum_shell, provider_bytes, group_bytes, mode):
    pair_bytes, pair_work = _pair_requirements(ns, ng)
    byte_count = (provider_bytes + group_bytes + pair_bytes + 8192*ng*n
                  + 512*n*n + 96*maximum_shell**4)
    # Audit and execution still touch all ordered members. Only retained
    # schedule memory is quadratic in shell count; work is not quadratic.
    _, work = _quartet_requirements(n, ns, ng, maximum_shell, 0, 0, mode)
    return byte_count, work + pair_work


@dataclass(frozen=True, slots=True)
class QuartetTopology:
    representative: tuple[int, int, int, int]
    members: tuple[QuartetImage, ...]
    stabilizer: _QuartetStabilizer


@dataclass(frozen=True, slots=True, init=False, eq=False)
class PairQuartetSchedule:
    """Reiterable shell topology, retaining O(|G| n_shells**2) integers.

    Index algebra only: no source, angular-matrix or state qualification.
    At most one O(8 |G|) quartet orbit is produced at a time. There is no
    list of all quartets, pair products, representative hashes or images.
    """
    group: FiniteGroup
    destinations: np.ndarray
    pairs: np.ndarray
    pair_maps: np.ndarray
    orbit_representatives: np.ndarray
    transporters: np.ndarray
    representatives: np.ndarray
    inverses: np.ndarray
    mode: str
    orbit_count: int
    required_bytes: int
    required_work: int

    def __init__(self, group, destinations, *, budget: Budget, mode='spatial'):
        if not isinstance(group, FiniteGroup) or mode not in ('none', 'permutation', 'spatial'):
            raise ValueError('ordinary molecular group and explicit quartet mode required')
        _array(destinations, 'int64', 2)
        ng, ns = destinations.shape
        if ng != group.order or ns < 1:
            raise ValueError('one nonempty shell permutation per group operation required')
        byte_count, work = _pair_requirements(ns, ng)
        byte_count += group._native.memory.bytes
        budget.admit(byte_count, work)
        expected = np.arange(ns)
        if not np.array_equal(destinations[group.identity], expected):
            raise ValueError('shell identity action failed')
        for g in range(ng):
            if group.antiunitary(g) or any(group.cocycle(g, h) != (0, 0, 0) for h in range(ng)):
                raise ValueError('pair schedule requires ordinary non-antiunitary molecular actions')
            if not np.array_equal(np.sort(destinations[g]), expected):
                raise ValueError('shell action must be a permutation')
        for g, h in product(range(ng), repeat=2):
            if not np.array_equal(destinations[g, destinations[h]], destinations[group.product(g, h)]):
                raise ValueError('shell permutations violate the group law')
        pairs = np.array([(a, b) for a in range(ns) for b in range(a, ns)], dtype=np.int64)
        npair = len(pairs)
        maps = np.empty((ng, npair), np.int64)
        for g in range(ng):
            mapped = np.sort(destinations[g, pairs], axis=1)
            a, b = mapped.T
            maps[g] = a*ns-a*(a-1)//2+(b-a)
        orbit_reps = np.full(npair, -1, np.int64)
        transporters = np.full(npair, -1, np.int64)
        reps = []
        operations = (group.identity,) + tuple(g for g in range(ng) if g != group.identity)
        for a in range(npair):
            if orbit_reps[a] >= 0:
                continue
            reps.append(a)
            for g in operations:
                target = int(maps[g, a])
                if orbit_reps[target] < 0:
                    orbit_reps[target], transporters[target] = a, g
        inverses = np.array([group.inverse(g) for g in range(ng)], np.int64)
        fixed = [int(np.count_nonzero(maps[g] == np.arange(npair))) for g in range(ng)]
        # Burnside on unordered pairs of pair objects: the exchange term is
        # fix(g^2), not fix(g). This includes equal-pair stabilizers exactly.
        numerator = sum(fixed[g]**2+fixed[group.product(g, g)] for g in range(ng))
        if numerator % (2*ng):
            raise ValueError('nonintegral pair-product orbit count')
        count = (ns**4 if mode == 'none' else npair*(npair+1)//2
                 if mode == 'permutation' else numerator//(2*ng))
        for name, value in dict(group=group, destinations=_freeze(destinations), pairs=_freeze(pairs),
                pair_maps=_freeze(maps), orbit_representatives=_freeze(orbit_reps),
                transporters=_freeze(transporters), representatives=_freeze(np.array(reps, np.int64)),
                inverses=_freeze(inverses), mode=mode, orbit_count=count,
                required_bytes=byte_count, required_work=work).items():
            object.__setattr__(self, name, value)

    def __len__(self):
        return self.orbit_count

    @property
    def retained_array_bytes(self):
        return sum(a.nbytes for a in (self.destinations, self.pairs, self.pair_maps,
            self.orbit_representatives, self.transporters, self.representatives, self.inverses))

    def _expand(self, representative):
        ng, identity = self.group.order, self.group.identity
        operations = ((identity,) if self.mode != 'spatial' else
                      (identity,) + tuple(g for g in range(ng) if g != identity))
        permutations = range(1 if self.mode == 'none' else 8)
        members, masks = {}, bytearray(ng)
        for g in operations:
            mapped = tuple(int(self.destinations[g, s]) for s in representative)
            for p in permutations:
                target = tuple(mapped[i] for i in _PERMUTATIONS[p])
                members.setdefault(target, QuartetImage(target, g, p))
                if target == representative:
                    masks[g] |= 1 << p
        stabilizer = _QuartetStabilizer(bytes(masks))
        if len(members)*len(stabilizer) != len(operations)*len(permutations):
            raise ValueError('quartet orbit-stabilizer identity failed')
        return QuartetTopology(representative, tuple(members.values()), stabilizer)

    def __iter__(self):
        if self.mode == 'none':
            for quartet in product(range(self.destinations.shape[1]), repeat=4):
                yield self._expand(quartet)
            return
        npair = len(self.pairs)
        if self.mode == 'permutation':
            for a in range(npair):
                for b in range(a, npair):
                    yield self._expand(tuple(map(int, self.pairs[a]))+tuple(map(int, self.pairs[b])))
            return
        for a in self.representatives:
            stabilizer = np.flatnonzero(self.pair_maps[:, a] == a)
            seen = np.zeros(npair, dtype=bool)
            for b in range(npair):
                # Exchange is already accounted for between distinct G pair
                # orbits by keeping only the lower orbit on the left.
                if self.orbit_representatives[b] < a or seen[b]:
                    continue
                images = self.pair_maps[stabilizer, b]
                seen[images] = True
                if self.orbit_representatives[b] == a:
                    # Bring the exchanged left pair b back to a, then resolve
                    # its right pair under H_a. The transporter is a matrix
                    # group operation, not an assumed Abelian sign.
                    inverse = self.inverses[self.transporters[b]]
                    exchanged = self.pair_maps[inverse, a]
                    partner = int(np.min(self.pair_maps[stabilizer, exchanged]))
                    if b > partner:
                        continue
                yield self._expand(tuple(map(int, self.pairs[a]))+tuple(map(int, self.pairs[b])))


def _digest_tile(digest, representative, values):
    digest.update(np.asarray(representative, dtype='>i8').tobytes())
    digest.update(values.tobytes())


@dataclass(frozen=True, slots=True, init=False, eq=False)
class StreamingShellQuartetPlan:
    """Complete source audit with a streamed pair-stabilizer inventory.

    All qualification calls remain mandatory. A single ordered digest binds
    representatives instead of retaining one hash per quartet. Builds check
    that digest before returning either target; no partial result escapes.
    """
    source: object
    source_identity: str
    offsets: np.ndarray
    group: FiniteGroup
    inverse_blocks: tuple
    orbits: PairQuartetSchedule
    qualification_work: QuartetWork
    source_residual_norm: float
    maximum_stabilizer_residual: float
    source_tolerance: float
    required_bytes: int
    required_work: int
    retained_array_bytes: int
    mode: str
    representative_digest: str
    audit_identity: str
    screening: object
    bounds: object
    bound_queries: int
    bound_lookup_probes: int
    hierarchy_closure_refusals: int

    def __init__(self, group, actions, source, *, source_tolerance, budget: Budget, mode='spatial', maximum_omitted_norm=None, bounds=None):
        if not isinstance(group, FiniteGroup) or mode not in ('spatial', 'permutation', 'none'):
            raise ValueError('admitted molecular group and explicit quartet reduction mode required')
        if not isinstance(actions, tuple) or len(actions) != group.order or not all(isinstance(a, BlockSpaceAction) for a in actions):
            raise TypeError('one compact shell action per group operation is required')
        first = actions[group.identity]
        n, ns, ng = first.dimension, len(first.destinations), group.order
        provider_bytes = _integer(source.modeled_peak_bytes, 'provider bytes')
        bound_bytes = bound_work = 0
        if bounds is not None:
            from .symmetry_quartets_screening import _immutable_source
            from .symmetry_quartets_bounds import _bounds_overhead
            if maximum_omitted_norm is None or mode != 'spatial':
                raise ValueError('producer bounds require an explicit spatial streaming mask')
            _immutable_source(source, session=False)
            bound_bytes, bound_work = _bounds_overhead(bounds, source._provider, ns)
        required_bytes, required_work = _streaming_requirements(
            n, ns, ng, 0, provider_bytes, group._native.memory.bytes, mode)
        required_bytes += bound_bytes
        required_work += bound_work
        budget.admit(required_bytes, required_work)
        maximum_shell = int(np.max(np.diff(first.offsets)))
        required_bytes, required_work = _streaming_requirements(
            n, ns, ng, maximum_shell, provider_bytes, group._native.memory.bytes, mode)
        required_bytes += bound_bytes
        required_work += bound_work
        budget.admit(required_bytes, required_work)
        source_tolerance = _tolerance(source_tolerance, 'source tolerance')
        _array(source.shell_offsets, 'int64', 1)
        if not np.array_equal(source.shell_offsets, first.offsets):
            raise ValueError('source and action shell layouts disagree')
        identity = source.source_identity
        _text(identity, 'source identity')
        inverse_blocks = _real_shell_actions(group, actions)
        schedule = PairQuartetSchedule(group, np.stack([a.destinations for a in actions]),
            budget=budget, mode=mode)
        screen_audit = None
        if maximum_omitted_norm is not None:
            from .symmetry_quartets_screening import _StreamingScreenAudit, _compact_screening_overhead
            if mode != 'spatial':
                raise ValueError('inline screening requires the spatial streaming schedule')
            extra_bytes, extra_work = _compact_screening_overhead(len(schedule))
            budget.admit(required_bytes+extra_bytes, required_work+extra_work)
            # The attached certificate stays live even when a later build
            # explicitly requests the unmasked source.
            required_bytes += extra_bytes
            required_work += extra_work
            if bounds is None:
                screen_audit = _StreamingScreenAudit(source, len(schedule), maximum_omitted_norm)
            else:
                from .symmetry_quartets_bounds import _ProducerBoundAudit
                from .symmetry_quartets_hierarchy import MolecularHierarchicalBounds, _HierarchicalBoundAudit
                audit_type = _HierarchicalBoundAudit if type(bounds) is MolecularHierarchicalBounds else _ProducerBoundAudit
                screen_audit = audit_type(source, len(schedule), maximum_omitted_norm, bounds)
        if isinstance(source, _MantaSource):
            source = source.with_session()
        audit = sha256(b'vibeqc-streamed-quartet-audit-v1\0')
        if bounds is not None:
            audit.update(b'retained-orbits-with-producer-bounds-v1\0')
            audit.update(bounds.identity.encode())
        for label in (identity, mode, source_tolerance.hex()):
            encoded = label.encode()
            audit.update(len(encoded).to_bytes(8, 'big'))
            audit.update(encoded)
        audit.update(first.offsets.tobytes())
        audit.update(schedule.destinations.tobytes())
        audit.update(np.asarray((group.order, group.identity), np.int64).tobytes())
        for g in range(group.order):
            audit.update(np.asarray([group.product(g,h) for h in range(group.order)], np.int64).tobytes())
        for row in inverse_blocks:
            for matrix in row:
                audit.update(matrix.tobytes())
        counters, residual, stabilizer_residual, count, digest = [0]*5, 0., 0., 0, sha256()
        for orbit in schedule:
            orbit_norm = orbit_residual = 0.
            count += 1
            representative = orbit.representative
            if bounds is not None and screen_audit.try_omit(orbit, first.offsets):
                audit.update(b'bound-omitted\0')
                audit.update(np.asarray(representative, np.int64).tobytes())
                continue
            original = _read(source, representative, first.offsets, identity, counters)
            _digest_tile(digest, representative, original)
            for g, p in orbit.stabilizer:
                image, _ = _transform(original, tuple(inverse_blocks[g][s] for s in representative), p)
                stabilizer_residual = max(stabilizer_residual, _norm(image-original))
            previous_g, rotated = None, None
            for member in orbit.members:
                if member.operation != previous_g:
                    rotated, _ = _transform(original, tuple(inverse_blocks[member.operation][s] for s in representative), 0)
                    previous_g = member.operation
                predicted = rotated.transpose(_PERMUTATIONS[member.permutation])
                actual = original if member.shells == representative else _read(source, member.shells, first.offsets, identity, counters)
                difference = _norm(predicted-actual)
                residual = math.hypot(residual, difference)
                audit.update(np.asarray((*member.shells, member.operation, member.permutation), np.int64).tobytes())
                audit.update(actual.tobytes())
                if screen_audit is not None:
                    orbit_norm = math.hypot(orbit_norm, _norm(actual))
                    orbit_residual = math.hypot(orbit_residual, difference)
            if screen_audit is not None:
                screen_audit.observe(orbit, original, orbit_norm, orbit_residual, first.offsets)
        skipped = 0 if bounds is None else screen_audit.tiles
        if count != len(schedule) or counters[0]+skipped != ns**4:
            raise ValueError('streamed quartet census failed')
        if source.source_identity != identity:
            raise ValueError('quartet source identity changed during admission')
        if max(residual, stabilizer_residual) > source_tolerance or not math.isfinite(residual):
            raise ValueError('finite quartet source is not equivariant; use unreduced source')
        for key, value in dict(source=source, source_identity=identity, offsets=first.offsets,
                group=group, inverse_blocks=inverse_blocks, orbits=schedule,
                qualification_work=QuartetWork(*counters), source_residual_norm=residual,
                maximum_stabilizer_residual=stabilizer_residual, source_tolerance=source_tolerance,
                required_bytes=required_bytes, required_work=required_work,
                retained_array_bytes=first.offsets.nbytes+sum(m.nbytes for row in inverse_blocks for m in row),
                mode=mode, representative_digest=digest.hexdigest(),
                audit_identity=audit.hexdigest(), bounds=bounds,
                bound_queries=0 if bounds is None else screen_audit.queries,
                bound_lookup_probes=getattr(screen_audit, 'probes', 0),
                hierarchy_closure_refusals=getattr(screen_audit, 'closure_refusals', 0)).items():
            object.__setattr__(self, key, value)
        object.__setattr__(self, 'screening', None if screen_audit is None else screen_audit.seal(self))

    @property
    def production_reduction_authorized(self):
        return False

    @property
    def citation_numerics(self):
        return ('molecular_quartet_pair_stabilizers',)

    def jk_batch(self, densities, *, budget: Budget, executor='numpy', targets=None, screening=None):
        from .symmetry_quartets_batch import quartet_jk_batch
        return quartet_jk_batch(self, densities, budget=budget, executor=executor,
                                targets=targets, screening=screening)

    def jk(self, density, *, budget: Budget, executor='numpy'):
        if self.bounds is not None:
            raise ValueError('partially audited source requires its fixed mask; use jk_batch or the unreduced provider')
        if executor == 'native':
            from .symmetry_quartets_native import native_quartet_jk
            return native_quartet_jk(self, density, budget=budget)
        if executor != 'numpy':
            raise ValueError('unknown quartet executor')
        _array(density, 'complex128', 2)
        n = int(self.offsets[-1])
        if density.shape != (n, n):
            raise ValueError('density and quartet source dimensions disagree')
        budget.admit(self.required_bytes, self.required_work)
        if not np.isfinite(density).all():
            raise ValueError('nonfinite quartet density')
        density_norm = _norm(density)
        bound = self.source_residual_norm*density_norm
        if not math.isfinite(bound) or (bound == 0 and self.source_residual_norm > 0 and density_norm > 0):
            raise ValueError('nonrepresentable quartet contraction error bound')
        j, k = np.zeros((n, n), complex), np.zeros((n, n), complex)
        counters, terms, calls, transport, digest = [0]*5, 0, 0, 0, sha256()
        for orbit in self.orbits:
            original = _read(self.source, orbit.representative, self.offsets, self.source_identity, counters)
            _digest_tile(digest, orbit.representative, original)
            previous_g, rotated = None, None
            for member in orbit.members:
                if member.operation != previous_g:
                    rotated, work = _transform(original, tuple(self.inverse_blocks[member.operation][s] for s in orbit.representative), 0)
                    transport += work
                    previous_g = member.operation
                tile = rotated.transpose(_PERMUTATIONS[member.permutation])
                p, q, r, s = (slice(int(self.offsets[a]), int(self.offsets[a+1])) for a in member.shells)
                j[p, q] += np.einsum('pqrs,sr->pq', tile, density[s, r], optimize=False)
                k[p, s] += np.einsum('pqrs,qr->ps', tile, density[q, r], optimize=False)
                terms += 2*tile.size
                calls += 2
        if digest.hexdigest() != self.representative_digest:
            raise ValueError('quartet representative changed; requalify the finite source')
        if self.source.source_identity != self.source_identity or not np.isfinite(j).all() or not np.isfinite(k).all():
            raise ValueError('changed source or nonfinite quartet contraction')
        return QuartetJKResult(_freeze(j), _freeze(k), QuartetWork(*counters), terms, calls, transport, bound)
