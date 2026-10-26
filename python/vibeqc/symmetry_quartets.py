"""Bounded, source-audited molecular shell-quartet execution.

Dupuis and King, Int. J. Quantum Chem. 11, 613 (1977),
doi:10.1002/qua.560110408: rotate shell distributions on a petite list.
This experimental consumer delegates integral evaluation to its provider.
It uses full AO blocks, not a non-Abelian reduced-product representation.
"""
from dataclasses import dataclass
from hashlib import sha256
from itertools import product
import math
from typing import Protocol

import numpy as np

from .symmetry_shared import (
    BlockSpaceAction, Budget, FiniteGroup, _array, _component_scale, _freeze,
    _integer, _scaled_frobenius, _text, _tolerance,
)


# Only real molecular AO ERIs admit these eight permutations without complex
# conjugation. A complex density does not change that integral convention.
_PERMUTATIONS = (
    (0, 1, 2, 3), (1, 0, 2, 3), (0, 1, 3, 2), (1, 0, 3, 2),
    (2, 3, 0, 1), (3, 2, 0, 1), (2, 3, 1, 0), (3, 2, 1, 0),
)


def _norm(array):
    scale = _component_scale(array)
    value = scale * _scaled_frobenius(array, scale)
    if not math.isfinite(value):
        raise ValueError('nonrepresentable quartet norm')
    return value


def _quartet_requirements(n, ns, ng, maximum_shell, provider_bytes, group_bytes, mode):
    pair_count = ns*(ns+1)//2
    maximum_orbits = ns**4 if mode == 'none' else pair_count*(pair_count+1)//2
    byte_count = (provider_bytes + group_bytes + 262144 + 1024*ns**4
                  + (192+2*ng)*maximum_orbits + 8192*ng*n + 512*n*n
                  + 96*maximum_shell**4)
    work = 100_000_000*ng + 128*ng*ns**4 + 512*n**4 + 1024*ng*(maximum_shell+1)*n**4
    return byte_count, work


@dataclass(frozen=True, slots=True)
class QuartetTile:
    """Owned or ephemeral ordered (p q | r s) values and actual work counts.

    The consumer copies values before requesting another tile. A cached
    dense reference reports zero shell/primitive calls, not invented savings.
    """
    values: np.ndarray
    source_identity: str
    shell_calls: int
    primitive_quartets_requested: int
    null_tiles: int
    primitive_precision: float


class QuartetSource(Protocol):
    """Stable finite source; custom callbacks are diagnostic declarations.

    Full numerical admission binds the supplied source at construction.
    Only an immutable producer can justify reusing it for later builds.
    """
    source_identity: str
    shell_offsets: np.ndarray
    modeled_peak_bytes: int

    def evaluate(self, ordered_quartet: tuple[int, int, int, int]) -> QuartetTile: ...


@dataclass(frozen=True, slots=True)
class QuartetImage:
    shells: tuple[int, int, int, int]
    operation: int
    permutation: int


@dataclass(frozen=True, slots=True)
class _QuartetStabilizer:
    # One byte per operation, one bit per ERI permutation. Keeping Python
    # (g,p) tuples here would cost O(8*|G|) objects per fixed quartet.
    masks: bytes

    def __len__(self):
        return sum(mask.bit_count() for mask in self.masks)

    def __iter__(self):
        for g, mask in enumerate(self.masks):
            for p in range(8):
                if mask & (1 << p):
                    yield g, p


@dataclass(frozen=True, slots=True)
class QuartetOrbit:
    representative: tuple[int, int, int, int]
    members: tuple[QuartetImage, ...]
    stabilizer: _QuartetStabilizer
    representative_digest: str
    source_norm: float
    residual_norm: float


@dataclass(frozen=True, slots=True)
class QuartetWork:
    tile_requests: int = 0
    shell_calls: int = 0
    primitive_quartets_requested: int = 0
    null_tiles: int = 0
    integral_values_received: int = 0


@dataclass(frozen=True, slots=True)
class QuartetJKResult:
    j: np.ndarray
    k: np.ndarray
    work: QuartetWork
    scalar_products: int
    block_contractions: int
    transport_scalar_products: int
    error_bound: float
    executor: str = 'numpy'


@dataclass(frozen=True, slots=True)
class QuartetJKBatchResult:
    """Independent density channels, with one aggregate source-work receipt."""
    j: np.ndarray
    k: np.ndarray
    work: QuartetWork
    scalar_products: int
    block_contractions: int
    transport_scalar_products: int
    error_bounds: np.ndarray
    source_identity: str
    executor: str
    j_channels: tuple[int, ...] = ()
    k_channels: tuple[int, ...] = ()
    screening: object = None


def _read(source, key, offsets, identity, counters):
    if source.source_identity != identity:
        raise ValueError('quartet source identity changed; requalify or use unreduced source')
    tile = source.evaluate(key)
    if not isinstance(tile, QuartetTile) or tile.source_identity != identity:
        raise ValueError('quartet tile source identity mismatch')
    _array(tile.values, 'float64', 4)
    if tile.values.shape != tuple(int(offsets[s+1]-offsets[s]) for s in key):
        raise ValueError('quartet tile shape mismatch')
    calls = _integer(tile.shell_calls, 'shell calls')
    primitives = _integer(tile.primitive_quartets_requested, 'primitive quartets')
    nulls = _integer(tile.null_tiles, 'null tiles')
    if nulls > calls or _tolerance(tile.primitive_precision, 'primitive precision') != 0:
        raise ValueError('quartet source requires zero primitive screening precision')
    if not np.isfinite(tile.values).all() or (nulls and np.any(tile.values)):
        raise ValueError('nonfinite or inconsistent null quartet tile')
    counters[0] += 1
    counters[1] += calls
    counters[2] += primitives
    counters[3] += nulls
    counters[4] += tile.values.size
    return tile.values.copy()


def _transform(values, matrices, permutation):
    """Covariant four-leg transport using inverse coefficient actions.

    E_target[pqrs] = W[ap]* W[bq] W[cr]* W[ds] E_source[abcd].
    W=U^-1. The admitted matrices are real; Euclidean unitarity is never
    assumed, including for normalized Cartesian d and higher shells.
    """
    transformed, work = values, 0
    for axis, matrix in enumerate(matrices):
        if np.array_equal(matrix, np.eye(matrix.shape[0])):
            continue
        work += transformed.size * matrix.shape[1]
        transformed = np.moveaxis(np.tensordot(transformed, matrix, axes=(axis, 0)), -1, axis)
    return transformed.transpose(_PERMUTATIONS[permutation]), work


def _real_shell_actions(group, actions):
    """Validate molecular block laws and return inverse coefficient actions."""
    first = actions[group.identity]
    ns, ng = len(first.destinations), group.order
    blocks = []
    for g, action in enumerate(actions):
        if group.antiunitary(g) or action.antiunitary:
            raise ValueError('antiunitary time reversal requires separate admission')
        if any(group.cocycle(g, h) != (0, 0, 0) for h in range(ng)):
            raise ValueError('periodic/projective quartet sources require a separate adapter')
        if (action.source != first.source or action.target != first.source
                or not np.array_equal(action.offsets, first.offsets)):
            raise ValueError('all actions must use the same complete named shell space')
        if np.any(action.matrices.imag):
            raise ValueError('eightfold quartet execution requires real AO actions')
        at, row = 0, []
        for s in range(ns):
            size = int(first.offsets[s+1]-first.offsets[s])
            row.append(_freeze(action.matrices[at:at+size*size].real.reshape(size, size)))
            at += size*size
        blocks.append(tuple(row))
    # The exact index law and numerical block law are separate. The latter
    # has a fixed representation tolerance, not the integral tolerance.
    for g, h, s in product(range(ng), range(ng), range(ns)):
        hs = int(actions[h].destinations[s])
        gh = group.product(g, h)
        if actions[g].destinations[hs] != actions[gh].destinations[s]:
            raise ValueError('shell permutations violate the group law')
        if _norm(blocks[g][hs] @ blocks[h][s] - blocks[gh][s]) > 1e-9:
            raise ValueError('shell matrices violate the group law')
    for s in range(ns):
        if first.destinations[s] != s or _norm(blocks[group.identity][s]-np.eye(blocks[group.identity][s].shape[0])) > 1e-9:
            raise ValueError('shell identity action failed')
    # The group's identity means no transport, including in the literal
    # unreduced fallback. Do not multiply by a numerically generated
    # near-identity Wigner matrix in that path.
    inverse_blocks = tuple(tuple(
        _freeze(np.eye(blocks[g][s].shape[0])) if g == group.identity
        else blocks[group.inverse(g)][int(actions[g].destinations[s])]
        for s in range(ns)) for g in range(ng))
    return inverse_blocks


@dataclass(frozen=True, slots=True, init=False, eq=False)
class ShellQuartetPlan:
    """Checked finite-source orbits with streamed, arbitrary-density J/K.

    Qualification evaluates every ordered shell quartet, including forbidden
    or screened-looking values, and checks representative stabilizers. Later
    builds evaluate only representatives and verify their exact snapshots.
    The source residual bounds the reconstructed tensor error, not floating
    point contraction error or an accumulated SCF/gradient error.
    """
    source: QuartetSource
    source_identity: str
    offsets: np.ndarray
    group: FiniteGroup
    inverse_blocks: tuple
    orbits: tuple[QuartetOrbit, ...]
    qualification_work: QuartetWork
    source_residual_norm: float
    maximum_stabilizer_residual: float
    source_tolerance: float
    required_bytes: int
    required_work: int
    retained_array_bytes: int
    mode: str
    audit_identity: str

    def __init__(self, group, actions, source, *, source_tolerance, budget: Budget, mode='spatial'):
        if not isinstance(group, FiniteGroup) or mode not in ('spatial', 'permutation', 'none'):
            raise ValueError('admitted molecular group and explicit quartet reduction mode required')
        if not isinstance(actions, tuple) or len(actions) != group.order or not all(isinstance(a, BlockSpaceAction) for a in actions):
            raise TypeError('one compact shell action per group operation is required')
        first = actions[group.identity]
        n, ns, ng = first.dimension, len(first.destinations), group.order
        provider_bytes = _integer(source.modeled_peak_bytes, 'provider bytes')
        # Reserve the complete bounded shell inventory, never an AO n^4
        # tensor. Python/allocator overhead is modeled conservatively, not RSS.
        required_bytes, required_work = _quartet_requirements(
            n, ns, ng, 0, provider_bytes, group._native.memory.bytes, mode)
        budget.admit(required_bytes, required_work)
        maximum_shell = int(np.max(np.diff(first.offsets)))
        required_bytes, required_work = _quartet_requirements(
            n, ns, ng, maximum_shell, provider_bytes, group._native.memory.bytes, mode)
        budget.admit(required_bytes, required_work)
        source_tolerance = _tolerance(source_tolerance, 'source tolerance')
        _array(source.shell_offsets, 'int64', 1)
        if not np.array_equal(source.shell_offsets, first.offsets):
            raise ValueError('source and action shell layouts disagree')
        identity = source.source_identity
        _text(identity, 'source identity')
        inverse_blocks = _real_shell_actions(group, actions)
        operations = (group.identity,) if mode != 'spatial' else (group.identity,) + tuple(g for g in range(ng) if g != group.identity)
        permutations = range(1 if mode == 'none' else 8)
        seen, schedules = set(), []
        for representative in product(range(ns), repeat=4):
            if representative in seen:
                continue
            members, stabilizer_masks = {}, bytearray(ng)
            for g in operations:
                mapped = tuple(int(actions[g].destinations[s]) for s in representative)
                for p in permutations:
                    target = tuple(mapped[i] for i in _PERMUTATIONS[p])
                    members.setdefault(target, QuartetImage(target, g, p))
                    if target == representative:
                        stabilizer_masks[g] |= 1 << p
            stabilizer = _QuartetStabilizer(bytes(stabilizer_masks))
            if len(members)*len(stabilizer) != len(operations)*len(permutations):
                raise ValueError('quartet orbit-stabilizer identity failed')
            if seen.intersection(members):
                raise ValueError('quartet orbits overlap')
            seen.update(members)
            schedules.append((representative, tuple(members.values()), stabilizer))
        del seen
        if isinstance(source, _MantaSource):
            # Engine construction follows complete shell-inventory admission.
            # One session is retained per plan; calls on a plan are serial.
            source = source.with_session()
        audit = sha256(b'vibeqc-finite-quartet-audit-v1\0')
        for label in (identity, mode, source_tolerance.hex()):
            encoded = label.encode()
            audit.update(len(encoded).to_bytes(8, 'big'))
            audit.update(encoded)
        audit.update(np.asarray((group.order, group.identity), np.int64).tobytes())
        for g in range(group.order):
            audit.update(np.asarray([group.product(g,h) for h in range(group.order)], np.int64).tobytes())
        audit.update(first.offsets.tobytes())
        for row in inverse_blocks:
            for matrix in row:
                audit.update(matrix.tobytes())
        counters, residual, stabilizer_residual, orbits = [0]*5, 0., 0., []
        for representative, members, stabilizer in schedules:
            orbit_norm = orbit_residual = 0.
            original = _read(source, representative, first.offsets, identity, counters)
            digest = sha256(original.tobytes()).hexdigest()
            for g, p in stabilizer:
                image, _ = _transform(original, tuple(inverse_blocks[g][s] for s in representative), p)
                stabilizer_residual = max(stabilizer_residual, _norm(image-original))
            # Unique members are grouped by their first transporter, so one
            # four-leg rotation serves several permutation views.
            previous_g, rotated = None, None
            for member in members:
                if member.operation != previous_g:
                    rotated, _ = _transform(original, tuple(inverse_blocks[member.operation][s] for s in representative), 0)
                    previous_g = member.operation
                predicted = rotated.transpose(_PERMUTATIONS[member.permutation])
                actual = original if member.shells == representative else _read(source, member.shells, first.offsets, identity, counters)
                difference = _norm(predicted-actual)
                residual = math.hypot(residual, difference)
                orbit_residual = math.hypot(orbit_residual, difference)
                orbit_norm = math.hypot(orbit_norm, _norm(actual))
                audit.update(np.asarray((*member.shells, member.operation, member.permutation), np.int64).tobytes())
                audit.update(actual.tobytes())
            if not math.isfinite(orbit_norm):
                raise ValueError('nonrepresentable finite-source orbit norm')
            orbits.append(QuartetOrbit(representative, members, stabilizer, digest,
                                      orbit_norm, orbit_residual))
        if source.source_identity != identity:
            raise ValueError('quartet source identity changed during admission')
        if max(residual, stabilizer_residual) > source_tolerance or not math.isfinite(residual):
            raise ValueError('finite quartet source is not equivariant; use unreduced source')
        for key, value in dict(source=source, source_identity=identity, offsets=first.offsets,
                group=group, inverse_blocks=inverse_blocks, orbits=tuple(orbits),
                qualification_work=QuartetWork(*counters), source_residual_norm=residual,
                maximum_stabilizer_residual=stabilizer_residual, source_tolerance=source_tolerance,
                required_bytes=required_bytes, required_work=required_work,
                retained_array_bytes=first.offsets.nbytes+sum(m.nbytes for row in inverse_blocks for m in row),
                mode=mode, audit_identity=audit.hexdigest()).items():
            object.__setattr__(self, key, value)

    @property
    def production_reduction_authorized(self):
        return False

    @property
    def citation_numerics(self):
        return ('molecular_quartet_symmetry',)

    def jk_batch(self, densities, *, budget: Budget, executor='numpy', targets=None, screening=None):
        """Share source/transport work across independent (batch,n,n) densities."""
        from .symmetry_quartets_batch import quartet_jk_batch
        return quartet_jk_batch(self, densities, budget=budget, executor=executor,
                                targets=targets, screening=screening)

    def jk(self, density, *, budget: Budget, executor='numpy'):
        """J[pq]=E[pqrs]D[sr], K[ps]=E[pqrs]D[qr], without state projection."""
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
        bound = self.source_residual_norm * density_norm
        if not math.isfinite(bound) or (bound == 0 and self.source_residual_norm > 0 and density_norm > 0):
            raise ValueError('nonrepresentable quartet contraction error bound')
        j, k = np.zeros((n, n), complex), np.zeros((n, n), complex)
        counters, terms, calls, transport = [0]*5, 0, 0, 0
        for orbit in self.orbits:
            original = _read(self.source, orbit.representative, self.offsets, self.source_identity, counters)
            if sha256(original.tobytes()).hexdigest() != orbit.representative_digest:
                raise ValueError('quartet representative changed; requalify the finite source')
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
        if self.source.source_identity != self.source_identity or not np.isfinite(j).all() or not np.isfinite(k).all():
            raise ValueError('changed source or nonfinite quartet contraction')
        return QuartetJKResult(_freeze(j), _freeze(k), QuartetWork(*counters), terms, calls, transport, bound)


def _shell_signature(shells):
    return tuple((s.atom_index, s.l, s.pure, tuple(s.origin),
                  tuple(s.exponents), tuple(s.coefficients)) for s in shells)


def _molecular_provider_kind(provider):
    """Exact immutable native types, never a caller's provider-name claim.

    MANTA's MD source has its own rounding and source identity. It enters
    through the same complete numerical audit, not the libint qualification.
    Range kernels, derivatives and periodic domains remain separate.
    """
    from . import _vibeqc_core as core
    kinds = {core._manta.ReferenceJKBuilder: 'libint', core._manta.DirectJKBuilder: 'manta'}
    kind = kinds.get(type(provider))
    if kind is None:
        raise TypeError('immutable native molecular quartet provider required; use unreduced builder')
    plan = provider.plan
    if (plan.provider != kind or getattr(plan, 'kernel', None) != 'coulomb'
            or getattr(plan, 'omega', None) != 0.):
        raise ValueError('unqualified molecular quartet provider or kernel')
    return kind


def manta_quartet_requirements(molecule, basis, provider, operation_count, *, budget: Budget, mode='spatial', schedule='stored', maximum_omitted_norm=None, bounds=None):
    """Complete count-only factory reservation, before shell/action snapshots.

    Returns a Budget whose exact limits admit construction. It includes
    provider inventory, all molecular actions and the bounded quartet plan.
    Native count-only helpers inspect extents, not radial/coordinate values.
    """
    from . import _vibeqc_core as core

    _molecular_provider_kind(provider)
    ng = _integer(operation_count, 'operation count', positive=True)
    if mode not in ('spatial', 'permutation', 'none') or schedule not in ('stored', 'streaming'):
        raise ValueError('invalid quartet reduction mode or schedule')
    if maximum_omitted_norm is not None:
        _tolerance(maximum_omitted_norm, 'maximum omitted norm')
        if schedule != 'streaming' or mode != 'spatial':
            raise ValueError('inline screening requires the spatial streaming schedule')
    p = provider.plan
    n, ns = int(basis.nbasis), int(basis.nshells)
    if bounds is not None and (maximum_omitted_norm is None or schedule != 'streaming' or mode != 'spatial'):
        raise ValueError('producer bounds require an explicit spatial streaming mask')
    from .symmetry_quartets_bounds import _bounds_overhead
    bound_bytes, bound_work = _bounds_overhead(bounds, provider, ns)
    if p.n_basis != n or p.n_shells != ns:
        raise ValueError('MANTA source and molecular basis counts disagree')
    metadata = core._symmetry_basis_snapshot_bytes(basis, budget.native())
    na = core._symmetry_molecule_count(molecule, budget.native())
    group = core._plan_symmetry_group(ng, False, budget.native())
    maximum_shell = math.isqrt(math.isqrt(int(p.tile_elements)))
    if maximum_shell**4 != p.tile_elements or maximum_shell < 1:
        raise ValueError('invalid native maximum quartet extent')
    requirements = _quartet_requirements
    if schedule == 'streaming':
        from .symmetry_quartets_streaming import _streaming_requirements
        requirements = _streaming_requirements
    byte_count, work = requirements(n, ns, ng, maximum_shell, p.modeled_peak_bytes, group.bytes, mode)
    byte_count += bound_bytes
    work += bound_work
    if maximum_omitted_norm is not None:
        from .symmetry_quartets_screening import _compact_screening_overhead
        # Count-only upper bound before actions/pair maps exist. The plan
        # later reserves the actual orbit count before allocating its mask.
        screen_bytes, screen_work = _compact_screening_overhead(ns**4)
        byte_count += screen_bytes
        work += screen_work
    # Include both borrowed and snapshotted native metadata, group-factory
    # workspace, and all action-factory inventories concurrently.
    group_bytes = 4096+128*ng*ng+256*ng+1024*na
    action_bytes = metadata+131072+4096*n+1024*na
    byte_count += 2*metadata + group_bytes + ng*action_bytes
    work += group.work + 1024*ng**3 + 256*ng*na**2
    cartesian_work = sum(core._plan_symmetry_cartesian_rotation(l, budget.native()).work
                         for l in range(p.maximum_l+1))
    work += ng*(256*na*na+512*n+7_000_000+cartesian_work)
    budget.admit(byte_count, work)
    return Budget(byte_count, work)


@dataclass(frozen=True, slots=True, init=False, eq=False)
class _MantaSource:
    """Thin adapter to an immutable native molecular provider, not an engine."""
    _provider: object
    _session: object
    _provider_kind: str
    shell_offsets: np.ndarray
    modeled_peak_bytes: int

    def __init__(self, provider, offsets, session=None):
        object.__setattr__(self, '_provider_kind', _molecular_provider_kind(provider))
        object.__setattr__(self, '_provider', provider)
        object.__setattr__(self, '_session', session)
        object.__setattr__(self, 'shell_offsets', _freeze(offsets))
        object.__setattr__(self, 'modeled_peak_bytes', int(provider.plan.modeled_peak_bytes))

    @property
    def source_identity(self):
        return self._provider.source_identity

    def with_session(self):
        return _MantaSource(self._provider, self.shell_offsets, self._provider.session())

    def evaluate(self, quartet):
        receipt = (self._session or self._provider).evaluate(quartet)
        if (receipt.get('provider') != self._provider_kind
                or receipt.get('kernel') != 'coulomb'
                or receipt.get('omega') != 0.):
            raise ValueError('unqualified quartet provider or kernel')
        required = ('values', 'source_identity', 'shell_calls',
                    'primitive_quartets_requested', 'null_tiles', 'primitive_precision')
        if any(field not in receipt for field in required):
            raise ValueError('quartet provider receipt lacks numerical source/work fields')
        return QuartetTile(*(receipt[field] for field in required))


def qualify_manta_quartets(molecule, basis, rotations, translations, *,
                          provider, source_tolerance, budget: Budget, mode='spatial', schedule='stored', maximum_omitted_norm=None, bounds=None, qualification='equivariance'):
    """Audit a libint or native MANTA MD full-Coulomb molecular source.

    No periodic images, complex integral kernels, density screening, RI/CD,
    BUPO or gradients are inferred from this gate. Native MD requires the
    full numerical audit; the libint producer bounds do not qualify it.
    Angular reconstruction requires the ordered-tile audit, reported as setup
    cost. Ordered-only qualification consumes native omission bounds and
    cannot authorize reconstruction. Provider/symmetry budgets stay explicit.
    """
    from .symmetry_ao import build_molecular_space_action, molecular_point_group
    from .symmetry_shared import SpaceIdentity

    if qualification not in ('equivariance', 'ordered-source'):
        raise ValueError('unknown quartet source qualification')
    if qualification == 'ordered-source' and (schedule != 'streaming' or bounds is None):
        raise ValueError('ordered-source qualification requires streamed producer bounds')
    _array(rotations, 'float64', 3)
    _array(translations, 'float64', 2)
    ng = len(rotations)
    if ng < 1 or rotations.shape != (ng, 3, 3) or translations.shape != (ng, 3):
        raise ValueError('molecular quartet operation shape mismatch')
    manta_quartet_requirements(molecule, basis, provider, ng, budget=budget, mode=mode,
        schedule=schedule, maximum_omitted_norm=maximum_omitted_norm, bounds=bounds)
    if _shell_signature(provider.shells) != _shell_signature(basis.shells()):
        raise ValueError('MANTA source and molecular basis snapshots disagree')
    group = molecular_point_group(molecule, rotations, translations,
        identity_label='molecular quartet source group', budget=budget)
    space = SpaceIdentity('finite molecular source', provider.source_identity,
                          'native AO shells', 'original AO metric')
    actions = tuple(build_molecular_space_action(molecule, basis,
        np.ascontiguousarray(r), np.ascontiguousarray(t), source=space, target=space,
        budget=budget) for r, t in zip(rotations, translations))
    plan_type = ShellQuartetPlan
    options = {}
    if schedule == 'streaming':
        from .symmetry_quartets_streaming import StreamingShellQuartetPlan
        plan_type = StreamingShellQuartetPlan
        options['maximum_omitted_norm'] = maximum_omitted_norm
        options['bounds'] = bounds
        options['qualification'] = qualification
    return plan_type(group, actions, _MantaSource(provider, actions[0].offsets),
        source_tolerance=source_tolerance, budget=budget, mode=mode, **options)
