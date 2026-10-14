"""Ordered fallback for exactly the qualified molecular P0 orbit mask.

The same immutable producer evaluates every retained ordered tile. No angular
transport, density projection, additional screening or source substitution is
performed. This is an execution alternative, not a new integral engine.
"""
from dataclasses import dataclass
from hashlib import sha256
import numpy as np

from . import _vibeqc_core as core
from .symmetry_shared import Budget, _array, _freeze
from .symmetry_quartets import QuartetJKBatchResult, QuartetWork, _norm, _read
from .symmetry_quartets_batch import _target_counts
from .symmetry_quartets_screening import QuartetScreening, _immutable_source, _screening_overhead
from .symmetry_quartets_streaming import StreamingShellQuartetPlan, _digest_tile


@dataclass(frozen=True)
class MolecularSourceQualification:
    """Evidence about an existing audit, never an unaudited producer seal.

    The exact native ReferenceJKBuilder constructor fixes real, molecular,
    full-Coulomb, derivative-zero, precision-zero support and normalization.
    Its immutable basis/source identity plus the numerical quartet audit
    qualify this mask. Neither a string nor this record can admit a provider.
    """
    producer_identity: str
    audit_identity: str
    source_identity: str
    policy_identity: str
    qualification_scope: str
    evidence: str = 'native-P0-constructor-and-numerical-audit'


def qualified_molecular_source(screening):
    if type(screening) is not QuartetScreening:
        raise TypeError('an immutable QuartetScreening certificate is required')
    plan = screening._plan
    _immutable_source(plan.source)
    screening._validate(plan)
    provider = plan.source._provider
    if (provider.plan.provider != 'libint'
            or provider.plan.n_basis != int(plan.offsets[-1])
            or provider.plan.n_shells != len(plan.offsets)-1):
        raise ValueError('producer and qualified molecular source disagree')
    return MolecularSourceQualification(screening.producer_identity,
        screening.audit_identity, screening.source_identity,
        screening.policy_identity, screening.receipt['qualification_scope'])


def ordered_quartet_requirements(screening, density_count, *, targets, budget):
    qualified_molecular_source(screening)
    plan = screening._plan
    nj, nk = _target_counts(density_count, targets)
    n, ns = int(plan.offsets[-1]), len(plan.offsets)-1
    maximum = max(m.shape[0] for m in plan.inverse_blocks[0])
    elements = sum(m.size for m in plan.inverse_blocks[0])
    native = core._plan_symmetry_quartet_accumulator(n, ns, 1, elements,
        maximum, budget.native(), density_count, nj, nk)
    sb, sw = _screening_overhead(plan, screening)
    size = plan.required_bytes + native.bytes + sb
    work = plan.required_work + native.work + sw
    budget.admit(size, work)
    return Budget(size, work), native, maximum


def ordered_quartet_jk(screening, densities, *, targets, budget):
    """Contract arbitrary complex densities against real ordered P0 tiles."""
    _array(densities, 'complex128', 3)
    count, n, last = densities.shape
    _, memory, maximum = ordered_quartet_requirements(screening, count,
        targets=targets, budget=budget)
    plan = screening._plan
    if (n, last) != (int(plan.offsets[-1]),)*2 or not np.isfinite(densities).all():
        raise ValueError('invalid ordered quartet density')
    j_channels = tuple(i for i, t in enumerate(targets) if 'j' in t)
    k_channels = tuple(i for i, t in enumerate(targets) if 'k' in t)
    bounds = np.array([screening.total_residual_norm*_norm(d) for d in densities])
    if (not np.isfinite(bounds).all() or any(b == 0. and
            screening.total_residual_norm > 0. and _norm(d) > 0.
            for b, d in zip(bounds, densities))):
        raise ValueError('nonrepresentable quartet contraction error bound')
    # Identity matrices, not the group's possibly non-orthogonal AO blocks.
    packed = np.concatenate([np.eye(int(b-a)).ravel()
        for a, b in zip(plan.offsets[:-1], plan.offsets[1:])])[None]
    flags = np.array([{'j': 1, 'k': 2, 'jk': 3}[t] for t in targets], np.uint8)
    accumulator = core._SymmetryQuartetAccumulator(plan.offsets, packed,
        densities, maximum, Budget(memory.bytes, memory.work).native(), flags)
    del packed, flags
    counters, digest = [0]*5, sha256()
    streaming = isinstance(plan, StreamingShellQuartetPlan)
    for index, orbit in enumerate(plan.orbits):
        if not screening.keep[index]:
            continue
        # Check representatives just as the reduced path does, using the
        # already-read tile. No extra audit reads or retained tile inventory.
        for member in orbit.members:
            tile = _read(plan.source, member.shells, plan.offsets,
                plan.source_identity, counters)
            if member.shells == orbit.representative:
                if streaming:
                    _digest_tile(digest, orbit.representative, tile)
                elif sha256(tile.tobytes()).hexdigest() != orbit.representative_digest:
                    raise ValueError('quartet representative changed; requalify the finite source')
            accumulator.accumulate(np.array(member.shells, np.int64), tile,
                np.array([(0, 0, *member.shells)], np.int64))
    if streaming and digest.hexdigest() != screening.retained_digest:
        raise ValueError('quartet representative changed; requalify the finite source')
    qualified_molecular_source(screening)
    j, k, counts = accumulator.finish()
    tiles = (len(plan.offsets)-1)**4-screening.omitted_ordered_tiles
    values = n**4-screening.omitted_integral_values
    channels = len(j_channels)+len(k_channels)
    if (counts['representatives'] != tiles or counters[0] != tiles
            or counts['scalar_products'] != channels*values
            or counts['block_contractions'] != channels*tiles):
        raise ValueError('ordered quartet execution census mismatch')
    if not np.isfinite(j).all() or not np.isfinite(k).all():
        raise ValueError('nonfinite ordered quartet contraction')
    return QuartetJKBatchResult(_freeze(j.reshape(len(j_channels), n, n)),
        _freeze(k.reshape(len(k_channels), n, n)), QuartetWork(*counters),
        counts['scalar_products'], counts['block_contractions'],
        counts['transport_scalar_products'], _freeze(bounds),
        screening.source_identity, 'ordered', j_channels, k_channels, screening)
