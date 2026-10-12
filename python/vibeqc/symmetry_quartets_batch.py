"""Bounded independent-density J/K channels sharing admitted real ERI tiles.

Dupuis and King, doi:10.1002/qua.560110408, supply the shell transport
ordering. Batching reuses that transport by linearity; it neither selects
electronic-state irreps nor qualifies a new numerical integral source.
"""
from hashlib import sha256
import math

import numpy as np

from .symmetry_shared import Budget, _array, _freeze, _integer
from .symmetry_quartets import (
    QuartetJKBatchResult, QuartetWork, ShellQuartetPlan, _PERMUTATIONS,
    _norm, _read, _transform,
)
from .symmetry_quartets_streaming import StreamingShellQuartetPlan, _digest_tile


def _target_counts(count, targets):
    """Validate immutable per-input selectors without allocating an inventory."""
    if targets is None:
        return count, count
    if not isinstance(targets, tuple) or len(targets) != count:
        raise TypeError('targets must be a tuple with one J/K selector per density')
    nj = nk = 0
    for choice in targets:
        if not isinstance(choice, str) or choice not in ('j', 'k', 'jk'):
            raise ValueError('each density must request j, k or jk')
        nj += 'j' in choice
        nk += 'k' in choice
    return nj, nk


def quartet_batch_requirements(plan, density_count, *, budget: Budget, executor='numpy', targets=None, screening=None):
    """Count-only live-plan and execution reservation, before reading payload."""
    if not isinstance(plan, (ShellQuartetPlan, StreamingShellQuartetPlan)):
        raise TypeError('an admitted molecular quartet plan is required')
    count = _integer(density_count, 'density count', positive=True)
    nj, nk = _target_counts(count, targets)
    from .symmetry_quartets_screening import _screening_overhead
    screen_bytes, screen_work = _screening_overhead(plan, screening)
    if executor == 'native':
        from .symmetry_quartets_native import native_quartet_requirements
        native = native_quartet_requirements(plan, budget=budget, density_count=count, targets=targets)
        size, work = native.maximum_bytes+screen_bytes, native.maximum_work+screen_work
        budget.admit(size, work)
        return Budget(size, work)
    if executor != 'numpy':
        raise ValueError('unknown quartet executor')
    n = int(plan.offsets[-1])
    # Inputs + owned density snapshot + targets + immutable exports. The live
    # plan already reserves tile, schedule and transform workspace. NumPy's
    # einsum has one additional batch shell-matrix temporary, below this cap.
    size = plan.required_bytes + 65536 + 32*(count+nj+nk)*n*n + 128*count + screen_bytes
    # Keep the original all-target reservation; omit only modeled contraction
    # work for absent targets, not source audit or angular transport work.
    work = plan.required_work*(count+1) - 16*(2*count-nj-nk)*n**4 + screen_work
    budget.admit(size, work)
    return Budget(size, work)


def quartet_jk_batch(plan, densities, *, budget: Budget, executor='numpy', targets=None, screening=None):
    _array(densities, 'complex128', 3)
    count, n, last = densities.shape
    quartet_batch_requirements(plan, count, budget=budget, executor=executor,
                                targets=targets, screening=screening)
    if (n, last) != (int(plan.offsets[-1]),)*2:
        raise ValueError('density and quartet source dimensions disagree')
    if not np.isfinite(densities).all():
        raise ValueError('nonfinite quartet density')
    j_channels = tuple(i for i in range(count) if targets is None or 'j' in targets[i])
    k_channels = tuple(i for i in range(count) if targets is None or 'k' in targets[i])
    nj, nk = len(j_channels), len(k_channels)
    bounds = np.empty(count)
    for channel, density in enumerate(densities):
        norm = _norm(density)
        residual = plan.source_residual_norm if screening is None else screening.total_residual_norm
        bound = residual*norm
        if not math.isfinite(bound) or (bound == 0 and residual > 0 and norm > 0):
            raise ValueError('nonrepresentable quartet contraction error bound')
        bounds[channel] = bound
    native = executor == 'native'
    if native:
        from .symmetry_quartets_native import core, _requirements
        _, memory, maximum = _requirements(plan, budget, count, targets)
        packed = np.stack([np.concatenate([m.ravel() for m in row]) for row in plan.inverse_blocks])
        flags = None if targets is None else np.array([{'j':1, 'k':2, 'jk':3}[t] for t in targets], np.uint8)
        accumulator = core._SymmetryQuartetAccumulator(plan.offsets, packed, densities,
            maximum, Budget(memory.bytes, memory.work).native(), flags)
        del packed, flags
    else:
        # Source callbacks must not be able to change this build's densities.
        densities = _freeze(densities)
        j, k = np.zeros((nj,n,n), complex), np.zeros((nk,n,n), complex)
    counters, digest = [0]*5, sha256()
    terms = blocks = transport = representatives = 0
    streaming = isinstance(plan, StreamingShellQuartetPlan)
    for index, orbit in enumerate(plan.orbits):
        if screening is not None and not screening.keep[index]:
            continue
        original = _read(plan.source, orbit.representative, plan.offsets, plan.source_identity, counters)
        if streaming:
            _digest_tile(digest, orbit.representative, original)
        elif sha256(original.tobytes()).hexdigest() != orbit.representative_digest:
            raise ValueError('quartet representative changed; requalify the finite source')
        if native:
            members = np.array([(m.operation, m.permutation, *m.shells) for m in orbit.members], np.int64)
            accumulator.accumulate(np.array(orbit.representative, np.int64), original, members)
            continue
        representatives += 1
        previous_g = None
        for member in orbit.members:
            if member.operation != previous_g:
                rotated, work = _transform(original,
                    tuple(plan.inverse_blocks[member.operation][s] for s in orbit.representative), 0)
                transport += work
                previous_g = member.operation
            tile = rotated.transpose(_PERMUTATIONS[member.permutation])
            p, q, r, s = (slice(int(plan.offsets[a]), int(plan.offsets[a+1])) for a in member.shells)
            if nj == count:
                j[:, p, q] += np.einsum('pqrs,bsr->bpq', tile, densities[:, s, r], optimize=False)
            else:
                for output, channel in enumerate(j_channels):
                    j[output, p, q] += np.einsum('pqrs,sr->pq', tile, densities[channel, s, r], optimize=False)
            if nk == count:
                k[:, p, s] += np.einsum('pqrs,bqr->bps', tile, densities[:, q, r], optimize=False)
            else:
                for output, channel in enumerate(k_channels):
                    k[output, p, s] += np.einsum('pqrs,qr->ps', tile, densities[channel, q, r], optimize=False)
            terms += (nj+nk)*tile.size
            blocks += nj+nk
    expected_digest = plan.representative_digest if streaming else None
    if streaming and screening is not None:
        expected_digest = screening.retained_digest
    if streaming and digest.hexdigest() != expected_digest:
        raise ValueError('quartet representative changed; requalify the finite source')
    if plan.source.source_identity != plan.source_identity:
        raise ValueError('quartet source identity changed during build')
    if native:
        j, k, counts = accumulator.finish()
        j, k = j.reshape(nj, n, n), k.reshape(nk, n, n)
        representatives, terms, blocks, transport = (counts[key] for key in
            ('representatives', 'scalar_products', 'block_contractions', 'transport_scalar_products'))
    expected_reps, expected_values = len(plan.orbits), n**4
    expected_tiles = (len(plan.offsets)-1)**4
    if screening is not None:
        expected_reps -= screening.omitted_representatives
        expected_values -= screening.omitted_integral_values
        expected_tiles -= screening.omitted_ordered_tiles
    if (representatives != expected_reps or terms != (nj+nk)*expected_values
            or blocks != (nj+nk)*expected_tiles):
        raise ValueError('quartet execution census mismatch')
    if not np.isfinite(j).all() or not np.isfinite(k).all():
        raise ValueError('nonfinite quartet contraction')
    return QuartetJKBatchResult(_freeze(j), _freeze(k), QuartetWork(*counters),
        terms, blocks, transport, _freeze(bounds),
        plan.source_identity if screening is None else screening.source_identity,
        executor, j_channels, k_channels, screening)
