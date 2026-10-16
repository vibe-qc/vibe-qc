"""Whole-orbit screening using MANTA's native bubble/error hierarchy.

Neese et al., doi:10.1021/acs.jpca.4c07415, Secs. 2.5-2.7: enclosing
extents and recursive near/far partitions. MANTA supplies the tree and its
Gaussian-tail ledger. This consumer adds a tensor-norm budget and orbit
closure; the geometric hierarchy is never an equivariance certificate.
"""
from dataclasses import dataclass, field
from hashlib import sha256
import math
from types import MappingProxyType

import numpy as np

from . import _vibeqc_core as core
from .symmetry_shared import Budget, _freeze, _integer, _tolerance
from .symmetry_quartets import _shell_signature
from .symmetry_quartets_bounds import _upper_product, _upper_hypot, _ProducerBoundAudit

_COVERAGE_CACHE_CAPACITY = 128
_COVERAGE_CACHE_BYTES = 4096 + 256*_COVERAGE_CACHE_CAPACITY


def _upper_sum(a, b):
    value = math.nextafter(a+b, math.inf) if a or b else 0.
    if not math.isfinite(value):
        raise ValueError('nonrepresentable hierarchical bound; use full audit')
    return value


def _weighted_bound(entry, entries):
    return _upper_product(entry, math.nextafter(math.sqrt(entries), math.inf))


def _pair(a, b):
    a, b = max(a, b), min(a, b)
    return a*(a+1)//2+b


def _native_budget(budget):
    native = core._manta.ResourceBudget()
    native.memory_bytes = budget.maximum_bytes
    native.shell_calls = native.primitive_quartets = budget.maximum_work
    native.scalar_terms = native.kernel_terms = budget.maximum_work
    return native


def _options(leaf_capacity, opening):
    leaf_capacity = _integer(leaf_capacity, 'bubble leaf capacity', positive=True)
    opening = _tolerance(opening, 'bubble opening')
    if opening >= 1.:
        raise ValueError('bubble opening must be below one')
    extents = core._manta.ExtentOptions()
    extents.moment_order = 0
    extents.retain_full_moments = True
    bubbles = core._manta.BubbleOptions()
    bubbles.leaf_capacity = leaf_capacity
    return extents, bubbles, opening


def molecular_hierarchical_bounds_requirements(basis, provider, *, budget: Budget,
        leaf_capacity=1, opening=.65):
    """Worst-case census before native envelopes, tree or Python snapshots."""
    if type(provider) is not core._manta.ReferenceJKBuilder:
        raise TypeError('hierarchical bounds require the immutable molecular P0 source')
    n, ns = int(basis.nbasis), int(basis.nshells)
    if (n, ns) != (provider.plan.n_basis, provider.plan.n_shells):
        raise ValueError('producer-bound basis dimensions disagree')
    extents, bubbles, _ = _options(leaf_capacity, opening)
    native = _native_budget(budget)
    plan = core._manta.plan_pair_extents(basis, extents, native)
    pairs = int(plan.shell_pairs)
    nodes, blocks = 2*pairs-1, pairs*(pairs+1)//2
    # A terminal partition can be quadratic in pair count. Reserve that
    # worst case before construction, including exported snapshots and the
    # sparse coverage map, even when actual accepted block storage is tiny.
    size = (plan.modeled_peak_bytes + provider.plan.modeled_peak_bytes
            + 4096 + 4096*nodes + 1024*blocks + 1024*pairs
            + 8*provider.plan.basis_payload_bytes + 32*(bubbles.maximum_depth+1)*pairs)
    work = (plan.scalar_terms + plan.kernel_terms + 131072*nodes*nodes
            + 4096*(bubbles.maximum_depth+1)*pairs*(1+pairs.bit_length()))
    budget.admit(size, work)
    return Budget(size, work)


@dataclass(frozen=True, slots=True, init=False, eq=False)
class MolecularHierarchicalBounds:
    """Immutable native hierarchy plus sparse bounded omission candidates.

    The stored blocks are candidates only. Symmetry admits their intersection
    with complete quartet orbits; an asymmetric tree cannot partially omit an
    orbit. Gaussian tails and node orientation/AO multiplicities are included.
    Arithmetic roundoff certification remains outside the analytic bound.
    """
    _provider: object = field(repr=False)
    _tree: object = field(repr=False)
    _model: object = field(repr=False)
    _coverage: object = field(repr=False)
    _ancestors: tuple = field(repr=False)
    _q: np.ndarray = field(repr=False)
    _v: np.ndarray = field(repr=False)
    shell_sizes: np.ndarray
    source_identity: str
    identity: str
    maximum_omitted_norm: float
    candidate_norm: float
    required_bytes: int
    required_work: int
    _receipt: object = field(repr=False)

    def __init__(self, basis, provider, *, maximum_omitted_norm, budget: Budget,
                 leaf_capacity=1, opening=.65):
        cap = _tolerance(maximum_omitted_norm, 'maximum omitted norm')
        reservation = molecular_hierarchical_bounds_requirements(
            basis, provider, budget=budget, leaf_capacity=leaf_capacity, opening=opening)
        options, bubble_options, opening = _options(leaf_capacity, opening)
        signature = _shell_signature(provider.shells)
        if signature != _shell_signature(basis.shells()):
            raise ValueError('producer-bound basis snapshot disagrees with source')
        native = _native_budget(budget)
        extents = core._manta.compute_pair_extents(basis, options, native)
        if signature != _shell_signature(basis.shells()):
            raise ValueError('producer-bound basis changed during construction')
        tree = core._manta.BubbleTree(extents, bubble_options, native)
        model = core._manta.BubbleErrorModel(tree, 0, native)
        nodes, order, pairs, envelopes = tree.nodes, tree.pair_order, extents.pairs, model.envelopes
        sizes = np.array([2*s[1]+1 if s[2] else (s[1]+1)*(s[1]+2)//2
                          for s in signature], np.int64)
        q, v = np.empty(len(pairs)), np.empty(len(pairs))
        weights = []
        for i, p in enumerate(pairs):
            a, b = map(int, p.shells)
            if not 0 <= b <= a < len(sizes) or _pair(a, b) != i:
                raise ValueError('unexpected native product-envelope ordering')
            q[i], v[i] = p.l1_norm_bound, p.potential_bound
            weights.append((1 if a == b else 2)*int(sizes[a])*int(sizes[b]))
        if not np.isfinite(q).all() or not np.isfinite(v).all() or np.any(q < 0.) or np.any(v < 0.):
            raise ValueError('invalid native product envelopes')
        # Parent weights sum all ordered AO pairs, not just shell-pair counts.
        # Python integers keep the multiplicity census exact.
        node_weights = [0]*len(nodes)
        ancestors = [None]*len(pairs)
        for k in reversed(range(len(nodes))):
            node = nodes[k]
            if node.children:
                node_weights[k] = sum(node_weights[c] for c in node.children)
            else:
                node_weights[k] = sum(weights[order[i]] for i in range(node.begin, node.end))
                chain, current = [], k
                while True:
                    chain.append(current)
                    if current == 0:
                        break
                    current = nodes[current].parent
                for i in range(node.begin, node.end):
                    ancestors[order[i]] = tuple(chain)
        if node_weights[0] != int(basis.nbasis)**2 or any(a is None for a in ancestors):
            raise ValueError('hierarchical ordered AO-pair census failed')
        cursor = tree.traversal(opening, native)
        coverage, total = {}, 0.
        queries = far_queries = tightened = refused_far = candidate_values = 0
        while (block := cursor.peek()) is not None:
            a, b = block.target, block.source
            ea, eb = envelopes[a], envelopes[b]
            entry = min(_upper_product(ea.full_moments[0], eb.potential),
                        _upper_product(eb.full_moments[0], ea.potential))
            queries += 1
            if block.far_candidate:
                try:
                    interaction = model.interaction(a, b)
                    # |M0_A T00 M0_B| + ALL native approximation-error terms
                    # bounds the full integral, not just its compact interior.
                    monopole = _upper_product(_upper_product(ea.full_moments[0],
                        abs(float(interaction.matrix.values[0, 0]))), eb.full_moments[0])
                    distance_bound = _upper_sum(monopole, interaction.error.total)
                    far_queries += 1
                    tightened += distance_bound < entry
                    entry = min(entry, distance_bound)
                except OverflowError:
                    # Native radial refusal leaves the independent full Q/V
                    # bound available; no zero/partial bound is substituted.
                    refused_far += 1
            values = node_weights[a]*node_weights[b]*(1 if a == b else 2)
            norm = _weighted_bound(entry, values)
            candidate = _upper_hypot(total, norm)
            if candidate <= cap:
                coverage[(min(a,b), max(a,b))] = entry
                total = candidate
                candidate_values += values
                cursor.accept()
            elif block.far_candidate:
                cursor.refine()
            else:
                cursor.accept()
        traversal = cursor.receipt
        if not traversal.complete or traversal.failed:
            raise ValueError('hierarchical screening traversal incomplete')
        digest = sha256(b'vibeqc-manta-P0-hierarchical-omission-v1\0')
        for label in (provider.source_identity, model.source_identity, traversal.partition_identity, cap.hex()):
            digest.update(label.encode()+b'\0')
        for (a,b), value in sorted(coverage.items()):
            digest.update(f'{a},{b}:{value.hex()};'.encode())
        receipt = dict(schema='vibeqc-manta-P0-hierarchical-omission-v1',
            source_identity=provider.source_identity, envelope_identity=extents.receipt.source_identity,
            tree_identity=tree.source_identity, model_identity=model.source_identity,
            partition_identity=traversal.partition_identity, kernel='full-coulomb',
            primitive_precision=0., finite_domain='all molecular quartets',
            bound_kind='node-QV-or-monopole-plus-Gaussian-tail', normalization='libint-shell-axial-v1',
            arithmetic_roundoff_included=False, operator_equivariance_certified=False,
            tree_equivariance_required=False, multipole_order=0, opening=opening,
            leaf_capacity=int(leaf_capacity), integral_shell_calls=0,
            shell_pairs=len(pairs), primitive_pairs=int(extents.receipt.primitive_pairs),
            envelope_evaluations=int(extents.receipt.envelope_evaluations), nodes=len(nodes),
            depth=tree.depth, node_bound_queries=queries, far_bound_queries=far_queries,
            tightened_node_bounds=int(tightened), radial_refusals=refused_far,
            visited_node_pairs=int(traversal.visited_node_pairs),
            accepted_candidate_blocks=len(coverage), candidate_integral_values=candidate_values,
            candidate_norm=total, maximum_omitted_norm=cap,
            coverage_payload_bytes=24*len(coverage),
            ancestor_index_bytes=8*sum(map(len, ancestors)),
            array_bytes=q.nbytes+v.nbytes+sizes.nbytes,
            required_bytes=reservation.maximum_bytes, required_work=reservation.maximum_work)
        identity = digest.hexdigest()
        receipt['identity'] = identity
        for key, value in dict(_provider=provider, _tree=tree, _model=model,
                _coverage=MappingProxyType(coverage), _ancestors=tuple(ancestors),
                _q=_freeze(q), _v=_freeze(v), shell_sizes=_freeze(sizes),
                source_identity=provider.source_identity, identity=identity,
                maximum_omitted_norm=cap, candidate_norm=total,
                required_bytes=reservation.maximum_bytes, required_work=reservation.maximum_work,
                _receipt=MappingProxyType(receipt)).items():
            object.__setattr__(self, key, value)

    def _validate(self, provider):
        if (type(provider) is not core._manta.ReferenceJKBuilder
                or provider.source_identity != self.source_identity
                or self._provider.source_identity != self.source_identity):
            raise ValueError('producer bounds belong to a different numerical source')

    def _indices(self, quartet):
        if not isinstance(quartet, tuple) or len(quartet) != 4:
            raise TypeError('four shell indices are required')
        indices = tuple(_integer(x, 'bound shell index') for x in quartet)
        if max(indices) >= len(self.shell_sizes):
            raise ValueError('bound shell index outside source')
        return indices

    def covered_bound(self, quartet):
        """Candidate tile bound or None, and actual ancestor-map probes."""
        a,b,c,d = self._indices(quartet)
        if not self._coverage:
            return None, 0
        probes = 0
        for na in self._ancestors[_pair(a,b)]:
            for nb in self._ancestors[_pair(c,d)]:
                probes += 1
                entry = self._coverage.get((min(na,nb), max(na,nb)))
                if entry is not None:
                    return _weighted_bound(entry, math.prod(int(self.shell_sizes[s])
                        for s in (a,b,c,d))), probes
        return None, probes

    def tile_frobenius_bound(self, quartet):
        """Full Q/V fallback or a tighter accepted hierarchical tile bound."""
        a,b,c,d = self._indices(quartet)
        ab,cd = _pair(a,b),_pair(c,d)
        entry = min(_upper_product(float(self._q[ab]),float(self._v[cd])),
                    _upper_product(float(self._q[cd]),float(self._v[ab])))
        norm = _weighted_bound(entry, math.prod(int(self.shell_sizes[s]) for s in (a,b,c,d)))
        covered, _ = self.covered_bound(quartet)
        return norm if covered is None else min(norm, covered)

    @property
    def receipt(self):
        return dict(self._receipt)


class _HierarchicalBoundAudit(_ProducerBoundAudit):
    policy = 'native-bubble-candidates-whole-orbit-intersection-v1'

    def __init__(self, source, count, cap, bounds):
        if float(cap) != bounds.maximum_omitted_norm:
            raise ValueError('hierarchical candidate budget changed; rebuild bounds')
        super().__init__(source, count, cap, bounds)
        self.probes = self.closure_refusals = 0
        self.cache_hits = self.cache_peak_entries = 0

    def try_omit(self, orbit, offsets):
        norm = 0.
        # These are identical real-ERI shell-pair labels, not spatially
        # equivalent bounds. A generic tree need not be group equivariant.
        # Keep only one bounded orbit-local cache; never retain all quartets.
        cache = {}
        for member in orbit.members:
            a,b,c,d = member.shells
            key = _pair(_pair(a,b), _pair(c,d))
            if key in cache:
                bound = cache[key]
                self.cache_hits += 1
            else:
                bound, probes = self.bounds.covered_bound(member.shells)
                self.probes += probes
                if len(cache) < _COVERAGE_CACHE_CAPACITY:
                    cache[key] = bound
                    self.cache_peak_entries = max(self.cache_peak_entries, len(cache))
            self.queries += 1
            if bound is None:
                self.closure_refusals += 1
                return False
            norm = _upper_hypot(norm, bound)
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
