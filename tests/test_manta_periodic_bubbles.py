"""Typed periodic pair-image hierarchy, density bounds and weighted adjoint."""
import gc
from itertools import combinations_with_replacement, product
from math import comb, factorial

import numpy as np
import pytest
from vibeqc import manta
from tests.test_manta_periodic_pairs import (
    CELL, IMAGES, K, WEIGHTS, _basis, source, density, options, reference_density,
    coefficients, moment_quadrature, ordered_fourier_density,
)


def bubble_options(cap=2, branching=3, depth=32):
    o = manta.BubbleOptions(); o.leaf_capacity = cap; o.branching = branching; o.maximum_depth = depth
    return o


def translation(indices, delta):
    """Independent multinomial translation of raw Cartesian moments."""
    indices = np.asarray(indices); a, b = indices[:, None, :], indices[None, :, :]
    difference = a-b; valid = np.all(difference >= 0, axis=2)
    pascal = np.array([[comb(i, j) if j <= i else 0 for j in range(13)] for i in range(13)])
    result = np.prod(pascal[a, b]*np.asarray(delta)[None, None, :]**np.maximum(difference, 0), axis=2)
    return np.where(valid, result, 0.)


def node_reference(basis, s, tree, d):
    moments, norms = reference_density(basis, s, d)
    indices = s.moment_panel(0).indices; entries = s.entries; order = tree.pair_order
    result = np.zeros((len(tree.nodes), s.plan.moments))
    for i, node in enumerate(tree.nodes):
        for p in order[node.begin:node.end]:
            result[i] += translation(indices, np.array(entries[p].extent.center)-node.center)@moments[p]
    return result, norms


def radial_shift(values, shift, order):
    return np.array([sum(comb(k, j)*shift**(k-j)*values[j] for j in range(k+1)) for k in range(order+1)])


@pytest.mark.parametrize('pure', [False, True])
@pytest.mark.parametrize('order', [0, 4, 8])
@pytest.mark.parametrize('cutoff', [None, 0.])
def test_upward_moments_and_density_bounds_against_independent_polynomials(pure, order, cutoff):
    basis = _basis((0, 2), pure); s = source(basis, order, cutoff)
    tree = manta.PeriodicBubbleTree(s, bubble_options()); d = density(basis.nbasis)
    actual = tree.contract(d); expected, norms = node_reference(basis, s, tree, d)
    np.testing.assert_allclose(actual.moments, expected, rtol=8e-12, atol=2e-11)
    degree = np.sum(s.moment_panel(0).indices, axis=1)
    assert np.all(abs(actual.moments) <= actual.full_moment_bounds[:, degree]*(1+2e-13))
    for i, node in enumerate(tree.nodes):
        full = np.zeros(order+1); tail = full.copy(); norm = 0.
        for p in tree.pair_order[node.begin:node.end]:
            e = s.entries[p].extent; r = np.linalg.norm(np.array(e.center)-node.center)
            full += norms[p]*radial_shift(e.full_moment_bounds, r, order)
            tail += norms[p]*radial_shift(e.tail_moment_bounds, r, order)
            norm += norms[p]
        assert np.all(actual.full_moment_bounds[i] >= full*(1-3e-14))
        assert np.all(actual.tail_moment_bounds[i] >= tail*(1-3e-14))
        assert actual.coefficient_l1[i] >= norm*(1-3e-14)
    assert actual.translation_calls == s.receipt.pairs+len(tree.nodes)-1
    assert actual.translation_calls*2 <= tree.plan.maximum_translation_calls
    assert actual.pair_receipt.source_identity == s.receipt.source_identity == tree.pair_source_identity
    assert actual.source_identity == tree.source_identity
    assert actual.qualification == 'periodic-pair-image-hierarchy-and-adjoint'


def test_root_moments_direct_gauss_hermite_at_node_origin_and_order_twelve():
    basis = _basis((0, 1)); s = source(basis, 12); t = manta.PeriodicBubbleTree(s, bubble_options(cap=1))
    d = density(basis.nbasis); f = t.contract(d); root = t.nodes[0]; indices = s.moment_panel(0).indices
    direct = np.zeros(s.plan.moments)
    for e in s.entries:
        a, b = [basis.shells()[i] for i in e.shells]; b.origin = (np.array(b.origin)+e.translation).tolist()
        direct += coefficients(e, d).ravel()@moment_quadrature(a, b, root.center, indices)
    np.testing.assert_allclose(f.moments[0], direct, rtol=2e-11, atol=4e-10)


@pytest.mark.parametrize('order', [0, 4, 8, 12])
def test_weighted_reverse_pass_all_nodes_and_energy_derivative(order):
    basis = _basis((0, 1)); s = source(basis, order); tree = manta.PeriodicBubbleTree(s, bubble_options(cap=1))
    d = density(basis.nbasis); h = density(basis.nbasis, 83)
    f = tree.contract(d); rng = np.random.default_rng(34)
    v = np.ascontiguousarray(rng.normal(size=f.moments.shape)/np.maximum(1., abs(f.moments)))
    j = tree.adjoint(v)
    lhs = np.sum(f.moments*v); rhs = np.einsum('k,kij,kji->', WEIGHTS, d, j.values).real
    assert lhs == pytest.approx(rhs, rel=2e-11, abs=2e-10)
    np.testing.assert_allclose(j.values, j.values.conj().transpose(0, 2, 1), atol=4e-12, rtol=3e-12)
    eps = 2e-5
    fd = np.sum((tree.contract(d+eps*h).moments-tree.contract(d-eps*h).moments)*v)/(2*eps)
    expected = np.einsum('k,kij,kji->', WEIGHTS, h, j.values).real
    assert fd == pytest.approx(expected, rel=2e-8, abs=2e-8)
    # Each ancestor potential is translated into this leaf once, with W^T.
    local = np.zeros((s.receipt.pairs, s.plan.moments)); indices = s.moment_panel(0).indices
    for i, node in enumerate(tree.nodes):
        for p in tree.pair_order[node.begin:node.end]:
            local[p] += translation(indices, np.array(s.entries[p].extent.center)-node.center).T@v[i]
    np.testing.assert_allclose(j.values, s.adjoint(local).values, atol=2e-10, rtol=2e-11)
    assert j.source_identity == tree.source_identity and j.pair.source_identity == s.receipt.source_identity
    assert j.translation_calls == f.translation_calls


@pytest.mark.parametrize('cap,branching,depth,opening', [(1, 2, 32, .8), (2, 8, 32, .8), (3, 3, 1, 0.), (1, 3, 0, .5)])
def test_periodic_inventory_enclosure_and_disjoint_traversal(cap, branching, depth, opening):
    s = source(_basis((0, 1, 2)), options=options(0, radius=.02))
    t = manta.PeriodicBubbleTree(s, bubble_options(cap, branching, depth))
    assert sorted(t.pair_order) == list(range(s.receipt.pairs))
    for node in t.nodes:
        for p in t.pair_order[node.begin:node.end]:
            e = s.entries[p].extent
            assert np.linalg.norm(np.array(node.center)-e.center)+e.radius <= node.radius*(1+3e-14)
    for refine in [False, True]:
        visited = []; traversal = t.traversal(opening)
        while (block := traversal.peek()) is not None:
            if refine and block.far_candidate:
                traversal.refine(); continue
            a, b = t.nodes[block.target], t.nodes[block.source]
            left, right = t.pair_order[a.begin:a.end], t.pair_order[b.begin:b.end]
            pairs = combinations_with_replacement(left, 2) if block.target == block.source else product(left, right)
            pairs = [tuple(sorted(p)) for p in pairs]; assert len(pairs) == block.pair_interactions
            visited.extend(pairs); traversal.accept()
        assert sorted(visited) == list(combinations_with_replacement(range(s.receipt.pairs), 2))
        assert traversal.receipt.complete and not traversal.receipt.failed
        assert traversal.receipt.pair_interactions == len(visited)
    with pytest.raises(ValueError, match='canonical shell-pair inventory'): manta.BubbleTree(s.extents)


def test_always_near_ownership_reordering_and_immutable_exports():
    basis = _basis((0, 1)); opts = options(4, radius=.05); opts.extents.pointwise_tolerance = opts.extents.tail_mass_tolerance = 1e-50
    s = source(basis, options=opts); t = manta.PeriodicBubbleTree(s, bubble_options(cap=1))
    assert all(n.always_near for n in t.nodes) and all(not b.far_candidate for b in t.traversal(.9))
    same = manta.PeriodicBubbleTree(source(basis, options=opts, images=IMAGES[::-1].copy()), bubble_options(cap=1))
    assert t.source_identity == same.source_identity
    changed = manta.PeriodicBubbleTree(source(basis, order=4, momenta=K*1.1), bubble_options(cap=1))
    assert changed.source_identity != t.source_identity
    d = density(basis.nbasis); f = t.contract(d); before = f.moments.copy()
    copy = f.moments; copy[:] = 999.; copy = f.full_moment_bounds; copy[:] = 0
    np.testing.assert_array_equal(f.moments, before)
    with pytest.raises(TypeError): manta.PeriodicBubbleDensity()
    with pytest.raises(TypeError): manta.PeriodicBubbleTree(s.extents)
    del s, basis; gc.collect()
    np.testing.assert_array_equal(t.contract(d).moments, before)
    assert f.density_identity != t.contract(d*2).density_identity
    assert f.density_identity == t.contract(np.asfortranarray(d)).density_identity


@pytest.mark.parametrize('field,planned', [('shell_calls','shell_calls'), ('primitive_quartets','primitive_pairs'),
    ('scalar_terms','scalar_terms'), ('kernel_terms','kernel_terms'), ('memory_bytes','modeled_peak_bytes')])
def test_exact_joint_budget_and_traversal_limits(field, planned):
    s = source(); opts = bubble_options(cap=1); p = manta.plan_periodic_bubbles(s, opts)
    b = manta.ResourceBudget(); setattr(b, field, getattr(p, planned))
    tree = manta.PeriodicBubbleTree(s, opts, b); f = tree.contract(density(s.plan.n_basis))
    assert np.all(np.isfinite(tree.adjoint(f.moments).values))
    setattr(b, field, getattr(p, planned)-1)
    with pytest.raises(ValueError, match='budget exceeded'): manta.PeriodicBubbleTree(s, opts, b)
    if field == 'shell_calls':
        b = manta.ResourceBudget(); b.shell_calls = 0; tr = tree.traversal(budget=b)
        with pytest.raises(ValueError, match='budget'): tr.peek()
        with pytest.raises(RuntimeError, match='cannot resume'): tr.peek()


def test_input_shape_dtype_failures_zero_and_subnormal_density():
    s = source(_basis((0,)), order=0, images=IMAGES[:1]); t = manta.PeriodicBubbleTree(s)
    d = np.zeros((3, 1, 1), complex); f = t.contract(d)
    assert np.count_nonzero(f.moments) == 0 and np.count_nonzero(f.full_moment_bounds) == 0
    d[:] = np.nextafter(0., 1.); f = t.contract(d)
    assert f.coefficient_l1[0] > 0 and f.full_moment_bounds[0, 0] > 0
    for bad in [d.real.copy(), np.ones((3, 2, 2), complex), np.full_like(d, np.nan), np.full_like(d, 1j)]:
        with pytest.raises((ValueError, TypeError)): t.contract(bad)
    for bad in [np.zeros((1, 1), np.float32), np.ones((10, 1)), np.full((1, 1), np.inf)]:
        with pytest.raises(ValueError): t.adjoint(bad)
    huge = np.lib.stride_tricks.as_strided(np.zeros(1), shape=(10**14, 1), strides=(8, 8))
    with pytest.raises(ValueError, match='shape'): t.adjoint(huge)


def finite_bubble_calculation(order=8, *, pure=True, skew=True, cap=2, seed=17, perturbation=0.):
    """Whole-mesh Coulomb action in a complete declared finite Fourier domain."""
    basis = _basis((0, 1), pure); cell = CELL if skew else np.eye(3)*3.9
    s = source(basis, order, lattice=cell); t = manta.PeriodicBubbleTree(s, bubble_options(cap=cap))
    d = density(basis.nbasis, seed)+perturbation*density(basis.nbasis, 97)
    f = t.contract(d); indices = np.array(s.moment_panel(0).indices)
    selected = [i for i, node in enumerate(t.nodes) if not node.children]
    vectors = IMAGES[1:]@(2*np.pi*np.linalg.inv(cell)); factors = 4*np.pi/np.sum(vectors*vectors, axis=1)/abs(np.linalg.det(cell))
    panels = np.zeros((len(t.nodes), len(vectors), len(indices)), complex)
    for node in selected:
        panel = np.ones((len(vectors), len(indices)), complex)
        for axis in range(3):
            panel *= (-1j*vectors[:, axis, None])**indices[None, :, axis]
            panel /= np.array([factorial(int(i)) for i in indices[:, axis]])
        panels[node] = panel*np.exp(-1j*vectors@np.array(t.nodes[node].center))[:, None]
    rho = np.einsum('nm,ngm->g', f.moments, panels)
    energy = float(.5*np.dot(factors, abs(rho)**2))
    local = np.einsum('g,g,ngm->nm', factors, rho.conj(), panels).real
    j = t.adjoint(np.ascontiguousarray(local)).values
    trace = float(.5*np.einsum('k,kij,kji->', WEIGHTS, d, j).real)
    reference_rho = ordered_fourier_density(basis, d, vectors, cell=cell)
    reference_energy = float(.5*np.dot(factors, abs(reference_rho)**2))
    reference_moments, _ = node_reference(basis, s, t, d)
    same_order = np.einsum('nm,ngm->g', reference_moments, panels)
    degree_p = np.sum(indices, axis=1) == order
    last = np.einsum('nm,ngm->g', f.moments[:, degree_p], panels[:, :, degree_p])
    rho_bound = f.full_moment_bounds[selected, order].sum()*np.linalg.norm(vectors, axis=1)**order/factorial(order)+abs(last)
    energy_bound = float(np.dot(factors, abs(rho)*rho_bound+.5*rho_bound**2))
    return dict(energy=energy, trace_energy=trace, reference_energy=reference_energy,
                error=abs(energy-reference_energy), same_order_error=float(np.max(abs(rho-same_order))),
                rho=rho, reference_rho=reference_rho, rho_bound=rho_bound, energy_bound=energy_bound,
                j=j, density=d, direction=density(basis.nbasis,97), moments=f.moments, full_bounds=f.full_moment_bounds,
                source_identity=t.source_identity, density_identity=f.density_identity, pair_source_identity=t.pair_source_identity,
                pairs=s.receipt.pairs, nodes=len(t.nodes), selected_nodes=len(selected), plan=t.plan)


@pytest.mark.parametrize('pure,skew', [(True, True), (False, False)])
@pytest.mark.parametrize('cap', [1, 3])
def test_complete_finite_coulomb_energy_and_controlled_taylor_error(pure, skew, cap):
    rows = [finite_bubble_calculation(order, pure=pure, skew=skew, cap=cap) for order in (0, 6, 12)]
    for row in rows:
        assert row['energy'] == pytest.approx(row['trace_energy'], rel=3e-12, abs=3e-12)
        assert row['same_order_error'] < 2e-11
        assert np.all(abs(row['rho']-row['reference_rho']) <= row['rho_bound']+5e-12)
        assert row['error'] <= row['energy_bound']+5e-12
    assert rows[-1]['error'] < rows[0]['error']
