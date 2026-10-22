"""Gaussian BUPO approximation error ledger, independent exact references."""
import itertools
from math import erf, pi, sqrt

import numpy as np
import pytest

import vibeqc as vq
from vibeqc import manta
from vibeqc import _vibeqc_core as core


def _fixture(pure=True, order=12, retain=True, tiny=False, far=35.):
    positions = [[0., .1, -.2], [2., -.2, .4], [far, 1., -.7], [far+2., -.5, .6]]
    mol = vq.Molecule([vq.Atom(2, p) for p in positions], 0, 1)
    shells = []
    for i, p in enumerate(positions):
        s = core.ShellInfo(); s.atom_index = i; s.l = i % 3; s.pure = pure; s.origin = p
        s.exponents = [.6, 1.4]; s.coefficients = [.8, -.25]; shells.append(s)
    basis = vq.BasisSet(mol, shells, "manta-bupo-bound-fixture", True)
    ext = manta.ExtentOptions(); ext.moment_order = order; ext.retain_full_moments = retain
    if tiny:
        ext.minimum_radius = ext.maximum_radius = .3
        ext.pointwise_tolerance = ext.tail_mass_tolerance = 1e6
    options = manta.BubbleOptions(); options.leaf_capacity = 1
    tree = manta.BubbleTree(manta.compute_pair_extents(basis, ext), options)
    return basis, tree


def _pair_ids(tree, node):
    n = tree.nodes[node]
    return tree.pair_order[n.begin:n.end]


@pytest.mark.parametrize("pure", [False, True])
@pytest.mark.parametrize("order", [0, 2, 4, 6, 9, 12])
def test_all_geometric_far_blocks_bound_independent_libint_integrals(pure, order):
    basis, tree = _fixture(pure, order)
    model = manta.BubbleErrorModel(tree, order)
    reference = manta.ReferenceJKBuilder(basis).session()
    blocks = [b for b in tree.traversal(.65) if b.far_candidate]
    assert blocks
    for block in blocks:
        interaction = model.interaction(block.target, block.source)
        bound = interaction.error
        assert bound.total >= bound.compact_remainder+bound.tail_penetration+bound.full_moment_tail
        assert bound.compact_remainder > 0 and bound.tail_penetration > 0 and bound.full_moment_tail > 0
        for ai, bi in itertools.product(_pair_ids(tree, block.target), _pair_ids(tree, block.source)):
            a, b = tree.pairs[ai], tree.pairs[bi]
            ma = manta.compute_pair_moments(basis, a.shells, tree.nodes[block.target].center, order)
            mb = manta.compute_pair_moments(basis, b.shells, tree.nodes[block.source].center, order)
            approximate = ma.values@interaction.matrix.values@mb.values.T
            exact = reference.evaluate([*a.shells, *b.shells])["values"].reshape(approximate.shape)
            fp_allowance = 2e-11*max(1., np.max(abs(exact)), np.max(abs(approximate)))
            assert np.max(abs(approximate-exact)) <= bound.total+fp_allowance
        reverse = model.interaction(block.source, block.target)
        assert reverse.error.total == pytest.approx(bound.total, rel=2e-12, abs=0.)
        np.testing.assert_allclose(reverse.matrix.values, interaction.matrix.values.T, atol=1e-15, rtol=3e-13)
        assert reverse.error.source_identity != bound.source_identity


def test_every_node_envelope_covers_translated_descendant_moments():
    basis, tree = _fixture(order=6)
    model = manta.BubbleErrorModel(tree, 6)
    envelopes = model.envelopes
    for node_id, node in enumerate(tree.nodes):
        for pair_id in _pair_ids(tree, node_id):
            pair = tree.pairs[pair_id]
            panel = manta.compute_pair_moments(basis, pair.shells, node.center, 6)
            for q, index in enumerate(panel.indices):
                assert np.max(abs(panel.values[:, q])) <= envelopes[node_id].full_moments[sum(index)]*(1+1e-12)
            assert envelopes[node_id].potential >= pair.potential_bound
            assert envelopes[node_id].tail_moments[0] >= pair.tail_mass_bound
    copied = envelopes[0].full_moments; copied.clear()
    assert len(model.envelopes[0].full_moments) == 7


def test_compact_order_ladder_and_explicit_full_gaussian_tail_term():
    basis, tree = _fixture(tiny=True)
    cursor = tree.traversal(.7)
    candidates = [b for b in cursor if b.far_candidate]
    # With theta below .4, raising the order from zero to twelve reduces the
    # geometric remainder by theta**12, below the stated four-decade target.
    selected = max((b for b in candidates if b.separation_ratio < .4), key=lambda b:b.separation_ratio)
    bounds = [manta.BubbleErrorModel(tree, order).bound(selected.target, selected.source) for order in [0, 2, 6, 12]]
    assert bounds[-1].compact_remainder < 1e-4*bounds[0].compact_remainder
    assert all(b.full_moment_tail > 0 and b.tail_penetration > 0 for b in bounds)
    assert bounds[-1].full_moment_tail > 100*bounds[-1].compact_remainder
    assert bounds[-1].total >= bounds[-1].full_moment_tail


@pytest.mark.parametrize("distance,exponent", [(2., .03), (8., .2), (30., .7)])
def test_s_gaussian_penetration_against_analytic_coulomb(distance, exponent):
    mol = vq.Molecule([vq.Atom(2, [0., 0., 0.]), vq.Atom(2, [distance, 0., 0.])], 0, 1)
    shells = []
    for i in range(2):
        s = core.ShellInfo(); s.atom_index = i; s.l = 0; s.pure = False
        s.origin = [i*distance, 0., 0.]; s.exponents = [exponent]; s.coefficients = [1.]; shells.append(s)
    basis = vq.BasisSet(mol, shells, "manta-gaussian-error", True)
    ext = manta.ExtentOptions(); ext.moment_order = 6; ext.retain_full_moments = True
    ext.minimum_radius = ext.maximum_radius = .1; ext.pointwise_tolerance = ext.tail_mass_tolerance = 1e10
    opt = manta.BubbleOptions(); opt.leaf_capacity = 1
    tree = manta.BubbleTree(manta.compute_pair_extents(basis, ext), opt)
    leaves = {tuple(tree.pairs[tree.pair_order[n.begin]].shells): i for i, n in enumerate(tree.nodes) if not n.children}
    model = manta.BubbleErrorModel(tree, 6); interaction = model.interaction(leaves[(0, 0)], leaves[(1, 1)])
    a = manta.compute_pair_moments(basis, [0, 0], [0., 0., 0.], 6).values[0]
    b = manta.compute_pair_moments(basis, [1, 1], [distance, 0., 0.], 6).values[0]
    approximate = a@interaction.matrix.values@b
    mass = (pi/(2*exponent))**1.5
    exact = mass*mass*erf(sqrt(exponent)*distance)/distance
    assert approximate == pytest.approx(mass*mass/distance, rel=3e-12)
    assert abs(approximate-exact) <= interaction.error.total


@pytest.mark.parametrize("field,plan_field", [("memory_bytes", "modeled_peak_bytes"), ("scalar_terms", "scalar_terms"), ("kernel_terms", "kernel_terms")])
def test_error_model_resource_boundaries(field, plan_field):
    _, tree = _fixture(order=6)
    plan = manta.plan_bubble_errors(tree, 6)
    budget = manta.ResourceBudget(); setattr(budget, field, getattr(plan, plan_field))
    model = manta.BubbleErrorModel(tree, 6, budget)
    block = next(b for b in tree.traversal(.7) if b.far_candidate)
    model.bound(block.target, block.source)
    model.auxiliary_interaction(0, _auxiliary_extent(_auxiliary(center=(100., 0., 0.)), 6))
    assert model.plan.modeled_peak_bytes == plan.modeled_peak_bytes
    setattr(budget, field, getattr(plan, plan_field)-1)
    with pytest.raises(ValueError, match="budget"):
        manta.BubbleErrorModel(tree, 6, budget)


def test_invalid_missing_moments_unsupported_and_overlapping_bubbles():
    _, tree = _fixture(order=4, retain=False)
    with pytest.raises(ValueError, match="retained"):
        manta.BubbleErrorModel(tree, 4)
    _, tree = _fixture(order=4)
    with pytest.raises(ValueError, match="retained"):
        manta.BubbleErrorModel(tree, 5)
    for order in [-1, 13]:
        with pytest.raises(ValueError, match="order"):
            manta.BubbleErrorModel(tree, order)
    model = manta.BubbleErrorModel(tree, 4)
    with pytest.raises(IndexError, match="index"):
        model.bound(len(tree.nodes), 0)
    with pytest.raises(ValueError, match="inventories"):
        model.bound(0, 1)
    with pytest.raises(ValueError, match="inventories"):
        model.bound(1, 1)
    pair = next((i, j) for i, a in enumerate(tree.nodes) for j, b in enumerate(tree.nodes)
                if i < j and not a.children and not b.children and np.linalg.norm(np.array(a.center)-b.center) <= a.radius+b.radius)
    with pytest.raises(ValueError, match="support"):
        model.bound(*pair)


def test_always_near_requires_exact_evaluation():
    basis, _ = _fixture(order=2)
    ext = manta.ExtentOptions(); ext.moment_order = 2; ext.retain_full_moments = True
    ext.minimum_radius = ext.maximum_radius = .001
    options = manta.BubbleOptions(); options.leaf_capacity = 1
    tree = manta.BubbleTree(manta.compute_pair_extents(basis, ext), options)
    assert tree.nodes[0].always_near
    model = manta.BubbleErrorModel(tree, 2)
    with pytest.raises(ValueError, match="always-near"):
        model.bound(0, 1)
    with pytest.raises(ValueError, match="always-near"):
        model.auxiliary_interaction(0, _auxiliary_extent(_auxiliary(), 2))


def test_extreme_radial_scale_refuses_instead_of_using_underflowed_high_order_kernel():
    _, tree = _fixture(order=3, far=1e100)
    model = manta.BubbleErrorModel(tree, 3)
    block = next(b for b in tree.traversal(.7) if b.far_candidate)
    with pytest.raises(OverflowError, match="radial scale"):
        model.bound(block.target, block.source)


def test_error_admission_refines_geometric_candidates_and_hashes_actual_partition():
    _, tree = _fixture(order=6)
    model = manta.BubbleErrorModel(tree, 6)
    receipts = []
    # Distant cross-center products can have nonzero envelopes far below
    # 1e-30. Zero forces every positive approximation bound to be refined.
    for tolerance in [1e3, 1e-6, 0.]:
        cursor = tree.traversal(.8)
        while (block := cursor.peek()) is not None:
            if block.far_candidate and model.bound(block.target, block.source).total > tolerance:
                cursor.refine()
            else:
                cursor.accept()
        receipts.append(cursor.receipt)
    assert len({r.source_identity for r in receipts}) == 1
    assert len({r.partition_identity for r in receipts}) >= 2
    assert all(r.complete and not r.failed for r in receipts)
    assert receipts[0].far_blocks > 0 and receipts[-1].far_blocks == 0
    assert receipts[-1].refined_candidates > receipts[0].refined_candidates
    assert len({r.pair_interactions for r in receipts}) == 1


def _auxiliary(l=2, pure=True, center=(70., -.3, .6), coefficient=.8, exponent=.6):
    mol = vq.Molecule([vq.Atom(2, list(center))], 0, 1)
    s = core.ShellInfo(); s.atom_index = 0; s.l = l; s.pure = pure; s.origin = list(center)
    s.exponents = [exponent, 1.4]; s.coefficients = [coefficient, -.25]
    return vq.BasisSet(mol, [s], "manta-auxiliary-error-fixture", True)


def _auxiliary_extent(basis, order, tiny=False):
    options = manta.ExtentOptions(); options.moment_order = order; options.retain_full_moments = True
    if tiny:
        options.minimum_radius = options.maximum_radius = .1
        options.pointwise_tolerance = options.tail_mass_tolerance = 1e10
    return manta.compute_auxiliary_extents(basis, options).shells[0]


@pytest.mark.parametrize("pure", [False, True])
@pytest.mark.parametrize("aux_l", [0, 2])
@pytest.mark.parametrize("order", [0, 6, 12])
def test_auxiliary_pair_bubble_error_bounds_independent_three_center_integrals(pure, aux_l, order):
    basis, tree = _fixture(pure, order, far=5.)
    auxiliary = _auxiliary(aux_l, pure); extent = _auxiliary_extent(auxiliary, order)
    model = manta.BubbleErrorModel(tree, order)
    reference = np.asarray(vq.compute_3c_eri(basis, auxiliary))
    counts = [2*s.l+1 if s.pure else (s.l+1)*(s.l+2)//2 for s in basis.shells()]
    offsets = np.cumsum([0]+counts)
    auxiliary_moments = manta.compute_auxiliary_moments(auxiliary, 0, extent.center, order).values
    largest_error = 0.
    for target, node in enumerate(tree.nodes):
        interaction = model.auxiliary_interaction(target, extent)
        bound = interaction.error
        assert bound.total >= bound.compact_remainder+bound.tail_penetration+bound.full_moment_tail
        assert min(bound.compact_remainder, bound.tail_penetration, bound.full_moment_tail) > 0.
        assert len(bound.source_identity) == 64
        for pair_id in _pair_ids(tree, target):
            a, b = tree.pairs[pair_id].shells
            panel = manta.compute_pair_moments(basis, [a, b], node.center, order)
            approximate = panel.values@interaction.matrix.values@auxiliary_moments.T
            exact = reference[:, offsets[a]:offsets[a+1], offsets[b]:offsets[b+1]].reshape(auxiliary.nbasis, -1).T
            error = np.max(abs(approximate-exact)); largest_error = max(largest_error, error)
            assert error <= bound.total+2e-11*max(1., np.max(abs(exact)), np.max(abs(approximate)))
    if order == 0:
        assert largest_error > 1e-7


@pytest.mark.parametrize("distance,orbital_exponent,auxiliary_exponent", [(2., .03, .045), (8., .2, .3), (30., .7, .4)])
def test_auxiliary_penetration_bound_against_two_unequal_analytic_gaussians(distance, orbital_exponent, auxiliary_exponent):
    def single(center, exponent):
        mol = vq.Molecule([vq.Atom(2, center)], 0, 1)
        shell = core.ShellInfo(); shell.atom_index = 0; shell.l = 0; shell.pure = False; shell.origin = center
        shell.exponents = [exponent]; shell.coefficients = [1.]
        return vq.BasisSet(mol, [shell], "manta-single-gaussian-penetration", True)
    orbital = single([0., 0., 0.], orbital_exponent)
    auxiliary = single([distance, 0., 0.], auxiliary_exponent)
    options = manta.ExtentOptions(); options.moment_order = 6; options.retain_full_moments = True
    options.minimum_radius = options.maximum_radius = .1
    options.pointwise_tolerance = options.tail_mass_tolerance = 1e10
    tree = manta.BubbleTree(manta.compute_pair_extents(orbital, options))
    extent = manta.compute_auxiliary_extents(auxiliary, options).shells[0]
    interaction = manta.BubbleErrorModel(tree, 6).auxiliary_interaction(0, extent)
    a = manta.compute_pair_moments(orbital, [0, 0], [0., 0., 0.], 6).values[0]
    b = manta.compute_auxiliary_moments(auxiliary, 0, extent.center, 6).values[0]
    p, q = 2*orbital_exponent, auxiliary_exponent
    masses = (pi/p)**1.5*(pi/q)**1.5
    approximate = a@interaction.matrix.values@b
    exact = masses*erf(sqrt(p*q/(p+q))*distance)/distance
    assert approximate == pytest.approx(masses/distance, rel=3e-12)
    assert abs(approximate-exact) <= interaction.error.total


def test_auxiliary_error_source_identity_support_moments_and_always_near_admission():
    _, tree = _fixture(order=6, far=5.)
    model = manta.BubbleErrorModel(tree, 6)
    first = _auxiliary_extent(_auxiliary(), 6)
    second = _auxiliary_extent(_auxiliary(coefficient=.9), 6)
    a = model.auxiliary_interaction(0, first)
    b = model.auxiliary_interaction(0, second)
    assert a.error.source_identity != b.error.source_identity
    np.testing.assert_array_equal(a.matrix.values, b.matrix.values)
    assert a.error.total != b.error.total
    with pytest.raises(IndexError, match="index"):
        model.auxiliary_interaction(len(tree.nodes), first)
    missing = manta.compute_auxiliary_extents(_auxiliary()).shells[0]
    with pytest.raises(ValueError, match="retained"):
        model.auxiliary_interaction(0, missing)
    with pytest.raises(ValueError, match="retained"):
        model.auxiliary_interaction(0, _auxiliary_extent(_auxiliary(), 4))
    overlapping = _auxiliary_extent(_auxiliary(center=tree.nodes[0].center), 6)
    with pytest.raises(ValueError, match="support"):
        model.auxiliary_interaction(0, overlapping)
    options = manta.ExtentOptions(); options.moment_order = 6; options.retain_full_moments = True
    options.minimum_radius = options.maximum_radius = .001
    always_near = manta.compute_auxiliary_extents(_auxiliary(), options).shells[0]
    assert always_near.always_near
    with pytest.raises(ValueError, match="always-near"):
        model.auxiliary_interaction(0, always_near)
    extreme = _auxiliary_extent(_auxiliary(center=[1e100, 0., 0.]), 6)
    with pytest.raises(OverflowError, match="radial scale"):
        model.auxiliary_interaction(0, extreme)
