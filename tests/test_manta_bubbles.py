"""Deterministic bounded bubble geometry and exhaustive streamed partitions."""
import gc
import itertools

import numpy as np
import pytest

import vibeqc as vq
from vibeqc import manta
from vibeqc import _vibeqc_core as core


def _extents(n=7, *, coincident=False, mixed=False, shift=(0., 0., 0.)):
    positions = np.array([[12.*i, 2.*(i % 3), -.7*(i % 2)] for i in range(n)])
    if coincident:
        positions[:] = 0.
    positions += np.array(shift)
    molecule = vq.Molecule([vq.Atom(2, p.tolist()) for p in positions], 0, 1)
    shells = []
    for i, p in enumerate(positions):
        s = core.ShellInfo(); s.atom_index = i; s.l = 0; s.pure = True
        s.origin = p.tolist(); s.exponents = [.00001 if mixed and i == 0 else .7]; s.coefficients = [1.]
        shells.append(s)
    basis = vq.BasisSet(molecule, shells, "manta-bubble-fixture", True)
    options = manta.ExtentOptions(); options.moment_order = 2
    if mixed:
        options.maximum_radius = 10.
    return manta.compute_pair_extents(basis, options)


def _tree(extents, leaf=1, branching=3, depth=32):
    options = manta.BubbleOptions(); options.leaf_capacity = leaf
    options.branching = branching; options.maximum_depth = depth
    return manta.BubbleTree(extents, options)


def _covered(tree, blocks):
    order = tree.pair_order; nodes = tree.nodes
    interactions = []
    for block in blocks:
        a, b = nodes[block.target], nodes[block.source]
        left, right = order[a.begin:a.end], order[b.begin:b.end]
        if block.target == block.source:
            assert not block.far_candidate
            pairs = list(itertools.combinations_with_replacement(left, 2))
        else:
            assert set(left).isdisjoint(right)
            pairs = list(itertools.product(left, right))
        assert block.pair_interactions == len(pairs)
        if block.far_candidate:
            assert not a.always_near and not b.always_near
            assert 0 < block.separation_ratio < 1.
            assert np.linalg.norm(np.array(a.center)-b.center) > a.radius+b.radius
        else:
            assert not a.children and not b.children
        interactions.extend(tuple(sorted(pair)) for pair in pairs)
    expected = set(itertools.combinations_with_replacement(range(len(order)), 2))
    assert len(interactions) == len(expected)
    assert set(interactions) == expected


@pytest.mark.parametrize("branching", [2, 3, 8])
@pytest.mark.parametrize("leaf,depth", [(1, 32), (4, 32), (1, 1), (1, 0)])
def test_tree_encloses_descendants_and_partition_covers_every_pair_once(branching, leaf, depth):
    extents = _extents(); tree = _tree(extents, leaf, branching, depth)
    nodes, order = tree.nodes, tree.pair_order
    assert sorted(order) == list(range(len(extents.pairs)))
    assert len(nodes) <= tree.plan.maximum_nodes and tree.depth <= depth
    for i, node in enumerate(nodes):
        assert node.end > node.begin
        if node.children:
            children = [nodes[j] for j in node.children]
            assert children[0].begin == node.begin and children[-1].end == node.end
            assert [x.end for x in children[:-1]] == [x.begin for x in children[1:]]
            assert max(c.end-c.begin for c in children)-min(c.end-c.begin for c in children) <= 1
            for child in children:
                assert child.parent == i and child.depth == node.depth+1
                assert np.linalg.norm(np.array(child.center)-node.center)+child.radius <= node.radius
        else:
            assert node.end-node.begin <= leaf or node.depth == depth
        for pair_id in order[node.begin:node.end]:
            pair = extents.pairs[pair_id]
            assert np.linalg.norm(np.array(pair.center)-node.center)+pair.radius <= node.radius
    for opening in [0., .5, .95]:
        traversal = tree.traversal(opening)
        blocks = list(traversal)
        _covered(tree, blocks)
        r = traversal.receipt
        assert r.complete and not r.failed and r.pair_interactions == len(order)*(len(order)+1)//2
        assert r.near_blocks+r.far_blocks == len(blocks)
        assert r.maximum_stack <= 1+70*tree.depth
        assert len(r.partition_identity) == 64
        if opening == 0:
            assert r.far_blocks == 0
    if depth == 32 and leaf == 1:
        assert traversal.receipt.far_blocks > 0
        assert len(blocks) < len(order)*(len(order)+1)//2


def test_all_candidates_can_be_refined_without_losing_or_repeating_interactions():
    tree = _tree(_extents())
    cursor = tree.traversal(.95); blocks = []
    with pytest.raises(RuntimeError, match="pending"):
        cursor.accept()
    initial = cursor.receipt
    while (block := cursor.peek()) is not None:
        assert cursor.peek().target == block.target  # peeking does no extra work
        if block.far_candidate:
            cursor.refine()
        else:
            blocks.append(block); cursor.accept()
    _covered(tree, blocks)
    assert cursor.receipt.far_blocks == 0 and cursor.receipt.refined_candidates > 0
    assert not initial.complete and initial.visited_node_pairs == 0
    exact = tree.traversal(0.)
    assert [(b.target, b.source) for b in blocks] == [(b.target, b.source) for b in exact]
    assert cursor.receipt.partition_identity != exact.receipt.partition_identity
    with pytest.raises(RuntimeError, match="candidate"):
        cursor.refine()
    assert cursor.peek() is None and cursor.receipt.complete


def test_mixed_always_near_descends_to_isolate_eligible_pairs():
    extents = _extents(mixed=True); tree = _tree(extents)
    assert tree.nodes[0].always_near
    flags = [p.always_near for p in extents.pairs]
    assert any(flags) and not all(flags)
    cursor = tree.traversal(.9); blocks = list(cursor); _covered(tree, blocks)
    assert cursor.receipt.far_blocks > 0
    for block in blocks:
        if block.far_candidate:
            for node_id in [block.target, block.source]:
                node = tree.nodes[node_id]
                assert not any(flags[k] for k in tree.pair_order[node.begin:node.end])


def test_coincident_centers_terminate_with_deterministic_tie_breaking():
    extents = _extents(coincident=True)
    a, b = _tree(extents, branching=8), _tree(extents, branching=8)
    assert a.pair_order == b.pair_order == list(range(len(extents.pairs)))
    assert a.source_identity == b.source_identity
    cursor = a.traversal(.99); blocks = list(cursor); _covered(a, blocks)
    assert cursor.receipt.far_blocks == 0
    other = b.traversal(.99); list(other)
    assert cursor.receipt.partition_identity == other.receipt.partition_identity


def test_translation_and_immutable_snapshot_lifetime():
    extents = _extents(); options = manta.BubbleOptions(); options.leaf_capacity = 1
    tree = manta.BubbleTree(extents, options); identity = tree.source_identity
    options.branching = 8; options.leaf_capacity = 200
    assert tree.source_identity == identity
    shifted = _tree(_extents(shift=(1., -2., 4.)))
    assert tree.pair_order == shifted.pair_order
    for a, b in zip(tree.nodes, shifted.nodes, strict=True):
        np.testing.assert_allclose(np.array(a.center)+[1., -2., 4.], b.center, atol=2e-14, rtol=0.)
        assert a.radius == pytest.approx(b.radius, rel=1e-14, abs=1e-14)
    assert tree.source_identity != shifted.source_identity
    assert tree.source_identity != _tree(extents, branching=8).source_identity
    copied_order = tree.pair_order; copied_order.clear()
    assert tree.pair_order
    cursor = tree.traversal(.8); receipt = cursor.receipt
    del extents, tree; gc.collect()
    assert len(list(cursor)) > 0 and cursor.receipt.complete
    assert receipt.visited_node_pairs == 0 and not receipt.complete


@pytest.mark.parametrize("field,plan_field", [("memory_bytes", "modeled_peak_bytes"), ("scalar_terms", "scalar_terms"), ("kernel_terms", "kernel_terms")])
def test_construction_resource_boundaries(field, plan_field):
    extents = _extents(); options = manta.BubbleOptions(); options.leaf_capacity = 1
    plan = manta.plan_bubbles(extents, options)
    budget = manta.ResourceBudget(); setattr(budget, field, getattr(plan, plan_field))
    tree = manta.BubbleTree(extents, options, budget)
    assert tree.plan.modeled_peak_bytes == plan.modeled_peak_bytes
    setattr(budget, field, getattr(plan, plan_field)-1)
    with pytest.raises(ValueError, match="budget"):
        manta.BubbleTree(extents, options, budget)


@pytest.mark.parametrize("field", ["memory_bytes", "shell_calls", "scalar_terms", "kernel_terms"])
def test_traversal_caps_abort_and_cannot_resume(field):
    tree = _tree(_extents()); baseline = tree.traversal(.8); list(baseline); receipt = baseline.receipt
    limits = dict(memory_bytes=receipt.modeled_peak_bytes, shell_calls=receipt.visited_node_pairs,
                  scalar_terms=receipt.pair_interactions, kernel_terms=128*receipt.visited_node_pairs)
    budget = manta.ResourceBudget(); setattr(budget, field, limits[field])
    exact = tree.traversal(.8, budget); list(exact)
    assert exact.receipt.complete and exact.receipt.partition_identity == receipt.partition_identity
    setattr(budget, field, limits[field]-1)
    if field == "memory_bytes":
        with pytest.raises(ValueError, match="budget"):
            tree.traversal(.8, budget)
    else:
        cursor = tree.traversal(.8, budget)
        with pytest.raises(ValueError, match="budget"):
            list(cursor)
        assert cursor.receipt.failed and not cursor.receipt.complete
        assert not cursor.receipt.partition_identity
        for action in [cursor.peek, cursor.accept, cursor.refine]:
            with pytest.raises(RuntimeError, match="cannot resume"):
                action()


def test_storage_plan_grows_linearly_in_pairs_and_depth_caps_keep_every_object():
    small, big = _extents(4), _extents(20)
    budget = manta.ResourceBudget(); budget.backend_margin_bytes = 65536
    options = manta.BubbleOptions(); options.leaf_capacity = 1
    plans = [manta.plan_bubbles(x, options, budget) for x in [small, big]]
    memory_ratio = (plans[1].modeled_peak_bytes-65536)/(plans[0].modeled_peak_bytes-65536)
    assert memory_ratio < 1.1*len(big.pairs)/len(small.pairs)
    options.maximum_depth = 1
    tree = manta.BubbleTree(big, options); cursor = tree.traversal(0.)
    blocks = list(cursor); _covered(tree, blocks)
    assert len(tree.nodes) == 4 and cursor.receipt.near_blocks == 6
    assert cursor.receipt.maximum_stack <= 6


@pytest.mark.parametrize("field,value", [("leaf_capacity", 0), ("branching", 1), ("branching", 9), ("maximum_depth", -1), ("maximum_depth", 65)])
def test_invalid_tree_controls(field, value):
    options = manta.BubbleOptions(); setattr(options, field, value)
    with pytest.raises(ValueError):
        manta.BubbleTree(_extents(1), options)


@pytest.mark.parametrize("opening", [-.1, 1., float("nan"), float("inf")])
def test_invalid_opening(opening):
    with pytest.raises(ValueError):
        _tree(_extents(1)).traversal(opening)
