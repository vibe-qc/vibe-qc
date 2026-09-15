"""Shared mathematical symmetry and independent molecular/chi consumers.

No test changes a solver switch or certifies finite-support production J/K.
"""
from __future__ import annotations

from dataclasses import replace
from math import hypot

import numpy as np
import pytest

import vibeqc as vq
from vibeqc import _vibeqc_core as core
from vibeqc.symmetry_shared import (
    BlockSpaceAction, Budget, EvidenceKind, FiniteGroup, OperatorContract,
    QualificationEvidence, SpaceIdentity, audit_equivariance, gather_orbit_adjoint,
    orbit_of, scatter_orbit, audit_metric_panels, audit_subspace_panels,
    BlochCharacter, audit_group_transport, audit_selected_group_transport,
    audit_group_operators,
    audit_selected_operators,
)

BUDGET = Budget(64 << 20, 10**9)
SPACE = SpaceIdentity("fixture-geometry", "fixture-basis", "AO", "native-order")
CONTRACT = OperatorContract("fixture-linear-Fock", "test-source", "full", "none", "molecular")


def group(table, *, anti=None, rotations=None, cocycle=None, identity=0, budget=BUDGET):
    table = np.asarray(table, dtype=np.int64)
    return FiniteGroup.from_table(table, np.zeros(len(table), dtype=np.uint8) if anti is None else anti,
        identity=identity, identity_label="test group", budget=budget,
        rotations=rotations, cocycle=cocycle)


def block(matrix, *, anti=False, source=SPACE, target=SPACE, budget=BUDGET):
    matrix = np.asarray(matrix, dtype=np.complex128)
    return BlockSpaceAction(source, target, np.array([0, len(matrix)], dtype=np.int64),
        np.array([0], dtype=np.int64), matrix.ravel(), antiunitary=anti, budget=budget)


def panel(n, columns=3):
    rng = np.random.default_rng(717)
    return np.ascontiguousarray(rng.normal(size=(n, columns))+1j*rng.normal(size=(n, columns)))


@pytest.mark.parametrize("anti", [False, True])
def test_equivariance_retains_reference_when_builder_reuses_output(anti):
    action = block([[0, 1], [1, 0]], anti=anti)
    density = np.diag([1., 2.]).astype(complex)
    scratch = np.empty((2, 2), dtype=complex)

    def builder(d):
        # This deliberately non-equivariant source returns the same workspace
        # on both calls, as an otherwise valid numerical backend may do.
        scratch[:] = d[0, 0] * np.eye(2)
        return scratch

    evidence = audit_equivariance(action, density, builder, contract=CONTRACT,
        tolerance=1e-12, probe_identity="reused Fock workspace")
    assert evidence.residual == pytest.approx(1.)
    assert not evidence.passed_probe


def test_equivariance_refuses_invalid_reference_before_second_build():
    calls = 0
    scratch = np.empty((2, 2), dtype=complex)

    def builder(d):
        nonlocal calls
        calls += 1
        scratch[:] = np.nan if calls == 1 else 0.
        return scratch

    with pytest.raises(ValueError, match="invalid operator"):
        audit_equivariance(block(np.eye(2)), np.eye(2, dtype=complex), builder,
            contract=CONTRACT, tolerance=1e-12, probe_identity="invalid first Fock")
    assert calls == 1


def test_equivariance_requires_target_builder_for_different_named_spaces():
    target = replace(SPACE, gauge="rescaled AO gauge")
    action = block(np.diag([2., 0.5]), target=target)
    calls = []

    def builder(d):
        calls.append(d)
        return np.zeros_like(d)

    with pytest.raises(ValueError, match="target.*builder"):
        audit_equivariance(action, np.eye(2, dtype=complex), builder,
            contract=CONTRACT, tolerance=1e-12, probe_identity="undeclared target operator")
    assert not calls


@pytest.mark.parametrize("anti", [False, True])
def test_equivariance_uses_target_gauge_builder_with_nonorthogonal_action(anti):
    target = replace(SPACE, gauge="rescaled conjugate gauge" if anti else "rescaled AO gauge")
    action = block(np.diag([2., 0.5]), target=target, anti=anti)
    density = np.array([[1., 0.2j], [-0.2j, 2.]], dtype=complex)
    h = np.array([[1., 0.3j], [-0.3j, 3.]], dtype=complex)
    # Target basis functions are rescaled source functions. Derive the
    # target formula in that basis, retaining the covariant AO factors.
    basis_change = np.diag([0.5, 2.]).astype(complex)
    target_metric = basis_change.conj().T @ basis_change
    target_h = basis_change.conj().T @ (h.conj() if anti else h) @ basis_change
    scratch = np.empty((2, 2), dtype=complex)
    calls = []

    def source_builder(d):
        calls.append("source")
        scratch[:] = h + d
        return scratch

    def target_builder(d):
        calls.append("target")
        scratch[:] = target_h + target_metric @ d @ target_metric
        return scratch

    evidence = audit_equivariance(action, density, source_builder,
        target_builder=target_builder, contract=CONTRACT, tolerance=1e-12,
        probe_identity="explicit source and target AO formulas")
    assert calls == ["source", "target"]
    assert evidence.passed_probe
    assert evidence.source_space == SPACE and evidence.target_space == target
    assert not evidence.production_reduction_authorized


def test_equivariance_target_callback_cannot_hide_a_different_operator():
    action = block(np.eye(2), target=replace(SPACE, gauge="target gauge"))
    evidence = audit_equivariance(action, np.eye(2, dtype=complex), lambda d: d.copy(),
        target_builder=lambda d: 2*d, contract=CONTRACT, tolerance=1e-12,
        probe_identity="mismatched target formula")
    assert evidence.residual == pytest.approx(1.)
    assert not evidence.passed_probe


def test_group_scalar_time_reversal_and_reordered_identity():
    g = group([[0, 1], [1, 0]], anti=np.array([0, 1], dtype=np.uint8))
    assert g.order == 2 and g.identity == 0
    assert g.product(1, 1) == 0 and g.inverse(1) == 1
    assert g.antiunitary(1) and not g.antiunitary(0)
    assert g.cocycle(1, 1) == (0, 0, 0)
    g = group([[1, 0], [0, 1]], identity=1)
    assert g.product(0, 0) == 1 and g.inverse(0) == 0


def test_conjugacy_admission_retains_the_borrowed_group_inventory():
    g = group([[0, 1], [1, 0]])
    scratch_bytes = 4096 + 256*g.order
    with pytest.raises(MemoryError, match="byte budget"):
        g.conjugacy_classes(budget=Budget(scratch_bytes, BUDGET.maximum_work))
    exact = Budget(scratch_bytes + g._native.memory.bytes, BUDGET.maximum_work)
    assert g.conjugacy_classes(budget=exact) == ((0,), (1,))


def test_group_full_fractional_translation_cocycle_is_not_reduced_mod_mesh():
    r = np.tile(np.eye(3, dtype=np.int64), (2, 1, 1))
    c = np.zeros((2, 2, 3), dtype=np.int64)
    c[1, 1] = [2**40+1, -7, 5]
    g = group([[0, 1], [1, 0]], rotations=r, cocycle=c)
    assert g.cocycle(1, 1) == (2**40+1, -7, 5)
    c[1, 1] = 0  # native group is an immutable owned snapshot
    assert g.cocycle(1, 1)[0] == 2**40+1


def test_noncommuting_lattice_group_and_twisted_translation_section():
    quarter_turn = np.array([[0, -1, 0], [1, 0, 0], [0, 0, 1]], dtype=np.int64)
    mirror = np.diag([-1, 1, 1]).astype(np.int64)
    rotations = np.array([np.linalg.matrix_power(quarter_turn, k) @ reflection
                          for reflection in (np.eye(3, dtype=np.int64), mirror)
                          for k in range(4)])
    table = np.empty((8, 8), dtype=np.int64)
    # Changing the integer translation representative of each point coset
    # generates an exact, nonzero section cocycle with a nontrivial twist.
    shifts = np.array([[0, 0, 0], [2, -3, 1], [-4, 2, 0], [1, 0, 2],
                       [3, 1, -2], [0, -5, 1], [2, 4, 0], [-1, 1, 3]], dtype=np.int64)
    cocycle = np.empty((8, 8, 3), dtype=np.int64)
    for g in range(8):
        for h in range(8):
            product = rotations[g] @ rotations[h]
            k = next(i for i, r in enumerate(rotations) if np.array_equal(r, product))
            table[g, h] = k
            cocycle[g, h] = shifts[g]+rotations[g] @ shifts[h]-shifts[k]
    admitted = group(table, rotations=rotations, cocycle=cocycle)
    assert admitted.product(1, 4) != admitted.product(4, 1)
    quotient = FiniteGroup.from_integer_rotations(
        rotations, identity_label="D4 rotation quotient", budget=BUDGET,
    )
    assert quotient.conjugacy_classes(budget=BUDGET) == (
        (0,), (1, 3), (2,), (4, 6), (5, 7),
    )
    for g in range(8):
        for h in range(8):
            assert quotient.product(g, h) == table[g, h]
    assert admitted.cocycle(1, 4) == tuple(cocycle[1, 4])
    cocycle[1, 4, 0] += 1
    with pytest.raises(ValueError, match="cocycle"):
        group(table, rotations=rotations, cocycle=cocycle)


@pytest.mark.parametrize("table,anti,message", [
    ([[0, 2], [1, 0]], [0, 0], "outside"),
    ([[0, 1], [1, 1]], [0, 0], "inverse"),
    ([[0, 1], [1, 0]], [0, 2], "zero or one"),
    ([[0, 1], [1, 0]], [1, 0], "identity"),
    ([[0, 1, 2], [1, 2, 0], [2, 0, 1]], [0, 1, 0], "grading"),
    ([[0, 1, 2], [1, 0, 0], [2, 0, 0]], [0, 0, 0], "associativity"),
])
def test_invalid_group_refused(table, anti, message):
    with pytest.raises(ValueError, match=message):
        group(table, anti=np.array(anti, dtype=np.uint8))


def test_lattice_rotation_and_cocycle_failures():
    r = np.tile(np.eye(3, dtype=np.int64), (2, 1, 1))
    c = np.zeros((2, 2, 3), dtype=np.int64)
    r[1] = -np.eye(3, dtype=np.int64)
    c[1, 1, 0] = 1  # inversion square cannot translate along reversed axis
    with pytest.raises(ValueError, match="cocycle"):
        group([[0, 1], [1, 0]], rotations=r, cocycle=c)
    c[:] = 0
    r[1, 0, 0] = 2
    with pytest.raises(ValueError, match="rotation product"):
        group([[0, 1], [1, 0]], rotations=r, cocycle=c)
    r[1, 0, 0] = 2**62
    with pytest.raises(OverflowError, match="integer overflow"):
        group([[0, 1], [1, 0]], rotations=r, cocycle=c)


def test_native_group_admission_before_payload_values():
    table = np.full((2, 2), -1, dtype=np.int64)
    with pytest.raises(ValueError, match="byte budget"):
        group(table, budget=Budget(1, 10**9))
    with pytest.raises(ValueError, match="work budget"):
        group(table, budget=Budget(100000, 1))
    with pytest.raises(OverflowError, match="count overflow"):
        core._plan_symmetry_group(2**62, True, BUDGET.native())
    valid = core._plan_symmetry_group(2, False, BUDGET.native())
    exact = group([[0, 1], [1, 0]], budget=Budget(valid.bytes, valid.work))
    assert exact.product(1, 1) == 0


@pytest.mark.parametrize("anti", [False, True])
def test_complex_block_transport_and_real_adjoint(anti):
    m = np.array([[1+2j, 2-1j], [-0.2j, 0.5]])
    action = block(m, anti=anti)
    x, y = panel(2), panel(2)[:, ::-1].copy()
    expected = m @ (x.conj() if anti else x)
    np.testing.assert_allclose(action.apply(x), expected, atol=1e-14)
    expected_adj = m.conj().T @ y
    if anti:
        expected_adj = expected_adj.conj()
    np.testing.assert_allclose(action.adjoint(y), expected_adj, atol=1e-14)
    assert np.vdot(action.apply(x), y).real == pytest.approx(np.vdot(x, action.adjoint(y)).real)


def test_block_permutation_and_immutable_snapshots():
    offsets = np.array([0, 2, 3, 5], dtype=np.int64)
    dest = np.array([2, 1, 0], dtype=np.int64)
    matrices = np.concatenate([np.array([[0, 1], [-1, 0]]).ravel(), [2], np.eye(2).ravel()]).astype(complex)
    a = BlockSpaceAction(SPACE, SPACE, offsets, dest, matrices, antiunitary=False, budget=BUDGET)
    x = panel(5)
    expected = np.vstack([x[3:5], 2*x[2:3], x[1:2], -x[0:1]])
    np.testing.assert_allclose(a.apply(x), expected)
    matrices[:] = 0
    np.testing.assert_allclose(a.apply(x), expected)
    for array in (a.offsets, a.destinations, a.matrices):
        with pytest.raises(ValueError):
            array.setflags(write=True)


def test_many_scalar_blocks_fit_linear_transport_budget():
    n = 4096
    # This budget admits linear descriptors and one panel, but cannot admit
    # a quadratic permutation census or a dense all-space rotation matrix.
    budget = Budget(128*n+8192, 128*n)
    action = BlockSpaceAction(SPACE, SPACE, np.arange(n+1, dtype=np.int64),
        (np.arange(n, dtype=np.int64)+1) % n, np.ones(n, dtype=complex),
        antiunitary=False, budget=budget)
    x = np.arange(n, dtype=np.complex128).reshape(n, 1)
    np.testing.assert_array_equal(action.apply(x), np.roll(x, 1, axis=0))


@pytest.mark.parametrize("anti", [False, True])
def test_nonorthogonal_metric_and_tensor_variance(anti):
    # Nonunitary AO representation obtained by changing basis in an inversion.
    v = np.array([[1, 0.3j], [0.2, 1.5]], dtype=complex)
    u = np.linalg.inv(v) @ np.diag([1., -1.]) @ (v.conj() if anti else v)
    a = block(u, anti=anti)
    metric = np.ascontiguousarray(v.conj().T @ v)
    np.testing.assert_allclose(a.pull_operator(metric), metric, atol=2e-14)
    assert np.linalg.norm(u.conj().T @ u-np.eye(2)) > 0.1
    x = panel(2, 2)
    d = np.ascontiguousarray(x @ x.conj().T)
    f = np.ascontiguousarray(x+x.conj().T)
    moved = u @ (d.conj() if anti else d) @ u.conj().T
    np.testing.assert_allclose(a.push_density(d), moved, atol=2e-14)
    assert np.trace(a.push_density(d) @ f).real == pytest.approx(np.trace(d @ a.pull_operator(f)).real)


@pytest.mark.parametrize("offsets,dest,mats,message", [
    ([0, 1, 2], [0, 0], [1, 1], "permutation"),
    ([0, 1, 3], [1, 0], [1]*5, "rank"),
    ([0, 0, 2], [0, 1], [1]*4, "increase"),
    ([0, 1, 2], [0, 2], [1, 1], "outside"),
    ([0, 2], [0], [1], "extent"),
    ([0, 1], [0], [np.nan], "nonfinite"),
])
def test_malformed_block_actions(offsets, dest, mats, message):
    with pytest.raises(ValueError, match=message):
        BlockSpaceAction(SPACE, SPACE, np.array(offsets, dtype=np.int64), np.array(dest, dtype=np.int64),
                         np.array(mats, dtype=complex), antiunitary=False, budget=BUDGET)


def test_transport_limits_zero_columns_and_nonfinite_payload():
    a = block(np.eye(2))
    assert a.apply(np.empty((2, 0), dtype=complex)).shape == (2, 0)
    with pytest.raises(ValueError, match="nonfinite"):
        a.apply(np.full((2, 1), np.inf, dtype=complex))
    with pytest.raises(TypeError, match="complex128"):
        a.apply(np.ones((2, 1)))
    with pytest.raises(MemoryError, match="budget"):
        block(np.eye(2), budget=Budget(1, 10**9))


def test_native_block_plan_is_enforced_before_payload_scan():
    offsets = np.array([0, 2], dtype=np.int64)
    destinations = np.array([0], dtype=np.int64)
    matrices = np.eye(2, dtype=complex).ravel()
    x = panel(2)
    plan = core._plan_symmetry_blocks(2, 1, 4, 3, BUDGET.native())
    result = core._apply_symmetry_blocks(offsets, destinations, matrices, x, False, False,
                                         Budget(plan.bytes, plan.work).native())
    np.testing.assert_allclose(result, x)
    matrices[:] = np.nan
    for budget, message in [(Budget(plan.bytes-1, plan.work), "byte budget"),
                            (Budget(plan.bytes, plan.work-1), "work budget")]:
        with pytest.raises(ValueError, match=message):
            core._apply_symmetry_blocks(offsets, destinations, matrices, x, False, False,
                                         budget.native())
    with pytest.raises(OverflowError, match="count overflow"):
        core._plan_symmetry_blocks(2**62, 1, 1, 2**62, BUDGET.native())


@pytest.mark.parametrize("n,expected", [(3, 2), (4, 3), (8, 5)])
def test_even_mesh_inversion_orbits_have_exact_stabilizers(n, expected):
    g = group([[0, 1], [1, 0]])
    remaining, orbits = set(range(n)), []
    while remaining:
        o = orbit_of(g, min(remaining), lambda op, p: (-p if op else p) % n, budget=BUDGET)
        orbits.append(o)
        remaining.difference_update(o.members)
        assert o.weight*len(o.stabilizer) == 2
    assert len(orbits) == expected
    if n % 2 == 0:
        o = orbit_of(g, n//2, lambda op, p: (-p if op else p) % n, budget=BUDGET)
        assert o.stabilizer == (0, 1) and o.weight == 1


def test_orbit_refuses_invalid_action_and_unrepresented_translation_cocycle():
    g = group([[0, 1], [1, 0]])
    with pytest.raises(ValueError, match="quotient"):
        orbit_of(g, 0, lambda op, p: p+op, budget=BUDGET)


def test_stabilizer_scatter_and_adjoint():
    g = group([[0, 1], [1, 0]])
    o = orbit_of(g, 0, lambda op, p: 0, budget=BUDGET)
    actions = (block(np.eye(2)), block(np.diag([1., -1.])))
    x = np.array([[2+1j], [0j]])
    np.testing.assert_allclose(scatter_orbit(o, x, actions, budget=BUDGET, tolerance=1e-12)[0], x)
    with pytest.raises(ValueError, match="stabilizer"):
        scatter_orbit(o, np.ones((2, 1), dtype=complex), actions, budget=BUDGET, tolerance=1e-12)
    # Free orbit: dense occupied mixing and antiunitary adjoint both retained.
    g = group([[0, 1], [1, 0]], anti=np.array([0, 1], dtype=np.uint8))
    o = orbit_of(g, 0, lambda op, p: (op+p) % 2, budget=BUDGET)
    actions = (block(np.eye(2)), block(np.array([[0, 1j], [1j, 0]]), anti=True))
    x = panel(2)
    y = (panel(2), panel(2)*2j)
    sx = scatter_orbit(o, x, actions, budget=BUDGET, tolerance=1e-12)
    lhs = sum(np.vdot(a, b).real for a, b in zip(sx, y))
    assert lhs == pytest.approx(np.vdot(x, gather_orbit_adjoint(o, y, actions, budget=BUDGET)).real)


def test_probe_evidence_does_not_authorize_reduction():
    a = block([[0, 1], [1, 0]])
    d = np.array([[1., 0.1j], [-0.1j, 0.2]], dtype=complex)
    good = audit_equivariance(a, d, lambda p: np.ascontiguousarray(2*p+np.eye(2)),
                             contract=CONTRACT, tolerance=1e-12, probe_identity="nonsymmetric density")
    assert good.passed_probe and not good.production_reduction_authorized
    bad = audit_equivariance(a, d, lambda p: np.ascontiguousarray(p+np.diag([0., 1.])),
                            contract=CONTRACT, tolerance=1e-12, probe_identity="broken source support")
    assert not bad.passed_probe and bad.residual == pytest.approx(1.)
    analytic = QualificationEvidence(CONTRACT, SPACE, SPACE, "equivariance", EvidenceKind.ANALYTIC,
                                      "method-owned derivation pending independent review")
    assert not analytic.production_reduction_authorized


def test_molecular_integral_consumer_and_group():
    from vibeqc.symmetry_ao import build_molecular_space_action, molecular_point_group
    mol = vq.Molecule([vq.Atom(8, [0., 0., 0.]), vq.Atom(1, [1.5, 0., -1.2]), vq.Atom(1, [-1.5, 0., -1.2])])
    basis = vq.BasisSet(mol, "sto-3g")
    rotations = np.array([np.eye(3), np.diag([-1., -1., 1.])])
    g = molecular_point_group(mol, rotations, np.zeros((2, 3)), identity_label="water C2", budget=BUDGET)
    assert g.product(1, 1) == 0
    a = build_molecular_space_action(mol, basis, rotations[1].copy(), np.zeros(3),
                                    source=SPACE, target=SPACE, budget=BUDGET)
    c = panel(basis.nbasis)
    old = vq.build_ao_permutation_matrix(basis, rotations[1], vq.atom_permutation_under_op(mol, rotations[1]))
    np.testing.assert_allclose(a.apply(c), old @ c, atol=1e-13)
    overlap = np.ascontiguousarray(vq.compute_overlap(basis), dtype=complex)
    metric_evidence = audit_metric_panels(a, c, overlap, overlap,
        contract=CONTRACT, tolerance=1e-10, probe_identity="molecular AO metric", budget=BUDGET)
    assert metric_evidence.passed_probe
    # Source-distinct unreduced molecular integrals, not a periodic surrogate.
    for operator in (vq.compute_overlap(basis), vq.compute_kinetic(basis), vq.compute_nuclear(basis, mol)):
        operator = np.ascontiguousarray(operator, dtype=complex)
        np.testing.assert_allclose(a.pull_operator(operator), operator, atol=1e-10)
    eri = np.asarray(vq.compute_eri(basis)).reshape((basis.nbasis,)*4)
    h = np.asarray(vq.compute_kinetic(basis))+np.asarray(vq.compute_nuclear(basis, mol))
    def fock(density):
        return np.ascontiguousarray(h+np.einsum("ijkl,kl->ij", eri, density)
                                   -0.5*np.einsum("ikjl,kl->ij", eri, density), dtype=complex)
    trial = panel(basis.nbasis, basis.nbasis)
    trial = np.ascontiguousarray(trial+trial.conj().T)
    evidence = audit_equivariance(a, trial, fock, contract=replace(CONTRACT,
        operator="molecular full four-center RHF", source_revision="current native test build"),
        tolerance=1e-10, probe_identity="water nonsymmetric Hermitian density")
    assert evidence.passed_probe
    assert not evidence.production_reduction_authorized


@pytest.mark.parametrize("anti", [False, True])
def test_molecular_rounded_identity_preserves_dense_and_compact_actions(anti):
    from vibeqc.symmetry_ao import build_molecular_space_action

    mol = vq.Molecule([vq.Atom(8, [0., 0., 0.]),
                       vq.Atom(1, [1.5, 0., -1.2]), vq.Atom(1, [-1.5, 0., -1.2])])
    basis = vq.BasisSet(mol, "6-31g*")
    rotation = np.eye(3)
    rotation[0, 1] = 4.5102810375396984e-17
    rotation[2, 2] = np.nextafter(1.0, 0.0)
    dense = vq.build_ao_permutation_matrix(
        basis, rotation, vq.atom_permutation_under_op(mol, rotation))
    action = build_molecular_space_action(mol, basis, rotation, np.zeros(3),
        source=SPACE, target=SPACE, budget=BUDGET, antiunitary=anti)
    coefficients = panel(basis.nbasis, 3)
    np.testing.assert_allclose(dense, np.eye(basis.nbasis), atol=1e-14, rtol=0.0)
    np.testing.assert_allclose(action.apply(coefficients),
        coefficients.conj() if anti else coefficients, atol=1e-14, rtol=0.0)
    for matrix in (vq.compute_overlap(basis), vq.compute_kinetic(basis),
                   vq.compute_nuclear(basis, mol)):
        matrix = np.ascontiguousarray(matrix, dtype=complex)
        np.testing.assert_allclose(action.pull_operator(matrix), matrix, atol=1e-12, rtol=0.0)


def test_molecular_radial_mismatch_fails_closed():
    from vibeqc.symmetry_ao import build_molecular_space_action, build_ao_permutation_matrix
    mol = vq.Molecule([vq.Atom(2, [-1., 0., 0.]), vq.Atom(2, [1., 0., 0.])])
    shells = [core.ShellInfo(i, 0, True, [0.8+i*0.1], [1.], list(atom.xyz))
              for i, atom in enumerate(mol.atoms)]
    basis = vq.BasisSet(mol, shells, "asymmetric radial", False)
    with pytest.raises(ValueError, match="radial"):
        build_ao_permutation_matrix(basis, -np.eye(3), np.array([1, 0]))
    with pytest.raises(ValueError, match="radial"):
        build_molecular_space_action(mol, basis, -np.eye(3), np.zeros(3), source=SPACE, target=SPACE, budget=BUDGET)


def test_molecular_profile_matching_preserves_repeated_and_reordered_shells():
    from vibeqc.symmetry_ao import build_molecular_space_action, build_ao_permutation_matrix
    mol = vq.Molecule([vq.Atom(2, [-1., 0., 0.]), vq.Atom(2, [1., 0., 0.])])
    # Three radial channels per atom, with the identical channel repeated.
    # Destination order differs, so positional shell matching would be wrong.
    radial = [(0, 0.8), (0, 1.2), (0, 0.8), (1, 1.2), (1, 0.8), (1, 0.8)]
    shells = [core.ShellInfo(atom, 0, True, [exponent], [1.], list(mol.atoms[atom].xyz))
              for atom, exponent in radial]
    basis = vq.BasisSet(mol, shells, "reordered radial channels", False)
    action = build_molecular_space_action(mol, basis, -np.eye(3), np.zeros(3),
                                         source=SPACE, target=SPACE, budget=BUDGET)
    x = panel(6)
    np.testing.assert_array_equal(action.destinations, [4, 3, 5, 1, 0, 2])
    np.testing.assert_allclose(action.apply(x), x[[4, 3, 5, 1, 0, 2]])
    np.testing.assert_allclose(action.apply(action.apply(x)), x)
    dense = build_ao_permutation_matrix(basis, -np.eye(3), np.array([1, 0]))
    np.testing.assert_array_equal(dense, np.eye(6)[[4, 3, 5, 1, 0, 2]])
    overlap = np.asarray(vq.compute_overlap(basis))
    np.testing.assert_allclose(dense @ overlap @ dense.T, overlap, atol=1e-13, rtol=0)


@pytest.mark.parametrize("anti", [False, True])
def test_chi_shared_bloch_space_adapter_has_full_half_translation(anti):
    from tests.test_periodic_ao_bloch_transport import _controls, _operation, _basis
    from vibeqc.periodic.chi.symmetry import ChiAOBlochSpaceAction
    system = vq.PeriodicSystem(3, np.diag([8., 16., 16.]),
        [vq.Atom(2, [0., 0., 0.]), vq.Atom(2, [4., 0., 0.])])
    basis = _basis(system, (0,))
    mesh = core._RegularKMesh([3, 1, 1], [0, 0, 0])
    action = ChiAOBlochSpaceAction(SPACE, replace(SPACE, gauge="k=-1/3" if anti else "k=1/3"), basis, system,
        _operation(translation=[0.5, 0., 0.]), mesh, 1, anti, *_controls())
    c = panel(2)
    # Native AO-centre convention: atom1 maps to atom0 with ell=-1.
    q = -1/3 if anti else 1/3
    x = c.conj() if anti else c
    expected = np.vstack([np.exp(-2j*np.pi*q)*x[1], x[0]])
    np.testing.assert_allclose(action.apply(c), expected, atol=1e-13)
    # Algebraic metric witness for the panel adapter. This identity metric is
    # not claimed to be the physical periodic overlap or a #704 source gate.
    metric = np.eye(2, dtype=complex)
    witness = audit_metric_panels(action, c, metric, metric, contract=CONTRACT,
        tolerance=1e-12, probe_identity="half-translation phase metric", budget=BUDGET)
    assert witness.passed_probe and not witness.production_reduction_authorized


def test_shared_metric_probe_reports_metric_loss_without_projection():
    metric = np.eye(2, dtype=complex)
    a = block([[1., 0.], [0., 0.5]])
    witness = audit_metric_panels(a, np.eye(2, dtype=complex), metric, metric,
        contract=CONTRACT, tolerance=1e-12, probe_identity="scaled local panel negative", budget=BUDGET)
    assert not witness.passed_probe and witness.residual == pytest.approx(0.75)


def subspace_probe(action, source, target, sm=None, tm=None, **kwargs):
    options = dict(source_subspace=replace(action.source, space="source occupied", gauge="localized"),
        target_subspace=replace(action.target, space="target occupied", gauge="mixed localized"),
        contract=CONTRACT, metric_tolerance=1e-10, leakage_tolerance=1e-10,
        probe_identity="retained-space probe", budget=BUDGET)
    options.update(kwargs)
    eye = np.eye(action.dimension, dtype=complex)
    return audit_subspace_panels(action, source, target, eye if sm is None else sm,
                                eye if tm is None else tm, **options)


@pytest.mark.parametrize("anti", [False, True])
def test_subspace_dense_mixing_and_nonorthogonal_ao_gauge(anti):
    # A non-Euclidean-unitary AO action in a rescaled AO basis. The same
    # retained span has independent complex source/target column gauges.
    scale = np.diag([2., 0.5, 3.])
    inverse = np.linalg.inv(scale)
    q = np.array([[1, 1j], [1j, 1]], dtype=complex)/np.sqrt(2)
    u = np.eye(3, dtype=complex)
    u[:2, :2] = q
    action = block(inverse @ u @ scale, anti=anti)
    source = np.ascontiguousarray(inverse[:, :2] @ q)
    target = np.ascontiguousarray(inverse[:, :2] @ q.conj())
    metric = np.ascontiguousarray(scale @ scale, dtype=complex)
    result = subspace_probe(action, source, target, metric, metric)
    assert result.passed_probe and not result.production_reduction_authorized
    coordinates = panel(2, 3)
    expected = action.apply(np.ascontiguousarray(source @ coordinates))
    recovered = target @ result.mixing @ (coordinates.conj() if anti else coordinates)
    np.testing.assert_allclose(recovered, expected, atol=1e-13)
    with pytest.raises(ValueError):
        result.mixing.setflags(write=True)
    with pytest.raises(ValueError):
        result.mixing[0, 0] = 9.


@pytest.mark.parametrize("anti", [False, True])
def test_subspace_separates_leakage_from_metric_loss(anti):
    c = np.ascontiguousarray(np.eye(3, dtype=complex)[:, :2])
    leaking = subspace_probe(block([[1,0,0],[0,0,1],[0,1,0]], anti=anti), c, c)
    assert leaking.metric.passed_probe
    assert leaking.containment.residual == pytest.approx(1/np.sqrt(2))
    assert not leaking.passed_probe
    scaled = subspace_probe(block(np.diag([.5, 1., 1.]), anti=anti), c, c)
    assert scaled.containment.passed_probe
    assert scaled.metric.residual == pytest.approx(.75)
    assert not scaled.passed_probe


@pytest.mark.parametrize("diagonal, moved", [([1,0,1], [1,1,0]), ([1,1,-1], [1,1,1])])
def test_subspace_leakage_cannot_cancel_in_indefinite_or_null_metric(diagonal, moved):
    c = np.array([[1.], [0.], [0.]], dtype=complex)
    u = np.eye(3, dtype=complex)
    u[:, 0] = moved
    metric = np.diag(diagonal).astype(complex)
    result = subspace_probe(block(u), c, c, metric, metric)
    assert result.metric.passed_probe
    assert result.containment.residual > .7
    assert not result.passed_probe


def test_subspace_snapshots_references_before_reused_action_workspace():
    source = np.array([[1.], [0.]], dtype=complex)
    target = np.array([[0.], [1.]], dtype=complex)
    tm = np.eye(2, dtype=complex)
    class ReusedAction:
        source = target = SPACE
        dimension = 2
        antiunitary = False
        def apply(self, c):
            assert not c.flags.writeable
            target[:] = c
            tm[:] = 0.
            return target
    result = subspace_probe(ReusedAction(), source, target, tm=tm)
    assert result.metric.passed_probe
    assert result.containment.residual == pytest.approx(1.)
    assert not result.passed_probe
    np.testing.assert_array_equal(source, [[1], [0]])


def test_subspace_keeps_finite_normalization_error_separate_from_leakage():
    source = np.array([[1.], [0.]], dtype=complex)
    target = source*(1+1e-7)
    result = subspace_probe(block(np.eye(2)), source, target,
                            metric_tolerance=1e-6, leakage_tolerance=1e-14)
    assert result.passed_probe
    np.testing.assert_allclose(target @ result.mixing, source, atol=1e-14, rtol=0)
    with pytest.raises(TypeError, match="audit_subspace_panels"):
        replace(result, mixing=np.eye(1, dtype=complex))


@pytest.mark.parametrize("scale", [1e-320,1e-310,1e-300])
@pytest.mark.parametrize("phase", [1.,.6+.8j])
@pytest.mark.parametrize("anti", [False,True])
def test_subspace_relative_leakage_survives_tiny_nonzero_transport(scale,phase,anti):
    c = np.array([[1.], [0.]], dtype=complex)
    result = subspace_probe(block([[0,0],[scale*phase,1]],anti=anti), c, c)
    assert result.containment.residual == pytest.approx(1.)
    assert not result.passed_probe


@pytest.mark.parametrize("scale,relative", [(1e-100,1e-200),(1.,1e-320),
                                            (1.,1e-200),(1e75,1e-200)])
@pytest.mark.parametrize("phase", [1.,.6+.8j])
@pytest.mark.parametrize("anti", [False,True])
def test_subspace_retains_tiny_relative_leakage(scale,relative,phase,anti):
    z = scale*relative*phase
    c = np.array([[1.],[0.]],dtype=complex)
    expected = hypot(z.real/scale,z.imag/scale)
    for factor in [.5,1.5]:
        result = subspace_probe(block([[scale,0],[z,1.]],anti=anti),c,c,
                                leakage_tolerance=factor*expected)
        residual = result.containment.residual
        assert residual > 0.
        assert residual == pytest.approx(expected,rel=1e-14,abs=np.spacing(expected))
        assert result.containment.passed_probe is (factor > 1.)
        assert not result.production_reduction_authorized


@pytest.mark.parametrize("failure", ["rank", "empty", "dependent", "metric", "identity", "nonfinite", "byte", "work"])
def test_subspace_admission_precedes_transport(failure):
    class ForbiddenAction:
        source = target = SPACE
        dimension = 2
        antiunitary = False
        def apply(self, c):
            pytest.fail("invalid subspace reached transport")
    source = np.eye(2, dtype=complex)
    target = source.copy()
    metric = source.copy()
    extra = {}
    error = ValueError
    if failure == "rank": target = target[:, :1].copy()
    if failure == "empty": source = target = np.empty((2,0), dtype=complex)
    if failure == "dependent": target[:, 1] = target[:, 0]
    if failure == "metric": metric[0,1] = .5j
    if failure == "identity": extra['target_subspace'] = replace(SPACE, geometry="wrong geometry")
    if failure == "nonfinite": source[0,0] = np.nan
    if failure == "byte":
        extra['budget'] = Budget(1, 10**9)
        source[:] = np.nan  # Metadata admission must precede payload scans.
        error = MemoryError
    if failure == "work":
        extra['budget'] = Budget(64 << 20, 1)
        source[:] = np.nan
    with pytest.raises(error, match="work budget" if failure == "work" else None):
        subspace_probe(ForbiddenAction(), source, target, tm=metric, **extra)


@pytest.mark.parametrize("failure", ["shape", "dtype", "nonfinite"])
def test_subspace_refuses_invalid_backend_output(failure):
    class BrokenAction:
        source = target = SPACE
        dimension = 2
        antiunitary = False
        def apply(self, c):
            if failure == "shape": return np.ones((2, 2), dtype=complex)
            if failure == "dtype": return np.ones((2, 1), dtype=float)
            return np.full((2, 1), np.nan, dtype=complex)
    c = np.array([[1.], [0.]], dtype=complex)
    with pytest.raises(TypeError if failure == "dtype" else ValueError):
        subspace_probe(BrokenAction(), c, c)


def test_molecular_retained_subspace_uses_unreduced_overlap_and_core_orbitals():
    from vibeqc.symmetry_ao import build_molecular_space_action
    from scipy.linalg import eigh
    mol = vq.Molecule([vq.Atom(8, [0.,0.,0.]), vq.Atom(1, [1.5,0.,-1.2]), vq.Atom(1, [-1.5,0.,-1.2])])
    basis = vq.BasisSet(mol, "sto-3g")
    metric = np.ascontiguousarray(vq.compute_overlap(basis), dtype=complex)
    h = np.asarray(vq.compute_kinetic(basis))+np.asarray(vq.compute_nuclear(basis, mol))
    _, c = eigh(h, metric)
    source = np.ascontiguousarray(c[:, :3])
    mix, _ = np.linalg.qr(panel(3, 3))
    target = np.ascontiguousarray(source @ mix)
    action = build_molecular_space_action(mol, basis, np.diag([-1.,-1.,1.]), np.zeros(3),
        source=SPACE, target=SPACE, budget=BUDGET)
    result = subspace_probe(action, source, target, metric, metric)
    assert result.passed_probe
    # Remove one retained core-Hamiltonian orbital and substitute its
    # orthogonal complement: a separately selected domain is not equivalent.
    target = np.ascontiguousarray(c[:, [0, 1, 3]])
    negative = subspace_probe(action, source, target, metric, metric)
    assert negative.metric.passed_probe and not negative.containment.passed_probe


@pytest.mark.parametrize("anti", [False, True])
def test_periodic_retained_subspace_preserves_half_translation_phase(anti):
    from tests.test_periodic_ao_bloch_transport import _controls, _operation, _basis
    from vibeqc.periodic.chi.symmetry import ChiAOBlochSpaceAction
    system = vq.PeriodicSystem(3, np.diag([8.,16.,16.]),
        [vq.Atom(2, [0.,0.,0.]), vq.Atom(2, [4.,0.,0.])])
    basis = _basis(system, (0,))
    action = ChiAOBlochSpaceAction(SPACE, replace(SPACE, gauge="target k"), basis, system,
        _operation(translation=[.5,0.,0.]), core._RegularKMesh([3,1,1],[0,0,0]),
        1, anti, *_controls())
    # Explicit one-AO retained span. This checks the phase convention and
    # domain selection, using an algebraic metric rather than a physical S.
    source = np.array([[0.], [1.]], dtype=complex)
    target = np.array([[1.], [0.]], dtype=complex)
    result = subspace_probe(action, source, target)
    q = -1/3 if anti else 1/3
    np.testing.assert_allclose(result.mixing, [[np.exp(-2j*np.pi*q)]], atol=1e-13)
    assert result.passed_probe and not result.production_reduction_authorized
    negative = subspace_probe(action, source, source)
    assert negative.metric.passed_probe and not negative.passed_probe


@pytest.mark.parametrize("anti", [False, True])
def test_periodic_retained_subspace_uses_unreduced_bloch_overlap(anti):
    from tests.test_periodic_ao_bloch_transport import _controls, _operation, _basis
    from vibeqc.periodic.chi.symmetry import ChiAOBlochSpaceAction
    from vibeqc.pbc_bipole_common import _bloch_sum_blocks
    system = vq.PeriodicSystem(3, np.diag([8.,16.,16.]),
        [vq.Atom(2, [0.,0.,0.]), vq.Atom(2, [4.,0.,0.])])
    basis = _basis(system, (0,))
    action = ChiAOBlochSpaceAction(SPACE, replace(SPACE, gauge="target k"), basis, system,
        _operation(translation=[.5,0.,0.]), core._RegularKMesh([3,1,1],[0,0,0]),
        1, anti, *_controls())
    options = core.LatticeSumOptions()
    options.cutoff_bohr = 20.
    overlap = core.compute_overlap_lattice(basis, system, options)
    blocks = [np.asarray(b, dtype=float) for b in overlap.blocks]
    def retained(q):
        # Direct unreduced overlap sum; q is fractional, the fold uses
        # Cartesian k. The chosen analytical half-translation eigenspace
        # comes from its phase convention, independently of action.apply.
        metric = np.ascontiguousarray(_bloch_sum_blocks(blocks, overlap.cells,
            np.array([2*np.pi*q/8., 0., 0.])), dtype=complex)
        c = np.array([[np.exp(-1j*np.pi*q)], [1.]], dtype=complex)
        c /= np.sqrt((c.conj().T @ metric @ c)[0,0].real)
        return metric, c
    sm, source = retained(1/3)
    q_target = -1/3 if anti else 1/3
    tm, target = retained(q_target)
    target *= np.exp(.37j)
    result = subspace_probe(action, source, target, sm, tm)
    assert result.passed_probe
    np.testing.assert_allclose(result.mixing,
        [[np.exp(-1j*np.pi*q_target-.37j)]], atol=1e-12, rtol=0)
    other = np.array([[1.], [0.]], dtype=complex)
    other -= target @ (target.conj().T @ tm @ other)
    other /= np.sqrt((other.conj().T @ tm @ other)[0,0].real)
    negative = subspace_probe(action, source, other, sm, tm)
    assert negative.metric.passed_probe and not negative.containment.passed_probe


def test_antiunitary_stabilizer_refuses_complex_representative():
    g = group([[0, 1], [1, 0]], anti=np.array([0, 1], dtype=np.uint8))
    o = orbit_of(g, 0, lambda op, p: 0, budget=BUDGET)
    actions = (block(np.eye(2)), block(np.eye(2), anti=True))
    real = np.ones((2, 1), dtype=complex)
    assert len(scatter_orbit(o, real, actions, budget=BUDGET, tolerance=1e-12)) == 1
    with pytest.raises(ValueError, match="stabilizer"):
        scatter_orbit(o, real*1j, actions, budget=BUDGET, tolerance=1e-12)


def test_orbit_gauge_identity_mismatch_is_not_silently_mixed():
    g = group([[0, 1], [1, 0]])
    o = orbit_of(g, 0, lambda op, p: (p+op) % 2, budget=BUDGET)
    actions = (block(np.eye(2)), block(np.eye(2), source=replace(SPACE, gauge="other occupied gauge")))
    with pytest.raises(ValueError, match="source space identity"):
        scatter_orbit(o, panel(2), actions, budget=BUDGET, tolerance=1e-12)


def test_native_payload_alignment_and_dtype_fail_closed():
    data = np.ones((2, 1), dtype=complex)
    raw = np.empty(data.nbytes+1, dtype=np.uint8)
    unaligned = np.ndarray(data.shape, dtype=np.complex128, buffer=raw, offset=1)
    unaligned[:] = data
    with pytest.raises(ValueError, match="aligned"):
        core._apply_symmetry_blocks(np.array([0, 2], dtype=np.int64), np.array([0], dtype=np.int64),
            np.eye(2, dtype=complex).ravel(), unaligned, False, False, BUDGET.native())
    with pytest.raises(ValueError, match="exact aligned"):
        core._make_symmetry_group(np.array([[0]], dtype=np.int32), np.array([0], dtype=np.uint8),
                                  0, None, None, BUDGET.native())


@pytest.mark.parametrize("kind", ["duplicate", "truncated", "missing_identity"])
def test_salc_rotation_quotient_refuses_non_groups(kind):
    from types import SimpleNamespace
    from vibeqc.symmetry_salc import _conjugacy_classes

    identity = np.eye(3, dtype=np.int64)
    turn = np.array([[0, -1, 0], [1, 0, 0], [0, 0, 1]], dtype=np.int64)
    rotations = {"duplicate": [identity, identity], "truncated": [identity, turn],
                 "missing_identity": [turn]}[kind]
    with pytest.raises(ValueError, match="group|identity"):
        _conjugacy_classes([SimpleNamespace(rotation=r) for r in rotations])


def test_integer_rotation_quotient_admits_before_copying_payload():
    invalid = np.zeros((2, 3, 3), dtype=np.int64)
    with pytest.raises(MemoryError, match="byte budget"):
        FiniteGroup.from_integer_rotations(invalid, identity_label="limited", budget=Budget(1, 10**9))
    with pytest.raises(ValueError, match="work budget"):
        FiniteGroup.from_integer_rotations(invalid, identity_label="limited", budget=Budget(10**6, 1))


def test_orbit_inventory_cannot_be_constructed_or_replaced_without_admission():
    from vibeqc.symmetry_shared import Orbit

    g = group([[0, 1], [1, 0]])
    with pytest.raises(TypeError, match="orbit_of"):
        Orbit(group=g, representative=0, members=(0,), transporters=((0, 1),), stabilizer=(0, 1))
    with pytest.raises(TypeError, match="admitted FiniteGroup"):
        orbit_of(object(), 0, lambda op, p: p, budget=BUDGET)
    admitted = orbit_of(g, 0, lambda op, p: (op+p) % 2, budget=BUDGET)
    with pytest.raises(TypeError, match="orbit_of"):
        replace(admitted, members=(0, 0))
    with pytest.raises(TypeError, match="admitted Orbit"):
        scatter_orbit(object(), panel(2), (block(np.eye(2)),), budget=BUDGET, tolerance=1e-12)


def test_orbit_transport_must_match_admitted_scalar_antiunitary_grading():
    g = group([[0, 1], [1, 0]])
    orbit = orbit_of(g, 0, lambda op, p: (op+p) % 2, budget=BUDGET)
    actions = (block(np.eye(2)), block(np.eye(2), anti=True))
    with pytest.raises(ValueError, match="antiunitary grading"):
        scatter_orbit(orbit, panel(2), actions, budget=BUDGET, tolerance=1e-12)
    with pytest.raises(ValueError, match="antiunitary grading"):
        gather_orbit_adjoint(orbit, (panel(2), panel(2)), actions, budget=BUDGET)


@pytest.mark.parametrize("descriptor", ["rotation", "cocycle"])
def test_native_lattice_descriptors_are_never_implicitly_materialized(descriptor):
    class DeferredArray:
        calls = 0
        def __array__(self, *args, **kwargs):
            self.calls += 1
            raise AssertionError("native admission attempted an implicit array allocation")
    deferred = DeferredArray()
    rotation = np.eye(3, dtype=np.int64)[None]
    cocycle = np.zeros((1, 1, 3), dtype=np.int64)
    if descriptor == "rotation":
        rotation = deferred
    else:
        cocycle = deferred
    with pytest.raises(TypeError, match="must be NumPy arrays"):
        core._make_symmetry_group(np.zeros((1, 1), dtype=np.int64),
            np.zeros(1, dtype=np.uint8), 0, rotation, cocycle, BUDGET.native())
    assert deferred.calls == 0


def mixing_witness(matrix, source=SPACE, target=SPACE, *, anti=False, **kwargs):
    action = block(matrix, source=source, target=target, anti=anti)
    c = np.eye(len(matrix), dtype=complex)
    return subspace_probe(action, c, c, source_subspace=source,
                          target_subspace=target, **kwargs)


def group_transport_probe(g, spaces, destinations, transports, **kwargs):
    options = dict(contract=CONTRACT, tolerance=1e-11,
                   probe_identity="all retained group operations", budget=BUDGET)
    options.update(kwargs)
    return audit_group_transport(g, spaces, destinations, transports, **options)


def test_bloch_character_evaluates_full_integer_images_before_float_conversion():
    c = BlochCharacter((2, -4, 6), 6)
    assert c == BlochCharacter((1, 1, 0), 3)
    image = (2**62+1, 0, 0)
    assert c.phase(image) == pytest.approx(c.phase((2,0,0)))
    assert abs(c.phase(image)-c.phase((1,0,0))) > 1.
    quarter = BlochCharacter((1,0,0), 4)
    assert [quarter.phase((i,0,0)) for i in range(4)] == [1., -1j, -1., 1j]
    assert BlochCharacter((5,0,0), 4) == quarter


@pytest.mark.parametrize("numerator,denominator", [
    ([1,0,0], 3), ((True,0,0), 3), ((2**63,0,0), 3),
    ((1,0,0), 0), ((1,0,0), True), ((1,0,0), 3.), ((1,0,0), 2**63),
])
def test_bloch_character_refuses_inexact_or_unbounded_descriptors(numerator, denominator):
    with pytest.raises((ValueError, TypeError)):
        BlochCharacter(numerator, denominator)


def test_group_transport_uses_the_intermediate_space_and_preserves_snapshots():
    g = group([[0,1],[1,0]])
    spaces = (SPACE, replace(SPACE, gauge="other retained gauge"))
    dest = np.array([[0,1],[1,0]], dtype=np.int64)
    u = np.array([[1,1j],[1j,1]], dtype=complex)/np.sqrt(2)
    v = np.diag([1j, np.exp(.3j)])
    m = v.conj().T @ u
    rows = ((mixing_witness(np.eye(2), spaces[0], spaces[0]),
             mixing_witness(np.eye(2), spaces[1], spaces[1])),
            (mixing_witness(m, spaces[0], spaces[1]),
             mixing_witness(m.conj().T, spaces[1], spaces[0])))
    result = group_transport_probe(g, spaces, dest, rows)
    assert result.passed_probe and not result.production_reduction_authorized
    assert np.linalg.norm(m @ m-np.eye(2)) > 1.  # Wrong source-row composition.
    wrong = (rows[0], (rows[1][0], mixing_witness(m, spaces[1], spaces[0])))
    bad = group_transport_probe(g, spaces, dest, wrong)
    assert bad.local_probes_passed and not bad.passed_probe
    assert bad.worst_composition[:2] == (1,1)
    dest[:] = 0
    np.testing.assert_array_equal(result.destinations, [[0,1],[1,0]])
    with pytest.raises(ValueError): result.destinations.setflags(write=True)
    with pytest.raises(TypeError, match="audit_group_transport"):
        replace(result, composition_residual=0.)


def test_group_transport_does_not_promote_a_nonisometric_representation():
    g = group([[0,1],[1,0]])
    rows = ((mixing_witness(np.eye(2)),), (mixing_witness([[1,2],[0,-1]]),))
    result = group_transport_probe(g, (SPACE,), np.zeros((2,1), dtype=np.int64), rows)
    assert result.identity_residual == 0.
    assert result.composition_residual == 0.
    assert not result.local_probes_passed and not result.passed_probe


def test_group_transport_requires_identity_not_only_local_isometry():
    g = group([[0]])
    result = group_transport_probe(g, (SPACE,), np.zeros((1,1), dtype=np.int64),
                                    ((mixing_witness([[-1]]),),))
    assert result.local_probes_passed
    assert result.identity_residual == pytest.approx(2.)
    assert not result.passed_probe


@pytest.mark.parametrize("fault", ["range", "grading", "rank", "contract", "space", "byte", "work"])
def test_group_transport_rejects_inconsistent_metadata_and_budget(fault):
    g = group([[0,1],[1,0]])
    dest = np.zeros((2,1), dtype=np.int64)
    rows = ((mixing_witness([[1]]),), (mixing_witness([[1]]),))
    extra = {}
    match = None
    error = ValueError
    if fault == "range": dest[1, 0] = 1
    if fault == "grading": rows = (rows[0], (mixing_witness([[1]], anti=True),))
    if fault == "rank": rows = (rows[0], (mixing_witness(np.eye(2)),))
    if fault == "contract":
        rows = (rows[0], (mixing_witness([[1]], contract=replace(CONTRACT, support="other support")),))
    if fault == "space": rows = (rows[0], (mixing_witness([[1]], target=replace(SPACE, gauge="wrong")),))
    if fault == "byte":
        dest[:] = -1
        extra['budget'] = Budget(1,10**9)
        error = MemoryError
    if fault == "work":
        dest[:] = -1
        extra['budget'] = Budget(64 << 20,1)
        match = "work budget"
    with pytest.raises(error, match=match):
        group_transport_probe(g, (SPACE,), dest, rows, **extra)


def test_group_transport_checks_the_exact_action_on_space_indices():
    g = group([[0,1],[1,0]])
    spaces = tuple(replace(SPACE, gauge=str(i)) for i in range(3))
    dest = np.array([[0,1,2],[1,2,0]], dtype=np.int64)  # A three-cycle is not C2.
    rows = tuple(tuple(mixing_witness([[1]], spaces[s], spaces[dest[op,s]])
                       for s in range(3)) for op in range(2))
    with pytest.raises(ValueError, match="destination composition"):
        group_transport_probe(g, spaces, dest, rows)
    dest[0] = [1,2,0]
    with pytest.raises(ValueError, match="destination identity"):
        group_transport_probe(g, spaces, dest, rows)


def test_group_transport_admits_payload_and_rank_work_before_matrix_products():
    g = group([[0]])
    spaces = (SPACE, replace(SPACE, gauge="disjoint rank-two space"))
    dest = np.array([[0,1]], dtype=np.int64)
    rows = ((mixing_witness([[1]]), mixing_witness(np.eye(2), spaces[1], spaces[1])),)
    result = group_transport_probe(g, spaces, dest, rows)
    assert result.passed_probe  # Disjoint group orbits may have different ranks.
    controls = 4096+g._native.memory.bytes+1024*2+2048*2
    with pytest.raises(MemoryError, match="byte budget"):
        group_transport_probe(g, spaces, dest, rows,
            budget=Budget(controls+2*dest.nbytes,10**9))
    with pytest.raises(ValueError, match="work budget"):
        group_transport_probe(g, spaces, dest, rows, budget=Budget(64 << 20,1024))


def half_translation_time_reversal_group():
    table = np.array([[(g%2+h%2)%2+2*((g//2)^(h//2)) for h in range(4)]
                      for g in range(4)], dtype=np.int64)
    images = np.zeros((4,4,3), dtype=np.int64)
    for g in range(4):
        for h in range(4): images[g,h,0] = (g%2+h%2)//2
    return group(table, anti=np.array([0,0,1,1], dtype=np.uint8),
        rotations=np.tile(np.eye(3, dtype=np.int64), (4,1,1)), cocycle=images)


def test_group_transport_refuses_dropped_or_inconsistent_bloch_characters():
    g = half_translation_time_reversal_group()
    spaces = (SPACE, replace(SPACE, gauge="negative k"))
    dest = np.array([[s^(op//2) for s in range(2)] for op in range(4)], dtype=np.int64)
    rows = tuple(tuple(mixing_witness([[1]], spaces[s], spaces[dest[op,s]], anti=bool(op//2))
                       for s in range(2)) for op in range(4))
    with pytest.raises(ValueError, match="explicit Bloch characters"):
        group_transport_probe(g, spaces, dest, rows)
    with pytest.raises(ValueError, match="scalar cocycle law"):
        group_transport_probe(g, spaces, dest, rows,
            characters=(BlochCharacter((1,0,0),3),)*2)
    gamma = group_transport_probe(g, spaces, dest, rows,
        characters=(BlochCharacter((0,0,0),1),)*2)
    assert gamma.passed_probe


@pytest.mark.parametrize("mesh_size", [2,3])
def test_periodic_retained_group_tracks_antiunitary_composition_and_lattice_phases(mesh_size):
    from tests.test_periodic_ao_bloch_transport import _controls, _operation, _basis
    from vibeqc.periodic.chi.symmetry import ChiAOBlochSpaceAction
    from vibeqc.pbc_bipole_common import _bloch_sum_blocks
    system = vq.PeriodicSystem(3, np.diag([8.,16.,16.]),
        [vq.Atom(2,[0.,0.,0.]), vq.Atom(2,[4.,0.,0.])])
    basis = _basis(system,(0,))
    mesh = core._RegularKMesh([mesh_size,1,1],[0,0,0])
    labels = [1] if mesh_size == 2 else [1,-1]
    ambient = SpaceIdentity("two He, half translation along 8-bohr x cell",
        "one test s shell per site", "AO", "native Bloch order")
    contract = OperatorContract("overlap", "direct lattice sum", "20 bohr cutoff", "none", "Bloch")
    spaces = tuple(replace(ambient, space="retained bands", gauge=f"k={q}/{mesh_size}") for q in labels)
    characters = tuple(BlochCharacter((q,0,0),mesh_size) for q in labels)
    options = core.LatticeSumOptions()
    options.cutoff_bohr = 20.
    overlap = core.compute_overlap_lattice(basis,system,options)
    blocks = [np.asarray(b,dtype=float) for b in overlap.blocks]
    metrics, panels = [], []
    for q in labels:
        metric = np.ascontiguousarray(_bloch_sum_blocks(blocks,overlap.cells,
            np.array([2*np.pi*q/(8*mesh_size),0.,0.])),dtype=complex)
        if mesh_size == 2:
            w,v = np.linalg.eigh(metric)
            gauge,_ = np.linalg.qr(panel(2,2))
            c = (v / np.sqrt(w)) @ gauge
        else:
            c = np.array([[np.exp(-1j*np.pi*q/mesh_size)],[1.]],dtype=complex)
            c /= np.sqrt((c.conj().T @ metric @ c)[0,0].real)
            c *= np.exp(.21j*q)
        metrics.append(metric)
        panels.append(np.ascontiguousarray(c))
    g = half_translation_time_reversal_group()
    dest = np.array([[s^(op//2) if len(spaces)==2 else 0 for s in range(len(spaces))]
                     for op in range(4)],dtype=np.int64)
    rows = []
    for op in range(4):
        row = []
        for s,q in enumerate(labels):
            target = dest[op,s]
            action = ChiAOBlochSpaceAction(ambient,ambient,basis,system,
                _operation(translation=[.5*(op%2),0.,0.]),mesh,q%mesh_size,bool(op//2),*_controls())
            row.append(subspace_probe(action,panels[s],panels[target],metrics[s],metrics[target],
                source_subspace=spaces[s],target_subspace=spaces[target],contract=contract))
        rows.append(tuple(row))
    rows = tuple(rows)
    result = group_transport_probe(g,spaces,dest,rows,characters=characters,contract=contract)
    assert result.passed_probe and not result.production_reduction_authorized
    assert result.composition_residual < 2e-12
    selected_panels = tuple(np.eye(t.mixing.shape[0],dtype=complex)*np.exp(.19j*(s+1))
                            for s,t in enumerate(rows[g.identity]))
    selected = selected_group_probe(result,selected_panels)
    assert selected.passed_probe and selected.group_transport.characters == characters
    kinetic = core.compute_kinetic_lattice(basis,system,options)
    kinetic_blocks = [np.asarray(b,dtype=float) for b in kinetic.blocks]
    operators, parent_operators = [], []
    for q,c,selection in zip(labels,panels,selected_panels):
        ao = _bloch_sum_blocks(kinetic_blocks,kinetic.cells,
            np.array([2*np.pi*q/(8*mesh_size),0.,0.]))
        coefficients = c @ selection
        operators.append(np.ascontiguousarray(coefficients.conj().T @ ao @ coefficients))
        parent_operators.append(np.ascontiguousarray(c.conj().T @ ao @ c))
    operator_result = group_operator_probe(selected,tuple(operators),
        contract=replace(contract,operator="direct Bloch kinetic integrals"))
    assert operator_result.passed_probe and operator_result.parent is selected
    reducing = selected_operator_probe(group_operator_probe(result,tuple(parent_operators),
        contract=operator_result.probes[0][0].contract),selected)
    assert reducing.passed_probe
    broken_operators = tuple(a.copy() for a in operators)
    broken_operators[0][0,0] += .4
    assert not group_operator_probe(selected,broken_operators).passed_probe
    if mesh_size == 2:
        # At the zone boundary the two half-translation eigenvalues are
        # interchanged by scalar TR. A single line cannot retain both.
        cut = selected_group_probe(result,(np.array([[1],[0]],dtype=complex),))
        assert not cut.passed_probe
    # Discarding integer images gives an ordinary quotient with the same
    # table and grading, but it cannot describe these retained Bloch actions.
    quotient = group([[g.product(a,b) for b in range(4)] for a in range(4)],
                     anti=np.array([0,0,1,1],dtype=np.uint8))
    wrong = group_transport_probe(quotient,spaces,dest,rows,contract=contract)
    assert wrong.local_probes_passed and not wrong.passed_probe
    assert wrong.composition_residual > 1.


def test_molecular_retained_group_uses_noncommuting_operations_and_physical_orbitals():
    from itertools import permutations
    from scipy.linalg import eigh
    from vibeqc.symmetry_ao import build_molecular_space_action, molecular_point_group
    mol = vq.Molecule([vq.Atom(2,[2.,0.,0.]),vq.Atom(2,[0.,2.,0.]),vq.Atom(2,[0.,0.,2.])])
    basis = vq.BasisSet(mol,"sto-3g")
    rotations = np.ascontiguousarray([np.eye(3)[list(p)] for p in permutations(range(3))])
    g = molecular_point_group(mol,rotations,np.zeros((6,3)),identity_label="three-site permutations",budget=BUDGET)
    assert any(g.product(a,b)!=g.product(b,a) for a in range(6) for b in range(6))
    metric = np.ascontiguousarray(vq.compute_overlap(basis),dtype=complex)
    h = np.asarray(vq.compute_kinetic(basis))+np.asarray(vq.compute_nuclear(basis,mol))
    energies,orbitals = eigh(h,metric)
    assert abs(energies[1]-energies[2]) < 1e-12
    gauge,_ = np.linalg.qr(panel(2,2))
    c = np.ascontiguousarray(orbitals[:,1:] @ gauge)
    ambient = SpaceIdentity("three He at Cartesian 2-bohr axis points", "sto-3g", "AO", "native order")
    contract = OperatorContract("overlap and core Hamiltonian", "direct molecular integrals",
        "full", "none", "molecular")
    space = replace(ambient,space="degenerate core eigenspace",gauge="complex column gauge")
    rows = []
    actions = []
    for rotation in rotations:
        action = build_molecular_space_action(mol,basis,rotation.copy(),np.zeros(3),
            source=ambient,target=ambient,budget=BUDGET)
        actions.append(action)
        rows.append((subspace_probe(action,c,c,metric,metric,source_subspace=space,
                                    target_subspace=space,contract=contract),))
    dest = np.zeros((6,1),dtype=np.int64)
    result = group_transport_probe(g,(space,),dest,tuple(rows),contract=contract)
    assert result.passed_probe
    selected = selected_group_probe(result,(np.ascontiguousarray(gauge),))
    assert selected.passed_probe
    selected_c = c @ gauge
    selected_h = np.ascontiguousarray(selected_c.conj().T @ h @ selected_c)
    operator_result = group_operator_probe(selected,(selected_h,),contract=contract)
    assert operator_result.passed_probe and operator_result.parent is selected
    reducing = selected_operator_probe(group_operator_probe(result,
        (np.ascontiguousarray(c.conj().T @ h @ c),),contract=contract),selected)
    assert reducing.passed_probe
    # Exercise an actual physical rank reduction: the three-dimensional
    # orthonormal AO span onto the two-dimensional core eigenspace.
    full_c = np.ascontiguousarray(orbitals,dtype=complex)
    full_space = replace(space,space="full orthonormal AO span",gauge="core eigenvectors")
    full_rows = tuple((subspace_probe(action,full_c,full_c,metric,metric,
        source_subspace=full_space,target_subspace=full_space,contract=contract),) for action in actions)
    full_group = group_transport_probe(g,(full_space,),dest,full_rows,contract=contract)
    full_operators = group_operator_probe(full_group,
        (np.ascontiguousarray(full_c.conj().T @ h @ full_c),),contract=contract)
    eigen_selection = selected_group_probe(full_group,
        (np.ascontiguousarray(full_c.conj().T @ metric @ c),))
    rank_reduced = selected_operator_probe(full_operators,eigen_selection)
    assert rank_reduced.passed_probe
    assert rank_reduced.operator_transport.operators[0].shape == (2,2)
    assert rank_reduced.containment[0].residual < 1e-12
    assert rank_reduced.adjoint_containment[0].residual < 1e-12
    broken_h = selected_h.copy()
    broken_h[0,0] += .4
    assert not group_operator_probe(selected,(broken_h,)).passed_probe
    # Cutting one column from the physical two-dimensional degenerate
    # eigenspace breaks the noncommuting molecular action.
    cut = selected_group_probe(result,(np.array([[1],[0]],dtype=complex),))
    assert not cut.passed_probe
    assert all(t.metric.passed_probe for row in cut.group_transport.transports for t in row)
    # An inconsistent operation phase preserves each individual subspace.
    class RephasedAction:
        source = target = ambient
        antiunitary = False
        dimension = basis.nbasis
        def apply(self, panel): return 1j*actions[1].apply(panel)
    rows[1] = (subspace_probe(RephasedAction(),c,c,metric,metric,source_subspace=space,
                              target_subspace=space,contract=contract),)
    negative = group_transport_probe(g,(space,),dest,tuple(rows),contract=contract)
    assert negative.local_probes_passed and not negative.passed_probe


def periodic_state_probe(case, *, subspace="correlated_occupied", **kwargs):
    from tests.test_periodic_orbital_sewing import _controls
    from vibeqc.symmetry_shared import audit_periodic_state_subspace
    system,basis,op,state,source,tr,*_ = case
    options,inventory,caps = _controls(full=state.n_basis == state.n_effective_orbitals)
    controls = dict(subspace=subspace,sewing_options=options,sewing_inventory=inventory,
        sewing_caps=caps,metric_tolerance=1e-9,leakage_tolerance=1e-9,
        probe_identity="native-state bridge",budget=BUDGET)
    controls.update(kwargs)
    return audit_periodic_state_subspace(state,basis,system,op,source,tr,**controls)


@pytest.mark.parametrize("tr", [False,True])
@pytest.mark.parametrize("shift", [(0,0,0),(1,0,0),(1,1,1)])
@pytest.mark.parametrize("subspace", ["frozen_core","correlated_occupied","virtual"])
def test_periodic_state_bridge_uses_actual_masks_mesh_and_complex_gauges(tr,shift,subspace):
    from tests.test_periodic_orbital_sewing import _fixture
    case = _fixture(tr=tr,shift=shift,frozen=True)
    result = periodic_state_probe(case,subspace=subspace)
    assert result.state is case[3]
    assert result.passed_probe and not result.production_reduction_authorized
    assert result.source_index == case[4] and result.target_index == 2
    bands = {"frozen_core":(0,),"correlated_occupied":(1,),"virtual":(2,3)}[subspace]
    assert result.source_bands == bands and result.target_bands == bands
    np.testing.assert_allclose(result.transport.mixing,case[6][np.ix_(bands,bands)],atol=2e-12)
    mesh = core._RegularKMesh(case[3].mesh,case[3].is_shift)
    for index,character in ((result.source_index,result.source_character),
                            (result.target_index,result.target_character)):
        for image in ((1,0,0),(0,1,0),(0,0,1),(3,-2,1)):
            assert character.phase(image) == pytest.approx(
                np.exp(-2j*np.pi*np.dot(mesh.fractional_at(index),image)))
    assert result.transport.metric.contract.source_revision == case[3].state_identity_sha256
    with pytest.raises(ValueError): result.operation_rotation.setflags(write=True)
    with pytest.raises(ValueError): result.transport.mixing.setflags(write=True)
    with pytest.raises(TypeError,match="audit_periodic_state_subspace"):
        replace(result,target_index=0)


def test_periodic_state_bridge_retains_noncontiguous_target_band_masks():
    from tests.test_periodic_orbital_sewing import _fixture
    case = _fixture(frozen=True,target_mask_swap=True)
    result = periodic_state_probe(case,subspace="frozen_core")
    assert result.source_bands == (0,) and result.target_bands == (1,)
    assert result.passed_probe


@pytest.mark.parametrize("change", ["source_gauge","discarded_drift"])
def test_periodic_state_bridge_names_change_with_actual_numerical_state(change):
    from tests.test_periodic_orbital_sewing import _fixture
    original = periodic_state_probe(_fixture(retained=3))
    changed = periodic_state_probe(_fixture(retained=3,**{change: True if change=="source_gauge" else 1.}))
    assert original.state.state_identity_sha256 != changed.state.state_identity_sha256
    assert original.transport.metric.source_space != changed.transport.metric.source_space
    assert original.transport.metric.contract != changed.transport.metric.contract


@pytest.mark.parametrize("change", ["basis", "geometry"])
def test_periodic_state_bridge_context_fingerprints_bind_supplied_geometry_and_radials(change):
    from tests.test_periodic_orbital_sewing import _fixture
    case = list(_fixture())  # Scalar TR: both contexts admit the same numerical state.
    original = periodic_state_probe(case)
    if change == "basis":
        shells = list(case[1].shells())
        changed_shells = [core.ShellInfo(s.atom_index,s.l,s.pure,
            [1.1*x for x in s.exponents],list(s.coefficients),list(s.origin)) for s in shells]
        case[1] = core.BasisSet(case[0].unit_cell_molecule(),changed_shells,"changed-radial",False)
    else:
        case[0].unit_cell = [core.Atom(6,[0,0,0])]
    changed = periodic_state_probe(case)
    assert original.state is changed.state
    a,b = original.transport.metric.source_space,changed.transport.metric.source_space
    assert (a.basis != b.basis) if change=="basis" else (a.geometry != b.geometry)
    assert original.passed_probe and changed.passed_probe
    # Context binding is not authentication of the physical source of F/S/C.
    assert not changed.sewing.physical_source_symmetry_certified


@pytest.mark.parametrize("fault,match", [
    ("mix_core","cross-subspace"), ("energy_drift","Roothaan|energy"),
    ("broken_subspace","cross-subspace|Roothaan"),
])
def test_periodic_state_bridge_preserves_native_mask_and_stationarity_gates(fault,match):
    from tests.test_periodic_orbital_sewing import _fixture
    case = _fixture(frozen=True,**{fault: 0.1 if fault=="energy_drift" else True})
    with pytest.raises(ValueError,match=match): periodic_state_probe(case)


@pytest.mark.parametrize("fault", ["empty", "unknown", "bytes", "work", "aggregate_bytes", "aggregate_work"])
def test_periodic_state_bridge_refuses_invalid_selections_and_budgets(fault):
    from tests.test_periodic_orbital_sewing import _fixture
    case = _fixture()
    extra = {}
    exception = ValueError
    if fault == "empty": extra['subspace'] = 'frozen_core'
    if fault == "unknown": extra['subspace'] = 'all orbitals'
    if fault == "bytes": extra['budget'] = Budget(1,10**9); exception = MemoryError
    if fault == "work": extra['budget'] = Budget(64 << 20,1)
    n,r = case[3].n_basis,case[3].n_effective_orbitals
    if fault == "aggregate_bytes":
        extra['budget'] = Budget(65536+case[3].resident_bytes+1024*n*n+1024*n*r+2048*r*r,10**9)
        exception = MemoryError
    if fault == "aggregate_work":
        extra['budget'] = Budget(64 << 20,256*n*n*r+512*n*r*r+64*n*n+128*n*r)
    with pytest.raises(exception): periodic_state_probe(case,**extra)


def test_periodic_state_bridge_cannot_promote_retained_state_to_full_ao_scope():
    from tests.test_periodic_orbital_sewing import _fixture, _controls
    with pytest.raises(ValueError,match="full-AO scope"):
        periodic_state_probe(_fixture(retained=3),sewing_options=_controls(full=True)[0])


def test_periodic_state_bridge_refuses_observable_context_mutation(monkeypatch):
    from tests.test_periodic_orbital_sewing import _fixture
    import vibeqc.symmetry_shared as shared
    case = _fixture()
    original = shared.audit_subspace_panels
    def mutate(*args,**kwargs):
        result = original(*args,**kwargs)
        case[0].unit_cell = [core.Atom(6,[0,0,0])]
        return result
    monkeypatch.setattr(shared,"audit_subspace_panels",mutate)
    with pytest.raises(ValueError,match="context changed"):
        periodic_state_probe(case)


def test_periodic_state_bridge_detaches_operation_and_state_matrix_copies():
    from tests.test_periodic_orbital_sewing import _fixture
    case = _fixture()
    result = periodic_state_probe(case)
    matrix = result.transport.mixing.copy()
    before = result.operation_translation.copy()
    with pytest.raises(AttributeError): case[2].translation = np.ones(3)
    case[3].coefficients(case[4])[:] = 0
    np.testing.assert_array_equal(result.operation_translation,before)
    np.testing.assert_array_equal(result.transport.mixing,matrix)
    assert result.passed_probe


def test_periodic_state_bridge_keeps_stricter_shared_probe_failures():
    from tests.test_periodic_orbital_sewing import _fixture
    result = periodic_state_probe(_fixture(),leakage_tolerance=0.)
    assert result.sewing.full_ao_scope_audited
    assert result.transport.containment.residual > 0.
    assert not result.passed_probe


@pytest.fixture(scope="module",params=[False,True])
def gamma_inventory_state(request):
    return make_gamma_inventory_state(request.param)


def make_gamma_inventory_state(two_sites):
    """Physical molecular S/H at Gamma, not a periodic SCF/source capture."""
    from scipy.linalg import eigh
    from tests.test_periodic_ao_bloch_transport import _basis
    atoms = [core.Atom(1,[0,0,0]),core.Atom(1,[4,0,0])] if two_sites else [core.Atom(2,[0,0,0])]
    system = core.PeriodicSystem(3,np.eye(3)*8,atoms)
    if two_sites:
        basis = _basis(system,(0,))
    else:
        shells = [core.ShellInfo(0,0,True,[a],[1.],[0,0,0]) for a in (.8,.3)]
        basis = core.BasisSet(system.unit_cell_molecule(),shells,"two s radial channels",False)
    s = np.asarray(vq.compute_overlap(basis),dtype=complex)
    h = np.asarray(vq.compute_kinetic(basis)+vq.compute_nuclear(basis,system.unit_cell_molecule()),dtype=complex)
    e,c = eigh(h,s)
    c = c*np.exp(1j*np.array([.37,.61]))
    data = core._PeriodicRestrictedMeanFieldInput()
    data.calculation_identity = 'd'*64
    data.periodic_dimension = 3
    data.mesh = (1,1,1)
    data.is_shift = (0,0,0)
    data.reciprocal_lattice = system.reciprocal_lattice()
    data.converged = True
    data.n_basis = data.n_effective_orbitals = 2
    data.electrons_per_cell = 2
    data.reference_energy_per_cell = 2*e[0]
    data.minimum_band_gap_hartree = 1e-5
    data.add_kpoint(k_cartesian=np.zeros(3),weight=1.,overlap=s,fock=h,coefficients=c,
        orbital_energies=e,occupations=[2.,0.],frozen_core_mask=[0,0],
        correlated_occupied_mask=[1,0],virtual_mask=[0,1])
    return system,basis,core._make_periodic_restricted_mean_field_state(data),two_sites


def inventory_bridges(fixture, rotations, translations=None, anti=None, subspace="correlated_occupied"):
    from tests.test_periodic_ao_bloch_transport import _operation
    system,basis,state,*_ = fixture
    translations = np.zeros((len(rotations),3)) if translations is None else translations
    anti = [False]*len(rotations) if anti is None else anti
    return tuple(tuple(periodic_state_probe((system,basis,_operation(rotation,translation),state,k,bool(tr)),
                       subspace=subspace) for k in range(state.n_kpoints))
                 for rotation,translation,tr in zip(rotations,translations,anti))


def inventory_probe(g,bridges,**kwargs):
    from vibeqc.symmetry_shared import audit_periodic_state_group
    options = dict(seitz_tolerance=1e-10,composition_tolerance=1e-9,
                   probe_identity="actual periodic operation inventory",budget=BUDGET)
    options.update(kwargs)
    return audit_periodic_state_group(g,bridges,**options)


def test_periodic_inventory_detects_wrong_table_hidden_by_gamma_panels(gamma_inventory_state):
    rotations = np.array([np.eye(3),np.diag([1,-1,-1]),np.diag([-1,1,-1]),np.diag([-1,-1,1])],dtype=np.int64)
    bridges = inventory_bridges(gamma_inventory_state,rotations)
    correct = group([[g^h for h in range(4)] for g in range(4)])
    result = inventory_probe(correct,bridges)
    assert result.passed_probe and result.state is gamma_inventory_state[2]
    assert not result.production_reduction_authorized
    wrong = group([[(g+h)%4 for h in range(4)] for g in range(4)])
    rows = tuple(tuple(b.transport for b in row) for row in bridges)
    blind = group_transport_probe(wrong,(rows[0][0].metric.source_space,),np.zeros((4,1),dtype=np.int64),
                                  rows,contract=rows[0][0].metric.contract)
    assert blind.passed_probe  # Every s-AO action is identity at Gamma.
    with pytest.raises(ValueError,match="rotation product"):
        inventory_probe(wrong,bridges)
    with pytest.raises(TypeError,match="audit_periodic_state_group"):
        replace(result,seitz_tolerance=1.)


@pytest.mark.parametrize("translation", [0.,1.])
def test_periodic_inventory_refuses_duplicate_cosets_hidden_at_gamma(gamma_inventory_state,translation):
    bridges = inventory_bridges(gamma_inventory_state,[np.eye(3)]*2,[[0,0,0],[translation,0,0]])
    with pytest.raises(ValueError,match="duplicate periodic Seitz coset"):
        inventory_probe(group([[0,1],[1,0]]),bridges)


def test_periodic_inventory_allows_identical_spatial_parts_with_different_grading(gamma_inventory_state):
    bridges = inventory_bridges(gamma_inventory_state,[np.eye(3)]*2,anti=[False,True])
    result = inventory_probe(group([[0,1],[1,0]],anti=np.array([0,1],dtype=np.uint8)),bridges)
    assert result.passed_probe
    assert result.maximum_fractional_seitz_residual == 0.


@pytest.mark.parametrize("image", [0,2**62+1])
@pytest.mark.parametrize("gamma_inventory_state", [True], indirect=True)
def test_periodic_inventory_checks_full_images_even_at_gamma(gamma_inventory_state,image):
    bridges = inventory_bridges(gamma_inventory_state,[np.eye(3)]*2,[[0,0,0],[.5,0,0]])
    images = np.zeros((2,2,3),dtype=np.int64)
    images[1,1,0] = image
    g = group([[0,1],[1,0]],rotations=np.tile(np.eye(3,dtype=np.int64),(2,1,1)),cocycle=images)
    rows = tuple(tuple(item.transport for item in row) for row in bridges)
    blind = group_transport_probe(g,(rows[0][0].metric.source_space,),np.zeros((2,1),dtype=np.int64),
        rows,contract=rows[0][0].metric.contract,characters=(bridges[0][0].source_character,))
    assert blind.passed_probe  # Every lattice character is one at Gamma.
    with pytest.raises(ValueError,match="full lattice cocycle"):
        inventory_probe(g,bridges)
    images[1,1,0] = 1
    exact = group([[0,1],[1,0]],rotations=np.tile(np.eye(3,dtype=np.int64),(2,1,1)),cocycle=images)
    assert inventory_probe(exact,bridges).passed_probe


@pytest.mark.parametrize("fault", ["type","shape","owner","mask","context","grading","byte","work","aggregate_byte","aggregate_work"])
def test_periodic_inventory_refuses_inconsistent_proofs_and_budget(gamma_inventory_state,fault):
    g = group([[0,1],[1,0]],anti=np.array([0,1],dtype=np.uint8))
    bridges = inventory_bridges(gamma_inventory_state,[np.eye(3)]*2,anti=[False,True])
    extra = {}
    error = ValueError
    if fault == "type": bridges=(bridges[0],(object(),)); error=TypeError
    if fault == "shape": bridges=(bridges[0],())
    if fault == "owner":
        # A distinct immutable owner, even with the same numerical state, is
        # refused so retained-state storage cannot multiply behind admission.
        other = make_gamma_inventory_state(gamma_inventory_state[3])
        assert other[2] is not gamma_inventory_state[2]
        assert other[2].state_identity_sha256 == gamma_inventory_state[2].state_identity_sha256
        bridges=(bridges[0],inventory_bridges(other,[np.eye(3)],anti=[True])[0])
    if fault == "mask":
        other = inventory_bridges(gamma_inventory_state,[np.eye(3)],anti=[True],subspace="virtual")
        bridges=(bridges[0],other[0])
    if fault == "context":
        system,basis,state,two_sites = gamma_inventory_state
        other_system = core.PeriodicSystem(3,np.asarray(system.lattice),list(system.unit_cell))
        other_system.charge = 1
        other = inventory_bridges((other_system,basis,state),[np.eye(3)],anti=[True])
        bridges=(bridges[0],other[0])
    if fault == "grading": g=group([[0,1],[1,0]])
    if fault == "byte": extra['budget']=Budget(1,10**10); error=MemoryError
    if fault == "work": extra['budget']=Budget(64 << 20,1)
    if fault in ("aggregate_byte","aggregate_work"):
        admitted=inventory_probe(g,bridges)
        if fault == "aggregate_byte":
            extra['budget']=Budget(admitted.admitted_bytes-1,admitted.admitted_work); error=MemoryError
        else: extra['budget']=Budget(admitted.admitted_bytes,admitted.admitted_work-1)
    with pytest.raises(error): inventory_probe(g,bridges,**extra)


def test_periodic_inventory_accepts_nonfirst_identity_and_noncommuting_rotations():
    from itertools import permutations
    fixture = make_gamma_inventory_state(False)
    rotations = np.ascontiguousarray([np.eye(3,dtype=np.int64)[list(p)] for p in permutations(range(3))])
    rotations = np.roll(rotations,3,axis=0)
    g = FiniteGroup.from_integer_rotations(rotations,identity_label="noncommuting scalar-orbital example",budget=BUDGET)
    assert g.identity == 3 and any(g.product(a,b)!=g.product(b,a) for a in range(6) for b in range(6))
    result = inventory_probe(g,inventory_bridges(fixture,rotations))
    assert result.passed_probe
    assert result.group_transport.group.identity == g.identity


def test_periodic_inventory_preserves_small_seitz_residual_and_strict_threshold():
    fixture = make_gamma_inventory_state(False)
    bridges = inventory_bridges(fixture,[np.eye(3)],[[1e-12,0,0]])
    result = inventory_probe(group([[0]]),bridges,seitz_tolerance=2e-12)
    assert result.maximum_fractional_seitz_residual == pytest.approx(1e-12,abs=0,rel=1e-14)
    with pytest.raises(ValueError,match="full lattice cocycle"):
        inventory_probe(group([[0]]),bridges,seitz_tolerance=0.)


def test_periodic_inventory_preserves_stricter_failing_local_probe():
    fixture = make_gamma_inventory_state(False)
    from tests.test_periodic_ao_bloch_transport import _operation
    bridge = periodic_state_probe((fixture[0],fixture[1],_operation(),fixture[2],0,True),leakage_tolerance=0.)
    assert not bridge.passed_probe
    identity = inventory_bridges(fixture,[np.eye(3)])[0]
    result = inventory_probe(group([[0,1],[1,0]],anti=np.array([0,1],dtype=np.uint8)),(identity,(bridge,)))
    assert not result.passed_probe and not result.production_reduction_authorized


@pytest.mark.parametrize("value", [True,-1,float('nan'),float('inf'),1e-3,"1e-10"])
def test_periodic_inventory_refuses_invalid_seitz_tolerance(value):
    with pytest.raises((ValueError,TypeError),match="seitz_tolerance"):
        inventory_probe(group([[0]]),(),seitz_tolerance=value)


def selected_group_probe(parent, selections, **kwargs):
    options = dict(selection_identity="caller-selected test columns",metric_tolerance=1e-11,
        leakage_tolerance=1e-11,composition_tolerance=1e-11,
        probe_identity="selected group witness",budget=BUDGET)
    options.update(kwargs)
    return audit_selected_group_transport(parent,selections,**options)


def selection_parent(*, anti=False):
    g = group([[0,1],[1,0]],anti=np.array([0,anti],dtype=np.uint8))
    spaces = (SPACE,replace(SPACE,gauge="other parent coordinates"))
    m = np.array([[1,1j],[1j,1]],dtype=complex)/np.sqrt(2)
    reverse = m.T if anti else m.conj().T
    rows = ((mixing_witness(np.eye(2),spaces[0],spaces[0]),
             mixing_witness(np.eye(2),spaces[1],spaces[1])),
            (mixing_witness(m,spaces[0],spaces[1],anti=anti),
             mixing_witness(reverse,spaces[1],spaces[0],anti=anti)))
    parent = group_transport_probe(g,spaces,np.array([[0,1],[1,0]],dtype=np.int64),rows)
    a = np.array([[1],[2j]],dtype=complex)/np.sqrt(5)
    b = np.ascontiguousarray(m @ (a.conj() if anti else a)*np.exp(.31j))
    return parent,(a,b)


@pytest.mark.parametrize("anti", [False,True])
def test_selected_group_tracks_dense_target_gauges_and_conjugation(anti):
    parent,panels = selection_parent(anti=anti)
    result = selected_group_probe(parent,panels)
    assert parent.passed_probe and result.passed_probe
    assert result.parent is parent and not result.production_reduction_authorized
    assert result.group_transport.contract is parent.contract
    np.testing.assert_array_equal(result.group_transport.destinations,parent.destinations)
    expected = panels[1].conj().T @ parent.transports[1][0].mixing @ (panels[0].conj() if anti else panels[0])
    np.testing.assert_allclose(result.group_transport.transports[1][0].mixing,expected,atol=1e-14)
    wrong = selected_group_probe(parent,(panels[0],np.ascontiguousarray(panels[1][::-1].conj()*[[1],[-1]])))
    assert not wrong.passed_probe
    assert wrong.group_transport.transports[1][0].metric.passed_probe
    assert wrong.group_transport.transports[1][0].containment.residual == pytest.approx(1.)


def test_selected_group_snapshots_bind_actual_columns_parent_gauge_and_policy():
    parent,panels = selection_parent()
    result = selected_group_probe(parent,panels)
    original = tuple(x.copy() for x in panels)
    changed_gauge = selected_group_probe(parent,tuple(np.ascontiguousarray(x*1j) for x in panels))
    changed_policy = selected_group_probe(parent,panels,selection_identity="different declared threshold policy")
    renamed = tuple(replace(s,gauge=s.gauge+" renamed") for s in parent.spaces)
    rows = tuple(tuple(mixing_witness(t.mixing,renamed[s],renamed[int(parent.destinations[g,s])],
        anti=t.antiunitary) for s,t in enumerate(row)) for g,row in enumerate(parent.transports))
    renamed_parent = group_transport_probe(parent.group,renamed,parent.destinations,rows)
    changed_parent = selected_group_probe(renamed_parent,panels)
    assert result.group_transport.spaces != changed_gauge.group_transport.spaces
    assert result.group_transport.spaces != changed_policy.group_transport.spaces
    assert result.group_transport.spaces != changed_parent.group_transport.spaces
    for x,y,z in zip(panels,result.selections,original):
        x[:] = 0
        np.testing.assert_array_equal(y,z)
        with pytest.raises(ValueError): y.setflags(write=True)
    with pytest.raises(TypeError,match="audit_selected_group_transport"):
        replace(result,selection_identity="forged")
    nested = selected_group_probe(result,tuple(np.ones((1,1),dtype=complex) for _ in panels))
    assert nested.parent is result and nested.passed_probe


@pytest.mark.parametrize("failure", ["local", "composition"])
def test_selected_group_cannot_erase_failed_parent_by_discarding_bad_block(failure):
    g = group([[0,1],[1,0]])
    m = np.array([[1,2],[0,-1]],dtype=complex) if failure=="local" else np.diag([1.,1j])
    parent = group_transport_probe(g,(SPACE,),np.zeros((2,1),dtype=np.int64),
        ((mixing_witness(np.eye(2)),),(mixing_witness(m),)))
    assert not parent.passed_probe
    result = selected_group_probe(parent,(np.array([[1],[0]],dtype=complex),))
    assert result.group_transport.passed_probe and not result.passed_probe
    nested = selected_group_probe(result,(np.ones((1,1),dtype=complex),))
    assert nested.group_transport.passed_probe and not nested.passed_probe


def test_selected_group_allows_distinct_ranks_in_disjoint_orbits():
    g = group([[0,1],[1,0]])
    spaces = (SPACE,replace(SPACE,space="second orbit"))
    row = tuple(mixing_witness(np.eye(r),s,s) for r,s in zip((2,3),spaces))
    parent = group_transport_probe(g,spaces,np.array([[0,1],[0,1]],dtype=np.int64),(row,row))
    result = selected_group_probe(parent,(np.eye(2,dtype=complex)[:,:1].copy(),np.eye(3,dtype=complex)[:,:2].copy()))
    assert result.passed_probe
    assert tuple(t.mixing.shape for t in result.group_transport.transports[0]) == ((1,1),(2,2))


@pytest.mark.parametrize("fault", ["parent","budget_type","list","count","dtype","shape","zero_rank",
    "rank_orbit","nonfinite","nonorthogonal","policy","probe","bytes","work","aggregate_bytes","aggregate_work"])
def test_selected_group_refuses_invalid_selections_and_admission(fault):
    parent,panels = selection_parent()
    options = {}
    if fault == "parent": parent = parent.group
    if fault == "budget_type": options['budget'] = None
    if fault == "list": panels = list(panels)
    if fault == "count": panels = panels[:1]
    if fault == "dtype": panels = (panels[0].real.copy(),panels[1])
    if fault == "shape": panels = (np.ones((3,1),dtype=complex),panels[1])
    if fault == "zero_rank": panels = (np.empty((2,0),dtype=complex),panels[1])
    if fault == "rank_orbit": panels = (np.eye(2,dtype=complex),panels[1])
    if fault == "nonfinite": panels[0][0,0] = np.nan
    if fault == "nonorthogonal": panels[0][:] *= 2
    if fault == "policy": options['selection_identity'] = " "
    if fault == "probe": options['probe_identity'] = " "
    if fault == "bytes": options['budget'] = Budget(1,10**9)
    if fault == "work": options['budget'] = Budget(64 << 20,1)
    if fault.startswith("aggregate"):
        result = selected_group_probe(parent,panels)
        options['budget'] = (Budget(result.admitted_bytes-1,10**9) if fault=="aggregate_bytes"
                             else Budget(64 << 20,result.admitted_work-1))
    with pytest.raises((ValueError,TypeError,MemoryError)):
        selected_group_probe(parent,panels,**options)


@pytest.mark.parametrize("field", ["metric_tolerance","leakage_tolerance","composition_tolerance"])
@pytest.mark.parametrize("value", [True,-1,float('nan'),float('inf'),"1e-10"])
def test_selected_group_refuses_invalid_tolerances(field,value):
    parent,panels = selection_parent()
    with pytest.raises((ValueError,TypeError)):
        selected_group_probe(parent,panels,**{field:value})


def test_selected_group_admits_before_scanning_payload(monkeypatch):
    import vibeqc.symmetry_shared as shared
    parent,panels = selection_parent()
    admitted = selected_group_probe(parent,panels)
    def forbidden(*args,**kwargs): raise AssertionError("payload scanned before admission")
    # Tolerance checks use isfinite too; allow scalar checks only.
    original = shared.np.isfinite
    monkeypatch.setattr(shared.np,"isfinite",lambda x: original(x) if np.ndim(x)==0 else forbidden())
    with pytest.raises(MemoryError):
        selected_group_probe(parent,panels,budget=Budget(admitted.admitted_bytes-1,10**9))


def group_operator_probe(parent, operators, **kwargs):
    options = dict(contract=replace(CONTRACT,operator="independent retained operator"),
        tolerance=1e-11,probe_identity="static operator family",budget=BUDGET)
    options.update(kwargs)
    return audit_group_operators(parent,operators,**options)


@pytest.mark.parametrize("anti", [False,True])
@pytest.mark.parametrize("hermitian", [False,True])
def test_group_operators_follow_dense_gauges_and_antiunitary_conjugation(anti,hermitian):
    parent,_ = selection_parent(anti=anti)
    a = np.array([[1,.2+.7j],[.4-.3j,3]],dtype=complex)
    if hermitian: a = np.ascontiguousarray(a+a.conj().T)
    m = parent.transports[1][0].mixing
    b = np.ascontiguousarray(m @ (a.conj() if anti else a) @ m.conj().T)
    result = group_operator_probe(parent,(a,b))
    assert result.passed_probe and not result.production_reduction_authorized
    assert result.parent is parent and len(result.probes)==2
    for g,row in enumerate(result.probes):
        for s,p in enumerate(row):
            assert p.contract != parent.contract
            assert p.source_space == parent.spaces[s]
            assert p.target_space == parent.spaces[parent.destinations[g,s]]
            assert p.kind is EvidenceKind.NUMERICAL
            assert p.residual < 1e-12
    # A real-only fixture would hide this missing or extra conjugation.
    wrong = np.ascontiguousarray(m @ (a if anti else a.conj()) @ m.conj().T)
    rejected = group_operator_probe(parent,(a,wrong))
    assert not rejected.passed_probe and rejected.probes[1][0].residual > .1


def test_group_operators_preserve_snapshots_and_absolute_tolerance():
    parent,_ = selection_parent()
    a = np.eye(2,dtype=complex)
    b = a.copy()*1.25
    result = group_operator_probe(parent,(a,b),tolerance=.4)
    assert result.passed_probe
    assert result.probes[1][0].residual == pytest.approx(np.sqrt(2)*.25)
    assert not group_operator_probe(parent,(a,b),tolerance=.3).passed_probe
    for original,snapshot in zip((a,b),result.operators):
        saved = original.copy()
        original[:] = 0
        np.testing.assert_array_equal(snapshot,saved)
        with pytest.raises(ValueError): snapshot.setflags(write=True)
    with pytest.raises(TypeError,match="audit_group_operators"):
        replace(result,parent=parent)


@pytest.mark.parametrize("failure", ["local","composition"])
def test_group_operators_cannot_erase_parent_failure_with_scalar_operator(failure):
    g = group([[0,1],[1,0]])
    m = np.array([[1,2],[0,-1]],dtype=complex) if failure=="local" else np.diag([1.,1j])
    parent = group_transport_probe(g,(SPACE,),np.zeros((2,1),dtype=np.int64),
        ((mixing_witness(np.eye(2)),),(mixing_witness(m),)))
    for supplied in (parent,selected_group_probe(parent,(np.array([[1],[0]],dtype=complex),))):
        rank = 2 if supplied is parent else 1
        result = group_operator_probe(supplied,(np.eye(rank,dtype=complex),))
        assert all(p.passed_probe for row in result.probes for p in row)
        assert result.parent is supplied and not result.passed_probe


def test_group_operators_allow_distinct_ranks_in_disjoint_orbits():
    g = group([[0,1],[1,0]])
    spaces = (SPACE,replace(SPACE,space="second orbit"))
    row = tuple(mixing_witness(np.eye(r),s,s) for r,s in zip((2,3),spaces))
    parent = group_transport_probe(g,spaces,np.array([[0,1],[0,1]],dtype=np.int64),(row,row))
    result = group_operator_probe(parent,(np.eye(2,dtype=complex),panel(3,3)))
    assert result.passed_probe


def test_group_operators_projected_pass_does_not_claim_discarded_space_invariance():
    g = group([[0,1],[1,0]])
    parent = group_transport_probe(g,(SPACE,),np.zeros((2,1),dtype=np.int64),
        ((mixing_witness(np.eye(2)),),(mixing_witness(np.diag([1.,-1.])),)))
    selected = selected_group_probe(parent,(np.array([[1],[0]],dtype=complex),))
    assert selected.passed_probe
    # This operator couples the two irreducible spaces, although its
    # compression onto the selected line is zero and hence covariant.
    a = np.array([[0,1],[1,0]],dtype=complex)
    assert not group_operator_probe(parent,(a,)).passed_probe
    q = selected.selections[0]
    compressed = group_operator_probe(selected,(np.ascontiguousarray(q.conj().T @ a @ q),))
    assert compressed.passed_probe and not compressed.production_reduction_authorized
    assert np.linalg.norm(a @ q-q @ compressed.operators[0]) == pytest.approx(1.)


@pytest.mark.parametrize("fault", ["parent","budget_type","contract","list","count","dtype","ndim",
    "shape","contiguous","nonfinite","probe","bytes","work","aggregate_bytes","aggregate_work"])
def test_group_operators_refuse_invalid_inventory_and_admission(fault):
    parent,_ = selection_parent()
    operators = (np.eye(2,dtype=complex),np.eye(2,dtype=complex))
    options = {}
    if fault == "parent": parent = parent.group
    if fault == "budget_type": options['budget'] = None
    if fault == "contract": options['contract'] = None
    if fault == "list": operators = list(operators)
    if fault == "count": operators = operators[:1]
    if fault == "dtype": operators = (operators[0].real.copy(),operators[1])
    if fault == "ndim": operators = (operators[0].ravel(),operators[1])
    if fault == "shape": operators = (np.eye(3,dtype=complex),operators[1])
    if fault == "contiguous": operators = (operators[0].T,operators[1])
    if fault == "nonfinite": operators[0][0,0] = np.nan
    if fault == "probe": options['probe_identity'] = " "
    if fault == "bytes": options['budget'] = Budget(1,10**9)
    if fault == "work": options['budget'] = Budget(64 << 20,1)
    if fault.startswith("aggregate"):
        result = group_operator_probe(parent,operators)
        options['budget'] = (Budget(result.admitted_bytes-1,10**9) if fault=="aggregate_bytes"
                             else Budget(64 << 20,result.admitted_work-1))
    with pytest.raises((ValueError,TypeError,MemoryError)):
        group_operator_probe(parent,operators,**options)


@pytest.mark.parametrize("value", [True,-1,float('nan'),float('inf'),"1e-10"])
def test_group_operators_refuse_invalid_tolerance(value):
    parent,_ = selection_parent()
    with pytest.raises((ValueError,TypeError)):
        group_operator_probe(parent,(np.eye(2,dtype=complex),)*2,tolerance=value)


def test_group_operators_admit_before_scanning_or_copying(monkeypatch):
    import vibeqc.symmetry_shared as shared
    parent,_ = selection_parent()
    operators = (np.eye(2,dtype=complex),)*2
    admitted = group_operator_probe(parent,operators)
    def forbidden(*args,**kwargs): raise AssertionError("payload accessed before admission")
    original = shared.np.isfinite
    monkeypatch.setattr(shared.np,"isfinite",lambda x: original(x) if np.ndim(x)==0 else forbidden())
    monkeypatch.setattr(shared,"_freeze",forbidden)
    for budget,error in ((Budget(admitted.admitted_bytes-1,10**9),MemoryError),
                         (Budget(64 << 20,admitted.admitted_work-1),ValueError)):
        with pytest.raises(error,match="budget exceeded"):
            group_operator_probe(parent,operators,budget=budget)


def test_group_operators_refuse_overflow_in_finite_payload_arithmetic():
    parent,_ = selection_parent()
    with pytest.raises(ValueError,match="nonfinite operator covariance residual"):
        group_operator_probe(parent,(np.full((2,2),1e308,dtype=complex),np.eye(2,dtype=complex)))


@pytest.mark.parametrize("anti", [False,True])
@pytest.mark.parametrize("scale", [1e-250,1e-200,1e-170,1.,1e170,1e200,1e308])
def test_group_operators_scaled_norm_preserves_absolute_tolerance(anti,scale):
    g = group([[0,1],[1,0]],anti=np.array([0,anti],dtype=np.uint8))
    parent = group_transport_probe(g,(SPACE,),np.zeros((2,1),dtype=np.int64),
        ((mixing_witness(np.eye(2)),),(mixing_witness([[0,1],[1,0]],anti=anti),)))
    a = np.diag([0.,scale]).astype(complex)
    # The swap commutator has exactly two entries of magnitude scale.
    # In particular, never use pytest.approx's default absolute tolerance
    # here: it would accept the underflowed zero for the tiny cases.
    expected = np.sqrt(2)*scale
    refused = group_operator_probe(parent,(a,),tolerance=.5*scale)
    accepted = group_operator_probe(parent,(a,),tolerance=1.5*scale)
    assert refused.probes[1][0].residual == pytest.approx(expected,rel=1e-14,abs=0.)
    assert accepted.probes[1][0].residual == pytest.approx(expected,rel=1e-14,abs=0.)
    assert not refused.passed_probe and accepted.passed_probe
    assert not accepted.production_reduction_authorized


@pytest.mark.parametrize("anti", [False,True])
@pytest.mark.parametrize("scale", [1e-320,1e-310])
@pytest.mark.parametrize("phase", [1.,.6+.8j])
def test_group_operators_preserve_subnormal_complex_residuals(anti,scale,phase):
    g = group([[0,1],[1,0]],anti=np.array([0,anti],dtype=np.uint8))
    parent = group_transport_probe(g,(SPACE,),np.zeros((2,1),dtype=np.int64),
        ((mixing_witness(np.eye(2)),),(mixing_witness([[0,1],[1,0]],anti=anti),)))
    z = scale*phase
    a = np.diag([0.,z]).astype(complex)
    # Use the actual rounded real components as the independent norm oracle.
    # At subnormal scales, one ulp is much larger than a relative 1e-14.
    expected = hypot(z.real,z.imag,z.real,z.imag)
    refused = group_operator_probe(parent,(a,),tolerance=.5*scale)
    accepted = group_operator_probe(parent,(a,),tolerance=1.5*scale)
    assert 0. < refused.probes[1][0].residual
    assert abs(refused.probes[1][0].residual-expected) <= np.spacing(expected)
    assert not refused.passed_probe and accepted.passed_probe
    assert accepted.parent is parent and not accepted.production_reduction_authorized


@pytest.mark.parametrize("scale,relative", [(1e-100,1e-200),(1.,1e-320),
                                            (1.,1e-200),(1e200,1e-200),(1e308,1e-308)])
@pytest.mark.parametrize("phase", [1.,.6+.8j])
@pytest.mark.parametrize("anti", [False,True])
def test_group_operators_small_residual_beside_large_commuting_part(scale,relative,phase,anti):
    g = group([[0,1],[1,0]],anti=np.array([0,anti],dtype=np.uint8))
    parent = group_transport_probe(g,(SPACE,),np.zeros((2,1),dtype=np.int64),
        ((mixing_witness(np.eye(2)),),(mixing_witness([[0,1],[1,0]],anti=anti),)))
    z = scale*relative*phase
    a = np.array([[scale,0.],[z,scale]],dtype=complex)
    expected = hypot(z.real,z.imag,z.real,z.imag)
    for factor in [.5,1.5]:
        result = group_operator_probe(parent,(a,),tolerance=factor*expected)
        residual = result.probes[1][0].residual
        assert residual > 0.
        assert residual == pytest.approx(expected,rel=1e-14,abs=np.spacing(expected))
        assert result.passed_probe is (factor > 1.)
        assert not result.production_reduction_authorized


def selected_operator_probe(parent, selection, **kwargs):
    options = dict(covariance_tolerance=1e-11,leakage_tolerance=1e-11,
        probe_identity="selected reducing operator subspace",budget=BUDGET)
    options.update(kwargs)
    return audit_selected_operators(parent,selection,**options)


def identity_operator_selection(a,q,**kwargs):
    parent = group_transport_probe(group([[0]]),(SPACE,),np.zeros((1,1),dtype=np.int64),
        ((mixing_witness(np.eye(a.shape[0])),),))
    operators = group_operator_probe(parent,(a,))
    selection = selected_group_probe(parent,(q,),**kwargs)
    return operators,selection


@pytest.mark.parametrize("anti", [False,True])
def test_selected_operators_follow_complex_gauges_and_inherit_contract(anti):
    parent,panels = selection_parent(anti=anti)
    q = panels[0]
    a = np.ascontiguousarray(5*np.eye(2)-3*q @ q.conj().T)
    m = parent.transports[1][0].mixing
    b = np.ascontiguousarray(m @ (a.conj() if anti else a) @ m.conj().T)
    operators = group_operator_probe(parent,(a,b))
    selection = selected_group_probe(parent,panels)
    result = selected_operator_probe(operators,selection)
    assert result.passed_probe and not result.production_reduction_authorized
    assert result.parent is operators and result.selection is selection
    assert result.operator_transport.parent is selection
    for s,(forward,adjoint) in enumerate(zip(result.containment,result.adjoint_containment)):
        assert forward.contract == operators.probes[0][s].contract == adjoint.contract
        assert forward.source_space == parent.spaces[s]
        assert forward.target_space == selection.group_transport.spaces[s]
        assert forward.residual < 1e-12 and adjoint.residual < 1e-12
        np.testing.assert_allclose(result.operator_transport.operators[s],[[2]],atol=1e-12)
        with pytest.raises(ValueError): result.operator_transport.operators[s].setflags(write=True)
    with pytest.raises(TypeError,match="audit_selected_operators"):
        replace(result,parent=operators)
    further = selected_group_probe(selection,tuple(np.ones((1,1),dtype=complex)*np.exp(.2j)
                                                   for _ in panels))
    nested = selected_operator_probe(result,further)
    assert nested.passed_probe and nested.parent is result


@pytest.mark.parametrize("scale", [1e-320,1e-310,1e-250,1.,1e250])
@pytest.mark.parametrize("phase", [1.,.6+.8j])
def test_selected_operators_expose_leakage_hidden_by_zero_compression(scale,phase):
    a = np.array([[0,scale*phase],[scale*phase,0]],dtype=complex)
    operators,selection = identity_operator_selection(a,np.array([[1],[0]],dtype=complex))
    assert operators.passed_probe and selection.passed_probe
    result = selected_operator_probe(operators,selection)
    assert result.operator_transport.passed_probe and not result.passed_probe
    assert result.containment[0].residual == pytest.approx(1.)
    assert result.adjoint_containment[0].residual == pytest.approx(1.)


@pytest.mark.parametrize("scale,relative", [(1e-100,1e-200),(1.,1e-320),
                                            (1.,1e-200),(1e200,1e-200),(1e308,1e-308)])
@pytest.mark.parametrize("phase", [1.,.6+.8j])
def test_selected_operators_preserve_tiny_relative_and_adjoint_leakage(scale,relative,phase):
    z = scale*relative*phase
    a = np.array([[scale,z.conjugate()],[z,0.]],dtype=complex)
    parent,selection = identity_operator_selection(a,np.array([[1.],[0.]],dtype=complex))
    expected = hypot(z.real/scale,z.imag/scale)
    for factor in [.5,1.5]:
        result = selected_operator_probe(parent,selection,leakage_tolerance=factor*expected)
        for evidence in result.containment+result.adjoint_containment:
            assert evidence.residual > 0.
            assert evidence.residual == pytest.approx(expected,rel=1e-14,abs=np.spacing(expected))
            assert evidence.passed_probe is (factor > 1.)
        assert result.passed_probe is (factor > 1.)
        assert result.parent is parent and not result.production_reduction_authorized


def test_selected_operator_norm_ratio_combines_exponents_before_rounding():
    tiny = np.nextafter(0.,1.)
    a = np.zeros((10,10),dtype=complex)
    a[0,0] = 4.
    a[1:,0] = a[0,1:] = tiny
    q = np.eye(10,dtype=complex)[:,:1].copy()
    parent,selection = identity_operator_selection(a,q)
    # sqrt(9*tiny**2)/4 = 0.75*tiny rounds to tiny. Dividing the
    # component scales first rounds tiny/4 to zero and loses the answer.
    for tolerance in [0.,tiny]:
        result = selected_operator_probe(parent,selection,leakage_tolerance=tolerance)
        assert result.containment[0].residual == tiny
        assert result.adjoint_containment[0].residual == tiny
        assert result.passed_probe is bool(tolerance > 0.)


def test_selected_operator_refuses_nonzero_unrepresentable_ratio():
    tiny = np.nextafter(0.,1.)
    a = np.array([[4.,tiny],[tiny,0.]],dtype=complex)
    parent,selection = identity_operator_selection(a,np.array([[1.],[0.]],dtype=complex))
    with pytest.raises(ValueError,match="nonfinite selected operator leakage"):
        selected_operator_probe(parent,selection,leakage_tolerance=0.)


def test_selected_operators_check_adjoint_coupling_for_nonhermitian_operator():
    a = np.array([[1,2],[0,3]],dtype=complex)
    operators,selection = identity_operator_selection(a,np.array([[1],[0]],dtype=complex))
    result = selected_operator_probe(operators,selection)
    assert result.operator_transport.passed_probe and result.containment[0].passed_probe
    assert result.adjoint_containment[0].residual == pytest.approx(2/np.sqrt(5))
    assert not result.passed_probe


def test_selected_operators_zero_action_and_finite_gram_error():
    q = np.array([[1.001],[0]],dtype=complex)
    for a,expected in ((np.zeros((2,2),dtype=complex),0),
                       (np.diag([2,3]).astype(complex),2)):
        operators,selection = identity_operator_selection(a,q,metric_tolerance=.01)
        result = selected_operator_probe(operators,selection)
        assert result.passed_probe
        # Q^dagger A Q would leave the admitted Gram error in the result.
        np.testing.assert_allclose(result.operator_transport.operators[0],[[expected]],atol=1e-14)
        assert result.containment[0].residual < 1e-14


def test_selected_operators_keep_leakage_failure_through_further_selection():
    a = np.array([[2,0,0],[0,1,1],[0,1,3]],dtype=complex)
    operators,selection = identity_operator_selection(a,np.eye(3,dtype=complex)[:,:2].copy())
    first = selected_operator_probe(operators,selection)
    assert first.operator_transport.passed_probe and not first.passed_probe
    second_selection = selected_group_probe(selection,(np.array([[1],[0]],dtype=complex),))
    second = selected_operator_probe(first,second_selection)
    assert second.parent is first and second.operator_transport.passed_probe
    assert all(p.passed_probe for p in second.containment+second.adjoint_containment)
    assert not second.passed_probe


@pytest.mark.parametrize("failure", ["operator","selection"])
def test_selected_operators_preserve_parent_covariance_and_selection_failures(failure):
    parent,_ = selection_parent()
    a = np.eye(2,dtype=complex)
    operators = group_operator_probe(parent,(a,a*(2 if failure=="operator" else 1)))
    q = np.array([[1],[0]],dtype=complex)
    other = q if failure=="selection" else np.ascontiguousarray(parent.transports[1][0].mixing @ q)
    selection = selected_group_probe(parent,(q,other))
    result = selected_operator_probe(operators,selection)
    assert all(p.passed_probe for p in result.containment+result.adjoint_containment)
    assert not result.passed_probe


def test_selected_operators_allow_different_orbit_ranks():
    spaces = (SPACE,replace(SPACE,space="second orbit"))
    row = tuple(mixing_witness(np.eye(r),s,s) for r,s in zip((2,3),spaces))
    parent = group_transport_probe(group([[0]]),spaces,np.array([[0,1]],dtype=np.int64),(row,))
    operators = group_operator_probe(parent,(np.eye(2,dtype=complex),np.eye(3,dtype=complex)))
    selection = selected_group_probe(parent,(np.array([[1],[0]],dtype=complex),
                                             np.eye(3,dtype=complex)[:,:2].copy()))
    result = selected_operator_probe(operators,selection)
    assert result.passed_probe
    assert tuple(a.shape for a in result.operator_transport.operators) == ((1,1),(2,2))


@pytest.mark.parametrize("fault", ["parent","selection","unrelated_parent","budget_type","probe",
    "bytes","work","aggregate_bytes","aggregate_work"])
def test_selected_operators_refuse_invalid_parents_and_admission(fault):
    a,q = np.eye(2,dtype=complex),np.array([[1],[0]],dtype=complex)
    operators,selection = identity_operator_selection(a,q)
    options = {}
    if fault == "parent": operators = operators.parent
    if fault == "selection": selection = selection.group_transport
    if fault == "unrelated_parent": _,selection = identity_operator_selection(a,q)
    if fault == "budget_type": options['budget'] = None
    if fault == "probe": options['probe_identity'] = " "
    if fault == "bytes": options['budget'] = Budget(1,10**9)
    if fault == "work": options['budget'] = Budget(64 << 20,1)
    if fault.startswith("aggregate"):
        result = selected_operator_probe(operators,selection)
        options['budget'] = (Budget(result.admitted_bytes-1,10**9) if fault=="aggregate_bytes"
                             else Budget(64 << 20,result.admitted_work-1))
    with pytest.raises((TypeError,ValueError,MemoryError)):
        selected_operator_probe(operators,selection,**options)


@pytest.mark.parametrize("field", ["covariance_tolerance","leakage_tolerance"])
@pytest.mark.parametrize("value", [True,-1,float('nan'),float('inf'),"1e-10"])
def test_selected_operators_refuse_invalid_tolerance(field,value):
    operators,selection = identity_operator_selection(np.eye(2,dtype=complex),
        np.array([[1],[0]],dtype=complex))
    with pytest.raises((TypeError,ValueError)):
        selected_operator_probe(operators,selection,**{field:value})


def test_selected_operators_admit_before_payload_work(monkeypatch):
    import vibeqc.symmetry_shared as shared
    operators,selection = identity_operator_selection(np.eye(2,dtype=complex),
        np.array([[1],[0]],dtype=complex))
    admitted = selected_operator_probe(operators,selection)
    def forbidden(*args,**kwargs): raise AssertionError("payload work before admission")
    original = shared.np.isfinite
    monkeypatch.setattr(shared.np,"isfinite",lambda x: original(x) if np.ndim(x)==0 else forbidden())
    monkeypatch.setattr(shared.np.linalg,"solve",forbidden)
    monkeypatch.setattr(shared,"audit_group_operators",forbidden)
    for budget,error in ((Budget(admitted.admitted_bytes-1,10**9),MemoryError),
                         (Budget(64 << 20,admitted.admitted_work-1),ValueError)):
        with pytest.raises(error,match="budget exceeded"):
            selected_operator_probe(operators,selection,budget=budget)


def test_selected_operators_refuse_overflow_during_projection():
    operators,selection = identity_operator_selection(np.full((2,2),1e308,dtype=complex),
        np.array([[1],[1]],dtype=complex)/np.sqrt(2))
    with pytest.raises(ValueError,match="nonfinite selected operator"):
        selected_operator_probe(operators,selection)
