"""Paired hierarchical BUPO-J versus independent dense libint contractions."""
import gc

import numpy as np
import pytest

import vibeqc as vq
from vibeqc import manta
from vibeqc import _vibeqc_core as core


def _basis(pure=True, far=60., angular=(0, 1, 2, 0), rotation=None, shift=(0., 0., 0.)):
    positions = np.array([[0., .1, -.2], [2., -.2, .4], [far, 1., -.7], [far+2., -.5, .6]])
    if rotation is not None:
        positions = positions@rotation.T
    positions += shift
    mol = vq.Molecule([vq.Atom(2, p.tolist()) for p in positions], 0, 1)
    shells = []
    for i, l in enumerate(angular):
        s = core.ShellInfo(); s.atom_index = i; s.l = l; s.pure = pure; s.origin = positions[i].tolist()
        s.exponents = [.6, 1.4]; s.coefficients = [.8, -.25]; shells.append(s)
    return vq.BasisSet(mol, shells, "manta-bupo-fixture", True)


def _options(backend="manta", order=6, tolerance=1e-10, opening=.8, leaf=1, branching=3):
    o = manta.BupoOptions(); o.near_backend = backend; o.order = order
    o.integral_tolerance = tolerance; o.opening = opening; o.bubbles.leaf_capacity = leaf; o.bubbles.branching = branching
    return o


def _density(n, kind="indefinite"):
    rng = np.random.default_rng(52); x = rng.normal(size=(n, n))
    if kind == "psd":
        return x@x.T/n
    if kind == "sparse":
        d = np.zeros((n, n)); d[0, 0] = .7; d[-1, -1] = -.2; d[1, -2] = d[-2, 1] = .4
        return d
    return (x+x.T)/2


def _exact(basis, d):
    return np.einsum("ijkl,kl->ij", np.asarray(vq.compute_eri(basis)), d)


@pytest.mark.parametrize("backend", ["manta", "libint"])
@pytest.mark.parametrize("pure", [False, True])
@pytest.mark.parametrize("kind", ["indefinite", "psd", "sparse"])
def test_near_only_uses_each_unordered_pair_quartet_once_and_matches_dense_reference(backend, pure, kind):
    basis = _basis(pure); d = _density(basis.nbasis, kind)
    builder = manta.BupoJBuilder(basis, _options(backend, opening=0.))
    result = builder.build(d); expected = _exact(basis, d)
    np.testing.assert_allclose(result.J, expected, atol=2e-11, rtol=5e-12)
    np.testing.assert_allclose(result.J, result.J.T, atol=2e-12, rtol=0.)
    p, r = builder.plan, result.receipt; pairs = basis.nshells*(basis.nshells+1)//2
    assert r.near_shell_calls == p.near_shell_calls == pairs*(pairs+1)//2
    assert r.near_primitive_quartets == p.near_primitive_quartets == 16*r.near_shell_calls
    assert r.far_blocks == p.far_blocks == r.moment_panels == p.moment_panels == 0
    assert r.approximation_j_bound == r.compact_j_bound == r.penetration_j_bound == r.moment_tail_j_bound == 0.
    assert r.near_provider == backend and r.moment_provider == "manta"
    assert r.near_primitive_evaluations_measured == (backend == "manta")
    assert r.coulomb_energy == pytest.approx(.5*np.sum(d*expected), abs=2e-11)
    assert r.partition_identity == p.partition_identity and r.source_identity == builder.source_identity
    assert r.traversal_visits == p.traversal_visits and r.modeled_peak_bytes == p.modeled_peak_bytes
    assert r.wall_seconds > 0


@pytest.mark.parametrize("backend", ["manta", "libint"])
@pytest.mark.parametrize("pure", [False, True])
@pytest.mark.parametrize("order,tolerance", [(0, .1), (2, .1), (6, 1e-5), (12, 1e-10)])
def test_actual_far_field_with_arbitrary_density_and_density_weighted_error_ledger(backend, pure, order, tolerance):
    basis = _basis(pure); d = _density(basis.nbasis)
    builder = manta.BupoJBuilder(basis, _options(backend, order, tolerance))
    result = builder.build(d); reference = _exact(basis, d); p, r = builder.plan, result.receipt
    error = np.max(abs(result.J-reference))
    assert error <= r.approximation_j_bound+2e-11*max(1., np.max(abs(reference)))
    assert p.maximum_integral_error <= tolerance
    assert r.density_l1 == pytest.approx(np.sum(abs(d)), rel=1e-14)
    assert r.approximation_j_bound >= p.maximum_integral_error*np.sum(abs(d))
    # The total is rounded upward separately from each component. Their
    # independently maximized sum therefore needs the documented FP headroom.
    components = r.compact_j_bound+r.penetration_j_bound+r.moment_tail_j_bound
    assert r.approximation_j_bound <= np.nextafter(components*(1+1024*np.finfo(float).eps), np.inf)
    assert r.far_blocks == p.far_blocks > 0 and r.near_shell_calls < 55
    assert r.moment_panels == p.moment_panels == 2*p.shell_pairs
    assert r.moment_primitive_pairs == p.moment_primitive_pairs == 4*r.moment_panels
    assert r.near_shell_calls == p.near_shell_calls and r.near_primitive_quartets == p.near_primitive_quartets
    np.testing.assert_allclose(result.J, result.J.T, atol=2e-12, rtol=0.)
    if order == 0:
        assert error > 1e-6  # proves that meaningful Coulomb work used the approximation
    if order == 12:
        assert error < 1e-8 and abs(r.coulomb_energy-.5*np.sum(d*reference)) < 1e-8


@pytest.mark.parametrize("order,tolerance", [(0, .1), (6, .1), (12, 1e-8)])
def test_fixed_operator_linearity_reciprocity_and_energy_derivative(order, tolerance):
    basis = _basis(); builder = manta.BupoJBuilder(basis, _options(order=order, tolerance=tolerance))
    a = _density(basis.nbasis); b = _density(basis.nbasis, "psd")
    ra, rb = builder.build(a), builder.build(b)
    combined = builder.build(.7*a-.3*b)
    np.testing.assert_allclose(combined.J, .7*ra.J-.3*rb.J, atol=5e-11, rtol=5e-12)
    assert np.sum(a*rb.J) == pytest.approx(np.sum(b*ra.J), abs=2e-10)
    step = 2e-4
    derivative = (builder.build(a+step*b).receipt.coulomb_energy-builder.build(a-step*b).receipt.coulomb_energy)/(2*step)
    assert derivative == pytest.approx(np.sum(b*ra.J), abs=1e-8)
    assert len({r.receipt.source_identity for r in [ra, rb, combined]}) == 1
    assert len({r.receipt.partition_identity for r in [ra, rb, combined]}) == 1
    assert len({r.receipt.density_identity for r in [ra, rb, combined]}) == 3
    np.testing.assert_array_equal(builder.build(np.zeros_like(a)).J, np.zeros_like(a))


def test_tightening_to_direct_and_repartitioning_preserve_same_analytic_limit():
    basis = _basis(); d = _density(basis.nbasis); expected = _exact(basis, d)
    bounds, errors = [], []
    for tolerance in [.1, 1e-3, 1e-6, 0.]:
        builder = manta.BupoJBuilder(basis, _options(order=6, tolerance=tolerance))
        result = builder.build(d); bounds.append(builder.plan.maximum_integral_error)
        errors.append(np.max(abs(result.J-expected)))
    assert bounds[-1] == 0 and errors[-1] < 2e-11
    assert bounds[-1] <= min(bounds[:-1]) and max(errors) < 1e-8
    identities = set()
    for leaf, branching in [(1, 2), (2, 8), (150, 3)]:
        builder = manta.BupoJBuilder(basis, _options(order=12, tolerance=1e-11, leaf=leaf, branching=branching))
        result = builder.build(d); identities.add(builder.source_identity)
        np.testing.assert_allclose(result.J, expected, atol=1e-8, rtol=0.)
    assert len(identities) == 3


def test_rotation_and_translation_of_scalar_basis_preserve_coulomb_limit():
    angle = .47; rotation = np.array([[np.cos(angle), -np.sin(angle), 0.], [np.sin(angle), np.cos(angle), 0.], [0., 0., 1.]])
    original = _basis(angular=(0, 0, 0, 0)); moved = _basis(angular=(0, 0, 0, 0), rotation=rotation, shift=(.3, -.6, .2))
    d = _density(original.nbasis); results = []
    for basis in [original, moved]:
        builder = manta.BupoJBuilder(basis, _options(order=12, tolerance=1e-11))
        result = builder.build(d); results.append(result)
        np.testing.assert_allclose(result.J, _exact(basis, d), atol=1e-8, rtol=0.)
    np.testing.assert_allclose(results[0].J, results[1].J, atol=1e-8, rtol=0.)
    assert results[0].receipt.source_identity != results[1].receipt.source_identity


def test_extent_failure_is_always_near_and_outputs_outlive_builder():
    basis = _basis(far=4.); d = _density(basis.nbasis)
    options = _options(tolerance=1e6); options.extents.minimum_radius = options.extents.maximum_radius = .001
    builder = manta.BupoJBuilder(basis, options)
    result = builder.build(d); saved = result.J.copy(); receipt = result.receipt
    assert result.receipt.far_blocks == 0 and result.receipt.near_shell_calls == 55
    np.testing.assert_allclose(result.J, _exact(basis, d), atol=2e-11, rtol=5e-12)
    options.integral_tolerance = 0.; options.bubbles.leaf_capacity = 200
    assert builder.build(d).receipt.source_identity == receipt.source_identity
    copied = result.J; copied[:] = -333.
    np.testing.assert_array_equal(result.J, saved)
    del builder, basis; gc.collect()
    np.testing.assert_array_equal(result.J, saved)


@pytest.mark.parametrize("opening", [0., .8])
def test_optional_component_timings_preserve_the_operator_and_account_for_executed_work(opening):
    basis = _basis(); density = _density(basis.nbasis)
    options = _options(opening=opening, tolerance=1e-5)
    plain = manta.BupoJBuilder(basis, options); reference = plain.build(density)
    options.collect_timings = True
    profiled = manta.BupoJBuilder(basis, options); result = profiled.build(density)
    assert plain.source_identity == profiled.source_identity
    np.testing.assert_array_equal(result.J, reference.J)
    assert plain.plan.partition_identity == profiled.plan.partition_identity
    assert not plain.plan.timings_collected and profiled.plan.timings_collected
    assert not reference.receipt.timings_collected and result.receipt.timings_collected
    setup_names = ["provider_setup_wall_seconds", "extent_setup_wall_seconds", "tree_setup_wall_seconds",
                   "error_model_setup_wall_seconds", "partition_setup_wall_seconds"]
    build_names = ["validation_wall_seconds", "leaf_moments_wall_seconds", "upward_wall_seconds",
                   "far_admission_wall_seconds", "far_apply_wall_seconds", "near_integrals_wall_seconds",
                   "near_digest_wall_seconds", "downward_wall_seconds", "reverse_moments_wall_seconds",
                   "finalize_wall_seconds"]
    assert all(getattr(plain.plan, name) == 0. for name in setup_names+["setup_wall_seconds"])
    assert all(getattr(reference.receipt, name) == 0. for name in build_names)
    assert 0. < sum(getattr(profiled.plan, name) for name in setup_names) <= profiled.plan.setup_wall_seconds
    assert 0. < sum(getattr(result.receipt, name) for name in build_names) <= result.receipt.wall_seconds
    assert result.receipt.near_integrals_wall_seconds > 0. and result.receipt.near_digest_wall_seconds > 0.
    for name in ["leaf_moments_wall_seconds", "upward_wall_seconds", "far_apply_wall_seconds",
                 "downward_wall_seconds", "reverse_moments_wall_seconds"]:
        assert (getattr(result.receipt, name) > 0.) == (opening > 0.)
    for name in ["near_shell_calls", "near_primitive_quartets", "far_blocks", "moment_panels", "traversal_visits"]:
        assert getattr(result.receipt, name) == getattr(reference.receipt, name)


@pytest.mark.parametrize("field,plan_field", [("memory_bytes", "modeled_peak_bytes"), ("shell_calls", "shell_calls"), ("primitive_quartets", "primitive_products"), ("scalar_terms", "scalar_terms"), ("kernel_terms", "kernel_terms")])
def test_aggregate_setup_and_build_admission_boundaries(field, plan_field):
    basis = _basis(); options = _options(order=2, tolerance=.1)
    baseline = manta.BupoJBuilder(basis, options); plan = baseline.plan
    budget = manta.ResourceBudget(); setattr(budget, field, getattr(plan, plan_field))
    exact = manta.BupoJBuilder(basis, options, budget)
    result = exact.build(_density(basis.nbasis))
    assert exact.source_identity == baseline.source_identity and result.receipt.modeled_peak_bytes == plan.modeled_peak_bytes
    setattr(budget, field, getattr(plan, plan_field)-1)
    with pytest.raises(ValueError, match="budget"):
        manta.BupoJBuilder(basis, options, budget)


def test_near_only_single_shell_reported_caps_admit_all_stages():
    mol = vq.Molecule([vq.Atom(2, [0., 0., 0.])], 0, 1)
    s = core.ShellInfo(); s.atom_index = 0; s.l = 0; s.pure = True; s.origin = [0., 0., 0.]; s.exponents = [.7]; s.coefficients = [1.]
    basis = vq.BasisSet(mol, [s], "manta-single-bupo", True); options = _options(opening=0., order=12)
    plan = manta.BupoJBuilder(basis, options).plan
    budget = manta.ResourceBudget(); budget.memory_bytes = plan.modeled_peak_bytes
    budget.shell_calls = plan.shell_calls; budget.primitive_quartets = plan.primitive_products
    budget.scalar_terms = plan.scalar_terms; budget.kernel_terms = plan.kernel_terms
    result = manta.BupoJBuilder(basis, options, budget).build(np.ones((1, 1)))
    np.testing.assert_allclose(result.J, _exact(basis, np.ones((1, 1))), atol=2e-12, rtol=3e-12)


def test_invalid_density_and_exchange_refusal():
    basis = _basis(); builder = manta.BupoJBuilder(basis, _options(order=2))
    n = basis.nbasis; d = _density(n)
    invalid = [d.astype(np.float32), d.astype(complex), np.ones((n+1, n+1)), np.eye(2*n)[::2, ::2], np.full_like(d, np.nan), np.full_like(d, np.inf)]
    nonsymmetric = d.copy(); nonsymmetric[0, 1] += .1; invalid.append(nonsymmetric)
    for value in invalid:
        with pytest.raises(ValueError):
            builder.build(value)
    with pytest.raises(OverflowError, match="density norm"):
        builder.build(np.full_like(d, np.finfo(float).max))
    with pytest.raises(ValueError, match="exchange"):
        builder.build_K(d)
    np.testing.assert_allclose(builder.build_g_rhf(d, 0.), builder.build_J(d), atol=0., rtol=0.)
    with pytest.raises(ValueError, match="exchange"):
        builder.build_g_rhf(d, 1.)
    np.testing.assert_array_equal(builder.build(np.asfortranarray(d)).J, builder.build(d).J)


@pytest.mark.parametrize("field,value", [("order", -1), ("order", 13), ("opening", -1.), ("opening", 1.), ("integral_tolerance", -1.), ("integral_tolerance", float("nan")), ("near_backend", "automatic")])
def test_invalid_algorithm_controls(field, value):
    options = _options(); setattr(options, field, value)
    with pytest.raises(ValueError):
        manta.BupoJBuilder(_basis(), options)


@pytest.mark.parametrize("pure", [False, True])
def test_l6_far_and_near_construction_against_optional_libcint(pure):
    gto = pytest.importorskip("pyscf.gto")
    positions = [[0., 0., 0.], [100., .3, -.2]]
    mol = vq.Molecule([vq.Atom(2, p) for p in positions], 0, 1); shells = []
    for i, l in enumerate([6, 0]):
        s = core.ShellInfo(); s.atom_index = i; s.l = l; s.pure = pure; s.origin = positions[i]
        s.exponents = [.7]; s.coefficients = [1.]; shells.append(s)
    basis = vq.BasisSet(mol, shells, "manta-bupo-l6", False)
    reference = gto.M(atom=[("He1", positions[0]), ("He2", positions[1])],
                      basis={"He1": [[6, [.7, 1.]]], "He2": [[0, [.7, 1.]]]}, unit="Bohr", cart=not pure, verbose=0)
    eri = reference.intor("int2e_sph" if pure else "int2e_cart")
    if not pure:
        factors = np.ones(basis.nbasis); factors[:28] = np.sqrt(13/(4*np.pi))
        eri *= np.einsum("i,j,k,l->ijkl", factors, factors, factors, factors)
    d = _density(basis.nbasis)
    builder = manta.BupoJBuilder(basis, _options(order=12, tolerance=1e-6))
    result = builder.build(d)
    assert result.receipt.far_blocks > 0 and result.receipt.near_shell_calls < 6
    np.testing.assert_allclose(result.J, np.einsum("ijkl,kl->ij", eri, d), atol=2e-10, rtol=5e-11)
